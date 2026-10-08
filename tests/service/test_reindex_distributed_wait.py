"""Reindex must observe remote ACKs rather than a stale local request tracker."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.service import reindex_executor as module
from openviking.service.reindex_executor import ReindexExecutor
from openviking.service.task_work_index import bind_task_context
from openviking.telemetry.request_wait_tracker import RequestWaitTracker
from openviking_cli.session.user_id import UserIdentifier


@pytest.mark.parametrize("distributed", [True, False])
async def test_reindex_waits_for_remote_ack_with_stale_local_roots(monkeypatch, distributed):
    uri = "viking://resources/tester/resources/probe"
    release = AsyncMock()
    agfs = SimpleNamespace(pathlock_acquire_tree=AsyncMock(return_value="lease"),
                           pathlock_as_borrowed=AsyncMock(return_value="borrowed"),
                           pathlock_release=release)
    fs = SimpleNamespace(_uri_to_path=lambda uri, ctx: "/probe",
                         stat=AsyncMock(return_value={"isDir": True}), _async_agfs=agfs)
    service = SimpleNamespace(viking_fs=fs, vikingdb_manager=SimpleNamespace(has_queue_manager=True))
    monkeypatch.setattr(module, "get_service", lambda: service)
    monkeypatch.setattr(module, "get_current_telemetry", lambda: SimpleNamespace(telemetry_id="tm_remote"))
    local = RequestWaitTracker()
    local.cleanup("tm_remote")
    monkeypatch.setattr(module, "get_request_wait_tracker", lambda: local)
    remote_ack = asyncio.Event()
    entered_wait = asyncio.Event()
    async def shared_wait(task_id, current_work_id):
        assert task_id == "reindex-task"
        assert current_work_id == ""
        entered_wait.set()
        await remote_ack.wait()
    task_tracker = SimpleNamespace(distributed_queue_enabled=distributed,
                                   wait_for_descendants=AsyncMock(side_effect=shared_wait))
    monkeypatch.setattr(module, "get_task_tracker", lambda: task_tracker)
    executor = ReindexExecutor()
    async def produce_work(**kwargs):
        local.register_embedding_root("tm_remote", "remote-message")
        kwargs["run"].counters.rebuilt_records = 3
        # A previous failed message must not be hidden by successful remote ACKs.
        local.mark_embedding_failed("tm_remote", "failed-message", "existing embedding error")
    executor._reindex_resource = AsyncMock(side_effect=produce_work)
    ctx = RequestContext(user=UserIdentifier("acc1", "tester"), role=Role.ADMIN)
    with bind_task_context("reindex-task", "acc1", "tester"):
        operation = asyncio.create_task(executor._run(uri=uri, object_type="resource",
                                    mode="vectors_only", ctx=ctx))
    try:
        await asyncio.sleep(0.02)
        assert not operation.done(), "must wait while durable work is outstanding"
        if distributed:
            await asyncio.wait_for(entered_wait.wait(), timeout=0.1)
            remote_ack.set()
        else:
            assert not entered_wait.is_set()
            local.mark_embedding_done("tm_remote", "remote-message", vector_written=True)
        result = await asyncio.wait_for(operation, timeout=0.2)
        assert result["status"] == "completed"
        assert result["failed_records"] == 1
        assert result["rebuilt_records"] == 2
        assert "existing embedding error" in result["warnings"][0]
        assert local.is_complete("tm_remote")  # cleaned up by the executor
        release.assert_awaited_once_with("lease")
    finally:
        if not operation.done():
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
        local.cleanup("tm_remote")
