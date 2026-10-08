import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from openviking.service.task_tracker import TaskStatus
from openviking.service.task_work_index import TASK_WORK_ID_FIELD
from openviking.storage.queuefs.add_resource_msg import AddResourceMsg
from openviking.storage.queuefs.add_resource_processor import AddResourceProcessor
from openviking.storage.queuefs.process_result import ProcessOutcome


@pytest.mark.asyncio
@pytest.mark.parametrize("user_cancelled", [False, True])
async def test_interrupted_ingest_keeps_source_for_takeover(monkeypatch, user_cancelled):
    tracker = SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(status=TaskStatus.PENDING)),
        start=AsyncMock(), complete=AsyncMock(), fail=AsyncMock(),
        get_task_auth=AsyncMock(return_value={}), wait_for_descendants=AsyncMock(),
        is_cancellation_requested=Mock(return_value=user_cancelled),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.add_resource_processor.get_task_tracker",
        lambda: tracker,
    )
    source_exists = True

    async def remove_source(*args, **kwargs):
        nonlocal source_exists
        source_exists = False

    async def execute(*args, **kwargs):
        assert source_exists, "replacement worker lost the staged source"
        return {"status": "success", "root_uri": "viking://resources/takeover"}

    service = SimpleNamespace(
        execute_add_resource_job=AsyncMock(side_effect=asyncio.CancelledError()),
        _link_resource_reason_memory=AsyncMock(), record_watch_execution=AsyncMock(),
    )
    fs = SimpleNamespace(
        _async_agfs=SimpleNamespace(pathlock_release=AsyncMock()),
        delete_temp=AsyncMock(side_effect=remove_source),
    )
    msg = AddResourceMsg(
        task_id="takeover", root_uri="viking://resources/takeover",
        account_id="acme", user_id="alice", role="user", watch_task_id="watch-1",
        staged_source={
            "temp_uri": "viking://temp/acme/takeover",
            "source_uri": "viking://temp/acme/takeover/source/doc.md",
            "source_type": "local", "original_source": "/tmp/doc.md", "meta": {},
        },
    )
    data = msg.to_dict()
    data[TASK_WORK_ID_FIELD] = "parent-work"
    with pytest.raises(asyncio.CancelledError):
        await AddResourceProcessor(service, "AddResource", fs).on_dequeue(data)

    if user_cancelled:
        assert not source_exists
        service.record_watch_execution.assert_awaited_once()
    else:
        assert source_exists
        service.record_watch_execution.assert_not_awaited()
        tracker.fail.assert_not_awaited()
        service.execute_add_resource_job.side_effect = execute
        result = await AddResourceProcessor(service, "AddResource", fs).on_dequeue(data)
        assert result.outcome is ProcessOutcome.SUCCESS
        assert not source_exists
