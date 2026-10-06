import asyncio
from unittest.mock import AsyncMock

import pytest

from openviking.pyagfs import AsyncAGFSClient
from openviking.service.deletion import _DeletionProcessor
from openviking.storage.queuefs.named_queue import NamedQueue


class _DeletionService:
    async def _process(self, _message):
        return "cleanup is retryable"


@pytest.mark.asyncio
async def test_cleanup_error_is_raised_for_queue_recovery():
    processor = _DeletionProcessor(_DeletionService(), asyncio.get_running_loop())

    with pytest.raises(RuntimeError, match="cleanup is retryable"):
        await processor.on_dequeue(
            {
                "task_id": "task-1",
                "account_id": "_system",
                "user_id": "root",
                "target": {"account_id": "account-1", "user_id": None},
            }
        )


@pytest.mark.asyncio
async def test_cleanup_error_skips_named_queue_ack(monkeypatch):
    transport = AsyncMock(spec=AsyncAGFSClient)
    transport.read.return_value = (
        b'{"id":"message-1","data":{"task_id":"task-1",'
        b'"account_id":"_system","user_id":"root",'
        b'"target":{"account_id":"account-1","user_id":null}}}'
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.named_queue.AsyncAGFSClient",
        lambda _client: transport,
    )

    processor = _DeletionProcessor(_DeletionService(), asyncio.get_running_loop())
    queue = NamedQueue(
        object(),
        "/queue",
        "account_delete",
        dequeue_handler=processor,
    )

    assert await queue.dequeue() is None
    transport.write.assert_not_awaited()
