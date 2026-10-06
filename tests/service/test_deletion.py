import asyncio

import pytest

from openviking.service.deletion import _DeletionProcessor


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
