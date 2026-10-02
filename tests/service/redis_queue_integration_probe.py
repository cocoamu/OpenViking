# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Local two-process QueueFS probe; run only against a disposable Redis instance."""

import asyncio
import os
import sys

from openviking.pyagfs import get_binding_client
from openviking.service.task_queue_middleware import TaskWorkQueueMiddleware
from openviking.service.task_store import PersistentTaskStore
from openviking.service.task_tracker import TaskStatus, TaskTracker
from openviking.service.task_work_index import TaskWorkIndex, bind_task_context
from openviking.storage.queuefs.named_queue import NamedQueue
from openviking.utils.agfs_utils import RagfsBindingConfig, mount_agfs_backend
from openviking_cli.utils.config.agfs_config import AGFSConfig
from openviking_cli.utils.config.cache_config import CacheConfig


ACCOUNT = "probe-account"
USER = "probe-user"
TASK_ID = os.environ.get("OV_PROBE_TASK_ID", "redis-queue-cross-process-probe")


class QueueSource:
    def __init__(self, queue):
        self.queue = queue

    async def has_task_work(self, task_id, *, exclude_work_ids=frozenset()):
        from openviking.service.task_work_index import extract_task_metadata

        return any(
            (metadata := extract_task_metadata(message)) is not None
            and metadata.task_id == task_id
            and metadata.work_id not in exclude_work_ids
            for message in await self.queue.snapshot()
        )


async def main(mode):
    root = os.environ["OV_PROBE_ROOT"]
    endpoint = os.environ["OV_PROBE_REDIS"]
    namespace = os.environ["OV_PROBE_NAMESPACE"]
    agfs_config = AGFSConfig.model_validate(
        {
            "backend": "local",
            "path": root,
            "queuefs": {
                "backend": "cache",
                "cache_key_prefix": namespace,
            },
            "pathlock": {"provider": "cache", "namespace": namespace},
        }
    )
    cache_config = CacheConfig(
        provider="redis", params={"mode": "standalone", "endpoints": [endpoint]}
    )
    binding = RagfsBindingConfig(agfs=agfs_config, cache=cache_config)
    client_type, _ = get_binding_client()
    client = client_type(config=binding.to_binding_dict())
    mount_agfs_backend(client, binding)

    index = TaskWorkIndex()
    middleware = TaskWorkQueueMiddleware(index)
    queue = NamedQueue(client, "/queue", "AddResource", middlewares=(middleware,))
    tracker = TaskTracker(store=PersistentTaskStore(client, distributed_writes=True))
    tracker.attach_work_index(index)
    tracker.attach_queue_manager(QueueSource(queue))
    middleware.attach_task_tracker(tracker)

    if mode == "produce":
        await tracker.create("add_resource", account_id=ACCOUNT, user_id=USER, task_id=TASK_ID)
        await tracker.start(TASK_ID, account_id=ACCOUNT, user_id=USER)
        with bind_task_context(TASK_ID, ACCOUNT, USER):
            await queue.enqueue({"task_id": TASK_ID, "account_id": ACCOUNT, "user_id": USER})
        await tracker.complete(TASK_ID, {"ok": True}, account_id=ACCOUNT, user_id=USER)
        assert (await tracker.get(TASK_ID, account_id=ACCOUNT, user_id=USER)).status == TaskStatus.RUNNING
        assert len(await queue.snapshot()) == 1
    elif mode in {"claim", "crash"}:
        message = await queue.dequeue_raw()
        assert message is not None
        assert (await tracker.get(TASK_ID, account_id=ACCOUNT, user_id=USER)).status == TaskStatus.RUNNING
        if mode == "crash":
            # Simulate a worker dying after dequeue and before ACK. The caller
            # intentionally terminates this process without touching the queue.
            print(f"{mode}: claimed message {message['id']} and exiting without ack")
            return
    elif mode == "consume":
        index.rebuild({"AddResource": await queue.snapshot()})
        message = await queue.dequeue_raw()
        assert message is not None
        await queue.ack(message["id"], message)
        assert (await tracker.get(TASK_ID, account_id=ACCOUNT, user_id=USER)).status == TaskStatus.COMPLETED
        assert not await queue.snapshot()
    else:
        raise ValueError(mode)
    print(f"{mode}: passed")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
