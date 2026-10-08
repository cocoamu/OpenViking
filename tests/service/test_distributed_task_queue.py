# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Two tracker instances sharing durable task records and queued messages."""

import asyncio
import json

import pytest

from openviking.service.task_queue_middleware import TaskWorkQueueMiddleware
from openviking.service.task_store import PersistentTaskStore
from openviking.service.task_tracker import TaskStatus, TaskTracker
from openviking.service.task_work_index import TaskWorkIndex, bind_task_context, extract_task_metadata
from openviking.storage.queuefs.process_result import ProcessOutcome, ProcessResult
from openviking.storage.queuefs.queue_middleware import AckContext, EnqueueContext, ProcessContext
from tests.test_task_tracker import _FakeAgfs

pytestmark = pytest.mark.asyncio
OWNER = {"account_id": "acme", "user_id": "alice"}


class SharedMessages:
    def __init__(self):
        self.messages = {}

    async def has_task_work(self, task_id, *, exclude_work_ids=frozenset(), exclude_queue_names=frozenset()):
        return any(
            (metadata := extract_task_metadata(message)) is not None
            and metadata.task_id == task_id
            and metadata.work_id not in exclude_work_ids
            and message.get("queue_name") not in exclude_queue_names
            for message in self.messages.values()
        )

    async def enqueue(self, middleware, task_id, *, before_commit=None, queue_name="AddResource"):
        ctx = EnqueueContext(queue_name, {"task_id": task_id, **OWNER})

        async def write(ctx):
            if before_commit is not None:
                await before_commit()
            message = {"id": f"message-{len(self.messages) + 1}", "data": json.dumps(ctx.payload), "queue_name": queue_name}
            self.messages[message["id"]] = message
            ctx.committed = True
            return message["id"]

        with bind_task_context(task_id, **OWNER):
            message_id = await middleware.enqueue(ctx, write)
        return self.messages[message_id]

    async def ack(self, middleware, message, *, fail=False):
        ctx = AckContext(message.get("queue_name", "AddResource"), message["id"], message)

        async def remove(ctx):
            if fail:
                raise OSError("transport unavailable")
            self.messages.pop(ctx.message_id)
            ctx.committed = True

        await middleware.ack(ctx, remove)


def two_pods(distributed=True):
    agfs = _FakeAgfs()
    shared = SharedMessages()
    pods = []
    for _ in range(2):
        tracker = TaskTracker(store=PersistentTaskStore(agfs, distributed_writes=distributed))
        index = TaskWorkIndex()
        tracker.attach_work_index(index)
        tracker.attach_queue_manager(shared)
        middleware = TaskWorkQueueMiddleware(index)
        middleware.attach_task_tracker(tracker)
        pods.append((tracker, middleware, index))
    return shared, pods


async def test_cross_pod_ack_finishes_task_after_last_durable_message():
    shared, ((a, producer, a_index), (b, consumer, _)) = two_pods()
    task = await a.create("add_resource", **OWNER)
    await a.start(task.task_id, **OWNER)
    message = await shared.enqueue(producer, task.task_id)
    await a.complete(task.task_id, {"ok": True}, **OWNER)
    assert (await b.get(task.task_id, **OWNER)).status == TaskStatus.RUNNING

    await shared.ack(consumer, message)

    assert (await a.get(task.task_id, **OWNER)).status == TaskStatus.COMPLETED
    assert a_index.has_work(task.task_id)
    assert not shared.messages


async def test_remote_cancel_waits_for_owner_work_to_settle():
    shared, ((a, producer, _), (b, consumer, _)) = two_pods()
    task = await a.create("add_resource", **OWNER)
    await a.start(task.task_id, **OWNER)
    message = await shared.enqueue(producer, task.task_id)

    cancelled = await b.cancel(task.task_id, **OWNER)
    assert cancelled.status == TaskStatus.CANCELLING
    assert (await a.get(task.task_id, **OWNER)).status == TaskStatus.CANCELLING

    result = await producer.process(
        ProcessContext("AddResource", message, cancel=lambda: asyncio.sleep(0, result=ProcessResult.cancelled())),
        lambda _ctx: asyncio.sleep(0, result=ProcessResult.success()),
    )
    assert result.outcome is ProcessOutcome.CANCELLED
    await shared.ack(consumer, message)
    assert (await a.get(task.task_id, **OWNER)).status == TaskStatus.CANCELLED


async def test_cancel_waits_for_in_flight_enqueue_commit():
    shared, ((a, producer, _), (b, _, _)) = two_pods()
    task = await a.create("add_resource", **OWNER)
    await a.start(task.task_id, **OWNER)
    writing = asyncio.Event()
    release = asyncio.Event()

    async def before_commit():
        writing.set()
        await release.wait()

    enqueue = asyncio.create_task(shared.enqueue(producer, task.task_id, before_commit=before_commit))
    await asyncio.wait_for(writing.wait(), 2)
    cancel = asyncio.create_task(b.cancel(task.task_id, **OWNER))
    await asyncio.sleep(0)
    assert not cancel.done()
    release.set()
    await enqueue
    assert (await cancel).status == TaskStatus.CANCELLING


async def test_ack_failure_replay_skips_terminal_task_handler():
    shared, ((a, producer, _), (b, consumer, _)) = two_pods()
    task = await a.create("add_resource", **OWNER)
    await a.start(task.task_id, **OWNER)
    message = await shared.enqueue(producer, task.task_id)
    await a.complete(task.task_id, {"ok": True}, **OWNER)

    with pytest.raises(OSError, match="transport unavailable"):
        await shared.ack(consumer, message, fail=True)
    assert (await a.get(task.task_id, **OWNER)).status == TaskStatus.COMPLETED
    assert message["id"] in shared.messages

    async def forbidden(_ctx):
        pytest.fail("terminal task replayed its handler")

    result = await consumer.process(
        ProcessContext("AddResource", message, cancel=forbidden), forbidden
    )
    assert result.outcome is ProcessOutcome.SUCCESS
    await shared.ack(consumer, message)
    assert not shared.messages


async def test_failure_on_one_pod_survives_last_ack_on_another():
    shared, ((a, producer, _), (b, consumer, _)) = two_pods()
    task = await a.create("add_resource", **OWNER)
    await a.start(task.task_id, **OWNER)
    failed = await shared.enqueue(producer, task.task_id)
    last = await shared.enqueue(producer, task.task_id)
    await a.complete(task.task_id, {"ok": True}, **OWNER)

    result = await consumer.process(
        ProcessContext("AddResource", failed, cancel=lambda: asyncio.sleep(0)),
        lambda _ctx: asyncio.sleep(0, result=ProcessResult.failed("child failed")),
    )
    assert result.outcome is ProcessOutcome.FAILED
    await shared.ack(consumer, failed)
    assert (await a.get(task.task_id, **OWNER)).status == TaskStatus.RUNNING
    await shared.ack(producer, last)
    terminal = await b.get(task.task_id, **OWNER)
    assert terminal.status == TaskStatus.FAILED
    assert terminal.error == "child failed"


async def test_direct_active_work_blocks_distributed_finalization():
    _shared, ((a, _producer, _), _) = two_pods()
    task = await a.create("add_resource", **OWNER)
    await a.start(task.task_id, **OWNER)
    a.register_running_task(task.task_id)
    await a.complete(task.task_id, {"ok": True}, **OWNER)
    assert (await a.get(task.task_id, **OWNER)).status == TaskStatus.RUNNING
    await a.unregister_running_task(task.task_id)
    assert (await a.get(task.task_id, **OWNER)).status == TaskStatus.COMPLETED


async def test_remote_cancel_waits_for_direct_work_on_another_pod():
    _shared, ((a, _producer, _), (b, _consumer, _)) = two_pods()
    task = await a.create("admin_reindex", **OWNER)
    await a.start(task.task_id, **OWNER)
    worker_id = await a.register_running_task_shared(task.task_id, **OWNER)

    cancelled = await b.cancel(task.task_id, **OWNER)

    assert cancelled.status == TaskStatus.CANCELLING
    assert (await b.get(task.task_id, **OWNER)).status == TaskStatus.CANCELLING
    await a.unregister_running_task_shared(task.task_id, worker_id=worker_id, **OWNER)
    assert (await b.get(task.task_id, **OWNER)).status == TaskStatus.CANCELLED


@pytest.mark.parametrize("current_work_id", ["parent-work", None])
async def test_distributed_wait_limits_shared_scans_and_observes_remote_ack(current_work_id):
    shared, ((a, producer, index), (_, consumer, _)) = two_pods()
    task = await a.create("add_resource", **OWNER)
    message = await shared.enqueue(producer, task.task_id)
    calls = 0
    original = shared.has_task_work

    async def count_scan(*args, **kwargs):
        nonlocal calls
        calls += 1
        return await original(*args, **kwargs)

    shared.has_task_work = count_scan
    waiting = asyncio.create_task(a.wait_for_descendants(task.task_id, current_work_id))
    try:
        await asyncio.sleep(0.25)
        assert not waiting.done(), "direct reindex must record its wait without a queue delivery ID"
        recorded = await a.get(task.task_id, **OWNER)
        event = recorded.execution_events["items"][-1]
        assert event["kind"] == "waiting_for_descendants"
        assert event.get("operation") == current_work_id
        assert calls <= 2, "one waiting task must not repeatedly scan all queues in a quarter second"
        await shared.ack(consumer, message)
        await asyncio.wait_for(waiting, 2)
        assert index.has_work(task.task_id), "remote ACK deliberately leaves this local index stale"
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


async def test_retry_roots_do_not_deadlock_descendants_or_finalize_before_last_ack():
    shared, ((a, producer, _), (b, consumer, _)) = two_pods()
    task = await a.create("add_resource", **OWNER)
    await a.start(task.task_id, **OWNER)
    roots = [await shared.enqueue(producer, task.task_id) for _ in range(3)]
    child = await shared.enqueue(producer, task.task_id, queue_name="Embedding")
    await a.complete(task.task_id, {"root_uri": "viking://resources/probe"}, **OWNER)
    waits = [asyncio.create_task(tracker.wait_for_descendants(
        task.task_id, extract_task_metadata(root).work_id, current_queue_name="AddResource"
    )) for tracker, root in zip((a, b), roots)]
    try:
        await asyncio.sleep(0.1)
        assert not any(w.done() for w in waits), "both parents must wait for real embedding work"
        await shared.ack(consumer, child)
        await asyncio.wait_for(asyncio.gather(*waits), 2)
        assert (await b.get(task.task_id, **OWNER)).status == TaskStatus.RUNNING
        for root in roots[:-1]:
            await shared.ack(consumer, root)
            assert (await a.get(task.task_id, **OWNER)).status == TaskStatus.RUNNING
        await shared.ack(consumer, roots[-1])
        assert (await a.get(task.task_id, **OWNER)).status == TaskStatus.COMPLETED
    finally:
        for w in waits:
            w.cancel()
        await asyncio.gather(*waits, return_exceptions=True)


async def test_local_retry_roots_do_not_block_real_children_or_finalization_check():
    shared, ((tracker, middleware, _), _) = two_pods(distributed=False)
    task = await tracker.create("add_resource", **OWNER)
    roots = [await shared.enqueue(middleware, task.task_id) for _ in range(2)]
    child = await shared.enqueue(middleware, task.task_id, queue_name="Embedding")
    waiting = asyncio.create_task(tracker.wait_for_descendants(
        task.task_id, extract_task_metadata(roots[0]).work_id, current_queue_name="AddResource"
    ))
    try:
        await asyncio.sleep(0.1)
        assert not waiting.done()
        await shared.ack(middleware, child)
        await asyncio.wait_for(waiting, 0.5)
        assert tracker.has_work(task.task_id), "retry roots still block finalization"
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
