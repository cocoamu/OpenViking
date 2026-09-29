# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Adapt the existing in-memory TaskWorkIndex to queue lifecycle middleware."""

import asyncio
from contextlib import suppress
from typing import Any

from openviking.service.task_work_index import (
    TaskWorkIndex,
    TaskWorkRejected,
    bind_task_context,
    extract_task_metadata,
    prepare_task_payload,
)
from openviking.storage.queuefs.process_result import ProcessOutcome, ProcessResult
from openviking.storage.queuefs.queue_middleware import (
    AckContext,
    AckNext,
    ClearContext,
    ClearNext,
    EnqueueContext,
    EnqueueNext,
    ProcessContext,
    ProcessNext,
    QueueMiddleware,
)
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


class TaskWorkQueueMiddleware(QueueMiddleware):
    """Keep QueueFS as the durable source of truth; no new task state or store."""

    def __init__(self, index: TaskWorkIndex) -> None:
        self._index = index
        self._tracker: Any = None

    def attach_task_tracker(self, tracker: Any) -> None:
        self._tracker = tracker

    def _distributed(self, metadata: Any) -> bool:
        if metadata is None or self._tracker is None:
            return False
        if not self._tracker.distributed_queue_enabled:
            return False
        if not metadata.account_id or not metadata.user_id:
            raise RuntimeError("Distributed task work requires account and user ownership")
        return True

    async def enqueue(self, ctx: EnqueueContext, call_next: EnqueueNext) -> str:
        if not isinstance(ctx.payload, dict):
            return await call_next(ctx)
        ctx.payload, metadata = prepare_task_payload(ctx.payload)
        if self._distributed(metadata):
            async with self._tracker._task_guard(
                metadata.task_id, metadata.account_id, metadata.user_id
            ):
                current = await self._tracker.get(
                    metadata.task_id,
                    account_id=metadata.account_id,
                    user_id=metadata.user_id,
                )
                if current is not None and current.status.value in (
                    "cancelling", "cancelled", "completed", "failed"
                ):
                    raise TaskWorkRejected(
                        f"Task {metadata.task_id} is no longer active"
                    )
                if not self._index.register(ctx.queue, metadata):
                    raise TaskWorkRejected(
                        f"Task {metadata.task_id} is cancelling; rejected work for {ctx.queue}"
                    )
                try:
                    return await call_next(ctx)
                finally:
                    if not ctx.committed:
                        await self._index.discard(ctx.queue, metadata, finalize=False)
        if not self._index.register(ctx.queue, metadata):
            raise TaskWorkRejected(
                f"Task {metadata.task_id} is cancelling; rejected work for {ctx.queue}"
            )
        try:
            return await call_next(ctx)
        finally:
            if metadata is not None and not ctx.committed:
                await self._index.discard(ctx.queue, metadata)

    async def process(self, ctx: ProcessContext, call_next: ProcessNext) -> ProcessResult:
        metadata = extract_task_metadata(ctx.message)
        if metadata is None:
            return await call_next(ctx)

        distributed = self._distributed(metadata)
        if distributed:
            current = await self._tracker.get(
                metadata.task_id,
                account_id=metadata.account_id,
                user_id=metadata.user_id,
            )
            if current is not None and current.status.value in (
                "completed", "failed", "cancelled"
            ):
                return ProcessResult.success()

        async def watch_cancellation() -> None:
            while True:
                await asyncio.sleep(2)
                try:
                    current = await self._tracker.get(
                        metadata.task_id,
                        account_id=metadata.account_id,
                        user_id=metadata.user_id,
                    )
                except Exception:
                    logger.warning("Could not refresh task %s cancellation", metadata.task_id)
                    continue
                if current is not None and current.status.value == "cancelling":
                    self._index.cancel_active(metadata.task_id)
                    return

        active_task = asyncio.current_task()
        watcher = None
        with bind_task_context(metadata.task_id, metadata.account_id, metadata.user_id):
            if self._index.cancellation_requested(metadata.task_id):
                result = await ctx.cancel()
                if result.outcome is ProcessOutcome.FAILED:
                    self._index.record_failure(metadata.task_id, result.error)
                return result
            if active_task is not None:
                self._index.register_active(metadata.task_id, active_task)
                if distributed:
                    watcher = asyncio.create_task(watch_cancellation())
            try:
                result = await call_next(ctx)
            except asyncio.CancelledError:
                if self._index.cancellation_requested(metadata.task_id):
                    # Handler unwinding must finish before this delivery is settled.
                    return ProcessResult.cancelled()
                raise
            except Exception as exc:
                self._index.record_failure(metadata.task_id, str(exc))
                if distributed:
                    await self._tracker.fail(
                        metadata.task_id, str(exc), metadata.account_id, metadata.user_id
                    )
                raise
            finally:
                if watcher is not None:
                    watcher.cancel()
                    with suppress(asyncio.CancelledError):
                        await watcher
                if active_task is not None:
                    self._index.unregister_active(metadata.task_id, active_task)
            if result.outcome is ProcessOutcome.FAILED:
                self._index.record_failure(metadata.task_id, result.error)
                if distributed:
                    await self._tracker.fail(
                        metadata.task_id,
                        result.error,
                        metadata.account_id,
                        metadata.user_id,
                    )
            return result

    async def ack(self, ctx: AckContext, call_next: AckNext) -> None:
        metadata = extract_task_metadata(ctx.message) if ctx.message is not None else None
        if self._distributed(metadata):
            async with self._tracker._task_guard(
                metadata.task_id, metadata.account_id, metadata.user_id
            ):
                prepared = await self._index.prepare_ack(
                    ctx.queue, metadata, finalize=False
                )
                try:
                    await self._tracker._finalize_task_locked(
                        metadata.task_id,
                        metadata.account_id,
                        metadata.user_id,
                        exclude_work_ids=frozenset({metadata.work_id}),
                    )
                    await call_next(ctx)
                finally:
                    if prepared is not None and not ctx.committed:
                        self._index.rollback_ack(ctx.queue, prepared)
            return
        prepared = None
        try:
            if ctx.message is not None:
                prepared = await self._index.prepare_ack(ctx.queue, ctx.message)
            await call_next(ctx)
        finally:
            if prepared is not None and not ctx.committed:
                self._index.rollback_ack(ctx.queue, prepared)

    async def clear(self, ctx: ClearContext, call_next: ClearNext) -> None:
        metadata = [extract_task_metadata(message) for message in ctx.messages]
        if any(self._distributed(item) for item in metadata):
            raise RuntimeError("Distributed tracked queue work cannot be cleared in bulk")
        prepared = []
        try:
            for message in ctx.messages:
                metadata = await self._index.prepare_ack(ctx.queue, message)
                if metadata is not None:
                    prepared.append(metadata)
            await call_next(ctx)
        finally:
            if not ctx.committed:
                for metadata in prepared:
                    self._index.rollback_ack(ctx.queue, metadata)
