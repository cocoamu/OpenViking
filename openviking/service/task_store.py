# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Internal storage backends for TaskTracker."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol

from openviking.pyagfs import AsyncAGFSClient
from openviking.pyagfs.async_client import fs_ctx_from_agfs_path
from openviking.pyagfs.exceptions import AGFSAlreadyExistsError, AGFSNotFoundError
from openviking.service.task_tracker_concurrency import StoreIOLimiter, run_to_completion

SYSTEM_TASK_ACCOUNT_ID = "_system"
SYSTEM_TASK_USER_ID = "root"


class TaskStore(Protocol):
    async def create(self, task: Any) -> None: ...

    async def update(self, task: Any) -> None: ...

    async def get(
        self,
        task_id: str,
        *,
        account_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]: ...

    async def list(
        self, account_id: str, *, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]: ...

    async def list_page(
        self,
        account_id: str,
        *,
        user_id: str,
        limit: int,
        before: tuple[float, str] | None = None,
        prune_expired: Callable[[Dict[str, Any]], Awaitable[bool]] | None = None,
        io_limiter: StoreIOLimiter | None = None,
        **filters: Any,
    ) -> List[Dict[str, Any]]: ...

    async def delete(
        self, task_id: str, *, account_id: str, user_id: Optional[str] = None
    ) -> None: ...


class PersistentTaskStore:
    """Persist task records into AGFS under account-scoped system task directories."""

    ROOT_PREFIX = "/local"
    SYSTEM_DIRNAME = "_system"
    TASKS_DIRNAME = "tasks"

    def __init__(self, agfs: Any, *, distributed_writes: bool = False) -> None:
        self._agfs = agfs if isinstance(agfs, AsyncAGFSClient) else AsyncAGFSClient(agfs)
        self._ensured_task_dirs: set[tuple[str, str]] = set()
        self.distributed_writes = distributed_writes
        self._active_lease: ContextVar[tuple[str, Dict[str, Any]] | None] = ContextVar(
            "task_store_active_lease", default=None
        )

    @asynccontextmanager
    async def lock_task(
        self, task_id: str, *, account_id: str, user_id: str
    ) -> AsyncIterator[None]:
        if not self.distributed_writes:
            yield
            return
        await self._ensure_task_dir(account_id, user_id)
        path = self._task_path(account_id, user_id, task_id)
        lease = await self._agfs.pathlock_acquire_exact(path, timeout_secs=30.0)
        token = self._active_lease.set((path, lease))
        try:
            yield
        finally:
            self._active_lease.reset(token)
            await run_to_completion(lambda: self._agfs.pathlock_release(lease))

    @asynccontextmanager
    async def lock_business(
        self, account_id: str, user_id: str, task_type: str, resource_id: str
    ) -> AsyncIterator[None]:
        if not self.distributed_writes:
            yield
            return
        await self._ensure_task_dir(account_id, user_id)
        key = "\0".join((account_id, user_id, task_type, resource_id)).encode("utf-8")
        path = f"{self._task_dir(account_id, user_id)}/.business-{hashlib.sha256(key).hexdigest()}"
        lease = await self._agfs.pathlock_acquire_exact(path, timeout_secs=30.0)
        try:
            yield
        finally:
            await run_to_completion(lambda: self._agfs.pathlock_release(lease))

    async def create(self, task: Any) -> None:
        await self._write_task(task)

    async def update(self, task: Any) -> None:
        await self._write_task(task)

    async def get(
        self,
        task_id: str,
        *,
        account_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not account_id or not user_id:
            return None
        path = self._task_path(account_id, user_id, task_id)
        try:
            fs_ctx = fs_ctx_from_agfs_path(path)
            if self.distributed_writes:
                fs_ctx["bypass_cache"] = "true"
            raw = await self._agfs.read(path, fs_ctx=fs_ctx)
        except (AGFSNotFoundError, FileNotFoundError):
            return None
        return json.loads(_decode_bytes(raw))

    async def list(self, account_id: str, *, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if user_id is None:
            try:
                root = self._task_root_dir(account_id)
                fs_ctx = fs_ctx_from_agfs_path(root)
                if self.distributed_writes:
                    fs_ctx["bypass_cache"] = "true"
                owners = await self._agfs.ls(root, fs_ctx=fs_ctx)
            except (AGFSNotFoundError, FileNotFoundError):
                return []
            tasks: List[Dict[str, Any]] = []
            for owner in owners:
                if owner.get("isDir") and owner.get("name") not in (".", ".."):
                    tasks.extend(await self.list(account_id, user_id=owner["name"]))
            return tasks
        directory = self._task_dir(account_id, user_id)
        try:
            fs_ctx = fs_ctx_from_agfs_path(directory)
            if self.distributed_writes:
                fs_ctx["bypass_cache"] = "true"
            items = await self._agfs.ls(directory, fs_ctx=fs_ctx)
        except (AGFSNotFoundError, FileNotFoundError):
            return []
        tasks: List[Dict[str, Any]] = []
        for item in items:
            path = item.get("path") or f"{directory}/{item.get('name', '')}"
            if not path.endswith(".json"):
                continue
            try:
                fs_ctx = fs_ctx_from_agfs_path(path)
                if self.distributed_writes:
                    fs_ctx["bypass_cache"] = "true"
                raw = await self._agfs.read(path, fs_ctx=fs_ctx)
            except (AGFSNotFoundError, FileNotFoundError):
                continue
            tasks.append(json.loads(_decode_bytes(raw)))
        return tasks

    async def list_page(
        self,
        account_id: str,
        *,
        user_id: str,
        limit: int,
        before: tuple[float, str] | None = None,
        prune_expired: Callable[[Dict[str, Any]], Awaitable[bool]] | None = None,
        io_limiter: StoreIOLimiter | None = None,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        """Scan durable records, retaining only the requested page in memory.

        AGFS has no indexed query API. This bounds record memory, but still
        performs O(n) reads; it deliberately does not load all tasks into the tracker.
        """
        from openviking.service.task_pagination import matches

        async def read(operation: str, factory: Callable[[], Awaitable[Any]]) -> Any:
            if io_limiter is None:
                return await factory()
            return await io_limiter.run(operation, lambda: run_to_completion(factory))

        directory = self._task_dir(account_id, user_id)
        try:
            fs_ctx = fs_ctx_from_agfs_path(directory)
            if self.distributed_writes:
                fs_ctx["bypass_cache"] = "true"
            entries = await read(
                "list_page_ls", lambda: self._agfs.ls(directory, fs_ctx=fs_ctx)
            )
        except (AGFSNotFoundError, FileNotFoundError):
            return []
        heap: list = []
        for entry in entries:
            path = entry.get("path") or f"{directory}/{entry.get('name', '')}"
            if not path.endswith(".json"):
                continue
            try:
                fs_ctx = fs_ctx_from_agfs_path(path)
                if self.distributed_writes:
                    fs_ctx["bypass_cache"] = "true"
                task = json.loads(
                    _decode_bytes(
                        await read(
                            "list_page_read",
                            lambda path=path, fs_ctx=fs_ctx: self._agfs.read(
                                path, fs_ctx=fs_ctx
                            ),
                        )
                    )
                )
            except (AGFSNotFoundError, FileNotFoundError):
                continue
            if task.get("account_id") != account_id or task.get("user_id") != user_id:
                continue
            if prune_expired is not None and await prune_expired(task):
                continue
            key = (float(task["created_at"]), str(task["task_id"]))
            if before is not None and key >= before:
                continue
            if not matches(task, **filters):
                continue
            item = (*key, task)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif key > heap[0][:2]:
                heapq.heapreplace(heap, item)
        return [item[2] for item in sorted(heap, reverse=True)]

    async def delete(self, task_id: str, *, account_id: str, user_id: Optional[str] = None) -> None:
        """Delete a task record, succeeding if it has already been removed."""
        if not user_id:
            return
        path = self._task_path(account_id, user_id, task_id)
        fs_ctx = self._write_fs_ctx(path)
        try:
            await self._agfs.rm(
                path,
                force=True,
                fs_ctx=fs_ctx,
                auto_pathlock=False,
            )
        except (AGFSNotFoundError, FileNotFoundError):
            return

    async def _write_task(self, task: Any) -> None:
        account_id = getattr(task, "account_id", None)
        user_id = getattr(task, "user_id", None)
        if not account_id or not user_id:
            raise ValueError("PersistentTaskStore requires account_id and user_id")
        await self._ensure_task_dir(account_id, user_id)
        path = self._task_path(account_id, user_id, task.task_id)
        payload = json.dumps(_task_to_payload(task), ensure_ascii=False).encode("utf-8")
        try:
            await self._write_task_payload(path, payload)
        except (AGFSNotFoundError, FileNotFoundError):
            self._ensured_task_dirs.discard((account_id, user_id))
            await self._ensure_task_dir(account_id, user_id)
            await self._write_task_payload(path, payload)

    async def _ensure_task_dir(self, account_id: str, user_id: str) -> None:
        cache_key = (account_id, user_id)
        if cache_key in self._ensured_task_dirs:
            return

        seen_paths: set[str] = set()
        for path in self._task_dir_chain(account_id, user_id):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            await self._mkdir_if_missing(path)
        self._ensured_task_dirs.add(cache_key)

    def _task_dir_chain(self, account_id: str, user_id: str) -> tuple[str, ...]:
        return (
            self._account_dir(account_id),
            self._system_dir(account_id),
            self._task_root_dir(account_id),
            self._task_dir(account_id, user_id),
        )

    async def _write_task_payload(self, path: str, payload: bytes) -> None:
        await self._agfs.write(
            path, payload, fs_ctx=self._write_fs_ctx(path), auto_pathlock=False
        )

    def _write_fs_ctx(self, path: str) -> Dict[str, str]:
        ctx = fs_ctx_from_agfs_path(path)
        if self.distributed_writes:
            active = self._active_lease.get()
            if active is None or active[0] != path:
                raise RuntimeError("Task mutation requires its owned PathLock lease")
            ctx["lease_ref"] = active[1]["lease_ref"]
        return ctx

    async def _mkdir_if_missing(self, path: str) -> None:
        try:
            await self._agfs.mkdir(path)
        except AGFSAlreadyExistsError:
            return
        except Exception as exc:
            if "already exists" in str(exc).lower():
                return
            raise

    def _account_dir(self, account_id: str) -> str:
        return f"{self.ROOT_PREFIX}/{account_id}"

    def _system_dir(self, account_id: str) -> str:
        if account_id == SYSTEM_TASK_ACCOUNT_ID:
            return self._account_dir(account_id)
        return f"{self._account_dir(account_id)}/{self.SYSTEM_DIRNAME}"

    def _task_root_dir(self, account_id: str) -> str:
        return f"{self._system_dir(account_id)}/{self.TASKS_DIRNAME}"

    def _task_dir(self, account_id: str, user_id: str) -> str:
        return f"{self._task_root_dir(account_id)}/{user_id}"

    def _task_path(self, account_id: str, user_id: str, task_id: str) -> str:
        return f"{self._task_dir(account_id, user_id)}/{task_id}.json"


def _task_to_payload(task: Any) -> Dict[str, Any]:
    status = getattr(task, "status", None)
    return {
        **deepcopy(getattr(task, "_extra_fields", {})),
        "task_id": task.task_id,
        "task_type": task.task_type,
        "status": status.value if hasattr(status, "value") else status,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "resource_id": task.resource_id,
        "account_id": task.account_id,
        "user_id": task.user_id,
        "meta": deepcopy(task.meta),
        "stage": task.stage,
        "result": deepcopy(task.result),
        "error": task.error,
        "execution_events": deepcopy(task.execution_events),
        "auth": deepcopy(task.auth),
    }


def _decode_bytes(raw: Any) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)
