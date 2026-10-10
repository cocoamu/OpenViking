"""Bound large document conversion outside the long-lived server process."""

from __future__ import annotations

import asyncio
import ctypes
import gc
import json
import logging
import math
import os
import signal
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_MIB = 1024 * 1024
_ADMISSION_LOCK = threading.Lock()
_ACTIVE_WORKERS = 0
_NEXT_RECLAIM_AT = 0.0


class DocumentWorkerError(RuntimeError):
    """A conversion worker failed while the server remained alive."""


@dataclass(frozen=True)
class WorkerSettings:
    memory_bytes: int = 3072 * _MIB
    resident_bytes: int = 1024 * _MIB
    max_workers: int = 1
    timeout_seconds: float = 600
    admission_timeout_seconds: float = 900

    @classmethod
    def from_environment(cls) -> "WorkerSettings":
        settings = cls(
            memory_bytes=int(os.getenv("OPENVIKING_DOCUMENT_WORKER_MEMORY_MB", "3072")) * _MIB,
            resident_bytes=int(os.getenv("OPENVIKING_DOCUMENT_WORKER_RESIDENT_MB", "1024")) * _MIB,
            max_workers=int(os.getenv("OPENVIKING_DOCUMENT_WORKER_CONCURRENCY", "1")),
            timeout_seconds=float(os.getenv("OPENVIKING_DOCUMENT_WORKER_TIMEOUT", "600")),
            admission_timeout_seconds=float(
                os.getenv("OPENVIKING_DOCUMENT_WORKER_ADMISSION_TIMEOUT", "900")
            ),
        )
        budgets = (settings.memory_bytes, settings.resident_bytes, settings.max_workers,
                   settings.timeout_seconds, settings.admission_timeout_seconds)
        if any(not math.isfinite(value) or value <= 0 for value in budgets):
            raise ValueError("Document worker budgets must be positive")
        return settings


def should_isolate(path: Path) -> bool:
    """Bound native document decoding even when its compressed input is small."""
    if sys.platform != "linux" or os.getenv("OPENVIKING_DOCUMENT_ISOLATION", "1") == "0":
        return False
    threshold = int(os.getenv("OPENVIKING_DOCUMENT_WORKER_THRESHOLD_MB", "4")) * _MIB
    if threshold <= 0:
        raise ValueError("Document worker threshold must be positive")
    try:
        size = path.stat().st_size
    except OSError:
        return False
    # Compressed byte size cannot bound decoded pages, XML, or native image buffers.
    # Keep plain small inputs on their existing path, but never let a small native
    # document bypass the process-global slot and child address/RSS limits.
    native_suffixes = {
        ".pdf", ".doc", ".docx", ".docm", ".odt", ".rtf",
        ".ppt", ".pptx", ".pptm", ".pps", ".ppsx", ".ppsm", ".pot", ".odp",
        ".xls", ".xlsx", ".xlsm", ".xlsb", ".ods", ".epub",
    }
    return path.suffix.lower() in native_suffixes or size >= threshold


def _memory_snapshot() -> tuple[int, int] | None:
    """Read the container's current usage and limit, supporting cgroup v1/v2."""
    for current, limit in (
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        ("/sys/fs/cgroup/memory/memory.usage_in_bytes",
         "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            used, capacity = int(Path(current).read_text()), int(Path(limit).read_text())
            if 0 <= used and 0 < capacity < (1 << 60):
                return used, capacity
        except (OSError, ValueError):
            continue
    return None


def _try_acquire(settings: WorkerSettings) -> bool:
    global _ACTIVE_WORKERS
    with _ADMISSION_LOCK:
        if _ACTIVE_WORKERS >= settings.max_workers:
            return False
        snapshot = _memory_snapshot()
        if snapshot is not None:
            used, capacity = snapshot
            # Account for reservations not yet resident. Already-running workers
            # may consume their entire allowance after this snapshot is taken.
            # RLIMIT_AS includes mapped native libraries, not just physical RAM.
            # Reserve RSS separately, and supervise it while the child runs.
            reserved = (_ACTIVE_WORKERS + 1) * min(settings.resident_bytes, settings.memory_bytes)
            if used + reserved > int(capacity * 0.85):
                return False
        _ACTIVE_WORKERS += 1
        return True


def _reclaim_parent_memory() -> None:
    """Return unused allocator pages; live objects and admission limits stay intact."""
    before = _memory_snapshot()
    gc.collect()
    try:
        trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if trim is not None:
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            trim(0)
    except (OSError, AttributeError):
        logger.debug("Parent allocator reclaim unavailable", exc_info=True)
    logger.info("Document worker parent memory reclaim before=%s after=%s",
                before, _memory_snapshot())


async def _maybe_reclaim_parent_memory(settings: WorkerSettings) -> None:
    """Try reclaim at most once per 30s when idle native workers cannot fit."""
    global _NEXT_RECLAIM_AT
    if sys.platform != "linux":
        return
    with _ADMISSION_LOCK:
        snapshot = _memory_snapshot()
        now = time.monotonic()
        if (_ACTIVE_WORKERS or snapshot is None or now < _NEXT_RECLAIM_AT
                or snapshot[0] + min(settings.resident_bytes, settings.memory_bytes)
                <= int(snapshot[1] * 0.85)):
            return
        _NEXT_RECLAIM_AT = now + 30
    await asyncio.to_thread(_reclaim_parent_memory)


@asynccontextmanager
async def _worker_slot(settings: WorkerSettings):
    global _ACTIVE_WORKERS
    started, next_log = time.monotonic(), time.monotonic()
    while not _try_acquire(settings):
        await _maybe_reclaim_parent_memory(settings)
        elapsed = time.monotonic() - started
        if elapsed >= settings.admission_timeout_seconds:
            raise DocumentWorkerError("DOCUMENT_WORKER_ADMISSION_TIMEOUT: memory budget unavailable")
        if time.monotonic() >= next_log:
            logger.info("Document worker waiting for capacity active=%s memory=%s",
                        _ACTIVE_WORKERS, _memory_snapshot())
            next_log = time.monotonic() + 30
        await asyncio.sleep(0.25)
    try:
        yield
    finally:
        with _ADMISSION_LOCK:
            _ACTIVE_WORKERS -= 1


async def _finish_cleanup(task: asyncio.Task):
    """A cancelled caller must still reap its child before releasing the slot."""
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise


async def _stop_worker(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=2)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


def _worker_resident_bytes(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


async def _wait_worker(process: asyncio.subprocess.Process, settings: WorkerSettings) -> None:
    waiting = asyncio.create_task(process.wait())
    deadline = time.monotonic() + settings.timeout_seconds
    try:
        while not waiting.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            try:
                await asyncio.wait_for(asyncio.shield(waiting), min(0.1, remaining))
            except asyncio.TimeoutError:
                resident = _worker_resident_bytes(process.pid)
                snapshot = _memory_snapshot()
                if (resident is not None and resident > settings.resident_bytes) or (
                    snapshot is not None and snapshot[0] > int(snapshot[1] * 0.85)
                ):
                    raise DocumentWorkerError("DOCUMENT_WORKER_MEMORY_LIMIT: physical memory budget exceeded")
        await waiting
    finally:
        if not waiting.done():
            waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


async def convert_document_in_worker(
    kind: str,
    path: Path,
    *,
    storage,
    resource_name: str | None,
    pdf_config: dict[str, Any] | None = None,
    max_table_rows: int = 1000,
) -> dict[str, Any]:
    """Isolate native decoding; return text and metadata, not image byte arrays."""
    if kind not in {"pdf", "anydoc"}:
        raise ValueError(f"Unsupported document worker kind: {kind}")
    settings = WorkerSettings.from_environment()
    async with _worker_slot(settings):
        with tempfile.TemporaryDirectory(prefix="ov-document-worker-") as directory:
            directory = Path(directory)
            request_file, result_file = directory / "request.json", directory / "result.json"
            request = {
                "kind": kind,
                "source": str(path.resolve()),
                "media_dir": str(storage.media_dir.resolve()),
                "resource_name": resource_name or path.stem,
                "pdf_config": pdf_config or {},
                "max_table_rows": max_table_rows,
                "memory_bytes": settings.memory_bytes,
                "parent_pid": os.getpid(),
                "staging_dir": str(directory / "media-stage"),
            }
            request_file.write_text(json.dumps(request), encoding="utf-8")
            request_file.chmod(0o600)
            env = dict(os.environ, MALLOC_ARENA_MAX="4", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            runner = Path(__file__).with_name("_document_worker_runner.py")
            started = time.monotonic()
            spawn = asyncio.create_task(asyncio.create_subprocess_exec(
                sys.executable, str(runner), str(request_file), str(result_file),
                start_new_session=True, env=env,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            ))
            process = None
            try:
                try:
                    process = await asyncio.shield(spawn)
                except asyncio.CancelledError:
                    # Cancellation can arrive after fork but before Process is returned.
                    process = await _finish_cleanup(spawn)
                    raise
                try:
                    await _wait_worker(process, settings)
                except asyncio.TimeoutError as exc:
                    raise DocumentWorkerError("DOCUMENT_WORKER_TIMEOUT: local conversion exceeded budget") from exc
                if not result_file.is_file():
                    raise DocumentWorkerError(
                        f"DOCUMENT_WORKER_EXIT: local converter exited with code {process.returncode}"
                    )
                if result_file.stat().st_size > 64 * _MIB:
                    raise DocumentWorkerError("DOCUMENT_WORKER_RESULT_LIMIT: conversion result is too large")
                payload = json.loads(result_file.read_text(encoding="utf-8"))
                if process.returncode != 0 or payload.get("status") != "ok":
                    error = payload.get("error", {})
                    raise DocumentWorkerError(
                        "DOCUMENT_WORKER_FAILED: " + str(error.get("type", "ConversionError"))
                        + ": " + str(error.get("message", "local conversion failed"))[:1000]
                    )
                result = payload.get("result")
                if not isinstance(result, dict) or not isinstance(result.get("markdown"), str):
                    raise DocumentWorkerError("DOCUMENT_WORKER_PROTOCOL: invalid conversion result")
                logger.info("Document worker completed kind=%s pid=%s seconds=%.3f peak_rss_kib=%s",
                            kind, process.pid, time.monotonic() - started,
                            payload.get("stats", {}).get("peak_rss_kib"))
                return result
            finally:
                if process is not None:
                    cleanup = asyncio.create_task(_stop_worker(process))
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        await _finish_cleanup(cleanup)
                        raise
