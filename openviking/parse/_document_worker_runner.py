"""Private subprocess entrypoint; apply limits before importing native parsers."""

from __future__ import annotations

import ctypes
import json
import logging
import os
import resource
import shutil
import signal
import sys
from dataclasses import asdict
from pathlib import Path


def _protect_worker(memory_bytes: int, parent_pid: int) -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    allowance = memory_bytes if hard == resource.RLIM_INFINITY else min(memory_bytes, hard)
    resource.setrlimit(resource.RLIMIT_AS, (allowance, allowance))
    # Linux automatically kills this child if the server is killed or restarted.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Unable to protect document worker lifecycle")
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def _publish_media(source: Path, destination: Path) -> None:
    """Copy only successful conversion media, without materializing all bytes."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("Document worker produced an invalid media path")
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        if not target.resolve().is_relative_to(root):
            raise ValueError("Document worker media escaped storage directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def _convert(request: dict) -> dict:
    from openviking_cli.utils.storage import StoragePath

    storage = StoragePath(Path(request["staging_dir"]))
    source = Path(request["source"])
    if request["kind"] == "pdf":
        from openviking.parse.parsers.pdf import PDFParser
        from openviking_cli.utils.config.parser_config import PDFConfig

        config = PDFConfig(strategy="local", **request["pdf_config"])
        markdown, meta = PDFParser(config)._convert_local_sync(
            source, storage, request["resource_name"]
        )
        result = {"markdown": markdown, "meta": meta}
    elif request["kind"] == "anydoc":
        from openviking.parse.parsers.anydoc_converter import AnyDocConverter

        result = asdict(AnyDocConverter().convert(
            source, resource_name=request["resource_name"], storage=storage,
            max_table_rows=request["max_table_rows"],
        ))
    else:
        raise ValueError("Unsupported document conversion kind")
    _publish_media(storage.media_dir, Path(request["media_dir"]))
    return result


def main() -> int:
    request_file, result_file = map(Path, sys.argv[1:3])
    request = json.loads(request_file.read_text(encoding="utf-8"))
    logging.disable(logging.CRITICAL)
    try:
        _protect_worker(request["memory_bytes"], request["parent_pid"])
        result = _convert(request)
        payload = {
            "status": "ok", "result": result,
            "stats": {"peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        }
    except BaseException as exc:
        payload = {"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)[:1000]}}
    result_file.write_text(json.dumps(payload), encoding="utf-8")
    result_file.chmod(0o600)
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
