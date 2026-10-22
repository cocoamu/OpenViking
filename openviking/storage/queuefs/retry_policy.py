"""Bounded retry policy for durable queue deliveries."""
import os
DEFAULT_MAX_ATTEMPTS = 3


def max_attempts() -> int:
    try:
        return max(1, int(os.getenv("OPENVIKING_QUEUE_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_ATTEMPTS


def failure_category(exc: BaseException) -> str:
    text = str(exc).lower()
    if isinstance(exc, (TimeoutError, asyncio_timeout_error())) or "timeout" in text:
        return "timeout"
    if any(
        word in text
        for word in ("malformed", "invalid", "magic bytes", "bad request", "status 400")
    ):
        return "malformed"
    return "upstream"


def asyncio_timeout_error():
    import asyncio
    return asyncio.TimeoutError


def should_retry(attempt: int, category: str) -> bool:
    return category == "upstream" and attempt + 1 < max_attempts()
