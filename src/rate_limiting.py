"""Rate limiting backends for the FastAPI service.

Provides an optional Redis-backed fixed-window limiter for multi-instance
deployments and an in-memory fallback for local development.
"""

from __future__ import annotations

import time
from collections import deque

from src.utils import get_logger

LOGGER = get_logger(__name__)

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    redis = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False


class InMemoryRateLimiter:
    """Simple per-process sliding-window rate limiter."""

    def __init__(self, buckets: dict[str, deque[float]]) -> None:
        self._buckets = buckets

    def is_rate_limited(
        self,
        client_id: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int | None]:
        now = time.monotonic()
        window_start = now - window_seconds
        bucket = self._buckets[client_id]

        while bucket and bucket[0] < window_start:
            bucket.popleft()

        if len(bucket) >= limit:
            retry_after = max(1, int(bucket[0] + window_seconds - now)) if bucket else 1
            return True, retry_after

        bucket.append(now)
        return False, None


class RedisRateLimiter:
    """Redis-backed fixed-window rate limiter for distributed deployments."""

    def __init__(self, redis_url: str) -> None:
        if not REDIS_AVAILABLE:
            raise RuntimeError("redis package is not installed.")
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def ping(self) -> bool:
        """Check whether Redis is reachable."""
        return bool(self._client.ping())

    def is_rate_limited(
        self,
        client_id: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int | None]:
        now = int(time.time())
        window_id = now // window_seconds
        key = f"churn:rate_limit:{client_id}:{window_id}"
        count = int(self._client.incr(key))
        if count == 1:
            self._client.expire(key, window_seconds)

        if count > limit:
            retry_after = max(1, window_seconds - (now % window_seconds))
            return True, retry_after

        return False, None


def build_redis_rate_limiter(redis_url: str) -> RedisRateLimiter | None:
    """Create a Redis limiter when configured and reachable."""
    if not redis_url:
        return None
    if not REDIS_AVAILABLE:
        LOGGER.warning("CHURN_REDIS_URL is set but redis is not installed. Falling back to memory.")
        return None

    try:
        limiter = RedisRateLimiter(redis_url)
        limiter.ping()
        LOGGER.info("Redis-backed rate limiting enabled.")
        return limiter
    except Exception as exc:  # pragma: no cover - depends on runtime Redis availability
        LOGGER.warning("Redis unavailable for rate limiting (%s). Falling back to memory.", exc)
        return None