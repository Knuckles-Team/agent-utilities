"""Per-tenant token-bucket rate limiting for the API gateway.

CONCEPT:OS-5.23 — Gateway Middle-Tier Hardening.

Pure-ASGI middleware mounted by
:func:`agent_utilities.gateway.graph_api.register_graph_routes` INSIDE the
OS-5.14 :class:`~agent_utilities.security.request_identity.ActorIdentityMiddleware`
so the server-minted :class:`~agent_utilities.security.brain_context.ActorContext`
is already in scope. Bucket key precedence:

1. ``ActorContext.tenant_id`` (multi-tenant isolation)
2. authenticated ``actor_id`` (per-service/user when no tenant claim)
3. client IP (unauthenticated legacy traffic)

Disabled by default (``GATEWAY_RATE_LIMIT=0``). The configured rate is the
sustained requests/second; ``GATEWAY_RATE_BURST`` is the bucket capacity
(default 2× rate). Rejections are ``429`` with a ``Retry-After`` header and
are counted on ``agent_utilities_gateway_rate_limited_total{tenant}``.

Scope note (documented, deliberate): buckets are in-memory and PER-PROCESS.
With ``GATEWAY_WORKERS=N`` or N replicas the effective limit is N× the
configured rate. Precise distributed limiting is a later
(state-externalization) concern — see ``docs/architecture/gateway_scaling.md``.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from typing import Any

from agent_utilities.observability.gateway_metrics import GATEWAY_RATE_LIMITED
from agent_utilities.security.brain_context import current_actor
from agent_utilities.security.request_identity import HEALTH_PATHS

logger = logging.getLogger(__name__)

# Liveness probes and Prometheus scrapes must never be throttled.
EXEMPT_PATHS: frozenset[str] = HEALTH_PATHS | {"/metrics"}

# Bucket-map hygiene: prune idle buckets so hostile/unbounded key churn
# (e.g. IP-keyed traffic) cannot grow the map without bound.
_PRUNE_THRESHOLD = 4096
_PRUNE_IDLE_SECONDS = 600.0


class _TokenBucket:
    """Classic token bucket: ``capacity`` tokens, refilled at ``rate``/sec."""

    __slots__ = ("capacity", "rate", "tokens", "updated")

    def __init__(self, rate: float, capacity: float, now: float) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity  # new buckets start full → burst headroom
        self.updated = now

    def consume(self, now: float) -> tuple[bool, float]:
        """Take one token. Returns ``(allowed, retry_after_seconds)``."""
        elapsed = max(0.0, now - self.updated)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.updated = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True, 0.0
        return False, (1.0 - self.tokens) / self.rate


class GatewayRateLimitMiddleware:
    """Pure-ASGI per-tenant token-bucket rate limiter (CONCEPT:OS-5.23)."""

    def __init__(
        self,
        app: Any,
        rate: float | None = None,
        burst: float | None = None,
    ) -> None:
        from agent_utilities.core.config import config

        self.app = app
        self.rate = float(rate if rate is not None else config.gateway_rate_limit or 0)
        raw_burst = float(
            burst if burst is not None else config.gateway_rate_burst or 0
        )
        # Default burst = 2× sustained rate; never below 1 token or the
        # very first request would be rejected.
        self.burst = raw_burst if raw_burst > 0 else max(self.rate * 2.0, 1.0)
        self._buckets: dict[str, _TokenBucket] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _bucket_key(self, scope: dict[str, Any]) -> str:
        actor = current_actor()
        if actor.tenant_id:
            return actor.tenant_id
        if actor.authenticated and actor.actor_id:
            return actor.actor_id
        client = scope.get("client")
        return str(client[0]) if client else "unknown"

    def _consume(self, key: str) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= _PRUNE_THRESHOLD:
                    cutoff = now - _PRUNE_IDLE_SECONDS
                    self._buckets = {
                        k: b for k, b in self._buckets.items() if b.updated >= cutoff
                    }
                bucket = self._buckets[key] = _TokenBucket(self.rate, self.burst, now)
            return bucket.consume(now)

    # ------------------------------------------------------------------
    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or self.rate <= 0
            or scope.get("path", "") in EXEMPT_PATHS
        ):
            await self.app(scope, receive, send)
            return

        key = self._bucket_key(scope)
        allowed, retry_after = self._consume(key)
        if allowed:
            await self.app(scope, receive, send)
            return

        GATEWAY_RATE_LIMITED.labels(tenant=key).inc()
        retry = max(1, math.ceil(retry_after))
        body = json.dumps(
            {
                "error": "rate limit exceeded",
                "tenant": key,
                "retry_after": retry,
            }
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"retry-after", str(retry).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


__all__ = ["EXEMPT_PATHS", "GatewayRateLimitMiddleware"]
