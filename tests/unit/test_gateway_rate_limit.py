"""Tests for the per-tenant token-bucket rate limiter (CONCEPT:AU-OS.observability.no-op-without-metrics).

Covers: bucket math, burst capacity, per-tenant isolation, the
disabled-by-default contract, the 429 response shape (Retry-After + JSON
body), exemption of /metrics and health routes, and bucket-key fallback
(tenant → authenticated actor → client IP).
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.gateway.rate_limit import (
    EXEMPT_PATHS,
    GatewayRateLimitMiddleware,
    _TokenBucket,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

# ---------------------------------------------------------------------------
# ASGI plumbing helpers
# ---------------------------------------------------------------------------


async def _ok_app(scope, receive, send):  # noqa: ARG001
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


async def _call(mw, path="/api/graph/query", client=("10.0.0.9", 1234)):
    sent: list[dict] = []

    async def send(msg):
        sent.append(msg)

    async def receive():
        return {"type": "http.request"}

    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [],
        "client": client,
    }
    await mw(scope, receive, send)
    return sent


def _status(sent):
    return sent[0]["status"]


def _actor(tenant="", actor_id="svc", authenticated=True):
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("user",),
        tenant_id=tenant,
        authenticated=authenticated,
    )


# ---------------------------------------------------------------------------
# Token-bucket math
# ---------------------------------------------------------------------------


class TestTokenBucket:
    def test_burst_then_refill(self):
        bucket = _TokenBucket(rate=2.0, capacity=4.0, now=100.0)
        # full bucket: 4 instant requests pass
        for _ in range(4):
            allowed, _retry = bucket.consume(100.0)
            assert allowed
        allowed, retry = bucket.consume(100.0)
        assert not allowed
        assert retry == pytest.approx(0.5)  # 1 token at 2/sec
        # after 1s, 2 tokens refilled
        assert bucket.consume(101.0)[0]
        assert bucket.consume(101.0)[0]
        assert not bucket.consume(101.0)[0]

    def test_refill_caps_at_capacity(self):
        bucket = _TokenBucket(rate=10.0, capacity=3.0, now=0.0)
        bucket.consume(0.0)
        for _ in range(3):  # long idle → still only `capacity` tokens
            allowed, _ = bucket.consume(1000.0)
            assert allowed
        assert not bucket.consume(1000.0)[0]

    def test_clock_regression_is_safe(self):
        bucket = _TokenBucket(rate=1.0, capacity=1.0, now=100.0)
        assert bucket.consume(100.0)[0]
        allowed, retry = bucket.consume(99.0)  # monotonic should prevent this anyway
        assert not allowed
        assert retry > 0


# ---------------------------------------------------------------------------
# Middleware behaviour
# ---------------------------------------------------------------------------


class TestRateLimitMiddleware:
    async def test_disabled_by_default(self):
        # AgentConfig default GATEWAY_RATE_LIMIT=0 → pure pass-through
        mw = GatewayRateLimitMiddleware(_ok_app)
        assert mw.rate == 0
        for _ in range(20):
            assert _status(await _call(mw)) == 200

    async def test_burst_defaults_to_twice_rate(self):
        mw = GatewayRateLimitMiddleware(_ok_app, rate=5.0)
        assert mw.burst == 10.0

    async def test_explicit_burst_honoured(self):
        mw = GatewayRateLimitMiddleware(_ok_app, rate=5.0, burst=3.0)
        assert mw.burst == 3.0

    async def test_burst_then_429(self):
        mw = GatewayRateLimitMiddleware(_ok_app, rate=1.0, burst=2.0)
        with use_actor(_actor(tenant="acme")):
            assert _status(await _call(mw)) == 200
            assert _status(await _call(mw)) == 200
            sent = await _call(mw)
        assert _status(sent) == 429

    async def test_429_shape(self):
        mw = GatewayRateLimitMiddleware(_ok_app, rate=1.0, burst=1.0)
        with use_actor(_actor(tenant="acme")):
            await _call(mw)
            sent = await _call(mw)
        assert _status(sent) == 429
        headers = dict(sent[0]["headers"])
        assert headers[b"content-type"] == b"application/json"
        retry_after = int(headers[b"retry-after"])
        assert retry_after >= 1
        body = json.loads(sent[1]["body"])
        assert body["error"] == "rate limit exceeded"
        assert body["tenant"] == "acme"
        assert body["retry_after"] == retry_after

    async def test_per_tenant_isolation(self):
        mw = GatewayRateLimitMiddleware(_ok_app, rate=1.0, burst=1.0)
        with use_actor(_actor(tenant="acme")):
            assert _status(await _call(mw)) == 200
            assert _status(await _call(mw)) == 429
        # a different tenant has its own untouched bucket
        with use_actor(_actor(tenant="globex")):
            assert _status(await _call(mw)) == 200

    async def test_key_fallback_actor_id_then_ip(self):
        mw = GatewayRateLimitMiddleware(_ok_app, rate=1.0, burst=1.0)
        # authenticated, no tenant claim → keyed by actor id
        with use_actor(_actor(tenant="", actor_id="svc-a")):
            await _call(mw)
        assert "svc-a" in mw._buckets
        # unauthenticated (ambient system actor) → keyed by client IP
        await _call(mw, client=("203.0.113.7", 999))
        assert "203.0.113.7" in mw._buckets

    async def test_exempt_paths_never_limited(self):
        mw = GatewayRateLimitMiddleware(_ok_app, rate=1.0, burst=1.0)
        assert "/metrics" in EXEMPT_PATHS
        assert "/api/health" in EXEMPT_PATHS
        for _ in range(10):
            assert _status(await _call(mw, path="/metrics")) == 200
            assert _status(await _call(mw, path="/api/health")) == 200

    async def test_non_http_scope_passes_through(self):
        mw = GatewayRateLimitMiddleware(_ok_app, rate=1.0, burst=1.0)
        called = {}

        async def ws_app(scope, receive, send):  # noqa: ARG001
            called["yes"] = True

        mw.app = ws_app
        await mw({"type": "websocket", "path": "/ws"}, None, None)
        assert called.get("yes")

    async def test_rate_limited_metric_incremented(self, monkeypatch):
        from agent_utilities.gateway import rate_limit as rl_mod

        calls: list[dict] = []

        class FakeCounter:
            def labels(self, **kw):
                calls.append(kw)
                return self

            def inc(self, *a, **k):
                pass

        monkeypatch.setattr(rl_mod, "GATEWAY_RATE_LIMITED", FakeCounter())
        mw = GatewayRateLimitMiddleware(_ok_app, rate=1.0, burst=1.0)
        with use_actor(_actor(tenant="acme")):
            await _call(mw)
            await _call(mw)
        assert {"tenant": "acme"} in calls
