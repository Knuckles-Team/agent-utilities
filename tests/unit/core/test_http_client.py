"""Tests for the canonical outbound HTTP client factory.

Every inline ``httpx.Client(...)`` / ``httpx.AsyncClient(...)`` construction
is being strangled onto ``agent_utilities.core.http_client`` — these tests pin
the unified defaults (finite timeout, verify=True, standard headers) and the
optional ResiliencePolicy-backed transport retry.
"""

from __future__ import annotations

import httpx
import pytest

from agent_utilities.core.http_client import (
    DEFAULT_HTTP_TIMEOUT_S,
    create_async_http_client,
    create_http_client,
    http_retry_policy,
    standard_headers,
)

# --------------------------------------------------------------------------- #
# unified defaults
# --------------------------------------------------------------------------- #


def test_default_timeout_is_finite():
    with create_http_client() as client:
        assert client.timeout == httpx.Timeout(DEFAULT_HTTP_TIMEOUT_S)


def test_infinite_timeout_rejected():
    with pytest.raises(ValueError, match="finite timeout"):
        create_http_client(timeout=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite timeout"):
        create_async_http_client(timeout=None)  # type: ignore[arg-type]


def test_verify_defaults_true(monkeypatch):
    captured: dict = {}
    real_client = httpx.Client

    def _capture(**kwargs):
        captured.update(kwargs)
        return real_client(**kwargs)

    monkeypatch.setattr(httpx, "Client", _capture)
    with create_http_client():
        pass
    assert captured["verify"] is True


def test_timeout_accepts_httpx_timeout_object():
    with create_http_client(timeout=httpx.Timeout(120.0)) as client:
        assert client.timeout == httpx.Timeout(120.0)


def test_standard_headers_applied_and_caller_wins():
    assert standard_headers()["User-Agent"].startswith("agent-utilities/")
    with create_http_client() as client:
        assert client.headers["user-agent"].startswith("agent-utilities/")
    with create_http_client(headers={"User-Agent": "custom/1", "X-Extra": "y"}) as c:
        assert c.headers["user-agent"] == "custom/1"
        assert c.headers["x-extra"] == "y"


def test_httpx_kwargs_passthrough_limits_and_transport():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": 1}))
    limits = httpx.Limits(max_connections=7)
    with create_http_client(transport=transport, limits=limits) as client:
        resp = client.get("http://unit.test/")
        assert resp.json() == {"ok": 1}


# --------------------------------------------------------------------------- #
# ResiliencePolicy-backed retry
# --------------------------------------------------------------------------- #


def _flaky_transport(failures: int) -> httpx.MockTransport:
    state = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] <= failures:
            raise httpx.ConnectError("transient connect failure", request=request)
        return httpx.Response(200, json={"attempts": state["n"]})

    return httpx.MockTransport(_handler)


def test_sync_retry_recovers_from_transport_failures():
    policy = http_retry_policy(max_attempts=3, backoff_base_s=0.0, jitter=False)
    with create_http_client(transport=_flaky_transport(2), retry=policy) as client:
        resp = client.get("http://unit.test/")
        assert resp.json() == {"attempts": 3}


def test_sync_retry_exhaustion_raises_original_error():
    policy = http_retry_policy(max_attempts=2, backoff_base_s=0.0, jitter=False)
    with create_http_client(transport=_flaky_transport(5), retry=policy) as client:
        with pytest.raises(httpx.ConnectError):
            client.get("http://unit.test/")


def test_retry_does_not_retry_http_status_errors():
    calls = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503)

    policy = http_retry_policy(max_attempts=3, backoff_base_s=0.0, jitter=False)
    transport = httpx.MockTransport(_handler)
    with create_http_client(transport=transport, retry=policy) as client:
        resp = client.get("http://unit.test/")
    # A response (even 5xx) is not a transport failure: exactly one call.
    assert resp.status_code == 503
    assert calls["n"] == 1


async def test_async_retry_recovers_from_transport_failures():
    policy = http_retry_policy(max_attempts=3, backoff_base_s=0.0, jitter=False)
    async with create_async_http_client(
        transport=_flaky_transport(2), retry=policy
    ) as client:
        resp = await client.get("http://unit.test/")
        assert resp.json() == {"attempts": 3}


async def test_async_client_defaults():
    async with create_async_http_client() as client:
        assert client.timeout == httpx.Timeout(DEFAULT_HTTP_TIMEOUT_S)
        assert client.headers["user-agent"].startswith("agent-utilities/")
