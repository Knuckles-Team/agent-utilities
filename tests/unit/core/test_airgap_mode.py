"""Tests for AIRGAP_MODE (CONCEPT:AU-OS.deployment.airgap-mode).

``reports/surpass-6mo/04-five-intersections.md`` (§3) found the sovereign/
self-hosted substrate strong but named two concrete gaps: no named air-gap
mode/flag, and no gate proving it actually blocks external calls. These
tests ARE that gate — they assert the ONE flag (``AIRGAP_MODE``) makes the
canonical outbound HTTP factory (and the LLM client constructor built on
top of it) refuse any request to a non-local host, fail-closed, before the
request is ever sent; and that everything is unchanged when the flag is off.
"""

from __future__ import annotations

import httpx
import pytest

from agent_utilities.core.http_client import (
    AirgapViolation,
    airgap_guard_transport,
    create_async_http_client,
    create_http_client,
    is_local_host,
)

# --------------------------------------------------------------------------- #
# is_local_host — pure classification, no I/O
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "LOCALHOST",
        "127.0.0.1",
        "::1",
        "10.0.0.18",  # RFC1918 class A
        "172.16.4.4",  # RFC1918 class B
        "192.168.1.1",  # RFC1918 class C
        "169.254.1.1",  # link-local
    ],
)
def test_is_local_host_true_for_loopback_private_link_local(host):
    assert is_local_host(host) is True


@pytest.mark.parametrize(
    "host",
    [
        "example.com",
        "8.8.8.8",
        "93.184.216.34",
        "vllm.arpa",  # hostname — no DNS lookup performed, treated as non-local
        "",
    ],
)
def test_is_local_host_false_for_external_or_unverifiable(host):
    assert is_local_host(host) is False


# --------------------------------------------------------------------------- #
# airgap_guard_transport — no-op when the flag is off
# --------------------------------------------------------------------------- #


def test_airgap_guard_transport_is_noop_when_flag_off(monkeypatch):
    monkeypatch.delenv("AIRGAP_MODE", raising=False)
    assert airgap_guard_transport(None, is_async=False) is None
    sentinel = httpx.MockTransport(lambda request: httpx.Response(200))
    assert airgap_guard_transport(sentinel, is_async=False) is sentinel


def test_airgap_guard_transport_wraps_when_flag_on(monkeypatch):
    monkeypatch.setenv("AIRGAP_MODE", "true")
    sentinel = httpx.MockTransport(lambda request: httpx.Response(200))
    wrapped = airgap_guard_transport(sentinel, is_async=False)
    assert wrapped is not sentinel
    assert wrapped is not None


# --------------------------------------------------------------------------- #
# create_http_client / create_async_http_client — end-to-end gate
# --------------------------------------------------------------------------- #


def _ok_transport() -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": 1}))


def test_airgap_off_by_default_external_host_reachable(monkeypatch):
    monkeypatch.delenv("AIRGAP_MODE", raising=False)
    with create_http_client(transport=_ok_transport()) as client:
        resp = client.get("https://example.com/")
        assert resp.json() == {"ok": 1}


def test_airgap_on_blocks_external_host(monkeypatch):
    monkeypatch.setenv("AIRGAP_MODE", "true")
    with create_http_client(transport=_ok_transport()) as client:
        with pytest.raises(AirgapViolation, match="example.com"):
            client.get("https://example.com/")


def test_airgap_on_allows_loopback_and_private_hosts(monkeypatch):
    monkeypatch.setenv("AIRGAP_MODE", "true")
    with create_http_client(transport=_ok_transport()) as client:
        resp = client.get("http://127.0.0.1:8000/health")
        assert resp.json() == {"ok": 1}
        resp = client.get("http://10.0.0.18:8000/v1/models")
        assert resp.json() == {"ok": 1}
        resp = client.get("http://localhost:11434/api/tags")
        assert resp.json() == {"ok": 1}


def test_airgap_composes_with_retry(monkeypatch):
    """The air-gap check must fire even when a ResiliencePolicy retry transport
    is also configured — and must NOT be retried (it's not a transport error)."""
    from agent_utilities.core.http_client import http_retry_policy

    monkeypatch.setenv("AIRGAP_MODE", "true")
    calls = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200)

    policy = http_retry_policy(max_attempts=3, backoff_base_s=0.0, jitter=False)
    transport = httpx.MockTransport(_handler)
    with create_http_client(transport=transport, retry=policy) as client:
        with pytest.raises(AirgapViolation):
            client.get("https://example.com/")
    assert calls["n"] == 0  # blocked before the wrapped/retried transport ever ran


async def test_async_airgap_on_blocks_external_host(monkeypatch):
    monkeypatch.setenv("AIRGAP_MODE", "true")
    async with create_async_http_client(transport=_ok_transport()) as client:
        with pytest.raises(AirgapViolation, match="example.com"):
            await client.get("https://example.com/")


async def test_async_airgap_on_allows_local(monkeypatch):
    monkeypatch.setenv("AIRGAP_MODE", "true")
    async with create_async_http_client(transport=_ok_transport()) as client:
        resp = await client.get("http://127.0.0.1:9000/")
        assert resp.json() == {"ok": 1}


# --------------------------------------------------------------------------- #
# model_factory wiring — the LLM-call egress path also gets the guard
# --------------------------------------------------------------------------- #


def test_model_factory_wires_airgap_guard_into_client(monkeypatch):
    from agent_utilities.core import model_factory

    captured: dict = {}

    class _SpyClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _SpyClient)
    monkeypatch.setattr(
        model_factory,
        "get_model_config",
        lambda mid=None: {
            "id": "internal",
            "provider": "openai",
            "base_url": "http://127.0.0.1:8000/v1",
        },
    )
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")
    monkeypatch.setenv("AIRGAP_MODE", "true")

    model_factory.create_model(provider="openai", model_id="internal")

    assert captured.get("transport") is not None


def test_model_factory_no_guard_transport_when_airgap_off(monkeypatch):
    from agent_utilities.core import model_factory

    captured: dict = {}

    class _SpyClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _SpyClient)
    monkeypatch.setattr(
        model_factory,
        "get_model_config",
        lambda mid=None: {
            "id": "internal",
            "provider": "openai",
            "base_url": "http://127.0.0.1:8000/v1",
        },
    )
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")
    monkeypatch.delenv("AIRGAP_MODE", raising=False)

    model_factory.create_model(provider="openai", model_id="internal")

    assert captured.get("transport") is None
