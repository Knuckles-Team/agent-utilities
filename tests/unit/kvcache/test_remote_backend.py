"""Unit tests for the epistemic-graph remote KV-cache connector.

CONCEPT:AU-KG.backend.kvcache-vllm-connector — exercises the LMCache/vLLM remote-backend contract
(:class:`EpistemicGraphKVBackend`) against a MOCK HTTP server implemented with
:class:`httpx.MockTransport`, so no live engine is needed. Covers: get/put/
contains round-trip, 404 → None miss, bearer-token propagation, error →
graceful miss (never crash the inference path), and stats parsing.
"""

from __future__ import annotations

import json

import httpx
import pytest

from agent_utilities.core.http_client import create_http_client
from agent_utilities.kvcache import (
    EpistemicGraphKVBackend,
    KvCacheConfig,
    KvCacheStats,
)
from agent_utilities.kvcache.config import _addr_to_base_url

BASE = "http://kv.test"


class _FakeKvServer:
    """In-memory stand-in for the EG-187 ``/kv`` HTTP surface (CONCEPT:AU-KG.backend.kvcache-vllm-connector).

    Implements the wire contract exactly: PUT stores verbatim under the opaque
    key (201 new / 200 dedup), GET returns bytes or 404, HEAD probes existence,
    ``/exists`` returns JSON, ``/stats`` returns occupancy counters.
    """

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.requests: list[httpx.Request] = []
        self.get_hits = 0
        self.get_misses = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path

        if path == "/kv/stats":
            body = {
                "unique_blocks": len(self.store),
                "total_refs": len(self.store),
                "resident_bytes": sum(len(v) for v in self.store.values()),
                "logical_bytes": sum(len(v) for v in self.store.values()),
                "dedup_savings_bytes": 0,
                "dedup_hits": 0,
                "get_hits": self.get_hits,
                "get_misses": self.get_misses,
            }
            return httpx.Response(200, json=body)

        if path.endswith("/exists"):
            key = path[len("/kv/") : -len("/exists")]
            return httpx.Response(200, json={"hash": key, "exists": key in self.store})

        key = path[len("/kv/") :]
        if request.method == "PUT":
            new = key not in self.store
            self.store[key] = request.content
            return httpx.Response(201 if new else 200)
        if request.method == "HEAD":
            return httpx.Response(200 if key in self.store else 404)
        if request.method == "GET":
            if key in self.store:
                self.get_hits += 1
                return httpx.Response(
                    200,
                    content=self.store[key],
                    headers={"Content-Type": "application/octet-stream"},
                )
            self.get_misses += 1
            return httpx.Response(404)
        return httpx.Response(405)


def _backend(server: _FakeKvServer, **client_kwargs: object) -> EpistemicGraphKVBackend:
    client = create_http_client(
        base_url=BASE,
        transport=httpx.MockTransport(server.handler),
        **client_kwargs,  # type: ignore[arg-type]
    )
    return EpistemicGraphKVBackend(KvCacheConfig(base_url=BASE), client=client)


def test_put_get_contains_round_trip_kg_2_306() -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — put then get returns the same bytes; contains flips."""
    server = _FakeKvServer()
    backend = _backend(server)

    key, blob = "token-hash-abc", b"\x00\x01paged-kv-block\xff"
    assert backend.contains(key) is False
    assert backend.get(key) is None

    assert backend.put(key, blob) is True
    assert backend.contains(key) is True
    assert backend.get(key) == blob


def test_put_dedup_hit_returns_true_kg_2_306() -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — a re-PUT (200 dedup) is still an accepted store."""
    server = _FakeKvServer()
    backend = _backend(server)
    backend.put("H", b"page")
    # Second PUT of the same key is a dedup hit (200) — connector reports success.
    assert backend.put("H", b"page") is True


def test_get_miss_returns_none_kg_2_306() -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — a 404 maps to None (compute-from-scratch)."""
    server = _FakeKvServer()
    backend = _backend(server)
    assert backend.get("never-stored") is None


def test_exists_endpoint_probe_kg_2_306() -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — GET /kv/<hash>/exists parses the JSON flag."""
    server = _FakeKvServer()
    backend = _backend(server)
    assert backend.exists("H") is False
    backend.put("H", b"page")
    assert backend.exists("H") is True


def test_bearer_token_sent_when_configured_kg_2_306() -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — every request carries Authorization: Bearer <token>."""
    server = _FakeKvServer()
    client = create_http_client(
        base_url=BASE,
        headers={"Authorization": "Bearer sekret"},
        transport=httpx.MockTransport(server.handler),
    )
    backend = EpistemicGraphKVBackend(
        KvCacheConfig(base_url=BASE, token="sekret"), client=client
    )
    backend.put("H", b"page")
    backend.get("H")

    assert server.requests, "expected the mock server to record requests"
    for req in server.requests:
        assert req.headers.get("Authorization") == "Bearer sekret"


def test_build_client_sets_bearer_header_kg_2_306() -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — the owned client is built with the bearer header."""
    backend = EpistemicGraphKVBackend(KvCacheConfig(token="abc123"))
    try:
        assert backend._client.headers.get("Authorization") == "Bearer abc123"
    finally:
        backend.close()


def test_no_token_means_no_auth_header_kg_2_306() -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — anonymous config sends no Authorization header."""
    backend = EpistemicGraphKVBackend(KvCacheConfig())
    try:
        assert "Authorization" not in backend._client.headers
    finally:
        backend.close()


def test_get_transport_error_degrades_to_miss_kg_2_306() -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — a transport error is a miss, never a raised exception."""

    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("engine unreachable")

    client = create_http_client(base_url=BASE, transport=httpx.MockTransport(boom))
    backend = EpistemicGraphKVBackend(KvCacheConfig(base_url=BASE), client=client)
    # None of these may raise — the inference path must survive an engine outage.
    assert backend.get("H") is None
    assert backend.contains("H") is False
    assert backend.exists("H") is False
    assert backend.put("H", b"page") is False
    assert backend.stats() == KvCacheStats()


def test_unexpected_status_is_miss_kg_2_306() -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — a 500 on get/put degrades, not crashes."""

    def five_hundred(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = create_http_client(
        base_url=BASE, transport=httpx.MockTransport(five_hundred)
    )
    backend = EpistemicGraphKVBackend(KvCacheConfig(base_url=BASE), client=client)
    assert backend.get("H") is None
    assert backend.put("H", b"page") is False


def test_stats_parse_kg_2_306() -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — GET /kv/stats parses into a typed KvCacheStats."""
    server = _FakeKvServer()
    backend = _backend(server)
    backend.put("A", b"aaaa")
    backend.put("B", b"bb")
    backend.get("A")  # a hit
    backend.get("missing")  # a miss

    stats = backend.stats()
    assert isinstance(stats, KvCacheStats)
    assert stats.unique_blocks == 2
    assert stats.resident_bytes == 6
    assert stats.get_hits == 1
    assert stats.get_misses == 1


def test_stats_malformed_json_degrades_kg_2_306() -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — non-JSON /kv/stats returns zeroed stats, no raise."""

    def bad_json(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    client = create_http_client(base_url=BASE, transport=httpx.MockTransport(bad_json))
    backend = EpistemicGraphKVBackend(KvCacheConfig(base_url=BASE), client=client)
    assert backend.stats() == KvCacheStats()


def test_key_is_percent_encoded_kg_2_306() -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — reserved chars in an opaque key are URL-encoded."""
    server = _FakeKvServer()
    backend = _backend(server)
    key = "prefix/with spaces+slash"
    backend.put(key, b"blob")
    # Round-trips despite reserved characters in the token-hash key.
    assert backend.get(key) == b"blob"
    put_req = server.requests[0]
    assert (
        "%2F" in put_req.url.raw_path.decode() or "%2f" in put_req.url.raw_path.decode()
    )


def test_context_manager_closes_owned_client_kg_2_306() -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — the connector closes a client it owns on exit."""
    with EpistemicGraphKVBackend(KvCacheConfig()) as backend:
        client = backend._client
    assert client.is_closed is True


def test_injected_client_not_closed_kg_2_306() -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — an injected client's lifecycle is the caller's."""
    server = _FakeKvServer()
    client = create_http_client(
        base_url=BASE, transport=httpx.MockTransport(server.handler)
    )
    with EpistemicGraphKVBackend(KvCacheConfig(base_url=BASE), client=client):
        pass
    assert client.is_closed is False
    client.close()


@pytest.mark.parametrize(
    ("addr", "expected"),
    [
        ("127.0.0.1:9130", "http://127.0.0.1:9130"),
        ("9130", "http://127.0.0.1:9130"),
        ("engine-host:9200", "http://engine-host:9200"),
        ("http://kv.internal:9130", "http://kv.internal:9130"),
        ("1", "http://127.0.0.1:9130"),
        ("", "http://127.0.0.1:9130"),
    ],
)
def test_addr_to_base_url_kg_2_306(addr: str, expected: str) -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — EG-187 bind values coerce to a client base URL."""
    assert _addr_to_base_url(addr) == expected


def test_config_from_env_kg_2_306(monkeypatch: pytest.MonkeyPatch) -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — from_env mirrors the engine EG-187 variables."""
    monkeypatch.setenv("EPISTEMIC_GRAPH_KVCACHE_ADDR", "engine:9130")
    monkeypatch.setenv("EPISTEMIC_GRAPH_KVCACHE_TOKEN", "tok")
    monkeypatch.setenv("EPISTEMIC_GRAPH_KVCACHE_TIMEOUT_S", "0.5")
    monkeypatch.delenv("EPISTEMIC_GRAPH_KVCACHE_URL", raising=False)

    cfg = KvCacheConfig.from_env()
    assert cfg.base_url == "http://engine:9130"
    assert cfg.token == "tok"
    assert cfg.timeout_s == 0.5


def test_config_from_env_explicit_url_wins_kg_2_306(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — an explicit URL overrides the derived ADDR."""
    monkeypatch.setenv("EPISTEMIC_GRAPH_KVCACHE_URL", "http://explicit:1234/")
    monkeypatch.setenv("EPISTEMIC_GRAPH_KVCACHE_ADDR", "ignored:9130")
    cfg = KvCacheConfig.from_env()
    assert cfg.base_url == "http://explicit:1234"


def test_stats_json_module_used_kg_2_306() -> None:
    """CONCEPT:AU-KG.backend.kvcache-vllm-connector — smoke that the JSON contract shape parses fully."""
    payload = {
        "unique_blocks": 128,
        "total_refs": 512,
        "resident_bytes": 4194304,
        "logical_bytes": 16777216,
        "dedup_savings_bytes": 12582912,
        "dedup_hits": 384,
        "get_hits": 900,
        "get_misses": 40,
    }
    stats = KvCacheStats.model_validate(json.loads(json.dumps(payload)))
    assert stats.dedup_savings_bytes == 12582912
    assert stats.total_refs == 512


# ── auth precedence: JWT (OIDC) first, static token fallback, else anonymous ──
def _has_auth_header(client: httpx.Client) -> bool:
    return any(k.lower() == "authorization" for k in client.headers)


def test_static_token_used_when_oidc_absent(monkeypatch):
    """OIDC not configured ⇒ the connector falls back to the static
    EPISTEMIC_GRAPH_KVCACHE_TOKEN bearer (the documented OpenBao-sourced option)."""
    monkeypatch.setattr(
        "agent_utilities.mcp.client_credentials.child_auth", lambda existing: None
    )
    b = EpistemicGraphKVBackend(KvCacheConfig(base_url=BASE, token="static-tok"))
    assert b._client.headers.get("Authorization") == "Bearer static-tok"
    b.close()


def test_oidc_auth_takes_precedence_over_static_token(monkeypatch):
    """OIDC configured ⇒ the self-refreshing ClientCredentialsAuth is used
    (per-request bearer), and NO frozen static Authorization header is baked in —
    even when a static token is also present. JWT-first."""

    class _FakeAuth(httpx.Auth):
        def auth_flow(self, request):
            request.headers["Authorization"] = "Bearer minted-jwt"
            yield request

    fake = _FakeAuth()
    monkeypatch.setattr(
        "agent_utilities.mcp.client_credentials.child_auth", lambda existing: fake
    )
    b = EpistemicGraphKVBackend(KvCacheConfig(base_url=BASE, token="static-tok"))
    assert not _has_auth_header(b._client)  # no frozen static header
    assert b._client.auth is fake  # per-request refreshing auth attached
    b.close()
