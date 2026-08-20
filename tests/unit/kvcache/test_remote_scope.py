"""NE-183 remote KV tenant/authentication fixtures."""

from __future__ import annotations

import httpx

from agent_utilities.core.http_client import create_http_client
from agent_utilities.kvcache import EpistemicGraphKVBackend, KvCacheConfig

BASE = "http://kv.test"


def _client(handler):  # noqa: ANN001 - focused MockTransport fixture
    return create_http_client(
        base_url=BASE,
        headers={"Authorization": "Bearer worker-token"},
        transport=httpx.MockTransport(handler),
    )


def test_strict_remote_scope_denies_missing_auth_and_bindings() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201)

    # An injected client with no Authorization is not made eligible merely by
    # naming a tenant or putting a token in an unrelated config object.
    client = create_http_client(base_url=BASE, transport=httpx.MockTransport(handler))
    backend = EpistemicGraphKVBackend(
        KvCacheConfig(
            base_url=BASE,
            tenant_ref="tenant-a",
            principal_ref="worker-a",
            token="not-attached-to-injected-client",
            require_tenant_scope=True,
        ),
        client=client,
    )
    assert backend.put("same-key", b"secret") is False
    assert backend.get("same-key") is None
    assert requests == []
    client.close()


def test_tenant_scope_salts_keys_and_binds_worker_headers() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201 if request.method == "PUT" else 404)

    a_client = _client(handler)
    b_client = _client(handler)
    a = EpistemicGraphKVBackend(
        KvCacheConfig(
            base_url=BASE,
            tenant_ref="tenant-a",
            principal_ref="worker-a",
            require_tenant_scope=True,
        ),
        client=a_client,
    )
    b = EpistemicGraphKVBackend(
        KvCacheConfig(
            base_url=BASE,
            tenant_ref="tenant-b",
            principal_ref="worker-b",
            require_tenant_scope=True,
        ),
        client=b_client,
    )
    assert a.put("same-logical-key", b"payload") is True
    assert b.get("same-logical-key") is None
    assert len(requests) == 2
    assert requests[0].url.path != requests[1].url.path
    assert requests[0].headers["X-Epistemic-Tenant"] == "tenant-a"
    assert requests[0].headers["X-Epistemic-Principal"] == "worker-a"
    assert requests[1].headers["X-Epistemic-Tenant"] == "tenant-b"
    assert requests[1].headers["X-Epistemic-Principal"] == "worker-b"
    a_client.close()
    b_client.close()
