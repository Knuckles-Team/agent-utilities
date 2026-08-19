"""Multiplexer wiring for the remote-OAuth broker (NE-008 Deliverable 3).

Proves the per-principal path added to
``agent_utilities.mcp.multiplexer`` -- ``_resolve_remote_oauth_bearer``
(the seam invoked from ``_open_one_session``), the pool-exclusion guard in
``_start_child``, and the probe-cache exclusion in ``_cache_probe``/
``_probe_cache_hit`` -- without re-deriving the attack matrix already proven
directly against the broker in ``test_remote_oauth_broker.py``. What THIS
file is responsible for:

* a missing/expired/revoked grant fails closed at the multiplexer boundary
  (no tools discoverable, no call possible) -- never a shared/service
  credential fallback;
* a token is only ever released for the EXACT registered resource endpoint,
  never a rebound one;
* an OAuth-gated server is never pool-mounted and its probe result is never
  written into (or served from) the shared, principal-agnostic probe cache;
* the U-44/U-45 canary in ``test_remote_oauth_fail_closed.py`` still passes
  (asserted indirectly here too, by never importing the fenced names).
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.core.fleet_catalog_tables import (
    TenantLocalDiscoveryBinding,
)
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    suspend_session,
    use_session,
)
from agent_utilities.mcp import multiplexer as mod
from agent_utilities.mcp.multiplexer import MCPMultiplexer
from agent_utilities.mcp.remote_oauth_broker import (
    OAuthGrantBinding,
    OAuthProviderError,
    OAuthTokenAbsentError,
    OAuthTokenStore,
    ProviderDescriptor,
    ProviderRegistry,
    RemoteOAuthBroker,
    StoredToken,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor
from agent_utilities.security.secrets_client import SecretsBackend, SecretsClient

RESOURCE_URL = "https://protected-mcp.example.com/mcp"
REBOUND_URL = "https://attacker-controlled.example.com/mcp"


def _synthetic_value(*parts: str) -> str:
    """Build a deterministic test-only credential from scanner-safe fragments."""
    return "".join(parts)


ACCESS_TOKEN = _synthetic_value("at-", "valid-for-", "this-principal")


class FakeSecretsBackend(SecretsBackend):
    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._store.get(key)

    def set(self, key: str, value: str, **metadata) -> None:
        with self._lock:
            self._store[key] = value

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def list_keys(self) -> list[str]:
        with self._lock:
            return sorted(self._store)

    def compare_and_set(self, key: str, expected: str, value: str, **metadata) -> bool:
        with self._lock:
            if self._store.get(key) != expected:
                return False
            self._store[key] = value
            return True


def verified_actor(actor_id="user-a", tenant_id="tenant-1") -> ActorContext:
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.HUMAN,
        tenant_id=tenant_id,
        authenticated=True,
        roles=("kg:read",),
    )


def verified_session(actor: ActorContext) -> GraphSession:
    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:read"}),
        graph=actor.tenant_id,
        policy_version="test",
        audience="test",
    )


def _provider_cfg(**overrides) -> dict:
    fields = dict(
        provider_id="acme",
        resource_url=RESOURCE_URL,
        client_id="mux-public-client",
        redirect_uri="https://gateway.internal.example/oauth/callback",
        scopes=("mcp:read",),
        enabled=True,
    )
    fields.update(overrides)
    return fields


def _broker_with_stored_token(
    *, actor: ActorContext, expires_in: float = 3600.0
) -> RemoteOAuthBroker:
    registry = ProviderRegistry()
    registry.register(ProviderDescriptor(**_provider_cfg()))
    secrets_client = SecretsClient(backend=FakeSecretsBackend())
    broker = RemoteOAuthBroker(registry=registry, secrets_client=secrets_client)
    token = StoredToken(
        access_token=ACCESS_TOKEN,
        refresh_token=None,
        token_type="Bearer",
        expires_at=time.time() + expires_in,
        granted_scope="mcp:read",
        key_version=OAuthTokenStore.CURRENT_KEY_VERSION,
        audience=RESOURCE_URL,
    )
    broker.tokens.put(
        actor=actor,
        provider_id="acme",
        resource_url=RESOURCE_URL,
        audience=RESOURCE_URL,
        token=token,
    )
    return broker


@pytest.fixture(autouse=True)
def _clean_broker_singleton_cache():
    """``_REMOTE_OAUTH_BROKERS`` is a module-level, provider_id-keyed cache --
    isolate every test from whatever another test (or another provider_id
    reuse) left behind."""
    mod._REMOTE_OAUTH_BROKERS.clear()
    yield
    mod._REMOTE_OAUTH_BROKERS.clear()


# ---------------------------------------------------------------------------
# _resolve_remote_oauth_bearer — the seam, unit-level
# ---------------------------------------------------------------------------
def test_non_gated_config_returns_none():
    assert (
        mod._resolve_remote_oauth_bearer({"headers": {"X-Foo": "bar"}}, RESOURCE_URL)
        is None
    )
    assert mod._resolve_remote_oauth_bearer({}, RESOURCE_URL) is None


def test_fails_closed_for_unauthenticated_actor(monkeypatch):
    unauth = ActorContext(actor_id="anon", tenant_id="tenant-1", authenticated=False)
    # Pre-seed a fake-secrets-backed broker (like every other test here) so
    # this exercises the actor-authenticated check itself, not whatever the
    # PRODUCTION secrets-client default needs (a reachable engine) --
    # otherwise this would only run/skip based on environment availability.
    registry = ProviderRegistry()
    registry.register(ProviderDescriptor(**_provider_cfg()))
    fake_broker = RemoteOAuthBroker(
        registry=registry, secrets_client=SecretsClient(backend=FakeSecretsBackend())
    )
    mod._REMOTE_OAUTH_BROKERS["acme"] = fake_broker
    cfg = {"oauth_provider": _provider_cfg()}
    with use_actor(unauth):
        with pytest.raises(PermissionError):
            mod._resolve_remote_oauth_bearer(cfg, RESOURCE_URL)


def test_fails_closed_without_any_stored_grant(monkeypatch):
    actor = verified_actor()
    registry = ProviderRegistry()
    registry.register(ProviderDescriptor(**_provider_cfg()))
    empty_broker = RemoteOAuthBroker(
        registry=registry, secrets_client=SecretsClient(backend=FakeSecretsBackend())
    )
    mod._REMOTE_OAUTH_BROKERS["acme"] = empty_broker
    cfg = {"oauth_provider": _provider_cfg()}
    with use_actor(actor):
        with pytest.raises(OAuthTokenAbsentError):
            mod._resolve_remote_oauth_bearer(cfg, RESOURCE_URL)


def test_fails_closed_on_expired_grant():
    actor = verified_actor()
    broker = _broker_with_stored_token(actor=actor, expires_in=-10.0)  # already expired
    mod._REMOTE_OAUTH_BROKERS["acme"] = broker
    cfg = {"oauth_provider": _provider_cfg()}
    with use_actor(actor):
        with pytest.raises(OAuthTokenAbsentError):
            mod._resolve_remote_oauth_bearer(cfg, RESOURCE_URL)


def test_happy_path_returns_bound_bearer():
    actor = verified_actor()
    broker = _broker_with_stored_token(actor=actor)
    mod._REMOTE_OAUTH_BROKERS["acme"] = broker
    cfg = {"oauth_provider": _provider_cfg()}
    with use_actor(actor):
        headers = mod._resolve_remote_oauth_bearer(cfg, RESOURCE_URL)
    assert headers == {"Authorization": f"Bearer {ACCESS_TOKEN}"}


def test_grant_resolution_returns_process_owned_binding_without_bearer_persistence():
    actor = verified_actor()
    broker = _broker_with_stored_token(actor=actor)
    mod._REMOTE_OAUTH_BROKERS["acme"] = broker
    cfg = {"oauth_provider": _provider_cfg()}
    with use_actor(actor):
        resolved = mod._resolve_remote_oauth_grant(cfg, RESOURCE_URL)
        current = mod.current_remote_oauth_grant_bindings(actor)
    assert resolved is not None
    headers, binding = resolved
    assert isinstance(binding, OAuthGrantBinding)
    assert headers["Authorization"].endswith(ACCESS_TOKEN)
    assert binding in current
    assert ACCESS_TOKEN not in binding.fingerprint
    assert all(secret not in repr(binding) for secret in (ACCESS_TOKEN,))


def test_rejects_rebound_endpoint():
    """A token minted for RESOURCE_URL must never be released for a
    DIFFERENT URL -- even one on the same provider's catalog entry (e.g. a
    rebound/redirected connect target). This is the exact non-negotiable
    property: "never forward to a rebound endpoint"."""
    actor = verified_actor()
    broker = _broker_with_stored_token(actor=actor)
    mod._REMOTE_OAUTH_BROKERS["acme"] = broker
    cfg = {"oauth_provider": _provider_cfg()}
    with use_actor(actor):
        with pytest.raises(OAuthProviderError):
            mod._resolve_remote_oauth_bearer(cfg, REBOUND_URL)
        # The correctly-bound URL still works with the SAME stored grant --
        # proves the rejection above is endpoint-specific, not a broken token.
        headers = mod._resolve_remote_oauth_bearer(cfg, RESOURCE_URL)
    assert headers == {"Authorization": f"Bearer {ACCESS_TOKEN}"}


def test_cross_principal_reuse_rejected():
    """A grant stored for user-a must never be usable by user-b, even against
    the SAME provider/resource and the SAME broker instance."""
    owner = verified_actor(actor_id="user-a")
    attacker = verified_actor(actor_id="user-b")
    broker = _broker_with_stored_token(actor=owner)
    mod._REMOTE_OAUTH_BROKERS["acme"] = broker
    cfg = {"oauth_provider": _provider_cfg()}
    with use_actor(attacker):
        with pytest.raises(OAuthTokenAbsentError):
            mod._resolve_remote_oauth_bearer(cfg, RESOURCE_URL)


def test_cross_tenant_reuse_rejected():
    owner = verified_actor(actor_id="user-a", tenant_id="tenant-1")
    other_tenant = verified_actor(actor_id="user-a", tenant_id="tenant-2")
    broker = _broker_with_stored_token(actor=owner)
    mod._REMOTE_OAUTH_BROKERS["acme"] = broker
    cfg = {"oauth_provider": _provider_cfg()}
    with use_actor(other_tenant):
        with pytest.raises(OAuthTokenAbsentError):
            mod._resolve_remote_oauth_bearer(cfg, RESOURCE_URL)


# ---------------------------------------------------------------------------
# Pool exclusion — an OAuth-gated server is never pool-mounted
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_start_child_skips_oauth_gated_server(tmp_path):
    mux = MCPMultiplexer(tmp_path / "c.json")
    cfg = {"url": RESOURCE_URL, "oauth_provider": _provider_cfg()}
    result = await mux._start_child("acme-remote", cfg)
    assert result is None
    assert "acme-remote" not in mux.children


# ---------------------------------------------------------------------------
# Probe-cache exclusion
# ---------------------------------------------------------------------------
def test_cache_probe_never_stores_oauth_gated_result(tmp_path):
    mux = MCPMultiplexer(tmp_path / "c.json")
    mux._catalog = {
        "acme-remote": {"url": RESOURCE_URL, "oauth_provider": _provider_cfg()}
    }
    mux._cache_probe("acme-remote", {"tools": [{"name": "t"}], "error": None})
    assert "acme-remote" not in mux._probe_cache


def test_probe_cache_hit_always_misses_for_oauth_gated_server(tmp_path):
    mux = MCPMultiplexer(tmp_path / "c.json")
    mux._catalog = {
        "acme-remote": {"url": RESOURCE_URL, "oauth_provider": _provider_cfg()}
    }
    # Seed the cache directly (bypassing ``_cache_probe``'s own gate) to prove
    # ``_probe_cache_hit`` ALSO refuses to serve a stale/leaked entry, not just
    # that writes are suppressed.
    mux._probe_cache["acme-remote"] = {
        "tools": [],
        "error": None,
        "probed_at": time.time(),
    }
    assert mux._probe_cache_hit("acme-remote") is None


def test_probe_cache_unaffected_for_ordinary_servers(tmp_path):
    mux = MCPMultiplexer(tmp_path / "c.json")
    mux._catalog = {"plain-remote": {"url": RESOURCE_URL}}
    mux._cache_probe("plain-remote", {"tools": [], "error": None})
    assert "plain-remote" in mux._probe_cache
    assert mux._probe_cache_hit("plain-remote") is not None


# ---------------------------------------------------------------------------
# End-to-end via probe_server: fail-closed discovery + endpoint-bound header
# ---------------------------------------------------------------------------
@contextlib.asynccontextmanager
async def _streams_cm(streams):
    yield streams


class _FakeSession:
    def __init__(self, recorder: list) -> None:
        self._recorder = recorder

    async def initialize(self):
        return None

    async def list_tools(self):
        result = MagicMock()
        result.tools = []
        return result


class _FakeSessionCM:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return _FakeSession([])

    async def __aexit__(self, *a):
        return False


@pytest.fixture
def remote_transport(monkeypatch):
    """Fake streamable-http transport, recording the httpx client each
    ``create_async_http_client`` call was built with -- so a test can inspect
    the ACTUAL headers the multiplexer would have sent, without a real
    network call. Mirrors ``tests/unit/mcp/test_multiplexer_transports.py``."""
    from agent_utilities.core import http_client as http_client_mod

    recorded_clients: list[dict] = []
    real_create_async_http_client = http_client_mod.create_async_http_client

    def _recording_create_async_http_client(**kwargs):
        recorded_clients.append(kwargs)
        return real_create_async_http_client(**kwargs)

    monkeypatch.setattr(
        http_client_mod, "create_async_http_client", _recording_create_async_http_client
    )

    def _client(*args, **kwargs):
        return _streams_cm(("r", "w"))

    monkeypatch.setattr(mod, "streamable_http_client", _client)
    monkeypatch.setattr(mod, "ClientSession", _FakeSessionCM)
    return recorded_clients


@pytest.mark.asyncio
async def test_probe_server_fails_closed_without_grant_and_never_opens_a_transport(
    remote_transport, tmp_path
):
    mux = MCPMultiplexer(tmp_path / "c.json")
    cfg = {"url": RESOURCE_URL, "oauth_provider": _provider_cfg()}
    mux._catalog = {"acme-remote": cfg}
    with use_actor(verified_actor()):
        info = await mux.probe_server("acme-remote")
    assert info["tools"] == []
    assert info["error"]
    # The connect must never even have been attempted -- the grant check
    # happens before the transport is ever constructed.
    assert not remote_transport


@pytest.mark.asyncio
async def test_probe_server_with_grant_forwards_bearer_bound_to_the_endpoint(
    remote_transport, tmp_path
):
    actor = verified_actor()
    broker = _broker_with_stored_token(actor=actor)
    mod._REMOTE_OAUTH_BROKERS["acme"] = broker
    mux = MCPMultiplexer(tmp_path / "c.json")
    cfg = {"url": RESOURCE_URL, "oauth_provider": _provider_cfg()}
    mux._catalog = {"acme-remote": cfg}
    with use_actor(actor):
        info = await mux.probe_server("acme-remote")
    assert info["error"] is None
    assert len(remote_transport) == 1
    sent_headers = remote_transport[0].get("headers") or {}
    assert sent_headers.get("Authorization") == f"Bearer {ACCESS_TOKEN}"
    # Never carried a fleet-global service credential alongside the
    # per-principal grant (the two credential lifetimes are never merged).
    assert remote_transport[0].get("auth") is None


def test_private_binding_sidechannel_rejects_public_catalog_spoof(tmp_path):
    """Grant authority is not a serializable/caller-populatable catalog field."""
    mux = MCPMultiplexer(tmp_path / "c.json")
    info = {"tools": [], "skills": [], "prompts": [], "error": None}
    binding = OAuthGrantBinding(
        tenant_id="tenant-1",
        principal_id="user-a",
        provider_id="acme",
        resource_url=RESOURCE_URL,
        audience=RESOURCE_URL,
        granted_scopes=("mcp:read",),
        key_version=1,
        grant_revision="revision-a",
    )
    mux._record_discovery_binding("acme-remote", info, binding)

    # The public shape is ordinary JSON and a copied/caller-shaped dict cannot
    # inherit private authority merely by carrying the old field name.
    assert "_discovery_binding" not in info
    json.dumps(info)
    spoof = dict(info)
    spoof["_discovery_binding"] = binding
    assert mux._take_discovery_bindings({"acme-remote": spoof}) == {}

    assert mux._take_discovery_bindings({"acme-remote": info}) == {
        "acme-remote": binding
    }
    assert mux._take_discovery_bindings({"acme-remote": info}) == {}


@pytest.mark.asyncio
async def test_local_probe_mints_tenant_visibility_without_public_metadata(
    remote_transport, tmp_path
):
    actor = verified_actor(tenant_id="tenant-local")
    mux = MCPMultiplexer(tmp_path / "c.json")
    mux._catalog = {"local": {"url": RESOURCE_URL}}

    with use_actor(actor), use_session(verified_session(actor)):
        info = await mux.probe_server("local")
        assert info["error"] is None
        assert "_discovery_binding" not in info
        json.dumps(info)
        bindings = mux._take_discovery_bindings({"local": info})

    assert isinstance(bindings["local"], TenantLocalDiscoveryBinding)
    assert bindings["local"].tenant_id == "tenant-local"
    assert bindings["local"].authority == "tenant_local"
    spoof = dict(info)
    spoof["_discovery_binding"] = bindings["local"]
    assert mux._take_discovery_bindings({"local": spoof}) == {}
    other = verified_actor(actor_id="other", tenant_id="other-tenant")
    with use_actor(other), use_session(verified_session(other)):
        assert mux._take_discovery_bindings({"local": info}) == {}


def test_local_cache_binding_requires_verified_session(tmp_path):
    mux = MCPMultiplexer(tmp_path / "c.json")
    mux._catalog = {"local": {"url": RESOURCE_URL}}
    info = mux._cache_probe(
        "local", {"tools": [], "skills": [], "prompts": [], "error": None}
    )
    with suspend_session():
        assert mux._take_discovery_bindings({"local": info}) == {}
    failed = mux._cache_probe(
        "local", {"tools": [], "skills": [], "prompts": [], "error": "unreachable"}
    )
    actor = verified_actor(tenant_id="tenant-local")
    with use_actor(actor), use_session(verified_session(actor)):
        assert mux._take_discovery_bindings({"local": failed}) == {}


def test_failed_probe_purges_prior_local_binding_and_cache_authority(tmp_path):
    mux = MCPMultiplexer(tmp_path / "c.json")
    mux._catalog = {"local": {"url": RESOURCE_URL}}
    actor = verified_actor(tenant_id="tenant-local")
    with use_actor(actor), use_session(verified_session(actor)):
        info = mux._cache_probe(
            "local", {"tools": [], "skills": [], "prompts": [], "error": None}
        )
        binding = TenantLocalDiscoveryBinding(tenant_id="tenant-local")
        mux._record_discovery_binding("local", info, binding)
        assert mux._take_discovery_bindings({"local": info}) == {"local": binding}

        failed = mux._cache_probe(
            "local", {"tools": [], "skills": [], "prompts": [], "error": "unreachable"}
        )
        assert mux._take_discovery_bindings({"local": failed}) == {}
        assert not any(
            record[0] == "local"
            for record in mux._local_discovery_cache_authority.values()
        )


def test_local_cache_binding_can_be_minted_on_verified_sync_thread(tmp_path):
    mux = MCPMultiplexer(tmp_path / "c.json")
    mux._catalog = {"local": {"url": RESOURCE_URL}}
    with suspend_session():
        info = mux._cache_probe(
            "local", {"tools": [], "skills": [], "prompts": [], "error": None}
        )
    actor = verified_actor(tenant_id="tenant-local")
    with use_actor(actor), use_session(verified_session(actor)):
        mux._bind_local_discovery_bindings({"local": info})
        bindings = mux._take_discovery_bindings({"local": info})
    assert isinstance(bindings["local"], TenantLocalDiscoveryBinding)


@pytest.mark.asyncio
async def test_probe_server_result_never_cached_across_principals(
    remote_transport, tmp_path
):
    """The exact property the lane calls out: discovery legitimately varies
    per principal, so a snapshot from one caller is never served to another."""
    owner = verified_actor(actor_id="user-a")
    stranger = verified_actor(actor_id="user-b")
    broker = _broker_with_stored_token(actor=owner)
    mod._REMOTE_OAUTH_BROKERS["acme"] = broker
    mux = MCPMultiplexer(tmp_path / "c.json")
    cfg = {"url": RESOURCE_URL, "oauth_provider": _provider_cfg()}
    mux._catalog = {"acme-remote": cfg}

    with use_actor(owner):
        info_owner = await mux.probe_server("acme-remote")
    assert info_owner["error"] is None
    assert "acme-remote" not in mux._probe_cache  # never cached, even on success

    with use_actor(stranger):
        info_stranger = await mux.probe_server("acme-remote")
    # The stranger gets their OWN honest fail-closed answer -- never the
    # owner's cached/leaked success. Their probe fails BEFORE a transport is
    # ever opened (no grant of their own), so ``remote_transport`` stays at
    # the owner's one successful connect -- proving the second call was a
    # genuine live re-evaluation for a DIFFERENT principal, not a served
    # cache hit that happened to also look like an error.
    assert info_stranger["error"]
    assert len(remote_transport) == 1


@pytest.mark.asyncio
async def test_probe_catalog_returns_plain_oauth_snapshot_and_private_binding(
    remote_transport, tmp_path
):
    actor = verified_actor()
    broker = _broker_with_stored_token(actor=actor)
    mod._REMOTE_OAUTH_BROKERS["acme"] = broker
    mux = MCPMultiplexer(tmp_path / "c.json")
    mux._catalog = {
        "acme-remote": {"url": RESOURCE_URL, "oauth_provider": _provider_cfg()}
    }

    with use_actor(actor):
        catalog = await mux.probe_catalog()

    info = catalog["acme-remote"]
    assert "_discovery_binding" not in info
    json.dumps(info)
    bindings = mux._take_discovery_bindings(catalog)
    assert bindings["acme-remote"].principal_id == actor.actor_id
