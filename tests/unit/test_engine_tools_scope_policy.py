"""Tests for AU-P0-6: per-action scope/policy gating + bounded graph-view pool
on the low-level ``engine_<domain>`` MCP tools (``mcp/tools/engine_tools.py``).

Covers:
  (a) an ADMIN action (tenants/resharding/consensus/rbac/admin family) invoked
      by a non-admin actor is DENIED (raises ``PermissionError``, fail-closed).
  (b) a normal read/write action by a normal (non-admin) actor is ALLOWED.
  (c) only an explicit GraphSession ``kg:admin`` scope allows an admin action;
      a role alone never bypasses session authority.
  (d) low-level graph views are bounded and eviction never closes the process
      transport.
  (e) fail-closed classification for a hypothetical un-classified domain.

No live engine is required — the wire client is monkeypatched exactly like
``tests/unit/test_engine_api_coverage.py``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import engine_tools
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor


def _fake_client_factory():
    """A fake ``SyncEpistemicGraphClient`` with recording sub-clients for every
    domain touched by these tests."""
    calls: list[tuple[str, str, dict]] = []

    def _sub(domain: str):
        def _make(name):
            def _call(**kwargs):
                calls.append((domain, name, kwargs))
                return {"ok": True, "domain": domain, "method": name}

            return _call

        class _Sub:
            def __getattr__(self, name):
                return _make(name)

        return _Sub()

    class _Client:
        def __getattr__(self, name):
            return _sub(name)

    return _Client(), calls


@pytest.fixture(autouse=True)
def _fresh_client_pool(monkeypatch):
    """Isolate the module-level client-pool singleton across tests."""
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", None)
    yield
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", None)


NON_ADMIN_ACTOR = ActorContext(
    actor_id="principal:marketing",
    actor_type=ActorType.AI_AGENT,
    roles=("marketing",),
    tenant_id="acme",
    authenticated=True,
)


def _session(actor: ActorContext, *scopes: str) -> GraphSession:
    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset(scopes),
        graph="tenant-acme-graph",
        policy_version="policy-v1",
        audience="agent-services",
    )


# ── (a) admin action denied for a non-admin actor ────────────────────────────
@pytest.mark.parametrize(
    ("domain", "action", "params"),
    [
        ("tenants", "list", {}),
        ("resharding", "catalog_list", {}),
        (
            "consensus",
            "register_identity",
            {"agent_id": "a", "role": "Worker", "teams": [], "signature": "s"},
        ),
        ("rbac", "list", {}),
        ("admin", "backup", {}),
    ],
)
def test_admin_action_denied_for_non_admin_actor(monkeypatch, domain, action, params):
    kg_server.ensure_tools_registered()
    client, _calls = _fake_client_factory()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    tool = kg_server.REGISTERED_TOOLS[f"engine_{domain}"]
    session = _session(NON_ADMIN_ACTOR, "kg:read", "kg:write")
    with use_actor(NON_ADMIN_ACTOR), use_session(session):
        with pytest.raises(PermissionError, match="kg:admin"):
            asyncio.run(
                tool(action=action, params_json=json.dumps(params), graph="")
            )


def test_unknown_domain_defaults_to_admin_fail_closed():
    """A hypothetical future domain this map hasn't classified yet is ADMIN by
    default — never silently open (AU-P0-6 guardrail)."""
    assert engine_tools._is_admin_domain("some_future_namespace") is True
    assert engine_tools.action_policy("some_future_namespace", "list")["admin"] is True


# ── (b) normal read/write allowed for a non-admin actor ──────────────────────
@pytest.mark.parametrize(
    ("domain", "action", "params"),
    [
        ("nodes", "has", {"node_id": "n1"}),
        ("edges", "has", {"source_id": "a", "target_id": "b"}),
        ("query", "sql", {"query": "SELECT 1"}),
        ("broker", "publish", {"exchange": "ex", "routing_key": "rk", "payload": "hi"}),
        ("graphlearn", "predict", {}),
    ],
)
def test_normal_action_allowed_for_non_admin_actor(monkeypatch, domain, action, params):
    kg_server.ensure_tools_registered()
    client, calls = _fake_client_factory()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    tool = kg_server.REGISTERED_TOOLS[f"engine_{domain}"]
    session = _session(NON_ADMIN_ACTOR, "kg:read", "kg:write")
    with use_actor(NON_ADMIN_ACTOR), use_session(session):
        out = json.loads(
            asyncio.run(tool(action=action, params_json=json.dumps(params), graph=""))
        )
    assert out.get("ok") is True, out
    assert calls == [(domain, action, params)]


def test_served_profile_enforces_read_write_scope_for_normal_domains(monkeypatch):
    from agent_utilities.knowledge_graph.core.session import ScopeError

    client, calls = _fake_client_factory()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)
    actor = ActorContext(
        actor_id="principal:opaque",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("kg:read",),
        tenant_id="tenant-a",
        authenticated=True,
    )
    reader = GraphSession(
        actor=actor,
        tenant="tenant-a",
        graph="tenant-a-graph",
        scopes=frozenset({"kg:read"}),
    )
    with use_actor(actor), use_session(reader):
        out = json.loads(engine_tools._dispatch("nodes", {"has"}, "has", "{}", ""))
        assert out["ok"] is True
        with pytest.raises(ScopeError):
            engine_tools._dispatch("nodes", {"add"}, "add", "{}", "")

    writer = _session(actor, "kg:read", "kg:write")
    with use_actor(actor), use_session(writer):
        out = json.loads(engine_tools._dispatch("nodes", {"add"}, "add", "{}", ""))
        assert out["ok"] is True
    assert [call[:2] for call in calls] == [("nodes", "has"), ("nodes", "add")]


# ── (c) admin actor / admin GraphSession scope IS allowed ────────────────────
def test_admin_role_without_admin_scope_is_denied(monkeypatch):
    kg_server.ensure_tools_registered()
    client, calls = _fake_client_factory()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    admin_actor = ActorContext(
        actor_id="principal:ops",
        actor_type=ActorType.AI_AGENT,
        roles=("admin",),
        tenant_id="tenant-ops",
        authenticated=True,
    )
    tool = kg_server.REGISTERED_TOOLS["engine_tenants"]
    session = _session(admin_actor, "kg:read", "kg:write")
    with use_actor(admin_actor), use_session(session), pytest.raises(
        PermissionError, match="kg:admin"
    ):
        asyncio.run(tool(action="list", params_json="{}", graph=""))
    assert calls == []


def test_admin_action_allowed_via_graph_session_scope(monkeypatch):
    """A non-admin-role actor with an explicit GraphSession ``kg:admin`` scope
    is also let through because GraphSession scopes are the sole authority."""
    kg_server.ensure_tools_registered()
    client, calls = _fake_client_factory()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    tool = kg_server.REGISTERED_TOOLS["engine_resharding"]
    session = _session(NON_ADMIN_ACTOR, "kg:admin")
    with use_actor(NON_ADMIN_ACTOR), use_session(session):
        out = json.loads(
            asyncio.run(
                tool(action="catalog_list", params_json="{}", graph="")
            )
        )
    assert out.get("ok") is True, out
    assert calls == [("resharding", "catalog_list", {})]


# ── new namespaces registered (audited gap #3) ───────────────────────────────
def test_new_namespaces_registered_and_admin_flagged():
    kg_server.ensure_tools_registered()
    for domain in ("broker", "rbac", "admin", "graphlearn"):
        tool_name = f"engine_{domain}"
        assert domain in engine_tools.ENGINE_DOMAINS, f"{domain} not discovered"
        assert tool_name in kg_server.REGISTERED_TOOLS
        assert kg_server.ACTION_TOOL_ROUTES.get(tool_name) == f"/engine/{domain}"
    assert engine_tools._is_admin_domain("rbac") is True
    assert engine_tools._is_admin_domain("admin") is True
    assert engine_tools._is_admin_domain("broker") is False
    assert engine_tools._is_admin_domain("graphlearn") is False


# ── (d) bounded graph-view pool over one process client ──────────────────────
def test_client_pool_is_bounded_lru_and_evicts(monkeypatch):
    """Capacity evicts an LRU graph view without closing shared transport."""
    from agent_utilities.knowledge_graph.core.tenant_engine_pool import (
        TenantEnginePool,
    )

    created: list[str] = []
    closed: list[str] = []

    class _FakeWireClient:
        def __init__(self, key: str) -> None:
            self.key = key

        def close(self) -> None:
            closed.append(self.key)

    def factory(key: str):
        created.append(key)
        return _FakeWireClient(key)

    pool = TenantEnginePool(capacity=2, factory=factory, on_evict=engine_tools._client_evict)
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", pool)

    engine_tools._client_for("g1")
    engine_tools._client_for("g2")
    engine_tools._client_for("g3")  # over capacity → evicts g1 (LRU)

    assert created == ["g1", "g2", "g3"]
    assert closed == []
    assert set(pool.warm_tenants()) == {"g2", "g3"}

    # Touching many more distinct graphs never grows the warm set past capacity.
    for i in range(4, 50):
        engine_tools._client_for(f"g{i}")
    assert len(pool.warm_tenants()) == 2
    assert pool.stats()["evictions"] == 47  # 49 acquires - 2 resident


def test_client_pool_warm_hit_reuses_same_client(monkeypatch):
    from agent_utilities.knowledge_graph.core.tenant_engine_pool import (
        TenantEnginePool,
    )

    created: list[str] = []

    def factory(key: str):
        created.append(key)
        return object()

    pool = TenantEnginePool(capacity=4, factory=factory, on_evict=lambda k, c: None)
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", pool)

    a = engine_tools._client_for("g1")
    b = engine_tools._client_for("g1")
    assert a is b
    assert created == ["g1"]


def test_client_pool_empty_graph_uses_stable_sentinel_key(monkeypatch):
    """``graph=""`` (deployment default) must hash to ONE stable pool key, not
    be treated as tenant-routed (which would resolve a different graph name
    entirely — see ``_client_factory``'s sentinel translation)."""
    from agent_utilities.knowledge_graph.core.tenant_engine_pool import (
        TenantEnginePool,
    )

    created: list[str] = []

    def factory(key: str):
        created.append(key)
        return object()

    pool = TenantEnginePool(capacity=4, factory=factory, on_evict=lambda k, c: None)
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", pool)

    a = engine_tools._client_for("")
    b = engine_tools._client_for("")
    assert a is b
    assert created == [engine_tools._DEFAULT_GRAPH_POOL_KEY]


def test_method_graph_parameter_inherits_verified_session_graph(monkeypatch):
    from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
    from agent_utilities.security.brain_context import ActorContext

    session = GraphSession(
        actor=ActorContext(
            actor_id="principal:opaque",
            tenant_id="tenant-a",
            authenticated=True,
        ),
        tenant="tenant-a",
        graph="tenant-a-graph",
        scopes=frozenset({"kg:read"}),
    )
    with use_session(session):
        assert engine_tools._resolve_graph_name("") == "tenant-a-graph"
        with pytest.raises(PermissionError, match="retarget"):
            engine_tools._resolve_graph_name("tenant-b-graph")
