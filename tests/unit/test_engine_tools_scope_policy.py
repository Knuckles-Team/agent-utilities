"""Tests for AU-P0-6: per-action scope/policy gating + bounded engine client pool
on the low-level ``engine_<domain>`` MCP tools (``mcp/tools/engine_tools.py``).

Covers:
  (a) an ADMIN action (tenants/resharding/consensus/rbac/admin family) invoked
      by a non-admin actor is DENIED (raises ``PermissionError``, fail-closed).
  (b) a normal read/write action by a normal (non-admin) actor is ALLOWED.
  (c) an admin actor (role OR an explicit GraphSession ``kg:admin`` scope) IS
      allowed to invoke an admin action.
  (d) the low-level engine client pool is bounded: exceeding capacity evicts
      the least-recently-used connection instead of growing without limit.
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
    actor_id="agent:marketing",
    actor_type=ActorType.AI_AGENT,
    roles=("marketing",),
    tenant_id="acme",
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
    with use_actor(NON_ADMIN_ACTOR):
        with pytest.raises(PermissionError, match="ADMIN-only"):
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
    with use_actor(NON_ADMIN_ACTOR):
        out = json.loads(
            asyncio.run(tool(action=action, params_json=json.dumps(params), graph=""))
        )
    assert out.get("ok") is True, out
    assert calls == [(domain, action, params)]


# ── (c) admin actor / admin GraphSession scope IS allowed ────────────────────
def test_admin_action_allowed_for_admin_role_actor(monkeypatch):
    kg_server.ensure_tools_registered()
    client, calls = _fake_client_factory()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    admin_actor = ActorContext(
        actor_id="agent:ops",
        actor_type=ActorType.AI_AGENT,
        roles=("admin",),
        tenant_id="",
    )
    tool = kg_server.REGISTERED_TOOLS["engine_tenants"]
    with use_actor(admin_actor):
        out = json.loads(
            asyncio.run(tool(action="list", params_json="{}", graph=""))
        )
    assert out.get("ok") is True, out
    assert calls == [("tenants", "list", {})]


def test_admin_action_allowed_via_graph_session_scope(monkeypatch):
    """A non-admin-role actor with an explicit GraphSession ``kg:admin`` scope
    is also let through (GraphSession.scopes is the other authority AU-P0-6
    checks — see ``_enforce_admin_scope``)."""
    kg_server.ensure_tools_registered()
    client, calls = _fake_client_factory()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    tool = kg_server.REGISTERED_TOOLS["engine_resharding"]
    session = GraphSession(actor=NON_ADMIN_ACTOR, scopes=frozenset({"kg:admin"}))
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


# ── (d) bounded client pool ───────────────────────────────────────────────────
def test_client_pool_is_bounded_lru_and_evicts(monkeypatch):
    """Exceeding pool capacity evicts the LRU connection (closing it) instead
    of letting the resident connection count grow without bound (audited
    gap #2)."""
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
    assert closed == ["g1"]
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
