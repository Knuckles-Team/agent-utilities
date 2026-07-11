"""Tenant-isolation end-to-end tests (workstream AU-P0-5).

Covers the three gaps closed by this workstream:

  (a) an authorization-sensitive cache (derived properties, mandatory
      markings) must NEVER serve tenant A's value to tenant B — distinct
      :class:`GraphSession`/tenant scopes must produce distinct cache entries.
  (b) a Postgres-backed connection checkout (the KG backend's pool AND the
      unified state-store pool) must set a tenant context (``app.tenant_id``
      GUC) on every connection BEFORE the caller's SQL runs.
  (c) a secured read (mandatory-marking enforcement) is tenant-scoped once an
      actor/session tenant is set — a marking applied under one tenant must
      not restrict (or leak into) another tenant's same-id node.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_utilities.core import state_store
from agent_utilities.knowledge_graph.backends.postgresql_backend import (
    PostgreSQLBackend,
    _SingleConnPool,
)
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.ontology import permissioning as pm
from agent_utilities.knowledge_graph.ontology.derived_properties import (
    DerivedBacking,
    DerivedProperty,
    DerivedPropertyEngine,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext

pytestmark = pytest.mark.concept("AU-P0-5")


# ---------------------------------------------------------------------------
# (a) Derived-property cache does not leak across tenants
# ---------------------------------------------------------------------------


class _TenantAwareGraph:
    """A stub CYPHER backend whose answer depends on the AMBIENT session's
    tenant — mirrors a real tenant-scoped facade (``KnowledgeGraph.query``),
    so this test proves the cache key (not just the backing call) is
    tenant-aware."""

    def query(self, expression: str, params: dict) -> list[dict[str, Any]]:
        from agent_utilities.knowledge_graph.core.session import current_session

        session = current_session()
        tenant = session.tenant if session is not None else ""
        return [{"v": f"secret-for-{tenant or 'unscoped'}"}]


@pytest.fixture
def tenant_prop() -> DerivedProperty:
    return DerivedProperty(
        name="tenant_secret",
        backing=DerivedBacking.CYPHER,
        expression="MATCH (n {id: $id}) RETURN n.secret AS v",
        cacheable=True,
    )


def test_derived_property_cache_isolates_tenants(tenant_prop):
    engine = DerivedPropertyEngine()
    graph = _TenantAwareGraph()
    obj = {"id": "shared-object-id"}  # SAME object id for both tenants

    session_a = GraphSession(tenant="acme")
    session_b = GraphSession(tenant="globex")

    with use_session(session_a):
        res_a = engine.compute(obj, tenant_prop, graph, session=session_a)
    with use_session(session_b):
        res_b = engine.compute(obj, tenant_prop, graph, session=session_b)

    assert res_a.value == "secret-for-acme"
    # The regression this guards: pre-P0-5 the cache key was
    # (name, sig, object_id) only, so this second call — same property, same
    # object id — would hit tenant A's cached entry and return "acme"'s value
    # to globex.
    assert res_b.value == "secret-for-globex"
    assert res_a.value != res_b.value

    # A repeat call for tenant A is a genuine cache hit of ITS OWN value.
    with use_session(session_a):
        res_a2 = engine.compute(obj, tenant_prop, graph, session=session_a)
    assert res_a2.cached is True
    assert res_a2.value == "secret-for-acme"


def test_derived_property_cache_unscoped_caller_unaffected(tenant_prop):
    """Additive/back-compat: an unscoped caller (no session) behaves exactly
    like before this fix — one shared cache bucket, still a cache hit."""
    engine = DerivedPropertyEngine()
    graph = _TenantAwareGraph()
    obj = {"id": "unscoped-object"}

    res1 = engine.compute(obj, tenant_prop, graph)
    res2 = engine.compute(obj, tenant_prop, graph)
    assert res1.cached is False
    assert res2.cached is True
    assert res1.value == res2.value == "secret-for-unscoped"


# ---------------------------------------------------------------------------
# (a bonus) Mandatory-marking registry does not leak across tenants
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_markings():
    pm.clear_markings()
    yield
    pm.clear_markings()


def test_marking_cache_isolates_tenants_with_same_node_id():
    # Two tenants both minted a node called "doc:shared" (id collision across
    # tenants is realistic — connectors/ids are not guaranteed globally
    # unique). A marking applied under one tenant must not appear under another.
    pm.apply_marking("doc:shared", pm.Marking("secret"), tenant="acme")

    assert pm.markings_for("doc:shared", tenant="acme") == {"secret"}
    assert pm.markings_for("doc:shared", tenant="globex") == set()
    assert pm.markings_for("doc:shared", tenant="") == set()


def test_marking_cache_unscoped_caller_unaffected():
    """No explicit tenant + no ambient actor/session -> the same single ""
    bucket as before markings were tenant-keyed."""
    pm.apply_marking("doc:legacy", pm.Marking("pii"))
    assert pm.markings_for("doc:legacy") == {"pii"}


# ---------------------------------------------------------------------------
# (c) Secured read (mandatory marking enforcement) is tenant-scoped
# ---------------------------------------------------------------------------


def _actor(tenant_id: str, actor_id: str) -> ActorContext:
    return ActorContext(actor_id=actor_id, actor_type=ActorType.AI_AGENT, tenant_id=tenant_id)


def test_restricted_view_marking_is_tenant_scoped():
    pm.apply_marking("doc:shared", pm.Marking("secret"), tenant="acme")

    acme_actor = _actor("acme", "alice")  # no marking:secret role -> denied
    globex_actor = _actor("globex", "bob")  # same node id, different tenant -> unmarked there

    objs = [{"id": "doc:shared", "v": 1}]
    assert pm.restricted_view(objs, acme_actor) == []
    assert pm.restricted_view(objs, globex_actor) == objs


def test_restricted_view_marking_clears_for_holder_of_the_role():
    pm.apply_marking("doc:shared", pm.Marking("secret"), tenant="acme")
    cleared_actor = ActorContext(
        actor_id="carol",
        actor_type=ActorType.AI_AGENT,
        tenant_id="acme",
        roles=("marking:secret",),
    )
    objs = [{"id": "doc:shared", "v": 1}]
    assert pm.restricted_view(objs, cleared_actor) == objs


# ---------------------------------------------------------------------------
# (b) Every pooled Postgres connection gets a tenant context before use
# ---------------------------------------------------------------------------


@pytest.fixture
def pg_backend():
    """A PostgreSQLBackend wired to a MagicMock connection (no live DB)."""
    backend = PostgreSQLBackend.__new__(PostgreSQLBackend)
    backend._dsn = "postgresql://test:test@localhost:5432/test"
    backend._graph_name = "test_graph"
    backend._pool_min = 1
    backend._pool_max = 2
    backend._pggraph_schema = "public"
    backend._known_tables = set()
    backend._node_tables = set()
    backend._pggraph_available = False
    backend._pgvector_available = False
    backend._paradedb_available = False

    mock_cur = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    backend._pool = _SingleConnPool(mock_conn)
    return backend, mock_cur


def test_pg_backend_conn_checkout_sets_tenant_guc(pg_backend):
    backend, mock_cur = pg_backend
    session = GraphSession(tenant="acme")
    with use_session(session):
        with backend._conn():
            pass
    executed = [c.args[0] for c in mock_cur.execute.call_args_list]
    assert any(
        "app.tenant_id" in sql and "acme" in sql for sql in executed
    ), executed


def test_pg_backend_conn_checkout_unscoped_sets_empty_guc(pg_backend):
    """No ambient session/actor -> GUC still runs, scoped to "" (unrestricted,
    the historical/system path) — additive/back-compat, not a silent no-op."""
    backend, mock_cur = pg_backend
    with backend._conn():
        pass
    executed = [c.args[0] for c in mock_cur.execute.call_args_list]
    assert any("app.tenant_id" in sql for sql in executed), executed


def test_pg_backend_conn_checkout_raises_when_real_tenant_set_fails(pg_backend):
    """Fail-CLOSED: if the ``SET app.tenant_id`` for a NON-EMPTY tenant fails,
    the checkout must ABORT (raise out of _conn()) — never yield a pooled
    connection still carrying the previous checkout's tenant (a cross-tenant
    leak). The caller body must never run."""
    backend, mock_cur = pg_backend
    mock_cur.execute.side_effect = RuntimeError("SET app.tenant_id failed")
    session = GraphSession(tenant="acme")
    ran = False
    with use_session(session):
        with pytest.raises(RuntimeError):
            with backend._conn():
                ran = True  # pragma: no cover - must not be reached
    assert ran is False


def test_pg_backend_conn_checkout_unscoped_tolerates_set_failure(pg_backend):
    """Fail-OPEN only for the empty/system baseline: a failed SET with NO
    tenant to leak into stays best-effort — the checkout still yields."""
    backend, mock_cur = pg_backend
    mock_cur.execute.side_effect = RuntimeError("SET app.tenant_id failed")
    ran = False
    with backend._conn():  # no ambient tenant -> "" -> best-effort
        ran = True
    assert ran is True


# -- state-store (sessions/turns/usage/...) shared pool --------------------


class _FakeCursor:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    def execute(self, sql: str, params: Any = ()) -> _FakeCursor:
        self._conn.calls.append((" ".join(sql.split()), tuple(params or ())))
        self.description = None
        return self

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    @property
    def rowcount(self) -> int:
        return 0


class _FakeConn:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def execute(self, sql: str, params: Any = ()) -> _FakeCursor:
        return self.cursor().execute(sql, params)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


class _FakePool:
    def __init__(self):
        self.conn = _FakeConn()
        self.put_back: list[Any] = []

    @contextmanager
    def connection(self):
        yield self.conn

    def getconn(self):
        return self.conn

    def putconn(self, conn):
        self.put_back.append(conn)


@pytest.fixture(autouse=True)
def _clean_state_store():
    state_store._migrated_stores.clear()
    yield
    state_store._migrated_stores.clear()


@pytest.fixture
def pg_state_pool(monkeypatch):
    pool = _FakePool()
    monkeypatch.setattr(state_store, "postgres_state_enabled", lambda: True)
    monkeypatch.setattr(state_store, "state_pool", lambda: pool)
    return pool


def test_state_store_connection_checkout_sets_tenant_guc(pg_state_pool):
    session = GraphSession(tenant="acme")
    with use_session(session):
        conn = state_store.open_state_connection("test", lambda: "/nonexistent")
    calls = [sql for sql, _ in pg_state_pool.conn.calls]
    assert any("app.tenant_id" in sql and "acme" in sql for sql in calls), calls
    conn.close()


def test_state_store_connection_checkout_unscoped_is_empty_guc(pg_state_pool):
    conn = state_store.open_state_connection("test", lambda: "/nonexistent")
    calls = [sql for sql, _ in pg_state_pool.conn.calls]
    assert any("app.tenant_id" in sql for sql in calls), calls
    conn.close()


def test_state_store_checkout_raises_when_real_tenant_set_fails(pg_state_pool):
    """Fail-CLOSED: a NON-EMPTY tenant whose GUC can't be set aborts the
    checkout (raises out of open_state_connection) AND returns the borrowed
    connection to the pool — never hands back a stale-tenant connection."""

    def _boom(sql, params=()):
        raise RuntimeError("SET app.tenant_id failed")

    pg_state_pool.conn.execute = _boom  # type: ignore[method-assign]
    session = GraphSession(tenant="acme")
    with use_session(session):
        with pytest.raises(RuntimeError):
            state_store.open_state_connection("test", lambda: "/nonexistent")
    # borrowed connection was returned to the pool on the abort path
    assert pg_state_pool.put_back == [pg_state_pool.conn]


def test_state_store_checkout_unscoped_tolerates_set_failure(pg_state_pool):
    """Fail-OPEN only for the empty/system baseline: no tenant to leak into,
    so a failed SET stays best-effort and the checkout still succeeds."""

    def _boom(sql, params=()):
        raise RuntimeError("SET app.tenant_id failed")

    pg_state_pool.conn.execute = _boom  # type: ignore[method-assign]
    conn = state_store.open_state_connection("test", lambda: "/nonexistent")
    assert conn.dialect == "postgres"
    conn.close()
