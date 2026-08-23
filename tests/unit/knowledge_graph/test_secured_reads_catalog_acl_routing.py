"""SQL-authoritative ACL routing (CONCEPT:AU-KG.ingest.fleet-catalog-acl-projection).

Closes the deferred item on ``secured_reads._durable_access_rows``: the
production incident measured two *unlabeled* Cypher full scans per fleet
tool; the label-scoped candidates fixed "unlabeled", this fixes "at all" for
a fleet ``Tool``/``MCPServer``/``Skill`` id whose relational catalog row
(``fleet_catalog_tables``) already carries a durable ACL stamp written from
the SAME policy the KG node write uses.

Uses the REAL ``fleet_catalog_tables`` SQL emulator
(``test_fleet_catalog_tables._FakeGraphCompute``) so the catalog rows this
suite queries are the product of the REAL writer path
(``write_fleet_catalog``), not a hand-authored double standing in for it --
the seam under test (``catalog_acl_rows``'s SQL query shapes against a real
migrated schema) is exercised for real, only the Cypher backend below it is
a test double.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core import fleet_catalog_tables as fct
from agent_utilities.knowledge_graph.core import secured_reads as sr
from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor
from tests.unit.knowledge_graph.test_fleet_catalog_tables import _FakeGraphCompute


@pytest.fixture
def brain():
    reset_company_brain()
    yield get_company_brain()
    reset_company_brain()


@pytest.fixture(autouse=True)
def _reset_ddl_cache():
    fct._ensured_stores.clear()
    yield
    fct._ensured_stores.clear()


def _actor(tenant: str = "tenant-a", actor_id: str = "sync-actor") -> ActorContext:
    return ActorContext(
        actor_id,
        ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id=tenant,
        authenticated=True,
    )


def _session(tenant: str = "tenant-a", actor_id: str = "sync-actor") -> GraphSession:
    return GraphSession(
        actor=_actor(tenant, actor_id),
        tenant=tenant,
        scopes=frozenset({"kg:read", "kg:write"}),
        graph="g",
        policy_version="v1",
        audience="test",
    )


class _RecordingCypherBackend:
    """A minimal ``execute_read``-only ACL backend, seeded with zero or more
    rows -- stands in for the durable KG store ``_durable_access_rows``'s
    Cypher path reads from. Records every query so a test can assert it was
    (or was not) reached at all."""

    def __init__(self, rows: list[dict] | None = None) -> None:
        self._rows = rows or []
        self.queries: list[tuple[str, dict]] = []

    def execute_read(self, query: str, params: dict, **_kw) -> list[dict]:
        self.queries.append((query, params))
        wanted = set(params.get("ids", []))
        return [row for row in self._rows if row.get("id") in wanted]


class _RejectingCypherBackend:
    """Every Cypher read fails -- models an infrastructure failure below the
    SQL fast path, never a route to "grant"."""

    def execute_read(self, _query, _params, **_kw):
        raise RuntimeError("durable ACL store rejected the hydration query")


class _CatalogBackedEngine:
    """A single fake engine exposing BOTH the SQL surface
    (``graph_compute.sql_exec``, what ``fleet_catalog_tables``/
    ``catalog_acl_rows`` write to and query) and the Cypher surface
    (``backend.execute_read``, what ``_durable_access_rows``'s label-scoped
    fallback reads) -- the same engine object real code sees both through.
    """

    def __init__(self, cypher_backend) -> None:
        self.graph_compute = _FakeGraphCompute()
        self.backend = cypher_backend

    def for_graph(self, _graph_name: str) -> "_CatalogBackedEngine":
        return self


def _server_catalog() -> dict:
    return {
        "srv": {
            "error": None,
            "tools": [
                {
                    "name": "t1",
                    "description": "d",
                    "inputSchema": {"properties": {"x": {"type": "string"}}},
                }
            ],
            "skills": [],
            "prompts": [],
        }
    }


def _seed_catalog_row(engine: _CatalogBackedEngine, tenant: str) -> None:
    """Write one real ``mcp_tools`` row through the real writer path, ACL
    columns included, under a verified actor/session -- this is the SAME
    write ``fleet-tool-schema-sync`` performs in production."""
    with use_actor(_actor(tenant)), use_session(_session(tenant)):
        result = fct.write_fleet_catalog(
            engine,
            _server_catalog(),
            discovery_bindings={
                "srv": fct.TenantLocalDiscoveryBinding(tenant_id=tenant)
            },
        )
    assert result["status"] == "ok"
    assert result["tools_written"] == 1


def test_sql_authoritative_id_resolves_via_sql_without_a_cypher_round_trip(
    monkeypatch, brain
):
    cypher = _RecordingCypherBackend(rows=[])
    engine = _CatalogBackedEngine(cypher)
    _seed_catalog_row(engine, "tenant-a")

    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)
    with use_actor(_actor("tenant-a")):
        rows = sr._durable_access_rows(["tool_srv_t1"])

    assert rows["tool_srv_t1"]["classification"] == "confidential"
    assert rows["tool_srv_t1"]["owner_id"] == "sync-actor"
    assert rows["tool_srv_t1"]["shared_scope"] == "private"
    assert rows["tool_srv_t1"]["external_access"] is None
    assert rows["tool_srv_t1"]["tenant_id"] == "tenant-a"
    # The dominant cost this closes: zero Cypher round trips for an id the
    # catalog can fully answer for.
    assert cypher.queries == []


def test_non_fleet_id_falls_back_to_cypher_unchanged(monkeypatch, brain):
    """An id the catalog has never heard of (not a fleet node at all) must
    still resolve through the existing, already-correct Cypher path -- the
    SQL fast path adds a lookup, it never narrows what the Cypher path could
    already answer."""
    cypher = _RecordingCypherBackend(
        rows=[
            {
                "id": "memory:some-id",
                "tenant_id": "tenant-a",
                "classification": "public",
                "external_access": None,
                "owner_id": None,
                "shared_scope": None,
            }
        ]
    )
    engine = _CatalogBackedEngine(cypher)
    # The catalog exists (migrated) but has never heard of this id.
    with use_actor(_actor("tenant-a")), use_session(_session("tenant-a")):
        fct.ensure_fleet_catalog_tables(engine)

    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)
    with use_actor(_actor("tenant-a")):
        rows = sr._durable_access_rows(["memory:some-id"])

    assert rows["memory:some-id"]["tenant_id"] == "tenant-a"
    assert rows["memory:some-id"]["classification"] == "public"
    # It genuinely went through Cypher -- proving the fast path correctly
    # declined to answer rather than silently omitting the id.
    assert len(cypher.queries) >= 1


def test_catalog_row_without_a_stamped_acl_also_falls_back_to_cypher(
    monkeypatch, brain
):
    """A catalog row that exists (the id IS in ``mcp_tools``) but was written
    before an ACL was ever stamped for it (``acl_classification`` NULL --
    the exact shape a legacy/pre-migration row backfill leaves) must be
    treated identically to "not in the catalog at all": SQL has no opinion,
    Cypher decides. A present-but-blank row must never be read as
    "unrestricted"."""
    cypher = _RecordingCypherBackend(
        rows=[
            {
                "id": "tool_srv_t1",
                "tenant_id": "tenant-a",
                "classification": "confidential",
                "external_access": None,
                "owner_id": "principal:owner",
                "shared_scope": "private",
            }
        ]
    )
    engine = _CatalogBackedEngine(cypher)
    with use_actor(_actor("tenant-a")), use_session(_session("tenant-a")):
        fct.ensure_fleet_catalog_tables(engine)
    # Seed the row directly with a blank ACL stamp, bypassing the writer --
    # models a row from before this migration's writer change landed.
    gc = engine.graph_compute
    gc.tables[fct.TABLE_MCP_TOOLS]["tool_srv_t1"] = {
        "id": "tool_srv_t1",
        "tenant_id": "tenant-a",
        "server_id": "mcp_server_srv",
        "server_name": "srv",
        "name": "t1",
        "description": "d",
        "input_schema": "{}",
        "schema_digest": fct._schema_digest({}),
        "tool_mode": "verbose",
        "enabled": True,
        "discovery_authority_kind": fct.DISCOVERY_AUTHORITY_TENANT_LOCAL,
        "discovery_principal": "",
        "discovery_grant_digest": "",
        "revision": 1,
        "idempotency_key": "seed",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "kg_node_id": "tool_srv_t1",
        "acl_classification": None,
        "acl_owner_id": None,
        "acl_shared_scope": None,
    }

    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)
    with use_actor(_actor("tenant-a")):
        rows = sr._durable_access_rows(["tool_srv_t1"])

    # Resolved via Cypher, not silently dropped or "unrestricted".
    assert rows["tool_srv_t1"]["owner_id"] == "principal:owner"
    assert len(cypher.queries) >= 1


def test_failed_sql_lookup_falls_back_to_cypher_and_never_grants_by_itself(
    monkeypatch, brain
):
    """A raising SQL surface (a transient engine error) must degrade to the
    Cypher path, exactly like "no catalog opinion" -- never surface as a
    grant, and never crash the read."""

    class _RaisingGraphCompute:
        def sql_exec(self, _statement):
            raise RuntimeError("engine SQL surface unavailable")

    class _Engine:
        def __init__(self, cypher) -> None:
            self.graph_compute = _RaisingGraphCompute()
            self.backend = cypher

        def for_graph(self, _graph_name: str) -> "_Engine":
            return self

    cypher = _RecordingCypherBackend(rows=[])  # Cypher also finds nothing.
    engine = _Engine(cypher)
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)

    with use_actor(_actor("tenant-a")):
        rows = sr._durable_access_rows(["tool_srv_t1"])
    assert rows == {}


def test_absent_or_failed_lookup_never_grants_permission_end_to_end(
    monkeypatch, brain
):
    """End-to-end through ``permit()``: when neither SQL nor Cypher can
    produce ACL material for an id, the node stays unregistered and
    ``permit()`` denies -- the fail-closed contract this whole read path
    exists to guarantee is unaffected by the new fast path being added."""
    cypher = _RecordingCypherBackend(rows=[])  # nothing anywhere
    engine = _CatalogBackedEngine(cypher)
    with use_actor(_actor("tenant-a")), use_session(_session("tenant-a")):
        fct.ensure_fleet_catalog_tables(engine)  # catalog exists, just empty

    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)
    with use_actor(_actor("tenant-a")):
        assert sr.permit(["nonexistent-id"]) == []
