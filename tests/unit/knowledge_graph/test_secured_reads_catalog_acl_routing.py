"""EG fleet-catalog ACL routing (CONCEPT:AU-KG.ingest.fleet-catalog-acl-projection).

EH-345 (2026-09-22) replaced the deleted ``fleet_catalog_tables.catalog_acl_rows``
SQL fast path with ``secured_reads._catalog_acl_hits`` reading
``FleetCatalogClient.lookup`` directly. This file (rewritten from its
predecessor, which built the real rows through ``fleet_catalog_tables``'s SQL
writer -- deleted along with that module) exercises the fast path with fake
``fleet_catalog`` clients implementing the same synchronous method surface
``SyncEpistemicGraphClient.fleet_catalog`` exposes, and confirms every miss
mode (no client configured, lookup raises, id absent from the answer) falls
through to the existing Cypher path unchanged -- never silently grants.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.core import secured_reads as sr
from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor


@pytest.fixture
def brain():
    reset_company_brain()
    yield get_company_brain()
    reset_company_brain()


def _actor(tenant: str = "tenant-a", actor_id: str = "sync-actor") -> ActorContext:
    return ActorContext(
        actor_id,
        ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id=tenant,
        authenticated=True,
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


def _component_ref(
    id: str, *, tenant_scope: bool, principal: str = ""
) -> SimpleNamespace:
    visibility = (
        SimpleNamespace(scope="tenant")
        if tenant_scope
        else SimpleNamespace(scope="principal", principal=principal)
    )
    acl = SimpleNamespace(
        tenant_id="tenant-a", visibility=visibility, publisher="pub-1"
    )
    return SimpleNamespace(id=id, acl=acl)


class _FakeFleetCatalog:
    def __init__(self, entries: list[SimpleNamespace] | None = None) -> None:
        self.entries = entries or []
        self.lookup_calls: list[list[str]] = []
        self.raises = False

    def lookup(self, ids: list[str], grant_digests=()):
        self.lookup_calls.append(list(ids))
        if self.raises:
            raise RuntimeError("engine fleet-catalog surface unavailable")
        wanted = set(ids)
        matched = [e for e in self.entries if sr._fleet_row_id_and_acl(e)[0] in wanted]
        return SimpleNamespace(rows=matched)


class _CatalogBackedEngine:
    """A single fake engine exposing BOTH the EG fleet-catalog client
    (``graph_compute.client.fleet_catalog``, what ``_catalog_acl_hits`` reads)
    and the Cypher surface (``backend.execute_read``, what
    ``_durable_access_rows``'s label-scoped fallback reads) -- the same
    engine object real code sees both through."""

    def __init__(self, cypher_backend, fleet_catalog: _FakeFleetCatalog | None) -> None:
        self.graph_compute = SimpleNamespace(
            client=SimpleNamespace(fleet_catalog=fleet_catalog)
        )
        self.backend = cypher_backend

    def for_graph(self, _graph_name: str) -> _CatalogBackedEngine:
        return self


# ---------------------------------------------------------------------------
# _catalog_acl_hits (the rewritten fast path) directly
# ---------------------------------------------------------------------------


def test_catalog_acl_hits_maps_tenant_scope_to_public_org():
    entry = SimpleNamespace(
        kind="tool",
        row=SimpleNamespace(
            component=_component_ref("mcp:svc/tool/t1", tenant_scope=True)
        ),
    )
    fc = _FakeFleetCatalog([entry])
    hits = sr._catalog_acl_hits(
        _CatalogBackedEngine(_RecordingCypherBackend(), fc),
        ["mcp:svc/tool/t1"],
        "tenant-a",
    )
    assert hits["mcp:svc/tool/t1"] == {
        "tenant_id": "tenant-a",
        "classification": "PUBLIC",
        "external_access": None,
        "owner_id": "pub-1",
        "shared_scope": "org",
    }


def test_catalog_acl_hits_maps_principal_scope_to_confidential_private():
    entry = SimpleNamespace(
        kind="tool",
        row=SimpleNamespace(
            component=_component_ref(
                "mcp:svc/tool/t2", tenant_scope=False, principal="alice"
            )
        ),
    )
    fc = _FakeFleetCatalog([entry])
    hits = sr._catalog_acl_hits(
        _CatalogBackedEngine(_RecordingCypherBackend(), fc),
        ["mcp:svc/tool/t2"],
        "tenant-a",
    )
    assert hits["mcp:svc/tool/t2"]["classification"] == "CONFIDENTIAL"
    assert hits["mcp:svc/tool/t2"]["owner_id"] == "alice"
    assert hits["mcp:svc/tool/t2"]["shared_scope"] == "private"


def test_catalog_acl_hits_no_client_is_empty_not_an_error():
    hits = sr._catalog_acl_hits(
        _CatalogBackedEngine(_RecordingCypherBackend(), None),
        ["mcp:svc/tool/t1"],
        "tenant-a",
    )
    assert hits == {}


def test_catalog_acl_hits_raising_lookup_is_empty_not_an_error():
    fc = _FakeFleetCatalog([])
    fc.raises = True
    hits = sr._catalog_acl_hits(
        _CatalogBackedEngine(_RecordingCypherBackend(), fc),
        ["mcp:svc/tool/t1"],
        "tenant-a",
    )
    assert hits == {}


# ---------------------------------------------------------------------------
# _durable_access_rows end to end: fast path vs Cypher fallback
# ---------------------------------------------------------------------------


def test_fleet_catalog_hit_resolves_without_a_cypher_round_trip(monkeypatch, brain):
    entry = SimpleNamespace(
        kind="tool",
        row=SimpleNamespace(
            component=_component_ref("mcp:svc/tool/t1", tenant_scope=True)
        ),
    )
    cypher = _RecordingCypherBackend(rows=[])
    engine = _CatalogBackedEngine(cypher, _FakeFleetCatalog([entry]))
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)

    with use_actor(_actor("tenant-a")):
        rows = sr._durable_access_rows(["mcp:svc/tool/t1"])

    assert rows["mcp:svc/tool/t1"]["classification"] == "PUBLIC"
    assert rows["mcp:svc/tool/t1"]["tenant_id"] == "tenant-a"
    # The dominant cost this closes: zero Cypher round trips for an id the
    # fleet catalog can fully answer for.
    assert cypher.queries == []


def test_non_fleet_id_falls_back_to_cypher_unchanged(monkeypatch, brain):
    """An id the fleet catalog has never heard of (not a fleet node at all,
    or its ConnectorPack hasn't been imported yet) must still resolve
    through the existing, already-correct Cypher path -- the fast path adds
    a lookup, it never narrows what the Cypher path could already answer."""
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
    engine = _CatalogBackedEngine(cypher, _FakeFleetCatalog([]))
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)

    with use_actor(_actor("tenant-a")):
        rows = sr._durable_access_rows(["memory:some-id"])

    assert rows["memory:some-id"]["tenant_id"] == "tenant-a"
    assert rows["memory:some-id"]["classification"] == "public"
    # It genuinely went through Cypher -- proving the fast path correctly
    # declined to answer rather than silently omitting the id.
    assert len(cypher.queries) >= 1


def test_no_fleet_catalog_client_falls_back_to_cypher(monkeypatch, brain):
    """A deployment where EG's fleet-catalog surface isn't wired yet (client
    is None -- the pre-eg-fleet-catalog-landing state) must degrade to
    Cypher exactly like a miss, never crash or silently deny."""
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
    engine = _CatalogBackedEngine(cypher, None)
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)

    with use_actor(_actor("tenant-a")):
        rows = sr._durable_access_rows(["tool_srv_t1"])

    assert rows["tool_srv_t1"]["owner_id"] == "principal:owner"
    assert len(cypher.queries) >= 1


def test_failed_fleet_catalog_lookup_falls_back_to_cypher_and_never_grants_by_itself(
    monkeypatch, brain
):
    """A raising fleet-catalog surface (a transient engine error) must
    degrade to the Cypher path, exactly like "no catalog opinion" -- never
    surface as a grant, and never crash the read."""
    fc = _FakeFleetCatalog([])
    fc.raises = True
    cypher = _RecordingCypherBackend(rows=[])  # Cypher also finds nothing.
    engine = _CatalogBackedEngine(cypher, fc)
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)

    with use_actor(_actor("tenant-a")):
        rows = sr._durable_access_rows(["tool_srv_t1"])
    assert rows == {}


def test_absent_or_failed_lookup_never_grants_permission_end_to_end(monkeypatch, brain):
    """End-to-end through ``permit()``: when neither the fleet catalog nor
    Cypher can produce ACL material for an id, the node stays unregistered
    and ``permit()`` denies -- the fail-closed contract this whole read path
    exists to guarantee is unaffected by which fast path feeds it."""
    cypher = _RecordingCypherBackend(rows=[])  # nothing anywhere
    engine = _CatalogBackedEngine(cypher, _FakeFleetCatalog([]))
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)

    with use_actor(_actor("tenant-a")):
        permitted = sr.permit(["tool_srv_t1"])
    assert "tool_srv_t1" not in permitted
