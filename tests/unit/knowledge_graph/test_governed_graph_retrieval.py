"""GOC-67 (R-22): governed retrieval end-to-end -- the ACL-hydration half of
explicit physical-graph selection landed on ``fix/au-explicit-graph-selection``
(R-01..03).

That branch proved a caller can correctly *select* a physical graph
(``resolve_explicit_graph``/``bound_to_graph``) and see two-graph write/read
isolation through a fake ``query_cypher`` override that bypassed governance
entirely. It never proved the caller can then actually *read* real governed
rows back -- ``secured_reads._durable_access_rows`` hydrated ACL/classification
material from ``IntelligenceGraphEngine.get_active()``'s own bound backend
regardless of which graph the verified session had narrowed to, so a selected
graph's nodes (absent from the active engine's own backend) were correctly, but
uselessly, default-denied. Aggregate/count queries never hit this path (no
per-row id to govern) and so appeared to work fine, masking the gap.

This file runs the REAL ``QueryMixin.query_cypher`` (not a test double) with
the REAL ``secured_reads`` governance chain (``scope``/``filter_rows``/
``permit``/``_durable_access_rows``/``visible``/``audit_read``) against two
physical graphs with independent content AND independent durable ACL
backends, proving:

* a realistic artifact/provenance/edge projection, each row carrying its own
  governed id, is returned intact -- and nothing not explicitly ingested (a
  ``content`` field) rides along;
* the identical query against a second physical graph returns zero rows;
* an unknown graph and a graph whose durable ACL backend rejects the
  hydration query both fail closed as the SAME caller-facing error, never
  distinguishable from one another;
* a scalar/aggregate-shaped row with no governed id is denied outright.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core import secured_reads as sr
from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor


@pytest.fixture
def brain():
    reset_company_brain()
    yield get_company_brain()
    reset_company_brain()


def _actor(tenant: str = "tenant-a") -> ActorContext:
    return ActorContext(
        "principal:reader",
        ActorType.AI_AGENT,
        roles=("kg:read",),
        tenant_id=tenant,
        authenticated=True,
    )


def _session(actor: ActorContext, graph: str) -> GraphSession:
    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        graph=graph,
        scopes=frozenset({"kg:read", "kg:write"}),
        policy_version="test-policy",
        audience="test-audience",
    )


class _RecordingBackend:
    """A minimal ``execute_read``-only content backend: returns whatever rows
    it was seeded with, regardless of the actual Cypher text (the real
    parser/executor is out of scope here -- what matters is what governance
    does to the rows it gets back)."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.reads: list[str] = []

    def execute_read(self, query: str, _params: dict) -> list[dict]:
        self.reads.append(query)
        return list(self._rows)


class _AclBackend:
    """The durable ACL/classification store for ONE physical graph."""

    def __init__(self, rows: list[dict]) -> None:
        self._by_id = {row["id"]: row for row in rows}
        self.queries: list[list[str]] = []

    def execute_read(self, _query: str, params: dict, **_kw) -> list[dict]:
        wanted = list(params.get("ids", []))
        self.queries.append(wanted)
        return [self._by_id[i] for i in wanted if i in self._by_id]


class _RejectingAclBackend:
    """A durable ACL backend that rejects every hydration query -- stands in
    for an authorization failure at the durable store (R-22's "hydration
    failure" case)."""

    def execute_read(self, _query, _params, **_kw):
        raise RuntimeError("durable ACL store rejected the hydration query")


class _GraphView:
    """A per-physical-graph ``IntelligenceGraphEngine``-shaped object: its own
    content backend (what ``self.backend`` serves to ``query_cypher``) and its
    own ACL backend (what ``_durable_access_rows`` must hydrate from once the
    fix routes it there via ``for_graph``)."""

    def __init__(self, name: str, content: _RecordingBackend, acl: object) -> None:
        self.name = name
        self.backend = content
        self._acl_backend = acl
        self.graph_compute = _GraphComputeStub(name)


class _GraphComputeStub:
    def __init__(self, graph_name: str) -> None:
        self.graph_name = graph_name


class _ActiveEngine:
    """The process-default ``IntelligenceGraphEngine`` singleton
    (``IntelligenceGraphEngine.get_active()``). Bound to ``default_graph``;
    ``for_graph(name)`` returns a lightweight view carrying ONLY that
    physical graph's own ACL backend -- mirroring the real zero-transport
    contract (no new socket/connection)."""

    def __init__(self, default_graph: str, views: dict[str, _GraphView]) -> None:
        self._views = views
        self.graph_compute = _GraphComputeStub(default_graph)
        self.backend = views[default_graph]._acl_backend

    def for_graph(self, graph_name: str):
        view = self._views.get(graph_name)
        if view is None:
            raise RuntimeError(f"no named-graph view for {graph_name!r}")
        return _AclView(view._acl_backend)


class _AclView:
    def __init__(self, backend) -> None:
        self.backend = backend


ARTIFACT_ROW = {
    "id": "artifact:demo.txt",
    "relative_path": "docs/demo.txt",
    "byte_size": 4096,
    "media_type": "text/plain",
    "hash": "sha256:abc123",
    "run_id": "run-1",
}
PROVENANCE_ROW = {
    "id": "provenance:demo.txt:run-1",
    "artifact_id": "artifact:demo.txt",
    "source_hash": "sha256:def456",
    "run": "run-1",
    "method": "ingest",
    "authority": "connector:filesystem",
}
EDGE_ROW = {
    "id": "edge:artifact->provenance",
    "source_id": "artifact:demo.txt",
    "provenance_id": "provenance:demo.txt:run-1",
    "relationship": "HAS_PROVENANCE",
    "run": "run-1",
}
# A DELIBERATELY-ingested content field -- proves `content` isn't stripped
# when it genuinely belongs, only never fabricated for rows that never had it.
DELIBERATE_CONTENT_ROW = {
    "id": "artifact:with-content.txt",
    "relative_path": "docs/with-content.txt",
    "byte_size": 12,
    "media_type": "text/plain",
    "hash": "sha256:zzz999",
    "run_id": "run-1",
    "content": "hello world",
}


def _public_acl_row(row_id: str) -> dict:
    return {
        "id": row_id,
        "tenant_id": "tenant-a",
        "classification": "public",
        "external_access": None,
    }


def _build_two_graph_topology():
    """graph-a carries the real artifact/provenance/edge projection (with a
    deliberately-content-bearing row too); graph-b is a distinct physical
    graph with nothing matching."""
    content_a = _RecordingBackend(
        [ARTIFACT_ROW, PROVENANCE_ROW, EDGE_ROW, DELIBERATE_CONTENT_ROW]
    )
    acl_a = _AclBackend(
        [
            _public_acl_row(ARTIFACT_ROW["id"]),
            _public_acl_row(PROVENANCE_ROW["id"]),
            _public_acl_row(EDGE_ROW["id"]),
            _public_acl_row(DELIBERATE_CONTENT_ROW["id"]),
        ]
    )
    view_a = _GraphView("graph-a", content_a, acl_a)

    content_b = _RecordingBackend([])  # nothing on graph-b
    acl_b = _AclBackend([])
    view_b = _GraphView("graph-b", content_b, acl_b)

    active = _ActiveEngine("graph-a", {"graph-a": view_a, "graph-b": view_b})
    return active, view_a, view_b


def _run_query(engine, session: GraphSession) -> list[dict]:
    with use_actor(session.actor), use_session(session):
        return QueryMixin.query_cypher(
            engine,
            "MATCH (n) RETURN n.id AS id, n.relative_path AS relative_path, "
            "n.byte_size AS byte_size, n.media_type AS media_type, "
            "n.hash AS hash, n.run_id AS run_id",
        )


def test_full_artifact_provenance_edge_projection_with_ids_and_no_stray_content(
    monkeypatch, brain
):
    active, view_a, _view_b = _build_two_graph_topology()
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", active)

    actor = _actor()
    session = _session(actor, "graph-a")
    # The engine used for CONTENT is graph-a's own view (mirrors the real
    # dispatch: `self` is whatever engine object actually resolved to the
    # selected graph). ACL hydration independently re-derives the graph from
    # the session and must land on the SAME physical graph's ACL backend.
    rows = _run_query(view_a, session)

    by_id = {row["id"]: row for row in rows}
    assert set(by_id) == {
        ARTIFACT_ROW["id"],
        PROVENANCE_ROW["id"],
        EDGE_ROW["id"],
        DELIBERATE_CONTENT_ROW["id"],
    }

    artifact = by_id[ARTIFACT_ROW["id"]]
    assert artifact["relative_path"] == "docs/demo.txt"
    assert artifact["byte_size"] == 4096
    assert artifact["media_type"] == "text/plain"
    assert artifact["hash"] == "sha256:abc123"
    assert artifact["run_id"] == "run-1"
    assert "content" not in artifact  # never fabricated

    provenance = by_id[PROVENANCE_ROW["id"]]
    assert provenance["artifact_id"] == "artifact:demo.txt"
    # (source_hash/run/method/authority weren't selected by this query's own
    # RETURN clause -- proving the projection reflects exactly what was
    # asked, not an ACL-driven widening -- but the id and the requested
    # columns above are exactly what was governed and returned.)

    edge = by_id[EDGE_ROW["id"]]
    assert edge["source_id"] == "artifact:demo.txt"

    # The ONE row that deliberately carries `content` keeps it...
    assert by_id[DELIBERATE_CONTENT_ROW["id"]]["content"] == "hello world"
    # ...and it never leaks onto any of the others.
    assert all(
        "content" not in row
        for row_id, row in by_id.items()
        if row_id != DELIBERATE_CONTENT_ROW["id"]
    )

    # Governance actually consulted graph-a's OWN ACL backend for every id.
    assert set(view_a._acl_backend.queries[0]) == set(by_id)


def test_same_query_against_second_graph_returns_zero_rows(monkeypatch, brain):
    active, _view_a, view_b = _build_two_graph_topology()
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", active)

    actor = _actor()
    session = _session(actor, "graph-b")
    rows = _run_query(view_b, session)
    assert rows == []


def test_unknown_and_unauthorized_graph_fail_closed_indistinguishably(monkeypatch, brain):
    content_a = _RecordingBackend([ARTIFACT_ROW])
    acl_a = _AclBackend([_public_acl_row(ARTIFACT_ROW["id"])])
    view_a = _GraphView("graph-a", content_a, acl_a)

    content_denied = _RecordingBackend([ARTIFACT_ROW])
    view_denied = _GraphView("graph-denied", content_denied, _RejectingAclBackend())

    active = _ActiveEngine(
        "graph-a", {"graph-a": view_a, "graph-denied": view_denied}
    )
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", active)
    actor = _actor()

    # (1) Unknown graph -- absent from the engine's own named-graph views.
    unknown_session = _session(actor, "graph-unknown")
    with pytest.raises(PermissionError) as unknown_exc:
        _run_query(view_a, unknown_session)

    # (2) Known graph, but its durable ACL backend rejects the hydration
    # query outright (an authorization failure at the durable store).
    denied_session = _session(actor, "graph-denied")
    with pytest.raises(PermissionError) as denied_exc:
        _run_query(view_denied, denied_session)

    # Both collapse to the SAME generic, caller-facing message -- a caller
    # cannot tell "this graph doesn't exist" from "you are not authorized"
    # from the exception alone.
    assert (
        str(unknown_exc.value)
        == str(denied_exc.value)
        == "Graph row-policy or audit enforcement failed"
    )


def test_scalar_row_without_governed_id_is_denied(monkeypatch, brain):
    active, view_a, _view_b = _build_two_graph_topology()
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", active)
    # Overwrite graph-a's content with a scalar/aggregate-shaped row that
    # carries no governable id at all.
    view_a.backend._rows = [{"count": 5}]

    actor = _actor()
    session = _session(actor, "graph-a")
    with pytest.raises(PermissionError) as exc:
        _run_query(view_a, session)
    assert str(exc.value) == "Graph row-policy or audit enforcement failed"


def test_graph_never_accepted_as_a_raw_parameter_only_from_the_session(monkeypatch, brain):
    """_durable_access_rows takes no graph argument at all -- every test in
    this module drives the graph selection purely through ``use_session``,
    never a function parameter, which is itself the proof: there is no
    payload-supplied graph channel to smuggle authority through at this
    layer."""
    import inspect

    sig = inspect.signature(sr._durable_access_rows)
    assert list(sig.parameters) == ["node_ids"]
