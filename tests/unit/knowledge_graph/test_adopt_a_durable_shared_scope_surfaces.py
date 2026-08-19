"""NE-042 acceptance — durable shared-scope ACL hydration (`4755f261`).

Gate: public/shared/private visibility must hold for FIELD, EDGE, TRAVERSAL,
and SEARCH projections, both before and after a cache reset and a restart,
including peer denial (a principal outside the share set gets zero rows).

`4755f261` fixed exactly one thing: ``secured_reads._durable_access_rows``'
Cypher fallback was silently dropping ``_shared_scope`` (aliased
``shared_scope``), so ``_hydrate_missing_acls`` never saw organization-share
evidence for a durable row that only exists as raw persisted node
properties (not yet cached in the in-process ``CompanyBrain`` permission
store) -- exactly the "governed projection returns empty rows for org-shared
data after restart" defect. ``tests/unit/knowledge_graph/test_secured_reads.
py`` proves this exhaustively at the ``permit()``/``_durable_access_rows()``
primitive layer. What it does NOT prove is that every one of the four
DIFFERENT CALLER surfaces the gate names actually reaches that primitive
uniformly:

* **field**  -- ``secured_reads.filter_rows``/``visible`` applied to a plain
  node-projection row (the same primitive ``query_cypher`` uses).
* **edge**   -- the identical primitive applied to an EDGE-shaped row (a
  nested node dict under a relationship alias, e.g. Cypher ``RETURN r``) --
  ``_row_node_id`` extracts identity the same way for both shapes, but that
  equivalence was never asserted with an org-shared row specifically.
* **traversal** -- ``QueryMixin.search_dci`` (multi-hop graph traversal
  retrieval, CONCEPT:AU-KG.memory.auto-similarity-memory-graph, the ``mode=
  'dci'`` MCP surface). See DEFECT below.
* **search** -- ``QueryMixin.search_hybrid`` -> ``_enforce_acl_on_results``
  -> the SAME ``visible(filter_rows(...))`` chain.

Cache reset / restart is approximated in-process (see each test's docstring
for exactly what is and is not exercised) by keeping the SAME durable-row
source (a monkeypatched ``_durable_access_rows``, standing in for the
persisted node properties that survive a restart) while calling
``reset_company_brain()`` (which clears the in-memory ``CompanyBrain``
permission cache -- the thing that does NOT survive a restart) between
reads. A genuine process restart is REQUIRES-LIVE-ACCEPTANCE; see the
run report.

STATUS for the traversal surface: DEFECT FOUND, not proven. See
``test_traversal_surface_search_dci_applies_no_acl_enforcement_at_all``.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core import secured_reads as sr
from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def brain():
    reset_company_brain()
    yield get_company_brain()
    reset_company_brain()


def _actor(
    actor_id: str = "principal:someone-else", *, tenant: str = "tenant-a"
) -> ActorContext:
    return ActorContext(
        actor_id, ActorType.AI_AGENT, roles=(), tenant_id=tenant, authenticated=True
    )


def _org_shared_row(node_id: str, *, owner: str = "principal:owner") -> dict:
    return {
        "tenant_id": "tenant-a",
        "classification": "confidential",
        "external_access": None,
        "owner_id": owner,
        "shared_scope": "org",
    }


# ---------------------------------------------------------------------------
# FIELD surface
# ---------------------------------------------------------------------------


def test_field_projection_admits_an_org_shared_row_for_a_same_tenant_non_owner(
    monkeypatch, brain
):
    monkeypatch.setattr(
        sr,
        "_durable_access_rows",
        lambda _ids: {"artifact-1": _org_shared_row("artifact-1")},
    )
    rows = [{"id": "artifact-1", "name": "field projection"}]
    with use_actor(_actor()):
        governed = sr.visible(sr.filter_rows(rows), _actor())
    assert [r["id"] for r in governed] == ["artifact-1"]


def test_field_projection_peer_denial_cross_tenant_gets_zero_rows(monkeypatch, brain):
    """Peer denial: a principal OUTSIDE the tenant (never in the org-share
    set) gets zero rows, not a redacted row."""
    monkeypatch.setattr(
        sr,
        "_durable_access_rows",
        lambda _ids: {"artifact-1": _org_shared_row("artifact-1")},
    )
    peer = _actor("principal:outsider", tenant="tenant-b")
    with use_actor(peer):
        governed = sr.visible(sr.filter_rows([{"id": "artifact-1"}]), peer)
    assert governed == []


# ---------------------------------------------------------------------------
# EDGE surface
# ---------------------------------------------------------------------------


def test_edge_projection_admits_an_org_shared_edge_row_for_a_same_tenant_non_owner(
    monkeypatch, brain
):
    """An edge-shaped result row (a nested node dict under a relationship
    alias, mirroring a Cypher ``MATCH ()-[r]->() RETURN r`` projection) must
    resolve identity and org-shared visibility exactly like a plain node row
    -- ``_row_node_id``/``filter_rows``/``visible`` are the same primitive for
    both shapes; this proves that equivalence holds for an org-shared row
    specifically, not just a plain PUBLIC/no-ACL one.
    """
    monkeypatch.setattr(
        sr, "_durable_access_rows", lambda _ids: {"edge-1": _org_shared_row("edge-1")}
    )
    edge_rows = [{"r": {"id": "edge-1", "type": "DEPENDS_ON"}}]
    with use_actor(_actor()):
        governed = sr.visible(sr.filter_rows(edge_rows), _actor())
    assert len(governed) == 1
    assert governed[0]["r"]["id"] == "edge-1"


def test_edge_projection_peer_denial_same_tenant_non_shared_gets_zero_rows(
    monkeypatch, brain
):
    """A PRIVATE (non-shared) edge is denied to a same-tenant non-owner peer."""
    private_row = {
        "tenant_id": "tenant-a",
        "classification": "confidential",
        "external_access": None,
        "owner_id": "principal:owner",
        "shared_scope": None,
    }
    monkeypatch.setattr(
        sr, "_durable_access_rows", lambda _ids: {"edge-2": private_row}
    )
    edge_rows = [{"r": {"id": "edge-2", "type": "DEPENDS_ON"}}]
    with use_actor(_actor()):
        governed = sr.visible(sr.filter_rows(edge_rows), _actor())
    assert governed == []


# ---------------------------------------------------------------------------
# SEARCH surface
# ---------------------------------------------------------------------------


class _StubRetriever:
    def __init__(self, nodes):
        self._nodes = nodes

    def retrieve_hybrid(self, query, **kwargs):
        return list(self._nodes)


class _Engine(QueryMixin):
    def __init__(self, nodes):
        self.hybrid_retriever = _StubRetriever(nodes)
        self.active_schema_pack = None


def _session(actor: ActorContext) -> GraphSession:
    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:read"}),
        graph="test-graph",
        policy_version="policy:test",
        audience="test-audience",
    )


def test_search_surface_search_hybrid_admits_an_org_shared_node(monkeypatch, brain):
    monkeypatch.setattr(
        sr, "_durable_access_rows", lambda _ids: {"doc-1": _org_shared_row("doc-1")}
    )
    actor = _actor()
    session = _session(actor)
    nodes = [{"id": "doc-1", "type": "Doc", "_score": 0.9, "status": "ACTIVE"}]
    with use_session(session):
        out = _Engine(nodes).search_hybrid("q", top_k=5, session=session)
    assert [n["id"] for n in out] == ["doc-1"]


def test_search_surface_peer_denial_cross_tenant_gets_zero_rows(monkeypatch, brain):
    monkeypatch.setattr(
        sr, "_durable_access_rows", lambda _ids: {"doc-1": _org_shared_row("doc-1")}
    )
    peer = _actor("principal:outsider", tenant="tenant-b")
    session = _session(peer)
    nodes = [{"id": "doc-1", "type": "Doc", "_score": 0.9, "status": "ACTIVE"}]
    with use_session(session):
        out = _Engine(nodes).search_hybrid("q", top_k=5, session=session)
    assert out == []


# ---------------------------------------------------------------------------
# Cache reset + restart approximation
# ---------------------------------------------------------------------------


def test_org_shared_visibility_holds_after_an_in_process_cache_reset(
    monkeypatch, brain
):
    """Approximates "restart": the durable row SOURCE (monkeypatched
    ``_durable_access_rows``, standing in for persisted node properties that
    survive a real restart) is unchanged across the reset, but
    ``reset_company_brain()`` clears the in-memory permission CACHE the fix
    populates on first hydration -- the thing that does NOT survive a real
    restart. Visibility must be recomputed correctly from the durable source
    both times, not merely remembered from the first hydration.

    This is an in-process approximation, not a real process restart -- see
    the run report for what a genuine restart would additionally need to
    prove (a fresh Python process / durable engine handle).
    """
    monkeypatch.setattr(
        sr, "_durable_access_rows", lambda _ids: {"doc-1": _org_shared_row("doc-1")}
    )
    actor = _actor()

    with use_actor(actor):
        assert sr.permit(["doc-1"]) == ["doc-1"]

    # Simulate the cache not surviving a restart: drop the in-memory
    # CompanyBrain permission store entirely. The durable source (still
    # monkeypatched, i.e. "still on disk") is untouched.
    reset_company_brain()

    with use_actor(actor):
        assert sr.permit(["doc-1"]) == ["doc-1"], (
            "org-shared visibility did not survive a cache reset -- "
            "re-hydration from the durable source failed the second time"
        )


def test_peer_denial_holds_after_an_in_process_cache_reset(monkeypatch, brain):
    """The negative half of the restart approximation above: a peer's denial
    must ALSO still hold after the cache is reset and re-hydrated -- a reset
    must never accidentally WIDEN access either.
    """
    monkeypatch.setattr(
        sr, "_durable_access_rows", lambda _ids: {"doc-1": _org_shared_row("doc-1")}
    )
    peer = _actor("principal:outsider", tenant="tenant-b")

    with use_actor(peer):
        assert sr.permit(["doc-1"]) == []

    reset_company_brain()

    with use_actor(peer):
        assert sr.permit(["doc-1"]) == []


# ---------------------------------------------------------------------------
# TRAVERSAL surface — DEFECT FOUND
# ---------------------------------------------------------------------------


class _FakeTraversalGraph:
    """Minimal double for the ``self.graph`` protocol ``search_dci`` walks
    (``has_node``/``get_successors``/``get_predecessors``/
    ``_get_node_properties_batch``) -- no ACL/session concept at all, mirroring
    the real low-level graph-view surface these calls actually hit.
    """

    def __init__(self, nodes: dict[str, dict], edges: list[tuple[str, str]]):
        self._nodes = nodes
        self._edges = edges

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_successors(self, node_id: str) -> list[str]:
        return [b for a, b in self._edges if a == node_id]

    def get_predecessors(self, node_id: str) -> list[str]:
        return [a for a, b in self._edges if b == node_id]

    def _get_node_properties_batch(self, ids: list[str]) -> dict[str, dict]:
        return {i: dict(self._nodes.get(i, {})) for i in ids}


class _DciEngine(QueryMixin):
    def __init__(
        self, nodes: dict[str, dict], edges: list[tuple[str, str]], seed_id: str
    ):
        self.graph = _FakeTraversalGraph(nodes, edges)
        self.active_schema_pack = None
        self._seed_id = seed_id

    def search_hybrid(self, query, top_k=5, **kwargs):  # noqa: ANN001
        # `search_dci`'s vector seed step -- pretend the seed itself is
        # authorized (this test isolates the TRAVERSAL/hop-expansion step,
        # which is what has no enforcement at all).
        node = dict(self.graph._nodes[self._seed_id])
        node["id"] = self._seed_id
        return [node]


def test_traversal_surface_search_dci_denies_a_cross_tenant_neighbor(
    monkeypatch, brain
):
    """NE-042 acceptance, inverted after NE-051 closed the defect it found.

    Originally a CHARACTERIZATION test: ``QueryMixin.search_dci`` had no
    ``session`` parameter and called no ``secured_reads`` primitive anywhere
    in its hop-expansion loop, so a traversal-discovered neighbour was
    hydrated and returned completely unfiltered. That was a sharper gap than
    the crowd-out defects, because there was no enforcement boundary at all.
    Its docstring said plainly that it documented the CURRENT behaviour and
    was "not the desired contract", so that a fix would make it fail.

    The fix landed (NE-051). The assertion is therefore INVERTED, not
    relaxed: the same cross-tenant neighbour that this test previously
    proved was returned must now be absent. The peer-denial setup mirrors
    ``test_search_surface_peer_denial_cross_tenant_gets_zero_rows`` above so
    the traversal surface is held to the identical boundary as the search
    surface -- which was the whole point of NE-042.
    """
    monkeypatch.setattr(
        sr,
        "_durable_access_rows",
        lambda _ids: {"denied-neighbor": _org_shared_row("denied-neighbor")},
    )
    peer = _actor("principal:outsider", tenant="tenant-b")
    session = _session(peer)

    nodes = {
        "seed": {"tenant_id": "tenant-a"},
        "denied-neighbor": {"tenant_id": "tenant-b", "classification": "confidential"},
    }
    edges = [("seed", "denied-neighbor")]
    engine = _DciEngine(nodes, edges, seed_id="seed")

    with use_session(session):
        results = engine.search_dci("q", max_hops=1, top_k=10, session=session)

    assert "denied-neighbor" not in {r["id"] for r in results}, (
        "search_dci returned a neighbour the caller is not entitled to see; "
        "NE-051's per-hop enforcement is not holding"
    )
