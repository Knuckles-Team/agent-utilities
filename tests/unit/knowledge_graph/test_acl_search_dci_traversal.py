#!/usr/bin/python
"""NE-051 — ``QueryMixin.search_dci`` had NO ACL enforcement at all.

``search_dci`` (agent_utilities/knowledge_graph/orchestration/engine_query.py)
is a multi-hop graph-traversal retrieval surface — seed via vector search,
then expand outward along raw graph edges (``self.graph.get_successors`` /
``get_predecessors``), hydrating and returning every discovered node
completely unfiltered. Unlike ``search_hybrid``/``query_cypher`` it had no
``session`` parameter and no call anywhere into ``secured_reads``.

Fixed to reuse the SAME ``QueryMixin._enforce_acl_on_results`` boundary
``search_hybrid`` already applies (per-node ACL + owner/scope + audit),
applied to the seed pool (via ``search_hybrid(..., session=...)``) AND to
every hop's newly-discovered neighbor batch, BEFORE that batch can enter
``results`` (a ``top_k`` slot) or seed the next hop's frontier.

Design decision under test: unlike ``search_hybrid`` (``session=None`` is a
documented permissive no-op, kept for ~20 pre-existing unfiltered callers),
``search_dci`` is FAIL-CLOSED — it always resolves a session (explicit or
ambient) and raises ``SessionRequiredError`` (a ``PermissionError``) when
neither exists, rather than ever returning an unfiltered traversal. This was
judged safe because ``search_dci`` has exactly one production caller (the
MCP ``graph_search`` ``mode="dci"`` branch in ``agent_utilities/mcp/tools/
query_tools.py``, which already has the served request's ambient session in
scope and was updated to pass it) and no pre-existing test exercised it
against a real engine.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    SessionRequiredError,
    suspend_session,
    use_session,
)
from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin
from agent_utilities.models.company_brain import DataClassification, NodeACL
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext


class _StubRetriever:
    """Seed pool for ``search_hybrid`` — returns a fixed list regardless of
    query, matching ``test_engine_query_acl_wiring.py``'s existing pattern."""

    def __init__(self, nodes: list[dict]) -> None:
        self._nodes = nodes

    def retrieve_hybrid(self, query, **kwargs):  # noqa: ARG002
        return list(self._nodes)


class _FakeTraversalGraph:
    """Minimal engine-graph double exposing exactly the surface ``search_dci``
    uses: ``has_node``, ``get_successors``/``get_predecessors`` (directed
    adjacency, both directions traversed), and the batched hydration
    primitive ``_get_node_properties_batch``."""

    def __init__(self, edges: dict[str, list[str]]) -> None:
        self._successors = {k: list(v) for k, v in edges.items()}
        self._predecessors: dict[str, list[str]] = {}
        for src, dsts in edges.items():
            for dst in dsts:
                self._predecessors.setdefault(dst, []).append(src)
        self._known = set(edges) | set(self._predecessors)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._known

    def get_successors(self, node_id: str) -> list[str]:
        return list(self._successors.get(node_id, []))

    def get_predecessors(self, node_id: str) -> list[str]:
        return list(self._predecessors.get(node_id, []))

    def _get_node_properties_batch(self, node_ids: list[str]) -> dict[str, dict]:
        return {nid: {"id": nid, "importance_score": 0.5} for nid in node_ids}


class _Engine(QueryMixin):
    def __init__(self, seeds: list[dict], graph: _FakeTraversalGraph) -> None:
        self.hybrid_retriever = _StubRetriever(seeds)
        self.active_schema_pack = None
        self.graph = graph


def _actor(actor_id: str = "reader-1") -> ActorContext:
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.AI_AGENT,
        roles=("reader",),
        tenant_id="acme",
        authenticated=True,
    )


def _session(actor: ActorContext | None = None) -> GraphSession:
    return GraphSession(
        actor=actor or _actor(),
        tenant=(actor or _actor()).tenant_id,
        scopes=frozenset({"kg:read"}),
        graph="test-graph",
        policy_version="policy:test",
        audience="test-audience",
    )


def _grant_public(*node_ids: str) -> None:
    permissions = get_company_brain().permissions
    for node_id in node_ids:
        permissions.set_acl(
            NodeACL(node_id=node_id, classification=DataClassification.PUBLIC)
        )


@pytest.fixture(autouse=True)
def brain():
    reset_company_brain()
    yield get_company_brain()
    reset_company_brain()


def _seed(node_id: str, score: float = 0.9) -> dict:
    return {"id": node_id, "type": "Doc", "_score": score, "status": "ACTIVE"}


# ---------------------------------------------------------------------------
# Fail-closed default
# ---------------------------------------------------------------------------


def test_no_session_at_all_raises_rather_than_returning_unfiltered_traversal():
    """FAIL-CLOSED: with no explicit session and no ambient session bound,
    search_dci must raise, not silently traverse unfiltered (the opposite
    default from search_hybrid's session=None no-op).

    The workspace test harness's own ``isolate_graph_compute_engine`` autouse
    fixture (tests/conftest.py) binds an ambient session for every test by
    default, so "no ambient session" has to be constructed explicitly via
    ``suspend_session`` here to reproduce the true no-authority case.
    """
    graph = _FakeTraversalGraph({"s1": ["n1"]})
    engine = _Engine([_seed("s1")], graph)

    with suspend_session(), pytest.raises(SessionRequiredError):
        engine.search_dci("q")


# ---------------------------------------------------------------------------
# Item 3: denied on every node -> zero rows
# ---------------------------------------------------------------------------


def test_principal_denied_on_every_node_gets_zero_results():
    """No ACL is registered for ANY node in this graph (AU-P0-4 default-deny),
    so a fully-denied principal must get back an empty list, not a partially
    or fully unfiltered traversal."""
    graph = _FakeTraversalGraph({"s1": ["n1", "n2"], "n1": ["n3"], "n2": ["n4"]})
    engine = _Engine([_seed("s1")], graph)
    session = _session()

    with use_session(session):
        results = engine.search_dci("q", max_hops=2, session=session)

    assert results == []


# ---------------------------------------------------------------------------
# Item 4: per-hop filtering; a denied mid-chain node does not leak its
# neighbors through the evidence chain.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Infrastructure failure -> raise, never fall back to unfiltered results
# ---------------------------------------------------------------------------


def test_acl_infrastructure_failure_raises_rather_than_returning_unfiltered_results(
    monkeypatch,
):
    from agent_utilities.knowledge_graph.core import secured_reads

    def _boom(_rows, _actor):
        raise PermissionError("Row visibility evaluation failed") from ValueError(
            "synthetic infrastructure failure"
        )

    monkeypatch.setattr(secured_reads, "filter_rows", lambda rows, _actor: rows)
    monkeypatch.setattr(secured_reads, "visible", _boom)

    _grant_public("s1")
    graph = _FakeTraversalGraph({"s1": []})
    engine = _Engine([_seed("s1")], graph)
    session = _session()

    with use_session(session), pytest.raises(PermissionError):
        engine.search_dci("q", session=session)
