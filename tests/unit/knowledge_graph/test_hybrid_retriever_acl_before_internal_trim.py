"""GOC-83-W04 (U-107/U-132) — ACL enforcement must run BEFORE
``HybridRetriever.retrieve_hybrid``'s OWN internal candidate-pool trim, not
just before ``search_hybrid``'s final return.

``engine_query.QueryMixin.search_hybrid`` was fixed (commit ``389f55657``) to
run ``_enforce_acl_on_results`` on ``retrieve_hybrid``'s RETURN VALUE before
``search_hybrid``'s own archive-filter/score_gate trim. That closes the leak
at ``search_hybrid``'s layer — but ``retrieve_hybrid`` itself ALREADY sorts
and trims its raw vector-arm candidates down to (a bounded multiple of)
``context_window`` internally, via ``_rerank_candidates``, entirely before
returning to ``search_hybrid``. A denied high-score candidate can consume one
of those internal slots and permanently exclude a lower-scored AUTHORIZED
candidate from ever reaching ``retrieve_hybrid``'s return value — at which
point NO downstream ACL pass, however early, can recover it; the candidate
was never fetched hydrated in the first place.

The existing regression test for this invariant
(``tests/unit/knowledge_graph/orchestration/test_engine_query_acl_wiring.py``)
uses a STUB retriever (``retrieve_hybrid`` just returns a fixed list) that
never exercises this internal sort/trim code path at all — it would pass
identically whether or not this deeper defect were fixed. These tests drive
the REAL ``HybridRetriever`` against a fake engine-graph double (the same
``_FakeVectorGraph`` contract as
``tests/unit/knowledge_graph/test_backlink_boost.py``'s existing kernel-free
harness), through the full ``QueryMixin.search_hybrid`` entrypoint, so the
internal engine-ANN-fetch → sort → trim path actually runs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin
from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import HybridRetriever
from agent_utilities.models.company_brain import ActorType, DataClassification, NodeACL
from agent_utilities.security.brain_context import ActorContext


def _actor(actor_id: str = "reader-1") -> ActorContext:
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.AI_AGENT,
        roles=("reader",),
        tenant_id="acme",
        authenticated=True,
    )


def _session(actor: ActorContext) -> GraphSession:
    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:read"}),
        graph="test-graph",
        policy_version="policy:test",
        audience="test-audience",
    )


class _NoCypherBackend:
    """Truthy backend whose Cypher ``execute`` yields nothing (forces the ANN
    path — matches ``test_backlink_boost.py``'s existing double)."""

    def execute(self, _q, _p=None):
        return []


class _FakeVectorGraph:
    """Engine-graph double: the vector arm reads ``semantic_search`` (engine
    ANN) and hydrates via ``_get_node_properties`` — the SAME contract
    ``test_backlink_boost.py`` already exercises. ``hits`` must be given in
    descending-score order: ``_engine_vector_search`` stops collecting once it
    has ``top_k`` hydrated results, it does not itself re-sort ``hits``.
    """

    def __init__(self, hits, props):
        self._hits = hits  # list[(id, score)], descending
        self._props = props  # dict[id -> props]

    def query_unified(self, _plan, **_k):
        return []  # no label seed in these tests → falls to native ANN below

    def semantic_search(self, _emb, _n=5):
        return list(self._hits)

    def _get_node_properties(self, nid):
        return dict(self._props.get(nid, {}))

    def has_node(self, nid):
        return nid in self._props

    def get_successors(self, _nid):
        return []

    def get_predecessors(self, _nid):
        return []


class _Engine(QueryMixin):
    def __init__(self, graph):
        self.graph = graph
        self.backend = _NoCypherBackend()
        self.active_schema_pack = None
        # enable_rerank=False: the default `ReasoningAwareReranker` reorders
        # by a query-relevance heuristic, which would make the exact-score
        # crowd-out scenario below non-deterministic. Disabling it makes
        # `_rerank_candidates` the plain `scored_nodes[:context_window]`
        # slice — deterministic on the `_score` this test controls.
        self.hybrid_retriever = HybridRetriever(self, enable_rerank=False)
        mock_embed = MagicMock()
        mock_embed.get_text_embedding.side_effect = lambda _text: [1.0, 0.0]
        self.hybrid_retriever.embed_model = mock_embed


@pytest.fixture
def brain():
    reset_company_brain()
    yield get_company_brain()
    reset_company_brain()


# Descending-score candidate pool: "denied-top" outranks every authorized
# candidate, including "auth-4" — the one that only fits inside a top-3 cut
# if "denied-top" was excluded BEFORE, not after, that cut.
_HITS = [
    ("denied-top", 0.99),
    ("auth-2", 0.90),
    ("auth-3", 0.80),
    ("auth-4", 0.70),
]
_PROPS = {
    "denied-top": {"id": "denied-top", "name": "d", "status": "ACTIVE"},
    "auth-2": {"id": "auth-2", "name": "a2", "status": "ACTIVE"},
    "auth-3": {"id": "auth-3", "name": "a3", "status": "ACTIVE"},
    "auth-4": {"id": "auth-4", "name": "a4", "status": "ACTIVE"},
}


@patch(
    "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model"
)
def test_denied_row_starves_an_authorized_row_inside_retrieve_hybrids_own_trim(
    _create_embed_mock, brain
):
    """CONTROL case, called DIRECTLY on ``HybridRetriever.retrieve_hybrid``
    (bypassing ``search_hybrid``'s own ``top_k*2`` window sizing and
    ``score_gate`` entirely, so this isolates EXACTLY the unit under test)
    with ``context_window=3`` and NO ``session`` — identical to every
    pre-existing internal caller. The internal engine-ANN fetch is bounded to
    exactly 3: "denied-top" occupies one of only 3 raw slots and "auth-4" is
    never even fetched, let alone hydrated or returned. This proves the
    scenario is a genuine raw-fetch-window bound (not a fixture artifact) —
    the next test proves a `session` closes it.
    """
    for node_id in ("auth-2", "auth-3", "auth-4"):
        brain.permissions.set_acl(
            NodeACL(node_id=node_id, classification=DataClassification.PUBLIC)
        )
    engine = _Engine(_FakeVectorGraph(hits=_HITS, props=_PROPS))

    out = engine.hybrid_retriever.retrieve_hybrid(
        "q", context_window=3, relevance_threshold=0.0, skip_quality_gate=True
    )
    ids = [n["id"] for n in out]
    assert "denied-top" in ids  # no session → completely unfiltered, as documented
    assert "auth-4" not in ids, (
        "control case failed: auth-4 should be unreachable at context_window=3 "
        f"regardless of ACL. Got {ids!r}"
    )


@patch(
    "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model"
)
def test_denied_high_score_row_no_longer_starves_an_authorized_row_via_internal_trim(
    _create_embed_mock, brain
):
    """GOC-83-W04 fix proof: called DIRECTLY on ``retrieve_hybrid`` (same
    isolation as the control case above) WITH a ``session`` and the SAME
    ``context_window=3``. ``retrieve_hybrid`` now overfetches its raw
    candidate pool (a bounded multiple of ``context_window``, not the bare
    ``context_window`` the control case used) and ACL-filters it BEFORE its
    own internal sort/trim-to-``context_window`` — so "denied-top" is removed
    from the pool before that cut, not after, and "auth-4" (unreachable in
    the no-session control case above, at the SAME ``context_window``) now
    legitimately fills the 3rd slot.
    """
    for node_id in ("auth-2", "auth-3", "auth-4"):
        brain.permissions.set_acl(
            NodeACL(node_id=node_id, classification=DataClassification.PUBLIC)
        )
    # "denied-top" gets NO registered ACL — denied fail-closed.
    engine = _Engine(_FakeVectorGraph(hits=_HITS, props=_PROPS))

    actor = _actor()
    session = _session(actor)
    with use_session(session):
        out = engine.hybrid_retriever.retrieve_hybrid(
            "q",
            context_window=3,
            relevance_threshold=0.0,
            skip_quality_gate=True,
            session=session,
        )
    ids = {n["id"] for n in out}
    assert "denied-top" not in ids
    assert ids == {"auth-2", "auth-3", "auth-4"}, (
        "a denied high-score row starved an authorized row inside "
        "retrieve_hybrid's OWN internal trim — ACL must be applied to the "
        f"RAW candidate pool before that trim, not just before search_hybrid's "
        f"return. Got {sorted(ids)}"
    )


# NOTE on why there is no "end-to-end through search_hybrid" variant of the
# above: search_hybrid's own score_gate caps the FINAL result to
# max_results=top_k, which is always < retrieve_hybrid's own
# context_window=top_k*2 — so a candidate only reachable because of THIS
# fix (starved by retrieve_hybrid's internal trim, recovered by the
# overfetch) sits at a rank between top_k+1 and 2*top_k and is therefore
# ALWAYS cut by search_hybrid's own max_results cap regardless of whether
# retrieve_hybrid's internal trim is fixed — an end-to-end assertion on
# search_hybrid's final output cannot discriminate this specific channel
# (it would pass identically with or without this fix, which would make it
# a "gate that never demonstrated against a known-bad input" — worthless as
# evidence, not just redundant). The two tests above, called directly
# against `retrieve_hybrid` at matched `context_window`, are the correct
# and sufficient isolation boundary for this channel; the "score
# normalization" and cross-principal "cache" channels ARE fully visible at
# search_hybrid's own boundary and are covered end-to-end in
# tests/unit/knowledge_graph/orchestration/test_engine_query_acl_wiring.py.
