"""ACL-aware vector/hybrid retrieval wiring (CONCEPT:AU-KG.retrieval.acl-aware-vector-retrieval).

``QueryMixin.search_hybrid`` (the engine method behind the served ``graph_search``
MCP tool) previously returned every ranked node completely unfiltered — no
per-node ACL check, no owner/scope visibility, no read audit — unlike the
guarded ``query_cypher`` Cypher path, which has enforced all three since
``AU-P0-4``. These tests drive the REAL ``secured_reads``/``company_brain``
permission stack (no mocking of the ACL seam itself — only the retriever is a
stub, matching ``tests/unit/knowledge_graph/test_search_score_contract.py``'s
existing kernel-free harness) to prove:

1. With no ``session`` argument, behaviour is byte-for-byte unchanged (the
   ~20 existing internal callers that never pass one are unaffected).
2. With a ``session``, a node with NO registered ACL is denied by default
   (CONCEPT:AU-P0-4 fail-closed) rather than silently returned.
3. With a ``session``, a node whose ACL grants the actor read access IS
   returned — enforcement is discriminating, not a blanket empty-result.
4. A read audit entry is actually recorded for the governed call.

GOC-83-W04 (U-107/U-132's "remaining half") extends this with the two leak
channels a denied row can escape through even once it is provably absent
from the *returned list itself* (the crowd-out test below covers that
literal top-k channel):

5. A denied outlier must not skew ``score_gate``'s fused-score mean/stddev
   fit and cause a DIFFERENT, unrelated authorized row to be trimmed by the
   weak-tail cut — i.e. the survivor COUNT and membership of the response
   must be identical whether or not a denied candidate was ever in the
   candidate pool (``test_denied_outlier_does_not_skew_score_gate_for_an_
   unrelated_authorized_row``).
6. Two principals querying the SAME stub retriever/candidate pool back to
   back must each get their own correctly-scoped result — nothing about the
   first principal's (unfiltered) candidate set or the first call's ACL
   outcome may be reused/cached across the second, differently-scoped call
   (``test_back_to_back_calls_under_different_principals_are_never_cross_
   contaminated``).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin
from agent_utilities.models.company_brain import ActorType, DataClassification, NodeACL
from agent_utilities.security.brain_context import ActorContext


def _actor(actor_id: str = "reader-1", tenant: str = "acme") -> ActorContext:
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.AI_AGENT,
        roles=("reader",),
        tenant_id=tenant,
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


class _StubRetriever:
    def __init__(self, nodes):
        self._nodes = nodes

    def retrieve_hybrid(self, query, **kwargs):  # noqa: ANN001, ANN003
        return list(self._nodes)


class _Engine(QueryMixin):
    def __init__(self, nodes) -> None:
        self.hybrid_retriever = _StubRetriever(nodes)
        self.active_schema_pack = None


@pytest.fixture
def brain():
    reset_company_brain()
    yield get_company_brain()
    reset_company_brain()


_NODES = [
    {"id": "public-doc", "type": "Doc", "_score": 0.9, "status": "ACTIVE"},
    {"id": "secret-doc", "type": "Doc", "_score": 0.8, "status": "ACTIVE"},
]


def test_no_session_is_unfiltered_passthrough(brain):
    """Zero behaviour change for the ~20 existing callers that pass no session."""
    out = _Engine(_NODES).search_hybrid("q", top_k=5)
    assert {n["id"] for n in out} == {"public-doc", "secret-doc"}


def test_session_denies_nodes_with_no_registered_acl(brain):
    """CONCEPT:AU-P0-4 — an unclassified node is denied by default, not returned."""
    actor = _actor()
    session = _session(actor)
    with use_session(session):
        out = _Engine(_NODES).search_hybrid("q", top_k=5, session=session)
    assert out == []


def test_session_admits_a_node_the_actor_is_permitted_to_read(brain):
    """Enforcement discriminates: a PUBLIC node is returned, an unclassified
    sibling in the SAME result set is not."""
    brain.permissions.set_acl(
        NodeACL(node_id="public-doc", classification=DataClassification.PUBLIC)
    )
    actor = _actor()
    session = _session(actor)
    with use_session(session):
        out = _Engine(_NODES).search_hybrid("q", top_k=5, session=session)
    assert [n["id"] for n in out] == ["public-doc"]


def test_session_records_a_read_audit(brain):
    brain.permissions.set_acl(
        NodeACL(node_id="public-doc", classification=DataClassification.PUBLIC)
    )
    actor = _actor()
    session = _session(actor)
    before = brain.provenance.read_count
    with use_session(session):
        _Engine(_NODES).search_hybrid("q", top_k=5, session=session)
    assert brain.provenance.read_count == before + 1


def test_denied_high_score_candidate_cannot_crowd_out_an_authorized_result(brain):
    """U-107/U-132 — ACL enforcement must run BEFORE the score gate trims to
    ``top_k``, not after.

    Regression for the crowd-out defect: an unauthorized (no registered ACL,
    denied fail-closed) candidate scored highest and an authorized PUBLIC
    candidate scored lowest. With ``top_k=1``, applying the score gate first
    keeps ONLY the denied high-score node — ACL then strips it and the
    legitimately authorized result never reaches the caller at all, even
    though it was retrieved and the actor is entitled to read it. Enforcing
    ACL on the raw candidate set first means the denied node never occupies
    the single top-k slot, so the authorized node is returned.
    """
    brain.permissions.set_acl(
        NodeACL(node_id="permitted-low", classification=DataClassification.PUBLIC)
    )
    nodes = [
        {"id": "denied-high", "type": "Doc", "_score": 0.95, "status": "ACTIVE"},
        {"id": "permitted-low", "type": "Doc", "_score": 0.1, "status": "ACTIVE"},
    ]
    actor = _actor()
    session = _session(actor)
    with use_session(session):
        out = _Engine(nodes).search_hybrid("q", top_k=1, session=session)
    assert [n["id"] for n in out] == ["permitted-low"]


def test_denied_outlier_does_not_skew_score_gate_for_an_unrelated_authorized_row(
    brain,
):
    """GOC-83-W04 — the "score normalization" leak channel: a denied row must
    not influence ``score_gate``'s fused mean/stddev fit even for an
    AUTHORIZED row it never directly competes with for a top-k slot.

    7 candidates: 5 clustered ~0.4-0.7 authorized rows, 1 low-scoring (0.398)
    AUTHORIZED row, and 1 denied (no registered ACL) row scored 0.758 — high,
    but not high enough to consume the single contested slot (``top_k=20`` is
    far larger than the candidate pool, so nothing is crowded out by RANK
    alone; this isolates the normalization channel from the crowd-out one
    already covered above).

    With the denied row mixed into the population BEFORE the weak-tail z-cut
    (the pre-fix order), its outlier score pulls the fused mean/stddev enough
    that the low authorized row's z-score falls to -1.08 (below the -1.0
    keep threshold) and it is wrongly trimmed — even though it was never
    going to lose a rank-based top-k contest. Enforcing ACL BEFORE
    score_gate (the fix) fits the mean/stddev over only the 6 authorized
    rows, where the same row's z-score is -0.96 (above threshold) and it
    survives. This is exact real arithmetic against the production
    ``score_gate``, not a hand-simulated approximation — the numbers were
    solved for exactly this crossing.
    """
    for node_id in ("c1", "c2", "c3", "c4", "c5", "low-authorized"):
        brain.permissions.set_acl(
            NodeACL(node_id=node_id, classification=DataClassification.PUBLIC)
        )
    # "denied-outlier" gets NO registered ACL — denied fail-closed.
    nodes = [
        {"id": "c1", "type": "Doc", "_score": 0.453, "status": "ACTIVE"},
        {"id": "c2", "type": "Doc", "_score": 0.691, "status": "ACTIVE"},
        {"id": "c3", "type": "Doc", "_score": 0.699, "status": "ACTIVE"},
        {"id": "c4", "type": "Doc", "_score": 0.422, "status": "ACTIVE"},
        {"id": "c5", "type": "Doc", "_score": 0.452, "status": "ACTIVE"},
        {"id": "low-authorized", "type": "Doc", "_score": 0.398, "status": "ACTIVE"},
        {"id": "denied-outlier", "type": "Doc", "_score": 0.758, "status": "ACTIVE"},
    ]
    actor = _actor()
    session = _session(actor)
    with use_session(session):
        out = _Engine(nodes).search_hybrid("q", top_k=20, session=session)
    ids = {n["id"] for n in out}
    assert "denied-outlier" not in ids  # the literal top-k/return-list channel
    # The normalization channel: the low-scoring AUTHORIZED row must survive.
    # Its presence/absence here depends ONLY on whether ACL ran before or
    # after score_gate's mean/stddev fit — not on its own rank or score.
    assert "low-authorized" in ids, (
        "a denied outlier skewed score_gate's fused-score normalization and "
        "wrongly trimmed an unrelated authorized row — ACL must run BEFORE "
        f"score_gate, not after. Got ids={sorted(ids)}"
    )
    assert ids == {"c1", "c2", "c3", "c4", "c5", "low-authorized"}


def test_back_to_back_calls_under_different_principals_are_never_cross_contaminated(
    brain,
):
    """GOC-83-W04 — the "cache" leak channel: nothing about one principal's
    call may leak into the very next call for a DIFFERENT principal, even
    against the identical stub retriever / candidate pool (same ``_Engine``
    instance, same nodes, same query string — the only thing that changes is
    the session). There is no cache in this path today; this is a permanent
    regression guard against one being added without per-principal scoping,
    proven by actually alternating two principals with disjoint grants twice
    each and checking every response, not just the first.
    """
    nodes = [
        {"id": "alpha-doc", "type": "Doc", "_score": 0.9, "status": "ACTIVE"},
        {"id": "beta-doc", "type": "Doc", "_score": 0.8, "status": "ACTIVE"},
    ]
    brain.permissions.set_acl(
        NodeACL(
            node_id="alpha-doc",
            classification=DataClassification.INTERNAL,
            data_owner="reader-alpha",
        )
    )
    brain.permissions.set_acl(
        NodeACL(
            node_id="beta-doc",
            classification=DataClassification.INTERNAL,
            data_owner="reader-beta",
        )
    )
    engine = _Engine(nodes)
    alpha = _session(_actor("reader-alpha"))
    beta = _session(_actor("reader-beta"))

    for _round in range(2):
        with use_session(alpha):
            out_alpha = engine.search_hybrid("q", top_k=5, session=alpha)
        assert [n["id"] for n in out_alpha] == ["alpha-doc"], (
            f"round {_round}: reader-alpha got {[n['id'] for n in out_alpha]!r} "
            "— a prior call's (or the other principal's) result leaked across"
        )

        with use_session(beta):
            out_beta = engine.search_hybrid("q", top_k=5, session=beta)
        assert [n["id"] for n in out_beta] == ["beta-doc"], (
            f"round {_round}: reader-beta got {[n['id'] for n in out_beta]!r} "
            "— a prior call's (or the other principal's) result leaked across"
        )
