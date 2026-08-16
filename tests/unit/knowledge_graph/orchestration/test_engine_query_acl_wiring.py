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
