"""NE-038 acceptance — GitLab requested-project narrowing and ACL-before-rank
(`389f5565`), the "cache identity" / citations half of the gate.

The gate requires: "a denied high-score row must never enter ranking,
trimming, reranking, citations, or cache identity ... construct a case where
a denied document would rank first, and prove it is absent from every one of
those five surfaces."

``engine_query.QueryMixin.search_hybrid`` was the fix target of `389f5565`:
its own ``_enforce_acl_on_results`` now runs BEFORE ``score_gate``'s
rank/trim, closing the literal crowd-out channel at THAT layer (see
``tests/unit/knowledge_graph/orchestration/test_engine_query_acl_wiring.py``
and ``tests/unit/knowledge_graph/test_hybrid_retriever_acl_before_internal_
trim.py`` for the exhaustive proof of that).

``ContextCompiler`` (``agent_utilities/knowledge_graph/retrieval/
context_compiler.py``) is a SEPARATE, higher-level "citations + cache
identity" surface layered on top of ``search_hybrid`` -- it is the thing
that literally computes a bundle ``cache_key`` (``compute_bundle_cache_key``)
and assembles the ``citations`` list the gate names. Its own ``_retrieve``
helper (line ~602-609) calls::

    search_hybrid(query, top_k=top_k, as_of=as_of or None)

WITHOUT forwarding the ``session=`` kwarg it already has in scope. Because
``QueryMixin._enforce_acl_on_results`` is deliberately opt-in on the EXPLICIT
``session`` argument only (never the ambient session), this omission means
``search_hybrid``'s internal ``score_gate`` trim runs COMPLETELY UNFILTERED
for every ``ContextCompiler.compile()`` call that reaches this branch (any
engine exposing ``search_hybrid`` and no explicit ``hybrid_retriever=``
override) -- the exact `389f5565` crowd-out defect, reopened one layer up,
outside that commit's own file list (``engine_query.py``/``gitlab_indexer.py``
only) and therefore never covered by its tests.

This was verified empirically (interactively, against this exact source
tree) before writing this test: with ``candidate_pool=top_k=1``, a denied
high-score candidate wins ``search_hybrid``'s internal top-1 trim before
``ContextCompiler``'s own ``enforce()`` policy step ever sees the candidate
pool, so the authorized low-score candidate is dropped from ``candidates``
itself -- not merely from the final bundle. The denied document is correctly
absent from citations/cache identity (as the gate requires), but so is the
UNRELATED AUTHORIZED document that should have appeared -- the sharper form
of the exact crowd-out defect `389f5565` closed at the ``search_hybrid``
layer, reopened here.

STATUS: DEFECT FOUND. ``test_authorized_low_score_document_survives_context_
compiler_when_a_denied_document_would_rank_first`` documents the desired
(currently unmet) contract and is expected to FAIL against the current
source -- see file:line evidence in its own docstring. It is intentionally
not marked ``xfail``: this program's instructions are that a gate must be
reported as unproven/failing, never disguised as passing.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.ontology.permissioning import (
    clear_markings,
    use_marking_authority,
)
from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin
from agent_utilities.knowledge_graph.retrieval.context_compiler import ContextCompiler
from agent_utilities.models.company_brain import ActorType, DataClassification, NodeACL
from agent_utilities.security.brain_context import ActorContext


class _FakeMarkingStore:
    """``ContextCompiler.compile`` resolves the mandatory-marking store on
    every call (see ``tests/retrieval/test_context_compiler.py``'s identical
    fixture docstring) regardless of whether a marking was ever applied.
    """

    @staticmethod
    def execute(_query, _params):
        return []


@pytest.fixture(autouse=True)
def _clean_state():
    reset_company_brain()
    clear_markings()
    with use_marking_authority(_FakeMarkingStore()):
        yield
    reset_company_brain()
    clear_markings()


class _StubRetriever:
    """Duck-typed ``HybridRetriever`` stand-in whose ``retrieve_hybrid`` just
    returns a fixed, RAW-scored candidate pool -- the real production
    ``QueryMixin.search_hybrid`` (not a hand-written per-test fake) is what
    performs the score-gate trim under test.
    """

    def __init__(self, nodes):
        self._nodes = nodes

    def retrieve_hybrid(self, query, **kwargs):
        return list(self._nodes)


class _Engine(QueryMixin):
    """A real ``QueryMixin.search_hybrid`` bound to a stub retriever -- this
    is what makes ``ContextCompiler._retrieve`` take the
    ``search_hybrid(...)`` branch (no explicit ``hybrid_retriever=``
    override), the branch the missing ``session=`` forward affects.
    """

    def __init__(self, nodes):
        self.hybrid_retriever = _StubRetriever(nodes)
        self.active_schema_pack = None


def _actor(actor_id: str = "reader-1") -> ActorContext:
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.AI_AGENT,
        roles=("reader",),
        tenant_id="acme",
        authenticated=True,
    )


def _session() -> GraphSession:
    return GraphSession(
        actor=_actor(),
        tenant="acme",
        scopes=frozenset({"kg:read"}),
        graph="test-graph",
        policy_version="policy:test",
        audience="test-audience",
    )


def test_authorized_low_score_document_survives_context_compiler_when_a_denied_document_would_rank_first():
    """DEFECT FOUND: ``ContextCompiler._retrieve`` (agent_utilities/
    knowledge_graph/retrieval/context_compiler.py, the ``search_hybrid(query,
    top_k=top_k, as_of=as_of or None)`` call around line 606) does not
    forward ``session=`` into ``search_hybrid``, so the `389f5565`
    ACL-before-rank fix never engages for this pipeline. With
    ``candidate_pool=top_k=1`` and a denied candidate outscoring an
    authorized one, the authorized candidate is silently dropped before
    ``ContextCompiler``'s own ``enforce()`` step ever runs -- proven by this
    assertion currently FAILING (``permitted-low`` is absent from
    ``bundle.citations``/``bundle.items``/``bundle.cache_key``'s backing id
    set, even though the actor is entitled to read it and no OTHER candidate
    would have consumed its slot under correct ACL-before-rank ordering).
    """
    brain = get_company_brain()
    brain.permissions.set_acl(
        NodeACL(node_id="permitted-low", classification=DataClassification.PUBLIC)
    )
    nodes = [
        {
            "id": "denied-high",
            "type": "Doc",
            "_score": 0.95,
            "status": "ACTIVE",
            "content": "denied content",
        },
        {
            "id": "permitted-low",
            "type": "Doc",
            "_score": 0.1,
            "status": "ACTIVE",
            "content": "permitted content",
        },
    ]
    session = _session()
    engine = _Engine(nodes)
    compiler = ContextCompiler(engine)

    with use_session(session):
        bundle = compiler.compile("q", session=session, top_k=1, candidate_pool=1)

    citation_ids = {c.node_id for c in bundle.citations}
    item_ids = {i.id for i in bundle.items}

    # The denied document must never appear -- this half already holds.
    assert "denied-high" not in citation_ids
    assert "denied-high" not in item_ids
    assert "denied-high" not in bundle.cache_key

    # DEFECT: the authorized document is ALSO absent -- it was crowded out of
    # the raw candidate pool by search_hybrid's own unfiltered trim before
    # ContextCompiler's policy step ever ran. A correct ACL-before-rank
    # pipeline (mirroring the fix already proven for search_hybrid itself)
    # would keep it.
    assert "permitted-low" in citation_ids, (
        "ContextCompiler crowd-out defect: an authorized low-score document "
        "was silently dropped because ContextCompiler._retrieve does not "
        "forward session= into search_hybrid, so the 389f5565 ACL-before-"
        "rank ordering never applies to this pipeline. See "
        "agent_utilities/knowledge_graph/retrieval/context_compiler.py "
        "_retrieve()'s search_hybrid(...) call."
    )


def test_context_compiler_is_safe_when_the_candidate_pool_is_generous_enough():
    """Sanity/contrast case: with a candidate pool larger than the raw
    candidate set (the common real-world default, ``candidate_pool=40``),
    ``search_hybrid``'s internal trim never has to choose between the denied
    and authorized candidates, so the defect above does not manifest and the
    authorized document reaches citations normally. This is not a substitute
    for the fix -- it only shows the defect is pool-size-dependent, not that
    it is absent.
    """
    brain = get_company_brain()
    brain.permissions.set_acl(
        NodeACL(node_id="permitted-low", classification=DataClassification.PUBLIC)
    )
    nodes = [
        {
            "id": "denied-high",
            "type": "Doc",
            "_score": 0.95,
            "status": "ACTIVE",
            "content": "denied content",
        },
        {
            "id": "permitted-low",
            "type": "Doc",
            "_score": 0.1,
            "status": "ACTIVE",
            "content": "permitted content",
        },
    ]
    session = _session()
    engine = _Engine(nodes)
    compiler = ContextCompiler(engine)

    with use_session(session):
        bundle = compiler.compile("q", session=session, top_k=10, candidate_pool=40)

    citation_ids = {c.node_id for c in bundle.citations}
    assert "denied-high" not in citation_ids
    assert "permitted-low" in citation_ids
