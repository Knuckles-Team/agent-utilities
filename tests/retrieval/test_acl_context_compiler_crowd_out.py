#!/usr/bin/python
"""NE-050 — ContextCompiler ACL-before-rank at the retrieval boundary.

``ContextCompiler._retrieve`` (agent_utilities/knowledge_graph/retrieval/
context_compiler.py) calls ``engine.search_hybrid(...)`` / a bare
retriever's ``retrieve_hybrid(...)`` WITHOUT forwarding the verified
``session`` it already resolved. Both of those methods already implement
ACL-before-rank enforcement (``QueryMixin._enforce_acl_on_results`` /
``HybridRetriever.retrieve_hybrid``'s own ``session`` docstring), but only
when a session is actually supplied — ``session=None`` is a documented
no-op kept for pre-existing unfiltered callers. Because ``_retrieve`` never
passed one through, a denied high-score candidate could consume a
``candidate_pool`` slot and crowd an authorized low-score candidate out of
``candidates`` entirely, BEFORE ``compile()``'s own ``enforce()`` policy
pass ever ran over the (already-trimmed) pool.

This suite is an independent proof of the fix (not a copy of the
acceptance-track's ``tests/retrieval/test_adopt_a_context_compiler_acl_
crowd_out.py`` on ``ne/adopt-au-acceptance-a``, which this track does not
own and must not modify) plus two requirements that test does not cover:
cache-identity safety and the ACL-infrastructure-failure contract.
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
    every call regardless of whether a marking was ever applied — every
    test here needs one installed (same fixture shape every other
    context-compiler test file uses)."""

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
    """Duck-typed ``HybridRetriever`` stand-in returning a fixed, RAW-scored
    candidate pool — the REAL ``QueryMixin.search_hybrid`` (not a
    hand-written per-test fake) performs ACL enforcement + score-gate trim
    under test, exactly like the engine's own served path does."""

    def __init__(self, nodes: list[dict]) -> None:
        self._nodes = nodes

    def retrieve_hybrid(self, query, **kwargs):  # noqa: ARG002
        return list(self._nodes)


class _Engine(QueryMixin):
    """A real ``QueryMixin.search_hybrid`` bound to a stub retriever — this is
    the branch of ``ContextCompiler._retrieve`` the missing ``session=``
    forward broke (no explicit ``hybrid_retriever=`` override)."""

    def __init__(self, nodes: list[dict]) -> None:
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


def _session(**kw) -> GraphSession:
    actor = kw.pop("actor", None) or _actor()
    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:read"}),
        graph="test-graph",
        policy_version=kw.pop("policy_version", "policy:test"),
        audience="test-audience",
        **kw,
    )


def _grant_public(*node_ids: str) -> None:
    permissions = get_company_brain().permissions
    for node_id in node_ids:
        permissions.set_acl(
            NodeACL(node_id=node_id, classification=DataClassification.PUBLIC)
        )


def _grant_to_actor_only(node_id: str, actor_id: str) -> None:
    """Grant read access to exactly ``actor_id`` (CONFIDENTIAL + explicit
    ``read_actors``, never PUBLIC) — a different authenticated actor is
    denied by ``check_permission`` (agent_utilities/knowledge_graph/core/
    company_brain.py), unlike PUBLIC which is visible to everyone."""
    get_company_brain().permissions.set_acl(
        NodeACL(
            node_id=node_id,
            classification=DataClassification.CONFIDENTIAL,
            read_actors=(actor_id,),
        )
    )


_DENIED_HIGH = {
    "id": "denied-high",
    "type": "Doc",
    "_score": 0.95,
    "status": "ACTIVE",
    "content": "content the actor may not see",
}
_PERMITTED_LOW = {
    "id": "permitted-low",
    "type": "Doc",
    "_score": 0.05,
    "status": "ACTIVE",
    "content": "content the actor may see",
}


class FakeKVBackend:
    """Minimal in-memory ``get``/``put`` duck-typed KV backend."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self.store.get(key)

    def put(self, key: str, value: bytes) -> bool:
        self.store[key] = value
        return True


def test_denied_high_score_document_does_not_crowd_out_authorized_low_score_document():
    """The core NE-050 assertion: with a candidate_pool sized to exactly the
    two-candidate pool, a denied high-score document must not consume the
    slot an authorized low-score document needed to survive into the bundle.
    """
    _grant_public("permitted-low")
    session = _session()
    engine = _Engine([_DENIED_HIGH, _PERMITTED_LOW])
    compiler = ContextCompiler(engine)

    with use_session(session):
        bundle = compiler.compile("q", session=session, top_k=1, candidate_pool=1)

    citation_ids = {c.node_id for c in bundle.citations}
    item_ids = {i.id for i in bundle.items}

    assert "denied-high" not in citation_ids
    assert "denied-high" not in item_ids
    assert "permitted-low" in citation_ids, (
        "authorized low-score document was crowded out of the raw candidate "
        "pool before compile()'s own enforce() step ever saw it — ACL is not "
        "running before rank/trim in ContextCompiler._retrieve"
    )
    assert "permitted-low" in item_ids


def test_denied_document_is_absent_from_cache_key_and_stored_cache_payload():
    """A denied document must not influence cache identity in EITHER
    direction: not present in the derived ``cache_key`` string, and not
    present in the bytes actually persisted under it (content, id, or
    citation)."""
    _grant_public("permitted-low")
    session = _session()
    engine = _Engine([_DENIED_HIGH, _PERMITTED_LOW])
    compiler = ContextCompiler(engine)
    kv = FakeKVBackend()

    with use_session(session):
        bundle = compiler.compile(
            "cache identity probe query, no relation to node content",
            session=session,
            top_k=1,
            candidate_pool=1,
            kv_backend=kv,
        )

    assert "denied-high" not in bundle.cache_key
    assert bundle.cache_key, "expected a non-empty cache key to have been computed"
    assert kv.store, "expected the assembled bundle to have been stored"

    stored_bytes = kv.store[bundle.cache_key]
    stored_text = stored_bytes.decode("utf-8")
    assert "denied-high" not in stored_text
    assert "content the actor may not see" not in stored_text


def test_second_denied_candidate_pool_never_reuses_a_first_principals_cache_entry():
    """Two principals with DIFFERENT ACL grants over the SAME raw candidate
    pool must never share a cache entry — the post-policy evidence-id set
    (which the cache key is derived from) differs between them, so a
    denied-for-principal-B document that was cached as part of principal A's
    (authorized) bundle must never be served back to B."""
    _grant_to_actor_only("permitted-low", "principal-a")  # B is NOT granted
    engine_a = _Engine([_DENIED_HIGH, _PERMITTED_LOW])
    engine_b = _Engine([_DENIED_HIGH, _PERMITTED_LOW])
    kv = FakeKVBackend()
    query = "cross-principal cache reuse probe, unrelated to node content"

    session_a = _session(actor=_actor("principal-a"))
    with use_session(session_a):
        bundle_a = ContextCompiler(engine_a).compile(
            query, session=session_a, top_k=2, candidate_pool=2, kv_backend=kv
        )

    session_b = _session(actor=_actor("principal-b"))
    with use_session(session_b):
        bundle_b = ContextCompiler(engine_b).compile(
            query, session=session_b, top_k=2, candidate_pool=2, kv_backend=kv
        )

    assert bundle_a.cache_key != bundle_b.cache_key
    assert {i.id for i in bundle_a.items} == {"permitted-low"}
    assert {i.id for i in bundle_b.items} == set()
    assert not bundle_b.kv_cache_hit


def test_acl_infrastructure_failure_raises_rather_than_returning_unfiltered_results(
    monkeypatch,
):
    """An enforcement infrastructure failure (not a denial — a defect in the
    ACL machinery itself) must raise, never silently degrade to the
    unfiltered candidate set. Mirrors the contract ``search_hybrid``/
    ``query_cypher`` already document and enforce."""
    from agent_utilities.knowledge_graph.core import secured_reads

    def _boom(_rows, _actor):
        raise PermissionError("Row visibility evaluation failed") from ValueError(
            "synthetic infrastructure failure"
        )

    monkeypatch.setattr(secured_reads, "filter_rows", lambda rows, _actor: rows)
    monkeypatch.setattr(secured_reads, "visible", _boom)

    _grant_public("permitted-low")
    session = _session()
    engine = _Engine([_DENIED_HIGH, _PERMITTED_LOW])
    compiler = ContextCompiler(engine)

    with use_session(session), pytest.raises(PermissionError):
        compiler.compile("q", session=session, top_k=2, candidate_pool=2)
