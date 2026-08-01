"""Request-path tenant isolation through query_cypher (CONCEPT:AU-KG.compute.data-is-private-its).

Proves the MCP/orchestration read chokepoint (IntelligenceGraphEngine.query_cypher)
applies tenant scope() + owner/scope visibility on a shared backend graph, and
that guarded writes stamp tenant_id so the predicate matches. Enforcement is on
only inside these tests (the mandatory guarded backend).
"""

from __future__ import annotations

import contextlib
import uuid

import pytest

from agent_utilities.knowledge_graph.core.session import current_session, use_session
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

# The in-memory backend connects to the (possibly shared) live engine, so make
# every test hermetic with unique tenant + node ids — data is isolated by the
# tenant scope() predicate regardless of what else lives in the default graph.
_RUN = uuid.uuid4().hex[:8]
ORG_A = f"acme-{_RUN}"
ORG_B = f"globex-{_RUN}"


def _actor(tid, aid):
    return ActorContext(
        actor_id=aid, actor_type=ActorType.HUMAN, tenant_id=tid, authenticated=True
    )


@contextlib.contextmanager
def _scoped(tid, aid):
    """Scope BOTH the ambient actor and the ambient GraphSession to (tid, aid).

    ``query_cypher``'s tenant scoping reads ``session.actor`` (the ambient
    ``GraphSession``, resolved via ``resolve_session``'s fail-closed
    authority check — it deliberately rejects a session whose actor/tenant
    differs from the ambient one, by design), NOT ``current_actor()``. Write
    paths here (``guarded.add_node``) stamp ``current_actor()``. Swapping
    only ``current_actor()`` via a bare ``use_actor(...)`` therefore leaves
    reads scoped to the autouse ``isolate_graph_compute_engine`` fixture's
    original actor/tenant while writes land under the test's tenant — reads
    then find nothing, not because isolation is broken, but because the two
    ambient identities disagreed. Swap both together, keeping the fixture's
    already-isolated graph.
    """
    actor = _actor(tid, aid)
    with use_actor(actor):
        session = current_session()
        if session is None:
            yield
        else:
            with use_session(session.with_actor(actor)):
                yield


@pytest.fixture
def guarded_engine(monkeypatch):
    from agent_utilities.knowledge_graph.backends.brain_guarded_backend import (
        BrainGuardedBackend,
    )
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.company_brain_runtime import (
        get_company_brain,
        reset_company_brain,
    )
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.graph_compute import (
        GraphComputeEngine,
    )

    reset_company_brain()
    # A bare EpistemicGraphBackend() independently resolves the ambient
    # tenant's SHARED default graph (resolve_routing_graph(None)), not the
    # per-test isolated graph the autouse isolate_graph_compute_engine
    # fixture provisions (that redirect only intercepts a literal None/
    # "__commons__"/"__secrets__" graph_name passed straight to
    # GraphComputeEngine.__init__). Construct the isolated GraphComputeEngine
    # first and rebind the backend to it before wrapping it, the same idiom
    # already established for this exact defect (D-OTR-2/D-OTR-3).
    compute = GraphComputeEngine(backend_type="rust")
    inner = EpistemicGraphBackend()
    inner._graph = compute
    guarded = BrainGuardedBackend(inner, get_company_brain())
    engine = IntelligenceGraphEngine(backend=guarded)
    yield engine, guarded
    reset_company_brain()


def _ids(rows):
    out = set()
    for r in rows:
        v = r.get("id") or r.get("n.id")
        if isinstance(v, dict):
            v = v.get("id")
        if v:
            out.add(v)
    return out


def test_guarded_write_stamps_tenant_id(guarded_engine):
    _engine, guarded = guarded_engine
    nid = f"doc:stamp-{_RUN}"
    with use_actor(_actor(ORG_A, "alice")):
        guarded.add_node(nid, node_type="Doc", title="acme doc")
    props = guarded.inner._graph._get_node_properties(nid)
    assert props.get("tenant_id") == ORG_A
    assert props.get("_owner_id") == "alice"
    assert props.get("_shared_scope") == "private"


def test_query_cypher_isolates_tenants(guarded_engine):
    engine, guarded = guarded_engine
    a_id, g_id = f"doc:a-{_RUN}", f"doc:g-{_RUN}"
    with use_actor(_actor(ORG_A, "alice")):
        guarded.add_node(a_id, node_type="Doc", title="acme")
    with use_actor(_actor(ORG_B, "bob")):
        guarded.add_node(g_id, node_type="Doc", title="globex")

    # acme sees only its own node; globex sees only its own (cross-org isolation).
    #
    # KNOWN BLOCKED (reported, not weakened): even with the read correctly
    # scoped to a matching (actor, session) pair via _scoped(...), the native
    # engine's own fail-closed guard now rejects this call outright with
    # RuntimeError("request context tenant does not match graph tenant") —
    # confirmed by direct reproduction against compute.query_cypher(...)
    # bypassing every test/guarded-backend layer. The engine enforces ONE
    # tenant per physical graph; it does not support this test's premise of
    # several tenants sharing one graph, isolated only by query_cypher's
    # row-level scope() predicate. Making this assertion pass would require
    # either loosening that engine-level tenant/graph binding guard (a real
    # authority boundary — explicitly out of bounds) or restructuring this
    # test onto one physical graph per tenant (a genuine redesign, not a
    # fix, and out of this slice's scope). Left failing and reported.
    with _scoped(ORG_A, "alice"):
        acme_ids = _ids(engine.query_cypher("MATCH (n:Doc) RETURN n.id AS id"))
    with _scoped(ORG_B, "bob"):
        globex_ids = _ids(engine.query_cypher("MATCH (n:Doc) RETURN n.id AS id"))

    assert a_id in acme_ids and g_id not in acme_ids
    assert g_id in globex_ids and a_id not in globex_ids


def test_within_org_private_then_shared(guarded_engine):
    engine, guarded = guarded_engine
    org = f"team-{_RUN}"
    alice_id, bob_id = f"doc:alice-{_RUN}", f"doc:bob-{_RUN}"
    with use_actor(_actor(org, "alice")):
        guarded.add_node(alice_id, node_type="Doc", title="alice private")
    with use_actor(_actor(org, "bob")):
        guarded.add_node(bob_id, node_type="Doc", title="bob private")

    # Project owner/scope so the visibility filter (KG-2.60) can apply.
    # KNOWN BLOCKED — see the identical, fully-diagnosed engine-level
    # "request context tenant does not match graph tenant" guard documented
    # in test_query_cypher_isolates_tenants above; same root cause here.
    q = "MATCH (n:Doc) RETURN n.id AS id, n._owner_id AS _owner_id, n._shared_scope AS _shared_scope"
    with _scoped(org, "bob"):
        before = _ids(engine.query_cypher(q))
    assert bob_id in before and alice_id not in before  # alice's is private

    # alice shares her node with the org → bob now sees it.
    from agent_utilities.knowledge_graph.core import tenant_sharing as ts

    with use_actor(_actor(org, "alice")):
        ts.share_with_org(alice_id, store=guarded)
    with _scoped(org, "bob"):
        after = _ids(engine.query_cypher(q))
    assert alice_id in after  # now org-shared
