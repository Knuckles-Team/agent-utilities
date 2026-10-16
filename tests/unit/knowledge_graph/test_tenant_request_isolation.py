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
from _test_engine import TEST_TENANT

from agent_utilities.knowledge_graph.core.session import current_session, use_session
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

# The engine is pinned to ONE tenant per process (EPISTEMIC_GRAPH_TENANT) and
# rejects every request whose signed context names another tenant
# (AUTH_TENANT_MISMATCH).  ORG_A is therefore the engine's own tenant; ORG_B is a
# foreign tenant whose rows can exist in the graph (stamped by the guarded write
# path) but whose reads the engine refuses.  The engine also admits only its
# registered principal (TEST_AGENT_ID), so every identity that READS through the
# engine is that principal; writes only stamp ownership and may name any actor.
# Node ids stay unique per run.
_RUN = uuid.uuid4().hex[:8]
ORG_A = TEST_TENANT
ORG_B = f"globex-{_RUN}"


def _actor(tid, aid):
    return ActorContext(
        actor_id=aid,
        actor_type=ActorType.HUMAN,
        roles=("test",),
        tenant_id=tid,
        authenticated=True,
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
def guarded_engine(monkeypatch, tiny_engine):
    # ``tiny_engine`` is depended on (its value unused) purely so this
    # fixture's real-engine need is visible in ``item.fixturenames`` --
    # ``tests/conftest.py``'s ``pytest_collection_modifyitems`` decides once
    # per xdist worker whether to pay for a real session engine by scanning
    # for exactly that fixture name (or ``engine_graph``). This file builds
    # its own ``GraphComputeEngine``/``BrainGuardedBackend`` directly instead
    # of using ``tiny_engine``'s return value, so without this explicit
    # dependency the worker-needs-engine classifier never saw it: run this
    # file alone (or grouped only with other non-engine tests/unit files)
    # and no session engine started, so ``GraphComputeEngine()`` below failed
    # to connect and the makereport hook turned that into a SKIP -- while the
    # exact same tests, grouped with an engine-requesting file, got a live
    # engine and genuinely FAILED. Same test, two different verdicts, purely
    # from incidental file/worker grouping. Depending on ``tiny_engine`` here
    # makes the real-engine need deterministic regardless of grouping.
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
