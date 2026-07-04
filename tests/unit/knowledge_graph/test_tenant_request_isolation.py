"""Request-path tenant isolation through query_cypher (CONCEPT:AU-KG.compute.data-is-private-its).

Proves the MCP/orchestration read chokepoint (IntelligenceGraphEngine.query_cypher)
applies tenant scope() + owner/scope visibility on a shared backend graph, and
that guarded writes stamp tenant_id so the predicate matches. Enforcement is on
only inside these tests (the guarded backend + KG_BRAIN_ENFORCE).
"""

from __future__ import annotations

import uuid

import pytest

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


@pytest.fixture
def guarded_engine(monkeypatch):
    monkeypatch.setenv("KG_BRAIN_ENFORCE", "1")
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

    reset_company_brain()
    guarded = BrainGuardedBackend(EpistemicGraphBackend(), get_company_brain())
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
        guarded.add_node(nid, type="Doc", title="acme doc")
    props = guarded.inner._graph._get_node_properties(nid)
    assert props.get("tenant_id") == ORG_A
    assert props.get("_owner_id") == "alice"
    assert props.get("_shared_scope") == "private"


def test_query_cypher_isolates_tenants(guarded_engine):
    engine, guarded = guarded_engine
    a_id, g_id = f"doc:a-{_RUN}", f"doc:g-{_RUN}"
    with use_actor(_actor(ORG_A, "alice")):
        guarded.add_node(a_id, type="Doc", title="acme")
    with use_actor(_actor(ORG_B, "bob")):
        guarded.add_node(g_id, type="Doc", title="globex")

    # acme sees only its own node; globex sees only its own (cross-org isolation).
    with use_actor(_actor(ORG_A, "alice")):
        acme_ids = _ids(engine.query_cypher("MATCH (n:Doc) RETURN n.id AS id"))
    with use_actor(_actor(ORG_B, "bob")):
        globex_ids = _ids(engine.query_cypher("MATCH (n:Doc) RETURN n.id AS id"))

    assert a_id in acme_ids and g_id not in acme_ids
    assert g_id in globex_ids and a_id not in globex_ids


def test_within_org_private_then_shared(guarded_engine):
    engine, guarded = guarded_engine
    org = f"team-{_RUN}"
    alice_id, bob_id = f"doc:alice-{_RUN}", f"doc:bob-{_RUN}"
    with use_actor(_actor(org, "alice")):
        guarded.add_node(alice_id, type="Doc", title="alice private")
    with use_actor(_actor(org, "bob")):
        guarded.add_node(bob_id, type="Doc", title="bob private")

    # Project owner/scope so the visibility filter (KG-2.60) can apply.
    q = "MATCH (n:Doc) RETURN n.id AS id, n._owner_id AS _owner_id, n._shared_scope AS _shared_scope"
    with use_actor(_actor(org, "bob")):
        before = _ids(engine.query_cypher(q))
    assert bob_id in before and alice_id not in before  # alice's is private

    # alice shares her node with the org → bob now sees it.
    from agent_utilities.knowledge_graph.core import tenant_sharing as ts

    ts.share_with_org(alice_id, store=guarded)
    with use_actor(_actor(org, "bob")):
        after = _ids(engine.query_cypher(q))
    assert alice_id in after  # now org-shared
