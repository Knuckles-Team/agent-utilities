"""NE-037 acceptance — explicit graph selection across async work (`92bb8578`),
the "local and redirected sessions" half of the gate.

``plans/graph-os-completion-program/archive/bug-analysis.md`` (R-02, the
predecessor invariant this mechanism extends) distinguishes two session
shapes a graph-selection narrowing helper must handle identically:

* a **local** session -- ``endpoint``/``placement_group``/``catalog_epoch``
  all unset (``None``), the ordinary in-process case where the caller never
  went through the engine's placement/routing catalog; and
* a **redirected** session -- one that already carries server-resolved
  placement metadata via :meth:`GraphSession.with_route` (the shape a
  request takes after the engine's placement catalog has routed it to a
  specific Raft group/shard/catalog epoch, CONCEPT:AU-KG.sharding.
  tenant-partitioned-sharding-hrw), i.e. "local or placement-redirected
  execution" from the archived bug-analysis' own wording.

:func:`_bound_to_explicit_ingest_graph` (the worker-side re-narrowing half of
U-06/GOC-67 this test's sibling ``test_bound_to_explicit_ingest_graph.py``
covers for the plain/local session only) must narrow onto the WorkItem's
persisted explicit graph and restore the caller's ORIGINAL session --
including its full placement/routing identity, not just its ``graph`` field
-- in both shapes.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.engine_tasks import (
    _bound_to_explicit_ingest_graph,
)
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    current_session,
    use_session,
)
from agent_utilities.security.brain_context import ActorContext, ActorType

_REDIRECTED_ENDPOINT = "shard-7.example.invalid:7443"


def _actor() -> ActorContext:
    return ActorContext(
        actor_id="worker-service",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id="test-tenant",
        authenticated=True,
    )


def _local_session(graph: str = "worker-default") -> GraphSession:
    """No endpoint/placement_group/catalog_epoch -- ordinary in-process case."""
    return GraphSession(
        actor=_actor(),
        tenant="test-tenant",
        scopes=frozenset({"kg:read", "kg:write"}),
        graph=graph,
        policy_version="test-policy",
        audience="test-audience",
    )


def _redirected_session(graph: str = "worker-default") -> GraphSession:
    """Carries server-resolved placement metadata via ``with_route`` -- the
    shape a request takes once the engine's placement catalog has already
    redirected it to a specific Raft group/shard/catalog epoch.
    """
    return _local_session(graph).with_route(
        endpoint=_REDIRECTED_ENDPOINT,
        placement_group=7,
        catalog_epoch=42,
    )


def test_local_session_narrows_and_fully_restores():
    session = _local_session()
    with use_session(session):
        with _bound_to_explicit_ingest_graph("kf-pilot:code-ingest"):
            narrowed = current_session()
            assert narrowed.graph == "kf-pilot:code-ingest"
            # Narrowing changes ONLY the graph; identity/tenant/scopes/policy/
            # placement (here: unset) are carried through unchanged.
            assert narrowed.actor is session.actor
            assert narrowed.tenant == session.tenant
            assert narrowed.scopes == session.scopes
            assert narrowed.endpoint is None
            assert narrowed.placement_group is None
        restored = current_session()
        assert restored.graph == "worker-default"
        assert restored.endpoint is None
        assert restored.placement_group is None
        assert restored == session


def test_redirected_session_narrows_and_fully_restores_placement():
    session = _redirected_session()
    assert session.endpoint == _REDIRECTED_ENDPOINT
    with use_session(session):
        with _bound_to_explicit_ingest_graph("kf-pilot:code-ingest"):
            narrowed = current_session()
            assert narrowed.graph == "kf-pilot:code-ingest"
            # The redirect (placement) identity must survive narrowing --
            # `with_graph` only replaces `graph`, never the routing fields.
            assert narrowed.endpoint == session.endpoint
            assert narrowed.placement_group == session.placement_group
            assert narrowed.catalog_epoch == session.catalog_epoch
            assert narrowed.actor is session.actor
            assert narrowed.tenant == session.tenant
        restored = current_session()
        # The caller's FULL redirected identity is restored -- not just the
        # graph name, but the placement routing it carried before narrowing.
        assert restored.graph == "worker-default"
        assert restored.endpoint == _REDIRECTED_ENDPOINT
        assert restored.placement_group == 7
        assert restored.catalog_epoch == 42
        assert restored == session


def test_local_and_redirected_sessions_narrow_to_the_same_graph_identically():
    """Both session shapes must behave IDENTICALLY on the one dimension that
    matters to this mechanism (which graph the narrowed session names) --
    the redirect/placement metadata is orthogonal and must never change
    which graph gets selected or how the restore is performed.
    """
    for session in (_local_session(), _redirected_session()):
        with use_session(session):
            with _bound_to_explicit_ingest_graph("kf-pilot:code-ingest"):
                assert current_session().graph == "kf-pilot:code-ingest"
            assert current_session().graph == "worker-default"
