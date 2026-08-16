"""U-06 — ``_bound_to_explicit_ingest_graph`` is the worker-side half of the
explicit codebase/document-ingest graph selector (the submission-side half,
``submit_task``'s ``graph`` kwarg, is covered by
``test_submit_task_explicit_graph.py``). It re-narrows the ambient verified
session onto a WorkItem's persisted ``graph`` metadata for the duration of
one async content write, mirroring ``mcp.kg_server.bound_to_graph`` (R-02)
without importing the MCP dispatch layer into KG core.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.engine_tasks import (
    _bound_to_explicit_ingest_graph,
)
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    current_session,
    suspend_session,
    use_session,
)
from agent_utilities.orchestration.work_item import WorkItemBackendUnavailable
from agent_utilities.security.brain_context import ActorContext, ActorType


def _session(graph: str = "") -> GraphSession:
    actor = ActorContext(
        actor_id="worker-service",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id="test-tenant",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant="test-tenant",
        scopes=frozenset({"kg:read", "kg:write"}),
        graph=graph,
        policy_version="test-policy",
        audience="test-audience",
    )


def test_empty_graph_is_a_no_op_and_preserves_the_ambient_session():
    with use_session(_session(graph="worker-default")):
        with _bound_to_explicit_ingest_graph(""):
            assert current_session().graph == "worker-default"
        # Unchanged after exit too.
        assert current_session().graph == "worker-default"


def test_explicit_graph_narrows_the_worker_session_and_restores_after():
    with use_session(_session(graph="worker-default")):
        with _bound_to_explicit_ingest_graph("kf-pilot:code-ingest"):
            assert current_session().graph == "kf-pilot:code-ingest"
        # The worker's own ambient graph is restored once the write is done —
        # a later unrelated task on the same worker thread must not stay
        # pinned to a previous job's explicit graph.
        assert current_session().graph == "worker-default"


def test_KNOWN_BAD_no_ambient_session_fails_closed_never_silently_unscoped():
    """Negative proof: if a worker somehow ran this with no verified session
    at all, silently proceeding unscoped would let the content write fall
    through to whatever default the underlying transport picks -- exactly
    the kind of silent fallback this whole theme forbids. It must raise,
    not degrade to a no-op.
    """
    # The suite's own ambient test session is deliberately suspended here —
    # the point under test is what happens with genuinely NO verified
    # authority, mirroring an unauthenticated boundary.
    with suspend_session():
        assert current_session() is None
        with pytest.raises(WorkItemBackendUnavailable):
            with _bound_to_explicit_ingest_graph("kf-pilot:code-ingest"):
                raise AssertionError("must never enter the body without a session")
