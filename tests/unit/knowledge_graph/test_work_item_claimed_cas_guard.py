"""BUG-11x regression: the engine's native WorkItem-authority guard
(``work_item_capability::validate_generic_method``, epistemic-graph)
unconditionally refuses ANY generic ``CompareAndSetNodeFields`` on an
already-claimed WorkItem row -- confirmed live against the real engine
(``RuntimeError: staged MutationBatch durable commit failed: native WorkItem
authority required for generic replacement``).

Before this fix, four call sites hit this unguarded: ``checkpoint_work_item``,
``request_work_item_input``, ``submit_work_item_input`` (all functionally
important -- checkpoint progress / human-input round-trip), and
``set_work_item_priority`` (non-critical scheduling metadata). The first
three now raise a clear, typed, actionable ``WorkItemBackendUnavailable``
instead of a bare transport ``RuntimeError``; the fourth degrades silently
(best-effort, no correctness impact), matching the existing
``_append_downstream``/``_reconcile_dependency_readiness`` pattern.

Requires a real engine (``EPISTEMIC_GRAPH_TEST_BINARY`` / the session-engine
fixture) -- see ``tests/conftest.py``'s hermetic-skip mechanism for a bare
environment with no isolated test engine.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin
from agent_utilities.orchestration import work_item as wi


class _Harness:
    """Real control-plane WorkItem surface, same shape as
    ``test_task_claim_cas.py``'s ``_ClaimHarness`` (both backend and control
    plane point at one real ``EpistemicGraphBackend`` -- no isolated
    ``__control__`` graph needed for these claim/CAS tests)."""

    _control = TaskManagerMixin._control
    _control_session_scope = TaskManagerMixin._control_session_scope
    _control_cypher = TaskManagerMixin._control_cypher
    _work_item_engine = TaskManagerMixin._work_item_engine

    def __init__(self, backend=None):
        self.backend = backend if backend is not None else EpistemicGraphBackend()
        self.control_backend = self.backend


def _claim(payload_ref: str, token: str):
    h = _Harness()
    engine = h._work_item_engine
    item_id = f"workitem:ingest_task:{payload_ref}"
    wi.submit_work_item(
        engine, kind="ingest_task", payload_ref=payload_ref, work_item_id=item_id
    )
    claim = wi.claim_specific(engine, item_id, token=token, lease_ttl_s=60.0)
    assert claim is not None, "setup: claim must win to reach the claimed state"
    wi.mark_running(engine, item_id, claim)
    return engine, item_id, claim


def test_checkpoint_work_item_raises_clear_typed_error_on_claimed_item():
    engine, item_id, claim = _claim("cp-guard-1", "cptoken:1:1")
    with pytest.raises(wi.WorkItemBackendUnavailable) as excinfo:
        wi.checkpoint_work_item(engine, item_id, claim, "checkpoint:guard-1")
    # The message must name the actual constraint, not just re-surface the
    # engine's opaque transport error verbatim.
    assert "checkpoint_work_item" in str(excinfo.value)
    assert "native WorkItem" in str(excinfo.value)


def test_request_work_item_input_raises_clear_typed_error_on_claimed_item():
    engine, item_id, claim = _claim("cp-guard-2", "cptoken:2:1")
    with pytest.raises(wi.WorkItemBackendUnavailable) as excinfo:
        wi.request_work_item_input(engine, item_id, claim, request={"prompt": "ok?"})
    assert "request_work_item_input" in str(excinfo.value)


def test_set_work_item_priority_degrades_to_false_on_claimed_item():
    """Unlike the three functionally-important sites above, priority is
    best-effort: it must return False (not raise) when the native authority
    refuses -- a missed priority bump is not a correctness regression."""
    engine, item_id, _claim_dict = _claim("cp-guard-3", "cptoken:3:1")
    result = wi.set_work_item_priority(engine, item_id, 3)
    assert result is False
