"""BUG-111 regression: the engine's native WorkItem-authority guard
(``work_item_capability::validate_generic_method``, epistemic-graph)
unconditionally refuses ANY generic ``CompareAndSetNodeFields`` on an
already-claimed WorkItem row (``RuntimeError: staged MutationBatch durable
commit failed: native WorkItem authority required for generic replacement``).

Four call sites used to hit this unguarded: ``checkpoint_work_item``,
``request_work_item_input``, ``submit_work_item_input`` (all functionally
important -- checkpoint progress / human-input round-trip), and
``set_work_item_priority`` (non-critical scheduling metadata). BUG-111 closes
the capability gap with a native, typed RPC (``Method::CasWorkItemMetadata``)
that performs an atomic CAS on a WorkItem's non-authority scheduling metadata
(checkpoint_id/metadata/prio_bucket) inside the SAME durable WorkItem
transaction ``ClaimWorkItem``/``RenewWorkItemLease`` use -- so all four sites
now SUCCEED against an already-claimed item instead of hitting the generic
-CAS authority guard. ``work_item.py``'s ``_cas_work_item_metadata`` still
raises :class:`~agent_utilities.orchestration.work_item.WorkItemBackendUnavailable`
for a genuine capability gap (missing RPC) or a malformed engine response --
never for an ordinary claimed-item CAS anymore.

Requires a real engine (the session-engine fixture) -- see
``tests/conftest.py``'s hermetic-skip mechanism for a bare environment with
no isolated test engine.
"""

from __future__ import annotations

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


def test_checkpoint_work_item_succeeds_on_claimed_item_via_native_rpc():
    engine, item_id, claim = _claim("cp-guard-1", "cptoken:1:1")
    assert wi.checkpoint_work_item(engine, item_id, claim, "checkpoint:guard-1") is True
    item = wi.get_work_item(engine, item_id)
    assert item is not None
    assert item.get("checkpoint_id") == "checkpoint:guard-1"


def test_request_work_item_input_succeeds_on_claimed_item_via_native_rpc():
    engine, item_id, claim = _claim("cp-guard-2", "cptoken:2:1")
    assert (
        wi.request_work_item_input(engine, item_id, claim, request={"prompt": "ok?"})
        is True
    )
    item = wi.get_work_item(engine, item_id)
    assert item is not None
    assert (item.get("metadata") or {}).get("pending_input_request") == {
        "prompt": "ok?"
    }


def test_set_work_item_priority_succeeds_on_claimed_item_via_native_rpc():
    """Priority is routed through the same native CasWorkItemMetadata RPC as
    the three functionally-important sites above -- it no longer needs to
    degrade silently on a claimed item, since the RPC never hits the
    generic-CAS authority guard in the first place."""
    engine, item_id, _claim_dict = _claim("cp-guard-3", "cptoken:3:1")
    assert wi.set_work_item_priority(engine, item_id, 3) is True
    item = wi.get_work_item(engine, item_id)
    assert item is not None
    assert item.get("prio_bucket") == 3
