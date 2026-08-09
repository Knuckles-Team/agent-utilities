"""High-stakes write-back: risk-tier + approval queue (CONCEPT:AU-KG.ingest.enterprise-source-extractor / KG-2.247).

The approval queue is engine-only (``:WritebackProposal`` nodes, no JSON fallback),
so these run against the REAL ephemeral engine the conftest provides (CONCEPT:
KG-2.238) — ``ProposalQueue()`` / ``run_writeback`` resolve the engine authority
via the OS-5.63 resolver, which is the session ``tiny_engine`` here.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.enrichment.writeback import (
    core,
    run_writeback,
)
from agent_utilities.knowledge_graph.enrichment.writeback.approval import (
    ProposalQueue,
    approve_proposal,
)


class _HighStakesSink:
    domain = "tesths"
    enable_flag = "TESTHS_ENABLE_WRITE"
    risk_tier = "high_stakes"

    def __init__(self):
        self.live_calls = 0

    def run(self, ctx, ops, *, dry_run):
        res = core.WritebackResult(target=self.domain)
        if dry_run:
            res.proposals.append({"op": "do_thing", "what": ops.get("what")})
        else:
            self.live_calls += 1
            res.created += 1
        return res


@pytest.fixture
def sink(monkeypatch, tiny_engine):
    # ``tiny_engine`` (CONCEPT:AU-KG.memory.provides-real-ephemeral-one) ensures the REAL ephemeral engine is up so
    # the engine-only ProposalQueue (no JSON fallback) has an authority to persist
    # :WritebackProposal nodes on.
    #
    # ProposalQueue(backend=None) -- the production path run_writeback() uses --
    # resolves its authority via require_engine_authority_backend(), which falls
    # back to a bare EpistemicGraphBackend() when no active backend is set. That
    # bare construction resolves its OWN routing graph via
    # resolve_routing_graph(None) *before* asking GraphComputeEngine for one, so
    # under the isolate_graph_compute_engine fixture (a tenant-bearing ambient
    # actor) it lands on a tenant graph this test never provisioned -- the same
    # isolation bug root-caused for test_kg_native_orchestration.py. Explicitly
    # setting the active backend to one bound to an already-isolated
    # GraphComputeEngine (constructed the same way the isolate fixture's own
    # engines are) makes require_engine_authority_backend() reuse it instead of
    # falling back to the divergent bare construction.
    from agent_utilities.knowledge_graph.backends import (
        get_active_backend,
        set_active_backend,
    )
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    compute = GraphComputeEngine(backend_type="rust")
    isolated_backend = object.__new__(EpistemicGraphBackend)
    isolated_backend._graph = compute
    isolated_backend.graph_name = compute.graph_name
    isolated_backend.create_schema()
    previous_backend = get_active_backend()
    set_active_backend(isolated_backend)

    s = _HighStakesSink()
    core.register_sink(s)
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)  # enabled
    yield s
    core._SINKS.pop("tesths", None)
    set_active_backend(previous_backend)


def test_high_stakes_live_request_is_queued_not_executed(sink):
    out = run_writeback("tesths", what="trade", dry_run=False)
    assert out["status"] == "queued"
    assert out["proposal_id"].startswith("wbp:tesths:")
    assert sink.live_calls == 0  # NEVER auto-executed
    assert out["proposals"][0]["op"] == "do_thing"
    pending = ProposalQueue().list(status="pending")
    assert len(pending) == 1


def test_approval_executes_the_queued_proposal(sink):
    out = run_writeback("tesths", what="trade", dry_run=False)
    pid = out["proposal_id"]
    assert sink.live_calls == 0

    approved = approve_proposal(pid)
    assert approved["status"] == "completed"
    assert approved["created"] == 1
    assert sink.live_calls == 1  # executed only after approval
    assert ProposalQueue().get(pid)["status"] == "approved"


def test_high_stakes_dry_run_previews_without_queueing(sink):
    out = run_writeback("tesths", what="trade", dry_run=True)
    assert out["status"] == "completed"
    assert out["dry_run"] is True
    assert sink.live_calls == 0
    assert ProposalQueue().list(status="pending") == []


# ---------------------------------------------------------------------------
# BUG-059 — ProposalQueue writes are ROUTED through stamp_ownership/
# stamp_classification, directly against a fake engine-authority backend (no
# live engine needed — isolates the governance behaviour from the engine
# resolution machinery the fixture above works around).
# ---------------------------------------------------------------------------


class _FakeApprovalBackend:
    """Minimal engine-authority-shaped backend for ProposalQueue."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}

    def execute(self, *_a, **_kw):  # pragma: no cover - only needs to exist
        return []

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props

    def get_node_properties(self, node_id):
        return self.nodes.get(node_id)

    def nodes_by_label(self, _label):
        return list(self.nodes.items())


def test_proposal_queue_enqueue_requires_a_bound_actor():
    """Known-bad input: no actor bound anywhere. BEFORE BUG-059's fix this
    silently minted an unowned, unreadable-by-anyone-but-privileged
    WritebackProposal node for the highest-stakes writes in the module
    ("finance trades, legal filings, destructive infra"). AFTER, it raises."""
    import contextvars

    from agent_utilities.security.brain_context import IdentityRequiredError

    backend = _FakeApprovalBackend()
    queue = ProposalQueue(backend=backend)

    def isolated():
        with pytest.raises(IdentityRequiredError):
            queue.enqueue("tesths", {"what": "trade"}, [{"op": "do_thing"}])

    contextvars.Context().run(isolated)
    assert backend.nodes == {}


def test_proposal_queue_enqueue_stamps_ownership_when_actor_bound():
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext, use_actor

    backend = _FakeApprovalBackend()
    queue = ProposalQueue(backend=backend)
    actor = ActorContext(
        actor_id="user:approver",
        actor_type=ActorType.HUMAN,
        tenant_id="tenant-fin",
        authenticated=True,
    )
    with use_actor(actor):
        pid = queue.enqueue("tesths", {"what": "trade"}, [{"op": "do_thing"}])

    props = backend.nodes[pid]
    assert props["_owner_id"] == "user:approver"
    assert props["tenant_id"] == "tenant-fin"
    assert props["classification"] == "confidential"


def test_proposal_queue_mark_preserves_original_owner_not_the_approver():
    """BUG-059: mark() (approve/reject) must NOT reassign ownership to
    whichever actor happens to approve the proposal — stamp_ownership's
    setdefault semantics only fill in a missing stamp, and mark() explicitly
    carries the original governance fields forward so a re-stamp under the
    approver's identity can never overwrite them."""
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext, use_actor

    backend = _FakeApprovalBackend()
    queue = ProposalQueue(backend=backend)
    proposer = ActorContext(
        actor_id="user:proposer",
        actor_type=ActorType.HUMAN,
        tenant_id="tenant-fin",
        authenticated=True,
    )
    with use_actor(proposer):
        pid = queue.enqueue("tesths", {"what": "trade"}, [{"op": "do_thing"}])

    approver = ActorContext(
        actor_id="user:approver",
        actor_type=ActorType.HUMAN,
        tenant_id="tenant-fin",
        authenticated=True,
    )
    with use_actor(approver):
        queue.mark(pid, "approved")

    props = backend.nodes[pid]
    assert props["status"] == "approved"
    assert props["_owner_id"] == "user:proposer"  # NOT the approver
