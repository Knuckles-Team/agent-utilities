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
