"""High-stakes write-back: risk-tier + approval queue (CONCEPT:KG-2.9)."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.enrichment.writeback import (
    approval,
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
def sink(monkeypatch, tmp_path):
    s = _HighStakesSink()
    core.register_sink(s)
    monkeypatch.setattr(approval, "_store_path", lambda: tmp_path / "proposals.json")
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)  # enabled
    yield s
    core._SINKS.pop("tesths", None)


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
