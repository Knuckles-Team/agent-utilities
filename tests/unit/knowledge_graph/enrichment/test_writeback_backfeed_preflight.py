"""``run_writeback``'s backfeed preflight wiring (CA-22, DEC-CA-07/P11).

Every write path -- both the 31 ``_DELTA_HANDLERS`` entries
(``source_sync._apply_with_preflight``) and this outbound ``run_writeback`` path --
calls ``evaluate_backfeed_preflight`` before committing. These tests cover the
opt-in signals (``ops["_sync_conflict"]``, ``expected_source_version``/
``current_source_version``, a sink's own ``backfeed`` attribute) and confirm the
additive property: a caller supplying NONE of them (every existing caller) sees
byte-identical behavior to pre-CA-22.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.writeback.core import (
    WritebackContext,
    WritebackResult,
    register_sink,
    run_writeback,
)
from agent_utilities.knowledge_graph.ontology.sync_conflict import (
    BackfeedCapabilitySpec,
    SyncConflict,
)


class _CapturingSink:
    domain = "backfeedtarget"
    enable_flag = "BACKFEEDTARGET_ENABLE_WRITE"

    def __init__(self, *, backfeed: BackfeedCapabilitySpec | None = None) -> None:
        self.run_calls = 0
        if backfeed is not None:
            self.backfeed = backfeed

    def run(self, ctx: WritebackContext, ops, *, dry_run: bool) -> WritebackResult:
        self.run_calls += 1
        return WritebackResult(target=self.domain, created=1)


def test_run_writeback_with_no_signal_is_a_noop_byte_identical_to_pre_ca22():
    sink = _CapturingSink()
    register_sink(sink)
    out = run_writeback("backfeedtarget", dry_run=True)
    assert out["status"] == "completed"
    assert sink.run_calls == 1


def test_run_writeback_stale_expected_version_is_refused_before_the_sink_runs():
    sink = _CapturingSink()
    register_sink(sink)
    out = run_writeback(
        "backfeedtarget",
        dry_run=True,
        expected_source_version="v1",
        current_source_version="v2",
    )
    assert out["status"] == "refused"
    assert "stale_expected_version" in out["reason"]
    assert sink.run_calls == 0


def test_run_writeback_matching_expected_version_proceeds():
    sink = _CapturingSink()
    register_sink(sink)
    out = run_writeback(
        "backfeedtarget",
        dry_run=True,
        expected_source_version="v2",
        current_source_version="v2",
    )
    assert out["status"] == "completed"
    assert sink.run_calls == 1


def test_run_writeback_conflict_with_no_sink_backfeed_is_refused_undeclared_capability():
    sink = _CapturingSink()
    register_sink(sink)
    conflict = SyncConflict(
        connector="backfeedtarget",
        node_id="n1",
        field="status",
        source_value="new",
        graph_value="old",
        policy="manual_review",
    )
    out = run_writeback(
        "backfeedtarget", dry_run=True, node_id="n1", _sync_conflict=conflict
    )
    assert out["status"] == "refused"
    assert "undeclared_capability" in out["reason"]
    assert sink.run_calls == 0


def test_run_writeback_conflict_with_enabled_sink_backfeed_queues_a_proposal_never_calls_sink():
    sink = _CapturingSink(
        backfeed=BackfeedCapabilitySpec(
            capabilities=["backfeedtarget.record.update"], approval_class="change"
        )
    )
    register_sink(sink)
    conflict = SyncConflict(
        connector="backfeedtarget",
        node_id="n1",
        field="status",
        source_value="new",
        graph_value="old",
        policy="manual_review",
    )
    out = run_writeback(
        "backfeedtarget", dry_run=True, node_id="n1", _sync_conflict=conflict
    )
    assert out["status"] == "queued"
    assert out["proposal"]["field"] == "status"
    assert out["proposal"]["approval_class"] == "change"
    assert sink.run_calls == 0  # never a live write -- the proposal IS the effect
