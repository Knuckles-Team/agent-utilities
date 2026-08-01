"""Closed-loop regression gate for failure remediations (CONCEPT:AU-AHE.harness.failure-evolution).

The failure-ingest tick builds a ``(spec) -> bool`` regression check and threads
it into the GovernedAutoMerger: a remediation auto-merges only when promoting it
does not coincide with a spiking failure, and every gap is recorded as a durable
eval regression case + reward nudge.

@pytest.mark.concept("AU-AHE.harness.failure-evolution")
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.adaptation.failure_analyzer import FailureAnalyzer
from agent_utilities.knowledge_graph.enrichment.orchestration import TeamSpec
from agent_utilities.knowledge_graph.research.auto_merge import (
    GovernedAutoMerger,
    MergePolicy,
)

pytestmark = pytest.mark.concept("AU-AHE.harness.failure-evolution")


class _Backend:
    """Trace backend whose current error count is controllable per workflow."""

    def __init__(self, current_errors):
        self._current = current_errors  # list of {"name": ...}

    async def get_error_observations(self, **k):
        return self._current


class _FakeFeedback:
    def __init__(self):
        self.recorded = []

    def record_correction(self, ctype, target_id, **kw):
        self.recorded.append((ctype, target_id, kw))


def _gaps():
    return [{"workflow": "loop", "signature": "sig1", "occurrences": 2}]


def _strong_team():
    return TeamSpec(
        name="Remediation Team",
        goal="Fix the loop timeout failure",
        lead="Lead",
        members=["Fixer", "Validator"],
        description="Complete remediation proposal.",
    )


class TestRegressionCheck:
    def test_passes_when_failure_cleared(self):
        analyzer = FailureAnalyzer(
            engine=None, trace_backend=_Backend([]), feedback=_FakeFeedback()
        )
        check = analyzer.make_regression_check(_gaps())
        assert check(_strong_team()) is True

    def test_holds_when_failure_spiking(self):
        spiking = [
            {"name": "loop"},
            {"name": "loop"},
            {"name": "loop"},
        ]  # 3 > baseline 2
        analyzer = FailureAnalyzer(
            engine=None, trace_backend=_Backend(spiking), feedback=_FakeFeedback()
        )
        check = analyzer.make_regression_check(_gaps())
        assert check(_strong_team()) is False

    def test_no_backend_allows_promotion(self):
        analyzer = FailureAnalyzer(engine=None, trace_backend=None, feedback=None)
        check = analyzer.make_regression_check(_gaps())
        assert check(_strong_team()) is True

    def test_records_eval_and_outcome_feedback(self):
        fb = _FakeFeedback()
        analyzer = FailureAnalyzer(engine=None, trace_backend=_Backend([]), feedback=fb)
        analyzer.make_regression_check(_gaps())(_strong_team())
        ctypes = {c[0] for c in fb.recorded}
        assert "eval" in ctypes
        assert "outcome" in ctypes


class TestMergerIntegration:
    def _merger(self, check):
        return GovernedAutoMerger(
            engine=None,
            policy=MergePolicy(enabled=True, require_governance_valid=False),
            regression_check=check,
            promoter=lambda spec: True,
        )

    def test_spiking_failure_blocks_auto_merge(self):
        spiking = [{"name": "loop"}, {"name": "loop"}, {"name": "loop"}]
        analyzer = FailureAnalyzer(
            engine=None, trace_backend=_Backend(spiking), feedback=_FakeFeedback()
        )
        ev = self._merger(analyzer.make_regression_check(_gaps())).consider(
            _strong_team()
        )
        assert ev.merged is False
        assert "regression detected" in ev.failures

    def test_cleared_failure_allows_auto_merge(self):
        analyzer = FailureAnalyzer(
            engine=None, trace_backend=_Backend([]), feedback=_FakeFeedback()
        )
        ev = self._merger(analyzer.make_regression_check(_gaps())).consider(
            _strong_team()
        )
        assert ev.merged is True
        assert ev.reason == "auto-merged"
