"""ARPO step-credit read-back from the run lifecycle (CONCEPT:AHE-3.15).

``write_back_step_credit`` existed but was never invoked from the live step
path — routing only ever learned from final answers. ``agent_runner`` now
credits every run's intermediate agent-steps into the capability reward-EMA on
completion (success AND failure), guarded so credit failures never break runs.

@pytest.mark.concept("AHE-3.15")
"""

from __future__ import annotations

import pytest

from agent_utilities.orchestration.agent_runner import (
    _extract_steps,
    _write_step_credit,
)

pytestmark = pytest.mark.concept("AHE-3.15")


class _CapIndex:
    def __init__(self):
        self.outcomes: list[tuple[str, float]] = []

    def record_outcome(self, id, success=None, reward=None, alpha=0.3):
        self.outcomes.append((id, reward))
        return reward


class _KG:
    def __init__(self, retrieval):
        self.retrieval = retrieval


class _Engine:
    def __init__(self, retrieval=None):
        self.knowledge_graph = _KG(retrieval) if retrieval is not None else None


class TestExtractSteps:
    def test_per_node_results_become_steps(self):
        result = {
            "results": {
                "researcher": "found 3 sources",
                "synthesizer": "",
                "output": "final answer",
            }
        }
        steps, ids = _extract_steps(result, "team_lead", success=True)
        assert ids == ["researcher", "synthesizer"]
        assert steps[0]["success"] is True
        assert steps[1]["success"] is False  # empty output = unsuccessful step

    def test_no_structure_collapses_to_single_step(self):
        steps, ids = _extract_steps("plain string", "solo_agent", success=False)
        assert ids == ["solo_agent"]
        assert steps == [{"action": "solo_agent", "success": False}]


class TestWriteStepCredit:
    def test_credits_each_agent_step(self):
        cap = _CapIndex()
        engine = _Engine(retrieval=cap)
        result = {"results": {"researcher": "ok", "verifier": "", "output": "x"}}
        written = _write_step_credit(engine, "run-1", "lead", result, success=True)
        assert written == 2
        rewards = dict(cap.outcomes)
        # the successful step earns strictly more credit than the failed one
        assert rewards["researcher"] > rewards["verifier"]
        assert all(0.0 < r < 1.0 for r in rewards.values())

    def test_failed_run_still_writes_credit(self):
        cap = _CapIndex()
        engine = _Engine(retrieval=cap)
        written = _write_step_credit(engine, "run-2", "solo", None, success=False)
        assert written == 1
        # single failed step on a failed trajectory lands below neutral
        assert cap.outcomes[0][1] < 0.5

    def test_successful_single_step_lands_above_neutral(self):
        cap = _CapIndex()
        engine = _Engine(retrieval=cap)
        _write_step_credit(engine, "run-3", "solo", "answer", success=True)
        assert cap.outcomes[0][1] > 0.5

    def test_no_capability_index_is_noop(self):
        assert _write_step_credit(_Engine(retrieval=None), "r", "a", None, True) == 0
        assert _write_step_credit(None, "r", "a", None, True) == 0

    def test_credit_failure_never_raises(self):
        class _Broken:
            def record_outcome(self, *a, **k):
                raise RuntimeError("boom")

        engine = _Engine(retrieval=_Broken())
        assert _write_step_credit(engine, "r", "a", None, True) == 0
