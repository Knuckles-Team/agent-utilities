"""Crash-resume tests for the durable-execution substrate (PA-R0.1).

Covers :class:`DurableRun` — the ONE crash-safe checkpoint substrate wired into
the autonomous evolution / SDD loop — and the live ``run_evolution_cycle`` entry
point: an interrupted run must RESUME from the last completed step on
re-invocation (already-completed steps skipped + replayed, the interrupted step
re-run) rather than restarting or losing state.
"""

from __future__ import annotations

import pytest

from agent_utilities.orchestration.durable_execution import DurableRun


def test_durable_run_resumes_interrupted_step(tmp_path):
    """A step that crashes mid-run leaves the run resumable; completed steps skip."""
    db = tmp_path / "durable.db"
    calls = {"a": 0, "b": 0, "c": 0}

    def a():
        calls["a"] += 1
        return {"a": "done"}

    def b():
        calls["b"] += 1
        return {"b": "done"}

    def c_crash():
        calls["c"] += 1
        raise RuntimeError("boom mid-step c")

    # Attempt 1: a + b complete, c crashes (simulating a kill -9 mid-step).
    run1 = DurableRun("sess:x", db_path=db)
    assert run1.resumed is False
    run1.step("a", a)
    run1.step("b", b)
    with pytest.raises(RuntimeError, match="boom mid-step c"):
        run1.step("c", c_crash)

    # Attempt 2: a fresh DurableRun on the SAME session resumes the SAME run.
    run2 = DurableRun("sess:x", db_path=db)
    assert run2.resumed is True
    assert run2.run_id == run1.run_id
    assert run2.is_done("a") is True
    assert run2.is_done("c") is False

    replayed_a = run2.step("a", a)  # already done → replayed, NOT re-run
    run2.step("b", b)

    def c_ok():
        calls["c"] += 1
        return {"c": "done"}

    run2.step("c", c_ok)  # interrupted step re-runs and now succeeds
    run2.finish()

    assert replayed_a == {"a": "done"}
    assert calls["a"] == 1, "completed step must NOT re-run on resume"
    assert calls["b"] == 1, "completed step must NOT re-run on resume"
    assert calls["c"] == 2, "interrupted step re-runs on resume"


def test_durable_run_finish_starts_fresh(tmp_path):
    """After finish(), the next run under the same session is brand new."""
    db = tmp_path / "durable.db"
    run1 = DurableRun("sess:y", db_path=db)
    run1.step("only", lambda: "ok")
    run1.finish()

    run2 = DurableRun("sess:y", db_path=db)
    assert run2.resumed is False
    assert run2.run_id != run1.run_id


def test_run_evolution_cycle_resumes_after_crash(monkeypatch):
    """A ``run_evolution_cycle`` interrupted mid-run resumes on re-invocation.

    The variant-evolution stage completes, then the skill-gap stage crashes. On
    re-invocation the completed evolve stage is REPLAYED (its expensive
    tournament/prune work is NOT re-run) and only the interrupted skill-gap stage
    re-executes — the run resumes instead of restarting or losing state.
    """
    from agent_utilities.harness.agentic_evolution_engine import (
        AgenticEvolutionEngine,
    )

    eng = AgenticEvolutionEngine()
    eng._initialized = True  # skip _lazy_init; inject fakes below

    class _Pool:
        def population_health(self, base_id):
            return {"spread": 0.5, "collapsed": False}

    eng._variant_pool = _Pool()  # truthy → evolve stage runs
    eng._memory_store = None
    eng._decentralized_memory = None
    eng._replay_buffer = None
    eng._self_play = None
    eng._fast_slow = None
    eng._skill_detector = object()  # truthy → skill-gap stage runs

    counts = {"tournament": 0, "skill_gap": 0}

    def _tournament(base_id, top_k=3):
        counts["tournament"] += 1
        return ["variant_a"]

    monkeypatch.setattr(eng, "tournament_select", _tournament)
    monkeypatch.setattr(eng, "prune_losers", lambda base_id, keep=3: 0)

    class _Gap:
        closest_skill = "deploy"
        similarity_score = 0.2
        suggested_name = "deploy_pods"

    def _detect(task_text):
        counts["skill_gap"] += 1
        if counts["skill_gap"] == 1:
            raise RuntimeError("crash mid skill-gap stage")
        return _Gap()

    monkeypatch.setattr(eng, "detect_skill_gap", _detect)

    # Attempt 1: evolve completes, skill-gap crashes (propagates out — a crash).
    with pytest.raises(RuntimeError, match="crash mid skill-gap"):
        eng.run_evolution_cycle("base_x", task_text="deploy k8s")
    assert counts["tournament"] == 1
    assert counts["skill_gap"] == 1

    # Attempt 2: resumes — evolve replayed (tournament NOT re-run), skill-gap re-runs.
    report = eng.run_evolution_cycle("base_x", task_text="deploy k8s")

    assert counts["tournament"] == 1, "completed evolve stage must NOT re-run on resume"
    assert counts["skill_gap"] == 2, "interrupted skill-gap stage re-runs on resume"
    assert report["winners"] == ["variant_a"], "replayed from the interrupted run"
    assert report["skill_gap"]["closest_skill"] == "deploy"
