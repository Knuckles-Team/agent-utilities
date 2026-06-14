#!/usr/bin/python
"""Tests for the superhuman-certification gate + adaptation benchmark (SAFE-1.6/1.7)."""

from __future__ import annotations

from agent_utilities.harness.adaptation_benchmark import AdaptationBenchmark
from agent_utilities.harness.sai_task import SpecializationTask, VerifierResult
from agent_utilities.harness.superhuman_gate import SuperhumanCertifier


# --------------------------------------------------------------------------- #
# SuperhumanCertifier (SAFE-1.6)
# --------------------------------------------------------------------------- #


def test_certifies_when_ci_clears_human_baseline():
    cert = SuperhumanCertifier(confidence=0.95)
    rewards = [0.9, 0.92, 0.88, 0.91, 0.93, 0.9, 0.89]
    res = cert.certify(rewards, human_baseline=0.7)
    assert res.certified is True
    assert res.ci_lower > 0.7


def test_refuses_when_below_baseline():
    cert = SuperhumanCertifier()
    rewards = [0.6, 0.62, 0.58, 0.61]
    res = cert.certify(rewards, human_baseline=0.7)
    assert res.certified is False
    assert "does not clear" in res.reason


def test_refuses_without_human_baseline():
    cert = SuperhumanCertifier()
    res = cert.certify([0.99, 0.99], human_baseline=None)
    assert res.certified is False
    assert "no human baseline" in res.reason


def test_saturation_blocks_certification():
    cert = SuperhumanCertifier(saturation_ceiling=0.98)
    # high rewards but a saturated pass-rate history → uninformative comparison
    res = cert.certify(
        [0.99, 0.99, 0.99], human_baseline=0.5, pass_rate_history=[0.99, 0.99, 0.99]
    )
    assert res.saturated is True
    assert res.certified is False


def test_certification_is_reproducible():
    cert = SuperhumanCertifier(seed=0)
    rewards = [0.8, 0.85, 0.82, 0.79, 0.83, 0.81]
    a = cert.certify(rewards, 0.6)
    b = cert.certify(rewards, 0.6)
    assert a.ci_lower == b.ci_lower and a.ci_upper == b.ci_upper


# --------------------------------------------------------------------------- #
# AdaptationBenchmark (SAFE-1.7)
# --------------------------------------------------------------------------- #


class _ScaffoldRewardVerifier:
    """Reward encoded directly in the scaffold name 'r=<x>' for a deterministic suite."""

    def verify(self, candidate: str) -> VerifierResult:
        try:
            r = float(candidate.split("r=")[1])
        except (IndexError, ValueError):
            r = 0.0
        return VerifierResult(reward=r, passed=r > 0, detail={})


def _task(task_id: str, best: float, human: float | None) -> SpecializationTask:
    return SpecializationTask(
        task_id=task_id,
        prompt_corpus=["r=0.2", f"r={best}"],
        verifier=_ScaffoldRewardVerifier(),
        target_tau=0.5,
        human_baseline=human,
    )


def test_benchmark_runs_suite_and_reports_leaderboard():
    bench = AdaptationBenchmark(rounds=1, certify_samples=3)
    runs = [
        (_task("super", best=0.9, human=0.7), lambda s: s),  # beats human
        (_task("sub", best=0.6, human=0.8), lambda s: s),  # below human
    ]
    entries = bench.run(runs)
    report = bench.report(entries)
    assert report["tasks"] == 2
    assert report["certified_superhuman"] == 1
    by_id = {e.task_id: e for e in entries}
    assert by_id["super"].certification["certified"] is True
    assert by_id["sub"].certification["certified"] is False
    # adaptation-speed metrics are present per task
    assert by_id["super"].metrics["sample_complexity"] is not None


def test_benchmark_is_reproducible():
    bench = AdaptationBenchmark(rounds=1, certify_samples=3)
    runs = [(_task("t", best=0.9, human=0.7), lambda s: s)]
    r1 = bench.report(bench.run(runs))
    r2 = bench.report(bench.run(runs))
    assert r1 == r2
