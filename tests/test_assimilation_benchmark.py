#!/usr/bin/python
from __future__ import annotations

"""Tests for the measured-lift assimilation benchmark suite (CONCEPT:AU-AHE.optimization.real-optimization-metric).

Each ``bench_*`` must return a :class:`BenchmarkResult` with the right metric and
``claim_reproduced is True`` under the fixed seed (the mechanism beats its
baseline in the paper's claimed direction); ``run_all`` returns seven results;
``to_markdown`` renders every row; and the whole suite is deterministic.
"""

import pytest

from agent_utilities.harness.assimilation_benchmark import (
    BenchmarkResult,
    bench_adore,
    bench_decentmem_bandit,
    bench_mlevolve,
    bench_pauserec,
    bench_scoregate,
    bench_sgs,
    bench_tasr,
    run_all,
    to_markdown,
)

# (bench_fn, expected metric substring) pairs.
_BENCHES = [
    (bench_pauserec, "NDCG"),
    (bench_scoregate, "precision"),
    (bench_tasr, "rounds"),
    (bench_adore, "Recall"),
    (bench_decentmem_bandit, "regret"),
    (bench_mlevolve, "best-metric"),
    (bench_sgs, "accepted-quality"),
]


@pytest.mark.parametrize("bench_fn, metric_substr", _BENCHES)
def test_bench_reproduces_claim(bench_fn, metric_substr) -> None:
    """Every benchmark returns a valid result that reproduces its paper's claim."""
    result = bench_fn(seed=0)
    assert isinstance(result, BenchmarkResult)
    assert metric_substr in result.metric
    assert result.claim_reproduced is True, (
        f"{result.name}: baseline={result.baseline} ours={result.ours} "
        f"lift={result.lift} (higher_is_better={result.higher_is_better})"
    )
    # The verdict must agree with a positive direction-aware lift.
    assert result.lift > 0.0
    assert result.detail  # every bench reports mechanism-specific detail


def test_lift_direction_is_consistent() -> None:
    """Lift sign matches the higher/lower-is-better convention for each result."""
    for result in run_all(seed=0):
        if result.higher_is_better:
            assert result.lift == pytest.approx(result.ours - result.baseline)
        else:
            assert result.lift == pytest.approx(result.baseline - result.ours)


def test_tasr_saves_rounds_at_equal_recall() -> None:
    """TASR uses strictly fewer rounds and reaches the same final answer."""
    result = bench_tasr(seed=0)
    assert result.metric == "rounds"
    assert result.ours < result.baseline
    assert result.detail["rounds_saved"] > 0
    assert result.detail["equal_final_answer"] is True
    assert result.detail["stop_reason"] == "answer_repeat"


def test_scoregate_holds_full_recall() -> None:
    """ScoreGate keeps the whole relevant cluster while raising precision."""
    result = bench_scoregate(seed=0)
    assert result.detail["ours_recall"] == pytest.approx(1.0)
    assert result.ours > result.baseline


def test_adore_recovers_expansion_only_docs() -> None:
    """ADORE beats one-shot exactly because it recovers expansion-only relevants."""
    result = bench_adore(seed=0)
    assert result.ours > result.baseline
    assert result.detail["rounds_run"] > 1


def test_mlevolve_uses_fusion() -> None:
    """Multi-branch search finds a higher best metric than a single branch."""
    result = bench_mlevolve(seed=0)
    assert result.ours > result.baseline


def test_sgs_guide_rejects_gamed_tasks() -> None:
    """The Guide raises accepted-task quality by rejecting gamed conjectures."""
    result = bench_sgs(seed=0)
    assert result.ours > result.baseline
    assert result.detail["gamed_rejected"] > 0


def test_run_all_returns_core_benchmarks() -> None:
    """run_all yields the seven deterministic rows (+ the trained-pause row when torch is present)."""
    results = run_all(seed=0)
    assert len(results) >= 7  # 8 when torch is installed (trained-pause-token bench)
    assert all(isinstance(r, BenchmarkResult) for r in results)
    assert all(r.claim_reproduced for r in results)


def test_to_markdown_renders_all_rows() -> None:
    """The Markdown table has a row per result plus a reproduced-count footer."""
    results = run_all(seed=0)
    md = to_markdown(results)
    for r in results:
        assert r.name in md
        assert r.metric in md
    assert "Claim reproduced" in md
    assert f"{len(results)}/{len(results)} claims reproduced" in md


def test_determinism_same_seed_same_numbers() -> None:
    """The whole suite is bit-for-bit reproducible under a fixed seed."""
    first = run_all(seed=0)
    second = run_all(seed=0)
    assert [(r.baseline, r.ours, r.lift) for r in first] == [
        (r.baseline, r.ours, r.lift) for r in second
    ]
