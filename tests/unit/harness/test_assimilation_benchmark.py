#!/usr/bin/python
from __future__ import annotations

"""Tests for the measured-lift assimilation benchmark suite (CONCEPT:AU-AHE.optimization.real-optimization-metric).

Each ``bench_*`` must return a :class:`BenchmarkResult` with the right metric and
``claim_reproduced is True`` under the fixed seed (the mechanism beats its
baseline in the paper's claimed direction) -- except PauseRec, whose claim does
not reproduce under the native RNG and is pinned at its measured values;
``run_all`` returns seven results;
``to_markdown`` renders every row; and the whole suite is deterministic.
"""

import pytest

# The compiled epistemic_graph.numeric kernel must be built for these tests; skip the whole module cleanly when it isn't, rather than erroring out collection (CONCEPT:AU-KG.compute.numeric-kernel).
pytest.importorskip("epistemic_graph.numeric")

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

# (bench_fn, expected metric substring) pairs for the benches whose paper
# claim reproduces under the engine's native seeded RNG. PauseRec is pinned
# separately below: its claim does not reproduce (research follow-up).
_BENCHES = [
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


def test_pauserec_measured_behavior_under_native_rng() -> None:
    """Pin what PauseRec measures today, not the paper's claim.

    Under the engine's native seeded RNG the query projection lands in the
    distractor cluster and two 0.5-blend pause steps do not pull the target
    back out, so neither arm retrieves a relevant item (NDCG@6 = 0 for both)
    and the claim is reported as NOT reproduced. Tracked as a research
    follow-up on the latent-refinement step; this pins the honest result so a
    change in either direction is visible.
    """
    result = bench_pauserec(seed=0)
    assert isinstance(result, BenchmarkResult)
    assert result.metric == "NDCG@6"
    assert result.baseline == 0.0
    assert result.ours == 0.0
    assert result.lift == 0.0
    assert result.claim_reproduced is False
    assert result.detail["pause_steps_ours"] == 2


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
    by_name = {r.name: r for r in results}
    assert by_name["PauseRec KG-2.93"].claim_reproduced is False  # see pinned test
    assert all(
        r.claim_reproduced for name, r in by_name.items() if name != "PauseRec KG-2.93"
    )


def test_to_markdown_renders_all_rows() -> None:
    """The Markdown table has a row per result plus a reproduced-count footer."""
    results = run_all(seed=0)
    md = to_markdown(results)
    for r in results:
        assert r.name in md
        assert r.metric in md
    assert "Claim reproduced" in md
    reproduced = sum(1 for r in results if r.claim_reproduced)
    assert reproduced == len(results) - 1  # PauseRec does not reproduce
    assert f"{reproduced}/{len(results)} claims reproduced" in md


def test_determinism_same_seed_same_numbers() -> None:
    """The whole suite is bit-for-bit reproducible under a fixed seed."""
    first = run_all(seed=0)
    second = run_all(seed=0)
    assert [(r.baseline, r.ours, r.lift) for r in first] == [
        (r.baseline, r.ours, r.lift) for r in second
    ]
