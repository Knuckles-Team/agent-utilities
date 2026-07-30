"""Unit tests for the unified reasoning-topology benchmark harness."""

from agent_utilities.graph.reasoning.benchmark import BenchmarkHarness
from agent_utilities.graph.reasoning.budgets import TerminationProof, TerminationReason


def _clean_success(**metrics):
    def run_fn():
        proof = TerminationProof(
            reason=TerminationReason.GOAL_REACHED, success=True, degraded=False
        )
        return True, proof, metrics

    return run_fn


def _degraded_but_correct(**metrics):
    def run_fn():
        proof = TerminationProof(
            reason=TerminationReason.LOOP_BUDGET_EXHAUSTED, success=True, degraded=True
        )
        return True, proof, metrics

    return run_fn


def test_run_task_records_wall_time_and_metrics():
    harness = BenchmarkHarness()
    record = harness.run_task(
        "cot",
        "task-1",
        _clean_success(
            tokens=100, tool_calls=2, cache_hits=1, cost_usd=0.01, grounded=True
        ),
    )
    assert record.topology == "cot"
    assert record.tokens == 100
    assert record.tool_calls == 2
    assert record.wall_time_s >= 0
    assert record.degraded is False


def test_stats_accuracy_excludes_degraded_runs_even_when_correct():
    harness = BenchmarkHarness()
    harness.run_task("rap", "t1", _clean_success(tokens=10))
    harness.run_task("rap", "t2", _degraded_but_correct(tokens=10))

    stats = harness.stats()["rap"]
    assert stats.n == 2
    # pass_rate counts BOTH as correct (looser axis)...
    assert stats.pass_rate == 1.0
    # ...but accuracy excludes the degraded run -- never a clean success.
    assert stats.accuracy == 0.5
    assert stats.reliability == 0.5


def test_stats_grounding_rate_ignores_tasks_with_no_grounding_notion():
    harness = BenchmarkHarness()
    harness.run_task("cot", "t1", _clean_success(grounded=True))
    harness.run_task("cot", "t2", _clean_success(grounded=False))
    harness.run_task("cot", "t3", _clean_success())  # no grounding notion at all

    stats = harness.stats()["cot"]
    assert stats.n == 3
    assert stats.grounding_rate == 0.5  # 1 of the 2 grounding-applicable tasks


def test_stats_cache_reuse_rate_is_hits_over_tool_calls():
    harness = BenchmarkHarness()
    harness.run_task("react", "t1", _clean_success(tool_calls=4, cache_hits=1))
    harness.run_task("react", "t2", _clean_success(tool_calls=4, cache_hits=3))

    stats = harness.stats()["react"]
    assert stats.cache_reuse_rate == 4 / 8


def test_stats_filters_by_topology():
    harness = BenchmarkHarness()
    harness.run_task("cot", "t1", _clean_success())
    harness.run_task("rap", "t2", _clean_success())

    only_rap = harness.stats(topology="rap")
    assert set(only_rap) == {"rap"}
