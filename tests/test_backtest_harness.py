from __future__ import annotations
"""Tests for CONCEPT:AHE-3.8 — Backtest Evaluation Harness."""


import os
import tempfile

import pytest

from agent_utilities.harness.continuous_evaluation_engine import (
    BacktestHarness,
    BacktestMetric,
    BacktestRunRecord,
)
from agent_utilities.models.knowledge_graph import (
    BacktestMetricNode,
    BacktestRunNode,
    RegistryEdgeType,
    RegistryNodeType,
)


@pytest.fixture
def harness(tmp_path):
    """Create a BacktestHarness with a temp database."""
    db_path = str(tmp_path / "test_backtest.db")
    return BacktestHarness(db_path=db_path)


class TestBacktestMetric:
    def test_creation(self):
        m = BacktestMetric(metric_name="sharpe_ratio", value=1.45)
        assert m.metric_name == "sharpe_ratio"
        assert m.value == 1.45
        assert m.window_index == 0
        assert m.is_passing is True

    def test_with_benchmark(self):
        m = BacktestMetric(
            metric_name="max_drawdown",
            value=0.12,
            benchmark_value=0.15,
            is_passing=True,
        )
        assert m.benchmark_value == 0.15


class TestBacktestHarness:
    def test_create_run(self, harness):
        run_id = harness.create_run(
            strategy_id="strat:mom",
            start_date="2024-01-01",
            end_date="2024-12-31",
            initial_capital=100_000.0,
            parameters={"lookback": 20},
        )
        assert run_id.startswith("bt:")
        record = harness.get_run(run_id)
        assert record is not None
        assert record.strategy_id == "strat:mom"
        assert record.initial_capital == 100_000.0
        assert record.status == "running"

    def test_record_metric(self, harness):
        run_id = harness.create_run(strategy_id="strat:test")
        harness.record_metric(run_id, "sharpe_ratio", 1.5)
        harness.record_metric(run_id, "max_drawdown", 0.08)

        record = harness.get_run(run_id)
        assert len(record.metrics) == 2
        assert record.metrics[0].metric_name == "sharpe_ratio"
        assert record.metrics[1].value == 0.08

    def test_complete_run(self, harness):
        run_id = harness.create_run(strategy_id="strat:test")
        harness.record_metric(run_id, "sharpe_ratio", 1.3)

        completed = harness.complete_run(
            run_id, final_capital=112_500.0, total_trades=47
        )
        assert completed is not None
        assert completed.status == "completed"
        assert completed.final_capital == 112_500.0
        assert completed.total_trades == 47
        assert completed.completed_at is not None

    def test_complete_nonexistent(self, harness):
        result = harness.complete_run("bt:nonexistent")
        assert result is None

    def test_walk_forward_metrics(self, harness):
        run_id = harness.create_run(
            strategy_id="strat:wf",
            walk_forward_windows=3,
        )
        # Record per-window metrics
        for i in range(1, 4):
            harness.record_metric(run_id, "sharpe_ratio", 1.0 + i * 0.1, window_index=i)
        # Record aggregate
        harness.record_metric(run_id, "sharpe_ratio", 1.2, window_index=0)

        record = harness.get_run(run_id)
        assert len(record.metrics) == 4
        windows = [m.window_index for m in record.metrics]
        assert 0 in windows  # aggregate
        assert 1 in windows  # window 1

    def test_compare_to_benchmark(self, harness):
        run_id = harness.create_run(strategy_id="strat:cmp", benchmark_id="bench:SP500")
        harness.record_metric(run_id, "sharpe_ratio", 1.5)
        harness.record_metric(run_id, "max_drawdown", 0.10)
        harness.record_metric(run_id, "win_rate", 0.55)

        comparisons = harness.compare_to_benchmark(
            run_id,
            benchmark_metrics={
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.15,
                "win_rate": 0.50,
            },
        )

        assert len(comparisons) == 3

        sharpe_cmp = next(c for c in comparisons if c.metric_name == "sharpe_ratio")
        assert sharpe_cmp.outperforms is True
        assert sharpe_cmp.delta == pytest.approx(0.3)

        dd_cmp = next(c for c in comparisons if c.metric_name == "max_drawdown")
        assert dd_cmp.outperforms is True  # lower drawdown = better

    def test_compare_nonexistent(self, harness):
        result = harness.compare_to_benchmark("bt:nope", {"sharpe_ratio": 1.0})
        assert result == []

    def test_list_runs(self, harness):
        harness.create_run(strategy_id="strat:A")
        harness.create_run(strategy_id="strat:B")
        harness.create_run(strategy_id="strat:A")

        all_runs = harness.list_runs()
        assert len(all_runs) == 3

        filtered = harness.list_runs(strategy_id="strat:A")
        assert len(filtered) == 2

    def test_purge(self, harness):
        harness.create_run(strategy_id="strat:del")
        harness.create_run(strategy_id="strat:del2")
        count = harness.purge()
        assert count == 2
        assert len(harness.list_runs()) == 0


class TestBacktestKGNodes:
    def test_run_node(self):
        node = BacktestRunNode(
            id="bt:001",
            name="Momentum V2 Backtest",
            strategy_id="strat:momentum_v2",
            start_date="2024-01-01",
            end_date="2024-12-31",
            initial_capital=100_000.0,
            final_capital=115_000.0,
            total_trades=120,
            parameters={"lookback": 20, "threshold": 0.02},
            walk_forward_windows=4,
            benchmark_id="bench:SP500",
        )
        assert node.type == RegistryNodeType.BACKTEST_RUN
        assert node.strategy_id == "strat:momentum_v2"
        assert node.walk_forward_windows == 4

    def test_metric_node(self):
        node = BacktestMetricNode(
            id="bm:001",
            name="Sharpe Ratio",
            metric_name="sharpe_ratio",
            value=1.45,
            benchmark_value=1.20,
            is_passing=True,
        )
        assert node.type == RegistryNodeType.BACKTEST_METRIC
        assert node.value == 1.45

    def test_edge_types(self):
        assert RegistryEdgeType.EVALUATED_STRATEGY == "evaluated_strategy"
        assert RegistryEdgeType.HAS_METRIC == "has_metric"
        assert RegistryEdgeType.COMPARED_TO_BENCHMARK == "compared_to_benchmark"

    def test_backtest_graph(self):
        import networkx as nx

        g = nx.MultiDiGraph()

        from agent_utilities.models.knowledge_graph import StrategyNode

        strat = StrategyNode(id="strat:v2", name="Momentum V2")
        bt = BacktestRunNode(id="bt:001", name="BT", strategy_id="strat:v2")
        m1 = BacktestMetricNode(
            id="bm:sr", name="Sharpe", metric_name="sharpe_ratio", value=1.5
        )
        m2 = BacktestMetricNode(
            id="bm:dd", name="Drawdown", metric_name="max_drawdown", value=0.1
        )

        for n in [strat, bt, m1, m2]:
            g.add_node(n.id, **n.model_dump())

        g.add_edge(bt.id, strat.id, type=RegistryEdgeType.EVALUATED_STRATEGY)
        g.add_edge(bt.id, m1.id, type=RegistryEdgeType.HAS_METRIC)
        g.add_edge(bt.id, m2.id, type=RegistryEdgeType.HAS_METRIC)

        assert g.out_degree(bt.id) == 3
        # Traverse from backtest to all metrics
        metric_ids = [
            t
            for _, t, d in g.out_edges(bt.id, data=True)
            if d.get("type") == RegistryEdgeType.HAS_METRIC
        ]
        assert len(metric_ids) == 2
