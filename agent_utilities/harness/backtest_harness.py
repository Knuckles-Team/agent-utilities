"""Backtest / Evaluation Harness — strategy validation and capture.

CONCEPT:AHE-3.8 — Backtest Evaluation Harness

Domain-agnostic backtesting and evaluation harness that extends
KGEvalCapture.  Records multi-dimensional evaluation results with
walk-forward validation windows and benchmark comparison.

Inspired by Vibe-Trading's 7 backtest engines with statistical validation
(Monte Carlo, Bootstrap CI, Walk-Forward).

Usage::

    from agent_utilities.harness.backtest_harness import BacktestHarness

    harness = BacktestHarness()
    run_id = harness.create_run(
        strategy_id="strategy:momentum_v2",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={"lookback": 20, "threshold": 0.02},
    )
    harness.record_metric(run_id, "sharpe_ratio", 1.45)
    harness.record_metric(run_id, "max_drawdown", 0.12)
    harness.complete_run(run_id, final_capital=112_500.0, total_trades=47)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = ".agent_workspace/backtest_log.db"


class BacktestMetric(BaseModel):
    """A single metric from a backtest run. CONCEPT:AHE-3.8"""

    metric_name: str
    value: float
    window_index: int = 0
    benchmark_value: float | None = None
    is_passing: bool = True


class BacktestRunRecord(BaseModel):
    """Complete record of a backtest run. CONCEPT:AHE-3.8"""

    run_id: str
    strategy_id: str = ""
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 100_000.0
    final_capital: float = 0.0
    total_trades: int = 0
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: str = "running"
    metrics: list[BacktestMetric] = Field(default_factory=list)
    walk_forward_windows: int = 0
    benchmark_id: str | None = None
    created_at: str = ""
    completed_at: str | None = None


class BacktestComparison(BaseModel):
    """Comparison result between a run and a benchmark. CONCEPT:AHE-3.8"""

    run_id: str
    benchmark_id: str
    metric_name: str
    run_value: float
    benchmark_value: float
    delta: float
    outperforms: bool


class BacktestHarness:
    """Domain-agnostic backtesting and evaluation harness.

    CONCEPT:AHE-3.8 — Backtest Evaluation Harness

    Records evaluation runs to a separate SQLite database to prevent
    KG contamination (same pattern as KGEvalCapture).

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._runs: dict[str, BacktestRunRecord] = {}
        self._db: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database schema."""
        try:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self._db_path)
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS backtest_runs (
                    run_id TEXT PRIMARY KEY,
                    strategy_id TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    initial_capital REAL,
                    final_capital REAL,
                    total_trades INTEGER,
                    parameters TEXT,
                    status TEXT,
                    walk_forward_windows INTEGER,
                    benchmark_id TEXT,
                    created_at TEXT,
                    completed_at TEXT
                )"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS backtest_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    metric_name TEXT,
                    value REAL,
                    window_index INTEGER DEFAULT 0,
                    benchmark_value REAL,
                    is_passing INTEGER DEFAULT 1,
                    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
                )"""
            )
            self._db.commit()
        except Exception as e:
            logger.warning("BacktestHarness DB init failed: %s", e)
            self._db = None

    def create_run(
        self,
        strategy_id: str = "",
        start_date: str = "",
        end_date: str = "",
        initial_capital: float = 100_000.0,
        parameters: dict[str, Any] | None = None,
        walk_forward_windows: int = 0,
        benchmark_id: str | None = None,
    ) -> str:
        """Create a new backtest run.

        CONCEPT:AHE-3.8

        Args:
            strategy_id: Reference to the strategy being evaluated.
            start_date: Evaluation period start (ISO date).
            end_date: Evaluation period end (ISO date).
            initial_capital: Starting capital.
            parameters: Strategy parameters for this run.
            walk_forward_windows: Number of walk-forward splits.
            benchmark_id: Optional benchmark for comparison.

        Returns:
            The generated run ID.
        """
        run_id = f"bt:{uuid.uuid4().hex[:8]}"
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        record = BacktestRunRecord(
            run_id=run_id,
            strategy_id=strategy_id,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            parameters=parameters or {},
            walk_forward_windows=walk_forward_windows,
            benchmark_id=benchmark_id,
            created_at=now,
        )
        self._runs[run_id] = record

        if self._db:
            try:
                self._db.execute(
                    """INSERT INTO backtest_runs
                    (run_id, strategy_id, start_date, end_date,
                     initial_capital, final_capital, total_trades,
                     parameters, status, walk_forward_windows,
                     benchmark_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        strategy_id,
                        start_date,
                        end_date,
                        initial_capital,
                        0.0,
                        0,
                        json.dumps(parameters or {}),
                        "running",
                        walk_forward_windows,
                        benchmark_id,
                        now,
                    ),
                )
                self._db.commit()
            except Exception as e:
                logger.warning("Failed to persist backtest run: %s", e)

        logger.info("Created backtest run %s for strategy %s", run_id, strategy_id)
        return run_id

    def record_metric(
        self,
        run_id: str,
        metric_name: str,
        value: float,
        window_index: int = 0,
        benchmark_value: float | None = None,
        is_passing: bool = True,
    ) -> None:
        """Record a metric for a backtest run.

        CONCEPT:AHE-3.8

        Args:
            run_id: The run to record the metric for.
            metric_name: Metric name (e.g., 'sharpe_ratio').
            value: Metric value.
            window_index: Walk-forward window index (0 = aggregate).
            benchmark_value: Benchmark comparison value.
            is_passing: Whether the metric passes its threshold.
        """
        metric = BacktestMetric(
            metric_name=metric_name,
            value=value,
            window_index=window_index,
            benchmark_value=benchmark_value,
            is_passing=is_passing,
        )

        if run_id in self._runs:
            self._runs[run_id].metrics.append(metric)

        if self._db:
            try:
                self._db.execute(
                    """INSERT INTO backtest_metrics
                    (run_id, metric_name, value, window_index,
                     benchmark_value, is_passing)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        metric_name,
                        value,
                        window_index,
                        benchmark_value,
                        int(is_passing),
                    ),
                )
                self._db.commit()
            except Exception as e:
                logger.warning("Failed to persist metric: %s", e)

    def complete_run(
        self,
        run_id: str,
        final_capital: float = 0.0,
        total_trades: int = 0,
        status: str = "completed",
    ) -> BacktestRunRecord | None:
        """Complete a backtest run and finalize results.

        CONCEPT:AHE-3.8

        Args:
            run_id: The run to complete.
            final_capital: Ending capital.
            total_trades: Total trades executed.
            status: Final status (completed, failed).

        Returns:
            The completed run record, or None if not found.
        """
        record = self._runs.get(run_id)
        if not record:
            return None

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record.final_capital = final_capital
        record.total_trades = total_trades
        record.status = status
        record.completed_at = now

        if self._db:
            try:
                self._db.execute(
                    """UPDATE backtest_runs
                    SET final_capital = ?, total_trades = ?,
                        status = ?, completed_at = ?
                    WHERE run_id = ?""",
                    (final_capital, total_trades, status, now, run_id),
                )
                self._db.commit()
            except Exception as e:
                logger.warning("Failed to update backtest run: %s", e)

        logger.info(
            "Completed backtest run %s: capital=%.2f, trades=%d",
            run_id,
            final_capital,
            total_trades,
        )
        return record

    def get_run(self, run_id: str) -> BacktestRunRecord | None:
        """Retrieve a backtest run by ID.

        Args:
            run_id: The run ID.

        Returns:
            The run record or None.
        """
        return self._runs.get(run_id)

    def compare_to_benchmark(
        self,
        run_id: str,
        benchmark_metrics: dict[str, float],
    ) -> list[BacktestComparison]:
        """Compare a run's metrics against benchmark values.

        CONCEPT:AHE-3.8

        Args:
            run_id: The run to compare.
            benchmark_metrics: Dict of metric_name → benchmark_value.

        Returns:
            List of comparison results.
        """
        record = self._runs.get(run_id)
        if not record:
            return []

        comparisons: list[BacktestComparison] = []
        for metric in record.metrics:
            if metric.metric_name in benchmark_metrics:
                bv = benchmark_metrics[metric.metric_name]
                delta = metric.value - bv
                # For drawdown-like metrics, lower is better
                if "drawdown" in metric.metric_name.lower():
                    outperforms = metric.value < bv
                else:
                    outperforms = metric.value > bv

                comparisons.append(
                    BacktestComparison(
                        run_id=run_id,
                        benchmark_id=record.benchmark_id or "default",
                        metric_name=metric.metric_name,
                        run_value=metric.value,
                        benchmark_value=bv,
                        delta=delta,
                        outperforms=outperforms,
                    )
                )
        return comparisons

    def list_runs(
        self,
        strategy_id: str | None = None,
        limit: int = 50,
    ) -> list[BacktestRunRecord]:
        """List backtest runs, optionally filtered by strategy.

        Args:
            strategy_id: Filter by strategy ID.
            limit: Maximum runs to return.

        Returns:
            List of run records.
        """
        runs = list(self._runs.values())
        if strategy_id:
            runs = [r for r in runs if r.strategy_id == strategy_id]
        return runs[:limit]

    def purge(self) -> int:
        """Delete all backtest data from the database.

        Returns:
            Number of runs purged.
        """
        count = len(self._runs)
        self._runs.clear()
        if self._db:
            try:
                self._db.execute("DELETE FROM backtest_metrics")
                self._db.execute("DELETE FROM backtest_runs")
                self._db.commit()
            except Exception as e:
                logger.warning("Failed to purge backtest data: %s", e)
        return count
