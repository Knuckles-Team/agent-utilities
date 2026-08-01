#!/usr/bin/python
from __future__ import annotations

"""Unified benchmark harness for reasoning-graph topologies.

CONCEPT:AU-AHE.evaluation.reasoning-topology-benchmark

Records accuracy, pass rate, grounding, tokens, wall time, tool calls, cache
reuse, cost, and reliability per topology run, so topology CHOICE becomes a
measurable router decision (consumed by :mod:`.policy`) instead of a
hardcoded preference.

Truthfulness (``AGENTS.md``): ``TopologyStats.accuracy`` counts a task correct
ONLY when it was both correct AND not budget-degraded — a degraded/truncated
run can still count toward the looser ``pass_rate`` axis (useful signal: "got
there anyway"), but never toward ``accuracy`` or ``reliability``.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

from .budgets import TerminationProof


@dataclass
class BenchmarkRecord:
    """One topology run's outcome against one benchmark task."""

    topology: str
    task_id: str
    correct: bool
    grounded: bool | None  # None when the task has no grounding notion
    tokens: int
    wall_time_s: float
    tool_calls: int
    cache_hits: int
    cost_usd: float
    degraded: bool
    termination_reason: str


@dataclass
class TopologyStats:
    """Aggregate statistics over all recorded runs of one topology."""

    topology: str
    n: int = 0
    accuracy: float = 0.0
    pass_rate: float = 0.0
    grounding_rate: float = 0.0
    mean_tokens: float = 0.0
    mean_wall_time_s: float = 0.0
    mean_tool_calls: float = 0.0
    cache_reuse_rate: float = 0.0
    mean_cost_usd: float = 0.0
    reliability: float = 0.0


class BenchmarkHarness:
    """Accumulates :class:`BenchmarkRecord`s and reports per-topology stats."""

    def __init__(self) -> None:
        self._records: list[BenchmarkRecord] = []

    @property
    def records(self) -> list[BenchmarkRecord]:
        return list(self._records)

    def record(self, record: BenchmarkRecord) -> None:
        self._records.append(record)

    def run_task(
        self,
        topology: str,
        task_id: str,
        run_fn: Callable[[], tuple[bool, TerminationProof, dict[str, float]]],
    ) -> BenchmarkRecord:
        """Run one task and record its :class:`BenchmarkRecord`.

        ``run_fn`` executes the topology and returns
        ``(correct, proof, metrics)`` where ``metrics`` may carry ``tokens``,
        ``tool_calls``, ``cache_hits``, ``cost_usd``, ``grounded`` (the
        harness fills in ``wall_time_s`` itself and any omitted metric
        defaults to ``0``/``None``).
        """
        start = time.monotonic()
        correct, proof, metrics = run_fn()
        wall_time_s = time.monotonic() - start
        grounded_metric = metrics.get("grounded")
        record = BenchmarkRecord(
            topology=topology,
            task_id=task_id,
            correct=bool(correct),
            grounded=None if grounded_metric is None else bool(grounded_metric),
            tokens=int(metrics.get("tokens", 0)),
            wall_time_s=wall_time_s,
            tool_calls=int(metrics.get("tool_calls", 0)),
            cache_hits=int(metrics.get("cache_hits", 0)),
            cost_usd=float(metrics.get("cost_usd", 0.0)),
            degraded=proof.degraded,
            termination_reason=proof.reason.value,
        )
        self.record(record)
        return record

    def stats(self, topology: str | None = None) -> dict[str, TopologyStats]:
        """Aggregate stats grouped by topology (or just ``topology`` if given)."""
        by_topology: dict[str, list[BenchmarkRecord]] = {}
        for rec in self._records:
            if topology is not None and rec.topology != topology:
                continue
            by_topology.setdefault(rec.topology, []).append(rec)

        out: dict[str, TopologyStats] = {}
        for name, records in by_topology.items():
            n = len(records)
            grounded_records = [r for r in records if r.grounded is not None]
            clean_correct = sum(1 for r in records if r.correct and not r.degraded)
            out[name] = TopologyStats(
                topology=name,
                n=n,
                accuracy=clean_correct / n,
                pass_rate=sum(1 for r in records if r.correct) / n,
                grounding_rate=(
                    sum(1 for r in grounded_records if r.grounded)
                    / len(grounded_records)
                    if grounded_records
                    else 0.0
                ),
                mean_tokens=sum(r.tokens for r in records) / n,
                mean_wall_time_s=sum(r.wall_time_s for r in records) / n,
                mean_tool_calls=sum(r.tool_calls for r in records) / n,
                cache_reuse_rate=sum(r.cache_hits for r in records)
                / max(1, sum(r.tool_calls for r in records)),
                mean_cost_usd=sum(r.cost_usd for r in records) / n,
                reliability=sum(1 for r in records if not r.degraded) / n,
            )
        return out
