"""Drive systems over a task at multiple scales and aggregate results (CONCEPT:AU-AHE.rlm.long-context-benchmark)."""

from __future__ import annotations

import time
from typing import Any

from .base import BenchResult, get_task
from .baselines import System


def _complete_costs(costs: list[float | None], count: int) -> list[float] | None:
    known = [cost for cost in costs if cost is not None]
    return known if count and len(costs) == count == len(known) else None


def _mean_known_cost(costs: list[float | None], count: int) -> float | None:
    known = _complete_costs(costs, count)
    if known is None:
        return None
    return round(sum(known) / count, 6)


def _build_bench_result(
    *,
    task: Any,
    system: System,
    scale: int,
    cases: list[Any],
    scores: list[float],
    costs: list[float | None],
    tokens: int,
    wall: float,
    max_depth: int,
    errors: int,
) -> BenchResult:
    count = len(scores)
    return BenchResult(
        task=task.name,
        complexity=task.complexity,
        system=system.name,
        scale=scale,
        accuracy=round(sum(scores) / count, 4) if count else 0.0,
        n=count,
        cost_usd=_mean_known_cost(costs, count),
        total_tokens=tokens // count if count else 0,
        wall_s=round(wall / count, 3) if count else 0.0,
        max_depth=max_depth,
        mode=cases[0].mode if cases else "synthetic",
        notes=f"{errors} case error(s)" if errors else "",
    )


async def run_benchmark(
    task_name: str,
    *,
    scales: list[int],
    systems: list[System] | None = None,
    cases_per_scale: int = 3,
    seed0: int = 0,
) -> list[BenchResult]:
    """Run ``systems`` over ``task_name`` at each scale, returning one :class:`BenchResult` each.

    ``systems`` defaults to the live trio (RLM, vanilla, compaction); pass explicit systems (e.g.
    with a fake completer) to run offline. Each (system, scale) cell averages ``cases_per_scale``
    independently-seeded cases. A system that errors on a case contributes a 0.0 score for that
    case rather than aborting the sweep.
    """
    if systems is None:
        from .baselines import CompactionSystem, RLMSystem, VanillaSystem

        systems = [RLMSystem(), VanillaSystem(), CompactionSystem()]

    task = get_task(task_name)
    results: list[BenchResult] = []
    for scale in scales:
        cases = [task.build(scale, seed=seed0 + k) for k in range(cases_per_scale)]
        for system in systems:
            scores: list[float] = []
            costs: list[float | None] = []
            tokens = 0
            max_depth = 0
            errors = 0
            t0 = time.perf_counter()
            for case in cases:
                try:
                    out = await system.answer(case)
                    scores.append(case.grade(out.prediction))
                    costs.append(out.cost_usd)
                    tokens += out.tokens
                    max_depth = max(max_depth, out.max_depth)
                except Exception:  # noqa: BLE001 — a failed case scores 0, sweep continues
                    scores.append(0.0)
                    costs.append(None)
                    errors += 1
            wall = time.perf_counter() - t0
            results.append(
                _build_bench_result(
                    task=task,
                    system=system,
                    scale=scale,
                    cases=cases,
                    scores=scores,
                    costs=costs,
                    tokens=tokens,
                    wall=wall,
                    max_depth=max_depth,
                    errors=errors,
                )
            )
    return results
