"""Drive systems over a task at multiple scales and aggregate results (CONCEPT:AHE-3.32)."""

from __future__ import annotations

import time

from .base import BenchResult, get_task
from .baselines import System


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
            cost = 0.0
            tokens = 0
            max_depth = 0
            errors = 0
            t0 = time.perf_counter()
            for case in cases:
                try:
                    out = await system.answer(case)
                    scores.append(case.grade(out.prediction))
                    cost += out.cost_usd
                    tokens += out.tokens
                    max_depth = max(max_depth, out.max_depth)
                except Exception:  # noqa: BLE001 — a failed case scores 0, sweep continues
                    scores.append(0.0)
                    errors += 1
            wall = time.perf_counter() - t0
            n = len(scores)
            note = f"{errors} case error(s)" if errors else ""
            results.append(
                BenchResult(
                    task=task.name,
                    complexity=task.complexity,
                    system=system.name,
                    scale=scale,
                    accuracy=round(sum(scores) / n, 4) if n else 0.0,
                    n=n,
                    cost_usd=round(cost / n, 6) if n else 0.0,
                    total_tokens=tokens // n if n else 0,
                    wall_s=round(wall / n, 3) if n else 0.0,
                    max_depth=max_depth,
                    mode=cases[0].mode if cases else "synthetic",
                    notes=note,
                )
            )
    return results
