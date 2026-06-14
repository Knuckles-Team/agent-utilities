#!/usr/bin/python
from __future__ import annotations

"""Reproducible adaptation-speed benchmark for the SAI factory (CONCEPT:SAFE-1.7).

The "time-to-superhuman leaderboard" the SAI paper calls for but never builds. It
runs the factory (AHE-3.29) over a *fixed, seeded* suite of specialization tasks
and reports, per task, the adaptation-speed metrics (AHE-3.27 — time-to-target,
sample-complexity, learning-AUC) and the superhuman-certification verdict
(SAFE-1.6). Because the factory and verifiers are deterministic and the certifier's
bootstrap is seeded, the whole report is reproducible run-to-run — the property a
progress tracker needs to be trustworthy.

This is the measurement counterpart to AU's self-baseline `ImprovementVelocity`
(SAFE-1.3): that tracks the *repo's* RSI cadence; this tracks *per-task adaptation
speed and superhuman status*, the SAI primary axes.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.harness.sai_task import SpecializationTask
from agent_utilities.harness.superhuman_gate import SuperhumanCertifier

# NOTE: SaiFactoryController is imported lazily inside ``run`` — it lives in
# knowledge_graph.research and imports back from this harness package, so a
# module-level import here creates a harness<->research cycle that breaks any cold
# import of sai_factory (harness/__init__ eagerly loads this module).

GenerateFn = Callable[[str], str]
#: One benchmark run: a task + the generator that proposes candidates for it.
BenchmarkRun = tuple[SpecializationTask, GenerateFn]


@dataclass
class BenchmarkEntry:
    """Per-task benchmark result."""

    task_id: str
    metrics: dict[str, Any]
    certification: dict[str, Any]


@dataclass
class AdaptationBenchmark:
    """Run the factory over a fixed task suite and report adaptation speed + cert."""

    rounds: int = 2
    certify_samples: int = 5
    certifier: SuperhumanCertifier = field(default_factory=SuperhumanCertifier)

    def run(self, runs: Sequence[BenchmarkRun]) -> list[BenchmarkEntry]:
        from agent_utilities.knowledge_graph.research.sai_factory import (
            SaiFactoryController,
        )

        entries: list[BenchmarkEntry] = []
        for task, generate_fn in runs:
            controller = SaiFactoryController(
                task, generate_fn, scaffolds=task.prompt_corpus
            )
            result = controller.run(rounds=self.rounds)
            # Re-evaluate the produced specialist to sample its reward distribution
            # (a point mass for deterministic verifiers; a spread for noisy ones).
            samples = [
                task.score(result.specialist.generate()).reward
                for _ in range(max(1, self.certify_samples))
            ]
            cert = self.certifier.certify(samples, task.human_baseline)
            entries.append(
                BenchmarkEntry(
                    task_id=task.task_id,
                    metrics=result.metrics(),
                    certification=cert.to_dict(),
                )
            )
        return entries

    @staticmethod
    def report(entries: Sequence[BenchmarkEntry]) -> dict[str, Any]:
        """A compact leaderboard: per-task speed + superhuman status, and totals."""
        rows = [
            {
                "task_id": e.task_id,
                "time_to_target_s": e.metrics.get("time_to_target_s"),
                "sample_complexity": e.metrics.get("sample_complexity"),
                "learning_auc": e.metrics.get("learning_auc"),
                "final_reward": e.metrics.get("final_specialist_reward"),
                "certified_superhuman": e.certification.get("certified"),
            }
            for e in entries
        ]
        return {
            "tasks": len(rows),
            "certified_superhuman": sum(1 for r in rows if r["certified_superhuman"]),
            "leaderboard": rows,
        }
