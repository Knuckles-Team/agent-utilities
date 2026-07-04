#!/usr/bin/python
from __future__ import annotations

"""Eval-set optimizer: optimize the EVALS themselves and close the compounding loop.

Source: GEPA enterprise learning loop — "Owning Your Token Capital / Enterprise
AI Learning Loop" (GEPA, arXiv:2507.19457). The article's compounding-IP thesis is
that the durable asset is not the prompt but the EVAL SET: start from a small seed
of expert annotations, then make every production failure a new eval case and
re-optimize the harness against the ever-growing eval set. The set compounds; the
harness is just tuned against it.

CONCEPT:AU-ORCH.execution.eval-set-optimization-compounding — Eval-Set Optimization & Compounding Learning Loop.

This complements :class:`agent_utilities.rlm.gepa.GEPAOptimizer`, which optimizes
PROMPTS. Here we optimize the EVAL SET (the source of truth) and close the
trace -> eval -> re-optimize loop so each failure permanently sharpens the bar.

Design constraints (repo policy):
    - No ``os.environ``, no stubs / ``NotImplementedError``, no back-compat shims.
    - Fully dependency-INJECTED: the caller supplies the judge (and an optional
      eval-refiner), so the whole module is unit-testable with NO LLM and is
      deterministic for fixed inputs.
    - Stdlib + dataclasses + typing only.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "EvalCase",
    "EvalResult",
    "EvalSet",
    "JudgeFn",
    "RefineFn",
    "EvalSetOptimizer",
]


@dataclass
class EvalCase:
    """A single evaluation case — one row of the compounding eval set.

    Attributes:
        case_id: Stable identity used for dedup across rounds.
        input: The query / task handed to the system under evaluation.
        expected: The org's desired answer, or a rubric anchor the judge scores against.
        source: Provenance — ``"seed"`` (expert annotation), ``"production_failure"``
            (harvested from a live miss), or ``"refined"`` (a sharpened case).
        weight: Relative importance when aggregating scores; expensive / high-value
            cases can be weighted up so the pass-rate reflects what matters.
    """

    case_id: str
    input: str
    expected: str
    source: str = "seed"
    weight: float = 1.0


@dataclass
class EvalResult:
    """The judged outcome of running the system against one :class:`EvalCase`."""

    case_id: str
    score: float
    passed: bool


# (system_output, case) -> result. Injected by the caller; never an LLM at the type level.
JudgeFn = Callable[[str, EvalCase], EvalResult]
# (failing_case, system_output) -> sharper case. Optional; sharpens harvested failures.
RefineFn = Callable[[EvalCase, str], EvalCase] | None


class EvalSet:
    """An ordered, id-deduplicated collection of :class:`EvalCase` — the compounding IP.

    Insertion order is preserved (so growth across rounds is observable and
    deterministic). Adding a case whose ``case_id`` already exists is a no-op, which
    keeps the same production failure from inflating the set on repeated rounds.
    """

    def __init__(self, cases: list[EvalCase] | None = None) -> None:
        self._cases: list[EvalCase] = []
        self._ids: set[str] = set()
        for case in cases or []:
            self.add(case)

    def add(self, case: EvalCase) -> None:
        """Append ``case`` unless a case with the same ``case_id`` is already present."""
        if case.case_id in self._ids:
            return
        self._ids.add(case.case_id)
        self._cases.append(case)

    def cases(self) -> list[EvalCase]:
        """Return a copy of the cases in insertion order (callers cannot mutate state)."""
        return list(self._cases)

    def __len__(self) -> int:
        return len(self._cases)

    def __contains__(self, item: object) -> bool:
        """Membership by ``case_id`` string, or by an :class:`EvalCase`'s ``case_id``."""
        if isinstance(item, EvalCase):
            return item.case_id in self._ids
        return item in self._ids


@dataclass
class EvalSetOptimizer:
    """Optimize the eval set itself + close the compounding learning loop (ORCH-1.55).

    The optimizer never tunes prompts. Its job is to (a) grow a high-signal eval set
    from a small expert seed plus every production failure, and (b) score successive
    system versions against that set so the harness is held to a monotonically
    rising bar. The eval set is the asset that compounds; the score is the proof.

    Args:
        eval_set: The (possibly seeded) set to grow and score against.
        judge_fn: Injected scorer mapping ``(system_output, case) -> EvalResult``.
            In production this wraps an LLM-as-judge; in tests it is a pure function.
        pass_threshold: A case passes when ``judge_fn`` returns a result whose
            ``passed`` is True OR (when the judge leaves that to the optimizer) whose
            ``score`` is ``>= pass_threshold``. The judge's explicit ``passed`` wins.
        refine_fn: Optional injected refiner that sharpens a harvested failure into a
            tighter eval case (e.g. tightening the rubric to the exact missed point).
    """

    eval_set: EvalSet
    judge_fn: JudgeFn
    pass_threshold: float = 0.6
    refine_fn: RefineFn = field(default=None)

    # ── seeding ───────────────────────────────────────────────────────────────

    def seed_from_annotations(self, annotations: list[tuple[str, str, str]]) -> int:
        """Turn expert ``(case_id, input, expected)`` annotations into seed eval cases.

        This is the small, expensive, human-authored kernel the article starts from.
        Returns the number of cases actually added (dedup by ``case_id`` means
        re-seeding the same annotations adds nothing).
        """
        before = len(self.eval_set)
        for case_id, input_text, expected in annotations:
            self.eval_set.add(
                EvalCase(
                    case_id=case_id,
                    input=input_text,
                    expected=expected,
                    source="seed",
                )
            )
        return len(self.eval_set) - before

    # ── scoring ───────────────────────────────────────────────────────────────

    def _judge(self, system_output: str, case: EvalCase) -> EvalResult:
        """Judge one output, honoring an explicit ``passed`` or falling back to the threshold."""
        result = self.judge_fn(system_output, case)
        # Respect the judge's explicit verdict; otherwise derive it from the threshold.
        passed = result.passed or result.score >= self.pass_threshold
        return EvalResult(case_id=result.case_id, score=result.score, passed=passed)

    def score_system(self, system_fn: Callable[[str], str]) -> dict[str, Any]:
        """Run ``system_fn`` over every case and judge each output.

        Returns a report with weight-aware ``pass_rate`` and ``mean_score`` plus the
        per-case ``results`` and the subset of ``failures`` (cases that did not pass,
        paired with the output that failed them). Deterministic for a deterministic
        ``system_fn`` and ``judge_fn``.
        """
        results: list[EvalResult] = []
        failures: list[tuple[EvalCase, str]] = []
        total_weight = 0.0
        passed_weight = 0.0
        weighted_score = 0.0

        for case in self.eval_set.cases():
            output = system_fn(case.input)
            result = self._judge(output, case)
            results.append(result)
            weight = case.weight
            total_weight += weight
            weighted_score += weight * result.score
            if result.passed:
                passed_weight += weight
            else:
                failures.append((case, output))

        pass_rate = passed_weight / total_weight if total_weight else 0.0
        mean_score = weighted_score / total_weight if total_weight else 0.0
        return {
            "pass_rate": pass_rate,
            "mean_score": mean_score,
            "results": results,
            "failures": failures,
        }

    # ── the compounding loop ───────────────────────────────────────────────────

    def harvest_failure(self, system_output: str, case: EvalCase) -> EvalCase:
        """Turn a production failure into a NEW eval case and add it to the set.

        The harvested case carries ``source="production_failure"`` and a derived
        ``case_id`` so it is stable and dedup-safe. When a ``refine_fn`` is injected,
        the raw harvested case is handed to it to be sharpened (the refiner may, for
        instance, re-anchor ``expected`` to the precise point that was missed and mark
        it ``source="refined"``). The (possibly refined) case is added and returned.
        """
        harvested = EvalCase(
            case_id=f"prodfail:{case.case_id}",
            input=case.input,
            expected=case.expected,
            source="production_failure",
            weight=case.weight,
        )
        if self.refine_fn is not None:
            harvested = self.refine_fn(harvested, system_output)
        self.eval_set.add(harvested)
        return harvested

    def optimize_round(self, system_fn: Callable[[str], str]) -> dict[str, Any]:
        """Run ONE compounding round against ``system_fn``.

        Scores the system, then harvests every failure into a new eval case — so the
        eval set GROWS by (at most) the number of distinct failures (dedup keeps a
        recurring failure from being added twice). The returned report captures the
        pass rate measured BEFORE the set grew (the honest score for this version),
        the number of newly added evals, the resulting eval-set size, and the
        failures that drove the growth.
        """
        report = self.score_system(system_fn)
        size_before = len(self.eval_set)
        for case, output in report["failures"]:
            self.harvest_failure(output, case)
        n_new = len(self.eval_set) - size_before
        return {
            "pass_rate_before": report["pass_rate"],
            "mean_score_before": report["mean_score"],
            "n_new_evals": n_new,
            "eval_set_size": len(self.eval_set),
            "failures": report["failures"],
        }

    def compounding_loop(
        self, system_fns: list[Callable[[str], str]]
    ) -> list[dict[str, Any]]:
        """Run :meth:`optimize_round` for each successive (improved) system version.

        Each round may grow the eval set, so later, better systems are judged against
        a strictly tougher bar — the compounding-IP property the article describes.
        Asserts the eval set is monotonically non-decreasing across rounds (it can
        only grow) and returns the per-round reports in order.
        """
        reports: list[dict[str, Any]] = []
        prev_size = len(self.eval_set)
        for system_fn in system_fns:
            report = self.optimize_round(system_fn)
            current_size = report["eval_set_size"]
            if current_size < prev_size:
                raise AssertionError(
                    "eval set shrank across a compounding round "
                    f"({prev_size} -> {current_size}); the eval set is the "
                    "compounding asset and must never lose cases"
                )
            prev_size = current_size
            reports.append(report)
        return reports
