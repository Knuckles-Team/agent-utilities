"""Pre-run go/no-go gates for research-craft discipline.

Operationalizes two pieces of practitioner wisdom as executable, blocking gates
that run *before* a scale run is allowed to consume compute:

1. **No gain claim survives an untuned / strong baseline.** "Be good at research"
   means tuning the baseline until it hurts: the graveyard of ML is full of gains
   that evaporated against a properly tuned baseline. A candidate must beat the
   baseline by a real margin or the "win" is rejected.
2. **Overfit a single batch first** (Karpathy, "A Recipe for Training Neural
   Networks" — "overfit a single batch first"). A model that cannot drive the
   loss down on one batch has a bug; catch it cheaply and fail fast rather than
   discovering it after a long, expensive scale run.

These gates are deterministic, dependency-free (stdlib + ``math``), and shaped to
compose alongside :mod:`agent_utilities.harness.quality_gates` — they share its
frozen-dataclass verdict style so a runner can persist both kinds of outcome.

CONCEPT:AHE-3.35
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["GateVerdict", "baseline_gate", "overfit_smoke_gate", "PreRunGate"]


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """Outcome of a single pre-run gate (or a composed set of them).

    Attributes:
        passed: ``True`` only when the gate's condition is satisfied.
        reason: A short, human-readable explanation of the verdict — empty-safe
            but always populated so a failing verdict is self-describing.
        detail: Structured, JSON-friendly evidence behind the verdict (the
            measured numbers, thresholds, and any sub-verdicts) for persistence.
    """

    passed: bool
    reason: str
    detail: dict[str, Any]


def baseline_gate(
    candidate_score: float,
    baseline_score: float,
    *,
    min_lift: float = 0.02,
    higher_is_better: bool = True,
) -> GateVerdict:
    """Reject a candidate that does not beat a (tuned) baseline by ``min_lift``.

    "The graveyard of ML is full of gains that evaporated against a properly
    tuned baseline." The lift is measured in the direction set by
    ``higher_is_better`` so the same gate works for accuracy-style metrics
    (higher wins) and loss/error-style metrics (lower wins).

    Args:
        candidate_score: The new approach's score.
        baseline_score: The strong/tuned baseline's score.
        min_lift: The minimum required improvement, in the metric's own units.
            Must be non-negative; a negative ``min_lift`` would let a regression
            pass and is rejected as a misconfiguration.
        higher_is_better: ``True`` when a larger score is better (accuracy);
            ``False`` when a smaller score is better (loss, error rate).

    Returns:
        A :class:`GateVerdict`; ``passed`` is ``True`` iff the realized lift is
        at least ``min_lift``.
    """
    if not math.isfinite(candidate_score) or not math.isfinite(baseline_score):
        return GateVerdict(
            passed=False,
            reason="non-finite score(s) supplied to baseline_gate",
            detail={
                "candidate_score": candidate_score,
                "baseline_score": baseline_score,
            },
        )
    if min_lift < 0:
        return GateVerdict(
            passed=False,
            reason=f"min_lift must be non-negative, got {min_lift}",
            detail={"min_lift": min_lift},
        )

    # Lift in the "good" direction: positive means the candidate improved.
    lift = (
        candidate_score - baseline_score
        if higher_is_better
        else baseline_score - candidate_score
    )
    passed = lift >= min_lift
    detail: dict[str, Any] = {
        "candidate_score": candidate_score,
        "baseline_score": baseline_score,
        "lift": lift,
        "min_lift": min_lift,
        "higher_is_better": higher_is_better,
    }
    if passed:
        reason = f"candidate beats baseline by {lift:.6g} >= min_lift {min_lift:.6g}"
    else:
        reason = (
            f"insufficient lift {lift:.6g} < min_lift {min_lift:.6g}: "
            "gain does not survive the tuned baseline"
        )
    return GateVerdict(passed=passed, reason=reason, detail=detail)


def overfit_smoke_gate(
    single_batch_loss_curve: Sequence[float],
    *,
    min_drop_frac: float = 0.5,
) -> GateVerdict:
    """Pass iff training on ONE batch can memorize it (loss collapses).

    Karpathy's "overfit a single batch first": before spending compute at scale,
    confirm the model/optimizer can drive the loss down on a single batch. The
    gate passes when the loss falls by at least ``min_drop_frac`` of its initial
    value. A flat or rising curve means the model cannot even memorize one batch
    — that is a bug (bad init, wrong loss wiring, dead gradients, a too-small
    learning rate) — so the gate fails fast.

    Args:
        single_batch_loss_curve: The per-step training loss recorded while
            fitting a single fixed batch, in step order.
        min_drop_frac: Required fractional drop from the initial loss, in
            ``[0, 1]``. ``0.5`` means the loss must at least halve.

    Returns:
        A :class:`GateVerdict`; ``passed`` is ``True`` only when the loss
        collapses by the required fraction. Empty or degenerate curves fail with
        a clear reason.
    """
    curve = list(single_batch_loss_curve)
    if not (0.0 <= min_drop_frac <= 1.0):
        return GateVerdict(
            passed=False,
            reason=f"min_drop_frac must be in [0, 1], got {min_drop_frac}",
            detail={"min_drop_frac": min_drop_frac},
        )
    if len(curve) < 2:
        return GateVerdict(
            passed=False,
            reason="loss curve needs at least 2 points to measure a drop",
            detail={"points": len(curve)},
        )
    if any(not math.isfinite(v) for v in curve):
        return GateVerdict(
            passed=False,
            reason="loss curve contains non-finite value(s) (nan/inf) — diverged",
            detail={"points": len(curve)},
        )

    initial = curve[0]
    final = min(curve)  # best loss reached, robust to a noisy last step
    if initial <= 0.0:
        return GateVerdict(
            passed=False,
            reason=f"initial loss must be positive to measure a drop, got {initial:.6g}",
            detail={"initial_loss": initial},
        )

    drop_frac = (initial - final) / initial
    passed = drop_frac >= min_drop_frac
    detail = {
        "initial_loss": initial,
        "final_loss": final,
        "drop_frac": drop_frac,
        "min_drop_frac": min_drop_frac,
        "points": len(curve),
    }
    if passed:
        reason = (
            f"single-batch loss dropped {drop_frac:.2%} >= {min_drop_frac:.2%}: "
            "model can memorize one batch"
        )
    else:
        reason = (
            f"single-batch loss only dropped {drop_frac:.2%} < {min_drop_frac:.2%}: "
            "model cannot overfit one batch — likely a bug, do not scale"
        )
    return GateVerdict(passed=passed, reason=reason, detail=detail)


class PreRunGate:
    """Compose baseline + overfit gates into one go/no-go before a scale run.

    Both research-craft disciplines must hold before compute is committed: the
    candidate must beat the tuned baseline (no evaporating gains) *and* the model
    must demonstrably overfit a single batch (no silent bug). The composed
    verdict passes only when both sub-gates pass, and its ``detail`` carries both
    sub-verdicts for persistence and post-mortem.

    CONCEPT:AHE-3.35
    """

    def __init__(self, *, min_lift: float = 0.02, min_drop_frac: float = 0.5) -> None:
        self.min_lift = min_lift
        self.min_drop_frac = min_drop_frac

    def evaluate(
        self,
        *,
        candidate_score: float,
        baseline_score: float,
        single_batch_loss_curve: Sequence[float],
        higher_is_better: bool = True,
    ) -> GateVerdict:
        """Run both sub-gates and combine them into one go/no-go verdict.

        Args:
            candidate_score: The candidate approach's score.
            baseline_score: The tuned baseline's score.
            single_batch_loss_curve: The single-batch overfit loss curve.
            higher_is_better: Metric direction for the baseline comparison.

        Returns:
            A :class:`GateVerdict` that ``passed`` only if both the baseline and
            overfit sub-gates pass; ``detail`` nests both sub-verdicts.
        """
        baseline = baseline_gate(
            candidate_score,
            baseline_score,
            min_lift=self.min_lift,
            higher_is_better=higher_is_better,
        )
        overfit = overfit_smoke_gate(
            single_batch_loss_curve,
            min_drop_frac=self.min_drop_frac,
        )
        passed = baseline.passed and overfit.passed
        if passed:
            reason = "go: candidate beats tuned baseline and model overfits one batch"
        elif not baseline.passed and not overfit.passed:
            reason = f"no-go: {baseline.reason}; and {overfit.reason}"
        elif not baseline.passed:
            reason = f"no-go (baseline): {baseline.reason}"
        else:
            reason = f"no-go (overfit): {overfit.reason}"
        return GateVerdict(
            passed=passed,
            reason=reason,
            detail={
                "baseline": {
                    "passed": baseline.passed,
                    "reason": baseline.reason,
                    "detail": baseline.detail,
                },
                "overfit": {
                    "passed": overfit.passed,
                    "reason": overfit.reason,
                    "detail": overfit.detail,
                },
            },
        )
