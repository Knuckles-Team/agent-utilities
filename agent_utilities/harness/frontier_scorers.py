#!/usr/bin/python
from __future__ import annotations

"""Non-saturating, superhuman-range progress signals.

CONCEPT:SAFE-1.1 — non-saturating superhuman progress tracking via relative scorers and a saturation detector that keep producing signal past the human or known-answer ceiling so a genuine capability jump is distinguishable from metric saturation

Every fixed-target benchmark (LongMemEval-S, the 1-5 quality gates, the frozen eval
corpus) clamps at a human-expert / known-answer ceiling — useless for telling two
superhuman systems apart, and unable to distinguish "real capability jump" from
"the metric saturated". This module adds signals that keep producing range past
that ceiling:

* ``CompressionScorer`` — a relative description-length / information-density score
  (an MDL-flavoured proxy for intelligence: more answer in fewer tokens). It is
  *informational* (always ``passed``) so it adds a non-saturating metric to the
  reliability suite without changing any guardrail's pass/fail.
* ``elo_from_duels`` — relative Elo-style ratings from agent-vs-agent duel outcomes
  (reusing the Bradley-Terry kernel); a ranking has no ceiling, so it separates
  superhuman agents the absolute scorers cannot.
* ``saturation_detector`` — flags an eval whose pass-rate has collapsed to the
  ceiling across recent agent versions and recommends promoting it to a relative
  scorer.

The live wiring is ``CompressionScorer`` in the reliability suite; the relative
ranking + saturation detector are the frontier surface a benchmark "frontier mode"
and the AHE-3.26 velocity ledger consume.
"""

from typing import Any

from agent_utilities.tools.eval_harness import EvalHarness, EvalResult


class CompressionScorer:
    """Informational relative score: answer information-density (description length).

    Rewards conveying the output in fewer characters than the input/context — a
    finite, ceiling-free MDL proxy. Always ``passed`` (never a guardrail); it only
    contributes a non-saturating ``compression_ratio`` metric.
    """

    name: str = "compression"

    def score(
        self, input_text: str, output_text: str, context: dict[str, Any] | None = None
    ) -> EvalResult:
        in_len = len(input_text or "")
        out_len = len(output_text or "")
        # Density: how much shorter the answer is than its input, in [0, 1].
        ratio = 0.0 if in_len == 0 else max(0.0, 1.0 - out_len / in_len)
        # A non-empty, non-trivial answer that compresses scores higher.
        score = ratio if out_len > 0 else 0.0
        return EvalResult(
            score=round(score, 4),
            passed=True,  # informational — never gates
            evaluator=self.name,
            metrics={
                "input_len": in_len,
                "output_len": out_len,
                "compression_ratio": round(ratio, 4),
            },
            reason="information-density (relative, non-saturating)",
        )


def elo_from_duels(
    duels: list[tuple[str, str]], *, iters: int = 200
) -> dict[str, float]:
    """Relative ratings from ``(winner, loser)`` agent-vs-agent duels.

    Reuses the Bradley-Terry kernel (`selection_operators`). A ranking has no
    human ceiling, so it keeps separating agents past benchmark saturation.
    Returns ``{agent_id: rating}`` (higher is stronger); ``{}`` for no duels.
    """
    if not duels:
        return {}
    from agent_utilities.harness.selection_operators import bradley_terry_scores

    items = sorted({a for pair in duels for a in pair})
    return bradley_terry_scores(items, list(duels), iters=iters)


def setter_solver_gap(solver_solved: int, setter_total: int) -> float:
    """Frontier difficulty the setter achieved: ``1 − solver_pass_rate``.

    In a setter-solver loop the setter generates problems at the solver's edge; the
    fraction the solver *fails* is a ceiling-free difficulty signal that rises as the
    solver improves (the setter must work harder). ``0`` when there are no problems.
    """
    if setter_total <= 0:
        return 0.0
    return max(0.0, 1.0 - solver_solved / setter_total)


def saturation_detector(
    pass_rates: list[float], *, ceiling: float = 0.98, window: int = 3
) -> dict[str, Any]:
    """Flag an eval whose recent pass-rate has collapsed to the ceiling.

    When the last ``window`` agent-version pass-rates are all ≥ ``ceiling`` the eval
    no longer distinguishes capability — recommend promoting it to a relative
    (frontier) scorer. Returns ``{saturated, recent_mean, window, reason}``.
    """
    recent = [float(r) for r in pass_rates][-window:]
    if len(recent) < window:
        return {
            "saturated": False,
            "recent_mean": round(sum(recent) / len(recent), 4) if recent else 0.0,
            "window": window,
            "reason": "insufficient history",
        }
    saturated = all(r >= ceiling for r in recent)
    return {
        "saturated": saturated,
        "recent_mean": round(sum(recent) / len(recent), 4),
        "window": window,
        "reason": (
            f"pass-rate ≥ {ceiling} across last {window} versions — promote to a "
            "relative/frontier scorer"
            if saturated
            else "still discriminating"
        ),
    }


def build_frontier_suite(scorers: list[Any] | None = None) -> EvalHarness:
    """An :class:`EvalHarness` of the non-saturating frontier scorers."""
    harness = EvalHarness()
    for scorer in scorers if scorers is not None else [CompressionScorer()]:
        harness.register(scorer)
    return harness
