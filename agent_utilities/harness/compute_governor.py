#!/usr/bin/python
from __future__ import annotations

"""Value-aware test-time-compute governor.

CONCEPT:AU-OS.scaling.value-test-time-compute — a value-aware test-time-compute governor that decides whether spending more reasoning compute is worth it by stopping once a result satisfices or the marginal expected quality gain per extra attempt falls below a threshold

The paper (§5.1) notes naive brute-force search "fails in virtually all non-toy
domains" — gains come from search *efficiency*, spending compute where marginal return
is highest. AU could scale test-time compute (diverse best-of-k, multi-paradigm
reasoning) but always spent a *fixed* amount, never allocating it value-optimally. This
governor adds the missing controller: given the results seen so far in an iterative
test-time search (e.g. the KG-2.68 router trying paradigms in reward order), it answers
"is another attempt worth it?" — stopping when the best result already satisfices, or
when the recent marginal quality gain per attempt has dropped below a floor, or a hard
cap is hit. Pure and stateless; the router's adaptive mode consumes it.
"""

from dataclasses import dataclass


@dataclass
class ComputeGovernor:
    """Decide whether to spend another test-time-compute attempt.

    Args:
        satisfice: Stop once the best score so far reaches this (no point continuing).
        min_marginal_gain: Stop when the best score improved by less than this over the
            previous attempt (diminishing returns).
        max_attempts: Hard cap on attempts regardless of marginal gain.
    """

    satisfice: float = 0.95
    min_marginal_gain: float = 0.02
    max_attempts: int = 4

    def should_continue(self, scores: list[float]) -> bool:
        """True if another attempt is worth it, given the scores seen so far.

        ``scores`` are the per-attempt result scores in the order produced. An empty
        list always continues (we have tried nothing yet).
        """
        if not scores:
            return True
        if len(scores) >= self.max_attempts:
            return False
        best = max(scores)
        if best >= self.satisfice:
            return False
        # Diminishing returns: did the most recent attempt meaningfully raise the best?
        prev_best = max(scores[:-1]) if len(scores) > 1 else 0.0
        marginal = max(scores) - prev_best
        return marginal >= self.min_marginal_gain

    def report(self, scores: list[float]) -> dict[str, float | int | bool]:
        """A small explanation of the stop decision (for audit/observability)."""
        best = max(scores) if scores else 0.0
        return {
            "attempts": len(scores),
            "best": round(best, 4),
            "satisficed": best >= self.satisfice,
            "at_cap": len(scores) >= self.max_attempts,
            "continue": self.should_continue(scores),
        }
