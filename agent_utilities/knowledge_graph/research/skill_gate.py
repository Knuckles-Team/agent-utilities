#!/usr/bin/python
from __future__ import annotations

"""Benchmark-gated promotion decision (CONCEPT:AU-AHE.optimization.skillopt-native-reflact).

The AU-native port of SkillOpt's (arXiv:2605.23904) ``skillopt/evaluation/gate.py::
evaluate_gate`` — a **pure decision function**, deliberately kept side-effect-free
exactly like the reference (the reference's own module docstring: "The trainer owns
side-effects ... This module is the pure decision function"). ``skill_evolution.py``
owns every side effect (persistence, action-policy consultation, promotion); this
module only answers "does the candidate beat the incumbent."

Ported algorithm (unchanged from the reference): a candidate is only accepted when it
STRICTLY improves the held-out score — ``cand_score > current_score``. A tie never
promotes (avoids churning a live skill on evaluation noise), matching the reference's
own comparison exactly.
"""

from dataclasses import dataclass
from typing import Literal

__all__ = ["GateAction", "SkillGateResult", "evaluate_promotion", "skill_gate"]

GateAction = Literal["accept_new_best", "accept", "reject"]


@dataclass(frozen=True)
class SkillGateResult:
    """Immutable outcome of the benchmark gate (mirrors SkillOpt's ``GateResult``)."""

    action: GateAction
    candidate_score: float
    incumbent_score: float
    best_score: float


def evaluate_promotion(candidate_score: float, incumbent_score: float) -> bool:
    """``True`` iff ``candidate_score`` strictly beats ``incumbent_score``.

    Direct port of SkillOpt's ``evaluate_gate``'s comparison
    (``cand_score > current_score``). This is the ONE line the whole benchmark-gated
    promotion contract rests on: a losing or tying candidate never promotes, no matter
    what ``action_policy`` would otherwise allow.
    """
    return float(candidate_score) > float(incumbent_score)


def skill_gate(
    candidate_score: float, incumbent_score: float, best_score: float | None = None
) -> SkillGateResult:
    """Full three-way gate — accept_new_best / accept / reject.

    Mirrors SkillOpt's ``evaluate_gate`` three-way action exactly: a candidate that
    beats the incumbent AND is the best score seen so far is ``accept_new_best``; one
    that beats the incumbent but not the running best is ``accept``; anything else is
    ``reject``. ``best_score`` defaults to ``incumbent_score`` (single-generation
    cycles have no separate running-best tracker yet — see the design doc §6).
    """
    best = incumbent_score if best_score is None else best_score
    if evaluate_promotion(candidate_score, incumbent_score):
        action: GateAction = "accept_new_best" if candidate_score > best else "accept"
    else:
        action = "reject"
    return SkillGateResult(
        action=action,
        candidate_score=float(candidate_score),
        incumbent_score=float(incumbent_score),
        best_score=float(best),
    )
