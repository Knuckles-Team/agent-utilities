"""EH-407: guardrail profile evolution proposed through EG ``Decide``.

The question: given one throttled cell's recorded history (EG keeps the last
AIMD steps on the cell), should its guardrail profile hold, move one step
tighter or one step looser along the operator's declared ladder? It is a
``policy`` question, so EG never explores it, and the answer is only a
PROPOSAL: tightening inside the declared bounds is applied by the evolution
service, loosening always waits for a human approval
(:mod:`agent_utilities.security.guardrail_evolution`). Only the moves the
ladder allows are offered, so no answer can leave the declared bounds.

The deterministic fallback is conservative: tighten after repeated
error-budget narrowings, loosen only after a long run of healthy windows at
the full declared capacity, otherwise hold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_utilities import decide
from agent_utilities.decide.options import Option
from agent_utilities.security.guardrail_profile import ProfileMove

QUESTION_ID = "au.guardrail.profile"
#: Narrowings in the recorded history that make the fallback tighten.
TIGHTEN_AFTER_NARROWINGS = 3
#: Counted healthy windows (and no narrowing) before the fallback proposes loosening.
LOOSEN_AFTER_HEALTHY_WINDOWS = 16
_HEALTHY = frozenset({"healthy_window", "within_budget", "at_declared_ceiling"})
_UNCOUNTED = frozenset({"insufficient_samples"})


@dataclass(frozen=True, slots=True)
class ThrottleEvidence:
    """What one cell's recorded throttle history says."""

    level: int
    levels: int
    windows: int
    narrowed: int
    healthy: int
    at_capacity: bool

    @classmethod
    def from_cell(
        cls, cell: Mapping[str, Any], level: int, levels: int
    ) -> ThrottleEvidence:
        throttle = cell.get("throttle") or {}
        history: Sequence[Mapping[str, Any]] = throttle.get("history") or ()
        counted = [step for step in history if step.get("reason") not in _UNCOUNTED]
        return cls(
            level=level,
            levels=levels,
            windows=len(counted),
            narrowed=sum(1 for step in counted if step.get("action") == "narrowed"),
            healthy=sum(1 for step in counted if step.get("reason") in _HEALTHY),
            at_capacity=int(throttle.get("ceiling", -1))
            == int(cell.get("capacity", 0)),
        )

    def legal_moves(self) -> list[ProfileMove]:
        moves = [ProfileMove.HOLD]
        if self.level < self.levels - 1:
            moves.append(ProfileMove.TIGHTEN)
        if self.level > 0:
            moves.append(ProfileMove.LOOSEN)
        return moves

    def features(self) -> dict[str, float]:
        windows = float(max(self.windows, 1))
        return {
            "narrowed_fraction": self.narrowed / windows,
            "healthy_fraction": self.healthy / windows,
            "windows": float(self.windows),
            "at_capacity": 1.0 if self.at_capacity else 0.0,
            "tightness": self.level / float(max(self.levels - 1, 1)),
        }


def fallback_move(evidence: ThrottleEvidence) -> ProfileMove:
    """The deterministic rule EG's answer is measured against."""
    legal = evidence.legal_moves()
    if evidence.narrowed >= TIGHTEN_AFTER_NARROWINGS:
        wanted = ProfileMove.TIGHTEN
    elif (
        evidence.narrowed == 0
        and evidence.windows >= LOOSEN_AFTER_HEALTHY_WINDOWS
        and evidence.healthy == evidence.windows
        and evidence.at_capacity
    ):
        wanted = ProfileMove.LOOSEN
    else:
        wanted = ProfileMove.HOLD
    return wanted if wanted in legal else ProfileMove.HOLD


def _options(evidence: ThrottleEvidence, heuristic: ProfileMove) -> list[Option]:
    facts = evidence.features()
    return [
        Option(
            move.value,
            {
                **facts,
                "step": {"hold": 0.0, "tighten": 1.0, "loosen": -1.0}[move.value],
                "heuristic": 1.0 if move is heuristic else 0.0,
            },
        )
        for move in evidence.legal_moves()
    ]


async def propose_move(evidence: ThrottleEvidence) -> tuple[ProfileMove, decide.Choice]:
    """The proposed move for one cell and the choice that proposed it."""
    heuristic = fallback_move(evidence)
    choice = await decide.achoose(
        QUESTION_ID, _options(evidence, heuristic), lambda: heuristic.value
    )
    legal = {move.value: move for move in evidence.legal_moves()}
    return legal.get(str(choice.option_id), ProfileMove.HOLD), choice


__all__ = [
    "LOOSEN_AFTER_HEALTHY_WINDOWS",
    "QUESTION_ID",
    "TIGHTEN_AFTER_NARROWINGS",
    "ThrottleEvidence",
    "fallback_move",
    "propose_move",
]
