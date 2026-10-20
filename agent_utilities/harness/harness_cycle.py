"""The propose–probe–compose–persist–reflect harness evolution cycle.

CONCEPT:AU-ORCH.execution.harness-evolution-cycle

For one task tick the harness proposes k candidate component versions from
the served improvement library, probes each under identical conditions
through the run loop, composes the adopted harness through the served
contrast decision, persists the version, the contrast and the lesson as
graph facts, and reflects the lesson for the next task.

The module holds no store, keeps no ranking and serves no operation. Every
capability it reaches is a host-injected port: the improvement-library
query, the contrast decision and the envelope writer are registered by the
host (the component versions, the lesson library and the decision live in
epistemic-graph). The candidate budget k is the candidate slot count of the
stage allocation the host carries on the run plan budget; below two slots
the cycle abstains with ``budget_below_contrast`` and keeps the current
harness. The measured contrast is the authority: the cycle never ranks
candidates of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

#: The closed refusal codes of one tick. Anything else is not a refusal.
ABSTAIN_REASONS: Final[tuple[str, ...]] = (
    "budget_below_contrast",
    "no_candidates",
    "probe_trace_missing",
    "compose_abstention",
)


class CycleRefusal(Exception):
    """The cycle abstained with a typed code; the current harness is kept."""

    def __init__(self, code: str) -> None:
        if code not in ABSTAIN_REASONS:
            raise ValueError(f"open refusal code: {code!r}")
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class HarnessCandidate:
    """One proposed candidate component version for one task.

    Carries references, not content: the component version reference is the
    graph fact the host resolves, and the shared seed, model and budget
    references bind the probe conditions.
    """

    candidate_id: str
    task_ref: str
    component_kind: str
    component_version_ref: str
    seed: str
    model_ref: str
    budget_ref: str


@dataclass(frozen=True)
class CycleOutcome:
    """The outcome of one cycle tick.

    ``abstain_reason`` is the closed set :data:`ABSTAIN_REASONS`; an abstained
    tick adopts nothing and carries no contrast or lesson reference.
    """

    task_ref: str
    k: int
    adopted_version_ref: str | None
    abstain_reason: str | None
    contrast_ref: str | None
    lesson_ref: str | None

    def __post_init__(self) -> None:
        if (
            self.abstain_reason is not None
            and self.abstain_reason not in ABSTAIN_REASONS
        ):
            raise ValueError(f"open refusal code: {self.abstain_reason!r}")
        if self.adopted_version_ref is not None and self.abstain_reason is not None:
            raise ValueError("an abstained tick adopts nothing")


__all__ = [
    "ABSTAIN_REASONS",
    "CycleOutcome",
    "CycleRefusal",
    "HarnessCandidate",
]
