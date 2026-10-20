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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

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


#: The served improvement-library query, driven by the host: the task class
#: in, the versions it returns out, best first. Each entry carries the
#: component kind and the component version reference of one version.
LibraryQuery = Callable[[str], Sequence[Mapping[str, Any]]]


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


def propose(
    *,
    task_ref: str,
    task_class: str,
    library: LibraryQuery,
    seed: str,
    model_ref: str,
    budget_ref: str,
    k: int,
) -> tuple[HarnessCandidate, ...]:
    """Build one candidate per version the improvement library returns.

    The library is the served improvement-library query (driven by the host);
    the cycle only reads its answer for ``task_class`` and stores nothing, so
    an empty answer refuses the tick with ``no_candidates``. At most ``k``
    candidates are kept, in the library's returned order.
    """
    versions = tuple(library(task_class))
    if not versions:
        raise CycleRefusal("no_candidates")
    candidates: list[HarnessCandidate] = []
    for index, version in enumerate(versions[:k]):
        candidates.append(
            HarnessCandidate(
                candidate_id=f"{task_ref}:candidate:{index}",
                task_ref=task_ref,
                component_kind=str(version["component_kind"]),
                component_version_ref=str(version["component_version_ref"]),
                seed=seed,
                model_ref=model_ref,
                budget_ref=budget_ref,
            )
        )
    return tuple(candidates)


__all__ = [
    "ABSTAIN_REASONS",
    "CycleOutcome",
    "CycleRefusal",
    "HarnessCandidate",
    "LibraryQuery",
    "propose",
]
