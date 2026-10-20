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

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from agent_utilities.graph.runner import execute_graph

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


#: The probe id of the current-harness probe (the contrast's run a).
CURRENT_PROBE_ID = "current"


@dataclass(frozen=True)
class ProbeTrace:
    """One probe under the shared conditions: its trace and its outcome.

    ``trace_id`` is the probe run's run id; the host records the run's events
    as the trace. ``conditions_digest`` binds the shared task, seed, model and
    budget references every probe of the tick ran under.
    """

    candidate_id: str
    trace_id: str
    conditions_digest: str
    status: str
    steps: int
    usage: Mapping[str, int]
    error: str | None


def conditions_digest(*, task: str, seed: str, model_ref: str, budget_ref: str) -> str:
    """The digest that binds the shared probe conditions.

    Every probe of one tick runs the same task with the same seed, model
    reference and budget reference; the digest of those four values is what
    the probe traces carry and what the engine checks against.
    """
    payload = json.dumps(
        {"task": task, "seed": seed, "model_ref": model_ref, "budget_ref": budget_ref},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def probe_candidates(
    *,
    graph: Any,
    config: Mapping[str, Any] | None,
    task: str,
    seed: str,
    model_ref: str,
    budget_ref: str,
    run_id: str,
    current_version_ref: str,
    candidates: Sequence[HarnessCandidate],
    mode: str = "ask",
    agent_name: str = "",
) -> tuple[ProbeTrace, ...]:
    """Probe the current harness and each candidate under one shared condition.

    Every probe enters through the run loop with the same task, seed, model
    reference and budget reference; the conditions digest binds them, and each
    probe records one trace under its own run id. The probe writes no graph
    state. A probe whose run produces no trace refuses the tick with
    ``probe_trace_missing``.
    """
    digest = conditions_digest(
        task=task, seed=seed, model_ref=model_ref, budget_ref=budget_ref
    )
    entries: list[tuple[str, str]] = []
    if current_version_ref:
        entries.append((CURRENT_PROBE_ID, current_version_ref))
    entries.extend(
        (candidate.candidate_id, candidate.component_version_ref)
        for candidate in candidates
    )
    traces: list[ProbeTrace] = []
    for candidate_id, version_ref in entries:
        trace_id = f"{run_id}:probe:{candidate_id}"
        try:
            result = await execute_graph(
                graph,
                config,
                task,
                run_id=trace_id,
                mode=mode,
                requested_model_id=model_ref,
                agent_name=agent_name,
                metadata={
                    "harness_probe": {
                        "candidate_id": candidate_id,
                        "component_version_ref": version_ref,
                        "seed": seed,
                    }
                },
            )
        except Exception:
            raise CycleRefusal("probe_trace_missing") from None
        if not isinstance(result, Mapping) or result.get("run_id") != trace_id:
            raise CycleRefusal("probe_trace_missing")
        usage = result.get("usage") or {}
        traces.append(
            ProbeTrace(
                candidate_id=candidate_id,
                trace_id=trace_id,
                conditions_digest=digest,
                status=str(result.get("status") or ""),
                steps=int(result.get("steps") or 0),
                usage={str(key): int(value) for key, value in dict(usage).items()},
                error=None if result.get("error") is None else str(result["error"]),
            )
        )
    return tuple(traces)


__all__ = [
    "ABSTAIN_REASONS",
    "CURRENT_PROBE_ID",
    "CycleOutcome",
    "CycleRefusal",
    "HarnessCandidate",
    "LibraryQuery",
    "ProbeTrace",
    "conditions_digest",
    "probe_candidates",
    "propose",
]
