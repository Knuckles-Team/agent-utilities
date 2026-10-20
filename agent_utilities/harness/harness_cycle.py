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
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from agent_utilities.graph.runner import execute_graph

from .edit_engine import Edit, apply_edits

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


@dataclass(frozen=True)
class ContrastDelta:
    """One candidate's measured contrast against the current harness's probe.

    The deltas are measurements the engine decides on: retry, tool
    selection, eval score and cost, each the candidate probe minus the
    current-harness probe. A positive eval score delta is the improvement
    the engine adopts on.
    """

    candidate_id: str
    run_a_trace_id: str
    run_b_trace_id: str
    retry_delta: int
    tool_selection_delta: int
    eval_score_delta: int
    cost_delta: int


@dataclass(frozen=True)
class ContrastRequest:
    """The payload of one served contrast decision call per tick."""

    task_ref: str
    task_class: str
    conditions_digest: str
    candidate_version_refs: tuple[str, ...]
    probe_trace_ids: tuple[str, ...]
    contrasts: tuple[ContrastDelta, ...]


@dataclass(frozen=True)
class ContrastAnswer:
    """The engine's answer: one adoption or one abstention, never both.

    ``abstain_reason`` is the engine's closed vocabulary, passed through.
    On an adoption ``adopted_component_kind`` and ``adopted_content`` carry
    the adopted component's kind and content for the in-run switch, the
    content resolved by the host from the graph.
    """

    adopted_version_ref: str | None = None
    abstain_reason: str | None = None
    adopted_component_kind: str = ""
    adopted_content: str = ""


#: The served contrast decision, driven by the host: one Decide call under
#: the contrast question kind per tick; the answer may be awaitable.
ContrastDecision = Callable[[ContrastRequest], Any]


def _usage_of(trace: ProbeTrace, key: str) -> int:
    return int(trace.usage.get(key, 0))


def _score_of(trace: ProbeTrace) -> int:
    """One probe's eval score: the run completed without an error, or not."""
    return 1 if trace.status == "completed" and trace.error is None else 0


def measure_contrasts(
    *,
    baseline: ProbeTrace | None,
    candidates: Sequence[HarnessCandidate],
    traces: Mapping[str, ProbeTrace],
) -> tuple[ContrastDelta, ...]:
    """Each candidate's measured contrast against the current harness's probe.

    Measurement only: the deltas the engine decides on are computed here and
    kept out of any ranking; without a current-harness probe the deltas
    measure against the empty baseline.
    """
    deltas: list[ContrastDelta] = []
    for candidate in candidates:
        trace = traces[candidate.candidate_id]
        if baseline is None:
            retry_delta = _usage_of(trace, "requests")
            tool_selection_delta = _usage_of(trace, "tool_calls")
            eval_score_delta = _score_of(trace)
            cost_delta = _usage_of(trace, "total_tokens")
            run_a = ""
        else:
            retry_delta = _usage_of(trace, "requests") - _usage_of(baseline, "requests")
            tool_selection_delta = _usage_of(trace, "tool_calls") - _usage_of(
                baseline, "tool_calls"
            )
            eval_score_delta = _score_of(trace) - _score_of(baseline)
            cost_delta = _usage_of(trace, "total_tokens") - _usage_of(
                baseline, "total_tokens"
            )
            run_a = baseline.trace_id
        deltas.append(
            ContrastDelta(
                candidate_id=candidate.candidate_id,
                run_a_trace_id=run_a,
                run_b_trace_id=trace.trace_id,
                retry_delta=retry_delta,
                tool_selection_delta=tool_selection_delta,
                eval_score_delta=eval_score_delta,
                cost_delta=cost_delta,
            )
        )
    return tuple(deltas)


def _field(value: Any, name: str) -> Any:
    """One duck-typed field of an answer the engine types itself."""
    return getattr(value, name, None)


def _normalize_answer(answer: Any) -> ContrastAnswer:
    if isinstance(answer, ContrastAnswer):
        normalized = answer
    elif isinstance(answer, Mapping):
        normalized = ContrastAnswer(
            adopted_version_ref=None
            if answer.get("adopted_version_ref") is None
            else str(answer["adopted_version_ref"]),
            abstain_reason=None
            if answer.get("abstain_reason") is None
            else str(answer["abstain_reason"]),
            adopted_component_kind=str(answer.get("adopted_component_kind") or ""),
            adopted_content=str(answer.get("adopted_content") or ""),
        )
    else:
        normalized = ContrastAnswer(
            adopted_version_ref=None
            if _field(answer, "adopted_version_ref") is None
            else str(_field(answer, "adopted_version_ref")),
            abstain_reason=None
            if _field(answer, "abstain_reason") is None
            else str(_field(answer, "abstain_reason")),
            adopted_component_kind=str(_field(answer, "adopted_component_kind") or ""),
            adopted_content=str(_field(answer, "adopted_content") or ""),
        )
    if normalized.adopted_version_ref is None and normalized.abstain_reason is None:
        raise ValueError(
            "the contrast decision answered with neither adoption nor abstention"
        )
    if (
        normalized.adopted_version_ref is not None
        and normalized.abstain_reason is not None
    ):
        raise ValueError(
            "the contrast decision answered with an adoption and an abstention"
        )
    return normalized


async def compose(
    *,
    task_ref: str,
    task_class: str,
    conditions_digest_value: str,
    candidates: Sequence[HarnessCandidate],
    baseline: ProbeTrace | None,
    traces: Mapping[str, ProbeTrace],
    decide: ContrastDecision,
) -> ContrastAnswer:
    """One served Decide call under the contrast question kind.

    Sends the probe traces and the measured contrasts; the engine answer is
    the adoption or the abstention. The cycle applies no local ranking: it
    sends, and the engine's answer stands. An engine abstention refuses the
    tick with ``compose_abstention``.
    """
    request = ContrastRequest(
        task_ref=task_ref,
        task_class=task_class,
        conditions_digest=conditions_digest_value,
        candidate_version_refs=tuple(
            candidate.component_version_ref for candidate in candidates
        ),
        probe_trace_ids=tuple(
            traces[candidate.candidate_id].trace_id for candidate in candidates
        ),
        contrasts=measure_contrasts(
            baseline=baseline, candidates=candidates, traces=traces
        ),
    )
    answer = decide(request)
    if inspect.isawaitable(answer):
        answer = await answer
    normalized = _normalize_answer(answer)
    if normalized.abstain_reason is not None:
        raise CycleRefusal("compose_abstention")
    return normalized


#: The generated-client envelope writer, driven by the host: one graph-fact
#: envelope in, the written fact's reference out.
EnvelopeWriter = Callable[[Mapping[str, Any]], str]


def _component_file(component_kind: str) -> str:
    """The in-run harness's own component file under the workspace root."""
    return f"harness/{component_kind}.md"


def persist(
    *,
    task_ref: str,
    task_class: str,
    conditions_digest_value: str,
    adopted: HarnessCandidate,
    contrast: ContrastDelta,
    answer: ContrastAnswer,
    write_envelope: EnvelopeWriter,
    root: str | Path = ".",
) -> tuple[str, str, str]:
    """Write the adopted version, the contrast and the lesson, then switch.

    Three graph-fact envelopes through the host's generated-client writer:
    the adopted component version, the measured contrast of the adoption
    and the lesson for the next task, which cites the evidence runs by
    trace id. The in-run harness switch reuses :func:`apply_edits`: the
    adopted component's content replaces the current content of the
    harness's own component file. The step writes no other state.
    """
    write_envelope(
        {
            "kind": "harness_component_version",
            "fact": {
                "task_ref": task_ref,
                "component_kind": adopted.component_kind,
                "component_version_ref": adopted.component_version_ref,
                "evidence_run_id": contrast.run_b_trace_id,
            },
        }
    )
    contrast_ref = write_envelope(
        {
            "kind": "harness_contrast",
            "fact": {
                "task_ref": task_ref,
                "conditions_digest": conditions_digest_value,
                "run_a_trace_id": contrast.run_a_trace_id,
                "run_b_trace_id": contrast.run_b_trace_id,
                "retry_delta": contrast.retry_delta,
                "tool_selection_delta": contrast.tool_selection_delta,
                "eval_score_delta": contrast.eval_score_delta,
                "cost_delta": contrast.cost_delta,
            },
        }
    )
    lesson_ref = write_envelope(
        {
            "kind": "harness_lesson",
            "fact": {
                "task_class": task_class,
                "changed_component_kind": adopted.component_kind,
                "what_changed": (
                    f"{adopted.component_kind} switched to "
                    f"{adopted.component_version_ref}"
                ),
                "evidence_run_ids": [contrast.run_a_trace_id, contrast.run_b_trace_id],
            },
        }
    )
    component_path = _component_file(adopted.component_kind)
    current_path = Path(root) / component_path
    current = current_path.read_text(encoding="utf-8") if current_path.exists() else ""
    result = apply_edits(
        [Edit(path=component_path, search=current, replace=answer.adopted_content)],
        root,
    )
    if not result.ok:
        raise RuntimeError(
            f"the in-run harness switch failed: {result.failures[0].reason}"
        )
    if answer.adopted_version_ref is None:
        raise ValueError("persist is called on an adoption only")
    return answer.adopted_version_ref, contrast_ref, lesson_ref


__all__ = [
    "ABSTAIN_REASONS",
    "CURRENT_PROBE_ID",
    "ContrastAnswer",
    "ContrastDecision",
    "ContrastDelta",
    "ContrastRequest",
    "CycleOutcome",
    "CycleRefusal",
    "EnvelopeWriter",
    "HarnessCandidate",
    "LibraryQuery",
    "ProbeTrace",
    "conditions_digest",
    "compose",
    "measure_contrasts",
    "persist",
    "probe_candidates",
    "propose",
]
