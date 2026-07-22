#!/usr/bin/python
from __future__ import annotations

"""SkillOpt-native ReflACT skill-markdown evolution cycle (CONCEPT:AU-AHE.optimization.skillopt-native-reflact).

Ground truth: SkillOpt (Microsoft, arXiv:2605.23904, ``open-source-libraries/SkillOpt``) treats
the skill markdown document as the ONLY trainable parameter of a frozen model and trains it
with a 6-stage per-step loop (``skillopt/engine/trainer.py::ReflACTTrainer``):

    ① Rollout   — execute episodes against the CURRENT skill
    ② Reflect   — turn failing episodes into failure patterns
    ③ Aggregate — merge patterns
    ④ Select    — budget-clip
    ⑤ Update    — apply edits to the skill document
    ⑥ Evaluate  — held-out validation gate: replace ONLY if the candidate beats the
                  incumbent (``skillopt/evaluation/gate.py::evaluate_gate``, ported here
                  as :func:`skill_gate.evaluate_promotion`)

This module is the AU-native port, wired into the SAME propose-only 24/7 flywheel every
other evolution family uses (``loop_controller.LoopController``), reusing:

* :class:`~agent_utilities.models.knowledge_graph.SkillVersionNode` — content-addressed,
  mirrors :class:`~agent_utilities.models.knowledge_graph.PromptVersionNode`'s lineage
  contract and :class:`~agent_utilities.models.knowledge_graph.ClaimNode`'s propose-only
  ``status`` contract.
* :class:`~agent_utilities.models.knowledge_graph.OutcomeEvaluationNode` — the SAME
  durable trajectory store :class:`~agent_utilities.harness.variant_pool.VariantPool`'s
  ``evaluate_fitness`` already reads back (no new database).
* ``agent_utilities.orchestration.action_policy`` — the SAME propose-and-veto gate every
  other mutating fleet action passes through, under the reserved
  ``kind="promote_skill_version"`` (shipped default: ``approval_required`` — a
  benchmark-winning candidate is NEVER auto-promoted out of the box, mirroring
  ``loop_controller._run_insight_validation``'s ``promote_mined_claim`` two-key
  discipline).

Pluggable signal interface (coordination note): Rollout/Evaluate/Reflect are built
against the abstract :class:`SkillSignalProvider` seam — NOT hardcoded to the internal
eval corpus — so a parallel "Native Langfuse" track can supply a Langfuse
dataset/score/trace-backed provider at the SAME three seams
(``train_tasks``/``holdout_tasks``, ``score``, ``failure_traces``) without touching the
cycle itself. :class:`InternalCorpusSignalProvider` is the zero-infra default shipped
here.

Foundation disclosure (see ``reports/skillopt-native-design.md`` §6 for the full list):
the default :func:`propose_revision` edit function and the default
:func:`default_static_executor` rollout executor are deterministic, non-LLM
placeholders — the real SkillOpt-parity LLM-driven Reflect/Update stage and a live
multi-backend rollout executor are follow-up work. Every seam here is an injection
point precisely so that follow-up work replaces a function, not this module's control
flow.
"""

import hashlib
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "SkillTask",
    "SkillTrajectory",
    "SkillSignalProvider",
    "InternalCorpusSignalProvider",
    "default_static_executor",
    "default_edit_fn",
    "run_rollout",
    "extract_failure_patterns",
    "propose_revision",
    "run_reflact_cycle",
    "record_transfer_score",
]


# ── ① types (SkillOpt's RolloutResult/RawPatch, ported minimally) ──────────────────


@dataclass(slots=True)
class SkillTask:
    """One eval-task item a skill is rolled out against.

    ``check`` is an optional ``Callable[[str], bool]`` pass/fail predicate over the
    rollout executor's output; when absent :class:`InternalCorpusSignalProvider` falls
    back to a non-empty-output check (never fabricates a pass).
    """

    id: str
    prompt: str
    check: Callable[[str], bool] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SkillTrajectory:
    """One Rollout episode's outcome — SkillOpt's ``RolloutResult``, ported."""

    task_id: str
    output: str
    success: bool
    score: float
    fail_reason: str = ""


# ── pluggable signal interface (coordination: Native Langfuse track) ───────────────


@runtime_checkable
class SkillSignalProvider(Protocol):
    """Abstract Rollout/Evaluate/Reflect signal source.

    Kept abstract on purpose: a "Native Langfuse" track builds a Langfuse
    dataset/score/trace-backed implementation of this SAME protocol (Langfuse
    datasets → ``train_tasks``/``holdout_tasks``; Langfuse scores → ``score``;
    Langfuse traces → ``failure_traces``) without touching :func:`run_reflact_cycle`.
    :class:`InternalCorpusSignalProvider` is the zero-infra default.
    """

    def train_tasks(self) -> list[SkillTask]:
        """The Rollout (①) minibatch — tasks the candidate is proposed against."""
        ...

    def holdout_tasks(self) -> list[SkillTask]:
        """The Evaluate (⑥) held-out set — NEVER seen by Aggregate/Select/Update."""
        ...

    def score(self, task: SkillTask, output: str) -> tuple[bool, float, str]:
        """Score one rollout output. Returns ``(success, score in [0,1], fail_reason)``."""
        ...

    def record_trajectory(
        self, skill_id: str, skill_version_id: str, trajectory: SkillTrajectory
    ) -> None:
        """Best-effort durable persistence of one rollout episode. Never raises."""
        ...

    def failure_traces(self, skill_id: str) -> list[dict[str, Any]]:
        """Prior failure traces available to the Reflect (②) stage (may be empty)."""
        ...


class InternalCorpusSignalProvider:
    """Zero-infra default :class:`SkillSignalProvider` — a plain task list.

    No LLM, no external eval harness: ``score`` runs the task's own ``check``
    predicate (or a non-empty-output fallback — never a fabricated pass). Rollout
    episodes ARE persisted to the shared durable store when ``engine`` is supplied —
    as :class:`~agent_utilities.models.knowledge_graph.OutcomeEvaluationNode`, the SAME
    node type :class:`~agent_utilities.harness.variant_pool.VariantPool`'s
    ``evaluate_fitness`` already reads back — so even the zero-infra default writes to
    the one shared trajectory store, not a private one.
    """

    def __init__(
        self,
        train: list[SkillTask],
        holdout: list[SkillTask],
        *,
        engine: Any = None,
    ) -> None:
        self._train = list(train)
        self._holdout = list(holdout)
        self.engine = engine

    def train_tasks(self) -> list[SkillTask]:
        return list(self._train)

    def holdout_tasks(self) -> list[SkillTask]:
        return list(self._holdout)

    def score(self, task: SkillTask, output: str) -> tuple[bool, float, str]:
        if callable(task.check):
            try:
                ok = bool(task.check(output))
            except Exception as e:  # noqa: BLE001 — a broken predicate scores as fail
                return False, 0.0, f"check_error: {e}"
            return ok, (1.0 if ok else 0.0), ("" if ok else "check_failed")
        ok = bool(output and output.strip())
        return ok, (1.0 if ok else 0.0), ("" if ok else "empty_output")

    def record_trajectory(
        self, skill_id: str, skill_version_id: str, trajectory: SkillTrajectory
    ) -> None:
        if self.engine is None:
            return
        try:
            from agent_utilities.models.knowledge_graph import OutcomeEvaluationNode

            node = OutcomeEvaluationNode(
                id=f"outcome:{skill_version_id}:{trajectory.task_id}",
                name=f"{skill_id}@{skill_version_id} / {trajectory.task_id}",
                reward=trajectory.score,
                success_criteria_met=(
                    [trajectory.task_id] if trajectory.success else []
                ),
                feedback_text=trajectory.fail_reason,
            )
            props = node.model_dump()
            props.pop("id", None)
            props["type"] = str(props.get("type", ""))
            self.engine.add_node(node.id, "outcome_evaluation", properties=props)
        except Exception as e:  # noqa: BLE001 — trajectory persistence is best-effort
            logger.debug("skill_evolution: record_trajectory failed: %s", e)

    def failure_traces(self, skill_id: str) -> list[dict[str, Any]]:
        # The internal provider has no separate trace store beyond the rollout
        # episodes themselves — a Langfuse-backed provider is expected to return
        # real prior traces here.
        return []


# ── ① Rollout ────────────────────────────────────────────────────────────────────


def default_static_executor(skill_content: str, task: SkillTask) -> str:  # noqa: ARG001
    """Foundation default rollout executor — a deterministic, zero-model placeholder.

    Treats the skill's OWN markdown as its rollout "output", so ``signal.score``
    becomes a deterministic completeness check ("does this skill's guidance satisfy
    the task's predicate") with zero model calls. A real executor that runs a target
    agent under the candidate skill (e.g. via ``graph_orchestrate execute_agent``) is
    follow-up work — see the design doc §6 disclosure. Kept as an injection point:
    every caller of :func:`run_rollout`/:func:`run_reflact_cycle` may pass its own
    ``executor: Callable[[str, SkillTask], str]``.
    """
    return skill_content


def run_rollout(
    skill_content: str,
    tasks: list[SkillTask],
    executor: Callable[[str, SkillTask], str],
    signal: SkillSignalProvider,
    *,
    skill_id: str = "",
    skill_version_id: str = "",
) -> list[SkillTrajectory]:
    """Stage ① — execute ``skill_content`` against every task, scored by ``signal``.

    Never raises: an executor failure scores as an empty-output trajectory rather
    than aborting the whole rollout (mirrors the reference's per-episode isolation —
    one failing episode never blocks the batch).
    """
    trajectories: list[SkillTrajectory] = []
    for task in tasks:
        try:
            output = executor(skill_content, task)
        except Exception as e:  # noqa: BLE001 — one bad episode never blocks the batch
            output = ""
            logger.debug("skill_evolution: executor failed for %s: %s", task.id, e)
        success, score, fail_reason = signal.score(task, output)
        traj = SkillTrajectory(
            task_id=task.id,
            output=output,
            success=success,
            score=score,
            fail_reason=fail_reason,
        )
        trajectories.append(traj)
        signal.record_trajectory(skill_id, skill_version_id, traj)
    return trajectories


def _mean_score(trajectories: list[SkillTrajectory]) -> float:
    if not trajectories:
        return 0.0
    return sum(t.score for t in trajectories) / len(trajectories)


# ── ② Reflect ────────────────────────────────────────────────────────────────────


def extract_failure_patterns(trajectories: list[SkillTrajectory]) -> list[str]:
    """Stage ② — group failing trajectories by ``fail_reason`` prefix.

    Minimal port of SkillOpt's ``trainer._extract_failure_patterns`` (grouping by
    ``fail_reason`` prefix); the reference's richer analyst-description enrichment
    (LLM-summarized failure descriptions) is not ported — see the design doc §6.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for t in trajectories:
        if t.success:
            continue
        reason = t.fail_reason or "unknown"
        prefix = reason.split(":", 1)[0].strip() or "unknown"
        groups[prefix].append(t.task_id)
    return [
        f"{reason} (×{len(ids)}: {', '.join(ids)})" for reason, ids in groups.items()
    ]


# ── ③④⑤ Aggregate / Select / Update ─────────────────────────────────────────────

_AUTO_SECTION_MARKER = "## Known failure patterns (auto-appended by skill evolution)"


def default_edit_fn(skill_content: str, patterns: list[str]) -> str:
    """Default Aggregate/Select/Update stage — a deterministic, non-LLM placeholder.

    Appends a bounded "Known failure patterns" section (never rewrites existing
    content — a conservative append-only textual edit, in the spirit of SkillOpt's
    bounded add/insert/replace/delete edit ops without needing an LLM to author it).
    Re-running strips the PRIOR auto-appended section first so repeated cycles don't
    accumulate duplicates. The real SkillOpt-parity LLM-driven proposer (mirroring
    ``rlm/gepa.py``'s ``ReflectiveMutator``) is follow-up work — see design doc §6.
    """
    if not patterns:
        return skill_content
    body = skill_content
    if _AUTO_SECTION_MARKER in body:
        head, _, _ = body.partition(_AUTO_SECTION_MARKER)
        body = head.rstrip()
    lines = "\n".join(f"- {p}" for p in patterns)
    return f"{body}\n\n{_AUTO_SECTION_MARKER}\n{lines}\n"


def propose_revision(
    skill_content: str,
    patterns: list[str],
    edit_fn: Callable[[str, list[str]], str] | None = None,
) -> str:
    """Stages ③④⑤ collapsed to one call — propose a candidate skill markdown.

    ``edit_fn`` is the injection point for a real LLM-driven proposer; defaults to
    :func:`default_edit_fn`.
    """
    fn = edit_fn or default_edit_fn
    return fn(skill_content, patterns)


# ── content addressing (mirrors PromptVersionNode.version_hash convention) ─────────


def version_hash(content: str) -> str:
    """Content-addressed hash — same convention as ``StructuredPrompt.version_hash``."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def skill_version_id(skill_id: str, content: str) -> str:
    return f"skill_version:{skill_id}:{version_hash(content)}"


# ── persistence (mirrors candidate_insight.py's engine.add_node convention) ────────


def _persist_skill_version(
    engine: Any,
    *,
    skill_id: str,
    candidate_id: str,
    content: str,
    parent_hash: str | None,
    status: str,
    origin: str,
    benchmark_score: float | None,
    benchmark_task_count: int,
    reflect_notes: list[str],
    errors: list[str],
) -> None:
    """Persist (or upsert) one ``:SkillVersion`` node. Best-effort; never raises."""
    if engine is None:
        return
    try:
        from agent_utilities.models.knowledge_graph import SkillVersionNode

        node = SkillVersionNode(
            id=candidate_id,
            name=f"{skill_id}@{candidate_id.rsplit(':', 1)[-1]}",
            skill_id=skill_id,
            version_hash=candidate_id.rsplit(":", 1)[-1],
            content=content,
            parent_hash=parent_hash,
            status=status,
            origin=origin,
            benchmark_score=benchmark_score,
            benchmark_task_count=benchmark_task_count,
            reflect_notes=list(reflect_notes),
        )
        props = node.model_dump()
        props.pop("id", None)
        props["type"] = str(props.get("type", ""))
        engine.add_node(node.id, "skill_version", properties=props)
    except Exception as e:  # noqa: BLE001 — persistence is best-effort
        errors.append(f"persist_skill_version {candidate_id}: {e}")


def record_transfer_score(
    engine: Any, skill_version_id_: str, model_id: str, score: float
) -> None:
    """Record a cross-model transfer score onto an existing ``:SkillVersion`` (§3.5).

    Wired + tested primitive: re-scores an existing version under a different target
    model and upserts ``transfer_scores[model_id]``. Does NOT sweep a model roster by
    itself — a caller (or a future orchestration stage) supplies the score, typically
    from re-running :func:`run_rollout` under a different ``executor``. Best-effort;
    never raises.
    """
    if engine is None:
        return
    try:
        get_node = getattr(engine, "get_node", None)
        existing: dict[str, Any] = {}
        if callable(get_node):
            existing = get_node(skill_version_id_) or {}
        transfer_scores = dict(existing.get("transfer_scores") or {})
        transfer_scores[model_id] = float(score)
        engine.add_node(
            skill_version_id_,
            "skill_version",
            properties={**existing, "transfer_scores": transfer_scores},
        )
    except Exception as e:  # noqa: BLE001 — best-effort audit overlay
        logger.debug("skill_evolution: record_transfer_score failed: %s", e)


# ── the full cycle ──────────────────────────────────────────────────────────────


def run_reflact_cycle(
    engine: Any,
    skill_id: str,
    incumbent_content: str,
    *,
    signal: SkillSignalProvider,
    executor: Callable[[str, SkillTask], str] | None = None,
    edit_fn: Callable[[str, list[str]], str] | None = None,
    incumbent_score: float | None = None,
) -> dict[str, Any]:
    """Run ONE ReflACT cycle for ``skill_id``. Returns a JSON-able report.

    ``Rollout(train) → Reflect → Aggregate/Select/Update → Evaluate(holdout) →
    skill_gate → action_policy.decide(kind="promote_skill_version") → promote``.
    Never raises — mirrors ``loop_controller._run_insight_validation``'s best-effort
    discipline: a failing sub-step degrades the report rather than crashing the
    caller's loop stage.

    Promotion contract (benchmark-gated, through the existing veto gate):

    1. A losing/tying candidate is persisted as ``status="rejected"`` and NEVER
       reaches ``action_policy`` — there is nothing to decide about a benchmark loss.
    2. A winning candidate is ALWAYS persisted as ``status="proposal"`` and ALWAYS
       run through ``action_policy.decide(kind="promote_skill_version")`` — the
       shipped default tier is ``approval_required``, so a benchmark win alone NEVER
       auto-promotes.
    3. Only when the action-policy decision is ``allow``/``allow_notify`` does the
       version flip to ``status="active"`` and gain a ``SUPERSEDES`` edge over the
       incumbent (the same edge-write convention
       ``candidate_insight.register_claim_materialization`` already uses for
       ``DERIVED_FROM``, and the same ``RegistryEdgeType.SUPERSEDES``
       :class:`~agent_utilities.harness.variant_pool.VariantPool.promote_winner`
       writes for a winning variant).
    """
    from agent_utilities.orchestration.action_policy import (
        ActionRequest,
        get_action_policy,
    )

    exec_fn = executor or default_static_executor
    report: dict[str, Any] = {
        "skill_id": skill_id,
        "train_n": 0,
        "holdout_n": 0,
        "failure_patterns": [],
        "candidate_version_id": None,
        "candidate_score": None,
        "incumbent_score": incumbent_score,
        "gate_action": None,
        "action_decision": None,
        "promoted": False,
        "errors": [],
    }

    try:
        train_tasks = signal.train_tasks()
        holdout_tasks = signal.holdout_tasks()
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"signal_tasks: {e}")
        return report
    report["train_n"] = len(train_tasks)
    report["holdout_n"] = len(holdout_tasks)

    incumbent_hash = version_hash(incumbent_content)
    incumbent_id = f"skill_version:{skill_id}:{incumbent_hash}"

    # ① Rollout (train)
    try:
        train_trajectories = run_rollout(
            incumbent_content,
            train_tasks,
            exec_fn,
            signal,
            skill_id=skill_id,
            skill_version_id=incumbent_id,
        )
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"rollout_train: {e}")
        train_trajectories = []

    # ② Reflect
    patterns = extract_failure_patterns(train_trajectories)
    report["failure_patterns"] = patterns
    if not patterns:
        report["gate_action"] = "skip_no_patches"
        return report

    # ③④⑤ Aggregate / Select / Update
    try:
        candidate_content = propose_revision(incumbent_content, patterns, edit_fn)
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"propose_revision: {e}")
        return report
    if candidate_content == incumbent_content:
        report["gate_action"] = "skip_no_change"
        return report

    candidate_hash = version_hash(candidate_content)
    candidate_id = f"skill_version:{skill_id}:{candidate_hash}"

    # ⑥ Evaluate — held-out
    try:
        if incumbent_score is None:
            incumbent_trajectories = run_rollout(
                incumbent_content,
                holdout_tasks,
                exec_fn,
                signal,
                skill_id=skill_id,
                skill_version_id=incumbent_id,
            )
            incumbent_score = _mean_score(incumbent_trajectories)
        candidate_trajectories = run_rollout(
            candidate_content,
            holdout_tasks,
            exec_fn,
            signal,
            skill_id=skill_id,
            skill_version_id=candidate_id,
        )
        candidate_score = _mean_score(candidate_trajectories)
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"evaluate: {e}")
        return report

    report["incumbent_score"] = incumbent_score
    report["candidate_score"] = candidate_score
    report["candidate_version_id"] = candidate_id

    from .skill_gate import evaluate_promotion

    wins = evaluate_promotion(candidate_score, incumbent_score)
    report["gate_action"] = "accept" if wins else "reject"

    _persist_skill_version(
        engine,
        skill_id=skill_id,
        candidate_id=candidate_id,
        content=candidate_content,
        parent_hash=incumbent_hash,
        status="proposal" if wins else "rejected",
        origin="reflact",
        benchmark_score=candidate_score,
        benchmark_task_count=len(holdout_tasks),
        reflect_notes=patterns,
        errors=report["errors"],
    )

    if not wins:
        # A benchmark loss never reaches action_policy — nothing to promote.
        return report

    action_policy = get_action_policy(engine)
    try:
        decision = action_policy.decide(
            ActionRequest(
                kind="promote_skill_version",
                target=skill_id,
                params={
                    "candidate_score": candidate_score,
                    "incumbent_score": incumbent_score,
                    "task_count": len(holdout_tasks),
                    "candidate_version_id": candidate_id,
                },
                source="loop_engine",
                reason=(
                    f"skill {skill_id} candidate {candidate_id} beats incumbent "
                    "on held-out eval"
                ),
            )
        )
    except Exception as e:  # noqa: BLE001 — fail closed, never crash
        report["errors"].append(f"action_policy: {e}")
        return report

    report["action_decision"] = decision.decision
    if decision.decision not in ("allow", "allow_notify"):
        return report

    # Two-key autonomy satisfied: benchmark win AND action-policy allow.
    try:
        if engine is not None:
            engine.add_edge(candidate_id, incumbent_id, relationship_type="SUPERSEDES")
        _persist_skill_version(
            engine,
            skill_id=skill_id,
            candidate_id=candidate_id,
            content=candidate_content,
            parent_hash=incumbent_hash,
            status="active",
            origin="reflact",
            benchmark_score=candidate_score,
            benchmark_task_count=len(holdout_tasks),
            reflect_notes=patterns,
            errors=report["errors"],
        )
        report["promoted"] = True
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"promote: {e}")

    return report
