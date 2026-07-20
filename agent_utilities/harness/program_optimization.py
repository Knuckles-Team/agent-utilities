"""Unified native program optimization — metrics, targets, driver, and evidence.

CONCEPT:AU-AHE.optimization.real-optimization-metric — Real optimization metric (replaces the exact-match placeholder).
CONCEPT:AU-AHE.optimization.optimizable-target-registry — optimizable-target registry.
CONCEPT:AU-AHE.optimization.few-shot-demo-set — Few-shot demo-set refinement.

The epistemic-graph ``ProgramOptimize`` job compiles governed program revisions from
typed training references and promotion evidence. This module owns the Agent Utilities
target registry, deterministic evaluation helpers, and the single engine dispatch.

It provides a **graded metric** built on the existing :class:`EvalRunner` scorer (with a
  dependency-free token-overlap fallback, so it runs offline) — optionally blended with
  the capability reward EMA; an **optimizable-target registry** with one handler per
  :class:`ComponentType`
  (system prompt, MCP tool description, agent skill) declaring how to load the artifact's
  text and name it; and a **driver** that submits one governed native job. Candidate
  promotion remains evidence-gated and propose-only at this layer.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .manifest import ComponentType

logger = logging.getLogger(__name__)

# A program metric is ``(example, prediction, trace=None) -> float | bool``.
OptimizationMetric = Callable[..., float]


# --------------------------------------------------------------------------- #
# CONCEPT:AU-AHE.optimization.real-optimization-metric — the real optimization metric
# --------------------------------------------------------------------------- #
def graded_score(expected: str, actual: str) -> float:
    """Graded [0, 1] similarity between ``expected`` and ``actual`` text.

    Reuses the existing :meth:`EvalRunner._semantic_similarity_eval` (embedding cosine
    when an embedder is reachable, token-overlap otherwise) so the metric degrades to a
    deterministic offline scorer instead of the brittle exact-match it replaces.
    """
    try:
        from .continuous_evaluation_engine import EvalRunner

        return float(EvalRunner._semantic_similarity_eval(expected or "", actual or ""))
    except Exception:  # noqa: BLE001 - the metric must never break a compile
        # Last-ditch token-overlap so the metric is always callable.
        e = set((expected or "").lower().split())
        a = set((actual or "").lower().split())
        if not e:
            return 1.0 if not a else 0.0
        return len(e & a) / len(e | a) if (e | a) else 0.0


def make_optimization_metric(
    *,
    threshold: float = 0.7,
    reward_fn: Callable[[Any], float] | None = None,
    reward_weight: float = 0.0,
    return_bool: bool = False,
) -> OptimizationMetric:
    """Build the graded program metric.

    The metric grades ``prediction.response`` against ``example.response`` via
    :func:`graded_score`, optionally blending a per-example ``reward_fn`` (e.g. the
    capability reward EMA for the artifact under optimization) weighted by
    ``reward_weight``. Returns a float in [0, 1], or a bool (``score >= threshold``) when
    ``return_bool`` is set.

    This is the single metric every text target reuses; pass a ``reward_fn`` to fold a
    live outcome signal into the optimization objective.
    """

    def metric(example: Any, prediction: Any, trace: Any = None) -> float:
        expected = getattr(example, "response", "") or ""
        actual = getattr(prediction, "response", "") or ""
        score = graded_score(expected, actual)
        if reward_fn is not None and reward_weight > 0.0:
            try:
                r = float(reward_fn(example))
            except Exception:  # noqa: BLE001
                r = 0.5
            score = (1.0 - reward_weight) * score + reward_weight * r
        if return_bool:
            return score >= threshold
        return score

    return metric


# --------------------------------------------------------------------------- #
# CONCEPT:AU-AHE.optimization.optimizable-target-registry — optimizable-target registry
# --------------------------------------------------------------------------- #
@dataclass
class OptimizationResult:
    """Reference-only metadata produced by native program optimization."""

    component_type: str
    artifact_ref: str
    compiled_state: dict[str, Any] = field(default_factory=dict)
    demonstration_refs: list[str] = field(default_factory=list)
    trainset_size: int = 0
    optimizer: str = ""
    confidence: float = 0.0
    # KG-trace-derived training examples blended into the native corpus.
    # how much of the trainset was real KG provenance vs self-supervised filler, so a caller/reviewer can see the closed loop worked.
    trace_derived_count: int = 0
    trace_failure_count: int = 0


@dataclass
class OptimizableTarget:
    """One native-program-optimizable artifact type.

    A handler declares, for a :class:`ComponentType`, how to (1) read the artifact's
    optimizable *text* from its file and (2) name it. The uniform program signature is
    ``context``/``task`` → ``response``, so a new target is data, not a new code path. The
    *apply side* (text → source file) is handled downstream by
    ``PhysicalDistillationEngine.distill_*`` and the GitOps committer.
    """

    component_type: str
    description: str
    load_text: Callable[[dict[str, Any]], str]
    task_name: Callable[[dict[str, Any]], str]
    kg_label: str


def _blueprint_text(bp: dict[str, Any]) -> str:
    """Instruction text of a system-prompt JSON blueprint (identity + instructions)."""
    parts: list[str] = []
    iden = bp.get("identity")
    if isinstance(iden, dict):
        parts.append("\n".join(f"{k}: {v}" for k, v in iden.items()))
    elif iden:
        parts.append(str(iden))
    inst = bp.get("instructions")
    if isinstance(inst, dict):
        parts.append("\n".join(str(v) for v in inst.values()))
    elif inst:
        parts.append(str(inst))
    text = "\n\n".join(p for p in parts if p).strip()
    return text or bp.get("metadata", {}).get("description", "")


# The built-in registry — real handlers at import (never an empty shell), mirroring the
# ontology registries' idiom. Keys are ComponentType *values* (StrEnum), so a caller can
# look a target up by either the enum or its string.
OPTIMIZABLE_TARGETS: dict[str, OptimizableTarget] = {
    "system_prompt": OptimizableTarget(
        component_type="system_prompt",
        description="An agent system-prompt blueprint.",
        load_text=_blueprint_text,
        task_name=lambda bp: str(bp.get("task", "agent")),
        kg_label="EvolvedPromptNode",
    ),
    # MCP tool descriptions are native optimizable targets.
    "tool_description": OptimizableTarget(
        component_type="tool_description",
        description="An MCP tool's LLM-facing description.",
        load_text=lambda a: str(a.get("description") or a.get("docstring") or ""),
        task_name=lambda a: str(a.get("name") or a.get("tool") or "tool"),
        kg_label="EvolvedToolDescriptionNode",
    ),
    # Agent skill SOP/description as a native optimizable target; the SOP
    # already reaches the model via mount_skill_unit (ORCH-1.28), apply side distill_skill.
    "skill": OptimizableTarget(
        component_type="skill",
        description="An agent skill's SOP / description.",
        load_text=lambda a: str(a.get("sop") or a.get("description") or ""),
        task_name=lambda a: str(a.get("name") or "skill"),
        kg_label="EvolvedSkillNode",
    ),
}


def get_target(component_type: ComponentType | str) -> OptimizableTarget | None:
    """Return the registered target handler for a component type (or None)."""
    key = getattr(component_type, "value", component_type)
    return OPTIMIZABLE_TARGETS.get(str(key))


# --------------------------------------------------------------------------- #
# CONCEPT:AU-AHE.optimization.few-shot-demo-set — few-shot demo-set refinement
# --------------------------------------------------------------------------- #
def run_program_optimization(
    target: OptimizableTarget,
    artifact: dict[str, Any],
    trainset: list[Any],
    *,
    holdout_fraction: float = 0.3,
    engine: Any = None,
) -> OptimizationResult | None:
    """Optimize a target through the sole native ``eg-program`` contract.

    CONCEPT:AU-AHE.optimization.trace-derived-training-examples — closing the cohesive-loop gap: before compiling, ``trainset``
    (the caller's self-supervised examples — may be empty) is BLENDED with real
    KG-trace-derived examples for this exact target
    (:func:`agent_utilities.harness.trace_examples.blend_trainset`) — recent
    ``Episode``/``ToolCall``/``OutcomeEvaluation`` provenance attributed to this
    prompt/tool/skill, success and failure alike. A prompt/tool/skill with FAILED
    traces now feeds the optimizer a real, KG-observed negative example (blank
    response, ``reward`` below threshold, ``failure_reason`` carrying WHY), and the
    native objective consumes that real reward, so "traces observed" becomes
    "training signal used" without a second code path.
    ``engine`` is optional and DEFAULT-ON: when omitted, :func:`trace_examples.
    gather_trace_examples` best-effort resolves the process-wide active KG engine
    instead, so this degrades to self-supervised-only rather than needing every
    caller to thread an engine through.

    Missing capability, invalid response, or native execution failure fails closed.
    """
    from agent_utilities.harness.trace_examples import (
        blend_trainset,
        record_trace_derived_finding,
    )

    blended, trace_stats = blend_trainset(engine, target, artifact, trainset)
    if not blended:
        return None
    from agent_utilities.harness.optimization_backend import (
        OptimizationRequest,
        try_native_optimization,
    )

    def invoke_native() -> OptimizationResult | None:
        serializable_examples: list[dict[str, Any]] = []
        for example in blended:
            if isinstance(example, dict):
                serializable_examples.append(dict(example))
                continue
            to_dict = getattr(example, "toDict", None)
            if callable(to_dict):
                candidate = to_dict()
                if isinstance(candidate, dict):
                    serializable_examples.append(dict(candidate))
                    continue
            serializable_examples.append(
                {
                    key: getattr(example, key)
                    for key in (
                        "context",
                        "task",
                        "response",
                        "reward",
                        "failure_reason",
                    )
                    if hasattr(example, key)
                }
            )
        native_artifact = {
            key: value for key, value in artifact.items() if key != "__file_path__"
        }
        request = OptimizationRequest(
            target=target.component_type,
            objective=OPTIMIZATION_TARGETS_META.get(target.component_type, {}).get(
                "metric", "graded held-out score"
            ),
            data={
                "artifact": native_artifact,
                "trainset": serializable_examples,
                "holdout_fraction": holdout_fraction,
            },
        )
        attempt = try_native_optimization(engine, request)
        if attempt.disposition == "completed":
            result_payload = attempt.payload["result"]
            if not isinstance(result_payload, dict):
                logger.error("eg-program returned an invalid optimization result")
                return None
            rows = result_payload.get("rows")
            if not isinstance(rows, list):
                logger.error("eg-program returned an invalid candidate shape")
                return None
            candidates = [
                row
                for row in rows
                if isinstance(row, dict) and row.get("kind") == "program_candidate"
            ]
            selected_candidates = [
                row for row in candidates if row.get("selected") is True
            ]
            if len(selected_candidates) != 1:
                logger.error("eg-program returned an invalid candidate selection")
                return None
            selected = selected_candidates[0]
            demonstration_refs = list(selected.get("demonstration_refs", []))
            compiled_state = {
                key: selected.get(key)
                for key in (
                    "id",
                    "program_ref",
                    "optimizer",
                    "execution",
                    "candidate_role",
                    "demonstration_refs",
                    "artifact_refs",
                    "composition_refs",
                    "instruction_ref",
                    "tool_policy_ref",
                    "model_profile_ref",
                    "evidence_refs",
                    "source_refs",
                    "proof_ids",
                    "contradiction_ids",
                    "modalities",
                )
            }
            from agent_utilities.prompting.structured import ProgramCompiledState

            compiled_state = ProgramCompiledState.model_validate(
                compiled_state
            ).model_dump()
            record_trace_derived_finding(engine, trace_stats)
            return OptimizationResult(
                component_type=target.component_type,
                artifact_ref=str(selected["id"]),
                compiled_state=compiled_state,
                demonstration_refs=demonstration_refs,
                trainset_size=len(blended),
                optimizer="eg-program",
                confidence=float(selected.get("confidence", 0.0)),
                trace_derived_count=trace_stats["trace_derived"],
                trace_failure_count=trace_stats["trace_failures"],
            )
        error_code = attempt.error_code or f"native_{attempt.disposition}"
        logger.error("eg-program optimization failed: %s", error_code)
        return None

    return invoke_native()


# --------------------------------------------------------------------------- #
# CONCEPT:AU-AHE.optimization.optimizable-target-registry — one entry point for every optimizable target
# --------------------------------------------------------------------------- #
# The metric + data source + driver for each target, so the optimize surface (MCP
# `graph_evolution action=optimize_component` + REST twin) is one dispatch, not six.
OPTIMIZATION_TARGETS_META: dict[str, dict[str, str]] = {
    "system_prompt": {
        "metric": "graded eval-corpus pass-rate",
        "driver": "evolution cycle (failure clusters)",
    },
    "tool_description": {
        "metric": "tool-selection accuracy (capability reward EMA)",
        "driver": "evolution cycle (failure clusters)",
    },
    "skill": {
        "metric": "skill-invocation reliability",
        "driver": "evolution cycle (failure clusters)",
    },
    "extraction": {
        "metric": "self-supervised dedup + canonical consistency",
        "driver": "optimize_extraction_prompt(documents)",
    },
    "concept_match": {
        "metric": "classification accuracy vs ADDRESSES edges",
        "driver": "optimize_concept_matcher(labeled_pairs)",
    },
    "routing": {
        "metric": "realized execution success",
        "driver": "optimize_routing_policy(traces)",
    },
}


def run_component_optimization(
    target_name: str,
    data: dict[str, Any] | None = None,
    *,
    engine: Any = None,
) -> dict[str, Any]:
    """Dispatch the native optimization pass for any registered target.

    The single reusable entry point both surfaces call — the scheduled
    optimization tick (over every :data:`SCHEDULABLE_TARGETS` via
    :func:`run_optimization_sweep`) AND the on-demand ``graph_evolution
    action=optimize_component``. The provider-free engine is the sole backend.
    """
    data = data or {}
    meta = OPTIMIZATION_TARGETS_META.get(target_name)
    if meta is None:
        return {
            "error": f"unknown optimization target: {target_name!r}",
            "targets": sorted(OPTIMIZATION_TARGETS_META),
        }

    from agent_utilities.harness.optimization_backend import (
        OptimizationRequest,
        try_native_optimization,
    )

    report: dict[str, Any] = {"target": target_name, "backend": "eg-program", **meta}
    started = time.monotonic()
    attempt = try_native_optimization(
        engine,
        OptimizationRequest(
            target=target_name,
            objective=meta["metric"],
            data=data,
            optimizer=str(data.get("optimizer") or "bootstrap_few_shot"),
        ),
    )
    report["duration_s"] = round(time.monotonic() - started, 3)
    if attempt.disposition == "completed":
        report.update(
            status=attempt.payload["status"], result=attempt.payload["result"]
        )
        logger.info(
            "optimization: target=%s status=%s duration=%.2fs",
            target_name,
            report["status"],
            report["duration_s"],
        )
        return report

    report.update(
        status="error",
        error_code=attempt.error_code or f"native_{attempt.disposition}",
    )
    logger.error(
        "optimization: target=%s failed after %.2fs: %s",
        target_name,
        report["duration_s"],
        report["error_code"],
    )
    return report


# --------------------------------------------------------------------------- #
# CONCEPT:AU-AHE.optimization.candidate-replaces-incumbent-only — scheduled optimization sweep (the daemon-tick twin)
# --------------------------------------------------------------------------- #
# The self-supervised targets the daemon can run unattended (the registry targets —
# system_prompt/tool_description/skill — are driven by the failure-cluster evolution
# cycle, not this sweep).
SCHEDULABLE_TARGETS: tuple[str, ...] = ("extraction", "concept_match", "routing")


def should_promote(
    baseline_score: float, candidate_score: float, *, min_delta: float = 0.0
) -> bool:
    """Promotion gate (CONCEPT:AU-AHE.optimization.candidate-replaces-incumbent-only): a candidate replaces the incumbent only when it
    beats it on the held-out metric by at least ``min_delta``. The criterion the sweep
    applies before an optimized artifact is allowed to supersede the live one."""
    return candidate_score >= baseline_score + min_delta


# --------------------------------------------------------------------------- #
# CONCEPT:AU-AHE.optimization.prompt-hardening-cycle — the prompt-hardening cycle (optimize → evaluate → propose)
# --------------------------------------------------------------------------- #
# The system-prompt leg persists only native compiled references. Demonstration content is
# resolved from the governed corpus for one evaluation context, then discarded. The
# candidate is scored against the agent's eval-corpus slice and a promote/reject decision
# is returned. The *apply* of the winning candidate is gated by
# ``EvolveAgent.apply_edits`` + ``KG_AGENT_AUTO_APPLY`` — this module only decides; it
# never writes source.


def _demo_fields(demo: Any) -> tuple[str, str, str]:
    """Extract ``(context, task, response)`` from a training row."""
    if isinstance(demo, dict):
        get = demo.get
    else:
        get = lambda k, d="": getattr(demo, k, d)  # noqa: E731 - tiny adapter
    context = str(get("context", "") or "")
    task = str(get("task", "") or get("query", "") or "")
    response = str(get("response", "") or get("expected_output", "") or "")
    return context, task, response


def render_program_prompt_for_execution(
    baseline: Any,
    demos: Sequence[Any],
    *,
    max_demos: int = 4,
) -> str:
    """Resolve examples into one ephemeral model context.

    Raw tasks and responses exist only in the returned execution string. Prompt
    blueprints, proposals, graph nodes, optimization results, and reports retain
    only the governed references in :class:`ProgramCompiledState`.
    """
    from agent_utilities.prompting.structured import render_ephemeral_demonstrations

    exemplars: list[dict[str, str]] = []
    for demo in list(demos)[:max_demos]:
        _ctx, task, response = _demo_fields(demo)
        if not (task or response):
            continue
        exemplars.append({"task": task, "response": response})
    return render_ephemeral_demonstrations(baseline.render(), exemplars)


def _bump_patch(version: str | None) -> str:
    """Bump the patch field of a ``major.minor.patch`` semver (default ``0.0.1``)."""
    if not version:
        return "0.0.1"
    parts = version.split(".")
    try:
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    except (ValueError, IndexError):
        return f"{version}+hardened"


def score_prompt_against_corpus(prompt_text: str, cases: Sequence[Any]) -> float:
    """Mean graded overlap of a prompt body with its eval-corpus expected outputs.

    CONCEPT:AU-AHE.optimization.prompt-hardening-cycle. The offline-deterministic proxy for "does this prompt embed the
    behavior the corpus rewards": each case's ``expected_output`` is scored against the
    rendered prompt via :func:`graded_score` (the same semantic scorer the program metric
    uses). A prompt that has folded in exemplars whose responses match the corpus scores
    strictly higher than the bare baseline — so the metric moves monotonically with real
    coverage, and a candidate enriched with *irrelevant* demos cannot beat baseline.
    """
    scored: list[float] = []
    for case in cases:
        expected = str(getattr(case, "expected_output", "") or "")
        if not expected:
            continue
        scored.append(graded_score(expected, prompt_text))
    return sum(scored) / len(scored) if scored else 0.0


@dataclass
class PromptHardeningOutcome:
    """The audit record of one prompt-hardening cycle (CONCEPT:AU-AHE.optimization.prompt-hardening-cycle).

    Carries content-safe review metadata: opaque agent/component references, the
    before/after metric, the promote decision, whether it was actually applied (vs held in
    shadow), and the candidate's content hash. ``status`` is one of ``no_data`` (no
    per-agent corpus), ``error`` (native or governed-store failure), ``rejected`` (did not
    beat baseline), ``proposed`` (beat baseline but auto-apply gated off — shadow), or
    ``applied`` (written to source under the gate).
    """

    agent_ref: str
    component_ref: str
    baseline_score: float = 0.0
    candidate_score: float = 0.0
    promote: bool = False
    applied: bool = False
    status: str = "no_data"
    trainset_size: int = 0
    optimizer: str = ""
    candidate_version_hash: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_ref": self.agent_ref,
            "component_ref": self.component_ref,
            "baseline_score": round(self.baseline_score, 4),
            "candidate_score": round(self.candidate_score, 4),
            "delta": round(self.candidate_score - self.baseline_score, 4),
            "promote": self.promote,
            "applied": self.applied,
            "status": self.status,
            "trainset_size": self.trainset_size,
            "optimizer": self.optimizer,
            "candidate_version_hash": self.candidate_version_hash,
            "detail": self.detail,
        }


def gather_optimization_data(
    engine: Any, target: str, *, limit: int = 50
) -> dict[str, Any]:
    """Best-effort production data for a self-supervised target (CONCEPT:AU-AHE.optimization.candidate-replaces-incumbent-only).

    Reads the live graph via ``engine.query_cypher`` for the data each optimizer needs —
    recent ``Document`` text (extraction), ``ADDRESSED_BY``-labeled concept/source pairs
    (concept_match), or ``ExecutionTrace`` rows (routing). Returns ``{}`` on any failure or
    when the engine/data is absent, so the sweep degrades to ``no_data`` rather than
    breaking the daemon.
    """
    if engine is None or not hasattr(engine, "query_cypher"):
        return {}

    def _rows(cypher: str) -> list[dict[str, Any]]:
        try:
            res = engine.query_cypher(cypher)
            return list(res) if res else []
        except Exception:  # noqa: BLE001 - data gathering is best-effort
            return []

    if target == "extraction":
        rows = _rows(
            f"MATCH (d:Document) WHERE d.content IS NOT NULL "
            f"RETURN d.content AS content LIMIT {limit}"
        )
        return {"documents": [str(r.get("content")) for r in rows if r.get("content")]}
    if target == "concept_match":
        rows = _rows(
            f"MATCH (c:Concept)-[:ADDRESSED_BY]->(s) "
            f"RETURN c.name AS concept, coalesce(s.content, s.name) AS article "
            f"LIMIT {limit}"
        )
        positives = [
            (str(r.get("article")), str(r.get("concept")), True)
            for r in rows
            if r.get("article") and r.get("concept")
        ]
        # Synthesize negatives by pairing each concept with a neighbour's article.
        negatives = (
            [
                (positives[(i + 1) % len(positives)][0], positives[i][1], False)
                for i in range(len(positives))
            ]
            if len(positives) > 1
            else []
        )
        return {"labeled_pairs": positives + negatives}
    if target == "routing":
        rows = _rows(
            f"MATCH (t:ExecutionTrace) "
            f"RETURN t.task_text AS task_text, t.primitive_used AS primitive_used, "
            f"t.success AS success LIMIT {limit}"
        )
        return {"traces": rows}
    return {}


def run_optimization_sweep(
    engine: Any = None, targets: Sequence[str] | None = None
) -> dict[str, Any]:
    """Propose-only optimization sweep over the schedulable targets (CONCEPT:AU-AHE.optimization.candidate-replaces-incumbent-only).

    The reusable core the daemon tick (``KG_OPTIMIZATION_ENABLED``) and the on-demand
    ``optimize_component task=all`` surface both call. For each target it gathers live
    data (:func:`gather_optimization_data`) and runs :func:`run_component_optimization`.
    It is **propose-only**: the optimizers persist optimization trajectories to the KG
    but nothing is auto-applied to source — promotion stays behind :func:`should_promote`
    and a future auto-apply gate (mirroring ``KG_GOLDEN_AUTO_MERGE``). Returns a
    per-target report (with ``optimized``/``failed`` target lists + ``duration_s``);
    never raises, but failures are logged loudly, not swallowed silently.
    """
    names = list(targets) if targets else list(SCHEDULABLE_TARGETS)
    t0 = time.monotonic()
    logger.info("optimization sweep: start targets=%s", names)
    report: dict[str, Any] = {}
    optimized: list[str] = []
    failed: list[str] = []
    for name in names:
        data = gather_optimization_data(engine, name)
        result = run_component_optimization(name, data, engine=engine)
        report[name] = result
        if result.get("status") in {"optimized", "proposed"}:
            optimized.append(name)
        elif result.get("status") == "error":
            failed.append(name)
    duration = time.monotonic() - t0

    if failed:
        logger.error(
            "optimization sweep: FAILURES targets=%s (optimized=%s, duration=%.2fs)",
            failed,
            optimized,
            duration,
        )
    else:
        logger.info(
            "optimization sweep: done optimized=%s duration=%.2fs",
            optimized,
            duration,
        )
    return {
        "targets": report,
        "optimized": optimized,
        "failed": failed,
        "propose_only": True,
        "duration_s": round(duration, 3),
    }
