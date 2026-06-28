from __future__ import annotations

"""Unified DSPy optimization subsystem — metric · targets · driver · demo refinement.

CONCEPT:AHE-3.39 — Real optimization metric (replaces the exact-match placeholder).
CONCEPT:AHE-3.40 — Optimizable-target registry + generalized DSPy driver.
CONCEPT:AHE-3.43 — Few-shot demo-set refinement.

DSPy optimizes anything expressible as a ``Signature`` (typed in→out) + a *metric* +
a *trainset*. The original wiring (`EvolveAgent._dspy_optimize_cluster`) was hardcoded
to one target (system prompts) and one metric (exact string match). This module is the
spine that generalizes both:

* a **graded metric** built on the existing :class:`EvalRunner` semantic scorer (with a
  dependency-free token-overlap fallback, so it runs offline) — optionally blended with
  the capability reward EMA — so optimization is steered by *quality*, not literal equality;
* an **optimizable-target registry**: one handler per :class:`ComponentType`
  (system prompt, MCP tool description, agent skill) declaring how to load the artifact's
  text, build its Signature, and persist the result — so a new target is a handler, not a
  new code path;
* a **driver** that compiles a target with any DSPy optimizer and then **refines the
  bootstrapped demo set** (drop-one ablation scored on a held-out slice) so a noisy demo
  cannot survive into the blueprint.

The non-prompt targets' *apply side* already exists (``PhysicalDistillationEngine``); this
supplies the *optimize side* they were missing.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .manifest import ComponentType

logger = logging.getLogger(__name__)

# A DSPy metric is ``(example, prediction, trace=None) -> float | bool``.
DspyMetric = Callable[..., float]


# --------------------------------------------------------------------------- #
# CONCEPT:AHE-3.39 — the real optimization metric
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
) -> DspyMetric:
    """Build a DSPy-compatible metric (CONCEPT:AHE-3.39).

    The metric grades ``prediction.response`` against ``example.response`` via
    :func:`graded_score`, optionally blending a per-example ``reward_fn`` (e.g. the
    capability reward EMA for the artifact under optimization) weighted by
    ``reward_weight``. Returns a float in [0, 1], or a bool (``score >= threshold``) when
    ``return_bool`` is set — both shapes DSPy optimizers accept.

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
# CONCEPT:AHE-3.40 — optimizable-target registry
# --------------------------------------------------------------------------- #
@dataclass
class OptimizationResult:
    """The artifacts a target optimization produced."""

    component_type: str
    file_path: str
    compiled_state: dict[str, Any] = field(default_factory=dict)
    demos: list[dict[str, Any]] = field(default_factory=list)
    optimized_instruction: str = ""
    trainset_size: int = 0
    optimizer: str = ""


@dataclass
class OptimizableTarget:
    """One DSPy-optimizable artifact type (CONCEPT:AHE-3.40).

    A handler declares, for a :class:`ComponentType`, how to (1) read the artifact's
    optimizable *text* from its file, (2) name it, and (3) write the compiled state back.
    The Signature is uniform (``context``, ``task`` → ``response`` with the artifact text
    as the instruction docstring) so a new target is data, not a new code path. The
    *apply side* (text → source file) is handled downstream by
    ``PhysicalDistillationEngine.distill_*`` and the GitOps committer.
    """

    component_type: str
    description: str
    load_text: Callable[[dict[str, Any]], str]
    task_name: Callable[[dict[str, Any]], str]
    kg_label: str

    def build_signature(self, artifact: dict[str, Any]) -> type:
        """Build a DSPy Signature whose instruction docstring is the artifact text."""
        import dspy

        text = self.load_text(artifact) or self.description
        name = self.task_name(artifact)

        class GeneratedSignature(dspy.Signature):
            context: str = dspy.InputField(desc="Context, history, and state.")
            task: str = dspy.InputField(desc="The task or query to satisfy.")
            response: str = dspy.OutputField(desc="The response or action payload.")

        GeneratedSignature.__doc__ = text
        safe = f"Signature_{self.component_type}_{name}".replace("-", "_").replace(
            " ", "_"
        )
        GeneratedSignature.__name__ = safe
        GeneratedSignature.__qualname__ = safe
        return GeneratedSignature


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
    # CONCEPT:AHE-3.41 — MCP tool descriptions become a DSPy-optimizable target (was
    # only heuristic-edited); apply side is PhysicalDistillationEngine.distill_mcp_tool.
    "tool_description": OptimizableTarget(
        component_type="tool_description",
        description="An MCP tool's LLM-facing description.",
        load_text=lambda a: str(a.get("description") or a.get("docstring") or ""),
        task_name=lambda a: str(a.get("name") or a.get("tool") or "tool"),
        kg_label="EvolvedToolDescriptionNode",
    ),
    # CONCEPT:AHE-3.42 — agent skill SOP/description as a DSPy-optimizable target; the SOP
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
# CONCEPT:AHE-3.43 — few-shot demo-set refinement
# --------------------------------------------------------------------------- #
def refine_demos(
    program: Any,
    holdout: Sequence[Any],
    metric: DspyMetric,
    *,
    min_demos: int = 1,
) -> list[Any]:
    """Drop-one demo ablation: keep only demos that don't hurt held-out score.

    CONCEPT:AHE-3.43. ``BootstrapFewShot`` selects demos but never re-checks them; a noisy
    demo then survives into the blueprint. This evaluates the program on ``holdout``,
    removes each demo whose removal does **not** lower the mean score, and returns the
    pruned demo list (never below ``min_demos``). Best-effort: any failure returns the
    original demos unchanged.
    """
    demos = list(getattr(program, "demos", []) or [])
    if len(demos) <= min_demos or not holdout:
        return demos

    def mean_score(demo_set: list[Any]) -> float:
        program.demos = demo_set
        total = 0.0
        n = 0
        for ex in holdout:
            try:
                pred = program(
                    context=getattr(ex, "context", ""),
                    task=getattr(ex, "task", ""),
                )
                total += float(metric(ex, pred))
                n += 1
            except Exception:  # noqa: BLE001 - a failed demo scores as worst
                n += 1
        return total / n if n else 0.0

    try:
        baseline = mean_score(demos)
        kept = list(demos)
        for demo in demos:
            if len(kept) <= min_demos:
                break
            trial = [d for d in kept if d is not demo]
            if mean_score(trial) >= baseline:
                kept = trial  # demo was dead weight or harmful — drop it
                baseline = mean_score(kept)
        program.demos = kept
        return kept
    except Exception as e:  # noqa: BLE001
        logger.debug("refine_demos failed, keeping original demos: %s", e)
        program.demos = demos
        return demos


# --------------------------------------------------------------------------- #
# CONCEPT:AHE-3.40 — the generalized driver
# --------------------------------------------------------------------------- #
def build_optimizer(optimizer_name: str, metric: DspyMetric) -> Any:
    """Construct a DSPy teleprompter by name, degrading to BootstrapFewShot.

    Centralizes the optimizer selection the original `_dspy_optimize_cluster` inlined, so
    every target shares one well-behaved fallback ladder (MIPROv2 →
    BootstrapFewShotWithRandomSearch → BootstrapFewShot).
    """
    from dspy.teleprompt import BootstrapFewShot

    if optimizer_name == "MIPROv2":
        try:
            from dspy.teleprompt import MIPROv2

            return MIPROv2(metric=metric, num_candidates=5, init_temperature=1.0)
        except ImportError:
            logger.warning("MIPROv2 unavailable; using BootstrapFewShot.")
    elif optimizer_name == "BootstrapFewShotWithRandomSearch":
        try:
            from dspy.teleprompt import BootstrapFewShotWithRandomSearch

            return BootstrapFewShotWithRandomSearch(
                metric=metric, max_bootstrapped_demos=3, num_candidate_programs=5
            )
        except ImportError:
            logger.warning(
                "BootstrapFewShotWithRandomSearch unavailable; using BootstrapFewShot."
            )
    return BootstrapFewShot(
        metric=metric, max_bootstrapped_demos=3, max_labeled_demos=3
    )


def run_dspy_optimization(
    target: OptimizableTarget,
    artifact: dict[str, Any],
    trainset: list[Any],
    *,
    metric: DspyMetric | None = None,
    optimizer_name: str = "BootstrapFewShot",
    holdout_fraction: float = 0.3,
) -> OptimizationResult | None:
    """Compile a target with DSPy, refine its demos, and return the result.

    CONCEPT:AHE-3.40/3.43. Splits ``trainset`` into a feedback set (used by the optimizer)
    and a held-out set (used by :func:`refine_demos`), compiles the target's Signature with
    the selected optimizer under the real metric, prunes the bootstrapped demos, and
    returns an :class:`OptimizationResult` (compiled state + kept demos). Returns ``None``
    when DSPy is unavailable or the trainset is empty. Never raises.
    """
    if not trainset:
        return None
    try:
        from agent_utilities.prompting.dspy_compiler import AgentTaskModule

        metric = metric or make_optimization_metric(return_bool=True)
        signature = target.build_signature(artifact)
        program = AgentTaskModule(signature)

        split = max(1, int(len(trainset) * (1.0 - holdout_fraction)))
        feedback, holdout = trainset[:split], trainset[split:]

        optimizer = build_optimizer(optimizer_name, metric)
        compiled = optimizer.compile(program, trainset=feedback)

        kept = (
            refine_demos(compiled, holdout, metric)
            if holdout
            else getattr(compiled, "demos", [])
        )

        return OptimizationResult(
            component_type=target.component_type,
            file_path=str(artifact.get("__file_path__", "")),
            compiled_state=compiled.dump_state(),
            demos=[d.toDict() if hasattr(d, "toDict") else dict(d) for d in kept],
            optimized_instruction=str(getattr(signature, "__doc__", "") or ""),
            trainset_size=len(trainset),
            optimizer=optimizer_name,
        )
    except Exception as e:  # noqa: BLE001 - optimization is best-effort
        logger.warning(
            "run_dspy_optimization failed for %s: %s", target.component_type, e
        )
        return None


# --------------------------------------------------------------------------- #
# CONCEPT:AHE-3.40 — one entry point for every optimizable target
# --------------------------------------------------------------------------- #
# The metric + data source + driver for each target, so the optimize surface (MCP
# `graph_orchestrate action=optimize_component` + REST twin) is one dispatch, not six.
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
    target_name: str, data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Dispatch a DSPy optimization pass for any target (CONCEPT:AHE-3.40).

    The single reusable entry point both surfaces call. For the registry targets
    (system_prompt / tool_description / skill) optimization is driven by the evolution
    cycle over real failure clusters, so this returns their metric/driver descriptor. For
    the self-supervised targets (extraction / concept_match / routing) it invokes the
    optimizer with ``data`` (documents / labeled_pairs / traces). Returns a JSON-able
    report; never raises.
    """
    data = data or {}
    meta = OPTIMIZATION_TARGETS_META.get(target_name)
    if meta is None:
        return {
            "error": f"unknown optimization target: {target_name!r}",
            "targets": sorted(OPTIMIZATION_TARGETS_META),
        }

    report: dict[str, Any] = {"target": target_name, **meta}
    try:
        if target_name in ("system_prompt", "tool_description", "skill"):
            report["status"] = "registered"
            report["note"] = (
                "Optimized by the evolution cycle when a failure cluster is attributed "
                "to this component; run via graph_orchestrate action='evolve' / the daemon."
            )
        elif target_name == "extraction":
            from agent_utilities.knowledge_graph.extraction.extraction_optimizer import (
                optimize_extraction_prompt,
            )

            res = optimize_extraction_prompt(data.get("documents", []) or [])
            report["status"] = "optimized" if res else "no_data_or_dspy_unavailable"
            report["result"] = res
        elif target_name == "concept_match":
            from agent_utilities.harness.policy_optimization import (
                optimize_concept_matcher,
            )

            res = optimize_concept_matcher(data.get("labeled_pairs", []) or [])
            report["status"] = "optimized" if res else "no_data_or_dspy_unavailable"
            report["result"] = res
        elif target_name == "routing":
            from agent_utilities.harness.policy_optimization import (
                optimize_routing_policy,
            )

            res = optimize_routing_policy(data.get("traces", []) or [])
            report["status"] = "optimized" if res else "no_data_or_dspy_unavailable"
            report["result"] = res
    except Exception as e:  # noqa: BLE001
        report["status"] = "error"
        report["error"] = str(e)
    return report


# --------------------------------------------------------------------------- #
# CONCEPT:AHE-3.46 — scheduled optimization sweep (the daemon-tick twin)
# --------------------------------------------------------------------------- #
# The self-supervised targets the daemon can run unattended (the registry targets —
# system_prompt/tool_description/skill — are driven by the failure-cluster evolution
# cycle, not this sweep).
SCHEDULABLE_TARGETS: tuple[str, ...] = ("extraction", "concept_match", "routing")


def should_promote(
    baseline_score: float, candidate_score: float, *, min_delta: float = 0.0
) -> bool:
    """Promotion gate (CONCEPT:AHE-3.46): a candidate replaces the incumbent only when it
    beats it on the held-out metric by at least ``min_delta``. The criterion the sweep
    applies before an optimized artifact is allowed to supersede the live one."""
    return candidate_score >= baseline_score + min_delta


# --------------------------------------------------------------------------- #
# CONCEPT:AHE-3.71 — the prompt-hardening cycle (optimize → evaluate → propose)
# --------------------------------------------------------------------------- #
# The system-prompt leg of the four artifacts, closed end-to-end: the DSPy demos (or,
# when no LM is reachable to run a compile, the agent's own labeled successes) are folded
# into the live :class:`StructuredPrompt` as exemplars, the candidate is scored against the
# agent's eval-corpus slice, and a promote/reject decision is returned. The *apply* of the
# winning candidate is gated by ``EvolveAgent.apply_edits`` + ``KG_AGENT_AUTO_APPLY`` — this
# module only decides; it never writes source.


def _demo_fields(demo: Any) -> tuple[str, str, str]:
    """Extract ``(context, task, response)`` from a DSPy ``Example`` or a plain dict."""
    if isinstance(demo, dict):
        get = demo.get
    else:
        get = lambda k, d="": getattr(demo, k, d)  # noqa: E731 - tiny adapter
    context = str(get("context", "") or "")
    task = str(get("task", "") or get("query", "") or "")
    response = str(get("response", "") or get("expected_output", "") or "")
    return context, task, response


def build_hardened_prompt(
    baseline: Any,
    demos: Sequence[Any],
    *,
    optimized_instruction: str = "",
    max_demos: int = 4,
) -> Any:
    """Fold optimized exemplars into a candidate :class:`StructuredPrompt` (CONCEPT:AHE-3.71).

    The hardening edit per BootstrapFewShot: the bootstrapped demos (each a real
    ``input → ideal response`` drawn from the agent's *passing* executions) are appended to
    the prompt body as a ``LEARNED EXEMPLARS`` block, and the prompt's ``prompt_version`` is
    bumped. Returns a NEW prompt object (deep copy); never mutates ``baseline``. When there
    is nothing to add (no demos, no optimized instruction) the copy is returned unchanged so
    the candidate simply ties baseline and is rejected by :func:`should_promote`.
    """
    from agent_utilities.prompting.structured import (
        PromptInstructions,
        StructuredPrompt,
    )

    candidate: StructuredPrompt = baseline.model_copy(deep=True)

    exemplars: list[str] = []
    for demo in list(demos)[:max_demos]:
        _ctx, task, response = _demo_fields(demo)
        if not (task or response):
            continue
        exemplars.append(
            f"- Input: {task.strip()[:400]}\n  Ideal response: {response.strip()[:400]}"
        )

    if not exemplars and not optimized_instruction.strip():
        return candidate

    instr = candidate.instructions or PromptInstructions()
    base_body = (instr.core_directive or "").strip()
    if optimized_instruction.strip() and optimized_instruction.strip() != base_body:
        base_body = optimized_instruction.strip()
    sections: list[str] = [base_body] if base_body else []
    if exemplars:
        sections.append(
            "### LEARNED EXEMPLARS (hardened from real execution outcomes)\n"
            + "\n".join(exemplars)
        )
    instr.core_directive = "\n\n".join(s for s in sections if s)
    candidate.instructions = instr
    candidate.prompt_version = _bump_patch(candidate.prompt_version)
    return candidate


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

    CONCEPT:AHE-3.71. The offline-deterministic proxy for "does this prompt embed the
    behavior the corpus rewards": each case's ``expected_output`` is scored against the
    rendered prompt via :func:`graded_score` (the same semantic scorer the DSPy metric
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
    """The audit record of one prompt-hardening cycle (CONCEPT:AHE-3.71).

    Carries everything a human/Claude needs to review the action: which agent, the
    before/after metric, the promote decision, whether it was actually applied (vs held in
    shadow), and the candidate's content hash. ``status`` is one of ``no_data`` (no
    per-agent corpus), ``rejected`` (did not beat baseline), ``proposed`` (beat baseline but
    auto-apply gated off — shadow), or ``applied`` (written to source under the gate).
    """

    agent_id: str
    prompt_path: str
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
            "agent_id": self.agent_id,
            "prompt_path": self.prompt_path,
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
    """Best-effort production data for a self-supervised target (CONCEPT:AHE-3.46).

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
    """Propose-only DSPy optimization sweep over the schedulable targets (CONCEPT:AHE-3.46).

    The reusable core the daemon tick and the on-demand ``optimize_component task=all``
    surface both call. For each target it gathers live data
    (:func:`gather_optimization_data`) and runs :func:`run_component_optimization`. It is
    **propose-only**: the optimizers persist optimization trajectories to the KG but
    nothing is auto-applied to source — promotion stays behind :func:`should_promote` and
    a future auto-apply gate (mirroring ``KG_GOLDEN_AUTO_MERGE``). Returns a per-target
    report; never raises.
    """
    names = list(targets) if targets else list(SCHEDULABLE_TARGETS)
    report: dict[str, Any] = {}
    optimized: list[str] = []
    for name in names:
        data = gather_optimization_data(engine, name)
        result = run_component_optimization(name, data)
        report[name] = result
        if result.get("status") == "optimized":
            optimized.append(name)
    return {"targets": report, "optimized": optimized, "propose_only": True}
