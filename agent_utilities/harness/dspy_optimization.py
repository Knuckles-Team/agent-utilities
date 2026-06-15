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
