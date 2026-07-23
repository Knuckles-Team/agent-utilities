from __future__ import annotations

"""KG-trace-derived native program training examples.

Closes the cohesive-loop gap between the observability flywheel — the machinery
that already mines ``:RunTrace``/``:ToolCall``/``:OutcomeEvaluation`` provenance
for FAILURE patterns (``knowledge_graph/research/trace_pattern_miner.py``,
``knowledge_graph/orchestration/engine_ahe.py::propose_new_skill_from_experience``,
``knowledge_graph/adaptation/feedback.py``) — and the native program optimizer
(``harness/program_optimization.py``). A prompt, tool description, or skill that produced FAILED traces in
production now becomes a labeled NEGATIVE example the optimization metric
penalizes automatically, and a passing trace becomes a real positive
demonstration — closing the loop from "traces observed" to "training signal
used" end to end.

Composition, not reinvention: reuses the SAME
``RunTrace -[:USED_TOOL]-> ToolCall`` and
``RunTrace -[:PRODUCED_OUTCOME]-> OutcomeEvaluation``
schema :mod:`trace_pattern_miner` mines the failure side of and ``engine_ahe``
mines the success side of, over the SAME ``engine.query_cypher`` surface every
other KG-reading optimizer helper here uses
(:func:`~agent_utilities.harness.program_optimization.gather_optimization_data`).
No new trace store or model call is introduced; this module only reads the graph
and builds provider-neutral data for the engine-owned job.
"""

import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_utilities.observability.trace_ontology import (
    TRACE_PRODUCED_OUTCOME_EDGE,
    TRACE_USED_TOOL_EDGE,
)
from agent_utilities.security.persistence_privacy import persistence_reference

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .program_optimization import OptimizableTarget

logger = logging.getLogger(__name__)

__all__ = [
    "TraceExample",
    "gather_trace_examples",
    "blend_trainset",
    "trace_reward_fn",
    "record_trace_derived_finding",
]

#: Cap on rows scanned per target — bounded like every other mining pass's row
#: LIMIT (mirrors ``trace_pattern_miner._TRACE_SCAN_LIMIT``).
DEFAULT_TRACE_LIMIT = 50

#: An outcome below this reward counts as a FAILURE (mirrors
#: ``trace_pattern_miner.FAILURE_REWARD_THRESHOLD`` / ``engine_ahe``'s own
#: "o.reward < 0.5" convention — kept in sync deliberately, not re-derived).
FAILURE_REWARD_THRESHOLD = 0.5

#: Default blend weight for the KG-observed reward inside an optimization metric.
DEFAULT_REWARD_WEIGHT = 0.3


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class TraceExample:
    """One KG-observed execution distilled into provider-neutral fields.

    ``reward`` carries the REAL outcome (``:OutcomeEvaluation.reward``, or the
    neutral prior when no outcome was recorded) so the optimization metric can
    be steered by it (:func:`trace_reward_fn`); ``failure_reason`` carries the
    ``OutcomeEvaluation.feedback_text`` for a failing trace — the signal a
    negative example teaches FROM (not just "this is bad" but "here's why").
    A failing example's ``response`` is deliberately left blank so
    the native compiler can never mistake a known-bad output for a
    demonstration to imitate; the failure still contributes to the metric via
    ``reward``/``failure_reason``.
    """

    context: str
    task: str
    response: str
    reward: float
    success: bool
    failure_reason: str = ""
    source_id: str = ""

    def to_payload(self) -> dict[str, Any]:
        """Render the bounded, JSON-compatible optimizer payload."""
        return {
            "context": self.context,
            "task": self.task,
            "response": self.response,
            "reward": self.reward,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "source": "kg_trace",
        }


def _auto_engine() -> Any:
    """Best-effort resolution of the live KG engine when none was passed in
    (CONCEPT:AU-AHE.optimization.trace-derived-training-examples — default-on: a caller that forgets to
    thread ``engine`` through still gets real traces when a process-wide engine
    is active)."""
    try:
        from agent_utilities.knowledge_graph.core.engine import (
            IntelligenceGraphEngine,
        )

        return IntelligenceGraphEngine.get_active()
    except Exception:  # noqa: BLE001 - no active engine is a normal, cold-start state
        return None


def _safe_task_name(target: Any, artifact: dict[str, Any]) -> str:
    try:
        return str(target.task_name(artifact) or "")
    except Exception:  # noqa: BLE001 - a malformed artifact must never break gathering
        return ""


def _query_by_tool(
    engine: Any, tool_name: str, limit: int, after_sequence: int
) -> list[dict[str, Any]]:
    """Traces that called ``tool_name``, joined to their outcome (if any).

    Reuses the exact canonical ``RunTrace`` trace/outcome schema
    :mod:`trace_pattern_miner` mines the failure side of.
    """
    try:
        rows = (
            engine.query_cypher(
                f"MATCH (r:RunTrace)-[:{TRACE_USED_TOOL_EDGE}]->(t:ToolCall {{tool_name: $name}}) "
                "WHERE r.event_sequence > $after_sequence "
                f"OPTIONAL MATCH (r)-[:{TRACE_PRODUCED_OUTCOME_EDGE}]->(o:OutcomeEvaluation) "
                "RETURN coalesce(r.task, '') AS context, "
                "coalesce(t.args, '') AS task_input, "
                "coalesce(t.result, '') AS result, "
                "o.reward AS reward, o.feedback_text AS feedback_text, "
                "r.event_sequence AS event_sequence "
                "ORDER BY event_sequence DESC "
                f"LIMIT {int(limit)}",
                {"name": tool_name, "after_sequence": int(after_sequence)},
            )
            or []
        )
    except Exception as exc:  # noqa: BLE001 - a query failure degrades, never raises
        logger.debug("trace_examples: tool-call query failed (%s)", type(exc).__name__)
        return []
    return [r for r in rows if isinstance(r, dict)]


def _query_by_tag(
    engine: Any, name: str, limit: int, after_sequence: int
) -> list[dict[str, Any]]:
    """Traces attributed to ``name`` (agent or skill), joined
    to their outcome (if any). Mirrors the ``agent:<id>``/skill-name tagging
    convention :meth:`~agent_utilities.knowledge_graph.adaptation.feedback.
    FeedbackService.record_action_outcome` already stamps on eval-corpus cases,
    through the same non-reversible references the runtime writes.
    """
    try:
        rows = (
            engine.query_cypher(
                "MATCH (r:RunTrace) WHERE r.event_sequence > $after_sequence "
                "AND (r.attribution_ref = $agent_ref OR r.skill_ref = $skill_ref) "
                f"OPTIONAL MATCH (r)-[:{TRACE_PRODUCED_OUTCOME_EDGE}]->(o:OutcomeEvaluation) "
                "RETURN coalesce(r.task, '') AS context, "
                "coalesce(r.task, '') AS task_input, '' AS result, "
                "o.reward AS reward, o.feedback_text AS feedback_text, "
                "r.event_sequence AS event_sequence ORDER BY event_sequence DESC "
                f"LIMIT {int(limit)}",
                {
                    "agent_ref": persistence_reference(
                        "agent", name, namespace="execution-trace"
                    ),
                    "skill_ref": persistence_reference(
                        "skill", name, namespace="execution-trace"
                    ),
                    "after_sequence": int(after_sequence),
                },
            )
            or []
        )
    except Exception as exc:  # noqa: BLE001 - a query failure degrades, never raises
        logger.debug("trace_examples: tag query failed (%s)", type(exc).__name__)
        return []
    return [r for r in rows if isinstance(r, dict)]


def _row_to_example(row: dict[str, Any]) -> TraceExample | None:
    """Turn one KG row into a labeled :class:`TraceExample`, or ``None`` when
    the row carries no usable context/task text."""
    context = str(row.get("context") or "")
    task = str(row.get("task_input") or "") or context
    response = str(row.get("result") or "")
    feedback_text = str(row.get("feedback_text") or "")
    raw_reward = row.get("reward")
    if raw_reward is None:
        # No OutcomeEvaluation recorded for this trace — a neutral, unlabeled
        # observation. Still worth keeping (real context/task text), but never
        # asserted as either a positive or a negative signal.
        reward = 0.5
        success = True
    else:
        try:
            reward = max(0.0, min(1.0, float(raw_reward)))
        except (TypeError, ValueError):
            reward = 0.5
        success = reward >= FAILURE_REWARD_THRESHOLD
    if not (context or task):
        return None
    if not success:
        # Deliberately blank: a failing trace's own output must never look like
        # a bootstrap-worthy demonstration (see TraceExample docstring).
        response = ""
    return TraceExample(
        context=context,
        task=task,
        response=response,
        reward=reward,
        success=success,
        failure_reason=feedback_text if not success else "",
    )


def gather_trace_examples(
    engine: Any,
    target: OptimizableTarget,
    artifact: dict[str, Any],
    *,
    limit: int = DEFAULT_TRACE_LIMIT,
    after_sequence: int = 0,
) -> list[TraceExample]:
    """Query the KG for recent traces attributable to ``target``/``artifact``.

    CONCEPT:AU-AHE.optimization.trace-derived-training-examples. Dispatches on ``target.component_type``:
    ``tool_description`` → traces that called the named tool;
    ``skill``/``system_prompt`` → traces carrying the matching opaque
    attribution reference. Bounded to ``limit`` rows and resumed with a numeric
    ``after_sequence`` cursor.
    Degrades to ``[]`` — never raises — when the engine is unavailable, the
    target has no resolvable name, or the query fails; callers then fall back
    to self-supervised examples alone (:func:`blend_trainset`).
    """
    engine = engine if engine is not None else _auto_engine()
    if engine is None or not hasattr(engine, "query_cypher"):
        return []
    name = _safe_task_name(target, artifact)
    if not name:
        return []
    component_type = str(getattr(target, "component_type", "") or "")
    if component_type == "tool_description":
        rows = _query_by_tool(engine, name, limit, after_sequence)
    elif component_type in ("skill", "system_prompt"):
        rows = _query_by_tag(engine, name, limit, after_sequence)
    else:
        rows = []
    examples: list[TraceExample] = []
    for row in rows:
        ex = _row_to_example(row)
        if ex is not None:
            ex.source_id = name
            examples.append(ex)
    return examples


def trace_reward_fn(example: Any) -> float:
    """The ``reward_fn`` every KG-trace-aware metric blends in
    (CONCEPT:AU-AHE.optimization.trace-derived-training-examples) — reads the real
    ``:OutcomeEvaluation``-derived reward :func:`gather_trace_examples` stamped
    onto the example, defaulting to a neutral ``0.5`` for self-supervised
    examples that never carried one. Never raises.

    Deliberately checks for ``None`` (missing), not falsiness — a genuine
    ``reward == 0.0`` (the worst, most informative failure) must reach the
    metric as ``0.0``, not get silently coerced back to the neutral default.
    """
    raw = (
        example.get("reward")
        if isinstance(example, dict)
        else getattr(example, "reward", None)
    )
    if raw is None:
        return 0.5
    try:
        return float(raw)
    except (TypeError, ValueError):  # noqa: BLE001 - a malformed reward never breaks the metric
        return 0.5


def blend_trainset(
    engine: Any,
    target: OptimizableTarget,
    artifact: dict[str, Any],
    self_supervised: Sequence[Any] | None = None,
    *,
    limit: int = DEFAULT_TRACE_LIMIT,
) -> tuple[list[Any], dict[str, Any]]:
    """Build the blended trainset: KG-trace-derived examples FIRST, the
    caller's self-supervised examples SECOND (CONCEPT:AU-AHE.optimization.trace-derived-training-examples).

    Blend, never replace: when the KG has no traces for this target the
    result is exactly ``self_supervised`` (cold-start still works); when it
    does, the real successes/failures lead the compiled few-shot set and the
    caller's own examples fill out the rest. Returns ``(trainset, stats)`` —
    ``stats`` is the observability record :func:`record_trace_derived_finding`
    logs/persists, and what callers report trace provenance from.
    """
    trace_examples = gather_trace_examples(engine, target, artifact, limit=limit)
    n_failures = sum(1 for ex in trace_examples if not ex.success)
    n_successes = len(trace_examples) - n_failures
    trace_rows = [ex.to_payload() for ex in trace_examples]
    self_rows = list(self_supervised or [])
    blended = trace_rows + self_rows
    stats = {
        "component_type": str(getattr(target, "component_type", "") or ""),
        "identifier_ref": persistence_reference(
            "optimization_target",
            _safe_task_name(target, artifact) or "unknown",
            namespace=str(getattr(target, "component_type", "") or "unknown"),
        ),
        "trace_derived": len(trace_examples),
        "trace_failures": n_failures,
        "trace_successes": n_successes,
        "self_supervised": len(self_rows),
        "total": len(blended),
    }
    return blended, stats


def record_trace_derived_finding(
    engine: Any,
    stats: dict[str, Any],
    *,
    node_type: str = "ProgramOptimizationFinding",
) -> str | None:
    """Emit a structured log + a best-effort KG note recording that an
    optimization pass drew from real traces (CONCEPT:AU-AHE.optimization.trace-derived-training-examples) — the
    observable end of the closed loop: traces → KG findings → training examples.

    Always logs (so the loop is visible even with no engine/persistence).
    Persists a ``:ProgramOptimizationFinding`` node when ``engine`` supports
    ``add_node`` (the SAME 3-arg ``add_node(id, label, properties=...)`` idiom
    ``ClaimFlywheel``/``engine_ahe`` already write through) — best-effort,
    never raises, and skipped entirely when there was nothing to record.
    Returns the finding id when persisted, else ``None``.
    """
    engine = engine if engine is not None else _auto_engine()
    component_type = stats.get("component_type", "")
    identifier_ref = stats.get("identifier_ref", "")
    if not stats.get("trace_derived"):
        logger.info(
            "program_optimization[%s/%s]: no KG traces found; optimizing on %d "
            "self-supervised example(s) only",
            component_type,
            identifier_ref,
            stats.get("self_supervised", 0),
        )
        return None
    logger.info(
        "program_optimization[%s/%s]: drew %d KG-trace example(s) (%d failures, "
        "%d successes) + %d self-supervised example(s) = %d total trainset",
        component_type,
        identifier_ref,
        stats["trace_derived"],
        stats["trace_failures"],
        stats["trace_successes"],
        stats["self_supervised"],
        stats["total"],
    )
    if engine is None or not hasattr(engine, "add_node"):
        return None
    finding_ref = persistence_reference(
        "finding",
        f"{component_type}:{identifier_ref}:{uuid.uuid4().hex}",
        namespace="program-optimization",
    )
    finding_id = f"program_trace_finding:{finding_ref}"
    try:
        engine.add_node(
            finding_id,
            node_type,
            properties={
                "component_type": component_type,
                "identifier_ref": identifier_ref,
                "trace_derived_count": stats["trace_derived"],
                "trace_failure_count": stats["trace_failures"],
                "trace_success_count": stats["trace_successes"],
                "self_supervised_count": stats["self_supervised"],
                "recorded_at": _now_iso(),
            },
        )
        return finding_id
    except Exception:  # noqa: BLE001 - the finding note is best-effort, never load-bearing
        logger.debug(
            "trace_examples: could not persist finding node for %s/%s",
            component_type,
            identifier_ref,
        )
        return None
