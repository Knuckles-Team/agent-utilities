from __future__ import annotations

"""KG-trace-derived DSPy training examples (CONCEPT:AU-AHE.optimization.trace-derived-training-examples).

Closes the cohesive-loop gap between the observability flywheel — the machinery
that already mines ``:Episode``/``:ToolCall``/``:OutcomeEvaluation`` provenance
for FAILURE patterns (``knowledge_graph/research/trace_pattern_miner.py``,
``knowledge_graph/orchestration/engine_ahe.py::propose_new_skill_from_experience``,
``knowledge_graph/adaptation/feedback.py``) — and the DSPy optimizer
(``harness/dspy_optimization.py``), which until now only ever compiled against
self-supervised / synthetic trainsets (``make_optimization_metric``'s own
``reward_fn``/``reward_weight`` parameters existed but no caller ever passed
them). A prompt, tool description, or skill that produced FAILED traces in
production now becomes a labeled NEGATIVE example the optimization metric
penalizes automatically, and a passing trace becomes a real positive
demonstration — closing the loop from "traces observed" to "training signal
used" end to end.

Composition, not reinvention: reuses the SAME
``Episode -[:USED_TOOL]-> ToolCall -[:PRODUCED_OUTCOME]-> OutcomeEvaluation``
schema :mod:`trace_pattern_miner` mines the failure side of and ``engine_ahe``
mines the success side of, over the SAME ``engine.query_cypher`` surface every
other KG-reading optimizer helper here uses
(:func:`~agent_utilities.harness.dspy_optimization.gather_optimization_data`).
No new trace store, no new LLM call, no bypass of the endpoint-safe DSPy LM
adapter (:mod:`agent_utilities.harness.dspy_lm_adapter`) — this module only
ever reads the graph and builds plain data; the compile itself still runs
under :func:`~agent_utilities.harness.dspy_lm_adapter.dspy_optimization_guard`.
"""

import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .dspy_optimization import OptimizableTarget

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

#: Default blend weight for the KG-observed reward inside the optimization
#: metric (see ``dspy_optimization.make_optimization_metric``'s ``reward_weight``)
#: — moderate by design: real outcomes steer the metric without drowning out
#: the text-quality signal, and drop to 0 automatically when there are no traces.
DEFAULT_REWARD_WEIGHT = 0.3


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class TraceExample:
    """One KG-observed execution, distilled into DSPy-``Example``-shaped fields.

    ``reward`` carries the REAL outcome (``:OutcomeEvaluation.reward``, or the
    neutral prior when no outcome was recorded) so the optimization metric can
    be steered by it (:func:`trace_reward_fn`); ``failure_reason`` carries the
    ``OutcomeEvaluation.feedback_text`` for a failing trace — the signal a
    negative example teaches FROM (not just "this is bad" but "here's why").
    A failing example's ``response`` is deliberately left blank so
    ``BootstrapFewShot`` can never mistake a known-bad output for a
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

    def to_dspy_example(self) -> Any:
        """Render as a ``dspy.Example`` (inputs: ``context``/``task``).

        Degrades to a plain dict when DSPy is not importable — every
        downstream consumer (the metric, the compile trainset) already
        tolerates either shape via ``getattr``/``.get``.
        """
        try:
            import dspy

            ex = dspy.Example(
                context=self.context,
                task=self.task,
                response=self.response,
            ).with_inputs("context", "task")
            # Extra, non-Signature attributes DSPy Examples tolerate — the
            # reward channel `trace_reward_fn` reads back out at metric time.
            ex.reward = self.reward
            ex.success = self.success
            ex.failure_reason = self.failure_reason
            ex.source = "kg_trace"
            return ex
        except Exception:  # noqa: BLE001 - dspy optional / construction best-effort
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


def _query_by_tool(engine: Any, tool_name: str, limit: int) -> list[dict[str, Any]]:
    """Episodes that called ``tool_name``, joined to their outcome (if any).

    Reuses the exact ``Episode -[:USED_TOOL]-> ToolCall`` /
    ``Episode -[:PRODUCED_OUTCOME]-> OutcomeEvaluation`` schema
    :mod:`trace_pattern_miner` mines the failure side of.
    """
    try:
        rows = (
            engine.query_cypher(
                "MATCH (e:Episode)-[:USED_TOOL]->(t:ToolCall {tool_name: $name}) "
                "OPTIONAL MATCH (e)-[:PRODUCED_OUTCOME]->(o:OutcomeEvaluation) "
                "RETURN coalesce(e.summary, e.description, '') AS context, "
                "coalesce(t.args, '') AS task_input, "
                "coalesce(t.result, '') AS result, "
                "o.reward AS reward, o.feedback_text AS feedback_text "
                "ORDER BY e.timestamp DESC "
                f"LIMIT {int(limit)}",
                {"name": tool_name},
            )
            or []
        )
    except Exception as e:  # noqa: BLE001 - a query failure degrades, never raises
        logger.debug("trace_examples: tool-call query failed for %s: %s", tool_name, e)
        return []
    return [r for r in rows if isinstance(r, dict)]


def _query_by_tag(engine: Any, name: str, limit: int) -> list[dict[str, Any]]:
    """Episodes tagged with ``name`` (an agent/task id or a skill name), joined
    to their outcome (if any). Mirrors the ``agent:<id>``/skill-name tagging
    convention :meth:`~agent_utilities.knowledge_graph.adaptation.feedback.
    FeedbackService.record_action_outcome` already stamps on eval-corpus cases,
    generalized to whatever attribution tag an ``Episode`` carries.
    """
    try:
        rows = (
            engine.query_cypher(
                "MATCH (e:Episode) WHERE any(tag IN e.tags WHERE tag CONTAINS $name) "
                "OPTIONAL MATCH (e)-[:PRODUCED_OUTCOME]->(o:OutcomeEvaluation) "
                "RETURN coalesce(e.summary, e.description, '') AS context, "
                "coalesce(e.summary, e.description, '') AS task_input, "
                "'' AS result, "
                "o.reward AS reward, o.feedback_text AS feedback_text "
                "ORDER BY e.timestamp DESC "
                f"LIMIT {int(limit)}",
                {"name": name},
            )
            or []
        )
    except Exception as e:  # noqa: BLE001 - a query failure degrades, never raises
        logger.debug("trace_examples: tag query failed for %s: %s", name, e)
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
        # No OutcomeEvaluation recorded for this episode — a neutral, unlabeled
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
) -> list[TraceExample]:
    """Query the KG for recent traces attributable to ``target``/``artifact``.

    CONCEPT:AU-AHE.optimization.trace-derived-training-examples. Dispatches on ``target.component_type``:
    ``tool_description`` → episodes that called the named tool
    (``Episode -[:USED_TOOL]-> ToolCall``); ``skill``/``system_prompt`` →
    episodes tagged with the skill/agent name. Bounded to ``limit`` rows.
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
        rows = _query_by_tool(engine, name, limit)
    elif component_type in ("skill", "system_prompt"):
        rows = _query_by_tag(engine, name, limit)
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
    examples that never carried one. Never raises."""
    try:
        if isinstance(example, dict):
            return float(example.get("reward", 0.5) or 0.5)
        return float(getattr(example, "reward", 0.5) or 0.5)
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
    trace_rows = [ex.to_dspy_example() for ex in trace_examples]
    self_rows = list(self_supervised or [])
    blended = trace_rows + self_rows
    stats = {
        "component_type": str(getattr(target, "component_type", "") or ""),
        "identifier": _safe_task_name(target, artifact) or "unknown",
        "trace_derived": len(trace_examples),
        "trace_failures": n_failures,
        "trace_successes": n_successes,
        "self_supervised": len(self_rows),
        "total": len(blended),
    }
    return blended, stats


def record_trace_derived_finding(
    engine: Any, stats: dict[str, Any], *, node_type: str = "DSPyTraceOptimizationFinding"
) -> str | None:
    """Emit a structured log + a best-effort KG note recording that an
    optimization pass drew from real traces (CONCEPT:AU-AHE.optimization.trace-derived-training-examples) — the
    observable end of the closed loop: traces → KG findings → DSPy examples.

    Always logs (so the loop is visible even with no engine/persistence).
    Persists a ``:DSPyTraceOptimizationFinding`` node when ``engine`` supports
    ``add_node`` (the SAME 3-arg ``add_node(id, label, properties=...)`` idiom
    ``ClaimFlywheel``/``engine_ahe`` already write through) — best-effort,
    never raises, and skipped entirely when there was nothing to record.
    Returns the finding id when persisted, else ``None``.
    """
    component_type = stats.get("component_type", "")
    identifier = stats.get("identifier", "")
    if not stats.get("trace_derived"):
        logger.info(
            "dspy_optimization[%s/%s]: no KG traces found; optimizing on %d "
            "self-supervised example(s) only",
            component_type,
            identifier,
            stats.get("self_supervised", 0),
        )
        return None
    logger.info(
        "dspy_optimization[%s/%s]: drew %d KG-trace example(s) (%d failures, "
        "%d successes) + %d self-supervised example(s) = %d total trainset",
        component_type,
        identifier,
        stats["trace_derived"],
        stats["trace_failures"],
        stats["trace_successes"],
        stats["self_supervised"],
        stats["total"],
    )
    if engine is None or not hasattr(engine, "add_node"):
        return None
    finding_id = f"dspy_trace_finding:{component_type}:{identifier}:{uuid.uuid4().hex[:10]}"
    try:
        engine.add_node(
            finding_id,
            node_type,
            properties={
                "component_type": component_type,
                "identifier": identifier,
                "trace_derived_count": stats["trace_derived"],
                "trace_failure_count": stats["trace_failures"],
                "trace_success_count": stats["trace_successes"],
                "self_supervised_count": stats["self_supervised"],
                "recorded_at": _now_iso(),
            },
        )
        return finding_id
    except Exception as e:  # noqa: BLE001 - the finding note is best-effort, never load-bearing
        logger.debug(
            "trace_examples: could not persist finding node for %s/%s: %s",
            component_type,
            identifier,
            e,
        )
        return None
