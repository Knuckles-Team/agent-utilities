"""Run-outcome-driven prompt evolution (propose-only).

CONCEPT:AU-AHE.optimization.run-outcome-prompt-evolution — run outcomes feed propose-only prompt evolution (AU-HARNESS-R007).

Closes the L5 loop from recorded agent runs to a reviewable prompt candidate:

    RunTrace -[PRODUCED_OUTCOME]-> OutcomeEvaluation   (per agent, after a cursor)
      → reference-only training rows
      → program optimizer (native eg-program DSPy-family job by default)
      → PromptVersion candidate (status="proposal")
        -[was_derived_from]-> every RunTrace it used

Composition, not a new subsystem:

- the trace schema, attribution reference and durable consumer cursor come from
  :mod:`agent_utilities.observability.trace_ontology`;
- the optimizer is :func:`~agent_utilities.harness.program_optimization.run_program_optimization`
  (the engine-owned ``ProgramOptimize`` job that replaced the in-process DSPy
  dependency in release 1.27); any callable with the :class:`PromptOptimizer`
  shape can replace it, so no DSPy import exists here;
- the candidate is the existing reference-only compiled prompt
  (:meth:`EvolveAgent._compiled_prompt_candidate`) recorded as the existing
  :class:`~agent_utilities.models.knowledge_graph.PromptVersionNode` lifecycle
  node, so :func:`~agent_utilities.knowledge_graph.research.evolution_state.artifact_evolution_summary`
  lists it for review;
- the schedule is the existing native optimization sweep
  (``KG_OPTIMIZATION_ENABLED`` / ``KG_OPTIMIZATION_INTERVAL``), which runs
  :func:`run_prompt_evolution_sweep` as its ``prompt_evolution`` target.

The module never writes a prompt file and never promotes a version. Promotion
stays behind the unified artifact promotion gate and human review.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_utilities.observability.trace_ontology import (
    TRACE_PRODUCED_OUTCOME_EDGE,
    TraceCursor,
    load_trace_cursor,
    save_trace_cursor,
)
from agent_utilities.security.persistence_privacy import persistence_reference

logger = logging.getLogger(__name__)

__all__ = [
    "PROMPT_EVOLUTION_TARGET",
    "PROPOSAL_ORIGIN",
    "PromptOptimizer",
    "RunOutcome",
    "gather_agent_run_outcomes",
    "native_prompt_optimizer",
    "packaged_prompt_targets",
    "propose_prompt_from_run_outcomes",
    "run_prompt_evolution_sweep",
]

#: Sweep target name the native optimization sweep dispatches to this module.
PROMPT_EVOLUTION_TARGET = "prompt_evolution"
#: ``origin`` stamped on every candidate this loop proposes.
PROPOSAL_ORIGIN = "run_outcome_program"
#: Fewest new attributed outcomes that justify one optimization pass.
MIN_RUN_OUTCOMES = 8
#: Row cap per agent per pass (bounded like every other trace miner).
RUN_OUTCOME_LIMIT = 50
_PROMPT_VERSION_LABEL = "prompt_version"
_DERIVED_FROM_EDGE = "was_derived_from"
_FAILURE_REWARD_THRESHOLD = 0.5
_SUCCESS_STATUSES = frozenset({"completed", "success", "succeeded", "ok"})

#: ``(artifact, trainset, *, engine) -> OptimizationResult | None``. The result
#: must carry a reference-only ``compiled_state`` and a ``confidence``.
PromptOptimizer = Callable[..., Any]


@dataclass(frozen=True)
class RunOutcome:
    """One attributed run and its terminal outcome, reference-only."""

    trace_id: str
    task_ref: str
    result_ref: str
    reward: float
    success: bool
    event_sequence: int

    def to_training_row(self) -> dict[str, Any]:
        """Return the bounded optimizer row; raw run content never appears."""
        return {
            "example_ref": self.trace_id,
            "trace_ref": self.trace_id,
            "context": "",
            "task": self.task_ref or self.trace_id,
            "response": self.result_ref if self.success else "",
            "reward": self.reward,
            "success": self.success,
            "failure_reason": "" if self.success else "run_outcome_failed",
            "source": "kg_trace",
        }


def _reward(row: Mapping[str, Any]) -> float:
    raw = row.get("reward")
    if raw is None:
        status = str(row.get("status") or "").lower()
        return 1.0 if status in _SUCCESS_STATUSES else 0.0
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.0


def _row_to_outcome(row: Mapping[str, Any]) -> RunOutcome | None:
    trace_id = str(row.get("id") or "")
    try:
        sequence = int(row.get("event_sequence") or 0)
    except (TypeError, ValueError):
        return None
    if not trace_id or sequence <= 0:
        return None
    reward = _reward(row)
    return RunOutcome(
        trace_id=trace_id,
        task_ref=str(row.get("task_digest") or ""),
        result_ref=str(row.get("result_digest") or ""),
        reward=reward,
        success=reward >= _FAILURE_REWARD_THRESHOLD,
        event_sequence=sequence,
    )


def gather_agent_run_outcomes(
    engine: Any,
    agent_id: str,
    *,
    after_sequence: int = 0,
    limit: int = RUN_OUTCOME_LIMIT,
) -> list[RunOutcome]:
    """Return recent ``RunTrace`` outcomes attributed to ``agent_id``.

    Matches the opaque ``attribution_ref`` the runtime stamps on every trace.
    A trace without an ``OutcomeEvaluation`` falls back to its own status.
    Degrades to ``[]`` when the engine or query is unavailable.
    """
    if engine is None or not hasattr(engine, "query_cypher"):
        return []
    agent_ref = persistence_reference("agent", agent_id, namespace="execution-trace")
    try:
        rows = engine.query_cypher(
            "MATCH (r:RunTrace) WHERE r.attribution_ref = $agent_ref "
            "AND r.event_sequence > $after_sequence "
            f"OPTIONAL MATCH (r)-[:{TRACE_PRODUCED_OUTCOME_EDGE}]->(o:OutcomeEvaluation) "
            "RETURN r.id AS id, r.status AS status, r.task_digest AS task_digest, "
            "r.result_digest AS result_digest, o.reward AS reward, "
            "r.event_sequence AS event_sequence "
            "ORDER BY event_sequence DESC "
            f"LIMIT {int(limit)}",
            {"agent_ref": agent_ref, "after_sequence": int(after_sequence)},
        )
    except Exception as exc:  # a read failure degrades to idle
        logger.warning("prompt_evolution: outcome query failed: %s", exc)
        return []
    outcomes = [_row_to_outcome(row) for row in rows or [] if isinstance(row, Mapping)]
    return [outcome for outcome in outcomes if outcome is not None]


def native_prompt_optimizer(
    artifact: dict[str, Any], trainset: list[dict[str, Any]], *, engine: Any
) -> Any:
    """Default optimizer: the native ``eg-program`` system-prompt job."""
    from .program_optimization import get_target, run_program_optimization

    target = get_target("system_prompt")
    if target is None:
        return None
    return run_program_optimization(target, artifact, trainset, engine=engine)


def _pending_proposal_exists(engine: Any, agent_id: str) -> bool:
    """Return whether this loop already holds a candidate awaiting review."""
    try:
        rows = engine.query_cypher(
            f"MATCH (v:{_PROMPT_VERSION_LABEL}) WHERE v.prompt_id = $prompt_id "
            "AND v.status = 'proposal' AND v.origin = $origin "
            "RETURN v.id AS id LIMIT 1",
            {"prompt_id": agent_id, "origin": PROPOSAL_ORIGIN},
        )
    except Exception as exc:  # unknown state: hold, never duplicate
        logger.warning("prompt_evolution: pending proposal read failed: %s", exc)
        return True
    return bool(rows)


def _consumer(agent_id: str) -> str:
    return f"{PROMPT_EVOLUTION_TARGET}:{agent_id}"


def _candidate_node(
    agent_id: str,
    baseline: Any,
    candidate: Any,
    result: Any,
    outcomes: list[RunOutcome],
) -> Any:
    """Build the reviewable ``PromptVersionNode`` for one candidate."""
    node = candidate.version(agent_id, parent_hash=baseline.version_hash())
    failures = sum(1 for outcome in outcomes if not outcome.success)
    return node.model_copy(
        update={
            "artifact_kind": "prompt",
            "artifact_id": agent_id,
            "status": "proposal",
            "origin": PROPOSAL_ORIGIN,
            "reward": float(getattr(result, "confidence", 0.0) or 0.0),
            "reward_source": str(getattr(result, "optimizer", "") or "optimizer"),
            "task_count": len(outcomes),
            "notes": [f"run_outcomes={len(outcomes)}", f"failures={failures}"],
        }
    )


def _persist_candidate(
    engine: Any, node: Any, compiled_state: Mapping[str, Any], trace_ids: list[str]
) -> None:
    """Write the candidate node and one provenance edge per source trace."""
    props = node.model_dump()
    props.pop("id", None)
    props["type"] = str(props.get("type", ""))
    props["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    props["program_compiled_state_json"] = json.dumps(
        dict(compiled_state), sort_keys=True
    )
    engine.add_node(node.id, _PROMPT_VERSION_LABEL, properties=props)
    for trace_id in trace_ids:
        engine.add_edge(node.id, trace_id, _DERIVED_FROM_EDGE)


def _report(agent_id: str, status: str, **extra: Any) -> dict[str, Any]:
    return {"agent_id": agent_id, "status": status, **extra}


def _eligible_outcomes(
    engine: Any, agent_id: str, min_outcomes: int
) -> tuple[list[RunOutcome], str]:
    """Return new outcomes past the agent cursor, or an idle reason."""
    if _pending_proposal_exists(engine, agent_id):
        return [], "pending_review"
    try:
        cursor = load_trace_cursor(engine, _consumer(agent_id))
    except (RuntimeError, ValueError):
        return [], "cursor_unavailable"
    outcomes = gather_agent_run_outcomes(
        engine, agent_id, after_sequence=cursor.event_sequence
    )
    if len(outcomes) < min_outcomes:
        return [], "no_data"
    if all(outcome.success for outcome in outcomes):
        return [], "no_failures"
    return outcomes, ""


def propose_prompt_from_run_outcomes(
    engine: Any,
    agent_id: str,
    prompt_path: str | Path,
    *,
    optimizer: PromptOptimizer | None = None,
    min_outcomes: int = MIN_RUN_OUTCOMES,
) -> dict[str, Any]:
    """Run one propose-only pass for one agent and return a report.

    Statuses: ``proposed`` (candidate recorded), ``pending_review`` (an earlier
    candidate awaits review), ``no_data``/``no_failures`` (nothing to learn),
    ``cursor_unavailable`` or ``error`` (the cursor stays put for a retry).
    """
    if engine is None or not hasattr(engine, "query_cypher"):
        return _report(agent_id, "no_data")
    outcomes, idle = _eligible_outcomes(engine, agent_id, min_outcomes)
    if idle:
        return _report(agent_id, idle)
    return _optimize_and_record(
        engine, agent_id, prompt_path, outcomes, optimizer or native_prompt_optimizer
    )


def _optimize_and_record(
    engine: Any,
    agent_id: str,
    prompt_path: str | Path,
    outcomes: list[RunOutcome],
    optimizer: PromptOptimizer,
) -> dict[str, Any]:
    """Optimize over ``outcomes``, record the candidate, then advance the cursor."""
    from agent_utilities.prompting.structured import StructuredPrompt

    from .evolve_agent import EvolveAgent

    baseline = StructuredPrompt.load(prompt_path)
    trainset = [outcome.to_training_row() for outcome in outcomes]
    result = optimizer(baseline.model_dump(exclude_none=True), trainset, engine=engine)
    if result is None or not getattr(result, "compiled_state", None):
        return _report(agent_id, "error", detail="optimizer_returned_no_candidate")

    candidate = EvolveAgent._compiled_prompt_candidate(
        str(prompt_path), result.compiled_state
    )
    node = _candidate_node(agent_id, baseline, candidate, result, outcomes)
    trace_ids = [outcome.trace_id for outcome in outcomes]
    cursor = TraceCursor.from_rows(
        [{"event_sequence": o.event_sequence} for o in outcomes]
    )
    try:
        _persist_candidate(engine, node, result.compiled_state, trace_ids)
        save_trace_cursor(engine, _consumer(agent_id), cursor)
    except Exception as exc:  # report, never raise into the sweep
        logger.error("prompt_evolution: candidate persist failed: %s", exc)
        return _report(agent_id, "error", detail="candidate_persist_failed")
    logger.info(
        "prompt_evolution: proposed %s from %d run outcome(s)", node.id, len(trace_ids)
    )
    return _report(agent_id, "proposed", version_id=node.id, trace_count=len(trace_ids))


def packaged_prompt_targets() -> dict[str, Path]:
    """Map agent ids to prompt blueprints: packaged base, then the XDG overlay."""
    from agent_utilities.core.paths import prompts_dir

    packaged = Path(__file__).resolve().parent.parent / "prompts"
    targets: dict[str, Path] = {}
    for directory in (packaged, prompts_dir()):
        if directory.is_dir():
            targets.update(
                {path.stem: path for path in sorted(directory.glob("*.json"))}
            )
    return targets


def _sweep_status(agents: dict[str, dict[str, Any]]) -> str:
    statuses = {report["status"] for report in agents.values()}
    if "proposed" in statuses:
        return "proposed"
    if "error" in statuses:
        return "error"
    return "no_data"


def run_prompt_evolution_sweep(
    engine: Any,
    *,
    targets: Mapping[str, str | Path] | None = None,
    optimizer: PromptOptimizer | None = None,
    min_outcomes: int = MIN_RUN_OUTCOMES,
) -> dict[str, Any]:
    """Run one propose-only pass for every known agent prompt.

    Returns the sweep-shaped report the native optimization sweep expects:
    ``status`` is ``proposed``, ``error`` or ``no_data`` (idle).
    """
    if engine is None or not hasattr(engine, "query_cypher"):
        return {"status": "no_data", "agents": {}}
    prompts = targets if targets is not None else packaged_prompt_targets()
    agents: dict[str, dict[str, Any]] = {}
    for agent_id, path in prompts.items():
        try:
            agents[agent_id] = propose_prompt_from_run_outcomes(
                engine, agent_id, path, optimizer=optimizer, min_outcomes=min_outcomes
            )
        except Exception as exc:  # one agent never blocks the rest
            logger.error("prompt_evolution: agent pass failed: %s", exc)
            agents[agent_id] = _report(agent_id, "error", detail="agent_pass_failed")
    status = _sweep_status(agents)
    report: dict[str, Any] = {"status": status, "agents": agents}
    if status == "error":
        report["error_code"] = "prompt_evolution_failed"
    return report
