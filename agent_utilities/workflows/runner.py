"""Workflow Runner — Execute Stored Workflows via Agent Runner.

CONCEPT:AU-ORCH.execution.workflow-lifecycle-management — Workflow Execution Engine

Bridges stored ``GraphPlan`` objects to live agent execution via
``run_agent()``. Respects step dependencies, executes parallel groups
concurrently, and tracks the entire execution as a Langfuse session.

Pipeline::

    ┌──────────────┐   load_workflow()   ┌──────────────┐
    │  KG Store     │ ──────────────────► │  GraphPlan    │
    └──────────────┘                      └──────┬───────┘
                                                 │
                                        execute_plan()
                                                 │
                        ┌────────────────────────┼────────────────────────┐
                        │                        │                        │
                  ┌─────▼─────┐           ┌──────▼──────┐          ┌─────▼─────┐
                  │ Step 0     │           │ Step 1      │          │ Step 2     │
                  │ (parallel) │           │ (parallel)  │          │ (depends   │
                  │ run_agent()│           │ run_agent() │          │  on 0,1)   │
                  └─────┬─────┘           └──────┬──────┘          └─────┬─────┘
                        │                        │                        │
                        └────────────────────────┼────────────────────────┘
                                                 │
                                        ┌────────▼────────┐
                                        │ WorkflowResult   │
                                        │ (outputs, mermaid,│
                                        │  trace_ids)       │
                                        └─────────────────┘

Usage::

    from agent_utilities.workflows.runner import WorkflowRunner

    runner = WorkflowRunner()

    # Execute a stored workflow by name
    result = await runner.execute_by_name("container_health_check", engine)

    # Execute a GraphPlan directly
    result = await runner.execute(plan, engine)

    # Access results
    print(result.summary())
    print(result.mermaid)

Process lineage close-out (CONCEPT:AU-ORCH.execution.best-effort-provenance)
---------------------------------------------

When an executed workflow was compiled from a descriptive BusinessProcess
(it carries a ``(:WorkflowDefinition)-[:REALIZES]->(:BusinessProcess)`` edge,
ORCH-1.41), completion closes the provenance loop: the run's ``RunTrace``
gets an ``EXECUTED_PROCESS`` edge to the BusinessProcess, so "which runs
executed this harvested process?" is a graph traversal.

**Lineage sink seam.** ``WorkflowRunner(lineage_sink=...)`` accepts an
optional callable invoked once per close-out with a normalized lineage
record::

    {
        "process_id": ...,            # BusinessProcess node id
        "process_external_id": ...,   # e.g. the Egeria GUID (externalToolId)
        "workflow_id": ...,           # WorkflowDefinition node id
        "workflow_name": ...,
        "run_id": ...,                # RunTrace session id
        "status": ...,                # completed | partial | failed
        "completed_steps": ..., "failed_steps": ...,
        "duration_ms": ..., "timestamp": ...,  # ISO-8601 UTC
    }

agent-utilities never imports a metadata server: a deployment wires the sink
to its lineage system of record — e.g. egeria-mcp's ``assert_lineage``::

    runner = WorkflowRunner(
        lineage_sink=lambda rec: egeria.assert_lineage(
            source_guid=rec["process_external_id"],
            target_name=rec["run_id"],
            status=rec["status"],
        )
    )

The sink is best-effort (exceptions are logged, never raised) and the
default ``None`` keeps the legacy behavior bit-for-bit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_utilities.models.graph import GraphPlan

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

# Registry for tracking workflow runs (both active and completed) within this process
_active_workflows: dict[str, WorkflowResult] = {}

# CONCEPT:AU-ORCH.execution.workflow-lifecycle-management — governance gate/approval step kind
# (autonomous-sdlc-loop-design.md §7.1 delta 3). A step whose ``kind`` is one of these is
# a suspend/resume checkpoint, not an ordinary agent-executed step.
GATE_KINDS = frozenset({"gate", "approval"})

# Statuses a StepResult can carry beyond "completed"/"failed" for a gate step.
STATUS_BLOCKED = "blocked_on_approval"
STATUS_REJECTED = "rejected"
STATUS_SKIPPED = "skipped"

# CONCEPT:AU-ORCH.routing.functional-role-resolution — evidentiary model-tier routing (ATG paper
# idea #3, ``models/sdd.py``'s ``Task.model_tier``). Deliberately NOT a new
# model registry: a "small"/"cheap" step (well-scoped, already-decomposed —
# exactly the case the paper shows a small model handles fine) maps onto the
# existing ``run_agent(reasoning_effort=...)`` per-call knob rather than a new
# tier->model-id table. Unrecognized/absent tiers pass ``None`` through
# unchanged (native default reasoning — today's behavior).
MODEL_TIER_REASONING_EFFORT: dict[str, str] = {
    "small": "low",
    "cheap": "low",
    "large": "high",
}


def _is_gate_step(step: Any) -> bool:
    return str(getattr(step, "kind", "task") or "task").lower() in GATE_KINDS


def _default_gate_checker(engine: Any, step: Any) -> str | None:
    """Default gate satisfaction check (§7.1 delta 3).

    A gate step is satisfied when an out-edge ``step -[:satisfiedBy]-> X``
    exists on the graph (written externally — an approval recorded, a Camunda
    user task completed, a DPIA approved). Returns ``"approved"``,
    ``"rejected"`` (edge carries ``decision: "rejected"``), or ``None``
    (pending — the run stays suspended). Best-effort: any read failure or a
    missing engine degrades to ``None`` (never silently auto-approves).
    """
    step_id = getattr(step, "id", "") or ""
    if not step_id or engine is None:
        return None

    def _decision(edata: dict[str, Any]) -> str:
        return (
            "rejected"
            if str(edata.get("decision") or "").lower() == "rejected"
            else "approved"
        )

    graph = getattr(engine, "graph", None)
    if graph is not None:
        try:
            for _src, _tgt, edata in graph.out_edges(step_id, data=True):
                rel = str(
                    (edata or {}).get("type") or (edata or {}).get("rel_type") or ""
                )
                if rel == "satisfiedBy":
                    return _decision(edata or {})
        except Exception as exc:  # noqa: BLE001 — read is best-effort
            logger.debug("[ORCH.gate] compute-graph gate check failed: %s", exc)

    backend = getattr(engine, "backend", None)
    if backend is not None:
        try:
            rows = backend.execute(
                "MATCH (s)-[r:satisfiedBy]->(x) WHERE s.id = $sid "
                "RETURN r.decision AS decision LIMIT 1",
                {"sid": step_id},
            )
            if rows:
                return _decision({"decision": rows[0].get("decision")})
        except Exception as exc:  # noqa: BLE001 — read is best-effort
            logger.debug("[ORCH.gate] backend gate check failed: %s", exc)
    return None


@dataclass
class StepResult:
    """Result from a single workflow step execution.

    CONCEPT:AU-ORCH.execution.workflow-lifecycle-management — Step Execution Result
    """

    step_index: int
    node_id: str
    task: str
    output: str
    status: str  # "completed", "failed", "skipped"
    duration_ms: float
    error: str | None = None
    trace_id: str | None = None
    # CONCEPT:AU-ORCH.routing.functional-role-resolution — the model-tier hint this step ran
    # under (ATG paper idea #3, ``models/sdd.py``'s ``Task.model_tier``), recorded
    # for observability even when unset (the default, ordinary-tier run).
    model_tier: str | None = None


@dataclass
class WorkflowResult:
    """Aggregate result from a full workflow execution.

    CONCEPT:AU-ORCH.execution.workflow-lifecycle-management — Workflow Execution Result

    Attributes:
        workflow_name: Name of the executed workflow.
        session_id: Langfuse session ID grouping all step traces.
        step_results: Per-step execution results.
        total_duration_ms: Wall-clock time for the entire workflow.
        status: Overall status ("completed", "partial", "failed").
        mermaid: Generated mermaid diagram of the execution.
    """

    workflow_name: str
    session_id: str
    step_results: list[StepResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    status: str = "pending"
    mermaid: str = ""

    @property
    def completed_steps(self) -> int:
        return sum(1 for r in self.step_results if r.status == "completed")

    @property
    def failed_steps(self) -> int:
        return sum(1 for r in self.step_results if r.status == "failed")

    def summary(self) -> str:
        """Generate a human-readable execution summary."""
        lines = [
            f"Workflow: {self.workflow_name}",
            f"Session: {self.session_id}",
            f"Status: {self.status}",
            f"Duration: {self.total_duration_ms:.0f}ms",
            f"Steps: {self.completed_steps}/{len(self.step_results)} completed, "
            f"{self.failed_steps} failed",
            "",
        ]
        for r in self.step_results:
            status_icon = "✅" if r.status == "completed" else "❌"
            lines.append(
                f"  {status_icon} [{r.step_index}] {r.node_id}: "
                f"{r.duration_ms:.0f}ms — {r.task[:60]}..."
            )
            if r.error:
                lines.append(f"      Error: {r.error[:120]}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "workflow_name": self.workflow_name,
            "session_id": self.session_id,
            "status": self.status,
            "total_duration_ms": self.total_duration_ms,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "step_results": [
                {
                    "step_index": r.step_index,
                    "node_id": r.node_id,
                    "task": r.task,
                    "output": r.output[:500],
                    "status": r.status,
                    "duration_ms": r.duration_ms,
                    "error": r.error,
                    "trace_id": r.trace_id,
                }
                for r in self.step_results
            ],
            "mermaid": self.mermaid,
        }


class WorkflowRunner:
    """Execute stored GraphPlan workflows via the agent_runner pipeline.

    CONCEPT:AU-ORCH.execution.workflow-lifecycle-management — Workflow Execution Engine

    Orchestrates step execution respecting dependencies:
    - Steps with no dependencies run in parallel
    - Steps with dependencies wait for all predecessors
    - Each step invokes ``run_agent()`` with the step's agent and task

    The entire workflow is tracked as a Langfuse session for
    end-to-end observability.

    For new workflows, prefer ``execute_via_parallel_engine()`` which
    routes through the single ``ParallelEngine`` entry point.
    """

    def __init__(
        self,
        max_steps_per_agent: int = 10,
        lineage_sink: Any | None = None,
        gate_checker: Callable[[Any, Any], str | None] | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            max_steps_per_agent: Max graph steps per individual agent call.
            lineage_sink: Optional callable receiving one normalized lineage
                record per process close-out (CONCEPT:AU-ORCH.execution.best-effort-provenance — see the
                module docstring). Lets a deployment wire egeria-mcp's
                ``assert_lineage`` (or any lineage SoR) without
                agent-utilities depending on it. Best-effort; default None.
            gate_checker: Optional ``(engine, step) -> "approved"|"rejected"|None``
                callable consulted for every ``kind="gate"``/``"approval"`` step
                (CONCEPT:AU-ORCH.execution.workflow-lifecycle-management, §7.1 delta 3). Defaults to
                :func:`_default_gate_checker` (a ``:satisfiedBy`` out-edge on the
                step). Overridable so a deployment can bind gate satisfaction to
                its own governance system (Camunda task completion, an
                ``:EscalationRequest`` resolution, ...).
        """
        self.max_steps_per_agent = max_steps_per_agent
        self.lineage_sink = lineage_sink
        self.gate_checker = gate_checker or _default_gate_checker

    async def execute_via_parallel_engine(
        self,
        plan: GraphPlan,
        engine: IntelligenceGraphEngine,
        workflow_name: str = "unnamed",
        query: str = "",
    ):
        """Execute a GraphPlan by delegating to ParallelEngine.

        CONCEPT:AU-ORCH.execution.workflow-parallel-bridge — Workflow → ParallelEngine Bridge

        Converts the ``GraphPlan`` to an ``ExecutionManifest`` and invokes
        ``ParallelEngine.execute()``, ensuring a single execution path for
        all workflow types.

        Args:
            plan: The GraphPlan to execute.
            engine: IntelligenceGraphEngine for resolution.
            workflow_name: Name for the workflow.
            query: Original user query.

        Returns:
            ``ExecutionResult`` from ``ParallelEngine``.
        """
        from agent_utilities.graph.parallel_engine import ParallelEngine
        from agent_utilities.models.execution_manifest import ExecutionManifest

        manifest = ExecutionManifest.from_graph_plan(
            plan,
            name=workflow_name,
            query=query,
        )
        pe = ParallelEngine(engine=engine)
        return await pe.execute(manifest)

    async def execute(
        self,
        plan: GraphPlan,
        engine: IntelligenceGraphEngine,
        workflow_name: str = "unnamed",
        trace_session: str | None = None,
        task: str | None = None,
    ) -> WorkflowResult:
        """Execute a GraphPlan step by step, respecting dependencies.

        CONCEPT:AU-ORCH.execution.workflow-lifecycle-management — Plan Execution
        """
        session_id = trace_session or f"wf-{uuid.uuid4().hex[:8]}"

        exec_res = await self.execute_via_parallel_engine(
            plan=plan,
            engine=engine,
            workflow_name=workflow_name,
            query=task
            or (
                plan.metadata.get("query", "")
                if hasattr(plan, "metadata") and plan.metadata
                else ""
            ),
        )

        step_results = []
        for wave_idx, w_res in enumerate(exec_res.wave_results):
            for r in w_res.results:
                step_results.append(
                    StepResult(
                        step_index=wave_idx,
                        node_id=r.agent_id,
                        # CONCEPT:AU-ORCH.execution.workflow-engine-wiring — ``AgentExecutionResult`` (the ParallelEngine
                        # wave result) carries no ``task`` field; it lives in its
                        # ``metadata``. Reading ``r.task`` raised AttributeError and
                        # crashed every wired ``execute_workflow`` run after the steps
                        # had already executed. Fall back through metadata → agent_id.
                        task=(
                            getattr(r, "task", None)
                            or (r.metadata or {}).get("task")
                            or r.agent_id
                        ),
                        output=r.output,
                        status="completed" if r.success else "failed",
                        duration_ms=r.duration_ms,
                        error=r.error or None,
                        trace_id=r.metadata.get("trace_id") or exec_res.execution_id,
                    )
                )

        status = "completed" if exec_res.success else "failed"
        if exec_res.wave_results and not exec_res.success:
            if any(r.success for w in exec_res.wave_results for r in w.results):
                status = "partial"

        result = WorkflowResult(
            workflow_name=workflow_name,
            session_id=exec_res.execution_id or session_id,
            step_results=step_results,
            total_duration_ms=exec_res.total_duration_ms,
            status=status,
            mermaid=exec_res.mermaid or "",
        )

        _active_workflows[result.session_id] = result
        # CONCEPT:AU-ORCH.execution.best-effort-provenance — close the descriptive↔executable provenance loop
        # for workflows compiled from a BusinessProcess (REALIZES, ORCH-1.41).
        self._close_out_process_lineage(engine, workflow_name, result)
        return result

    # ------------------------------------------------------------------
    # Process lineage close-out (CONCEPT:AU-ORCH.execution.best-effort-provenance)
    # ------------------------------------------------------------------

    def _find_realized_process(
        self, engine: IntelligenceGraphEngine, workflow_name: str
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        """Resolve (workflow_id, process_id, process_props) via REALIZES.

        Returns ``(None, None, {})`` for workflows not compiled from a
        BusinessProcess — the common case, which must stay zero-cost-ish.
        """
        backend = getattr(engine, "backend", None)
        if backend is not None:
            try:
                rows = backend.execute(
                    "MATCH (w:WorkflowDefinition)-[:REALIZES]->(p) "
                    "WHERE w.name = $name "
                    "RETURN w.id AS wid, p.id AS pid, "
                    "p.externalToolId AS external_id LIMIT 1",
                    {"name": workflow_name},
                )
                if rows:
                    return (
                        rows[0].get("wid"),
                        rows[0].get("pid"),
                        {"externalToolId": rows[0].get("external_id")},
                    )
            except Exception as exc:  # noqa: BLE001 — fall through to compute graph
                logger.debug("[ORCH-1.43] backend REALIZES lookup failed: %s", exc)

        graph = getattr(engine, "graph", None)
        if graph is not None:
            try:
                for nid, data in graph.nodes(data=True):
                    if (
                        data.get("type") != "WorkflowDefinition"
                        or data.get("name") != workflow_name
                    ):
                        continue
                    for _src, tgt, edata in graph.out_edges(nid, data=True):
                        rel = str(
                            (edata or {}).get("type")
                            or (edata or {}).get("rel_type")
                            or ""
                        ).upper()
                        if rel == "REALIZES":
                            try:
                                props = dict(graph.nodes[tgt])
                            except Exception:  # noqa: BLE001
                                props = {}
                            return nid, tgt, props
                    return nid, None, {}
            except Exception as exc:  # noqa: BLE001 — provenance is best-effort
                logger.debug("[ORCH-1.43] compute REALIZES lookup failed: %s", exc)
        return None, None, {}

    def _close_out_process_lineage(
        self,
        engine: IntelligenceGraphEngine,
        workflow_name: str,
        result: WorkflowResult,
    ) -> None:
        """Record RunTrace→BusinessProcess provenance + feed the lineage sink.

        CONCEPT:AU-ORCH.execution.best-effort-provenance — best-effort by design: lineage close-out must
        never fail a completed (or even failed) workflow run.
        """
        if engine is None:
            return
        try:
            workflow_id, process_id, process_props = self._find_realized_process(
                engine, workflow_name
            )
            if not process_id:
                return

            import time as _time

            ts = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
            trace_id = f"trace:{result.session_id}"
            # Upsert the workflow-level RunTrace (agent_runner's per-step
            # traces use their own run ids; this is the run's umbrella node).
            engine.add_node(
                trace_id,
                "RunTrace",
                properties={
                    "workflow_name": workflow_name,
                    "workflow_id": workflow_id,
                    "status": result.status,
                    "duration_ms": round(result.total_duration_ms, 1),
                    "timestamp": ts,
                },
            )
            engine.link_nodes(
                trace_id,
                process_id,
                "EXECUTED_PROCESS",
                properties={"status": result.status, "workflow_id": workflow_id},
            )
            logger.info(
                "[ORCH-1.43] lineage close-out: %s EXECUTED_PROCESS %s (%s)",
                trace_id,
                process_id,
                result.status,
            )

            if self.lineage_sink is not None:
                record = {
                    "process_id": process_id,
                    "process_external_id": process_props.get("externalToolId"),
                    "workflow_id": workflow_id,
                    "workflow_name": workflow_name,
                    "run_id": result.session_id,
                    "status": result.status,
                    "completed_steps": result.completed_steps,
                    "failed_steps": result.failed_steps,
                    "duration_ms": result.total_duration_ms,
                    "timestamp": ts,
                }
                try:
                    self.lineage_sink(record)
                except Exception as exc:  # noqa: BLE001 — sink is best-effort
                    logger.warning("[ORCH-1.43] lineage_sink failed: %s", exc)
        except Exception as exc:  # noqa: BLE001 — never fail the run on lineage
            logger.debug("[ORCH-1.43] lineage close-out skipped: %s", exc)

    async def execute_by_name(
        self,
        workflow_name: str,
        engine: IntelligenceGraphEngine,
        trace_session: str | None = None,
        task: str | None = None,
    ) -> WorkflowResult:
        """Load a stored workflow by name from the KG and execute its step-DAG.

        CONCEPT:AU-ORCH.execution.workflow-lifecycle-management / ORCH-1.95 — Named Workflow Execution. Loads the stored
        ``WorkflowDefinition``/``WorkflowStep`` DAG (the KG-2.97 ``WorkflowStore``
        shape) and runs each step through :func:`run_agent` in dependency-wave order
        — so a step that names a ``Server``/ingested ``Skill`` resolves to its real
        MCP toolset and runs the tool-calling loop on the LOCAL LLM, with each step's
        RunTrace + :ToolCall provenance (KG-2.296) written for free. This is the
        execution half of "ingested workflow → executed", routed here from
        ``graph_orchestrate action=execute_workflow``.
        """
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        store = WorkflowStore(engine)
        plan = store.load_workflow(workflow_name)
        if plan is None:
            raise ValueError(f"Workflow '{workflow_name}' not found in KG or catalog")

        return await self._execute_plan_via_agents(
            plan=plan,
            engine=engine,
            workflow_name=workflow_name,
            trace_session=trace_session,
            task=task,
        )

    async def resume(
        self,
        workflow_name: str,
        engine: IntelligenceGraphEngine,
        session_id: str,
        task: str | None = None,
    ) -> WorkflowResult:
        """Resume a run that :meth:`_execute_plan_via_agents` SUSPENDED on a gate.

        CONCEPT:AU-ORCH.execution.workflow-lifecycle-management — §7.1 delta 3. A suspended run
        persisted its per-step state as a ``:WorkflowRun {status:"suspended"}`` node
        (see :meth:`_persist_run_state`). Resuming reloads that state and re-drives
        the DAG from where it blocked; each still-blocked gate is re-checked via
        ``gate_checker`` (now that an ``:satisfiedBy`` edge — an approval / Camunda
        task completion / :EscalationRequest resolution — may have been recorded)
        and, if satisfied, the run continues. Idempotent: already-completed steps are
        NOT re-executed. Falls back to a fresh run when no suspended state exists.
        """
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        store = WorkflowStore(engine)
        plan = store.load_workflow(workflow_name)
        if plan is None:
            raise ValueError(f"Workflow '{workflow_name}' not found in KG or catalog")
        prior = self._load_run_state(engine, session_id)
        return await self._execute_plan_via_agents(
            plan=plan,
            engine=engine,
            workflow_name=workflow_name,
            trace_session=session_id,
            task=task,
            resume_state=prior,
        )

    async def resume_localized(
        self,
        workflow_name: str,
        engine: IntelligenceGraphEngine,
        session_id: str,
        failed_step: str,
        task: str | None = None,
        prior_result: WorkflowResult | None = None,
    ) -> WorkflowResult:
        """Resume a run after a STEP FAILURE (not a gate), re-executing ONLY the
        region ``failed_step`` actually invalidates — Atomic Task Graph paper
        idea #1 (arXiv:2607.01942,
        ``reports/paper-analysis-2607.01942.md`` §4 Rank 1), the same primitive
        :mod:`agent_utilities.observability.ci_recycle` uses for a CI failure.

        Unlike :meth:`resume` (which only ever re-drives from a SUSPENDED gate,
        never discarding a completed step), this computes
        :func:`agent_utilities.workflows.localized_repair.localized_repair_region`
        from ``failed_step`` over the plan's own ``TRANSITION_TO`` DAG and drops
        ONLY the invalidated steps from the prior run's completed/satisfied
        state before re-driving — every PRESERVED step (upstream + sibling
        branches ``failed_step`` never fed) keeps its already-recorded
        ``:RunTrace``/``:ToolCall`` result and is NOT re-executed. This is the
        region-preserving repair the paper's failure-localization idea
        describes, applied to the existing suspend/resume state machine rather
        than a blind whole-plan retry.

        The prior run's completed-step state is sourced from ``prior_result``
        when the caller already has it in hand (the common case — a run just
        completed/failed and its ``WorkflowResult`` is right there; a plain
        completed/failed/partial run has NO persisted ``:WorkflowRun`` state,
        only a SUSPENDED one does, see :meth:`_persist_run_state`), else it
        falls back to :meth:`_load_run_state` (a gate-suspended run resumed
        into a subsequent step failure).
        """
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore
        from agent_utilities.workflows.localized_repair import (
            localized_repair_region,
        )

        store = WorkflowStore(engine)
        plan = store.load_workflow(workflow_name)
        if plan is None:
            raise ValueError(f"Workflow '{workflow_name}' not found in KG or catalog")

        all_step_ids = [getattr(s, "id", "") or "" for s in plan.steps]
        region = localized_repair_region(
            failed_step, engine=engine, all_nodes=all_step_ids
        )
        invalidated = set(region["invalidated"]) | {failed_step}

        if prior_result is not None:
            prior_completed = {
                r.node_id: {
                    "output": r.output,
                    "status": r.status,
                    "node_id": r.node_id,
                }
                for r in prior_result.step_results
            }
            prior_satisfied = {
                r.node_id for r in prior_result.step_results if r.status == "completed"
            }
        else:
            prior = self._load_run_state(engine, session_id)
            prior_completed = dict(prior.get("completed") or {})
            prior_satisfied = set(prior.get("satisfied") or set())

        trimmed = {
            "completed": {
                sid: rec
                for sid, rec in prior_completed.items()
                if sid not in invalidated
            },
            "satisfied": {sid for sid in prior_satisfied if sid not in invalidated},
        }

        result = await self._execute_plan_via_agents(
            plan=plan,
            engine=engine,
            workflow_name=workflow_name,
            trace_session=session_id,
            task=task,
            resume_state=trimmed,
        )
        logger.info(
            "[ORCH.repair] workflow %s localized repair from %s: invalidated=%s preserved=%s",
            workflow_name,
            failed_step,
            sorted(invalidated),
            region["preserved"],
        )
        return result

    # ------------------------------------------------------------------
    # Suspend/resume state persistence (CONCEPT:AU-ORCH.execution.workflow-lifecycle-management, §7.1 delta 3)
    # ------------------------------------------------------------------

    def _persist_run_state(
        self,
        engine: Any,
        session_id: str,
        workflow_name: str,
        status: str,
        completed: dict[str, StepResult],
        satisfied: set[str],
        blocked_on: list[str],
    ) -> None:
        """Best-effort persist a run's step state so it can be resumed after a gate.

        Writes a ``:WorkflowRun`` node keyed on the session id carrying JSON of the
        completed step outputs + statuses, the satisfied set, and the blocked gate
        ids. Never raises — persistence must not fail a suspend.
        """
        if engine is None:
            return
        try:
            payload = {
                sid: {"output": r.output, "status": r.status, "node_id": r.node_id}
                for sid, r in completed.items()
            }
            engine.add_node(
                f"workflowrun:{session_id}",
                "WorkflowRun",
                properties={
                    "session_id": session_id,
                    "workflow_name": workflow_name,
                    "status": status,
                    "completed_json": json.dumps(payload, default=str),
                    "satisfied_json": json.dumps(sorted(satisfied)),
                    "blocked_on_json": json.dumps(sorted(blocked_on)),
                },
            )
        except Exception as exc:  # noqa: BLE001 — persistence is best-effort
            logger.debug("[ORCH.gate] run-state persist skipped: %s", exc)

    def _load_run_state(self, engine: Any, session_id: str) -> dict[str, Any]:
        """Reload a suspended run's persisted step state (``{}`` if none)."""
        if engine is None:
            return {}
        node_id = f"workflowrun:{session_id}"
        raw: dict[str, Any] = {}
        graph = getattr(engine, "graph", None)
        if graph is not None:
            try:
                data = graph.nodes[node_id]
                if data:
                    raw = dict(data)
            except Exception:  # noqa: BLE001 — fall through to backend
                raw = {}
        if not raw:
            backend = getattr(engine, "backend", None)
            if backend is not None:
                try:
                    rows = backend.execute(
                        "MATCH (r:WorkflowRun) WHERE r.id = $rid RETURN r",
                        {"rid": node_id},
                    )
                    if rows and isinstance(rows[0].get("r"), dict):
                        raw = dict(rows[0]["r"])
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[ORCH.gate] run-state load failed: %s", exc)
        if not raw:
            return {}
        try:
            return {
                "completed": json.loads(raw.get("completed_json") or "{}"),
                "satisfied": set(json.loads(raw.get("satisfied_json") or "[]")),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("[ORCH.gate] run-state decode failed: %s", exc)
            return {}

    async def _execute_plan_via_agents(
        self,
        plan: GraphPlan,
        engine: IntelligenceGraphEngine,
        workflow_name: str,
        trace_session: str | None = None,
        task: str | None = None,
        resume_state: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """Run a stored plan's steps via :func:`run_agent`, respecting dependencies.

        CONCEPT:AU-ORCH.execution.workflow-engine-wiring — wires the EXISTING ``run_agent`` executor (not a new one)
        into named-workflow execution: steps with satisfied dependencies run
        concurrently as a wave, each via ``run_agent(step.id, step.task, engine=...)``
        on the local LLM with its resolved MCP toolset. Upstream step outputs are
        threaded into a dependent step's context. ``run_agent`` records each step's
        own RunTrace + :ToolCall nodes, so workflow execution is fully visible over
        graph-os with zero extra plumbing.

        CONCEPT:AU-ORCH.execution.workflow-lifecycle-management — §7.1 delta 3: a ready step whose
        ``kind`` is ``"gate"``/``"approval"`` is NOT run by an agent; instead the
        ``gate_checker`` is consulted. Approved → the step completes and its
        on-success dependents proceed; rejected → the step and its on-success
        downstream are marked skipped (its ``on_reject`` target, if any, still runs);
        pending → the whole run SUSPENDS: state is persisted (``:WorkflowRun``) and a
        ``status="suspended"`` WorkflowResult is returned rather than blocking. Call
        :meth:`resume` once the gate's ``:satisfiedBy`` edge is recorded.
        """
        import time as _time

        from agent_utilities.orchestration.agent_runner import run_agent

        session_id = trace_session or f"wf-{uuid.uuid4().hex[:8]}"
        wf_started = _time.monotonic()

        steps = list(plan.steps)
        # Resolve per-step (agent_name, task) from the canonical WorkflowStep shape:
        # step.id is the resolvable agent/skill/server name, step.refined_subtask the
        # task (falling back to the step description / the workflow-level task).
        by_id: dict[str, Any] = {}
        for s in steps:
            sid = getattr(s, "id", "") or ""
            by_id[sid] = s
        completed: dict[str, StepResult] = {}
        outputs: dict[str, str] = {}
        # ``satisfied`` = step ids whose ON-SUCCESS path is cleared (completed tasks +
        # approved gates). A gate that rejected/pends is in ``completed`` but NOT here,
        # so its on-success dependents don't spuriously advance.
        satisfied: set[str] = set()

        # Rehydrate a resumed run's already-finished steps (idempotent resume).
        if resume_state:
            for sid, rec in (resume_state.get("completed") or {}).items():
                completed[sid] = StepResult(
                    step_index=0,
                    node_id=str(rec.get("node_id") or sid),
                    task="",
                    output=str(rec.get("output") or ""),
                    status=str(rec.get("status") or "completed"),
                    duration_ms=0.0,
                    trace_id=session_id,
                )
                outputs[sid] = str(rec.get("output") or "")
            satisfied |= set(resume_state.get("satisfied") or set())

        skipped: set[str] = set()
        wave_idx = 0
        suspended_gate: str | None = None

        def _mark_reject_downstream(gate_id: str, keep: str | None) -> None:
            """Skip the on-success transitive downstream of a rejected gate (except
            an explicit ``on_reject`` branch target ``keep``)."""
            frontier = [gate_id]
            seen: set[str] = set()
            while frontier:
                cur = frontier.pop()
                for s in steps:
                    sid = getattr(s, "id", "") or ""
                    if not sid or sid == keep or sid in seen:
                        continue
                    if cur in (getattr(s, "depends_on", None) or []):
                        seen.add(sid)
                        skipped.add(sid)
                        frontier.append(sid)

        def _remaining() -> list[Any]:
            return [
                s
                for s in steps
                if (getattr(s, "id", "") or "") not in completed
                and (getattr(s, "id", "") or "") not in skipped
            ]

        while _remaining() and suspended_gate is None:
            remaining = _remaining()
            ready = [
                s
                for s in remaining
                if all(
                    dep in completed or dep in skipped
                    for dep in (getattr(s, "depends_on", None) or [])
                )
            ]
            if not ready:
                # A dependency cycle / dangling dep — run the rest as one wave rather
                # than deadlock (the SHACL gate upstream guards malformed DAGs).
                ready = list(remaining)

            # Gate/approval steps are resolved by the gate_checker, not an agent.
            gate_steps = [s for s in ready if _is_gate_step(s)]
            agent_steps = [s for s in ready if not _is_gate_step(s)]

            gate_progressed = False
            for gstep in gate_steps:
                gsid = getattr(gstep, "id", "") or f"gate-{wave_idx}"
                verdict = self.gate_checker(engine, gstep)
                if verdict == "approved":
                    completed[gsid] = StepResult(
                        step_index=wave_idx,
                        node_id=gsid,
                        task=str(getattr(gstep, "refined_subtask", "") or gsid),
                        output="gate approved",
                        status="completed",
                        duration_ms=0.0,
                        trace_id=session_id,
                    )
                    outputs[gsid] = "gate approved"
                    satisfied.add(gsid)
                    gate_progressed = True
                elif verdict == "rejected":
                    completed[gsid] = StepResult(
                        step_index=wave_idx,
                        node_id=gsid,
                        task=str(getattr(gstep, "refined_subtask", "") or gsid),
                        output="gate rejected",
                        status=STATUS_REJECTED,
                        duration_ms=0.0,
                        error="gate rejected",
                        trace_id=session_id,
                    )
                    outputs[gsid] = "gate rejected"
                    _mark_reject_downstream(gsid, getattr(gstep, "on_reject", None))
                    gate_progressed = True
                else:
                    # Pending → suspend the whole run at the first blocking gate.
                    suspended_gate = gsid
                    break

            if suspended_gate is not None:
                break

            if not agent_steps:
                # Only gates this wave — if none progressed we would loop forever;
                # that can't happen here (a non-progressing gate suspends above).
                if not gate_progressed:
                    break
                wave_idx += 1
                continue

            async def _run_step(step: Any, wave: int = wave_idx) -> StepResult:
                sid = getattr(step, "id", "") or f"step-{wave}"
                step_task = (
                    getattr(step, "refined_subtask", None)
                    or getattr(step, "description", None)
                    or task
                    or sid
                )
                # Thread completed upstream outputs in as context.
                deps = getattr(step, "depends_on", None) or []
                ctx = "\n\n".join(
                    f"Output of '{d}':\n{outputs.get(d, '')}"
                    for d in deps
                    if outputs.get(d)
                )
                # CONCEPT:AU-ORCH.routing.functional-role-resolution — model-tier routing hint (ATG
                # paper idea #3). Only honored when the step didn't already pin an
                # exact model_id (which always wins); unrecognized/absent tiers pass
                # reasoning_effort=None through unchanged (today's default).
                tier = str(getattr(step, "model_tier", "") or "").lower() or None
                effort = (
                    MODEL_TIER_REASONING_EFFORT.get(tier)
                    if tier and not getattr(step, "model_id", None)
                    else None
                )
                t0 = _time.monotonic()
                try:
                    out = await run_agent(
                        agent_name=sid,
                        task=str(step_task),
                        engine=engine,
                        max_steps=self.max_steps_per_agent,
                        context=ctx or None,
                        session_id=session_id,
                        reasoning_effort=effort,
                    )
                    ok = not str(out).startswith("Agent execution failed")
                    return StepResult(
                        step_index=wave,
                        node_id=sid,
                        task=str(step_task),
                        output=str(out),
                        status="completed" if ok else "failed",
                        duration_ms=(_time.monotonic() - t0) * 1000,
                        error=None if ok else str(out)[:300],
                        trace_id=session_id,
                        model_tier=tier,
                    )
                except Exception as exc:  # noqa: BLE001 — one step must not kill the DAG
                    return StepResult(
                        step_index=wave,
                        node_id=sid,
                        task=str(step_task),
                        output="",
                        status="failed",
                        duration_ms=(_time.monotonic() - t0) * 1000,
                        error=str(exc)[:300],
                        model_tier=tier,
                        trace_id=session_id,
                    )

            results = await asyncio.gather(*(_run_step(s) for s in agent_steps))
            for step, res in zip(agent_steps, results, strict=False):
                sid = getattr(step, "id", "") or res.node_id
                completed[sid] = res
                outputs[sid] = res.output
                if res.status == "completed":
                    satisfied.add(sid)
            wave_idx += 1

        # A suspended run persists its state and returns early (not completed/failed).
        if suspended_gate is not None:
            for s in steps:
                if (getattr(s, "id", "") or "") == suspended_gate:
                    completed[suspended_gate] = StepResult(
                        step_index=wave_idx,
                        node_id=suspended_gate,
                        task=str(getattr(s, "refined_subtask", "") or suspended_gate),
                        output="",
                        status=STATUS_BLOCKED,
                        duration_ms=0.0,
                        error="awaiting gate satisfaction",
                        trace_id=session_id,
                    )
                    break
            step_results = [
                completed[getattr(s, "id", "")]
                for s in steps
                if getattr(s, "id", "") in completed
            ]
            result = WorkflowResult(
                workflow_name=workflow_name,
                session_id=session_id,
                step_results=step_results,
                total_duration_ms=(_time.monotonic() - wf_started) * 1000,
                status="suspended",
                mermaid=plan.to_mermaid(title=workflow_name)
                if hasattr(plan, "to_mermaid")
                else "",
            )
            _active_workflows[session_id] = result
            self._persist_run_state(
                engine,
                session_id,
                workflow_name,
                "suspended",
                {k: v for k, v in completed.items() if v.status != STATUS_BLOCKED},
                satisfied,
                [suspended_gate],
            )
            logger.info(
                "[ORCH.gate] workflow %s suspended on gate %s (session %s)",
                workflow_name,
                suspended_gate,
                session_id,
            )
            return result

        # Skipped (rejected-downstream) steps are recorded for visibility.
        for sid in skipped:
            if sid not in completed:
                completed[sid] = StepResult(
                    step_index=wave_idx,
                    node_id=sid,
                    task="",
                    output="",
                    status=STATUS_SKIPPED,
                    duration_ms=0.0,
                    error="skipped (upstream gate rejected)",
                    trace_id=session_id,
                )

        step_results = [
            completed[getattr(s, "id", "")]
            for s in steps
            if getattr(s, "id", "") in completed
        ]
        n_failed = sum(1 for r in step_results if r.status == "failed")
        n_ok = sum(1 for r in step_results if r.status == "completed")
        status = "completed" if n_failed == 0 else ("partial" if n_ok else "failed")

        result = WorkflowResult(
            workflow_name=workflow_name,
            session_id=session_id,
            step_results=step_results,
            total_duration_ms=(_time.monotonic() - wf_started) * 1000,
            status=status,
            mermaid=plan.to_mermaid(title=workflow_name)
            if hasattr(plan, "to_mermaid")
            else "",
        )
        _active_workflows[session_id] = result
        # Same provenance close-out as the manifest path (ORCH-1.43).
        self._close_out_process_lineage(engine, workflow_name, result)
        return result
