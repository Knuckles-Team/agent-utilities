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
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_utilities.models.graph import GraphPlan

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

# Registry for tracking workflow runs (both active and completed) within this process
_active_workflows: dict[str, WorkflowResult] = {}


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
    ) -> None:
        """Initialize the runner.

        Args:
            max_steps_per_agent: Max graph steps per individual agent call.
            lineage_sink: Optional callable receiving one normalized lineage
                record per process close-out (CONCEPT:AU-ORCH.execution.best-effort-provenance — see the
                module docstring). Lets a deployment wire egeria-mcp's
                ``assert_lineage`` (or any lineage SoR) without
                agent-utilities depending on it. Best-effort; default None.
        """
        self.max_steps_per_agent = max_steps_per_agent
        self.lineage_sink = lineage_sink

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

    async def _execute_plan_via_agents(
        self,
        plan: GraphPlan,
        engine: IntelligenceGraphEngine,
        workflow_name: str,
        trace_session: str | None = None,
        task: str | None = None,
    ) -> WorkflowResult:
        """Run a stored plan's steps via :func:`run_agent`, respecting dependencies.

        CONCEPT:AU-ORCH.execution.workflow-engine-wiring — wires the EXISTING ``run_agent`` executor (not a new one)
        into named-workflow execution: steps with satisfied dependencies run
        concurrently as a wave, each via ``run_agent(step.id, step.task, engine=...)``
        on the local LLM with its resolved MCP toolset. Upstream step outputs are
        threaded into a dependent step's context. ``run_agent`` records each step's
        own RunTrace + :ToolCall nodes, so workflow execution is fully visible over
        graph-os with zero extra plumbing.
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
        wave_idx = 0
        remaining = list(steps)

        while remaining:
            ready = [
                s
                for s in remaining
                if all(
                    dep in completed for dep in (getattr(s, "depends_on", None) or [])
                )
            ]
            if not ready:
                # A dependency cycle / dangling dep — run the rest as one wave rather
                # than deadlock (the SHACL gate upstream guards malformed DAGs).
                ready = list(remaining)

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
                t0 = _time.monotonic()
                try:
                    out = await run_agent(
                        agent_name=sid,
                        task=str(step_task),
                        engine=engine,
                        max_steps=self.max_steps_per_agent,
                        context=ctx or None,
                        session_id=session_id,
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
                        trace_id=session_id,
                    )

            results = await asyncio.gather(*(_run_step(s) for s in ready))
            for step, res in zip(ready, results, strict=False):
                sid = getattr(step, "id", "") or res.node_id
                completed[sid] = res
                outputs[sid] = res.output
            remaining = [
                s for s in remaining if (getattr(s, "id", "") or "") not in completed
            ]
            wave_idx += 1

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
