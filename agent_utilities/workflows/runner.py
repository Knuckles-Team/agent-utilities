"""Workflow Runner — Execute Stored Workflows via Agent Runner.

CONCEPT:ORCH-1.24 — Workflow Execution Engine

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
"""

from __future__ import annotations

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

    CONCEPT:ORCH-1.24 — Step Execution Result
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

    CONCEPT:ORCH-1.24 — Workflow Execution Result

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

    CONCEPT:ORCH-1.24 — Workflow Execution Engine

    Orchestrates step execution respecting dependencies:
    - Steps with no dependencies run in parallel
    - Steps with dependencies wait for all predecessors
    - Each step invokes ``run_agent()`` with the step's agent and task

    The entire workflow is tracked as a Langfuse session for
    end-to-end observability.

    For new workflows, prefer ``execute_via_parallel_engine()`` which
    routes through the single ``ParallelEngine`` entry point.
    """

    def __init__(self, max_steps_per_agent: int = 10) -> None:
        """Initialize the runner.

        Args:
            max_steps_per_agent: Max graph steps per individual agent call.
        """
        self.max_steps_per_agent = max_steps_per_agent

    async def execute_via_parallel_engine(
        self,
        plan: GraphPlan,
        engine: IntelligenceGraphEngine,
        workflow_name: str = "unnamed",
        query: str = "",
    ):
        """Execute a GraphPlan by delegating to ParallelEngine.

        CONCEPT:ORCH-1.25 — Workflow → ParallelEngine Bridge

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

        CONCEPT:ORCH-1.24 — Plan Execution
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
                        task=r.task,
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
        return result

    async def execute_by_name(
        self,
        workflow_name: str,
        engine: IntelligenceGraphEngine,
        trace_session: str | None = None,
        task: str | None = None,
    ) -> WorkflowResult:
        """Load a stored workflow by name from the KG and execute it.

        CONCEPT:ORCH-1.24 — Named Workflow Execution
        """
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        store = WorkflowStore(engine)
        plan = store.load_workflow(workflow_name)
        if plan is None:
            raise ValueError(f"Workflow '{workflow_name}' not found in KG or catalog")

        return await self.execute(
            plan=plan,
            engine=engine,
            workflow_name=workflow_name,
            trace_session=trace_session,
            task=task,
        )
