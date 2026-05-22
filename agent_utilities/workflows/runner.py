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

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_utilities.models.graph import ExecutionStep, GraphPlan

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
    """

    def __init__(self, max_steps_per_agent: int = 10) -> None:
        """Initialize the runner.

        Args:
            max_steps_per_agent: Max graph steps per individual agent call.
        """
        self.max_steps_per_agent = max_steps_per_agent

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

        Args:
            plan: The GraphPlan to execute.
            engine: IntelligenceGraphEngine for agent resolution.
            workflow_name: Name for the workflow (used in tracing).
            trace_session: Optional Langfuse session ID. Auto-generated if None.

        Returns:
            WorkflowResult with per-step outputs and aggregate metrics.
        """
        session_id = trace_session or f"wf-{uuid.uuid4().hex[:8]}"
        result = WorkflowResult(
            workflow_name=workflow_name,
            session_id=session_id,
        )
        _active_workflows[session_id] = result

        # Set up Langfuse session context
        try:
            from agent_utilities.harness.tracing import set_session_id

            set_session_id(session_id)
        except ImportError:
            pass

        logger.info(
            "[ORCH-1.24] Starting workflow '%s' with %d steps (session=%s)",
            workflow_name,
            len(plan.steps),
            session_id,
        )

        start_time = time.monotonic()

        # Build dependency graph: step_index → set of dependent step indices
        step_map: dict[str, int] = {}
        for i, step in enumerate(plan.steps):
            step_map[step.node_id] = i

        # Track completed step indices
        completed: set[int] = set()
        step_outputs: dict[int, str] = {}

        # Group steps into execution waves based on dependencies
        waves = self._build_execution_waves(plan)

        for wave_idx, wave in enumerate(waves):
            logger.info(
                "[ORCH-1.24] Executing wave %d: %d steps [%s]",
                wave_idx,
                len(wave),
                ", ".join(plan.steps[i].node_id for i in wave),
            )

            # Execute all steps in this wave concurrently
            tasks = []
            for step_idx in wave:
                step = plan.steps[step_idx]
                tasks.append(
                    self._execute_step(
                        step_index=step_idx,
                        step=step,
                        engine=engine,
                        prior_outputs=step_outputs,
                        task_input=task,
                    )
                )

            wave_results = await asyncio.gather(*tasks, return_exceptions=True)

            for step_idx, wave_result in zip(wave, wave_results, strict=False):
                if isinstance(wave_result, BaseException):
                    step_result = StepResult(
                        step_index=step_idx,
                        node_id=plan.steps[step_idx].node_id,
                        task=plan.steps[step_idx].refined_subtask or "",
                        output="",
                        status="failed",
                        duration_ms=0,
                        error=str(wave_result),
                    )
                else:
                    step_result = wave_result
                    if step_result.status == "completed":
                        completed.add(step_idx)
                        step_outputs[step_idx] = step_result.output

                result.step_results.append(step_result)

        # Finalize
        result.total_duration_ms = (time.monotonic() - start_time) * 1000

        if result.failed_steps == 0:
            result.status = "completed"
        elif result.completed_steps > 0:
            result.status = "partial"
        else:
            result.status = "failed"

        # Generate execution mermaid diagram
        try:
            result.mermaid = self._generate_execution_mermaid(plan, result)
        except Exception as e:
            logger.debug("Failed to generate mermaid: %s", e)

        # Record execution in KG
        self._record_execution(engine, result)

        logger.info(
            "[ORCH-1.24] Workflow '%s' %s: %d/%d steps, %.0fms",
            workflow_name,
            result.status,
            result.completed_steps,
            len(plan.steps),
            result.total_duration_ms,
        )

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

        Args:
            workflow_name: Name of the workflow in the KG.
            engine: IntelligenceGraphEngine for loading and execution.
            trace_session: Optional Langfuse session ID.
            task: Optional dynamic task/input to interpolate in steps.

        Returns:
            WorkflowResult with execution details.

        Raises:
            ValueError: If the workflow is not found in the KG.
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

    # -----------------------------------------------------------------------
    # Internal: Execution Waves
    # -----------------------------------------------------------------------

    def _build_execution_waves(
        self,
        plan: GraphPlan,
    ) -> list[list[int]]:
        """Build ordered execution waves from step dependencies.

        Steps with no unmet dependencies go into the earliest possible wave.
        This naturally handles both parallel and sequential steps.

        Returns:
            List of waves, each containing step indices to run concurrently.
        """
        num_steps = len(plan.steps)
        if num_steps == 0:
            return []

        # Build dependency index map
        step_id_to_idx: dict[str, int] = {}
        for i, step in enumerate(plan.steps):
            step_id_to_idx[step.node_id] = i

        # Resolve depends_on to indices
        deps: dict[int, set[int]] = {}
        for i, step in enumerate(plan.steps):
            dep_indices: set[int] = set()
            for dep_id in step.depends_on:
                if dep_id in step_id_to_idx:
                    dep_indices.add(step_id_to_idx[dep_id])
            deps[i] = dep_indices

        # Topological sort into waves
        waves: list[list[int]] = []
        remaining = set(range(num_steps))

        while remaining:
            # Find all steps whose dependencies are satisfied
            ready = []
            for idx in remaining:
                if deps[idx].issubset(set(range(num_steps)) - remaining):
                    ready.append(idx)

            if not ready:
                # Circular dependency — force remaining into one wave
                logger.warning(
                    "[ORCH-1.24] Circular dependency detected, forcing %d steps",
                    len(remaining),
                )
                ready = sorted(remaining)

            waves.append(sorted(ready))
            remaining -= set(ready)

        return waves

    # -----------------------------------------------------------------------
    # Internal: Step Execution
    # -----------------------------------------------------------------------

    async def _execute_step(
        self,
        step_index: int,
        step: ExecutionStep,
        engine: IntelligenceGraphEngine,
        prior_outputs: dict[int, str],
        task_input: str | None = None,
    ) -> StepResult:
        """Execute a single workflow step via run_agent().

        Args:
            step_index: Index of the step in the plan.
            step: The ExecutionStep to execute.
            engine: IntelligenceGraphEngine for agent resolution.
            prior_outputs: Outputs from previously completed steps.
            task_input: Optional dynamic task/input context/target to interpolate.

        Returns:
            StepResult with execution details.
        """
        from agent_utilities.orchestration.agent_runner import run_agent

        task = step.refined_subtask or f"Execute step: {step.node_id}"

        # Inject runtime task/input parameter if provided
        if task_input:
            if "{{task}}" in task:
                task = task.replace("{{task}}", task_input)
            elif "{{input}}" in task:
                task = task.replace("{{input}}", task_input)
            else:
                # If no explicit placeholder is used, append as helpful input context
                task = f"{task}\n\nTask Input: {task_input}"

        # Inject prior step context if access_list is set
        if step.access_list:
            context_parts = []
            for dep_id in step.access_list:
                # Find the dep index by node_id
                for idx, output in prior_outputs.items():
                    if output:
                        context_parts.append(
                            f"[Prior result from step {idx}]: {output[:300]}"
                        )
                        break

            if context_parts:
                task = f"{task}\n\nContext from prior steps:\n" + "\n".join(
                    context_parts
                )

        logger.info(
            "[ORCH-1.24] Executing step %d: agent=%s, task=%.80s...",
            step_index,
            step.node_id,
            task,
        )

        start = time.monotonic()
        try:
            output = await asyncio.wait_for(
                run_agent(
                    agent_name=step.node_id,
                    task=task,
                    max_steps=self.max_steps_per_agent,
                    engine=engine,
                ),
                timeout=step.timeout,
            )
            duration_ms = (time.monotonic() - start) * 1000

            return StepResult(
                step_index=step_index,
                node_id=step.node_id,
                task=step.refined_subtask or "",
                output=str(output),
                status="completed",
                duration_ms=duration_ms,
            )

        except TimeoutError:
            duration_ms = (time.monotonic() - start) * 1000
            return StepResult(
                step_index=step_index,
                node_id=step.node_id,
                task=step.refined_subtask or "",
                output="",
                status="failed",
                duration_ms=duration_ms,
                error=f"Step timed out after {step.timeout}s",
            )

        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            return StepResult(
                step_index=step_index,
                node_id=step.node_id,
                task=step.refined_subtask or "",
                output="",
                status="failed",
                duration_ms=duration_ms,
                error=str(e),
            )

    # -----------------------------------------------------------------------
    # Internal: Mermaid Generation
    # -----------------------------------------------------------------------

    def _generate_execution_mermaid(
        self,
        plan: GraphPlan,
        result: WorkflowResult,
    ) -> str:
        """Generate a mermaid diagram showing execution status.

        Color-codes nodes based on execution status:
        - Green: completed
        - Red: failed
        - Blue: in progress
        """
        # Update step statuses in the plan for to_mermaid()
        status_map: dict[int, str] = {}
        for sr in result.step_results:
            status_map[sr.step_index] = sr.status

        for i, step in enumerate(plan.steps):
            step.status = status_map.get(i, "pending")

        return plan.to_mermaid(title=f"Workflow: {result.workflow_name}")

    # -----------------------------------------------------------------------
    # Internal: KG Provenance
    # -----------------------------------------------------------------------

    def _record_execution(
        self,
        engine: IntelligenceGraphEngine,
        result: WorkflowResult,
    ) -> None:
        """Record workflow execution as a RunTrace in the KG."""
        import time as time_mod

        try:
            ts = time_mod.strftime("%Y-%m-%dT%H:%M:%SZ", time_mod.gmtime())
            trace_id = f"wf_trace:{result.session_id}"

            engine.add_node(
                trace_id,
                "RunTrace",
                properties={
                    "name": result.workflow_name,
                    "status": result.status,
                    "duration_ms": round(result.total_duration_ms, 1),
                    "agent_name": "workflow_runner",
                    "task": f"Workflow execution: {result.workflow_name}",
                    "timestamp": ts,
                    "metadata": str(result.to_dict())[:2000],
                },
            )
        except Exception as e:
            logger.debug("Failed to record workflow trace: %s", e)
