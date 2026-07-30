"""Governed dynamic-workflow adapter for the canonical GraphOS execution path.

CONCEPT:AU-ORCH.execution.dynamic-workflows

``pydantic-ai-harness`` offers a sandboxed, model-authored workflow tool.  GraphOS
does not let that tool become a second executor: this adapter accepts a reviewed,
bounded dynamic declaration, compiles it to ``GraphPlan``, and delegates exactly
once to ``ParallelEngine``.  The optional harness extra supplies interoperability
for deployments that need it; importing this module never requires that extra.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field, model_validator

from agent_utilities.models.execution_manifest import ExecutionManifest, ExecutionResult
from agent_utilities.models.graph import GraphPlan
from agent_utilities.models.sdd import Task


class WorkflowResourceLimits(BaseModel):
    """Resource limits carried into canonical per-step timeouts and provenance."""

    max_duration_secs: float = Field(default=300.0, gt=0, le=3600.0)
    max_concurrency: int = Field(default=8, ge=1, le=64)


class DelegationStep(BaseModel):
    """A reviewed dynamic step; its model selection cannot escape its menu."""

    id: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    model_menu: list[str] = Field(default_factory=list)
    model_id: str | None = None
    timeout_secs: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_model_selection(self) -> DelegationStep:
        if self.model_id and self.model_menu and self.model_id not in self.model_menu:
            raise ValueError("model_id must be one of model_menu")
        return self


class GovernedDynamicWorkflow(BaseModel):
    """Compile a bounded dynamic declaration to GraphPlan and execute it once.

    The adapter deliberately has no script evaluator and no sub-agent invocation
    method.  That keeps policy, trace propagation, budgets, cancellation, and KG
    persistence in the existing GraphOS engine.
    """

    name: str = "governed-dynamic-workflow"
    query: str = ""
    steps: list[DelegationStep]
    max_agent_calls: int = Field(default=50, ge=1, le=300)
    resource_limits: WorkflowResourceLimits = Field(
        default_factory=WorkflowResourceLimits
    )
    trace_context: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_budget_and_dag(self) -> GovernedDynamicWorkflow:
        if len(self.steps) > self.max_agent_calls:
            raise ValueError("steps exceed max_agent_calls")
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("dynamic workflow step ids must be unique")
        unknown = {dep for step in self.steps for dep in step.depends_on} - set(ids)
        if unknown:
            raise ValueError(
                f"dynamic workflow has unknown dependencies: {sorted(unknown)}"
            )
        return self

    def to_graph_plan(self) -> GraphPlan:
        """Compile the declaration to the canonical plan, preserving trace metadata."""
        tasks = []
        for step in self.steps:
            timeout = min(
                step.timeout_secs or self.resource_limits.max_duration_secs,
                self.resource_limits.max_duration_secs,
            )
            tasks.append(
                Task(
                    id=step.id,
                    description=step.description,
                    depends_on=step.depends_on,
                    parallel=not step.depends_on,
                    timeout=timeout,
                    model_id=step.model_id,
                    delegation_model_menu=step.model_menu,
                )
            )
        return GraphPlan(
            steps=tasks,
            metadata={
                "source": "governed_dynamic_workflow",
                "max_agent_calls": self.max_agent_calls,
                "resource_limits": self.resource_limits.model_dump(),
                "trace_context": self.trace_context,
            },
        )

    def to_manifest(self) -> ExecutionManifest:
        """Materialize the one supported execution input for ParallelEngine."""
        return ExecutionManifest.from_graph_plan(
            self.to_graph_plan(), name=self.name, query=self.query
        ).model_copy(
            update={
                "source": "governed_dynamic_workflow",
                "max_concurrency": self.resource_limits.max_concurrency,
            }
        )

    async def execute(
        self,
        parallel_engine: Any,
        *,
        graph_deps: Any | None = None,
        cancellation: asyncio.Event | None = None,
    ) -> ExecutionResult:
        """Execute through the supplied canonical ParallelEngine, never upstream code."""
        if cancellation is not None and cancellation.is_set():
            raise asyncio.CancelledError(
                "governed dynamic workflow cancelled before dispatch"
            )
        execution = asyncio.create_task(
            parallel_engine.execute(self.to_manifest(), graph_deps=graph_deps)
        )
        if cancellation is None:
            return await execution
        cancelled = asyncio.create_task(cancellation.wait())
        done, _pending = await asyncio.wait(
            {execution, cancelled}, return_when=asyncio.FIRST_COMPLETED
        )
        if cancelled in done:
            execution.cancel()
            try:
                await execution
            except asyncio.CancelledError:
                pass
            raise asyncio.CancelledError(
                "governed dynamic workflow cancelled during dispatch"
            )
        cancelled.cancel()
        return await execution
