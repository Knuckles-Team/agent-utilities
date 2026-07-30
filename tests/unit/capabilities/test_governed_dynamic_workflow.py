"""Live-path tests for the governed DynamicWorkflow adapter."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from agent_utilities.capabilities.governed_dynamic_workflow import (
    DelegationStep,
    GovernedDynamicWorkflow,
    WorkflowResourceLimits,
)


class RecordingParallelEngine:
    def __init__(self) -> None:
        self.manifest = None
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False

    async def execute(self, manifest, *, graph_deps=None):
        self.manifest = manifest
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return {"graph_deps": graph_deps, "source": manifest.source}


def test_compiles_model_menus_budgets_and_trace_into_canonical_manifest() -> None:
    workflow = GovernedDynamicWorkflow(
        name="review",
        query="review this change",
        max_agent_calls=2,
        resource_limits=WorkflowResourceLimits(max_duration_secs=10, max_concurrency=2),
        trace_context={"traceparent": "00-test"},
        steps=[
            DelegationStep(
                id="draft",
                description="draft findings",
                model_menu=["openai:gpt-5.6-luna", "openai:gpt-5.6-terra"],
                model_id="openai:gpt-5.6-luna",
                timeout_secs=60,
            ),
            DelegationStep(
                id="judge", description="judge findings", depends_on=["draft"]
            ),
        ],
    )

    manifest = workflow.to_manifest()
    assert manifest.source == "governed_dynamic_workflow"
    assert manifest.max_concurrency == 2
    assert manifest.metadata["trace_context"] == {"traceparent": "00-test"}
    assert manifest.agents[0].model_id == "openai:gpt-5.6-luna"
    assert manifest.agents[0].delegation_model_menu == [
        "openai:gpt-5.6-luna",
        "openai:gpt-5.6-terra",
    ]
    assert manifest.agents[0].timeout == 10
    assert manifest.agents[1].depends_on == ["draft"]


def test_rejects_budget_dag_and_model_menu_escapes() -> None:
    with pytest.raises(ValidationError, match="model_id must be one of model_menu"):
        DelegationStep(id="x", description="x", model_menu=["a"], model_id="b")
    with pytest.raises(ValidationError, match="steps exceed max_agent_calls"):
        GovernedDynamicWorkflow(
            max_agent_calls=1,
            steps=[
                DelegationStep(id="a", description="a"),
                DelegationStep(id="b", description="b"),
            ],
        )
    with pytest.raises(ValidationError, match="unknown dependencies"):
        GovernedDynamicWorkflow(
            steps=[DelegationStep(id="a", description="a", depends_on=["missing"])]
        )


async def test_execute_uses_parallel_engine_and_propagates_cancellation() -> None:
    workflow = GovernedDynamicWorkflow(steps=[DelegationStep(id="a", description="a")])
    engine = RecordingParallelEngine()
    cancellation = asyncio.Event()
    task = asyncio.create_task(
        workflow.execute(engine, graph_deps="deps", cancellation=cancellation)
    )
    await engine.started.wait()
    cancellation.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.cancelled is True
    assert engine.manifest.source == "governed_dynamic_workflow"
