"""Live-path tests for the governed upstream DynamicWorkflow boundary."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow

from agent_utilities.capabilities.governed_dynamic_workflow import (
    DelegationStep,
    DynamicWorkflowUnavailableError,
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


class RecordingOrchestrator:
    def __init__(self, engine=None) -> None:
        self.engine = engine
        self.calls: list[dict] = []

    async def execute_agent(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(
            {
                "output": f"{kwargs['agent_name']}:{kwargs['task']}",
                "run_summary": {"outcome": "ok"},
            }
        )


class RecordingTraceEngine:
    backend = None

    def __init__(self) -> None:
        self.nodes: list[tuple[str, str, dict]] = []
        self.edges: list[tuple[str, str, str]] = []

    def add_node(self, node_id, label, properties=None):
        self.nodes.append((node_id, label, dict(properties or {})))

    def link_nodes(self, source, target, relationship, properties=None):
        self.edges.append((source, target, relationship))


def _workflow_model() -> FunctionModel:
    async def respond(messages, _info):
        if any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart("workflow complete")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "run_workflow",
                    {
                        "code": (
                            'review = await reviewer(task="review change")\n'
                            'await summarizer(task="summarize " + review)'
                        )
                    },
                )
            ]
        )

    return FunctionModel(respond)


def test_compiles_model_menus_budgets_and_trace_into_static_fallback() -> None:
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
                allowed_tools=["read_code"],
                model_menu=["openai:model-a", "openai:model-b"],
                model_id="openai:model-a",
                timeout_secs=60,
            ),
            DelegationStep(
                id="judge", description="judge findings", depends_on=["draft"]
            ),
        ],
    )

    manifest = workflow.to_manifest()
    assert manifest.source == "governed_dynamic_workflow_static_fallback"
    assert manifest.max_concurrency == 2
    assert manifest.metadata["trace_context"] == {"traceparent": "00-test"}
    assert manifest.agents[0].model_id == "openai:model-a"
    assert manifest.agents[0].delegation_model_menu == [
        "openai:model-a",
        "openai:model-b",
    ]
    assert manifest.agents[0].tools == ["read_code"]
    assert manifest.agents[0].timeout == 10
    assert manifest.agents[1].depends_on == ["draft"]


def test_rejects_budget_dag_model_and_tool_contract_escapes() -> None:
    with pytest.raises(ValidationError, match="model_id must be one of model_menu"):
        DelegationStep(id="x", description="x", model_menu=["a"], model_id="b")
    with pytest.raises(ValidationError, match="required_tools must be allowed"):
        DelegationStep(
            id="x",
            description="x",
            allowed_tools=["read"],
            required_tools=["write"],
        )
    with pytest.raises(ValidationError, match="tool_server requires kind='skill'"):
        DelegationStep(id="x", description="x", tool_server="server")
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
    with pytest.raises(ValidationError, match="must form a DAG"):
        GovernedDynamicWorkflow(
            steps=[
                DelegationStep(id="a", description="a", depends_on=["b"]),
                DelegationStep(id="b", description="b", depends_on=["a"]),
            ]
        )


def test_builds_actual_upstream_harness_capability() -> None:
    workflow = GovernedDynamicWorkflow(
        steps=[DelegationStep(id="reviewer", description="review")]
    )

    capability = workflow.build_upstream_capability(RecordingOrchestrator())

    assert isinstance(capability, DynamicWorkflow)
    assert capability.max_agent_calls == 50
    assert capability.forward_usage is False
    assert capability.resource_limits["max_memory"] == 256 * 1024 * 1024


async def test_actual_harness_script_reenters_graphos_with_shared_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.core import contextual_model

    monkeypatch.setattr(contextual_model, "_context_compiler_enabled", lambda: False)
    workflow = GovernedDynamicWorkflow(
        name="review",
        query="review the change",
        max_agent_calls=2,
        resource_limits=WorkflowResourceLimits(
            max_concurrency=1,
            max_tokens_per_agent=4000,
            orchestrator_token_budget=8000,
        ),
        steps=[
            DelegationStep(
                id="reviewer",
                description="review",
                target_name="code-review-agent",
                allowed_tools=["read_code"],
                required_tools=["read_code"],
                model_class="economy",
            ),
            DelegationStep(
                id="summarizer",
                description="summarize",
                depends_on=["reviewer"],
                kind="skill",
                target_name="summarize-skill",
                tool_server="summary-api",
            ),
        ],
    )
    trace_engine = RecordingTraceEngine()
    orchestrator = RecordingOrchestrator(trace_engine)

    result = await workflow.execute(
        orchestrator,
        orchestrator_model=_workflow_model(),
    )

    assert result.output == "workflow complete"
    assert result.backend == "pydantic-ai-harness.dynamic_workflow.DynamicWorkflow"
    assert result.upstream_version == "0.14.0"
    assert len(result.script_evidence) == 1
    assert result.script_evidence[0].byte_count > 0
    assert len(result.script_evidence[0].sha256) == 64
    assert [call["agent_name"] for call in orchestrator.calls] == [
        "code-review-agent",
        "summarize-skill",
    ]
    assert {call["session_id"] for call in orchestrator.calls} == {
        result.workflow_run_id
    }
    assert all(call["include_run_summary"] for call in orchestrator.calls)
    assert orchestrator.calls[0]["allowed_tools"] == ["read_code"]
    assert orchestrator.calls[0]["required_tools"] == ["read_code"]
    assert orchestrator.calls[0]["model_class"] == "economy"
    assert orchestrator.calls[0]["budget_tokens"] == 4000
    assert orchestrator.calls[1]["skill_name"] == "summarize-skill"
    assert orchestrator.calls[1]["tool_server"] == "summary-api"
    assert [child.outcome for child in result.child_runs] == ["ok", "ok"]
    assert [child.agent_name for child in result.child_runs] == [
        "code-review-agent",
        "summarize-skill",
    ]
    assert any(
        label == "RunTrace" and node_id == result.trace_ref
        for node_id, label, _properties in trace_engine.nodes
    )
    assert all(
        (child.trace_ref, result.trace_ref, "PARENT_RUN") in trace_engine.edges
        for child in result.child_runs
    )


def test_from_graph_plan_hydrates_reviewed_agent_skill_and_tool_contracts() -> None:
    from agent_utilities.models.graph import GraphPlan
    from agent_utilities.models.sdd import Task

    workflow = GovernedDynamicWorkflow.from_graph_plan(
        GraphPlan(
            steps=[
                Task(
                    id="reviewer",
                    description="review",
                    assigned_to="code-review-agent",
                    metadata={
                        "allowed_tools": ["read_code"],
                        "required_tools": ["read_code"],
                        "reasoning_effort": "medium",
                    },
                ),
                Task(
                    id="summarizer",
                    description="summarize",
                    depends_on=["reviewer"],
                    metadata={
                        "skill_name": "summarize-skill",
                        "tool_server": "summary-api",
                    },
                ),
            ]
        ),
        name="review",
        query="review",
    )

    assert workflow.steps[0].kind == "agent"
    assert workflow.steps[0].target_name == "code-review-agent"
    assert workflow.steps[0].allowed_tools == ["read_code"]
    assert workflow.steps[0].required_tools == ["read_code"]
    assert workflow.steps[0].reasoning_effort == "medium"
    assert workflow.steps[1].kind == "skill"
    assert workflow.steps[1].target_name == "summarize-skill"
    assert workflow.steps[1].tool_server == "summary-api"


async def test_shared_cancellation_cleans_up_the_in_flight_dispatch() -> None:
    from agent_utilities.capabilities.governed_dynamic_workflow import (
        _await_with_cancellation,
    )

    started = asyncio.Event()
    cleaned_up = asyncio.Event()
    cancellation = asyncio.Event()

    async def dispatch() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up.set()

    execution = asyncio.create_task(_await_with_cancellation(dispatch(), cancellation))
    await started.wait()
    cancellation.set()

    with pytest.raises(asyncio.CancelledError):
        await execution
    assert cleaned_up.is_set()


async def test_execute_static_propagates_cancellation() -> None:
    workflow = GovernedDynamicWorkflow(steps=[DelegationStep(id="a", description="a")])
    engine = RecordingParallelEngine()
    cancellation = asyncio.Event()
    task = asyncio.create_task(
        workflow.execute_static(engine, graph_deps="deps", cancellation=cancellation)
    )
    await engine.started.wait()
    cancellation.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.cancelled is True
    assert engine.manifest is not None
    assert engine.manifest.source == "governed_dynamic_workflow_static_fallback"


def test_unavailable_upstream_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_utilities.capabilities import governed_dynamic_workflow as module

    monkeypatch.setattr(
        module,
        "_load_upstream_dynamic_workflow",
        lambda: (_ for _ in ()).throw(
            DynamicWorkflowUnavailableError("Harness missing")
        ),
    )
    workflow = GovernedDynamicWorkflow(
        steps=[DelegationStep(id="reviewer", description="review")]
    )

    with pytest.raises(DynamicWorkflowUnavailableError, match="Harness missing"):
        workflow.build_upstream_capability(RecordingOrchestrator())


def test_exact_model_override_requires_stored_dag() -> None:
    workflow = GovernedDynamicWorkflow(
        steps=[
            DelegationStep(
                id="reviewer",
                description="review",
                model_id="openai:fixed-model",
            )
        ]
    )

    with pytest.raises(
        DynamicWorkflowUnavailableError,
        match="model_id selection is not supported",
    ):
        workflow.build_upstream_capability(RecordingOrchestrator())


async def test_manager_fallback_is_only_for_upstream_unavailability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.capabilities import governed_dynamic_workflow as module
    from agent_utilities.knowledge_graph import workflow_store
    from agent_utilities.models.graph import GraphPlan
    from agent_utilities.models.sdd import Task
    from agent_utilities.orchestration.manager import Orchestrator

    engine: Any = SimpleNamespace()
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.engine = engine
    monkeypatch.setattr(
        workflow_store.WorkflowStore,
        "load_workflow",
        lambda _self, _name: GraphPlan(
            steps=[Task(id="reviewer", description="review")]
        ),
    )
    monkeypatch.setattr(
        module,
        "_load_upstream_dynamic_workflow",
        lambda: (_ for _ in ()).throw(
            DynamicWorkflowUnavailableError("Harness missing")
        ),
    )
    fallback = AsyncMock(return_value={"run_id": "run:fallback"})
    monkeypatch.setattr(orchestrator, "execute_workflow", fallback)

    result = await orchestrator.execute_dynamic_workflow(
        "review",
        unavailable_fallback="stored_dag",
        orchestrator_model=_workflow_model(),
    )

    assert result["backend"] == "stored_dag"
    assert result["fallback_used"] is True
    fallback.assert_awaited_once()
