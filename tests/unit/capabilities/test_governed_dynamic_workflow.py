"""Live-path tests for the governed upstream DynamicWorkflow boundary."""

from __future__ import annotations

import asyncio
import contextvars
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


class RecordingKGEngine:
    """A minimal engine double that round-trips node writes.

    ``RecordingTraceEngine`` above only records writes for assertion; it
    cannot answer a later read. The resume cache
    (``governed_dynamic_workflow._load_resume_cache``/``_save_resume_cache``)
    needs a real round trip across two independent ``execute()`` calls, so
    this double additionally implements the ``backend_type == "rust"``
    read shape (``has_node``/``__getitem__``) the resume cache checks.
    """

    backend = None
    backend_type = "rust"

    def __init__(self) -> None:
        self.node_store: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(self, node_id, label, properties=None):
        self.node_store[node_id] = dict(properties or {})

    def link_nodes(self, source, target, relationship, properties=None):
        self.edges.append((source, target, relationship))

    def has_node(self, node_id):
        return node_id in self.node_store

    def __getitem__(self, node_id):
        return self.node_store[node_id]


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


def _dual_reviewer_workflow_model() -> FunctionModel:
    """A conductor that fans a task across ``reviewer`` twice, then ``summarizer``.

    Deterministically re-issues the SAME script text on every attempt, so a
    resumed attempt exercises the resume cache under the exact scenario the
    model would produce after a genuine budget-halted retry: it does not know
    which calls already completed and simply re-asks for all of them.
    """

    async def respond(messages, _info):
        if any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart("workflow attempt complete")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "run_workflow",
                    {
                        "code": (
                            'r1 = await reviewer(task="pass one")\n'
                            'r2 = await reviewer(task="pass two")\n'
                            'await summarizer(task="combine")'
                        )
                    },
                )
            ]
        )

    return FunctionModel(respond)


def _resume_test_workflow(*, max_agent_calls: int) -> GovernedDynamicWorkflow:
    return GovernedDynamicWorkflow(
        name="resume-review",
        query="review and summarize",
        max_agent_calls=max_agent_calls,
        resource_limits=WorkflowResourceLimits(max_concurrency=1),
        steps=[
            DelegationStep(id="reviewer", description="review"),
            DelegationStep(
                id="summarizer", description="summarize", depends_on=["reviewer"]
            ),
        ],
    )


async def test_budget_halted_then_restarted_workflow_persists_no_duplicate_toolcalls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The acceptance test: budget halt then restart produces NO duplicate ToolCalls.

    Attempt 1 runs with ``max_agent_calls=2``: both `reviewer` calls succeed
    (real GraphOS dispatches -- the equivalent of a real ``:ToolCall`` each),
    then `summarizer` hits the Harness's own host-enforced budget ceiling and
    the run halts (a truthful, non-exceptional completion carrying the
    budget-exhausted terminal result). Attempt 2 reuses the SAME
    ``workflow_run_id`` against the SAME durable engine but a FRESH in-memory
    ``GovernedDynamicWorkflow``/runtime (simulating a process restart) and a
    larger budget. The model re-issues the identical script, unaware of what
    already completed -- proving the resume cache, not model good behavior,
    is what prevents duplicate work.
    """

    from agent_utilities.core import contextual_model

    monkeypatch.setattr(contextual_model, "_context_compiler_enabled", lambda: False)

    engine = RecordingKGEngine()
    orchestrator = RecordingOrchestrator(engine)
    workflow_run_id = "wf:resume-acceptance-test"

    attempt_one = _resume_test_workflow(max_agent_calls=2)
    result_one = await attempt_one.execute(
        orchestrator,
        orchestrator_model=_dual_reviewer_workflow_model(),
        workflow_run_id=workflow_run_id,
    )

    # Attempt 1: both reviewer calls actually dispatched through GraphOS;
    # summarizer never did (blocked by the harness's own per-run ceiling).
    calls_after_attempt_one = list(orchestrator.calls)
    assert [c["task"] for c in calls_after_attempt_one] == ["pass one", "pass two"]
    assert result_one.resumed is False

    attempt_two = _resume_test_workflow(max_agent_calls=3)
    result_two = await attempt_two.execute(
        orchestrator,
        orchestrator_model=_dual_reviewer_workflow_model(),
        workflow_run_id=workflow_run_id,
    )

    # Attempt 2 re-issued the SAME two reviewer calls plus summarizer. Only
    # summarizer is a NEW GraphOS dispatch -- the reviewer calls are replayed
    # from the persisted resume cache, so the total ToolCall-equivalent count
    # across BOTH attempts is exactly 3, never 5.
    all_calls = orchestrator.calls
    assert len(all_calls) == 3
    assert [c["task"] for c in all_calls] == ["pass one", "pass two", "combine"]

    # Truthfulness: attempt 2 is explicitly marked resumed, never reported as
    # an indistinguishable clean success.
    assert result_two.resumed is True
    assert set(result_two.replayed_step_ids) == {"reviewer"}
    replayed = [c for c in result_two.child_runs if c.outcome == "replayed"]
    assert len(replayed) == 2
    assert {c.step_id for c in replayed} == {"reviewer"}
    fresh = [c for c in result_two.child_runs if c.outcome == "ok"]
    assert [c.step_id for c in fresh] == ["summarizer"]

    # The generated script and the normalised call graph are stored as trace
    # artifacts (not merely returned to the caller).
    script_nodes = [
        (node_id, props)
        for node_id, props in engine.node_store.items()
        if node_id.startswith("workflow-script:")
    ]
    assert script_nodes
    assert all(node[1]["code"] for node in script_nodes)

    run_trace = next(
        props
        for node_id, props in engine.node_store.items()
        if node_id.startswith("trace:")
    )
    assert run_trace["graph_resume_supported"] is True
    assert run_trace["graph_topology_digest"]
    assert run_trace["graph_version_digest"]
    assert json.loads(run_trace["graph_transition_sequence"])

    # Restore-and-continue: a real conductor checkpoint exists for each
    # attempt (CheckpointMiddleware is now default-on for this path).
    assert result_one.checkpoint_ids
    assert result_two.checkpoint_ids


async def test_sub_agent_dispatch_inherits_ambient_context_across_the_monty_sandbox_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tenant/session contextvar set before ``execute()`` reaches every real
    GraphOS dispatch inside the Monty sandbox, unchanged -- the Monty
    boundary does not strip ambient context (CONCEPT:AU-ORCH.execution.dynamic-workflows).
    """

    from agent_utilities.core import contextual_model

    monkeypatch.setattr(contextual_model, "_context_compiler_enabled", lambda: False)

    tenant_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("test_tenant_ctx")

    class TenantRecordingOrchestrator(RecordingOrchestrator):
        async def execute_agent(self, **kwargs):
            kwargs["observed_tenant"] = tenant_ctx.get(None)
            return await super().execute_agent(**kwargs)

    orchestrator = TenantRecordingOrchestrator()
    workflow = GovernedDynamicWorkflow(
        steps=[
            DelegationStep(id="reviewer", description="review"),
            DelegationStep(
                id="summarizer", description="summarize", depends_on=["reviewer"]
            ),
        ]
    )

    token = tenant_ctx.set("tenant-acme")
    try:
        result = await workflow.execute(
            orchestrator, orchestrator_model=_workflow_model()
        )
    finally:
        tenant_ctx.reset(token)

    assert result.output == "workflow complete"
    assert len(orchestrator.calls) == 2
    assert all(call["observed_tenant"] == "tenant-acme" for call in orchestrator.calls)
    # Ambient context is caller-scoped, not leaked into a later unrelated call.
    assert tenant_ctx.get(None) is None


async def test_nested_dynamic_workflow_is_rejected_across_the_graphos_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sub-agent's own attempt to run a nested DynamicWorkflow is refused.

    Upstream enforces "workflows do not nest" through an asyncio contextvar
    set for the duration of the outer sandbox's ``call_tool``; GraphOS's
    sub-agent dispatch is a plain awaited call within that SAME context (no
    new task boundary), so the inner attempt inherits the flag with no
    GraphOS-side code required. This proves that inheritance holds across a
    REAL two-level GraphOS delegation, not just inside upstream's own tests.
    """

    from agent_utilities.core import contextual_model

    monkeypatch.setattr(contextual_model, "_context_compiler_enabled", lambda: False)

    inner_workflow = GovernedDynamicWorkflow(
        steps=[DelegationStep(id="leaf", description="leaf step")]
    )

    def _echoing_inner_model() -> FunctionModel:
        """Echo the ``run_workflow`` tool's raw return content into the final
        text, so the test can see the nesting-rejection payload rather than a
        content-blind "done" the way real deployed models would report it."""

        async def respond(messages, _info):
            for message in messages:
                for part in message.parts:
                    if isinstance(part, ToolReturnPart):
                        return ModelResponse(parts=[TextPart(str(part.content))])
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "run_workflow", {"code": 'await leaf(task="trigger nested")'}
                    )
                ]
            )

        return FunctionModel(respond)

    class NestingOrchestrator(RecordingOrchestrator):
        async def execute_agent(self, **kwargs):
            if kwargs["agent_name"] == "reviewer":
                nested_orchestrator = RecordingOrchestrator()
                nested_result = await inner_workflow.execute(
                    nested_orchestrator,
                    orchestrator_model=_echoing_inner_model(),
                )
                # The nested call must never re-enter GraphOS: no real
                # dispatch happened for the inner workflow's OWN catalog.
                assert nested_orchestrator.calls == []
                return json.dumps(
                    {
                        "output": str(nested_result.output),
                        "run_summary": {"outcome": "ok"},
                    }
                )
            return await super().execute_agent(**kwargs)

    outer_orchestrator = NestingOrchestrator()
    outer_workflow = GovernedDynamicWorkflow(
        steps=[DelegationStep(id="reviewer", description="review")]
    )

    def _single_call_model() -> FunctionModel:
        async def respond(messages, _info):
            for message in messages:
                for part in message.parts:
                    if isinstance(part, ToolReturnPart):
                        return ModelResponse(parts=[TextPart(str(part.content))])
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "run_workflow",
                        {"code": 'await reviewer(task="trigger nested")'},
                    )
                ]
            )

        return FunctionModel(respond)

    result = await outer_workflow.execute(
        outer_orchestrator, orchestrator_model=_single_call_model()
    )

    # The nested attempt's rejection message round-trips all the way back
    # through GraphOS's real dispatch into the outer script's own result --
    # proving the inheritance holds across a REAL two-level GraphOS
    # delegation, not merely inside upstream's own unit tests.
    assert "do not nest" in str(result.output)
    assert result.child_runs[0].outcome == "ok"


async def test_upstream_dispatch_is_cancelled_through_the_harness_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shared ``cancellation`` event reaches an in-flight catalog call even
    through the real Harness/Monty execution path, not just the static
    ``ParallelEngine`` fallback (already covered by
    ``test_execute_static_propagates_cancellation``).
    """

    from agent_utilities.core import contextual_model

    monkeypatch.setattr(contextual_model, "_context_compiler_enabled", lambda: False)

    started = asyncio.Event()
    release = asyncio.Event()
    cancelled_inside = asyncio.Event()

    class SlowOrchestrator(RecordingOrchestrator):
        async def execute_agent(self, **kwargs):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled_inside.set()
                raise
            return await super().execute_agent(**kwargs)  # pragma: no cover

    def _single_reviewer_model() -> FunctionModel:
        async def respond(messages, _info):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "run_workflow", {"code": 'await reviewer(task="slow")'}
                    )
                ]
            )

        return FunctionModel(respond)

    orchestrator = SlowOrchestrator()
    workflow = GovernedDynamicWorkflow(
        steps=[DelegationStep(id="reviewer", description="review")]
    )
    cancellation = asyncio.Event()

    task = asyncio.create_task(
        workflow.execute(
            orchestrator,
            orchestrator_model=_single_reviewer_model(),
            cancellation=cancellation,
        )
    )
    await started.wait()
    cancellation.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled_inside.is_set()
