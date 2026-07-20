"""Closed structured-response contract for GraphOS delegation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, get_args, get_origin, get_type_hints
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_utilities.graph.state import GraphDeps, GraphState
from agent_utilities.graph.verification import synthesizer_step
from agent_utilities.orchestration.response_format import validate_response_format


class _FakeMCP:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, *, name: str, **_metadata: Any):
        def capture(function: Any) -> Any:
            self.tools[name] = function
            return function

        return capture


@pytest.mark.asyncio
async def test_graph_orchestrate_propagates_closed_json_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.mcp import kg_server
    from agent_utilities.mcp.tools.analysis_tools import register_analysis_tools
    from agent_utilities.orchestration import manager

    captured: dict[str, Any] = {}

    class RecordingOrchestrator:
        def __init__(self, _engine: Any) -> None:
            pass

        async def execute_agent(self, **kwargs: Any) -> str:
            captured.update(kwargs)
            return json.dumps(
                {"output": '{"answer":"ok"}', "run_id": "run:opaque", "mermaid": None}
            )

    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    monkeypatch.setattr(manager, "Orchestrator", RecordingOrchestrator)
    mcp = _FakeMCP()
    register_analysis_tools(mcp)

    graph_orchestrate = mcp.tools["graph_orchestrate"]
    annotation = get_type_hints(graph_orchestrate)["response_format"]
    assert set(get_args(annotation)) == {"text", "json"}

    result = json.loads(
        await graph_orchestrate(
            task="Return a structured assessment.",
            agent_name="analysis-skill",
            max_steps=30,
            context="",
            budget_tokens=0,
            context_ref="",
            allowed_tools="",
            cred_ref="",
            open_channel=False,
            reasoning_effort="",
            model_class="standard",
            response_format="json",
        )
    )

    assert captured["response_format"] == "json"
    assert result["output"] == '{"answer":"ok"}'
    assert result["run_id"] == "run:opaque"

    with pytest.raises(ValueError, match="text, json"):
        await graph_orchestrate(
            task="Return a structured assessment.",
            agent_name="analysis-skill",
            max_steps=30,
            context="",
            budget_tokens=0,
            context_ref="",
            allowed_tools="",
            cred_ref="",
            open_channel=False,
            reasoning_effort="",
            model_class="standard",
            response_format="markdown",
        )


@pytest.mark.asyncio
async def test_orchestrator_and_runner_propagate_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.orchestration import agent_runner
    from agent_utilities.orchestration import manager as manager_module

    manager_call: dict[str, Any] = {}

    async def fake_run_agent(**kwargs: Any) -> str:
        manager_call.update(kwargs)
        return "ok"

    monkeypatch.setattr(manager_module, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        manager_module.Orchestrator, "_scan_task", lambda _self, _task: None
    )
    await manager_module.Orchestrator(object()).execute_agent(
        agent_name="analysis-skill",
        task="Assess the evidence.",
        response_format="json",
    )
    assert manager_call["response_format"] == "json"

    execution_config: dict[str, Any] = {}
    monkeypatch.setattr(
        agent_runner, "_resolve_agent_from_kg", lambda _engine, _name: {"type": "stub"}
    )
    monkeypatch.setattr(
        agent_runner,
        "_build_execution_config",
        lambda *_args, **_kwargs: {
            "agent_model": "synthetic-model",
            "selected_model_class": "standard",
        },
    )
    monkeypatch.setattr(agent_runner, "_record_execution_trace", lambda *a, **k: None)

    async def fake_execute_graph(*, config: dict[str, Any], **_kwargs: Any):
        execution_config.update(config)
        return {"results": {"output": '{"answer":"ok"}'}}

    monkeypatch.setattr(agent_runner, "_execute_graph", fake_execute_graph)
    await agent_runner.run_agent(
        agent_name="analysis-skill",
        task="Assess the evidence in detail.",
        engine=object(),
        response_format="json",
    )
    assert execution_config["response_format"] == "json"


@pytest.mark.asyncio
async def test_json_contract_bypasses_text_only_direct_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.graph import builder
    from agent_utilities.orchestration import agent_runner
    from agent_utilities.orchestration import engine as engine_module

    direct_completion = AsyncMock(
        side_effect=AssertionError("text-only direct completion was called")
    )
    monkeypatch.setattr(agent_runner, "_run_direct_completion", direct_completion)
    monkeypatch.setattr(
        builder,
        "create_graph_agent",
        lambda **_kwargs: (object(), {"tag_prompts": {}}),
    )
    graph_execution = AsyncMock(return_value={"results": {"output": '{"answer":"ok"}'}})
    monkeypatch.setattr(
        engine_module.AgentOrchestrationEngine, "execute_graph", graph_execution
    )

    result = await agent_runner._execute_graph(
        config={
            "tag_prompts": {},
            "execution_shape": SimpleNamespace(direct_complete=True),
            "response_format": "json",
        },
        query="Hello",
        run_id="run:" + "a" * 32,
        max_steps=3,
        agent_meta={"type": "unknown"},
    )

    assert result["results"]["output"] == '{"answer":"ok"}'
    direct_completion.assert_not_awaited()
    assert graph_execution.await_args.kwargs["config"]["response_format"] == "json"


def test_invalid_response_format_is_rejected_without_fallback() -> None:
    with pytest.raises(ValueError, match="text, json"):
        validate_response_format("markdown")
    with pytest.raises(ValueError, match="text, json"):
        GraphDeps(
            tag_prompts={},
            tag_env_vars={},
            mcp_toolsets=[],
            router_model=None,
            agent_model=None,
            response_format="markdown",  # type: ignore[arg-type]
        )


def _structured_context() -> MagicMock:
    ctx = MagicMock()
    ctx.state = GraphState(query="Assess the evidence.")
    ctx.state.results_registry = {"researcher": {"finding": "supported"}}
    ctx.deps = MagicMock()
    ctx.deps.agent_model = object()
    ctx.deps.event_queue = None
    ctx.deps.knowledge_engine = None
    ctx.deps.response_format = "json"
    ctx.deps.verifier_timeout = 5.0
    return ctx


class _NoopDistillationHook:
    def __init__(self, **_kwargs: Any) -> None:
        pass

    async def on_execution_complete(self, **_kwargs: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_json_synthesizer_emits_one_raw_compact_object() -> None:
    ctx = _structured_context()
    structured = {"answer": "supported", "confidence": 0.9}
    agent = MagicMock()
    agent.run = AsyncMock(return_value=SimpleNamespace(output=structured))

    with (
        patch(
            "agent_utilities.graph.verification.create_context_agent",
            return_value=agent,
        ) as create_agent,
        patch(
            "agent_utilities.graph.verification.load_specialized_prompts",
            return_value="Verify the evidence.",
        ),
        patch(
            "agent_utilities.workflows.distillation_hook.WorkflowDistillationHook",
            _NoopDistillationHook,
        ),
    ):
        result = await synthesizer_step(ctx)

    output = result.data.results["output"]
    assert output == '{"answer":"supported","confidence":0.9}'
    assert json.loads(output) == structured
    assert "```" not in output
    kwargs = create_agent.call_args.kwargs
    assert get_origin(kwargs["output_type"]) is dict
    assert get_args(kwargs["output_type"]) == (str, Any)


@pytest.mark.asyncio
async def test_json_synthesizer_does_not_repair_markdown_fences() -> None:
    ctx = _structured_context()
    agent = MagicMock()
    agent.run = AsyncMock(
        return_value=SimpleNamespace(output='```json\n{"answer":"unsupported"}\n```')
    )

    with (
        patch(
            "agent_utilities.graph.verification.create_context_agent",
            return_value=agent,
        ),
        patch(
            "agent_utilities.graph.verification.load_specialized_prompts",
            return_value="Verify the evidence.",
        ),
        patch(
            "agent_utilities.workflows.distillation_hook.WorkflowDistillationHook",
            _NoopDistillationHook,
        ),
    ):
        result = await synthesizer_step(ctx)

    assert result.data.results["output"] == (
        '{"error":"structured_synthesis_failed","results_available":true,'
        '"status":"degraded"}'
    )
    assert result.data.metadata["degraded"] is True
    assert result.data.metadata["outcome"] == "structured_synthesis_failed"
