"""Behaviour of the agent run loop (agent_utilities.graph) over fake host ports."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset

import agent_utilities
from agent_utilities import host_ports
from agent_utilities.graph import (
    AgentRunExecutor,
    GraphEventHandler,
    RunBudget,
    RunPlan,
    RunRequest,
    StopRule,
    create_graph_agent,
    default_run_plan,
    execute_graph,
    execute_graph_iter,
    initialize_graph_from_workspace,
    run,
)


def echo_model(log: list[int] | None = None) -> FunctionModel:
    """Reply with the last user prompt; record the history length per call."""

    def fn(messages, info: AgentInfo) -> ModelResponse:
        if log is not None:
            log.append(len(messages))
        prompt = messages[-1].parts[-1].content
        return ModelResponse(parts=[TextPart(f"echo:{prompt}")])

    return FunctionModel(fn)


def tool_model(tool: str) -> FunctionModel:
    """Call ``tool`` once, then answer with its return value."""

    def fn(messages, info: AgentInfo) -> ModelResponse:
        last = messages[-1].parts[-1]
        if getattr(last, "part_kind", "") == "tool-return":
            return ModelResponse(parts=[TextPart(str(last.content))])
        return ModelResponse(parts=[ToolCallPart(tool, {})])

    return FunctionModel(fn)


def plan_of(model, **kw) -> RunPlan:
    return RunPlan(model=model, steps=kw.pop("steps", ("q",)), **kw)


# spec: AU-BOUNDARY-R043.2
async def test_no_model_fails_closed(events):
    result = await run(RunRequest(task="hi"))
    assert result.status == "failed"
    assert "RunPlanUnavailable" in (result.error or "")
    assert [e["type"] for e in events] == ["error"]


# spec: AU-BOUNDARY-R043.2
@pytest.mark.parametrize(
    ("request_kw", "config", "check"),
    [
        (
            {"agent_name": "a"},
            {"specialists": {"a": "be A"}},
            lambda p: p.instructions == "be A",
        ),
        ({}, {"system_prompt": "sys"}, lambda p: p.instructions == "sys"),
        ({"context": "ctx"}, {}, lambda p: p.steps == ("ctx\n\nt",)),
        ({"max_steps": 2}, {}, lambda p: p.stop.max_steps == 2),
        ({"budget_tokens": 9}, {}, lambda p: p.budget.total_tokens_limit == 9),
        ({"allowed_tools": ("x",)}, {}, lambda p: p.tool_names == ("x",)),
        ({"model_id": "m2"}, {}, lambda p: p.model == "m2"),
    ],
)
def test_default_plan_reads_only_request_and_config(request_kw, config, check):
    plan = default_run_plan(
        RunRequest(task="t", **request_kw), {"agent_model": "m", **config}
    )
    assert check(plan)


# spec: AU-BOUNDARY-R043.1
@pytest.mark.parametrize("is_async", [False, True])
async def test_host_decision_provider_supplies_the_plan(is_async):
    seen: list[RunRequest] = []
    plan = plan_of(TestModel(custom_output_text="from-host"))

    def provide(request):
        seen.append(request)
        if is_async:

            async def later():
                return plan

            return later()
        return plan

    host_ports.register_run_decision_provider(provide)
    result = await run(RunRequest(task="t", agent_name="n"))
    assert (result.status, result.output) == ("completed", "from-host")
    assert seen[0].agent_name == "n"


# spec: AU-BOUNDARY-R043.2
async def test_provider_returning_non_plan_fails():
    host_ports.register_run_decision_provider(lambda request: {"model": "x"})
    result = await run(RunRequest(task="t"))
    assert result.status == "failed" and "RunPlan" in (result.error or "")


# spec: AU-BOUNDARY-R043.1
async def test_steps_share_one_history_and_emit_events(events):
    log: list[int] = []
    result = await run(
        RunRequest(task="t"), plan=plan_of(echo_model(log), steps=("a", "b", "c"))
    )
    assert result.output == "echo:c" and result.steps == 3
    assert log == sorted(log) and log[0] < log[-1]
    assert [e["type"] for e in events] == [
        "run_started",
        "node_transition",
        "node_transition",
        "node_transition",
        "graph_complete",
    ]
    assert result.usage["requests"] == 3


# spec: AU-BOUNDARY-R043.1
@pytest.mark.parametrize(
    ("stop", "budget", "status", "steps"),
    [
        (StopRule(stop_markers=("echo:b",)), RunBudget(), "stopped", 2),
        (StopRule(max_steps=1), RunBudget(), "completed", 1),
        (StopRule(), RunBudget(request_limit=0), "stopped", 1),
        (StopRule(), RunBudget(timeout_seconds=0), "stopped", 1),
    ],
)
async def test_stop_rules_and_budgets_are_enforced(stop, budget, status, steps):
    plan = plan_of(echo_model(), steps=("a", "b", "c"), stop=stop, budget=budget)
    result = await run(RunRequest(task="t"), plan=plan)
    assert (result.status, result.steps) == (status, steps)
    if budget != RunBudget():
        assert (result.error or "").startswith("budget:")


# spec: AU-BOUNDARY-R043.1
async def test_model_ids_resolve_through_the_host():
    asked: list[str] = []

    def models(model_id: str):
        asked.append(model_id)
        return TestModel(custom_output_text="ok")

    host_ports.register_model_provider(models)
    result = await run(RunRequest(task="t"), plan=plan_of("host:model"))
    assert (result.output, asked) == ("ok", ["host:model"])


# spec: AU-BOUNDARY-R043.1
async def test_granted_tools_bind_through_the_native_provider():
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup() -> str:
        return "tool-value"

    bound: list[tuple[list[str], str]] = []

    def build(names: list[str], toolset_id: str) -> FunctionToolset:
        bound.append((names, toolset_id))
        return toolset

    host_ports.register_native_toolset_provider(
        host_ports.NativeToolsetProvider(
            available_tools=lambda: ["lookup"], build_toolset=build
        )
    )
    plan = plan_of(tool_model("lookup"), tool_names=("lookup",))
    result = await run(RunRequest(task="t", run_id="r1"), plan=plan)
    assert result.output == "tool-value"
    assert bound == [(["lookup"], "run-r1")]


# spec: AU-BOUNDARY-R043.2
async def test_granted_tools_without_a_host_fail_closed():
    result = await run(
        RunRequest(task="t"), plan=plan_of(TestModel(), tool_names=("x",))
    )
    assert result.status == "failed"


async def test_json_response_format_returns_json_text():
    from agent_utilities.graph import HarnessConfig

    plan = plan_of(
        TestModel(custom_output_args={"a": 1}),
        harness=HarnessConfig(response_format="json"),
    )
    result = await run(RunRequest(task="t"), plan=plan)
    assert result.output == '{"a": 1}'


async def test_model_error_is_a_failed_terminal_event():
    def boom(messages, info):
        raise ValueError("model down")

    result = await run(RunRequest(task="t"), plan=plan_of(FunctionModel(boom)))
    assert result.status == "failed" and "model down" in (result.error or "")


# spec: AU-BOUNDARY-R043.1
async def test_bundle_runs_through_execute_graph():
    graph, config = create_graph_agent(
        tag_prompts={"ops": "be ops"}, agent_model="m", name="Ops"
    )
    host_ports.register_model_provider(lambda _id: echo_model())
    result = await execute_graph(graph, config, "hello", run_id="r9", agent_name="ops")
    assert (result["run_id"], result["output"]) == ("r9", "echo:hello")
    assert config["routing"] == {"router_model": None, "min_confidence": 0.0}


async def test_execute_graph_iter_calls_the_handler():
    graph, config = create_graph_agent(agent_model="m")
    host_ports.register_model_provider(lambda _id: echo_model())
    seen: list[str] = []

    class Handler(GraphEventHandler):
        async def handle_event(self, event):
            seen.append(event["type"])

    out = [
        e["type"]
        async for e in execute_graph_iter(graph, config, "q", handler=Handler())
    ]
    assert seen == out and out[-1] == "graph_complete"


# spec: AU-BOUNDARY-R043.2
def test_tool_endpoints_fail_closed_without_a_host():
    with pytest.raises(RuntimeError, match="graph-os host"):
        create_graph_agent(mcp_url="http://x")


def test_tool_endpoints_bind_through_the_remote_provider(tmp_path):
    host_ports.register_remote_toolset_provider(
        host_ports.RemoteToolsetProvider(
            build_url_toolset=lambda url, tid, timeout: f"url:{url}",
            load_config_toolsets=lambda path: [f"cfg:{path.name}"],
        )
    )
    graph, config = create_graph_agent(
        mcp_url="http://x", mcp_config=str(tmp_path / "c.json")
    )
    assert graph.toolsets == ("url:http://x", "cfg:c.json")
    assert config["toolsets"] == list(graph.toolsets)


@pytest.mark.parametrize("has_file", [True, False])
def test_workspace_prompt_becomes_instructions(tmp_path, has_file):
    if has_file:
        (tmp_path / "AGENTS.md").write_text("ws prompt")
    _, config = initialize_graph_from_workspace(workspace=tmp_path, agent_model="m")
    assert config["system_prompt"] == ("ws prompt" if has_file else "")


# spec: AU-BOUNDARY-R043.1
async def test_agent_run_executor_maps_a_host_request():
    host_ports.register_model_provider(lambda _id: echo_model())
    executor = AgentRunExecutor({"agent_model": "m"})
    request = SimpleNamespace(
        task="do",
        agent_name="a",
        run_id="r2",
        context=None,
        allowed_tools=None,
        max_steps=5,
        budget_tokens=None,
        response_format="text",
    )
    result = await executor.execute_agent(request, session=object())
    assert (result.run_id, result.output) == ("r2", "echo:do")
    with pytest.raises(RuntimeError):
        await AgentRunExecutor().execute_agent(request)


def test_top_level_exports_are_lazy():
    assert agent_utilities.execute_graph is execute_graph
    with pytest.raises(AttributeError):
        agent_utilities.not_a_name  # noqa: B018
