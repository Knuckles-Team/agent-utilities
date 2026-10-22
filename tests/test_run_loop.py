"""Behaviour of the agent run loop (agent_utilities.graph) over fake host ports."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RequestUsage, RunUsage

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
    iterate_run,
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
    with pytest.raises(RuntimeError, match="arbora host"):
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


# spec: AU-BOUNDARY-R043.1
@pytest.mark.parametrize("through_port", [False, True])
async def test_admitted_attachments_reach_only_first_model_input(through_port):
    from pydantic_ai import BinaryContent

    png = BinaryContent(data=b"\x89PNG\r\n\x1a\nfixture", media_type="image/png")
    text = BinaryContent(data=b"attached text", media_type="text/plain")
    captured = []

    def observe(messages, info):
        captured.append(messages[-1].parts[-1].content)
        return ModelResponse(parts=[TextPart("ok")])

    plan = plan_of(
        FunctionModel(observe),
        steps=("inspect", "continue"),
        input_media_types=("image/png", "text/plain"),
    )
    request = RunRequest(task="inspect", attachments=(png, text))
    if through_port:
        host_ports.register_run_decision_provider(lambda request: plan)
        result = await AgentRunExecutor().execute_agent(request)
    else:
        result = await run(request, plan=plan)
    assert result.status == "completed"
    assert captured == [["inspect", png, "attached text"], "continue"]


# spec: AU-BOUNDARY-R043.1
@pytest.mark.parametrize(
    ("parts", "media", "budget"),
    [
        ((b"x", "image/png"), (), {}),
        ((b"x", "application/pdf"), ("application/pdf",), {}),
        ((b"", "image/png"), ("image/png",), {}),
        ((b"long", "image/png"), ("image/png",), {"max_attachment_bytes": 3}),
        ((b"x", "image/png"), ("image/png",), {"max_attachments": 0}),
        ((b"x", "image/png"), ("image/png",), {"max_attachment_bytes": True}),
        ((b"\xff", "text/plain"), ("text/plain",), {}),
    ],
)
async def test_attachment_refusal_precedes_model_construction(
    parts, media, budget, events
):
    from pydantic_ai import BinaryContent

    host_ports.register_model_provider(lambda model: pytest.fail("model constructed"))
    result = await run(
        RunRequest(
            task="inspect",
            attachments=(BinaryContent(data=parts[0], media_type=parts[1]),),
        ),
        plan=plan_of(
            "must-not-resolve", input_media_types=media, budget=RunBudget(**budget)
        ),
    )
    assert result.status == "failed"
    assert [event["type"] for event in events] == ["error"]


# spec: AU-BOUNDARY-R043.1
async def test_attachment_budget_is_aggregate_and_urls_are_not_fetched(events):
    from pydantic_ai import BinaryContent, ImageUrl

    host_ports.register_model_provider(lambda model: pytest.fail("model constructed"))
    plan = plan_of(
        "must-not-resolve",
        input_media_types=("image/png",),
        budget=RunBudget(max_attachment_bytes=3),
    )
    for parts in [
        (BinaryContent(data=b"xx", media_type="image/png"),) * 2,
        (ImageUrl(url="https://example.invalid/image.png"),),
    ]:
        result = await run(RunRequest(task="inspect", attachments=parts), plan=plan)
        assert result.status == "failed"
    assert [event["type"] for event in events] == ["error", "error"]


@pytest.mark.spec("AU-BOUNDARY-R043.7")
async def test_zero_initial_usage_preserves_legacy_serial_step_budgets():
    plan = plan_of(
        echo_model(), steps=("a", "b", "c"), budget=RunBudget(request_limit=1)
    )
    another = plan_of(echo_model())
    assert plan.initial_usage == RunUsage()
    assert plan.initial_usage is not another.initial_usage
    assert plan.initial_elapsed_seconds == 0.0
    result = await run(RunRequest(task="t"), plan=plan)
    assert (result.status, result.steps, result.usage["requests"]) == (
        "completed",
        3,
        3,
    )
    assert plan.initial_usage == RunUsage()


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("through_port", [False, True])
async def test_initial_usage_is_counted_once_and_reusable(through_port, events):
    async def reply(messages, info):
        return ModelResponse(
            parts=[TextPart("ok")],
            usage=RequestUsage(input_tokens=2, output_tokens=1),
        )

    initial = RunUsage(requests=2, tool_calls=3, input_tokens=11, output_tokens=7)
    original = asdict(initial)
    budget = RunBudget(request_limit=4, tool_calls_limit=5, total_tokens_limit=30)
    original_budget = asdict(budget)
    plan = plan_of(
        FunctionModel(reply), steps=("a", "b"), initial_usage=initial, budget=budget
    )
    if through_port:
        host_ports.register_run_decision_provider(lambda request: plan)
    expected = {
        "requests": 4,
        "tool_calls": 3,
        "input_tokens": 15,
        "output_tokens": 9,
        "total_tokens": 24,
    }
    for _ in range(2):
        events.clear()
        result = await run(RunRequest(task="t"), plan=None if through_port else plan)
        assert (result.status, result.steps) == ("completed", 2)
        assert result.usage == expected
        assert events[-1]["result"]["usage"] == expected
        assert sum(event["type"] == "graph_complete" for event in events) == 1
        assert asdict(initial) == original
        assert plan.budget is budget and asdict(budget) == original_budget


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize(
    ("initial", "budget", "elapsed"),
    [(RunUsage(requests=amount), RunBudget(request_limit=3), 0) for amount in (3, 4)]
    + [
        (RunUsage(input_tokens=amount), RunBudget(total_tokens_limit=7), 0)
        for amount in (7, 8)
    ]
    + [
        (RunUsage(tool_calls=limit + 1), RunBudget(tool_calls_limit=limit), 0)
        for limit in (0, 2)
    ]
    + [(RunUsage(), RunBudget(timeout_seconds=5), elapsed) for elapsed in (5, 6)],
)
async def test_exhausted_initial_usage_refuses_before_any_binding(
    initial, budget, elapsed, events
):
    host_ports.register_model_provider(lambda _: pytest.fail("model resolved"))
    host_ports.register_native_toolset_provider(
        host_ports.NativeToolsetProvider(
            available_tools=lambda: pytest.fail("tools discovered"),
            build_toolset=lambda *_: pytest.fail("tools bound"),
        )
    )
    plan = plan_of(
        "unbound",
        tool_names=("effect",),
        initial_usage=initial,
        initial_elapsed_seconds=elapsed,
        budget=budget,
    )
    result = await run(RunRequest(task="t"), plan=plan)
    assert result.status == "stopped" and result.steps == 0
    assert (result.error or "").startswith("budget:")
    assert result.usage == {
        "requests": initial.requests,
        "tool_calls": initial.tool_calls,
        "input_tokens": initial.input_tokens,
        "output_tokens": initial.output_tokens,
        "total_tokens": initial.total_tokens,
    }
    assert events[-1]["result"]["usage"] == result.usage
    assert not any(e["type"] in ("node_transition", "task_started") for e in events)


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("limit", [0, 2])
@pytest.mark.parametrize("attempt_tool", [False, True])
async def test_exact_initial_tool_limit_allows_only_model_work(
    limit, attempt_tool, events
):
    effects = []
    prompts = []
    toolset = FunctionToolset()

    @toolset.tool_plain
    async def effect() -> str:
        effects.append("effect")
        return "done"

    async def reply(messages, info):
        prompts.append(messages[-1].parts[-1].content)
        return ModelResponse(
            parts=[ToolCallPart("effect", {}) if attempt_tool else TextPart("ok")],
            usage=RequestUsage(input_tokens=2, output_tokens=1),
        )

    initial = RunUsage(requests=1, tool_calls=limit, input_tokens=3, output_tokens=4)
    result = await run(
        RunRequest(task="t"),
        plan=plan_of(
            FunctionModel(reply),
            steps=("a", "b"),
            toolsets=(toolset,),
            initial_usage=initial,
            budget=RunBudget(tool_calls_limit=limit),
        ),
    )
    calls = 1 if attempt_tool else 2
    assert prompts == ["a", "b"][:calls]
    assert (result.status, result.steps, result.output) == (
        "stopped" if attempt_tool else "completed",
        calls,
        "" if attempt_tool else "ok",
    )
    assert effects == []
    assert result.usage == {
        "requests": 1 + calls,
        "tool_calls": limit,
        "input_tokens": 3 + 2 * calls,
        "output_tokens": 4 + calls,
        "total_tokens": 7 + 3 * calls,
    }
    assert initial == RunUsage(
        requests=1, tool_calls=limit, input_tokens=3, output_tokens=4
    )
    assert [event["type"] for event in events] == (
        ["run_started"]
        + ([] if attempt_tool else ["node_transition"] * calls)
        + ["graph_complete"]
    )
    assert events[-1]["result"] == result.as_dict()
    if attempt_tool:
        assert (result.error or "").startswith("budget:")
        assert "tool_calls_limit" in result.error
    else:
        assert result.error is None


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize(
    "field", ["requests", "tool_calls", "input_tokens", "output_tokens"]
)
@pytest.mark.parametrize("value", [-1, True, 1.5])
async def test_invalid_initial_counters_fail_before_model_resolution(
    field, value, events
):
    host_ports.register_model_provider(lambda _: pytest.fail("model resolved"))
    initial = RunUsage(**{field: value})
    result = await run(
        RunRequest(task="t"), plan=plan_of("unbound", initial_usage=initial)
    )
    assert result.status == "failed" and result.steps == 0
    assert [event["type"] for event in events] == ["error"]


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize(
    "elapsed", [-1, True, float("nan"), float("inf"), float("-inf")]
)
async def test_invalid_initial_elapsed_fails_before_model_resolution(elapsed, events):
    host_ports.register_model_provider(lambda _: pytest.fail("model resolved"))
    result = await run(
        RunRequest(task="t"),
        plan=plan_of("unbound", initial_elapsed_seconds=elapsed),
    )
    assert result.status == "failed" and result.steps == 0
    assert [event["type"] for event in events] == ["error"]


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize(
    ("initial", "budget", "elapsed", "calls", "tokens"),
    [
        (RunUsage(requests=1), RunBudget(request_limit=3), 0, 2, 4),
        (RunUsage(input_tokens=3), RunBudget(total_tokens_limit=5), 0, 1, 5),
        (RunUsage(), RunBudget(request_limit=1), 0.5, 1, 2),
        (RunUsage(), RunBudget(total_tokens_limit=2), 0.5, 1, 2),
    ],
)
async def test_initial_usage_shares_remaining_limits_across_serial_steps(
    initial, budget, elapsed, calls, tokens, events
):
    prompts = []

    async def reply(messages, info):
        prompts.append(messages[-1].parts[-1].content)
        return ModelResponse(
            parts=[TextPart("ok")],
            usage=RequestUsage(input_tokens=1, output_tokens=1),
        )

    result = await run(
        RunRequest(task="t"),
        plan=plan_of(
            FunctionModel(reply),
            steps=("a", "b", "c"),
            initial_usage=initial,
            initial_elapsed_seconds=elapsed,
            budget=budget,
        ),
    )
    assert result.status == "stopped" and len(prompts) == calls
    assert result.usage["requests"] == initial.requests + calls
    assert result.usage["total_tokens"] == tokens
    assert events[-1]["result"]["usage"] == result.usage


@pytest.mark.spec("AU-BOUNDARY-R043.7")
async def test_initial_tool_usage_is_shared_across_serial_steps(events):
    effects = []
    toolset = FunctionToolset()

    @toolset.tool_plain
    async def effect() -> str:
        effects.append("effect")
        return "done"

    result = await run(
        RunRequest(task="t"),
        plan=plan_of(
            tool_model("effect"),
            steps=("a", "b"),
            toolsets=(toolset,),
            initial_usage=RunUsage(tool_calls=1),
            budget=RunBudget(tool_calls_limit=2),
        ),
    )
    assert result.status == "stopped" and effects == ["effect"]
    assert result.usage["tool_calls"] == 2
    assert events[-1]["result"]["usage"] == result.usage


@pytest.mark.spec("AU-BOUNDARY-R043.7")
async def test_initial_usage_snapshot_precedes_host_model_resolution():
    initial = RunUsage(requests=2, input_tokens=3, output_tokens=1)

    async def reply(messages, info):
        return ModelResponse(
            parts=[TextPart("ok")],
            usage=RequestUsage(input_tokens=1, output_tokens=1),
        )

    def resolve(model_id):
        initial.requests = 99
        initial.input_tokens = 999
        return FunctionModel(reply)

    host_ports.register_model_provider(resolve)
    result = await run(
        RunRequest(task="t"), plan=plan_of("host:model", initial_usage=initial)
    )
    assert result.status == "completed"
    assert result.usage == {
        "requests": 3,
        "tool_calls": 0,
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }


@pytest.mark.spec("AU-BOUNDARY-R043.7")
async def test_unreported_tokens_are_not_estimated_by_the_harness():
    class UnreportedUsageModel(TestModel):
        async def request(self, messages, model_settings, model_request_parameters):
            return ModelResponse(parts=[TextPart("not a token measurement")])

    result = await run(
        RunRequest(task="a long prompt is not a usage measurement"),
        plan=plan_of(UnreportedUsageModel(), initial_usage=RunUsage(requests=1)),
    )
    assert result.status == "completed"
    assert result.usage == {
        "requests": 2,
        "tool_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


@pytest.mark.spec("AU-BOUNDARY-R043.7")
async def test_request_metadata_and_model_output_cannot_replace_host_initial_usage():
    spoof = '{"initial_usage":{"requests":999},"initial_elapsed_seconds":9999}'

    async def reply(messages, info):
        return ModelResponse(
            parts=[TextPart(spoof)],
            usage=RequestUsage(input_tokens=1, output_tokens=1),
        )

    plan = plan_of(
        FunctionModel(reply), steps=("a", "b"), initial_usage=RunUsage(requests=2)
    )
    host_ports.register_run_decision_provider(lambda request: plan)
    result = await run(
        RunRequest(
            task="t",
            metadata={
                "initial_usage": {"requests": 999},
                "initial_elapsed_seconds": 9999,
            },
        )
    )
    assert result.status == "completed" and result.output == spoof
    assert result.usage["requests"] == 4 and result.usage["total_tokens"] == 4


@pytest.mark.spec("AU-BOUNDARY-R043.7")
async def test_initial_elapsed_reduces_timeout_without_changing_received_budget(
    monkeypatch,
):
    delays = []
    timeout_factory = asyncio.timeout

    def capture_timeout(delay):
        delays.append(delay)
        return timeout_factory(None)

    monkeypatch.setattr(asyncio, "timeout", capture_timeout)
    budget = RunBudget(timeout_seconds=10)
    plan = plan_of(echo_model(), budget=budget, initial_elapsed_seconds=2.5)
    result = await run(RunRequest(task="t"), plan=plan)
    assert result.status == "completed" and delays == [7.5]
    assert plan.budget is budget and budget.timeout_seconds == 10


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("exit_kind", ["setup_failure", "model_failure", "close"])
async def test_initial_usage_survives_serial_early_exit(exit_kind, events):
    initial = RunUsage(requests=2, tool_calls=1, input_tokens=3, output_tokens=4)

    async def fail(messages, info):
        raise ValueError("model failed")

    def resolve(model_id):
        raise ValueError("setup failed")

    host_ports.register_model_provider(resolve)
    plan = plan_of(
        "host:model" if exit_kind == "setup_failure" else FunctionModel(fail),
        initial_usage=initial,
    )
    if exit_kind == "close":
        stream = iterate_run(RunRequest(task="t"), plan=plan)
        assert (await anext(stream))["type"] == "run_started"
        await stream.aclose()
        assert events[-1]["type"] == "run_cancelled"
        usage = events[-1]["result"]["usage"]
    else:
        result = await run(RunRequest(task="t"), plan=plan)
        assert result.status == "failed"
        usage = result.usage
        assert events[-1]["result"]["usage"] == usage
    assert usage == {
        "requests": 3 if exit_kind == "model_failure" else 2,
        "tool_calls": 1,
        "input_tokens": 3,
        "output_tokens": 4,
        "total_tokens": 7,
    }
    assert initial == RunUsage(
        requests=2, tool_calls=1, input_tokens=3, output_tokens=4
    )


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("initial", [None, {"requests": 1}])
async def test_initial_usage_requires_a_typed_host_ledger(initial, events):
    host_ports.register_model_provider(lambda _: pytest.fail("model resolved"))
    result = await run(
        RunRequest(task="t"), plan=plan_of("unbound", initial_usage=initial)
    )
    assert result.status == "failed" and result.steps == 0
    assert "initial_usage" in result.error
    assert [event["type"] for event in events] == ["error"]


@pytest.mark.spec("AU-BOUNDARY-R043.7")
async def test_incoherent_initial_token_total_fails_before_model_resolution(events):
    class IncoherentUsage(RunUsage):
        @property
        def total_tokens(self):
            return super().total_tokens + 1

    host_ports.register_model_provider(lambda _: pytest.fail("model resolved"))
    result = await run(
        RunRequest(task="t"),
        plan=plan_of("unbound", initial_usage=IncoherentUsage(input_tokens=1)),
    )
    assert result.status == "failed" and result.steps == 0
    assert "total_tokens" in result.error
    assert [event["type"] for event in events] == ["error"]
