"""Behaviour of the agent run loop (agent_utilities.graph) over fake host ports."""

from __future__ import annotations

import asyncio
import copy
import pickle
from collections import UserDict
from dataclasses import asdict, replace
from types import MappingProxyType, SimpleNamespace
from unittest.mock import ANY

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

# The measured-timing snapshot (AU-BOUNDARY-R043.1) every terminal result
# carries. Only the client clock readings vary between runs.
NO_MODEL_METRICS = {
    "source": "client_monotonic_clock",
    "run_elapsed_ms": ANY,
    "observed_model_requests": 0,
    "coverage": "complete",
    "model_requests": [],
}
ONE_FAILED_MODEL_REQUEST_METRICS = {
    "source": "client_monotonic_clock",
    "run_elapsed_ms": ANY,
    "observed_model_requests": 1,
    "coverage": "complete",
    "model_requests": [
        {
            "sequence": 1,
            "source": "client_monotonic_clock",
            "outcome": "failed",
            "provider_name": "function",
            "model_name": "function:respond:",
            "client_request_elapsed_ms": ANY,
        }
    ],
}


# The first model request completes with reported tokens; the second fails.
COMPLETED_THEN_FAILED_METRICS = {
    "source": "client_monotonic_clock",
    "run_elapsed_ms": ANY,
    "observed_model_requests": 2,
    "coverage": "complete",
    "model_requests": [
        {
            "sequence": 1,
            "source": "client_monotonic_clock",
            "client_request_elapsed_ms": ANY,
            "outcome": "completed",
            "model_name": "function:respond:",
            "provider_name": "function",
            "input_tokens": 2,
            "output_tokens": 1,
        },
        {
            "sequence": 2,
            "source": "client_monotonic_clock",
            "client_request_elapsed_ms": ANY,
            "outcome": "failed",
            "model_name": "function:respond:",
            "provider_name": "function",
        },
    ],
}


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


@pytest.fixture
def no_planning_failure_execution():
    """Fail immediately if a rejected host decision reaches an execution port."""
    host_ports.register_model_provider(lambda _: pytest.fail("model resolved"))
    host_ports.register_native_toolset_provider(
        host_ports.NativeToolsetProvider(
            available_tools=lambda: pytest.fail("tools discovered"),
            build_toolset=lambda *_: pytest.fail("tools bound"),
        )
    )
    host_ports.register_scaffold_publication_factory(
        lambda _: pytest.fail("execution scaffold bound")
    )


def register_failing_decision(error: BaseException, *, is_async: bool = False):
    """Exercise both public synchronous and awaitable decision-provider forms."""
    seen = []

    def provide(request):
        seen.append(request)
        if is_async:

            async def later():
                raise error

            return later()
        raise error

    host_ports.register_run_decision_provider(provide)
    return seen


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("through_iterator", [False, True])
@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        (ValueError, "planner output validation failed"),
        (RuntimeError, "plan publication failed"),
        (LookupError, "published plan readback failed"),
        (PermissionError, "authority expired after planning"),
        (TimeoutError, "publication timed out after planning"),
    ],
)
async def test_measured_planning_failure_is_a_single_reusable_terminal_result(
    is_async,
    through_iterator,
    error_type,
    message,
    no_planning_failure_execution,
    events,
):
    from agent_utilities.graph import RunPlanFailure

    initial = RunUsage(
        requests=7,
        tool_calls=2,
        input_tokens=11,
        output_tokens=13,
        cache_read_tokens=41,
        details={"provider-extra": 9},
    )
    original = asdict(initial)
    error = error_type(message)
    failure = RunPlanFailure(error, initial_usage=initial, initial_elapsed_seconds=2.5)
    seen = register_failing_decision(failure, is_async=is_async)
    request = RunRequest(
        task="t", model_id="unbound", allowed_tools=("effect",), budget_tokens=1
    )
    expected = {
        "run_id": request.run_id,
        "output": "",
        "status": "failed",
        "steps": 0,
        "usage": {
            "requests": 7,
            "tool_calls": 2,
            "input_tokens": 11,
            "output_tokens": 13,
            "total_tokens": 24,
        },
        "error": f"{error_type.__name__}: {message}",
        "execution_mode": "graph",
        "metrics": NO_MODEL_METRICS,
    }
    for _ in range(2):
        events.clear()
        if through_iterator:
            emitted = [event async for event in iterate_run(request)]
            assert emitted == events
            assert emitted[-1]["result"] == expected
        else:
            result = await run(request)
            assert result.as_dict() == expected
        assert [event["type"] for event in events] == ["error"]
        assert events[0]["result"] == expected
        assert events[0]["error"] == expected["error"]
        assert events[0]["sequence"] == 1 and events[0]["step"] == 0
        assert asdict(initial) == original
    assert seen == [request, request]


@pytest.mark.spec("AU-BOUNDARY-R043.7")
async def test_measured_planning_limit_failure_preserves_original_error_and_usage(
    no_planning_failure_execution, events
):
    from pydantic_ai.exceptions import UsageLimitExceeded

    from agent_utilities.graph import RunPlanFailure

    error = UsageLimitExceeded("planner exceeded its original token allowance")
    register_failing_decision(
        RunPlanFailure(error, initial_usage=RunUsage(requests=3, input_tokens=15))
    )
    result = await run(RunRequest(task="t", budget_tokens=2))
    assert (result.status, result.steps, result.error) == (
        "failed",
        0,
        f"UsageLimitExceeded: {error}",
    )
    assert result.usage == {
        "requests": 3,
        "tool_calls": 0,
        "input_tokens": 15,
        "output_tokens": 0,
        "total_tokens": 15,
    }
    assert [event["type"] for event in events] == ["error"]
    assert events[0]["result"] == result.as_dict()


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("through_iterator", [False, True])
@pytest.mark.parametrize("error_type", [asyncio.CancelledError, GeneratorExit])
async def test_measured_planning_cancellation_keeps_ledger_and_original_exception(
    is_async, through_iterator, error_type, no_planning_failure_execution, events
):
    from agent_utilities.graph import RunPlanFailure

    initial = RunUsage(requests=2, tool_calls=1, input_tokens=9, output_tokens=3)
    original = asdict(initial)
    error = error_type("cancel after planner response")
    cause = ValueError("original cancellation cause")
    error.__cause__ = cause
    register_failing_decision(
        RunPlanFailure(error, initial_usage=initial, initial_elapsed_seconds=0.25),
        is_async=is_async,
    )
    request = RunRequest(task="t", model_id="unbound", allowed_tools=("effect",))
    with pytest.raises(error_type) as caught:
        if through_iterator:
            async for _ in iterate_run(request):
                pytest.fail("planning cancellation yielded an execution event")
        else:
            await run(request)
    assert caught.value is error
    assert caught.value.__cause__ is cause
    assert [event["type"] for event in events] == ["run_cancelled"]
    assert events[0]["status"] == "stopped"
    assert events[0]["sequence"] == 1 and events[0]["step"] == 0
    assert events[0]["result"] == {
        "run_id": request.run_id,
        "output": "",
        "status": "stopped",
        "steps": 0,
        "usage": {
            "requests": 2,
            "tool_calls": 1,
            "input_tokens": 9,
            "output_tokens": 3,
            "total_tokens": 12,
        },
        "error": None,
        "execution_mode": "graph",
        "metrics": NO_MODEL_METRICS,
    }
    assert asdict(initial) == original


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "error_type", [RuntimeError, asyncio.CancelledError, GeneratorExit]
)
async def test_raw_planning_failures_cannot_claim_unmeasured_usage(
    is_async, error_type, no_planning_failure_execution, events
):
    error = error_type("raw provider failure")
    error.initial_usage = RunUsage(requests=999, input_tokens=999)
    error.initial_elapsed_seconds = 999.0
    register_failing_decision(error, is_async=is_async)
    request = RunRequest(
        task="t",
        metadata={
            "initial_usage": {"requests": 888, "input_tokens": 888},
            "initial_elapsed_seconds": 888.0,
        },
    )
    if error_type is RuntimeError:
        result = await run(request)
        assert (result.status, result.steps, result.usage, result.error) == (
            "failed",
            0,
            {},
            "RuntimeError: raw provider failure",
        )
        assert [event["type"] for event in events] == ["error"]
        assert events[0]["result"] == result.as_dict()
    else:
        with pytest.raises(error_type) as caught:
            await run(request)
        assert caught.value is error
        assert events == []


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize(
    "field", ["requests", "tool_calls", "input_tokens", "output_tokens"]
)
@pytest.mark.parametrize("value", [-1, True, 1.5, "3", None])
async def test_invalid_planning_failure_counters_reject_the_whole_ledger(
    field, value, no_planning_failure_execution, events
):
    from agent_utilities.graph import RunPlanFailure

    initial = RunUsage(requests=2, tool_calls=1, input_tokens=9, output_tokens=3)
    setattr(initial, field, value)
    original = asdict(initial)

    def provide(request):
        raise RunPlanFailure(ValueError("original failure"), initial_usage=initial)

    host_ports.register_run_decision_provider(provide)
    result = await run(RunRequest(task="t"))
    assert (result.status, result.steps, result.usage) == ("failed", 0, {})
    assert "initial_usage" in result.error
    assert "nonnegative integers" in result.error
    assert [event["type"] for event in events] == ["error"]
    assert events[0]["result"] == result.as_dict()
    assert asdict(initial) == original


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize(
    "elapsed", [-1, True, float("nan"), float("inf"), float("-inf"), "2", None]
)
async def test_invalid_planning_failure_elapsed_rejects_the_whole_ledger(
    elapsed, no_planning_failure_execution, events
):
    from agent_utilities.graph import RunPlanFailure

    initial = RunUsage(requests=2, input_tokens=9, output_tokens=3)

    def provide(request):
        raise RunPlanFailure(
            RuntimeError("original failure"),
            initial_usage=initial,
            initial_elapsed_seconds=elapsed,
        )

    host_ports.register_run_decision_provider(provide)
    result = await run(RunRequest(task="t"))
    assert (result.status, result.steps, result.usage) == ("failed", 0, {})
    assert "initial_elapsed_seconds" in result.error
    assert [event["type"] for event in events] == ["error"]
    assert events[0]["result"] == result.as_dict()
    assert initial == RunUsage(requests=2, input_tokens=9, output_tokens=3)


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize(
    "initial",
    [None, {"requests": 2}, SimpleNamespace(requests=2), RequestUsage(input_tokens=2)],
)
async def test_planning_failure_requires_a_typed_host_ledger(
    initial, no_planning_failure_execution, events
):
    from agent_utilities.graph import RunPlanFailure

    def provide(request):
        raise RunPlanFailure(RuntimeError("original failure"), initial_usage=initial)

    host_ports.register_run_decision_provider(provide)
    result = await run(RunRequest(task="t"))
    assert (result.status, result.steps, result.usage) == ("failed", 0, {})
    assert "initial_usage must be RunUsage" in result.error
    assert [event["type"] for event in events] == ["error"]
    assert events[0]["result"] == result.as_dict()


@pytest.mark.spec("AU-BOUNDARY-R043.7")
async def test_planning_failure_rejects_an_incoherent_host_token_total(
    no_planning_failure_execution, events
):
    from agent_utilities.graph import RunPlanFailure

    class IncoherentUsage(RunUsage):
        @property
        def total_tokens(self):
            return super().total_tokens + 1

    def provide(request):
        raise RunPlanFailure(
            RuntimeError("original failure"),
            initial_usage=IncoherentUsage(requests=2, input_tokens=9, output_tokens=3),
        )

    host_ports.register_run_decision_provider(provide)
    result = await run(RunRequest(task="t"))
    assert (result.status, result.steps, result.usage) == ("failed", 0, {})
    assert "initial_usage total_tokens" in result.error
    assert [event["type"] for event in events] == ["error"]
    assert events[0]["result"] == result.as_dict()


@pytest.mark.spec("AU-BOUNDARY-R043.7")
async def test_model_cannot_inject_a_second_planning_failure_ledger(events):
    from agent_utilities.graph import RunPlanFailure

    async def fail(messages, info):
        raise RunPlanFailure(
            RuntimeError("not a trusted host planning failure"),
            initial_usage=RunUsage(requests=999, input_tokens=999),
            initial_elapsed_seconds=999.0,
        )

    plan = plan_of(FunctionModel(fail), initial_usage=RunUsage(requests=2))
    result = await run(
        RunRequest(
            task="t",
            metadata={
                "initial_usage": {"requests": 888, "input_tokens": 888},
                "initial_elapsed_seconds": 888.0,
            },
        ),
        plan=plan,
    )
    assert (result.status, result.steps) == ("failed", 1)
    assert result.usage == {
        "requests": 3,
        "tool_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    assert [event["type"] for event in events] == ["run_started", "error"]
    assert events[-1]["result"] == result.as_dict()
    assert plan.initial_usage == RunUsage(requests=2)


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("error_type", [asyncio.CancelledError, GeneratorExit])
@pytest.mark.parametrize("invalid_part", ["counter", "elapsed"])
async def test_invalid_planning_cancellation_ledger_fails_before_cancellation(
    error_type, invalid_part, no_planning_failure_execution, events
):
    from agent_utilities.graph import RunPlanFailure

    initial = RunUsage(requests=2, input_tokens=9, output_tokens=3)
    if invalid_part == "counter":
        initial.requests = -1
    register_failing_decision(
        RunPlanFailure(
            error_type("cancel after planning"),
            initial_usage=initial,
            initial_elapsed_seconds=float("nan") if invalid_part == "elapsed" else 0,
        )
    )
    result = await run(RunRequest(task="t"))
    assert (result.status, result.steps, result.usage) == ("failed", 0, {})
    assert result.error.startswith("ValueError: initial_")
    assert [event["type"] for event in events] == ["error"]
    assert events[0]["result"] == result.as_dict()


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("requests", [0, 2])
async def test_planning_failure_does_not_estimate_unreported_tokens(
    requests, no_planning_failure_execution, events
):
    from agent_utilities.graph import RunPlanFailure

    initial = RunUsage(requests=requests)
    register_failing_decision(
        RunPlanFailure(
            RuntimeError("planner tokens were not reported"), initial_usage=initial
        )
    )
    result = await run(
        RunRequest(task="Prompt and output text must not become token measurements")
    )
    assert (result.status, result.steps) == ("failed", 0)
    assert result.usage == {
        "requests": requests,
        "tool_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    assert [event["type"] for event in events] == ["error"]
    assert events[0]["result"] == result.as_dict()
    assert initial == RunUsage(requests=requests)


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("error", [None, "not an exception", RuntimeError])
def test_planning_failure_requires_an_exception_instance(error):
    from agent_utilities.graph import RunPlanFailure

    with pytest.raises(TypeError, match="error must be an exception"):
        RunPlanFailure(error, initial_usage=RunUsage(requests=2))


@pytest.mark.spec("AU-BOUNDARY-R043.7")
async def test_task_cancellation_after_planner_consumption_reports_actual_usage(
    no_planning_failure_execution, events
):
    from agent_utilities.graph import RunPlanFailure

    consumed = asyncio.Event()
    publication = asyncio.Event()
    original_cancellations = []
    initial = RunUsage(requests=2, input_tokens=9, output_tokens=3)

    async def provide(request):
        consumed.set()
        try:
            await publication.wait()
        except asyncio.CancelledError as error:
            original_cancellations.append(error)
            raise RunPlanFailure(error, initial_usage=initial) from error
        pytest.fail("publication unexpectedly completed")

    host_ports.register_run_decision_provider(provide)
    request = RunRequest(task="t", model_id="unbound", allowed_tools=("effect",))
    execution = asyncio.create_task(run(request))
    try:
        await asyncio.wait_for(consumed.wait(), timeout=0.5)
        execution.cancel("stop while publishing the measured plan")
        with pytest.raises(asyncio.CancelledError) as caught:
            await execution
    finally:
        if not execution.done():
            execution.cancel()
        await asyncio.gather(execution, return_exceptions=True)
    assert len(original_cancellations) == 1
    assert caught.value is original_cancellations[0]
    assert str(caught.value) == "stop while publishing the measured plan"
    assert execution.cancelled()
    assert [event["type"] for event in events] == ["run_cancelled"]
    assert events[0]["status"] == "stopped"
    assert events[0]["result"] == {
        "run_id": request.run_id,
        "output": "",
        "status": "stopped",
        "steps": 0,
        "usage": {
            "requests": 2,
            "tool_calls": 0,
            "input_tokens": 9,
            "output_tokens": 3,
            "total_tokens": 12,
        },
        "error": None,
        "execution_mode": "graph",
        "metrics": NO_MODEL_METRICS,
    }
    assert initial == RunUsage(requests=2, input_tokens=9, output_tokens=3)


@pytest.fixture
def failed_run_result():
    from agent_utilities.graph import RunResult

    return RunResult(
        run_id="host-assigned-run-id",
        output="partial output",
        status="failed",
        steps=2,
        usage={
            "requests": 2,
            "tool_calls": 1,
            "input_tokens": 5,
            "output_tokens": 7,
            "total_tokens": 12,
        },
        error="ValueError: execution failed",
    )


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("error", ["ValueError: execution failed", "", None])
@pytest.mark.parametrize("mode", ["direct", "graph"])
def test_run_failed_is_public_and_preserves_runtime_error_message(
    error, mode, failed_run_result
):
    from agent_utilities import graph
    from agent_utilities.graph import RunFailed, RunResult

    result = replace(failed_run_result, error=error, execution_mode=mode)
    failure = RunFailed(result)
    assert "RunFailed" in graph.__all__ and graph.RunFailed is RunFailed
    assert isinstance(failure, RuntimeError)
    assert isinstance(failure.result, RunResult)
    assert failure.result is not result
    assert failure.result.as_dict() == result.as_dict()
    assert (str(failure), failure.args) == (
        error or "agent run failed",
        (error or "agent run failed",),
    )
    assert failure.run_id == result.run_id
    assert failure.usage == result.usage


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("mapping_type", [dict, UserDict, MappingProxyType])
@pytest.mark.parametrize("empty", [False, True])
def test_run_failed_snapshots_mapping_without_aliasing_or_estimating(
    mapping_type, empty, failed_run_result
):
    from agent_utilities.graph import RunFailed

    source = {} if empty else dict(failed_run_result.usage)
    result = replace(failed_run_result, usage=mapping_type(source))
    expected = result.as_dict()
    failure = RunFailed(result)
    if not isinstance(result.usage, MappingProxyType):
        result.usage.clear()
    source.clear()
    source["requests"] = 999
    assert failure.result.as_dict() == expected
    assert failure.usage == expected["usage"]
    with pytest.raises(TypeError):
        failure.usage["requests"] = 999
    with pytest.raises(TypeError):
        failure.result.usage["requests"] = 999
    exported = failure.result.as_dict()
    exported["run_id"] = "changed"
    exported["usage"]["requests"] = 999
    assert failure.result.as_dict() == expected


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("field", ["result", "run_id", "usage"])
def test_run_failed_receipt_properties_are_read_only(field, failed_run_result):
    from agent_utilities.graph import RunFailed

    failure = RunFailed(failed_run_result)
    with pytest.raises(AttributeError):
        setattr(failure, field, None)
    with pytest.raises(AttributeError):
        failure.result.run_id = "changed"
    with pytest.raises(AttributeError):
        failure.result.usage = {}
    assert failure.result.as_dict() == failed_run_result.as_dict()


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("method", ["copy", "deepcopy", "pickle"])
@pytest.mark.parametrize("empty", [False, True])
def test_run_failed_copy_and_pickle_preserve_receipt_and_exception_state(
    method, empty, failed_run_result
):
    from agent_utilities.graph import RunFailed

    result = replace(failed_run_result, usage={}) if empty else failed_run_result
    failure = RunFailed(result)
    failure.add_note("host diagnostic note")
    failure.context = {"phase": ["execution"]}
    if method == "pickle":
        restored = pickle.loads(pickle.dumps(failure))
    else:
        restored = getattr(copy, method)(failure)
    assert type(restored) is RunFailed and restored is not failure
    assert restored.args == failure.args and str(restored) == str(failure)
    assert restored.result.as_dict() == failure.result.as_dict()
    assert restored.run_id == failure.run_id and restored.usage == failure.usage
    assert restored.__notes__ == failure.__notes__
    assert restored.context == failure.context
    assert (restored.context is failure.context) is (method == "copy")
    with pytest.raises(TypeError):
        restored.result.usage["requests"] = 999


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("result", [None, {}, SimpleNamespace(), RuntimeError("x")])
def test_run_failed_rejects_non_result_receipts(result):
    from agent_utilities.graph import RunFailed

    with pytest.raises((TypeError, ValueError)):
        RunFailed(result)


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", value)
        for value in ("pending", "running", "completed", "stopped", None)
    ]
    + [("run_id", value) for value in ("", None, 1, False)]
    + [("output", value) for value in (None, 1, b"text")]
    + [("error", value) for value in (1, b"error", RuntimeError("not text"))]
    + [("steps", value) for value in (-1, True, False, 1.0, "1", None)]
    + [("execution_mode", value) for value in ("", "ask", None, 1)],
)
def test_run_failed_rejects_invalid_result_fields(field, value, failed_run_result):
    from agent_utilities.graph import RunFailed

    with pytest.raises((TypeError, ValueError)):
        RunFailed(replace(failed_run_result, **{field: value}))


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize(
    "usage", [None, [], (), [("requests", 2)], RunUsage(requests=2), False]
)
def test_run_failed_requires_a_usage_mapping(usage, failed_run_result):
    from agent_utilities.graph import RunFailed

    with pytest.raises((TypeError, ValueError)):
        RunFailed(replace(failed_run_result, usage=usage))


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize(
    "missing",
    ["requests", "tool_calls", "input_tokens", "output_tokens", "total_tokens"],
)
def test_run_failed_rejects_partial_usage_without_filling_counters(
    missing, failed_run_result
):
    from agent_utilities.graph import RunFailed

    usage = dict(failed_run_result.usage)
    del usage[missing]
    with pytest.raises((TypeError, ValueError)):
        RunFailed(replace(failed_run_result, usage=usage))


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("extra", ["cached_tokens", "details", 0])
def test_run_failed_rejects_noncanonical_usage_keys(extra, failed_run_result):
    from agent_utilities.graph import RunFailed

    with pytest.raises((TypeError, ValueError)):
        RunFailed(
            replace(failed_run_result, usage={**failed_run_result.usage, extra: 0})
        )


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize(
    "field", ["requests", "tool_calls", "input_tokens", "output_tokens", "total_tokens"]
)
@pytest.mark.parametrize("value", [-1, True, False, 1.0, "1", None])
def test_run_failed_rejects_invalid_usage_counters(field, value, failed_run_result):
    from agent_utilities.graph import RunFailed

    with pytest.raises((TypeError, ValueError)):
        RunFailed(
            replace(failed_run_result, usage={**failed_run_result.usage, field: value})
        )


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("total", [11, 13])
def test_run_failed_rejects_incoherent_total(total, failed_run_result):
    from agent_utilities.graph import RunFailed

    with pytest.raises((TypeError, ValueError)):
        RunFailed(
            replace(
                failed_run_result,
                usage={**failed_run_result.usage, "total_tokens": total},
            )
        )


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("counter", [0, 2**65])
def test_run_failed_preserves_exact_nonnegative_integer_counters(
    counter, failed_run_result
):
    from agent_utilities.graph import RunFailed

    usage = {key: counter for key in failed_run_result.usage}
    usage["total_tokens"] = counter * 2
    failure = RunFailed(replace(failed_run_result, usage=usage, steps=0))
    assert failure.usage == usage and failure.result.steps == 0


def register_measured_run_failure(phase, initial):
    """Bind public fake host ports for a failure at the selected execution stage."""
    from agent_utilities.graph import DependencyPlan, RunPlanFailure, RunTask

    calls = []

    async def respond(messages, info):
        prompt = messages[-1].parts[-1].content.split("\n", 1)[0]
        calls.append(prompt)
        if prompt == "fail":
            raise ValueError("execution failed")
        return ModelResponse(
            parts=[TextPart("partial output")],
            usage=RequestUsage(input_tokens=2, output_tokens=1),
        )

    def resolve(model_id):
        raise ValueError("setup failed")

    if phase == "planner":
        register_failing_decision(
            RunPlanFailure(ValueError("planner failed"), initial_usage=initial)
        )
        return calls
    host_ports.register_model_provider(resolve)
    dependency_plan = None
    if phase == "dag":
        dependency_plan = DependencyPlan(
            tasks=(RunTask("first", "first"), RunTask("fail", "fail", ("first",))),
            output_task_id="fail",
        )
    plan = plan_of(
        "host:model" if phase == "setup" else FunctionModel(respond),
        steps=() if phase == "dag" else ("first", "fail"),
        initial_usage=initial,
        dependency_plan=dependency_plan,
    )
    host_ports.register_run_decision_provider(lambda request: plan)
    return calls


async def failed_result_from_entrypoint(entrypoint, request, events):
    """Observe the same failed public result through each supported entrypoint."""
    from agent_utilities.graph import RunFailed

    if entrypoint == "executor":
        with pytest.raises(RunFailed) as caught:
            await AgentRunExecutor().execute_agent(request)
        result = caught.value.result.as_dict()
        assert caught.value.run_id == request.run_id
        assert caught.value.usage == result["usage"]
        assert caught.value.args == (result["error"],)
        return result
    if entrypoint == "run":
        return (await run(request)).as_dict()
    emitted = [event async for event in iterate_run(request)]
    assert emitted == events
    return emitted[-1]["result"]


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize(
    ("phase", "output", "steps", "requests", "input_tokens", "output_tokens", "error"),
    [
        ("planner", "", 0, 2, 3, 4, "ValueError: planner failed"),
        ("setup", "", 0, 2, 3, 4, "ValueError: setup failed"),
        ("model", "partial output", 2, 4, 5, 5, "ValueError: execution failed"),
        ("dag", "", 2, 4, 5, 5, "ValueError: execution failed"),
    ],
    ids=["planner", "setup", "model", "dag"],
)
@pytest.mark.parametrize("entrypoint", ["executor", "run", "iterate"])
async def test_failed_run_receipt_matches_the_single_terminal_aggregate(
    phase,
    output,
    steps,
    requests,
    input_tokens,
    output_tokens,
    error,
    entrypoint,
    events,
):
    initial = RunUsage(requests=2, tool_calls=1, input_tokens=3, output_tokens=4)
    original = asdict(initial)
    calls = register_measured_run_failure(phase, initial)
    request = RunRequest(task="t", run_id="actual-run")
    result = await failed_result_from_entrypoint(entrypoint, request, events)
    assert result == {
        "run_id": request.run_id,
        "output": output,
        "status": "failed",
        "steps": steps,
        "usage": {
            "requests": requests,
            "tool_calls": 1,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "error": error,
        "execution_mode": "graph",
        "metrics": NO_MODEL_METRICS if steps == 0 else COMPLETED_THEN_FAILED_METRICS,
    }
    assert calls == ["first", "fail"][:steps]
    terminals = [
        event
        for event in events
        if event["type"] in ("error", "graph_complete", "run_cancelled")
    ]
    assert terminals == [events[-1]]
    assert terminals[0]["type"] == "error"
    assert terminals[0]["result"] == result
    assert asdict(initial) == original


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("phase", ["planner", "setup", "default"])
async def test_executor_failure_preserves_missing_usage_as_empty(phase, events):
    from agent_utilities.graph import RunFailed

    def fail(*args):
        raise ValueError("no measured usage")

    if phase == "planner":
        host_ports.register_run_decision_provider(fail)
    elif phase == "setup":
        host_ports.register_model_provider(fail)
        host_ports.register_run_decision_provider(lambda request: plan_of("host:model"))
    request = RunRequest(task="unreported tokens must not be estimated")
    with pytest.raises(RunFailed) as caught:
        await AgentRunExecutor().execute_agent(request)
    assert caught.value.run_id == request.run_id
    assert caught.value.usage == {}
    assert caught.value.result.steps == 0
    assert events[-1]["result"] == caught.value.result.as_dict()
    assert sum(event["type"] == "error" for event in events) == 1


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("phase", ["planner", "setup", "model"])
async def test_executor_uses_its_own_aggregate_not_a_foreign_failure_receipt(
    phase, failed_run_result, events
):
    from agent_utilities.graph import RunFailed

    forged = RunFailed(replace(failed_run_result, run_id="foreign-run"))

    def fail(*args):
        raise forged

    if phase == "planner":
        host_ports.register_run_decision_provider(fail)
    else:
        host_ports.register_model_provider(fail)
        plan = plan_of(
            "host:model" if phase == "setup" else FunctionModel(fail),
            initial_usage=RunUsage(requests=2, input_tokens=3, output_tokens=4),
        )
        host_ports.register_run_decision_provider(lambda request: plan)
    request = RunRequest(task="t", run_id="actual-run")
    with pytest.raises(RunFailed) as caught:
        await AgentRunExecutor().execute_agent(request)
    assert caught.value is not forged
    assert caught.value.run_id == request.run_id
    assert caught.value.result.output == ""
    assert caught.value.result.steps == (1 if phase == "model" else 0)
    assert caught.value.usage == (
        {}
        if phase == "planner"
        else {
            "requests": 3 if phase == "model" else 2,
            "tool_calls": 0,
            "input_tokens": 3,
            "output_tokens": 4,
            "total_tokens": 7,
        }
    )
    assert caught.value.result.error == f"RunFailed: {forged}"
    assert events[-1]["result"] == caught.value.result.as_dict()
    assert sum(event["type"] == "error" for event in events) == 1
    assert (
        forged.result.as_dict()
        == replace(failed_run_result, run_id="foreign-run").as_dict()
    )


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("kind", ["completed", "stop_marker", "budget"])
async def test_executor_keeps_successful_and_stopped_return_contract(kind, events):
    from agent_utilities.graph import RunResult

    plan = plan_of(
        TestModel(custom_output_text="ok"),
        initial_usage=RunUsage(requests=2, input_tokens=3),
        stop=StopRule(stop_markers=("ok",)) if kind == "stop_marker" else StopRule(),
        budget=RunBudget(request_limit=2) if kind == "budget" else RunBudget(),
    )
    host_ports.register_run_decision_provider(lambda request: plan)
    result = await AgentRunExecutor().execute_agent(RunRequest(task="t"))
    assert isinstance(result, RunResult)
    assert result.status == ("completed" if kind == "completed" else "stopped")
    assert result.steps == (0 if kind == "budget" else 1)
    assert events[-1]["type"] == "graph_complete"
    assert events[-1]["result"] == result.as_dict()
    assert sum(event["type"] == "graph_complete" for event in events) == 1


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("error_type", [asyncio.CancelledError, GeneratorExit])
@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
async def test_executor_preserves_raw_and_measured_planner_cancellation(
    error_type, measured, is_async, no_planning_failure_execution, events
):
    from agent_utilities.graph import RunPlanFailure

    error = error_type("cancel during planning")
    cause = ValueError("original cause")
    error.__cause__ = cause
    initial = RunUsage(requests=2, input_tokens=3, output_tokens=4)
    register_failing_decision(
        RunPlanFailure(error, initial_usage=initial) if measured else error,
        is_async=is_async,
    )
    request = RunRequest(task="t")
    with pytest.raises(error_type) as caught:
        await AgentRunExecutor().execute_agent(request)
    assert caught.value is error and caught.value.__cause__ is cause
    if measured:
        assert [event["type"] for event in events] == ["run_cancelled"]
        assert events[0]["result"] == {
            "run_id": request.run_id,
            "output": "",
            "status": "stopped",
            "steps": 0,
            "usage": {
                "requests": 2,
                "tool_calls": 0,
                "input_tokens": 3,
                "output_tokens": 4,
                "total_tokens": 7,
            },
            "error": None,
            "execution_mode": "graph",
            "metrics": NO_MODEL_METRICS,
        }
    else:
        assert events == []
    assert initial == RunUsage(requests=2, input_tokens=3, output_tokens=4)


@pytest.mark.spec("AU-BOUNDARY-R043.7")
async def test_executor_task_cancellation_remains_cancelled_with_measured_usage(events):
    started = asyncio.Event()
    release = asyncio.Event()
    original_cancellations = []

    async def respond(messages, info):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError as error:
            original_cancellations.append(error)
            raise
        pytest.fail("model unexpectedly resumed")

    plan = plan_of(
        FunctionModel(respond),
        initial_usage=RunUsage(requests=2, input_tokens=3, output_tokens=4),
    )
    host_ports.register_run_decision_provider(lambda request: plan)
    request = RunRequest(task="t")
    execution = asyncio.create_task(AgentRunExecutor().execute_agent(request))
    try:
        await asyncio.wait_for(started.wait(), timeout=0.5)
        execution.cancel("stop model execution")
        with pytest.raises(asyncio.CancelledError) as caught:
            await execution
    finally:
        if not execution.done():
            execution.cancel()
        await asyncio.gather(execution, return_exceptions=True)
    assert execution.cancelled()
    assert len(original_cancellations) == 1
    assert isinstance(original_cancellations[0], asyncio.CancelledError)
    assert str(caught.value) == "stop model execution"
    assert [event["type"] for event in events] == ["run_started", "run_cancelled"]
    assert events[-1]["result"] == {
        "run_id": request.run_id,
        "output": "",
        "status": "stopped",
        "steps": 1,
        "usage": {
            "requests": 3,
            "tool_calls": 0,
            "input_tokens": 3,
            "output_tokens": 4,
            "total_tokens": 7,
        },
        "error": None,
        "execution_mode": "graph",
        "metrics": ONE_FAILED_MODEL_REQUEST_METRICS,
    }


@pytest.mark.spec("AU-BOUNDARY-R043.11")
@pytest.mark.parametrize("through_port", [False, True])
async def test_terminal_observer_preserves_host_mapping_tools_and_privacy(
    through_port, events
):
    from pydantic_ai.messages import PartStartEvent, ThinkingPart
    from pydantic_ai.models.function import DeltaThinkingPart, DeltaToolCall

    session = object()
    received, native, tool_calls = [], [], []
    tools = FunctionToolset()

    @tools.tool
    def native_read(ctx) -> str:
        tool_calls.append("read")
        return "read result"

    async def stream(messages, info):
        if messages[-1].parts[-1].part_kind == "tool-return":
            yield {0: DeltaThinkingPart(content="private reasoning")}
            yield "answer"
        else:
            yield {0: DeltaToolCall(name="native_read", json_args="{}")}

    request = RunRequest(
        task="inspect",
        agent_name="native",
        run_id="admitted-id",
        context="admitted-context",
        allowed_tools=("native_read",),
        max_steps=1,
        budget_tokens=500,
        metadata={"session": session},
    )

    def decide(value):
        received.append(value)
        return RunPlan(
            model=FunctionModel(stream_function=stream),
            steps=(value.task,),
            toolsets=(tools,),
        )

    async def observe(event):
        native.append(event)

    host_ports.register_run_decision_provider(decide)
    if through_port:
        result = await AgentRunExecutor().execute_agent(
            request, session=session, stream_observer=observe
        )
    else:
        result = await run(request, stream_observer=observe)
    assert result.status == "completed" and result.output == "answer"
    assert result.run_id == "admitted-id" and tool_calls == ["read"]
    assert received == [request]
    assert received[0].metadata["session"] is session
    assert any(
        isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart)
        for event in native
    )
    assert "private reasoning" not in str(events)


@pytest.mark.spec("AU-BOUNDARY-R043.11")
@pytest.mark.parametrize("through_port", [False, True])
@pytest.mark.parametrize("outcome", ["failure", "cancel", "budget"])
async def test_terminal_observer_keeps_failure_cancel_and_budget(
    through_port, outcome, events
):
    from agent_utilities.graph import RunFailed

    model_calls, observed = [], []

    async def stream(messages, info):
        model_calls.append("called")
        yield "first"
        yield "never retried"

    async def observe(event):
        observed.append(event)
        if outcome == "cancel":
            raise asyncio.CancelledError
        raise RuntimeError("secret observer detail")

    host_ports.register_run_decision_provider(
        lambda request: RunPlan(
            model=FunctionModel(stream_function=stream),
            steps=(request.task,),
            budget=RunBudget(request_limit=0 if outcome == "budget" else 2),
        )
    )
    request = RunRequest(task="inspect")
    execution = (
        AgentRunExecutor().execute_agent(request, stream_observer=observe)
        if through_port
        else run(request, stream_observer=observe)
    )
    if outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await execution
        assert events[-1]["type"] == "run_cancelled"
    elif outcome == "failure" and through_port:
        with pytest.raises(RunFailed) as caught:
            await execution
        assert "model stream observer failed" in caught.value.result.error
    else:
        result = await execution
        assert result.status == ("stopped" if outcome == "budget" else "failed")
    assert len(model_calls) == len(observed) == (0 if outcome == "budget" else 1)
    assert "secret observer detail" not in str(events)


@pytest.mark.spec("AU-BOUNDARY-R043.11")
async def test_native_executor_observers_are_per_invocation():
    from pydantic_ai.messages import PartStartEvent

    both_entered = asyncio.Event()
    entered, outputs = [], {"one": [], "two": []}

    async def stream(messages, info):
        label = messages[-1].parts[-1].content
        entered.append(label)
        if len(entered) == 2:
            both_entered.set()
        await both_entered.wait()
        yield label

    host_ports.register_run_decision_provider(
        lambda request: RunPlan(
            model=FunctionModel(stream_function=stream),
            steps=(request.task,),
        )
    )
    executor = AgentRunExecutor()

    async def execute(label):
        async def observe(event):
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                outputs[label].append(event.part.content)

        return await executor.execute_agent(
            RunRequest(task=label), stream_observer=observe
        )

    first, second = await asyncio.gather(execute("one"), execute("two"))
    assert [first.output, second.output] == ["one", "two"]
    assert outputs == {"one": ["one"], "two": ["two"]}
