"""Behaviour of the recursive run loop (agent_utilities.rlm)."""

from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from agent_utilities import host_ports
from agent_utilities.graph import RunPlan
from agent_utilities.rlm import RLMConfig, SandboxFatalError, classify_failure, run_rlm
from agent_utilities.rlm.telemetry import dominant_failure


class SandboxRejected(Exception):
    pass


# spec: AU-BOUNDARY-R043.3
@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (SandboxFatalError("x"), "sandbox_fatal"),
        (TimeoutError(), "host_tool_timeout"),
        (SandboxRejected("x"), "sandbox_escalated"),
        ("subprocess has been killed", "sandbox_fatal"),
        ("sandbox timeout", "sandbox_exec_timeout"),
        ("tool timeout", "host_tool_timeout"),
        ("SyntaxError: bad", "model_generated_bad_code"),
        ("output rejected", "evaluator_reject"),
        ("??", "unknown"),
    ],
)
def test_failures_are_classified(exc, expected):
    assert classify_failure(exc) == expected


def test_dominant_failure_follows_precedence():
    assert dominant_failure(["evaluator_reject", "sandbox_fatal"]) == "sandbox_fatal"
    assert dominant_failure([]) == "unknown"


def scripted(*calls: tuple[str, dict]) -> FunctionModel:
    """Make each tool call in turn (per run), then answer with the last return."""

    def fn(messages, info: AgentInfo) -> ModelResponse:
        returns = [
            p
            for m in messages
            for p in m.parts
            if getattr(p, "part_kind", "") == "tool-return"
        ]
        if len(returns) < len(calls):
            name, args = calls[len(returns)]
            return ModelResponse(parts=[ToolCallPart(name, args)])
        return ModelResponse(
            parts=[TextPart(str(returns[-1].content) if returns else "done")]
        )

    return FunctionModel(fn)


# spec: AU-BOUNDARY-R043.3
async def test_root_reads_slices_and_recurses():
    root = scripted(
        ("read_input", {"start": 2, "end": 5}), ("sub_query", {"question": "q"})
    )
    child = TestModel(call_tools=[], custom_output_text="child-answer")
    out = await run_rlm("t", "abcdefgh", config=RLMConfig(model=root, sub_model=child))
    assert out["ok"] and out["result"] == "child-answer"
    assert out["usage"]["requests"] >= 3 and out["max_depth"] == 3


# spec: AU-BOUNDARY-R043.3
async def test_read_input_is_capped_by_the_slice_limit():
    root = scripted(("read_input", {"start": 0, "end": 100}))
    out = await run_rlm("t", "x" * 50, config=RLMConfig(model=root, max_slice_chars=4))
    assert out["result"] == "xxxx"


# spec: AU-BOUNDARY-R043.3
@pytest.mark.parametrize("max_depth", [0, 1])
async def test_recursion_stops_at_max_depth(max_depth):
    root = scripted(("sub_query", {"question": "q"}))
    out = await run_rlm("t", "abc", config=RLMConfig(model=root, max_depth=max_depth))
    assert out["result"] == "refused: maximum recursion depth reached"


async def test_model_failure_returns_a_typed_failure():
    def boom(messages, info):
        raise RuntimeError("sandbox timeout")

    out = await run_rlm("t", config=RLMConfig(model=FunctionModel(boom)))
    assert (out["ok"], out["failure_class"]) == (False, "sandbox_exec_timeout")


# spec: AU-BOUNDARY-R043.3
async def test_fatal_failure_propagates():
    def boom(messages, info):
        raise SandboxFatalError("dead")

    with pytest.raises(SandboxFatalError):
        await run_rlm("t", config=RLMConfig(model=FunctionModel(boom)))


# spec: AU-BOUNDARY-R043.3
async def test_without_a_model_the_host_plan_supplies_it():
    host_ports.register_run_decision_provider(
        lambda req: RunPlan(
            model=TestModel(call_tools=[], custom_output_text=req.agent_name)
        )
    )
    out = await run_rlm("t")
    assert out["result"] == "rlm"


async def test_without_any_model_the_run_fails():
    out = await run_rlm("t")
    assert out["ok"] is False and "no model" in out["error"]


# spec: AU-BOUNDARY-R043.7
async def test_recursive_requests_refill_slots_without_holding_parent_permits():
    import asyncio

    started = {key: asyncio.Event() for key in ("A", "B", "C")}
    release = {key: asyncio.Event() for key in started}
    release["B"].set()
    active = peak = 0
    timeline = []

    async def root(messages, info):
        if any(
            getattr(p, "part_kind", "") == "tool-return" for p in messages[-1].parts
        ):
            return ModelResponse(parts=[TextPart("joined")])
        return ModelResponse(
            parts=[ToolCallPart("sub_query", {"question": key}) for key in started]
        )

    async def child(messages, info):
        nonlocal active, peak
        key = messages[-1].parts[-1].content.split("\n", 1)[0]
        active += 1
        peak = max(active, peak)
        timeline.append((key, "start"))
        started[key].set()
        try:
            await release[key].wait()
            timeline.append((key, "end"))
            return ModelResponse(parts=[TextPart(key)])
        finally:
            active -= 1

    parent = asyncio.create_task(
        run_rlm(
            "root",
            config=RLMConfig(
                model=FunctionModel(root),
                sub_model=FunctionModel(child),
                max_concurrent_requests=2,
            ),
        )
    )
    try:
        await asyncio.wait_for(started["C"].wait(), 0.5)
        assert ("A", "end") not in timeline
        assert peak == 2 and not parent.done()
        release["A"].set()
        release["C"].set()
        result = await parent
    finally:
        if not parent.done():
            parent.cancel()
        await asyncio.gather(parent, return_exceptions=True)
    assert result["ok"] and result["result"] == "joined"
    assert result["usage"]["requests"] == 5
    assert active == 0


# spec: AU-BOUNDARY-R043.7
@pytest.mark.parametrize("cancel", [False, True])
async def test_nested_recursion_capacity_one_finishes_or_drains_on_cancel(cancel):
    import asyncio

    deepest = asyncio.Event()
    release = asyncio.Event()
    drained = asyncio.Event()
    if not cancel:
        release.set()

    async def model(messages, info):
        if any(
            getattr(p, "part_kind", "") == "tool-return" for p in messages[-1].parts
        ):
            return ModelResponse(parts=[TextPart("done")])
        key = messages[-1].parts[-1].content.split("\n", 1)[0]
        if key in ("root", "child"):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "sub_query", {"question": "child" if key == "root" else "leaf"}
                    )
                ]
            )
        deepest.set()
        try:
            await release.wait()
            return ModelResponse(parts=[TextPart("leaf answer")])
        finally:
            drained.set()

    parent = asyncio.create_task(
        run_rlm(
            "root",
            config=RLMConfig(
                model=FunctionModel(model),
                max_concurrent_requests=1,
            ),
        )
    )
    await asyncio.wait_for(deepest.wait(), 0.5)
    if cancel:
        parent.cancel()
        with pytest.raises(asyncio.CancelledError):
            await parent
    else:
        result = await asyncio.wait_for(parent, 0.5)
        assert result["ok"] and result["usage"]["requests"] == 5
    assert drained.is_set()


# spec: AU-BOUNDARY-R043.7
@pytest.mark.parametrize(
    "request_limit,tool_limit,expected_calls", [(0, 10, 0), (1, 10, 1), (5, 2, 1)]
)
async def test_recursive_shared_budget_and_batch_refusal(
    request_limit, tool_limit, expected_calls
):
    calls = []

    async def model(messages, info):
        calls.append("call")
        return ModelResponse(
            parts=[ToolCallPart("sub_query", {"question": str(i)}) for i in range(3)]
        )

    result = await run_rlm(
        "root",
        config=RLMConfig(
            model=FunctionModel(model),
            request_limit=request_limit,
            tool_calls_limit=tool_limit,
        ),
    )
    assert result["ok"] is False
    assert len(calls) == expected_calls


# spec: AU-BOUNDARY-R043.7
async def test_host_zero_request_limit_is_not_widened_to_fallback():
    from agent_utilities.graph import RunBudget

    host_ports.register_run_decision_provider(
        lambda req: RunPlan(
            model=FunctionModel(lambda *_: pytest.fail("must not dispatch")),
            budget=RunBudget(request_limit=0),
        )
    )
    assert (await run_rlm("t"))["ok"] is False


# spec: AU-BOUNDARY-R043.7
async def test_repeated_recursive_cancellation_waits_for_async_child_cleanup():
    import asyncio

    entered, cleaning, finish, cleaned = (asyncio.Event() for _ in range(4))

    async def root(messages, info):
        return ModelResponse(parts=[ToolCallPart("sub_query", {"question": "leaf"})])

    async def child(messages, info):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await finish.wait()
            cleaned.set()

    parent = asyncio.create_task(
        run_rlm(
            "root",
            config=RLMConfig(
                model=FunctionModel(root),
                sub_model=FunctionModel(child),
                max_concurrent_requests=1,
            ),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 0.5)
        parent.cancel()
        await asyncio.wait_for(cleaning.wait(), 0.5)
        parent.cancel()
        assert not parent.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(parent, 0.5)
        assert cleaned.is_set()
    finally:
        finish.set()
        if not parent.done():
            parent.cancel()
        await asyncio.gather(parent, return_exceptions=True)


# spec: AU-BOUNDARY-R043.7
@pytest.mark.parametrize("tool_limit", [None, 50])
async def test_recursive_fallback_refuses_oversized_batch_before_child_fanout(
    tool_limit,
):
    children = 0

    async def root(messages, info):
        return ModelResponse(
            parts=[ToolCallPart("sub_query", {"question": "leaf"}) for _ in range(51)]
        )

    async def child(messages, info):
        nonlocal children
        children += 1
        return ModelResponse(parts=[TextPart("child")])

    result = await run_rlm(
        "root",
        config=RLMConfig(
            model=FunctionModel(root),
            sub_model=FunctionModel(child),
            tool_calls_limit=tool_limit,
        ),
    )
    assert result["ok"] is False and children == 0
    assert "tool_calls_limit" in result["error"]
