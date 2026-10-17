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
async def test_recursion_stops_at_max_depth():
    root = scripted(("sub_query", {"question": "q"}))
    out = await run_rlm("t", "abc", config=RLMConfig(model=root, max_depth=1))
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
