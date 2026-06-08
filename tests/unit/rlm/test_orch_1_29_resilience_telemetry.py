"""CONCEPT:ORCH-1.29 — RLM Resilience + Structured Telemetry."""

from __future__ import annotations

import asyncio

import pytest

from agent_utilities.rlm.telemetry import (
    RunTrace,
    SandboxFatalError,
    classify_failure,
    dominant_failure,
    with_tool_timeout,
)


@pytest.mark.concept(id="ORCH-1.29")
def test_classify_failure_taxonomy():
    assert classify_failure(SandboxFatalError("x")) == "sandbox_fatal"
    assert classify_failure(TimeoutError()) == "host_tool_timeout"
    assert (
        classify_failure(SyntaxError("unterminated string"))
        == "model_generated_bad_code"
    )
    assert classify_failure("evaluator reject: bad") == "evaluator_reject"
    assert classify_failure("something odd") == "unknown"


@pytest.mark.concept(id="ORCH-1.29")
def test_dominant_failure_precedence():
    assert (
        dominant_failure(["unknown", "host_tool_timeout", "sandbox_fatal"])
        == "sandbox_fatal"
    )
    assert (
        dominant_failure(["model_generated_bad_code", "unknown"])
        == "model_generated_bad_code"
    )
    assert dominant_failure([]) == "unknown"


@pytest.mark.concept(id="ORCH-1.29")
def test_runtrace_accumulates_steps_and_usage():
    t = RunTrace()
    t.add_step(code="x=1", output="", finish_reason="stop")
    t.add_step(code="bad(", failure_class="model_generated_bad_code")
    t.usage.prompt_tokens = 100
    t.usage.completion_tokens = 50
    t.usage.sub_lm_tokens = 25
    assert [s.index for s in t.steps] == [0, 1]
    assert t.usage.total == 175
    assert t.failure_summary() == "model_generated_bad_code"


@pytest.mark.concept(id="ORCH-1.29")
@pytest.mark.asyncio
async def test_with_tool_timeout_recoverable():
    async def slow():
        await asyncio.sleep(1)
        return "done"

    ok, val = await with_tool_timeout(slow(), seconds=0.01)
    assert (
        ok is False and "recoverable" in val.lower()
    )  # timeout is recoverable, not fatal

    async def fast():
        return 42

    ok2, val2 = await with_tool_timeout(fast(), seconds=1)
    assert ok2 is True and val2 == 42


@pytest.mark.concept(id="ORCH-1.29")
@pytest.mark.asyncio
async def test_with_tool_timeout_propagates_fatal():
    async def fatal():
        raise SandboxFatalError("sandbox died")

    with pytest.raises(SandboxFatalError):
        await with_tool_timeout(fatal(), seconds=1)
