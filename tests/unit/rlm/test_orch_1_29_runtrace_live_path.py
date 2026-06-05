"""CONCEPT:ORCH-1.29 — RunTrace is populated by the LIVE run_full_rlm loop (not just available).

Drives the existing `RLMEnvironment.run_full_rlm` with a fake agent + execute, and asserts a
structured RunTrace is built as a side effect — proving the telemetry is on the live path.
"""

from __future__ import annotations

import pytest

from agent_utilities.rlm.config import RLMConfig
from agent_utilities.rlm.repl import RLMEnvironment
from agent_utilities.rlm.telemetry import RunTrace


class _FakeRes:
    output = "```python\nx = 1\n```"  # a code block, never FINAL_VAR → loop runs to max_turns
    finish_reason = "stop"

    def all_messages(self):
        return []


class _FakeAgent:
    def __init__(self, **kwargs):
        pass

    async def run(self, prompt, message_history=None):
        return _FakeRes()


@pytest.mark.concept(id="ORCH-1.29")
@pytest.mark.asyncio
async def test_run_full_rlm_populates_runtrace(monkeypatch):
    monkeypatch.setattr("pydantic_ai.Agent", _FakeAgent)
    env = RLMEnvironment(context="some data", config=RLMConfig(metadata_only_root=False))

    async def _fake_execute(code):
        return {}, "stdout output here"

    env.execute = _fake_execute  # avoid real sandbox

    result = await env.run_full_rlm("analyze the data")

    # The live loop attached a structured RunTrace and recorded a step per iteration.
    assert isinstance(env.last_run_trace, RunTrace)
    assert len(env.last_run_trace.steps) >= 1
    step = env.last_run_trace.steps[0]
    assert "x = 1" in step.code and step.output.startswith("stdout output")
    assert env.last_run_trace.final_status == "partial"  # never hit FINAL_VAR
    assert "Max turns reached" in result


@pytest.mark.concept(id="ORCH-1.29")
@pytest.mark.asyncio
async def test_run_full_rlm_records_failure_class(monkeypatch):
    monkeypatch.setattr("pydantic_ai.Agent", _FakeAgent)
    env = RLMEnvironment(context="d", config=RLMConfig(metadata_only_root=False))

    async def _boom(code):
        raise SyntaxError("unterminated string literal")

    env.execute = _boom

    with pytest.raises(SyntaxError):
        await env.run_full_rlm("task")
    # The failing iteration was classified into the RunTrace before re-raising.
    assert env.last_run_trace.final_status == "failure"
    assert env.last_run_trace.steps[0].failure_class == "model_generated_bad_code"
