"""CONCEPT:ORCH-1.12 — Structured-output contracts on the RLM subagent fan-out.

Proves the article's "v2 (success)" pattern: subagents called from inside the
REPL return *schema-constrained, typed* values (e.g. a boolean), validated with
retry-don't-restart on a contract violation — not free-form prose.

The fake agent emits REPL code that the REAL `_execute_local` runs (FINAL_VAR is
a live helper), so these exercise the genuine validation path, not a mock of it.
"""

from __future__ import annotations

import pytest

from agent_utilities.rlm.config import RLMConfig
from agent_utilities.rlm.repl import RLMEnvironment


def _cfg() -> RLMConfig:
    # depth>0 sub-RLMs use the small model; metadata-only only affects depth 0.
    return RLMConfig(metadata_only_root=False, async_enabled=True)


class _ScriptedRes:
    def __init__(self, output: str):
        self.output = output
        self.finish_reason = "stop"

    def all_messages(self):
        return []


def _make_fake_agent(script):
    """Return a fake pydantic_ai.Agent class driven by ``script`` — a callable
    ``(call_index, system_prompt, prompt) -> code_str`` returning REPL code."""

    class _FakeAgent:
        def __init__(self, **kwargs):
            self._system_prompt = kwargs.get("system_prompt", "")
            self._calls = 0

        async def run(self, prompt, message_history=None):
            code = script(self._calls, self._system_prompt, prompt)
            self._calls += 1
            return _ScriptedRes(f"```python\n{code}\n```")

    return _FakeAgent


@pytest.mark.concept(id="ORCH-1.12")
@pytest.mark.asyncio
async def test_rlm_query_with_bool_schema_returns_typed_value(monkeypatch):
    """rlm_query(prompt, ctx, schema=bool) returns a real Python bool, not a string."""
    fake = _make_fake_agent(lambda i, sysp, p: "FINAL_VAR('result', True)")
    monkeypatch.setattr("pydantic_ai.Agent", fake)

    parent = RLMEnvironment(context="root", config=_cfg())
    result = await parent.rlm_query("is it relevant?", sub_context="chunk", schema=bool)

    assert result is True
    assert isinstance(result, bool)


@pytest.mark.concept(id="ORCH-1.12")
@pytest.mark.asyncio
async def test_subagent_schema_violation_retries_without_restart(monkeypatch):
    """A sub-RLM whose FINAL violates the schema gets schema+error feedback and retries."""

    def script(call_index, system_prompt, prompt):
        # First attempt returns an invalid (non-boolean) value; the in-loop
        # validation rejects it and feeds back the contract → retry succeeds.
        if call_index == 0:
            return "FINAL_VAR('result', 'definitely not a boolean')"
        return "FINAL_VAR('result', True)"

    fake = _make_fake_agent(script)
    monkeypatch.setattr("pydantic_ai.Agent", fake)

    parent = RLMEnvironment(context="root", config=_cfg())
    result = await parent.rlm_query(
        "relevant?", sub_context="chunk", schema={"type": "boolean"}
    )

    # The retry path produced a valid, coerced boolean (REPL state preserved).
    assert result is True


@pytest.mark.concept(id="ORCH-1.12")
@pytest.mark.asyncio
async def test_run_parallel_sub_calls_honours_per_call_schema(monkeypatch):
    """Each call dict may carry its own `schema`; results come back as typed values."""
    fake = _make_fake_agent(lambda i, sysp, p: "FINAL_VAR('result', True)")
    monkeypatch.setattr("pydantic_ai.Agent", fake)

    parent = RLMEnvironment(context="root", config=_cfg())
    results = await parent.run_parallel_sub_calls(
        [
            {"prompt": "chunk 0 relevant?", "context": "a", "schema": bool},
            {"prompt": "chunk 1 relevant?", "context": "b", "schema": bool},
        ]
    )

    assert results == [True, True]
    assert all(isinstance(r, bool) for r in results)
