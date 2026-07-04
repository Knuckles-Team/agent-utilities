"""CONCEPT:AU-ORCH.execution.predict-rlm-runtime — structured fan-out is reached on the LIVE run_full_rlm path.

Reproduces the article's NarrativeQA "boolean attention-mask" pattern end-to-end:
the root RLM writes REPL code that fans out boolean-schema subagents over chunks
and routes on the typed results. Also asserts (Wire-First) that the `schema=`
contract is actually surfaced in the agent-facing prompt — otherwise the model
would never emit it and the feature would be dead code.
"""

from __future__ import annotations

import pytest

from agent_utilities.rlm.config import RLMConfig
from agent_utilities.rlm.repl import RLMEnvironment

_CAPTURED_SYSTEM_PROMPTS: list[str] = []


class _Res:
    def __init__(self, output: str):
        self.output = output
        self.finish_reason = "stop"

    def all_messages(self):
        return []


class _DepthAwareFakeAgent:
    """Roots fan out boolean subagents; subagents (whose prompt carries the
    boolean output contract) return a typed True."""

    def __init__(self, **kwargs):
        sysp = kwargs.get("system_prompt", "")
        _CAPTURED_SYSTEM_PROMPTS.append(sysp)

    async def run(self, prompt, message_history=None, **kwargs):
        if "boolean" in str(
            prompt
        ):  # the injected output contract reached the sub-agent
            return _Res("```python\nFINAL_VAR('result', True)\n```")
        code = (
            "flags = await run_parallel_sub_calls([\n"
            "    {'prompt': 'chunk0 relevant?', 'context': 'aa', 'schema': {'type': 'boolean'}},\n"
            "    {'prompt': 'chunk1 relevant?', 'context': 'bb', 'schema': {'type': 'boolean'}},\n"
            "])\n"
            "FINAL_VAR('relevant_flags', flags)"
        )
        return _Res(f"```python\n{code}\n```")


@pytest.mark.concept(id="AU-ORCH.execution.predict-rlm-runtime")
@pytest.mark.asyncio
async def test_structured_fanout_on_live_path(monkeypatch):
    _CAPTURED_SYSTEM_PROMPTS.clear()
    monkeypatch.setattr("pydantic_ai.Agent", _DepthAwareFakeAgent)

    env = RLMEnvironment(
        context="full novel text", config=RLMConfig(metadata_only_root=False)
    )
    await env.run_full_rlm("Find chunks about Saltram's living situation.")

    # The parent received typed booleans from the fan-out — not prose to re-parse.
    assert env.vars["relevant_flags"] == [True, True]

    # Wire-First: the schema= contract is advertised in the agent-facing system prompt.
    assert any("schema=" in sp for sp in _CAPTURED_SYSTEM_PROMPTS)


@pytest.mark.concept(id="AU-ORCH.execution.predict-rlm-runtime")
def test_context_metadata_advertises_schema():
    """The metadata-only root access-instructions mention the schema= fan-out option."""
    env = RLMEnvironment(context="x" * 100, config=RLMConfig())
    meta = env._build_context_metadata()
    assert "schema" in meta
