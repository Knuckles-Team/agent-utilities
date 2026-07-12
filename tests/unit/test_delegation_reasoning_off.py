"""Reasoning is OFF by default (clobber fixed) and forced off for delegated tool loops.

CONCEPT:AU-ORCH.execution.delegation-reasoning-off — ``create_model`` defaults
``reasoning_effort="none"`` (thinking OFF), but pydantic-ai merges agent-over-model
``ModelSettings`` with a SHALLOW dict union, so ``create_agent``'s agent-level
``extra_body`` REPLACED the model-level one — silently discarding that default (and any
per-model override) for every ``create_agent``-built agent. On the reasoning chat model
that is ~18x per-turn latency that stacks across a delegation's model->tool->model turns
until it overruns the wall-clock and is mis-attributed to a blocked tool. These tests pin:

1. the merge (not clobber): the model's reasoning settings reach the winning agent layer;
2. per-model reasoning opt-in (e.g. "high") is preserved;
3. an explicit ``reasoning_effort`` arg wins; and
4. the single-server / focused-tools executor forces ``reasoning_effort="none"``.
"""

from __future__ import annotations

import asyncio

from agent_utilities.agent.factory import _resolve_agent_extra_body, create_agent


class _FakeModel:
    def __init__(self, extra_body):
        # mirrors OpenAIChatModel.settings (a ModelSettings TypedDict == plain dict),
        # or None for a settings-less model (e.g. the validation-mode TestModel).
        self.settings = {"extra_body": extra_body} if extra_body is not None else None


def test_merge_inherits_model_reasoning_default():
    """No arg + model default 'none' -> reaches the agent layer (thinking OFF fleet-wide)."""
    eb = _resolve_agent_extra_body(_FakeModel({"reasoning_effort": "none"}), None)
    assert eb.get("reasoning_effort") == "none"


def test_merge_preserves_per_model_override():
    """A per-model reasoning opt-in (e.g. 'high') is NOT clobbered to off."""
    eb = _resolve_agent_extra_body(_FakeModel({"reasoning_effort": "high"}), None)
    assert eb.get("reasoning_effort") == "high"


def test_explicit_reasoning_effort_wins():
    """An explicit arg overrides whatever the model carried."""
    eb = _resolve_agent_extra_body(_FakeModel({"reasoning_effort": "high"}), "none")
    assert eb.get("reasoning_effort") == "none"


def test_settings_less_model_still_takes_explicit():
    """A settings-less model (TestModel) has nothing to inherit but honors an explicit arg."""
    assert _resolve_agent_extra_body(_FakeModel(None), None) == {}
    assert _resolve_agent_extra_body(_FakeModel(None), "none") == {
        "reasoning_effort": "none"
    }


def test_create_agent_threads_explicit_reasoning_effort():
    """End-to-end: create_agent(reasoning_effort='none') lands it on agent extra_body."""
    agent, _ = create_agent(
        provider="openai",
        model_id="qwen/qwen3.6-27b",
        base_url="http://vllm.arpa/v1",
        api_key=None,
        mcp_toolsets=[],
        enable_skills=False,
        enable_universal_tools=False,
        name="t-off",
        system_prompt="x",
        reasoning_effort="none",
    )
    ms = getattr(agent, "model_settings", None) or getattr(
        agent, "_model_settings", None
    )
    assert dict(ms).get("extra_body", {}).get("reasoning_effort") == "none"


def test_single_server_delegation_disables_reasoning(monkeypatch):
    """LIVE PATH: ``_execute_single_server`` passes ``reasoning_effort='none'`` to create_agent."""
    from agent_utilities.agent import factory as factory_mod
    from agent_utilities.orchestration import agent_runner

    captured: dict = {}

    class _FakeResult:
        output = "done"

    class _FakeAgent:
        async def run(self, *a, **k):
            return _FakeResult()

    def _fake_create_agent(*args, **kwargs):
        captured.update(kwargs)
        return _FakeAgent(), []

    # _execute_single_server does a local ``from agent_utilities.agent.factory import
    # create_agent`` inside the function, so patch it at the factory module.
    monkeypatch.setattr(factory_mod, "create_agent", _fake_create_agent)
    monkeypatch.setattr(agent_runner, "_extract_tool_calls", lambda _r: [])

    config = {
        "mcp_toolsets": [object()],  # non-empty so it doesn't fail-loud on "no toolset"
        "provider": "openai",
        "agent_model": "qwen/qwen3.6-27b",
        "base_url": "http://vllm.arpa/v1",
        "api_key": None,
    }
    out = asyncio.run(
        agent_runner._execute_single_server(
            config, "list things", 4, {"type": "server"}, "scholarx-mcp"
        )
    )
    assert captured.get("reasoning_effort") == "none", captured
    assert out["results"]["output"] == "done"
