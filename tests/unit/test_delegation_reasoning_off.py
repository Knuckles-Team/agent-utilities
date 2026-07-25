"""Reasoning is OFF by default and forced off for delegated tool loops.

CONCEPT:AU-ORCH.execution.delegation-reasoning-off — reasoning rides on core
``ModelSettings.thinking`` (PA-R0.2, retiring the ``extra_body["reasoning_effort"]``
hack). ``create_model`` defaults ``reasoning_effort="none"`` -> ``thinking=False``, which
pydantic-ai translates to the top-level ``reasoning_effort='none'`` request parameter this
vLLM build honors. Because ``thinking`` is a top-level ``ModelSettings`` key it survives
the agent-over-model SHALLOW union on its own (no ``extra_body`` clobber). On the reasoning
chat model, thinking left ON is ~18x per-turn latency that stacks across a delegation's
model->tool->model turns until it overruns the wall-clock. These tests pin:

1. ``clamp_thinking_effort`` maps the reasoning vocabulary onto ``thinking`` levels;
2. ``_resolve_agent_extra_body`` carries the model's vLLM-only knobs up WITHOUT reasoning;
3. ``create_agent(reasoning_effort=...)`` lands ``thinking`` on the agent settings (off by
   default, an explicit level opts in); and
4. delegation leaves reasoning OFF by default (inherits) but is an opt-in CAPABILITY —
   a run turns it ON per-execution via ``reasoning_effort`` (run_agent/execute_agent).
"""

from __future__ import annotations

import asyncio

from agent_utilities.agent.factory import _resolve_agent_extra_body, create_agent
from agent_utilities.core.model_factory import clamp_thinking_effort


class _FakeModel:
    def __init__(self, extra_body):
        # mirrors OpenAIChatModel.settings (a ModelSettings TypedDict == plain dict),
        # or None for a settings-less model (e.g. the validation-mode TestModel).
        self.settings = {"extra_body": extra_body} if extra_body is not None else None


def test_clamp_maps_reasoning_vocabulary_to_thinking():
    """'none'/off/unknown -> False (thinking off); levels pass through; None -> None."""
    assert clamp_thinking_effort("none") is False
    assert clamp_thinking_effort("off") is False
    assert (
        clamp_thinking_effort("bogus") is False
    )  # unknown clamps OFF, never silently on
    assert clamp_thinking_effort("high") == "high"
    assert clamp_thinking_effort("minimal") == "minimal"
    assert clamp_thinking_effort(None) is None


def test_extra_body_carries_vllm_knobs_not_reasoning():
    """The agent extra_body carries the model's vLLM-only knobs (priority) up; reasoning
    is NOT here anymore (it lives on the top-level thinking key)."""
    eb = _resolve_agent_extra_body(_FakeModel({"priority": 5}))
    assert eb.get("priority") == 5
    assert "reasoning_effort" not in eb


def test_settings_less_model_yields_empty_extra_body():
    """A settings-less model (TestModel) contributes no extra_body."""
    assert _resolve_agent_extra_body(_FakeModel(None)) == {}


def test_create_agent_threads_reasoning_effort_onto_thinking():
    """End-to-end: create_agent(reasoning_effort='none') sets thinking=False on the agent
    settings (NOT reasoning_effort in extra_body); a level opts in; omitting it leaves the
    model's own thinking to survive the merge (no key set)."""
    common = dict(
        provider="openai",
        model_id="qwen/qwen3.6-27b",
        base_url="http://vllm.arpa/v1",
        api_key=None,
        mcp_toolsets=[],
        enable_skills=False,
        enable_universal_tools=False,
        system_prompt="x",
    )

    def _settings(agent):
        ms = getattr(agent, "model_settings", None) or getattr(
            agent, "_model_settings", None
        )
        return dict(ms)

    agent_off, _ = create_agent(name="t-off", reasoning_effort="none", **common)
    s_off = _settings(agent_off)
    assert s_off.get("thinking") is False
    assert "reasoning_effort" not in (s_off.get("extra_body") or {})

    agent_on, _ = create_agent(name="t-on", reasoning_effort="high", **common)
    assert _settings(agent_on).get("thinking") == "high"

    agent_default, _ = create_agent(name="t-default", **common)
    # No arg -> no thinking key, so the model's own thinking default survives the merge.
    assert "thinking" not in _settings(agent_default)


def _run_single_server(monkeypatch, config: dict) -> dict:
    """Drive _execute_single_server with a captured, faked create_agent; return kwargs."""
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
    base = {
        "mcp_toolsets": [object()],  # non-empty so it doesn't fail-loud on "no toolset"
        "provider": "openai",
        "agent_model": "qwen/qwen3.6-27b",
        "base_url": "http://vllm.arpa/v1",
        "api_key": None,
    }
    base.update(config)
    asyncio.run(
        agent_runner._execute_single_server(
            base, "list things", 4, {"type": "server"}, "scholarx-mcp"
        )
    )
    return captured


def test_single_server_delegation_inherits_off_by_default(monkeypatch):
    """No opt-in => reasoning_effort=None => create_agent inherits the model's OFF default."""
    captured = _run_single_server(monkeypatch, {})
    assert captured.get("reasoning_effort") is None, captured


def test_single_server_delegation_honors_reasoning_opt_in(monkeypatch):
    """A run that needs deliberation turns reasoning ON via config (like RLM)."""
    captured = _run_single_server(monkeypatch, {"reasoning_effort": "high"})
    assert captured.get("reasoning_effort") == "high", captured


def test_run_agent_threads_reasoning_effort_onto_config():
    """run_agent(reasoning_effort=...) is the per-execution capability seam onto config."""
    import inspect

    from agent_utilities.orchestration import agent_runner, manager

    # both the entry point and the manager expose the opt-in param
    assert "reasoning_effort" in inspect.signature(agent_runner.run_agent).parameters
    assert (
        "reasoning_effort"
        in inspect.signature(manager.Orchestrator.execute_agent).parameters
    )
