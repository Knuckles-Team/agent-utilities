"""Reasoning is OFF by default and forced off for delegated tool loops.

CONCEPT:AU-ORCH.execution.delegation-reasoning-off — regression fix. Reasoning is
computed on core ``ModelSettings.thinking`` (``clamp_thinking_effort``) AND, since this
suite was first written, ALSO as a raw wire-level ``extra_body`` directive
(``reasoning_wire_directives`` — vLLM's ``reasoning_effort`` + ``chat_template_kwargs.
enable_thinking``). Both are required: pydantic-ai's ``Model.prepare_request()`` only
forwards ``thinking`` into the actual request when the model's PROFILE is recognized as
reasoning-capable (``openai_model_profile()`` recognizes only OpenAI's own o-series/
gpt-5(.1+) naming); a local/custom reasoning model served through the generic ``openai``
provider — e.g. ``qwen/qwen3.6-27b`` behind vLLM — gets ``supports_thinking=False`` from
that heuristic, so ``thinking`` silently never reaches the wire regardless of its value,
and the model's OWN default (thinking ON) always wins. This was a LIVE regression: this
very test file used to assert ``"reasoning_effort" not in extra_body`` — i.e. it pinned
the exact behavior that made a "reasoning off by default" call measure ~22s instead of
~0.3s, because the disable directive was computed correctly and then never sent. These
tests now pin the corrected contract:

1. ``clamp_thinking_effort`` maps the reasoning vocabulary onto ``thinking`` levels;
2. ``reasoning_wire_directives`` maps the same vocabulary onto raw wire fields;
3. ``merge_extra_body`` folds a reasoning directive into ``extra_body`` WITHOUT dropping
   pre-existing keys (incl. a nested ``chat_template_kwargs``) — the original clobber bug
   this mechanism must never reintroduce;
4. ``_resolve_agent_extra_body`` carries the model's vLLM-only knobs AND its reasoning
   directive up to the agent level;
5. ``create_agent(reasoning_effort=...)`` lands BOTH ``thinking`` and the raw directive on
   the agent settings — off by default, an explicit level opts in and correctly
   RE-ENABLES the directive, and OMITTING it entirely still carries the model's own
   off-by-default directive through (the regression-pinning case); and
6. delegation leaves reasoning OFF by default (inherits) but is an opt-in CAPABILITY —
   a run turns it ON per-execution via ``reasoning_effort`` (run_agent/execute_agent).
"""

from __future__ import annotations

import asyncio

from agent_utilities.agent.factory import _resolve_agent_extra_body, create_agent
from agent_utilities.core.model_factory import (
    _openai_reasoning_settings,
    clamp_thinking_effort,
    merge_extra_body,
    reasoning_wire_directives,
)


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


def test_reasoning_wire_directives_disable():
    """effort='none' -> the raw disable directive (both vLLM spellings), so a call reaches
    the wire even when pydantic-ai's model-profile inference doesn't recognize this model
    as reasoning-capable (the regression: ``thinking`` alone silently no-ops there)."""
    directives = reasoning_wire_directives("none")
    assert directives == {
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_reasoning_wire_directives_enable_level():
    """An explicit effort level re-enables thinking on the wire, not just in ``thinking``."""
    directives = reasoning_wire_directives("high")
    assert directives == {
        "reasoning_effort": "high",
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_reasoning_wire_directives_empty_when_none_effort():
    """effort=None ('inherit' / leave the model's native default untouched) sends NO
    directive either way — distinct from 'none', which explicitly disables."""
    assert reasoning_wire_directives(None) == {}


def test_merge_extra_body_does_not_clobber_sibling_keys():
    """Folding the reasoning directive in must not drop pre-existing extra_body knobs
    (e.g. the vLLM scheduler ``priority`` hint or sampling-profile knobs) — the original
    agent-over-model shallow-union clobber this mechanism must never reintroduce."""
    base = {"priority": 5, "top_k": 20}
    merged = merge_extra_body(
        base,
        {
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert merged["priority"] == 5
    assert merged["top_k"] == 20
    assert merged["reasoning_effort"] == "none"
    assert merged["chat_template_kwargs"] == {"enable_thinking": False}


def test_merge_extra_body_deep_merges_chat_template_kwargs():
    """A nested ``chat_template_kwargs`` key already present in the base (e.g. an RLM- or
    operator-set knob) survives an overlay that ALSO sets ``chat_template_kwargs`` — a
    naive shallow union (``{**base, **overlay}``) would silently drop it."""
    base = {"chat_template_kwargs": {"some_other_knob": True}}
    overlay = {"chat_template_kwargs": {"enable_thinking": False}}
    merged = merge_extra_body(base, overlay)
    assert merged["chat_template_kwargs"] == {
        "some_other_knob": True,
        "enable_thinking": False,
    }


def test_default_reasoning_settings_carry_thinking_disable_directive_on_the_wire():
    """THE regression-pinning test: create_model's model-level default
    (``reasoning_effort="none"``) must reach the wire via ``extra_body``, not rely on
    ``ModelSettings.thinking`` alone — pydantic-ai silently drops ``thinking`` for a model
    whose profile isn't recognized as reasoning-capable (a custom/local reasoning model
    like ``qwen/qwen3.6-27b`` via a generic ``openai`` provider is exactly such a model),
    which is what made a "reasoning off by default" call measure ~22s instead of ~0.3s."""
    settings = _openai_reasoning_settings("none")
    assert settings["thinking"] is False
    extra = settings["extra_body"]
    assert extra["reasoning_effort"] == "none"
    assert extra["chat_template_kwargs"]["enable_thinking"] is False


def test_reasoning_opt_in_re_enables_wire_directive():
    """A per-execution reasoning_effort='high' opt-in correctly RE-ENABLES thinking on the
    wire (proves the opt-in path isn't broken by the disable-by-default fix)."""
    settings = _openai_reasoning_settings("high")
    assert settings["thinking"] == "high"
    extra = settings["extra_body"]
    assert extra["reasoning_effort"] == "high"
    assert extra["chat_template_kwargs"]["enable_thinking"] is True


def test_extra_body_carries_vllm_knobs_and_reasoning_directive():
    """The agent extra_body carries the model's vLLM-only knobs (priority) AND its
    reasoning directive up — both must survive, since the agent-over-model settings merge
    REPLACES (not deep-merges) extra_body wholesale if left uncarried."""
    eb = _resolve_agent_extra_body(_FakeModel({"priority": 5}))
    assert eb.get("priority") == 5
    assert "reasoning_effort" not in eb  # nothing to carry: this fake model set none

    eb_with_reasoning = _resolve_agent_extra_body(
        _FakeModel(
            {
                "priority": 5,
                "reasoning_effort": "none",
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
    )
    assert eb_with_reasoning.get("priority") == 5
    assert eb_with_reasoning.get("reasoning_effort") == "none"
    assert eb_with_reasoning.get("chat_template_kwargs") == {"enable_thinking": False}


def test_settings_less_model_yields_empty_extra_body():
    """A settings-less model (TestModel) contributes no extra_body."""
    assert _resolve_agent_extra_body(_FakeModel(None)) == {}


def test_create_agent_threads_reasoning_effort_onto_thinking(monkeypatch):
    """End-to-end: create_agent(reasoning_effort='none') sets BOTH thinking=False AND the
    raw extra_body directive (reasoning_effort='none' + chat_template_kwargs.
    enable_thinking=False) — ``thinking`` alone is not sufficient (see module docstring).
    A level opts in and correctly RE-ENABLES the directive. Omitting reasoning_effort
    entirely leaves no agent-level ``thinking`` key (the model's own survives the merge)
    but MUST still carry the model's own off-by-default wire directive through — the
    regression-pinning case: a routine call with no reasoning_effort anywhere in the call
    chain must still disable thinking on the wire, not silently send nothing.

    ``AGENT_UTILITIES_TESTING`` is forced off for this test: ``create_model`` short-circuits
    to a settings-less ``TestModel`` under it (hermetic-by-default for the rest of the
    suite), which would make the "default" assertions below vacuously pass without
    actually exercising ``_openai_reasoning_settings`` — ``setting()`` is a LIVE read, and
    no network call happens at model *construction* time, so this stays a fast, offline
    unit test while covering the real production model-construction path.
    """
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")
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
    eb_off = s_off.get("extra_body") or {}
    assert eb_off.get("reasoning_effort") == "none"
    assert eb_off.get("chat_template_kwargs", {}).get("enable_thinking") is False

    agent_on, _ = create_agent(name="t-on", reasoning_effort="high", **common)
    s_on = _settings(agent_on)
    assert s_on.get("thinking") == "high"
    eb_on = s_on.get("extra_body") or {}
    assert eb_on.get("reasoning_effort") == "high"
    assert eb_on.get("chat_template_kwargs", {}).get("enable_thinking") is True

    agent_default, _ = create_agent(name="t-default", **common)
    s_default = _settings(agent_default)
    # No explicit override -> no agent-level `thinking` key (the model's own survives the
    # merge)...
    assert "thinking" not in s_default
    # ...but the WIRE-level directive still carries the model's OWN default
    # (create_model's reasoning_effort="none") up through extra_body. THIS is the fix: a
    # routine call with NO explicit reasoning_effort anywhere still disables thinking on
    # the wire instead of silently sending nothing.
    eb_default = s_default.get("extra_body") or {}
    assert eb_default.get("reasoning_effort") == "none"
    assert eb_default.get("chat_template_kwargs", {}).get("enable_thinking") is False


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
