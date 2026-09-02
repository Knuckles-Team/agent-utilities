"""CONCEPT:AU-ORCH.execution.drop-rlm-completion-client — drop-in RLM client + model-family-aware prompt + AHE-3.32 usage capture."""

from __future__ import annotations

import pytest

from agent_utilities.rlm import RLM, RLMConfig, RLMResponse
from agent_utilities.rlm.prompts import build_system_prompt, infer_profile
from agent_utilities.rlm.telemetry import LMUsage

# ── model-family-aware prompt (ORCH-1.54) ──


def test_auto_profile_does_not_infer_behavior_from_identity():
    assert infer_profile("vendor-a/reasoner-large") == "concise"
    assert infer_profile("vendor-b/chat-small") == "concise"


def test_build_system_prompt_addenda():
    base = build_system_prompt("auto", "vendor-a/chat-small")
    concise = build_system_prompt("concise", "vendor-b/chat-medium")
    code_first = build_system_prompt("code-first", "vendor-c/reasoner-large")
    # All share the core helper contract.
    for p in (base, concise, code_first):
        assert "rlm_query" in p and "FINAL_VAR" in p
    # Family addenda differ and target their failure modes.
    assert "terse" in concise.lower()
    assert "narrate" in code_first.lower()
    assert concise == base and code_first != base


def test_prompt_family_pin_overrides_inference():
    pinned = build_system_prompt("concise", "vendor-a/chat-small")
    assert "terse" in pinned.lower()


def test_config_has_prompt_family_default_auto():
    assert RLMConfig().prompt_family == "auto"


# ── drop-in client (ORCH-1.54) ──


async def test_rlm_acompletion_maps_result(monkeypatch):
    async def fake_run_rlm(task, input_text="", *, config=None, **kw):
        # context-as-external-variable: the long prompt arrives as input_text
        assert input_text == "a very long document"
        return {"ok": True, "result": "the answer", "usage": {"total": 42}}

    monkeypatch.setattr("agent_utilities.rlm.client.run_rlm", fake_run_rlm)
    rlm = RLM(backend="vendor-a", backend_kwargs={"model_name": "chat-small"})
    resp = await rlm.acompletion("a very long document")
    assert isinstance(resp, RLMResponse)
    assert resp.response == "the answer" and resp.text == "the answer"
    assert resp.ok and resp.usage == {"total": 42}


async def test_rlm_acompletion_question_over_context(monkeypatch):
    seen = {}

    async def fake_run_rlm(task, input_text="", *, config=None, **kw):
        seen["task"] = task
        seen["input"] = input_text
        return {"ok": True, "result": "ok", "usage": {}}

    monkeypatch.setattr("agent_utilities.rlm.client.run_rlm", fake_run_rlm)
    rlm = RLM(backend="vendor-a", backend_kwargs={"model_name": "chat-small"})
    await rlm.acompletion("what is X?", context="big ctx")
    assert seen["task"] == "what is X?" and seen["input"] == "big ctx"


def test_rlm_backend_sets_root_model():
    rlm = RLM(backend="vendor-a", backend_kwargs={"model_name": "reasoner-large"})
    assert rlm.config.sub_llm_model_large == "vendor-a:reasoner-large"


async def test_sync_completion_inside_loop_raises():
    rlm = RLM(backend="vendor-a", backend_kwargs={"model_name": "chat-small"})
    with pytest.raises(RuntimeError, match="event loop"):
        rlm.completion("x")  # we are inside the asyncio test loop


# ── usage capture wiring (AHE-3.32) ──


def test_accumulate_root_usage():
    from agent_utilities.rlm.repl import _accumulate_root_usage

    class _U:
        request_tokens = 100
        response_tokens = 30

    class _Res:
        def usage(self):
            return _U()

    usage = LMUsage()
    _accumulate_root_usage(usage, _Res())
    assert usage.prompt_tokens == 100 and usage.completion_tokens == 30
    # second call accumulates
    _accumulate_root_usage(usage, _Res())
    assert usage.prompt_tokens == 200


def test_accumulate_root_usage_missing_is_noop():
    from agent_utilities.rlm.repl import _accumulate_root_usage

    usage = LMUsage()
    _accumulate_root_usage(usage, object())  # no usage() method
    assert usage.total == 0


def test_absorb_sub_usage_folds_total():
    from agent_utilities.rlm.repl import RLMEnvironment
    from agent_utilities.rlm.telemetry import RunTrace

    parent = RLMEnvironment(context="x")
    parent.last_run_trace = RunTrace()
    child = RLMEnvironment(context="y", depth=1)
    child.last_run_trace = RunTrace()
    child.last_run_trace.usage = LMUsage(prompt_tokens=10, completion_tokens=5)
    parent._absorb_sub_usage(child)
    assert parent.last_run_trace.usage.sub_lm_tokens == 15
