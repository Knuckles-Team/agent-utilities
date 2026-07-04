"""Unit + live-path tests for the dynamic KV-cache-layering policy.

CONCEPT:AU-ORCH.optimization.kvcache-worthiness-policy — verifies (1) the :class:`KVCacheLayeringPolicy`
cache-worthiness verdict across the one-off / long-prefix / multi-turn / large-
context signals, (2) that :func:`fold_kv_hint` folds the per-request
``kv_transfer_params={"lmcache.skip_save": ...}`` control into ``extra_body``
without clobbering other knobs, and (3) that the LIVE run seam
(:func:`attach_profile_resolver`) actually sets the hint on every chat call.
"""

from __future__ import annotations

from agent_utilities.agent.sampling_profile import attach_profile_resolver
from agent_utilities.kvcache.policy import (
    KVCacheDecision,
    KVCacheLayeringPolicy,
    decide,
    fold_kv_hint,
)


class _Part:
    def __init__(self, content: str) -> None:
        self.content = content


class _Msg:
    def __init__(self, content: str) -> None:
        self.parts = [_Part(content)]


# ── the scorer ───────────────────────────────────────────────────────────────


def test_one_off_short_prompt_skips_save():
    d = decide(system_prompt="You are a helpful bot.", user_prompt="hi")
    assert isinstance(d, KVCacheDecision)
    assert d.cache_worthy is False
    assert d.skip_save is True
    assert "one_off_short_prompt" in d.reasons
    assert d.kv_transfer_params == {"lmcache.skip_save": True}


def test_long_shared_prefix_is_cache_worthy():
    d = decide(system_prompt="x" * 8000, user_prompt="do a small thing")
    assert d.cache_worthy is True
    assert d.skip_save is False
    assert any("long_shared_prefix" in r for r in d.reasons)


def test_multi_turn_conversation_is_cache_worthy():
    hist = [_Msg("turn one text"), _Msg("turn two text")]
    d = decide(system_prompt="You are a bot.", user_prompt="next", message_history=hist)
    assert d.cache_worthy is True
    assert any("multi_turn" in r for r in d.reasons)


def test_large_rag_context_is_cache_worthy():
    d = decide(user_prompt="answer", rag_context="r" * 9000)
    assert d.cache_worthy is True
    assert any("rag" in r or "large_fixed_context" in r for r in d.reasons)


def test_thresholds_are_overridable_for_testing():
    pol = KVCacheLayeringPolicy(min_prefix_tokens=1, min_context_tokens=1)
    d = pol.decide(system_prompt="short", user_prompt="hi")
    assert d.cache_worthy is True  # tiny threshold flips the verdict


def test_disabled_policy_is_inert():
    pol = KVCacheLayeringPolicy(enabled=False)
    d = pol.decide(system_prompt="x" * 8000, user_prompt="hi")
    assert d.enabled is False
    assert d.kv_transfer_params == {}  # attach nothing → opportunistic default


# ── fold_kv_hint (settings merge) ────────────────────────────────────────────


def test_fold_sets_skip_save_true_for_one_off():
    ms = fold_kv_hint({"temperature": 0.2}, system_prompt="bot", user_prompt="hi")
    d = dict(ms)
    assert d["temperature"] == 0.2  # untouched
    assert d["extra_body"]["kv_transfer_params"]["lmcache.skip_save"] is True


def test_fold_preserves_existing_extra_body_knobs():
    ms = fold_kv_hint(
        {"extra_body": {"reasoning_effort": "none", "priority": 5}},
        system_prompt="x" * 8000,
        user_prompt="go",
    )
    eb = dict(ms)["extra_body"]
    assert eb["reasoning_effort"] == "none"  # sibling knob survives
    assert eb["priority"] == 5
    assert eb["kv_transfer_params"]["lmcache.skip_save"] is False  # worthy → store


def test_fold_is_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("KV_CACHE_LAYERING", "false")
    ms = fold_kv_hint({"temperature": 0.1}, system_prompt="x" * 8000, user_prompt="go")
    assert "kv_transfer_params" not in (dict(ms).get("extra_body") or {})


# ── LIVE run seam: the hint is set on every chat call ────────────────────────


def test_live_seam_sets_kv_hint_on_every_call():
    """attach_profile_resolver must fold the KV hint into per-call model_settings."""
    captured: dict = {}

    class FakeAgent:
        def run(self, user_prompt=None, **kwargs):
            captured["ms"] = kwargs.get("model_settings")
            return "ok"

    agent = FakeAgent()
    # A long, stable system prompt makes this a reuse-heavy agent (store).
    attach_profile_resolver(agent, {"temperature": 0.7}, system_prompt="P" * 8000)

    agent.run("do a thing")
    eb = dict(captured["ms"]).get("extra_body", {})
    assert "kv_transfer_params" in eb
    assert eb["kv_transfer_params"]["lmcache.skip_save"] is False  # reuse-heavy → store


def test_live_seam_skips_save_for_short_one_off_agent():
    captured: dict = {}

    class FakeAgent:
        def run(self, user_prompt=None, **kwargs):
            captured["ms"] = kwargs.get("model_settings")

    agent = FakeAgent()
    attach_profile_resolver(agent, {"temperature": 0.7}, system_prompt="tiny prompt")
    agent.run("hi")
    eb = dict(captured["ms"]).get("extra_body", {})
    assert eb["kv_transfer_params"]["lmcache.skip_save"] is True  # one-off → skip


def test_live_seam_folds_hint_even_under_explicit_settings():
    captured: dict = {}

    class FakeAgent:
        def run(self, user_prompt=None, **kwargs):
            captured["ms"] = kwargs.get("model_settings")

    agent = FakeAgent()
    attach_profile_resolver(agent, {"temperature": 0.7}, system_prompt="P" * 8000)
    agent.run("go", model_settings={"temperature": 0.42})
    ms = dict(captured["ms"])
    assert ms["temperature"] == 0.42  # caller's sampling still wins
    assert "kv_transfer_params" in ms.get("extra_body", {})  # KV hint still folded
