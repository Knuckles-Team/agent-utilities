"""Unit tests for :mod:`agent_utilities.caching.prompt_cache`.

CONCEPT:AU-ORCH.optimization.provider-prompt-cache. Proves:

* :class:`PromptCacheKey` hashes every scoping component — a change to any one mints a
  different fingerprint (the KVCheckpointKey-style "any change invalidates" contract).
* :func:`fold_prompt_cache_hint` is default-on (Anthropic cache directives +
  ``openai_prompt_cache_key`` folded with no caller action) and never clobbers an
  explicit caller-supplied value.
* The one opt-out (``AU_PROMPT_CACHE=false``) leaves settings completely untouched.
* :func:`prompt_cache_create_kwargs` mirrors the same contract for the raw-openai-client seam.
* :func:`usage_cache_tokens`/:func:`record_prompt_cache_usage` extract cache usage defensively
  (never raise) and are wired to the Prometheus telemetry.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agent_utilities.caching.prompt_cache import (
    PromptCacheKey,
    fold_prompt_cache_hint,
    prompt_cache_create_kwargs,
    record_prompt_cache_usage,
    resolve_prompt_cache_key,
    usage_cache_tokens,
)


def _base_key(**overrides: str) -> PromptCacheKey:
    fields = {
        "tenant": "tenant-a",
        "principal": "user-1",
        "policy_version": "v1",
        "model_identity": "anthropic:claude-x",
        "prompt_fingerprint": "abc123",
        "safety_posture": "strict",
    }
    fields.update(overrides)
    return PromptCacheKey(**fields)


class TestPromptCacheKeyFingerprint:
    def test_fingerprint_deterministic(self) -> None:
        assert _base_key().fingerprint == _base_key().fingerprint

    @pytest.mark.parametrize(
        "override",
        [
            {"tenant": "tenant-b"},
            {"principal": "user-2"},
            {"policy_version": "v2"},
            {"model_identity": "openai:gpt-x"},
            {"prompt_fingerprint": "different"},
            {"safety_posture": "on"},
        ],
    )
    def test_any_component_change_invalidates(self, override: dict[str, str]) -> None:
        assert _base_key().fingerprint != _base_key(**override).fingerprint

    def test_key_is_frozen(self) -> None:
        key = _base_key()
        with pytest.raises(ValidationError):
            key.tenant = "other"  # type: ignore[misc]


class TestResolvePromptCacheKey:
    def test_explicit_args_win_over_ambient(self) -> None:
        key = resolve_prompt_cache_key(
            system_prompt="hello",
            model_identity="openai:gpt",
            tenant="explicit-tenant",
            principal="explicit-principal",
            policy_version="explicit-policy",
        )
        assert key.tenant == "explicit-tenant"
        assert key.principal == "explicit-principal"
        assert key.policy_version == "explicit-policy"
        assert key.model_identity == "openai:gpt"
        assert key.prompt_fingerprint  # non-empty content hash

    def test_no_explicit_tenant_never_raises(self) -> None:
        # Whether or not a GraphSession happens to be ambient in this process, resolving
        # with no explicit tenant must never raise (best-effort fallback either way).
        key = resolve_prompt_cache_key(system_prompt="hi")
        assert isinstance(key.tenant, str)


class TestFoldPromptCacheHint:
    def test_default_on_sets_anthropic_and_openai_directives(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AU_PROMPT_CACHE", raising=False)
        settings = fold_prompt_cache_hint(
            {}, system_prompt="sys", model_identity="anthropic:claude-x", tenant="t1"
        )
        assert settings["anthropic_cache"] is True
        assert settings["anthropic_cache_instructions"] is True
        assert settings["anthropic_cache_tool_definitions"] is True
        assert isinstance(settings["openai_prompt_cache_key"], str)
        assert settings["openai_prompt_cache_key"]

    def test_never_clobbers_explicit_caller_settings(self) -> None:
        settings = fold_prompt_cache_hint(
            {"anthropic_cache": False, "openai_prompt_cache_key": "caller-supplied"},
            system_prompt="sys",
        )
        assert settings["anthropic_cache"] is False
        assert settings["openai_prompt_cache_key"] == "caller-supplied"

    def test_preserves_unrelated_settings(self) -> None:
        settings = fold_prompt_cache_hint({"temperature": 0.2}, system_prompt="sys")
        assert settings["temperature"] == 0.2

    def test_opt_out_leaves_settings_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AU_PROMPT_CACHE", "false")
        settings = fold_prompt_cache_hint({"temperature": 0.5}, system_prompt="sys")
        assert "anthropic_cache" not in settings
        assert "openai_prompt_cache_key" not in settings
        assert settings["temperature"] == 0.5

    def test_tenant_scoped_keys_differ_across_tenants(self) -> None:
        a = fold_prompt_cache_hint({}, system_prompt="sys", tenant="tenant-a")
        b = fold_prompt_cache_hint({}, system_prompt="sys", tenant="tenant-b")
        assert a["openai_prompt_cache_key"] != b["openai_prompt_cache_key"]


class TestPromptCacheCreateKwargs:
    def test_sets_prompt_cache_key_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AU_PROMPT_CACHE", raising=False)
        kwargs = prompt_cache_create_kwargs({}, system_prompt="sys", tenant="t1")
        assert kwargs["prompt_cache_key"]

    def test_never_overrides_explicit_value(self) -> None:
        kwargs = prompt_cache_create_kwargs(
            {"prompt_cache_key": "mine"}, system_prompt="sys", tenant="t1"
        )
        assert kwargs["prompt_cache_key"] == "mine"

    def test_does_not_mutate_input_dict(self) -> None:
        original: dict[str, str] = {}
        prompt_cache_create_kwargs(original, system_prompt="sys", tenant="t1")
        assert original == {}

    def test_opt_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AU_PROMPT_CACHE", "false")
        kwargs = prompt_cache_create_kwargs({}, system_prompt="sys", tenant="t1")
        assert "prompt_cache_key" not in kwargs


class TestUsageCacheTokens:
    def test_extracts_from_pydantic_ai_shaped_usage(self) -> None:
        usage = SimpleNamespace(cache_read_tokens=42, cache_write_tokens=7)
        assert usage_cache_tokens(usage) == (42, 7)

    def test_defensive_on_missing_attrs(self) -> None:
        assert usage_cache_tokens(SimpleNamespace()) == (0, 0)
        assert usage_cache_tokens(None) == (0, 0)

    def test_record_prompt_cache_usage_never_raises_and_returns_tokens(self) -> None:
        usage = SimpleNamespace(cache_read_tokens=10, cache_write_tokens=3)
        read, write = record_prompt_cache_usage(provider="anthropic", usage=usage)
        assert (read, write) == (10, 3)
        # A None usage must not raise either.
        assert record_prompt_cache_usage(provider="openai", usage=None) == (0, 0)
