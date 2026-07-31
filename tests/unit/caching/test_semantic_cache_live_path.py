"""Live-path test: the semantic cache wired into the REAL Seam-6 serving call
(``agent_utilities.knowledge_graph.retrieval.context_compiler_serving.bundle_chat_completion`` /
``bundle_async_chat_completion``), not just the isolated :class:`SemanticCache` class.

CONCEPT:AU-KG.memory.semantic-response-cache. This is the entry point every direct
OpenAI-compatible completion in agent-utilities is centralized through (module docstring of
``context_compiler_serving.py``), so proving the cache is actually consulted HERE — not merely
importable and unit-tested in isolation — is the Wire-First proof this capability is live, not
dead code.

Also proves the module-level default-on provider prompt-cache-key hint (Task A) is folded into
the SAME raw ``chat.completions.create`` call.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.caching.semantic_cache import SemanticCache, SemanticCachePolicy
from agent_utilities.knowledge_graph.retrieval.context_compiler import ContextBundle
from agent_utilities.knowledge_graph.retrieval.context_compiler_serving import (
    bundle_async_chat_completion,
    bundle_chat_completion,
)


class _FakeCompletions:
    def __init__(self, reply: str = "Paris") -> None:
        self.calls: list[dict[str, Any]] = []
        self._reply = reply

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        from types import SimpleNamespace

        message = SimpleNamespace(content=self._reply, role="assistant")
        choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
        return SimpleNamespace(choices=[choice], usage=SimpleNamespace(), model="m")


class _FakeAsyncCompletions(_FakeCompletions):
    async def create(self, **kwargs: Any) -> Any:  # type: ignore[override]
        return super().create(**kwargs)


class _FakeChat:
    def __init__(self, completions: Any) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: Any) -> None:
        self.chat = _FakeChat(completions)


def _bundle(
    *, tenant: str = "tenant-live", policy_version: str = "v1"
) -> ContextBundle:
    return ContextBundle(
        query="what is the capital of france",
        session_tenant=tenant,
        policy_version=policy_version,
    )


def _policy(**overrides: object) -> SemanticCachePolicy:
    fields: dict[str, object] = {
        "enabled": True,
        "side_effect_free": True,
        "freshness_tolerance_seconds": 300,
        "similarity_threshold": 0.9,
    }
    fields.update(overrides)
    return SemanticCachePolicy(**fields)  # type: ignore[arg-type]


def _embed(text: str) -> list[float]:
    vector = [0.0] * 26
    for ch in text.lower():
        if "a" <= ch <= "z":
            vector[ord(ch) - 97] += 1.0
    return vector


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch: pytest.MonkeyPatch) -> SemanticCache:
    """Fresh, master-switch-enabled cache instance patched in as the process default, so this
    test's entries never leak into (or are polluted by) any other test's cache state."""
    monkeypatch.setenv("AU_SEMANTIC_CACHE", "true")
    cache = SemanticCache(embed_fn=_embed)
    import agent_utilities.caching.semantic_cache as module

    monkeypatch.setattr(module, "_default_cache", cache)
    return cache


class TestSemanticCacheLivePathSync:
    def test_first_call_misses_and_stores_second_call_hits_without_a_live_request(
        self,
    ) -> None:
        bundle = _bundle()
        client = _FakeClient(_FakeCompletions(reply="Paris"))
        policy = _policy()

        first = bundle_chat_completion(
            bundle,
            "what is the capital of france",
            client=client,
            model="m",
            semantic_cache_policy=policy,
        )
        assert getattr(first, "au_cache_hit", False) is False
        assert len(client.chat.completions.calls) == 1
        assert first.choices[0].message.content == "Paris"

        second = bundle_chat_completion(
            bundle,
            "what is the capital of france",
            client=client,
            model="m",
            semantic_cache_policy=policy,
        )
        # The live client must NOT have been called a second time.
        assert len(client.chat.completions.calls) == 1
        assert second.au_cache_hit is True
        assert second.choices[0].message.content == "Paris"
        # A cache hit is never indistinguishable from a fresh call.
        assert second.au_cache_similarity is not None
        assert second.au_cache_age_seconds is not None
        assert second.au_cache_fingerprint

    def test_no_policy_never_touches_the_cache(self) -> None:
        bundle = _bundle()
        client = _FakeClient(_FakeCompletions(reply="Paris"))

        bundle_chat_completion(
            bundle, "what is the capital of france", client=client, model="m"
        )
        bundle_chat_completion(
            bundle, "what is the capital of france", client=client, model="m"
        )

        # No semantic_cache_policy => always a live call, every time (default-off, opt-in).
        assert len(client.chat.completions.calls) == 2

    def test_side_effecting_request_is_never_served_from_cache(self) -> None:
        bundle = _bundle()
        client = _FakeClient(_FakeCompletions(reply="Paris"))
        policy = _policy(side_effect_free=False)

        bundle_chat_completion(
            bundle,
            "what is the capital of france",
            client=client,
            model="m",
            semantic_cache_policy=policy,
        )
        second = bundle_chat_completion(
            bundle,
            "what is the capital of france",
            client=client,
            model="m",
            semantic_cache_policy=policy,
        )

        assert len(client.chat.completions.calls) == 2
        assert getattr(second, "au_cache_hit", False) is False

    def test_cross_tenant_bundle_never_shares_a_cache_hit(self) -> None:
        client = _FakeClient(_FakeCompletions(reply="Paris"))
        policy = _policy()

        bundle_chat_completion(
            _bundle(tenant="tenant-a"),
            "what is the capital of france",
            client=client,
            model="m",
            semantic_cache_policy=policy,
        )
        result = bundle_chat_completion(
            _bundle(tenant="tenant-b"),
            "what is the capital of france",
            client=client,
            model="m",
            semantic_cache_policy=policy,
        )

        assert len(client.chat.completions.calls) == 2
        assert getattr(result, "au_cache_hit", False) is False

    def test_default_on_prompt_cache_key_reaches_the_raw_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AU_PROMPT_CACHE", raising=False)
        bundle = _bundle()
        client = _FakeClient(_FakeCompletions())

        bundle_chat_completion(
            bundle, "what is the capital of france", client=client, model="m"
        )

        call = client.chat.completions.calls[0]
        assert call["prompt_cache_key"]

    def test_caller_supplied_prompt_cache_key_is_preserved(self) -> None:
        bundle = _bundle()
        client = _FakeClient(_FakeCompletions())

        bundle_chat_completion(
            bundle,
            "q",
            client=client,
            model="m",
            prompt_cache_key="caller-value",
        )

        assert client.chat.completions.calls[0]["prompt_cache_key"] == "caller-value"


class TestSemanticCacheLivePathAsync:
    @pytest.mark.asyncio
    async def test_first_call_misses_second_call_hits(self) -> None:
        bundle = _bundle()
        client = _FakeClient(_FakeAsyncCompletions(reply="Paris"))
        policy = _policy()

        first = await bundle_async_chat_completion(
            bundle,
            "what is the capital of france",
            client=client,
            model="m",
            semantic_cache_policy=policy,
        )
        assert getattr(first, "au_cache_hit", False) is False
        assert len(client.chat.completions.calls) == 1

        second = await bundle_async_chat_completion(
            bundle,
            "what is the capital of france",
            client=client,
            model="m",
            semantic_cache_policy=policy,
        )
        assert len(client.chat.completions.calls) == 1
        assert second.au_cache_hit is True
        assert second.choices[0].message.content == "Paris"
