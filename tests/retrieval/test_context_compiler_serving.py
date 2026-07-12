#!/usr/bin/python
"""Tests for the Seam 6 serving-layer wire (CONCEPT:AU-KG.retrieval.context-compiler-kv-seam, deep half).

Two things under test, both hermetic (no network, no live vLLM):

1. :meth:`ContextBundle.as_prompt_messages` — the bundle's ``as_text()`` renders
   as a BYTE-IDENTICAL system-message prefix across two calls with the same
   bundle and different ``turn_text``, and DIFFERS when the underlying bundle
   (evidence/policy/budget) differs — the prefix-cache-reuse contract vLLM's
   automatic prefix cache keys off of.
2. :func:`bundle_chat_completion` / :func:`resolve_bundle_chat_client` — the
   real-call wrapper builds the expected ``messages=`` and forwards them (plus
   any extra kwargs) to a ``chat.completions.create``-shaped client, and
   resolves ``base_url``/``model`` from ``config.default_chat_model`` with the
   documented fallback order, using a FAKE client/config — never touching the
   network.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.knowledge_graph.ontology.permissioning import clear_markings
from agent_utilities.knowledge_graph.retrieval.context_compiler import (
    DEFAULT_BUNDLE_SYSTEM_PREAMBLE,
    ContextCompiler,
)
from agent_utilities.knowledge_graph.retrieval.context_compiler_serving import (
    bundle_chat_completion,
    resolve_bundle_chat_client,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext


@pytest.fixture(autouse=True)
def _clean_state():
    reset_company_brain()
    clear_markings()
    yield
    reset_company_brain()
    clear_markings()


class FakeRetriever:
    def __init__(self, nodes: list[dict]) -> None:
        self._nodes = nodes

    def retrieve_hybrid(self, query, context_window=10, **kwargs):
        return list(self._nodes)[:context_window]


def _session() -> GraphSession:
    actor = ActorContext(actor_id="agent:test", actor_type=ActorType.AI_AGENT)
    return GraphSession(actor=actor, policy_version="v1")


_NODES_A = [
    {
        "id": "claim:a",
        "type": "Claim",
        "name": "Claim A",
        "description": "The premise claim, well sourced.",
        "score": 0.9,
        "confidence": 0.9,
        "source_refs": ["doc:1"],
    },
    {
        "id": "claim:b",
        "type": "Claim",
        "name": "Claim B",
        "description": "The dependent claim, worded very differently indeed.",
        "score": 0.85,
        "confidence": 0.7,
        "proof_ids": ["claim:a"],
    },
]

_NODES_B = [
    {
        "id": "claim:x",
        "type": "Claim",
        "name": "Claim X",
        "description": "A totally unrelated claim about a different topic.",
        "score": 0.9,
        "confidence": 0.9,
    },
]


def _compile(nodes: list[dict]) -> Any:
    compiler = ContextCompiler(FakeRetriever(nodes))
    return compiler.compile("test query", session=_session(), top_k=2, candidate_pool=2)


# --------------------------------------------------------------------------
# as_prompt_messages: stable prefix, varying suffix, no false prefix reuse
# --------------------------------------------------------------------------


def test_same_bundle_same_prefix_different_turn_text():
    bundle = _compile(_NODES_A)
    msgs1 = bundle.as_prompt_messages("What is claim A?")
    msgs2 = bundle.as_prompt_messages("Summarize the dispute.")

    assert msgs1[0]["role"] == "system"
    assert msgs2[0]["role"] == "system"
    # The stable prefix is BYTE-IDENTICAL regardless of the turn-specific suffix.
    assert msgs1[0]["content"] == msgs2[0]["content"]
    assert msgs1[0]["content"].startswith(DEFAULT_BUNDLE_SYSTEM_PREAMBLE)
    assert bundle.as_text() in msgs1[0]["content"]

    # The suffix (user message) is exactly the turn text and varies.
    assert msgs1[1] == {"role": "user", "content": "What is claim A?"}
    assert msgs2[1] == {"role": "user", "content": "Summarize the dispute."}


def test_same_bundle_called_twice_is_byte_identical():
    """Same bundle, same turn_text — the FULL messages payload is byte-identical,
    exactly what a repeated real call to vLLM needs for prefix-cache reuse."""
    bundle = _compile(_NODES_A)
    msgs1 = bundle.as_prompt_messages("same question")
    msgs2 = bundle.as_prompt_messages("same question")
    assert msgs1 == msgs2


def test_different_bundle_yields_different_prefix():
    bundle_a = _compile(_NODES_A)
    bundle_b = _compile(_NODES_B)

    msgs_a = bundle_a.as_prompt_messages("same question")
    msgs_b = bundle_b.as_prompt_messages("same question")

    # Different evidence -> different as_text() -> different system prefix ->
    # no false prefix-cache reuse across unrelated bundles.
    assert msgs_a[0]["content"] != msgs_b[0]["content"]
    # The turn-specific suffix is unaffected by which bundle it's attached to.
    assert msgs_a[1] == msgs_b[1] == {"role": "user", "content": "same question"}


def test_custom_system_preamble_is_honored_and_still_stable():
    bundle = _compile(_NODES_A)
    msgs1 = bundle.as_prompt_messages("q1", system_preamble="CUSTOM PREAMBLE\n")
    msgs2 = bundle.as_prompt_messages("q2", system_preamble="CUSTOM PREAMBLE\n")
    assert msgs1[0]["content"] == msgs2[0]["content"]
    assert msgs1[0]["content"].startswith("CUSTOM PREAMBLE\n")


def test_empty_bundle_still_produces_deterministic_messages():
    bundle = _compile([])
    msgs1 = bundle.as_prompt_messages("q")
    msgs2 = bundle.as_prompt_messages("q")
    assert msgs1 == msgs2
    assert "No context found" in msgs1[0]["content"]


# --------------------------------------------------------------------------
# bundle_chat_completion / resolve_bundle_chat_client — the real-call wrapper
# --------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"echo": kwargs}


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


def test_bundle_chat_completion_sends_expected_messages_and_kwargs():
    bundle = _compile(_NODES_A)
    client = _FakeClient()

    result = bundle_chat_completion(
        bundle,
        "What does claim A say?",
        client=client,
        model="qwen/qwen3.6-27b",
        max_tokens=16,
        temperature=0.0,
    )

    assert len(client.chat.completions.calls) == 1
    call = client.chat.completions.calls[0]
    assert call["model"] == "qwen/qwen3.6-27b"
    assert call["max_tokens"] == 16
    assert call["temperature"] == 0.0
    assert call["messages"] == bundle.as_prompt_messages("What does claim A say?")
    assert result == {"echo": call}


def test_bundle_chat_completion_two_calls_same_bundle_share_stable_prefix():
    """The wrapper's own request payload preserves the stable-prefix property —
    proving the wire (not just the raw messages helper) carries it end-to-end."""
    bundle = _compile(_NODES_A)
    client = _FakeClient()

    bundle_chat_completion(bundle, "turn one", client=client, model="m")
    bundle_chat_completion(bundle, "turn two", client=client, model="m")

    calls = client.chat.completions.calls
    assert len(calls) == 2
    assert calls[0]["messages"][0] == calls[1]["messages"][0]  # stable system prefix
    assert calls[0]["messages"][1] != calls[1]["messages"][1]  # varying suffix


def test_resolve_bundle_chat_client_falls_back_to_vllm_arpa(monkeypatch):
    from agent_utilities.core import config as config_module

    monkeypatch.setattr(
        type(config_module.config), "default_chat_model", property(lambda self: None)
    )

    client, model_id = resolve_bundle_chat_client()
    assert str(client.base_url).rstrip("/") == "http://vllm.arpa/v1"
    assert model_id == "default"


def test_resolve_bundle_chat_client_prefers_configured_default(monkeypatch):
    from agent_utilities.core import config as config_module

    class _FakeChatModelConfig:
        id = "qwen/qwen3.6-27b"
        base_url = "http://vllm.arpa/v1"
        api_key = "s3cr3t"

    monkeypatch.setattr(
        type(config_module.config),
        "default_chat_model",
        property(lambda self: _FakeChatModelConfig()),
    )

    client, model_id = resolve_bundle_chat_client()
    assert model_id == "qwen/qwen3.6-27b"
    assert str(client.base_url).rstrip("/") == "http://vllm.arpa/v1"
    assert client.api_key == "s3cr3t"


def test_resolve_bundle_chat_client_explicit_override_wins(monkeypatch):
    from agent_utilities.core import config as config_module

    class _FakeChatModelConfig:
        id = "some-other-model"
        base_url = "http://other.arpa/v1"
        api_key = "x"

    monkeypatch.setattr(
        type(config_module.config),
        "default_chat_model",
        property(lambda self: _FakeChatModelConfig()),
    )

    client, model_id = resolve_bundle_chat_client(
        base_url="http://override.arpa/v1", model="override-model"
    )
    assert model_id == "override-model"
    assert str(client.base_url).rstrip("/") == "http://override.arpa/v1"
