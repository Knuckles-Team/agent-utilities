"""Per-model registry base_url precedence in create_model.

Regression for a split-endpoint vLLM deployment: the router model (e.g. ``qwen-lite``
served on ``vllm-lite.arpa``) was being created with the graph-level default base_url
(``vllm.arpa``) that the engine threads through for every role, causing a 404 because
``vllm.arpa`` only serves the KG model. The registry is the source of truth for where a
model is served, so its per-model ``base_url`` must win over a caller-supplied default;
unregistered models must still honor an explicit base_url.
"""

from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from agent_utilities.core import model_factory


def _extract_base_url(model) -> str | None:
    prov = getattr(model, "_provider", None) or getattr(model, "provider", None)
    client = getattr(prov, "client", None) or getattr(prov, "_client", None)
    bu = getattr(client, "base_url", None)
    return str(bu) if bu is not None else None


def _unwrap_model(model):
    current = model
    for _ in range(8):
        wrapped = getattr(current, "wrapped", None)
        if wrapped is None:
            wrapped = getattr(current, "_wrapped", None)
        if wrapped is None:
            break
        current = wrapped
    return current


@pytest.mark.concept("ORCH-1.27")
def test_registry_base_url_wins_over_caller_default(monkeypatch):
    """A registered model uses its registry base_url even when a default is forced."""
    monkeypatch.setattr(
        model_factory,
        "get_model_config",
        lambda mid=None: {
            "id": "qwen-lite",
            "provider": "openai",
            "base_url": "http://vllm-lite.arpa/v1",
        },
    )
    # Disable the TestModel short-circuit so real base_url resolution runs
    # (constructing an OpenAIChatModel needs no network; only .run() would).
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")
    model = model_factory.create_model(
        provider="openai",
        model_id="qwen-lite",
        base_url="http://vllm.arpa/v1",  # graph-level default the engine forces
    )
    assert "vllm-lite.arpa" in (_extract_base_url(model) or "")


@pytest.mark.concept("ORCH-1.27")
def test_unregistered_model_honors_explicit_base_url(monkeypatch):
    """An unregistered model still routes to the caller's explicit base_url."""
    monkeypatch.setattr(model_factory, "get_model_config", lambda mid=None: None)
    # Disable the TestModel short-circuit so real base_url resolution runs
    # (constructing an OpenAIChatModel needs no network; only .run() would).
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")
    model = model_factory.create_model(
        provider="openai",
        model_id="totally-unregistered",
        base_url="http://custom.example/v1",
    )
    assert "custom.example" in (_extract_base_url(model) or "")


@pytest.mark.concept("ORCH-1.74")
@pytest.mark.asyncio
async def test_openai_compatible_model_merges_governed_and_agent_system_messages(
    monkeypatch,
):
    """Governed evidence plus dynamic agent instructions become one leading system role.

    This is the exact focused-tools failure shape from graph-os: the mandatory
    context wrapper contributes a leading evidence system part and Pydantic AI
    inserts the agent's dynamic instructions as another leading system message.
    Strict vLLM chat templates accept exactly one.
    """
    monkeypatch.setattr(model_factory, "get_model_config", lambda mid=None: None)
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")
    wrapped = model_factory.create_model(
        provider="openai",
        model_id="qwen/qwen3.6-27b",
        base_url="http://vllm.example/v1",
    )
    model = _unwrap_model(wrapped)
    messages = [
        ModelRequest(parts=[SystemPromptPart(content="governed evidence")]),
        ModelRequest(
            parts=[UserPromptPart(content="read one incident")],
            instructions="ServiceNow agent instructions",
        ),
    ]

    mapped = await model._map_messages(
        messages,
        ModelRequestParameters(),
    )

    assert [message["role"] for message in mapped] == ["system", "user"]
    assert mapped[0]["content"] == (
        "governed evidence\n\nServiceNow agent instructions"
    )
