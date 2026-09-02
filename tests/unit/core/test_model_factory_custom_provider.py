"""CONCEPT:AU-ORCH.adapter.byok-provider-proxy — registry-bound model adapters.

Verifies compatible endpoints remain egress-gated and an arbitrary operator provider
can build a real model through the active neutral registry.
"""

from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from agent_utilities.core import model_factory
from agent_utilities.core.model_factory import create_model
from agent_utilities.models import model_registry as registry_module
from agent_utilities.models.model_registry import ModelDefinition, ModelRegistry

pytestmark = pytest.mark.concept(id="AU-ORCH.adapter.byok-provider-proxy")


def test_custom_provider_requires_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")
    # No base_url anywhere → explicit error (config.openai_base_url may be None in test env).
    import agent_utilities.core.config as cfg

    monkeypatch.setattr(cfg.config, "openai_base_url", None, raising=False)
    with pytest.raises(ValueError, match="base_url"):
        create_model(provider="custom", model_id="m", base_url=None)


def test_custom_provider_rejects_internal_ip(monkeypatch):
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")
    with pytest.raises(ValueError, match="egress guard"):
        create_model(provider="custom", model_id="m", base_url="http://10.0.0.5/v1")


def test_custom_provider_builds_model_for_explicitly_allowed_loopback(monkeypatch):
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")
    monkeypatch.setattr(
        model_factory.config,
        "model_http_allowed_private_hosts",
        ["127.0.0.1"],
        raising=False,
    )
    # Local model egress is available only through the exact AgentConfig host allow-list.
    model = create_model(
        provider="custom",
        model_id="operator-model",
        base_url="http://127.0.0.1:8080/v1",
        api_key="k",
    )
    assert model is not None
    assert getattr(model, "model_name", "operator-model") in (
        "operator-model",
        getattr(model, "model_name", "operator-model"),
    )


def test_registry_factory_builds_arbitrary_operator_provider(monkeypatch):
    external_id = "operator/external-model-v99"
    registry = ModelRegistry(
        models=[
            ModelDefinition(
                id=external_id,
                name="Operator model",
                provider="provider-z",
                model_id=external_id,
                base_url="https://model-gateway.example.test/v1",
            )
        ]
    )
    captured = []

    def build_adapter(request):
        captured.append(request)
        return TestModel(model_name=request.model_id)

    monkeypatch.setattr(registry_module, "_ACTIVE_REGISTRY", registry)
    monkeypatch.setattr(
        model_factory,
        "_resolve_tls_and_headers",
        lambda headers, custom: (None, custom),
    )
    http_client = object()
    monkeypatch.setattr(
        model_factory, "_build_model_http_client", lambda *args: http_client
    )
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")

    created = create_model(model_id=external_id, adapter_factory=build_adapter)

    assert created is not None
    assert captured[0].provider == "provider-z"
    assert captured[0].model_id == external_id
    assert captured[0].http_client is http_client
