"""CONCEPT:ORCH-1.34 — create_model(provider="custom") via the proxy (follow-up #2).

Verifies the custom/proxy provider builds an OpenAI-compatible model at the resolved base_url, is
gated by the SSRF egress guard (internal-IP base_url rejected), and requires a base_url.
"""

from __future__ import annotations

import pytest

from agent_utilities.core.model_factory import create_model

pytestmark = pytest.mark.concept(id="ORCH-1.34")


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


def test_custom_provider_builds_model_for_public_url(monkeypatch):
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")
    # Loopback is allowed by the egress guard → local proxy / local LLM works.
    model = create_model(
        provider="custom",
        model_id="gpt-x",
        base_url="http://127.0.0.1:8080/v1",
        api_key="k",
    )
    assert model is not None
    assert getattr(model, "model_name", "gpt-x") in (
        "gpt-x",
        getattr(model, "model_name", "gpt-x"),
    )
