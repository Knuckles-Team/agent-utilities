"""Integration tests for the `GET /models` endpoint.

Exercises three scenarios:

1. Empty registry -> ``{"models": [], "default_id": null}``.
2. Single-model registry (default single-model bootstrap) -> registry
   populated with a tier-medium default entry.
3. Multi-model registry supplied explicitly -> all entries round-trip and
   ``default_id`` is the one marked ``is_default``.

The tests also verify `resolve_model_registry` reads `MODELS_CONFIG` from
the environment, and that `ModelRegistry` survives an HTTP round-trip via
`model_validate` (the web UI uses this contract).
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agent_utilities.models import (
    ModelCostRate,
    ModelDefinition,
    ModelRegistry,
)
from agent_utilities.server import build_agent_app
from agent_utilities.server.dependencies import resolve_model_registry


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.toolsets = []
    agent.to_a2a.return_value = MagicMock()
    return agent


def _build_client(**kwargs):
    """Construct a TestClient without mounting the web UI or ACP."""
    with patch(
        "agent_utilities.server.app.create_agent",
        return_value=(kwargs.pop("_agent"), []),
    ):
        app = build_agent_app(
            enable_web_ui=False,
            enable_acp=False,
            enable_otel=False,
            graph_bundle=("graph", {"valid_domains": []}),
            **kwargs,
        )
    return TestClient(app)


def test_models_endpoint_empty_registry(mock_agent, monkeypatch):
    """No model_registry + no bootstrap kwargs -> empty payload."""
    monkeypatch.delenv("MODELS_CONFIG", raising=False)
    client = _build_client(
        _agent=mock_agent,
        # Force-empty: clear kwargs and env so resolver returns empty.
        model_id=None,
        provider=None,
        base_url=None,
    )
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"models": [], "default_id": None}


def test_models_endpoint_single_model_bootstrap(mock_agent, monkeypatch):
    """Single-model kwargs auto-bootstrap a one-entry registry."""
    monkeypatch.delenv("MODELS_CONFIG", raising=False)
    client = _build_client(
        _agent=mock_agent,
        provider="openai",
        model_id="gpt-4o-mini",
        base_url=None,
    )
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["default_id"] == "openai:gpt-4o-mini"
    assert len(data["models"]) == 1
    only = data["models"][0]
    assert only["id"] == "openai:gpt-4o-mini"
    assert only["provider"] == "openai"
    assert only["model_id"] == "gpt-4o-mini"
    assert only["tier"] == "medium"
    assert only["is_default"] is True
    assert only["cost"] == {"input": 0.0, "output": 0.0}


def test_models_endpoint_multi_model_registry(mock_agent, monkeypatch):
    """Explicit multi-model registry round-trips verbatim."""
    monkeypatch.delenv("MODELS_CONFIG", raising=False)
    registry = ModelRegistry(
        models=[
            ModelDefinition(
                id="local",
                name="Local LM Studio",
                provider="openai",
                model_id="llama-3.2-3b-instruct",
                base_url="http://localhost:1234/v1",
                tier="light",
                is_default=True,
            ),
            ModelDefinition(
                id="mini",
                name="GPT-4o Mini",
                provider="openai",
                model_id="gpt-4o-mini",
                api_key_env="OPENAI_API_KEY",
                tier="medium",
                cost=ModelCostRate(input=0.15, output=0.6),
                tags=["code", "tools"],
            ),
            ModelDefinition(
                id="opus",
                name="Claude 3 Opus",
                provider="anthropic",
                model_id="claude-3-opus-20240229",
                api_key_env="ANTHROPIC_API_KEY",
                tier="heavy",
                cost=ModelCostRate(input=15, output=75),
                tags=["reasoning"],
            ),
        ]
    )
    client = _build_client(_agent=mock_agent, model_registry=registry)
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["default_id"] == "local"
    ids = [m["id"] for m in data["models"]]
    assert ids == ["local", "mini", "opus"]
    mini = next(m for m in data["models"] if m["id"] == "mini")
    assert mini["cost"] == {"input": 0.15, "output": 0.6}
    assert mini["tags"] == ["code", "tools"]
    assert mini["api_key_env"] == "OPENAI_API_KEY"

    # The payload is wire-compatible with the ModelRegistry model.
    roundtrip = ModelRegistry.model_validate({"models": data["models"]})
    _opus = roundtrip.get_by_id("opus")
    assert _opus is not None
    assert _opus.tier == "heavy"


def test_models_endpoint_picks_up_models_config_env(
    tmp_path, mock_agent, monkeypatch
):
    """MODELS_CONFIG env var is honoured when no explicit kwargs."""
    payload = {
        "models": [
            {
                "id": "from-env",
                "name": "From Env",
                "provider": "openai",
                "model_id": "gpt-4o-mini",
                "tier": "light",
                "is_default": True,
                "cost": {"input": 0.15, "output": 0.6},
            }
        ]
    }
    cfg = tmp_path / "models.json"
    cfg.write_text(json.dumps(payload))
    monkeypatch.setenv("MODELS_CONFIG", str(cfg))

    client = _build_client(_agent=mock_agent)
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["default_id"] == "from-env"
    assert data["models"][0]["provider"] == "openai"
    assert data["models"][0]["tier"] == "light"


def test_resolve_model_registry_prefers_explicit_registry(monkeypatch):
    """The caller-supplied registry wins over env and single-model fallback."""
    explicit = ModelRegistry(
        models=[
            ModelDefinition(
                id="explicit",
                name="Explicit",
                provider="openai",
                model_id="gpt-4o",
                tier="medium",
                is_default=True,
            )
        ]
    )
    monkeypatch.setenv("MODELS_CONFIG", "/tmp/does-not-exist.json")
    result = resolve_model_registry(
        registry=explicit,
        provider="ignored",
        model_id="ignored",
    )
    assert result is explicit
    _explicit_default = result.get_default()
    assert _explicit_default is not None
    assert _explicit_default.id == "explicit"


def test_resolve_model_registry_falls_back_to_empty(monkeypatch):
    """No registry, no env, no kwargs -> empty registry (not None)."""
    monkeypatch.delenv("MODELS_CONFIG", raising=False)
    result = resolve_model_registry()
    assert isinstance(result, ModelRegistry)
    assert result.models == []
    assert result.get_default() is None


def test_resolve_model_registry_missing_config_file_logs_and_falls_back(
    mock_agent, monkeypatch, tmp_path
):
    """Bad MODELS_CONFIG path must not crash server startup."""
    monkeypatch.setenv("MODELS_CONFIG", str(tmp_path / "nope.json"))
    # File does not exist -> resolver falls through to kwargs/empty.
    result = resolve_model_registry(
        provider="openai", model_id="gpt-4o-mini"
    )
    _missing_default = result.get_default()
    assert _missing_default is not None
    assert _missing_default.id == "openai:gpt-4o-mini"


def test_resolve_model_registry_invalid_yaml_falls_back_to_bootstrap(
    monkeypatch, tmp_path
):
    """Unreadable JSON -> do not crash; fall back to single-model kwargs."""
    cfg = tmp_path / "broken.json"
    cfg.write_text("{not: valid: json")
    monkeypatch.setenv("MODELS_CONFIG", str(cfg))
    result = resolve_model_registry(
        provider="openai", model_id="gpt-4o-mini"
    )
    _broken_default = result.get_default()
    assert _broken_default is not None
    assert _broken_default.id == "openai:gpt-4o-mini"


def test_models_endpoint_app_state_attached(mock_agent, monkeypatch):
    """The registry must be reachable via ``app.state.model_registry``."""
    monkeypatch.delenv("MODELS_CONFIG", raising=False)
    with patch(
        "agent_utilities.server.app.create_agent", return_value=(mock_agent, [])
    ):
        app = build_agent_app(
            provider="openai",
            model_id="gpt-4o-mini",
            enable_web_ui=False,
            enable_acp=False,
            enable_otel=False,
            graph_bundle=("graph", {"valid_domains": []}),
        )
    client = TestClient(app)
    # Trigger the factory by issuing a request so app.state is populated.
    client.get("/health")
    inner_app = app.app if hasattr(app, "app") else app
    reg = getattr(inner_app.state, "model_registry", None)
    assert reg is not None
    assert reg.get_default() is not None
    assert reg.get_default().model_id == "gpt-4o-mini"
    # Ensure the environment was not mutated by the test.
    assert "MODELS_CONFIG" not in os.environ
