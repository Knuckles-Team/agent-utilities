"""Default-model routing, verified TLS, and per-model static headers in create_model.

Three regressions on the model construction path:

* **Default routing** — when no ``model_id``/``role`` is supplied, the factory must route
  to the operator's DEFINED default chat model (``config.default_chat_model``), not a
  hardcoded ``qwen/qwen3.6-27b`` literal.
* **Verified TLS** — registered model endpoints use the runtime TLS profile and cannot
  disable certificate verification.
* **Per-model headers** — a registered model's reference-backed headers must be
  resolved in memory and sit under any call-site header.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.core import model_factory


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), 3_601, True])
def test_model_transport_rejects_unbounded_timeouts(timeout):
    with pytest.raises(ValueError, match="timeout"):
        model_factory._validated_http_options(timeout, None)


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "redirect.invalid"},
        {"Connection": "keep-alive"},
        {"X-Test": "value\r\ninjected: true"},
        {"not a header": "value"},
    ],
)
def test_model_transport_rejects_dangerous_headers(headers):
    with pytest.raises(ValueError, match="header"):
        model_factory._validated_http_options(30.0, headers)


def test_model_factory_rejects_unknown_provider_before_client_construction(monkeypatch):
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")
    monkeypatch.setattr(model_factory, "get_model_config", lambda mid=None: None)
    with pytest.raises(ValueError, match="unsupported model provider"):
        model_factory._create_model_impl(provider="unknown", model_id="model")


def _client(model):
    prov = getattr(model, "_provider", None) or getattr(model, "provider", None)
    return getattr(prov, "client", None) or getattr(prov, "_client", None)


def test_default_routing_uses_defined_default_not_hardcoded_qwen(monkeypatch):
    """No model_id/role → the factory resolves the DEFINED default chat model's id."""
    seen = {}

    def fake_get_model_config(mid=None):
        seen["id"] = mid
        return {
            "id": "house-model",
            "provider": "openai",
            "base_url": "http://house.arpa/v1",
        }

    monkeypatch.setattr(
        model_factory,
        "config",
        SimpleNamespace(
            default_chat_model=SimpleNamespace(id="house-model"),
            openai_base_url="http://house.arpa/v1",
            openai_api_key="k",
                model_tls_profile=None,
                model_tls_profile_ref=None,
                model_http_allowed_private_hosts=[],
            ),
    )
    monkeypatch.setattr(model_factory, "get_model_config", fake_get_model_config)
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")

    model_factory.create_model()  # no provider / model_id / role

    assert seen["id"] == "house-model"  # the defined default, NOT "qwen/qwen3.6-27b"


def test_per_model_reference_backed_headers_sent(monkeypatch):
    """Reference-backed headers land on the client's default headers."""
    monkeypatch.setenv("TEST_MODEL_HEADERS", '{"X-Client-Id":"synthetic-client"}')
    monkeypatch.setattr(
        model_factory,
        "get_model_config",
        lambda mid=None: {
            "id": "gw",
            "provider": "openai",
            "base_url": "https://gateway.arpa/v1",
            "headers_ref": "env://TEST_MODEL_HEADERS",
        },
    )
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")

    model = model_factory.create_model(provider="openai", model_id="gw")

    headers = getattr(_client(model), "default_headers", {}) or {}
    assert headers.get("X-Client-Id") == "synthetic-client"


def test_call_site_header_wins_over_per_model_header(monkeypatch):
    """An explicit custom_headers value overrides the per-model static header."""
    monkeypatch.setenv("TEST_MODEL_HEADERS", '{"X-Client-Id":"from-reference"}')
    monkeypatch.setattr(
        model_factory,
        "get_model_config",
        lambda mid=None: {
            "id": "gw",
            "provider": "openai",
            "base_url": "https://gateway.arpa/v1",
            "headers_ref": "env://TEST_MODEL_HEADERS",
        },
    )
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")

    model = model_factory.create_model(
        provider="openai",
        model_id="gw",
        custom_headers={"X-Client-Id": "from-call"},
    )

    headers = getattr(_client(model), "default_headers", {}) or {}
    assert headers.get("X-Client-Id") == "from-call"


def test_per_model_api_key_reference_is_resolved_in_memory(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_API_KEY", "synthetic-runtime-material")
    monkeypatch.setattr(
        model_factory,
        "get_model_config",
        lambda mid=None: {
            "id": "gw",
            "provider": "openai",
            "base_url": "https://gateway.arpa/v1",
            "api_key_ref": "env://TEST_MODEL_API_KEY",
        },
    )
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")

    model = model_factory.create_model(provider="openai", model_id="gw")

    assert _client(model).api_key == "synthetic-runtime-material"


def _reasoning_extra_body(model):
    settings = getattr(model, "settings", None)
    if not settings:
        return None
    return dict(settings).get("extra_body")


def test_per_model_reasoning_effort_pins_level(monkeypatch):
    """A configured reasoning_effort level is threaded into the request (extra_body)."""
    monkeypatch.setattr(
        model_factory,
        "get_model_config",
        lambda mid=None: {
            "id": "thinker",
            "provider": "openai",
            "base_url": "https://vllm.arpa/v1",
            "reasoning_effort": "high",
        },
    )
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")

    model = model_factory._create_model_impl(provider="openai", model_id="thinker")
    assert (_reasoning_extra_body(model) or {}).get("reasoning_effort") == "high"


def test_per_model_reasoning_effort_null_opts_into_native_reasoning(monkeypatch):
    """reasoning_effort=None (explicit null) sends NO override — the model reasons natively.

    Even though the caller default is 'none' (thinking off), the per-model null wins and no
    reasoning_effort is injected, so the model uses its own default behaviour.
    """
    monkeypatch.setattr(
        model_factory,
        "get_model_config",
        lambda mid=None: {
            "id": "native",
            "provider": "openai",
            "base_url": "https://vllm.arpa/v1",
            "reasoning_effort": None,
        },
    )
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")

    model = model_factory._create_model_impl(
        provider="openai", model_id="native", reasoning_effort="none"
    )
    # No reasoning_effort override present (settings is None, or extra_body lacks the key).
    assert (_reasoning_extra_body(model) or {}).get("reasoning_effort") is None


def test_reasoning_effort_inherit_keeps_caller_value(monkeypatch):
    """The default 'inherit' sentinel leaves the caller's reasoning_effort untouched."""
    monkeypatch.setattr(
        model_factory,
        "get_model_config",
        lambda mid=None: {
            "id": "plain",
            "provider": "openai",
            "base_url": "https://vllm.arpa/v1",
            "reasoning_effort": "inherit",
        },
    )
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")

    model = model_factory._create_model_impl(
        provider="openai", model_id="plain", reasoning_effort="low"
    )
    assert (_reasoning_extra_body(model) or {}).get("reasoning_effort") == "low"
