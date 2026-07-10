"""Default-model routing, per-model TLS, and per-model static headers in create_model.

Three regressions on the model construction path:

* **Default routing** — when no ``model_id``/``role`` is supplied, the factory must route
  to the operator's DEFINED default chat model (``config.default_chat_model``), not a
  hardcoded ``qwen/qwen3.6-27b`` literal.
* **Per-model TLS** — a registered model's ``ssl_verify`` (bool or CA-bundle path) must
  override the caller/global default for THAT endpoint only (e.g. an internal self-signed
  vLLM) without loosening TLS elsewhere.
* **Per-model headers** — a registered model's static ``headers`` (e.g. a gateway
  ``X-Client-Id``) must be sent on every request, sitting under any call-site header.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx

from agent_utilities.core import model_factory


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
        ),
    )
    monkeypatch.setattr(model_factory, "get_model_config", fake_get_model_config)
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")

    model_factory.create_model()  # no provider / model_id / role

    assert seen["id"] == "house-model"  # the defined default, NOT "qwen/qwen3.6-27b"


def test_per_model_ssl_verify_overrides_global(monkeypatch):
    """A registered model's ssl_verify=False wins over a caller ssl_verify=True."""
    captured: dict = {}

    class _SpyClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    # Subclass (not a bare function) so the openai SDK's isinstance(..., AsyncClient) holds.
    monkeypatch.setattr(httpx, "AsyncClient", _SpyClient)
    monkeypatch.setattr(
        model_factory,
        "get_model_config",
        lambda mid=None: {
            "id": "internal",
            "provider": "openai",
            "base_url": "https://vllm.internal/v1",
            "ssl_verify": False,
        },
    )
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")

    model_factory.create_model(provider="openai", model_id="internal", ssl_verify=True)

    assert captured.get("verify") is False  # per-model TLS won over the global default


def test_per_model_static_headers_sent(monkeypatch):
    """A registered model's static headers land on the client's default headers."""
    monkeypatch.setattr(
        model_factory,
        "get_model_config",
        lambda mid=None: {
            "id": "gw",
            "provider": "openai",
            "base_url": "https://gateway.arpa/v1",
            "headers": {"X-Client-Id": "svc-42"},
        },
    )
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")

    model = model_factory.create_model(provider="openai", model_id="gw")

    headers = getattr(_client(model), "default_headers", {}) or {}
    assert headers.get("X-Client-Id") == "svc-42"


def test_call_site_header_wins_over_per_model_header(monkeypatch):
    """An explicit custom_headers value overrides the per-model static header."""
    monkeypatch.setattr(
        model_factory,
        "get_model_config",
        lambda mid=None: {
            "id": "gw",
            "provider": "openai",
            "base_url": "https://gateway.arpa/v1",
            "headers": {"X-Client-Id": "from-config"},
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
