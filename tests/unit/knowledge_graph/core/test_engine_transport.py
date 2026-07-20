from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.core.engine_transport import (
    EngineTransportError,
    engine_client_transport_kwargs,
    native_endpoint_address,
)


def _config(**overrides):
    values = {
        "engine_tls_profile": None,
        "engine_tls_profile_ref": None,
        "engine_tls_server_name": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_plaintext_native_engine_is_loopback_only_by_default():
    assert engine_client_transport_kwargs(
        "tcp://127.0.0.1:9100", config=_config()
    ) == {}
    with pytest.raises(EngineTransportError, match="requires tls://"):
        engine_client_transport_kwargs(
            "tcp://engine.example.invalid:9100", config=_config()
        )


def test_tls_endpoint_uses_named_profile_and_cleans_material(monkeypatch):
    cleaned = []
    trust = SimpleNamespace(ssl_context=object(), cleanup=lambda: cleaned.append(True))
    captured = {}

    def resolve(service, *, profile_name=None, profile_ref=None):
        captured.update(
            service=service, profile_name=profile_name, profile_ref=profile_ref
        )
        return trust

    monkeypatch.setattr(
        "agent_utilities.core.transport_security.resolve_configured_tls_profile",
        resolve,
    )
    result = engine_client_transport_kwargs(
        "tls://engine.example.invalid:9100",
        config=_config(),
        profile_name="private-trust",
        profile_ref="vault://runtime/engine-tls",
        server_hostname="engine.example.invalid",
    )
    assert result == {
        "tls": trust.ssl_context,
        "tls_server_hostname": "engine.example.invalid",
    }
    assert captured == {
        "service": "ENGINE",
        "profile_name": "private-trust",
        "profile_ref": "vault://runtime/engine-tls",
    }
    assert cleaned == [True]


def test_native_endpoint_address_preserves_ipv6_authority():
    assert native_endpoint_address("tls://[::1]:9100") == ("[::1]:9100", True)
    assert native_endpoint_address("tcp://127.0.0.1:9100") == (
        "127.0.0.1:9100",
        False,
    )
