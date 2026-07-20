from __future__ import annotations

import pytest

from agent_utilities.messaging import bus_privacy


@pytest.fixture(autouse=True)
def _clear_identity_key_cache():
    bus_privacy._identity_key.cache_clear()
    yield
    bus_privacy._identity_key.cache_clear()


def test_reference_is_stable_idempotent_and_contains_no_input(monkeypatch):
    values = {
        "BUS_IDENTITY_HMAC_KEY_REF": "env://BUS_TEST_KEY",
        "GRAPH_SERVICE_AUTH_SECRET": "",
    }
    monkeypatch.setattr(
        bus_privacy, "setting", lambda name, default="": values.get(name, default)
    )
    monkeypatch.setattr(
        "agent_utilities.security.secrets_client.create_secrets_client",
        lambda: type("Secrets", (), {"resolve_ref": lambda self, _ref: "test-key"})(),
    )

    first = bus_privacy.bus_reference(
        "agent", "person@example.test", tenant="tenant-a"
    )
    second = bus_privacy.bus_reference(
        "agent", "person@example.test", tenant="tenant-a"
    )

    assert first == second
    assert bus_privacy.bus_reference("agent", first, tenant="tenant-a") == first
    assert "person" not in first and "tenant-a" not in first


def test_production_requires_operator_managed_identity_key(monkeypatch):
    monkeypatch.setattr(bus_privacy, "setting", lambda _name, default="": default)
    monkeypatch.setattr(
        "agent_utilities.core.profile_guard.is_production_profile", lambda: True
    )

    with pytest.raises(RuntimeError, match="BUS_IDENTITY_HMAC_KEY_REF"):
        bus_privacy.bus_reference("agent", "runtime-identity", tenant="tenant-a")


def test_content_is_sanitized_before_persistence(monkeypatch):
    monkeypatch.setattr(bus_privacy, "setting", lambda _name, default="": default)
    payload, metadata, report = bus_privacy.sanitize_bus_content(
        "reply to person@example.test from /home/local-account/work/item",
        {"host": "private-host", "token": "secret-value", "safe": "ok"},
    )

    assert "person@example.test" not in payload
    assert "/home/local-account" not in payload
    assert "private-host" not in metadata
    assert "secret-value" not in metadata
    assert '\"safe\":\"ok\"' in metadata
    assert report["redactions"] >= 4
