"""Focused tests for the database connector's lazy connection profile path."""

from __future__ import annotations

import pytest

from agent_utilities.protocols.source_connectors.connectors.database import (
    DatabaseConnector,
)


class _FakeSecrets:
    def __init__(self, rendered: str) -> None:
        self.rendered = rendered
        self.calls: list[str] = []

    def resolve_ref(self, ref: str) -> str:
        self.calls.append(ref)
        return self.rendered


class _FakeConnection:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


def test_connection_resolves_profile_and_caches_connector(monkeypatch):
    secrets = _FakeSecrets(
        '{"dsn":"sqlite:///records.db","kind":"sqlite",'
        '"tls_service":" service ","tls_profile":" profile "}'
    )
    connector = DatabaseConnector(
        query="SELECT 1",
        connection_profile_ref="env://DB_PROFILE",
        source_alias="records",
    )
    monkeypatch.setattr(
        "agent_utilities.security.secrets_client.create_secrets_client",
        lambda: secrets,
    )
    monkeypatch.setattr(
        "agent_utilities.protocols.universal_connector.UniversalConnector",
        _FakeConnection,
    )

    first = connector._connection()
    second = connector._connection()

    assert first is second
    assert secrets.calls == ["env://DB_PROFILE"]
    assert first.args == ("sqlite:///records.db",)
    assert first.kwargs["kind"] == "sqlite"
    assert first.kwargs["source_alias"] == "records"
    assert first.kwargs["tls_service"] == "service"
    assert first.kwargs["tls_profile"] == "profile"
    assert first.kwargs["tls_profile_ref"] is None
    assert first.kwargs["tls_resolver"] == secrets.resolve_ref


def test_connection_rejects_unsupported_profile_fields(monkeypatch):
    secrets = _FakeSecrets('{"dsn":"sqlite:///records.db","unexpected":true}')
    connector = DatabaseConnector(
        query="SELECT 1",
        connection_profile_ref="env://DB_PROFILE",
    )
    monkeypatch.setattr(
        "agent_utilities.security.secrets_client.create_secrets_client",
        lambda: secrets,
    )

    with pytest.raises(ValueError, match="unsupported fields"):
        connector._connection()
