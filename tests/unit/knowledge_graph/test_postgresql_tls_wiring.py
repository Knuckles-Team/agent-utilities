from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize("backend_type", ["postgresql", "age"])
def test_create_backend_forwards_complete_postgres_tls_contract(
    monkeypatch: pytest.MonkeyPatch,
    backend_type: str,
) -> None:
    from agent_utilities.knowledge_graph import backends
    from agent_utilities.knowledge_graph.backends import (
        age_backend,
        postgresql_backend,
    )

    captured: dict[str, object] = {}

    class Backend:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def create_schema(self) -> None:
            return None

    monkeypatch.setattr(backends, "setting", lambda _key, default="": default)
    if backend_type == "age":
        monkeypatch.setattr(age_backend, "AGEBackend", Backend)
    else:
        monkeypatch.setattr(postgresql_backend, "PostgreSQLBackend", Backend)

    tls_profile_config = {"system_trust": True, "trust_env": False}

    def resolve_profile(_reference: str) -> str | None:
        return None

    created = backends.create_backend(
        backend_type=backend_type,
        uri="postgresql://runtime-profile",
        db_name="agent_graph",
        tls_profile="database-trust",
        tls_profile_ref="secret://database/tls",
        tls_profile_config=tls_profile_config,
        profile_resolver=resolve_profile,
    )

    assert created is not None
    assert captured["tls_profile"] == "database-trust"
    assert captured["tls_profile_ref"] == "secret://database/tls"
    assert captured["tls_profile_config"] == tls_profile_config
    assert captured["profile_resolver"] is resolve_profile


def test_postgresql_pool_resolves_and_applies_complete_tls_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.core import transport_security
    from agent_utilities.knowledge_graph.backends.postgresql_backend import (
        PostgreSQLBackend,
    )

    trust = MagicMock()
    trust.psycopg_kwargs.return_value = {
        "sslmode": "verify-full",
        "sslrootcert": "runtime-ca-bundle.pem",
    }
    resolve = MagicMock(return_value=trust)
    monkeypatch.setattr(
        transport_security,
        "resolve_configured_tls_profile",
        resolve,
    )
    pool_options: dict[str, object] = {}

    class ConnectionPool:
        def __init__(self, dsn: str, **kwargs: object) -> None:
            pool_options.update(dsn=dsn, **kwargs)

        def close(self) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "psycopg_pool",
        SimpleNamespace(ConnectionPool=ConnectionPool),
    )
    tls_profile_config = {"system_trust": True, "trust_env": False}

    def resolve_profile(_reference: str) -> str | None:
        return None

    backend = PostgreSQLBackend(
        "postgresql://runtime-profile",
        tls_profile="database-trust",
        tls_profile_ref="secret://database/tls",
        tls_profile_config=tls_profile_config,
        profile_resolver=resolve_profile,
    )

    assert backend._ensure_pool() is backend._pool
    resolve.assert_called_once_with(
        "POSTGRES",
        profile_name="database-trust",
        profile_ref="secret://database/tls",
        profile=tls_profile_config,
        resolver=resolve_profile,
    )
    assert pool_options["kwargs"] == {
        "autocommit": False,
        "sslmode": "verify-full",
        "sslrootcert": "runtime-ca-bundle.pem",
    }
