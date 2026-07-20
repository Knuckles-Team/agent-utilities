from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.backends.contrib import neo4j_backend as module


def _trust(tmp_path, **overrides):
    values = {
        "configured": True,
        "verify_enabled": True,
        "ca_bundle_path": tmp_path / "runtime-ca.pem",
        "ca_directory": None,
        "client_cert_path": None,
        "client_key_path": None,
        "client_key_password": None,
        "proxy_url": None,
        "cleanup": MagicMock(),
    }
    values.update(overrides)
    values["ca_bundle_path"].write_text("synthetic", encoding="utf-8")
    return SimpleNamespace(**values)


def test_neo4j_custom_ca_profile_enables_encryption_and_database(
    tmp_path, monkeypatch
) -> None:
    graph_database = MagicMock()
    driver = graph_database.driver.return_value
    session = driver.session.return_value
    session.__enter__.return_value = session
    session.run.return_value = []
    trust = _trust(tmp_path)
    custom_ca = MagicMock(return_value="custom-ca-policy")
    monkeypatch.setattr(module, "GraphDatabase", graph_database)
    monkeypatch.setattr(module, "TrustCustomCAs", custom_ca)
    monkeypatch.setattr(
        module, "resolve_configured_tls_profile", lambda *_a, **_kw: trust
    )

    backend = module.Neo4jBackend(
        uri="neo4j://graph.example.test:7687",
        user="runtime-user",
        password="runtime-secret",
        database="domain-graph",
        tls_profile_ref="secret://graphs/domain/tls",
    )
    backend.execute("RETURN 1")

    options = graph_database.driver.call_args.kwargs
    assert options["encrypted"] is True
    assert options["trusted_certificates"] == "custom-ca-policy"
    assert "runtime-secret" not in repr(options["trusted_certificates"])
    session_options = driver.session.call_args.kwargs
    assert session_options == {"database": "domain-graph"}


def test_neo4j_rejects_http_proxy_profile(tmp_path, monkeypatch) -> None:
    graph_database = MagicMock()
    trust = _trust(tmp_path, proxy_url="https://proxy.example.test:8443")
    monkeypatch.setattr(module, "GraphDatabase", graph_database)
    monkeypatch.setattr(
        module, "resolve_configured_tls_profile", lambda *_a, **_kw: trust
    )

    with pytest.raises(ValueError, match="does not support HTTP proxy"):
        module.Neo4jBackend(
            uri="neo4j://graph.example.test:7687",
            user="runtime-user",
            password="runtime-secret",
            tls_profile_ref="secret://graphs/domain/tls",
        )

    graph_database.driver.assert_not_called()
    trust.cleanup.assert_called_once()


def test_neo4j_execute_read_uses_driver_read_transaction(tmp_path, monkeypatch) -> None:
    graph_database = MagicMock()
    driver = graph_database.driver.return_value
    session = driver.session.return_value
    session.__enter__.return_value = session
    transaction = MagicMock()
    transaction.run.return_value = [{"n": {"id": "node-1"}}]
    session.execute_read.side_effect = lambda operation: operation(transaction)
    trust = _trust(tmp_path)
    monkeypatch.setattr(module, "GraphDatabase", graph_database)
    monkeypatch.setattr(module, "TrustCustomCAs", MagicMock())
    monkeypatch.setattr(
        module, "resolve_configured_tls_profile", lambda *_a, **_kw: trust
    )

    backend = module.Neo4jBackend(
        uri="neo4j://graph.example.test:7687",
        user="runtime-user",
        password="runtime-secret",
        database="domain-graph",
    )

    assert backend.execute_read("MATCH (n) RETURN n") == [
        {"n": {"id": "node-1"}}
    ]
    session.execute_read.assert_called_once()
    transaction.run.assert_called_once_with("MATCH (n) RETURN n", {})
    driver.session.assert_called_with(database="domain-graph")
