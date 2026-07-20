from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from agent_utilities.core import transport_security as transport_security_module
from agent_utilities.core.config import AgentConfig
from agent_utilities.core.transport_security import (
    TransportSecurityError,
    resolve_configured_tls_profile,
    resolve_tls_profile,
    tls_environment_from_config,
)


def _certificate(*, common_name: str, ca: bool) -> tuple[str, str]:
    now = datetime.now(UTC)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.BasicConstraints(ca=ca, path_length=0 if ca else None), True
        )
        .sign(key, hashes.SHA256())
    )
    return (
        certificate.public_bytes(serialization.Encoding.PEM).decode("ascii"),
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode("ascii"),
    )


def test_named_secret_profile_materializes_runtime_ca_without_repr_leak(
    tmp_path,
) -> None:
    ca_pem, _key = _certificate(common_name="Synthetic Runtime Root", ca=True)
    catalog = {
        "profiles": {
            "private-trust": {
                "ca_bundle_ref": "vault://synthetic/ca",
                "system_trust": True,
            }
        }
    }

    def resolve(ref: str) -> str | None:
        return {
            "vault://synthetic/catalog": json.dumps(catalog),
            "vault://synthetic/ca": ca_pem,
        }.get(ref)

    trust = resolve_tls_profile(
        "SYNTHETIC_SERVICE",
        profile_name="private-trust",
        environ={"TLS_PROFILES_REF": "vault://synthetic/catalog"},
        resolver=resolve,
        destination_root=tmp_path,
    )

    assert trust.configured is True
    assert trust.verify_enabled is True
    assert trust.ca_bundle_path is not None
    assert trust.ca_bundle_path.is_file()
    assert trust.httpx_kwargs()["verify"] is trust.ssl_context
    assert trust.requests_kwargs()["verify"] == str(trust.ca_bundle_path)
    assert trust.pymongo_kwargs()["tlsCAFile"] == str(trust.ca_bundle_path)
    assert trust.psycopg_kwargs()["sslrootcert"] == str(trust.ca_bundle_path)
    assert trust.child_env()["REQUESTS_CA_BUNDLE"] == str(trust.ca_bundle_path)
    assert str(tmp_path) not in repr(trust)
    assert "vault://" not in repr(trust)

    materialized = trust.ca_bundle_path
    trust.cleanup()
    assert not materialized.exists()


def test_env_profile_bootstraps_without_graph_secret_backend(
    monkeypatch, tmp_path
) -> None:
    ca_pem, _key = _certificate(common_name="Synthetic Environment Root", ca=True)
    catalog = {
        "profiles": {
            "environment-trust": {
                "ca_bundle_ref": "env://TEST_TLS_CA_PEM",
            }
        }
    }
    monkeypatch.delenv("TEST_TLS_CATALOG", raising=False)
    monkeypatch.delenv("TEST_TLS_CA_PEM", raising=False)
    durable_resolver = MagicMock()
    durable_resolver.side_effect = AssertionError(
        "env refs must not initialize a durable secret backend"
    )

    with patch(
        "agent_utilities.security.secrets_client.create_secrets_client",
        side_effect=AssertionError(
            "env refs must not initialize the graph secret backend"
        ),
    ) as create_secrets_client:
        trust = resolve_tls_profile(
            "SYNTHETIC_SERVICE",
            profile_name="environment-trust",
            environ={
                "TLS_PROFILES_REF": "env://TEST_TLS_CATALOG",
                "TEST_TLS_CATALOG": json.dumps(catalog),
                "TEST_TLS_CA_PEM": ca_pem,
            },
            resolver=durable_resolver,
            destination_root=tmp_path,
        )

    create_secrets_client.assert_not_called()
    durable_resolver.assert_not_called()
    assert trust.ca_bundle_path is not None
    assert trust.ca_bundle_path.is_file()
    assert trust.child_env()["UV_NATIVE_TLS"] == "true"
    trust.cleanup()


def test_configured_tls_view_copies_only_referenced_environment_material() -> None:
    cfg = AgentConfig(TLS_PROFILES_REF="env://TEST_TLS_CATALOG")

    rendered = tls_environment_from_config(
        cfg,
        base_environ={
            "TEST_TLS_CATALOG": '{"profiles": {}}',
            "UNRELATED_RUNTIME_VALUE": "must-not-cross-boundary",
        },
    )

    assert rendered["TEST_TLS_CATALOG"] == '{"profiles": {}}'
    assert "UNRELATED_RUNTIME_VALUE" not in rendered


def test_configured_tls_view_projects_dedicated_certification_prometheus_selector() -> (
    None
):
    cfg = AgentConfig(
        CERT_PROMETHEUS_TLS_PROFILE="production-metrics",
        CERT_PROMETHEUS_TLS_PROFILE_REF="env://TEST_CERT_PROMETHEUS_TLS",
    )

    rendered = tls_environment_from_config(
        cfg,
        base_environ={
            "TEST_CERT_PROMETHEUS_TLS": '{"system_trust":true}',
            "UNRELATED_RUNTIME_VALUE": "must-not-cross-boundary",
        },
    )

    assert rendered["CERT_PROMETHEUS_TLS_PROFILE"] == "production-metrics"
    assert rendered["CERT_PROMETHEUS_TLS_PROFILE_REF"] == (
        "env://TEST_CERT_PROMETHEUS_TLS"
    )
    assert rendered["TEST_CERT_PROMETHEUS_TLS"] == '{"system_trust":true}'
    assert "UNRELATED_RUNTIME_VALUE" not in rendered


def test_configured_explicit_env_profile_uses_isolated_runtime_value(
    monkeypatch, tmp_path
) -> None:
    ca_pem, _key = _certificate(common_name="Synthetic Explicit Root", ca=True)
    monkeypatch.setenv(
        "TEST_EXPLICIT_TLS_PROFILE",
        json.dumps({"ca_bundle_pem": ca_pem, "system_trust": True}),
    )

    with patch(
        "agent_utilities.security.secrets_client.create_secrets_client",
        side_effect=AssertionError("env refs must not initialize a secret backend"),
    ) as create_secrets_client:
        trust = resolve_configured_tls_profile(
            "synthetic",
            profile_ref="env://TEST_EXPLICIT_TLS_PROFILE",
            config=AgentConfig(),
            destination_root=tmp_path,
        )

    create_secrets_client.assert_not_called()
    assert trust.ca_bundle_path is not None
    assert trust.verify_enabled is True
    trust.cleanup()


def test_explicit_tls_reference_is_isolated_from_ambient_selector(tmp_path) -> None:
    ca_pem, _key = _certificate(common_name="Synthetic Isolated Root", ca=True)
    trust = resolve_tls_profile(
        "SYNTHETIC_SERVICE",
        profile_ref="env://EXPLICIT_TLS_PROFILE",
        environ={
            "EXPLICIT_TLS_PROFILE": json.dumps({"ca_bundle_pem": ca_pem}),
            "TLS_PROFILE": "unrelated-ambient-profile",
            "TLS_PROFILE_REF": "env://UNRELATED_TLS_PROFILE",
            "UNRELATED_TLS_PROFILE": "not-json",
        },
        destination_root=tmp_path,
    )

    assert trust.source == "secret_ref"
    assert trust.ca_bundle_path is not None
    trust.cleanup()


def test_tls_validation_failure_removes_materialized_files(tmp_path) -> None:
    ca_pem, _key = _certificate(common_name="Synthetic Cleanup Root", ca=True)

    with pytest.raises(TransportSecurityError, match="tls_ca_source_ambiguous"):
        resolve_tls_profile(
            "SYNTHETIC_SERVICE",
            profile={
                "ca_bundle_pem": ca_pem,
                "ca_directory": str(tmp_path),
            },
            environ={},
            destination_root=tmp_path,
        )

    assert list(tmp_path.rglob("tls-*.pem")) == []


def test_tls_materialization_failure_removes_untracked_file(
    monkeypatch, tmp_path
) -> None:
    ca_pem, _key = _certificate(common_name="Synthetic Chmod Root", ca=True)
    original_chmod = Path.chmod

    def fail_material_chmod(path: Path, mode: int) -> None:
        if path.name.startswith("tls-"):
            raise OSError("synthetic chmod failure")
        original_chmod(path, mode)

    monkeypatch.setattr(Path, "chmod", fail_material_chmod)
    with pytest.raises(TransportSecurityError, match="tls_materialization_failed"):
        resolve_tls_profile(
            "SYNTHETIC_SERVICE",
            profile={"ca_bundle_pem": ca_pem},
            environ={},
            destination_root=tmp_path,
        )

    assert list(tmp_path.rglob("tls-*.pem")) == []


def test_tls_cleanup_retains_failed_unlink_for_process_exit_retry(
    monkeypatch, tmp_path
) -> None:
    ca_pem, _key = _certificate(common_name="Synthetic Retry Root", ca=True)
    trust = resolve_tls_profile(
        "SYNTHETIC_SERVICE",
        profile={"ca_bundle_pem": ca_pem},
        environ={},
        destination_root=tmp_path,
    )
    materialized = trust.ca_bundle_path
    assert materialized is not None
    original_unlink = Path.unlink
    failed_once = False

    def fail_once(path: Path, *args, **kwargs) -> None:
        nonlocal failed_once
        if path == materialized and not failed_once:
            failed_once = True
            raise OSError("synthetic unlink failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)

    trust.cleanup()

    assert materialized.exists()
    assert materialized in transport_security_module._MATERIALIZED_PATHS

    transport_security_module._cleanup_materialized()

    assert not materialized.exists()
    assert materialized not in transport_security_module._MATERIALIZED_PATHS


def test_inline_mtls_and_proxy_are_client_portable(tmp_path) -> None:
    ca_pem, _ = _certificate(common_name="Synthetic Root", ca=True)
    cert_pem, key_pem = _certificate(common_name="Synthetic Client", ca=False)

    trust = resolve_tls_profile(
        "SYNTHETIC_SERVICE",
        profile={
            "ca_bundle_pem": ca_pem,
            "client_cert_pem": cert_pem,
            "client_key_pem": key_pem,
            "proxy_url": "https://proxy.example.test:8443",
            "no_proxy": "localhost,127.0.0.1",
        },
        environ={},
        destination_root=tmp_path,
    )

    assert trust.httpx_kwargs()["proxy"] == "https://proxy.example.test:8443"
    requests_kwargs = trust.requests_kwargs()
    assert requests_kwargs["proxies"]["https"].startswith("https://")
    assert trust.client_bundle_path is not None
    assert requests_kwargs["cert"] == str(trust.client_bundle_path)
    assert trust.client_bundle_path.is_file()
    assert trust.child_env()["NO_PROXY"] == "localhost,127.0.0.1"
    child_env = trust.child_env(service="synthetic-service")
    assert child_env["SYNTHETIC_SERVICE_CLIENT_CERT"] == str(trust.client_bundle_path)
    assert child_env["SYNTHETIC_SERVICE_CLIENT_KEY"] == str(trust.client_bundle_path)
    trust.cleanup()


def test_encrypted_mtls_key_becomes_private_requests_bundle(tmp_path) -> None:
    cert_pem, key_pem = _certificate(common_name="Synthetic Client", ca=False)
    private_key = serialization.load_pem_private_key(
        key_pem.encode("ascii"), password=None
    )
    encrypted_key = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.BestAvailableEncryption(b"synthetic-password"),
    ).decode("ascii")

    trust = resolve_tls_profile(
        "SYNTHETIC_SERVICE",
        profile={
            "client_cert_pem": cert_pem,
            "client_key_pem": encrypted_key,
            "client_key_password_ref": "secret://synthetic/client-key-password",
        },
        environ={},
        resolver=lambda _ref: "synthetic-password",
        destination_root=tmp_path,
    )

    client_bundle = trust.requests_kwargs()["cert"]
    assert client_bundle == str(trust.client_bundle_path)
    assert "ENCRYPTED PRIVATE KEY" not in trust.client_bundle_path.read_text(
        encoding="ascii"
    )
    if os.name == "posix":
        assert stat.S_IMODE(trust.client_bundle_path.stat().st_mode) == 0o600
    trust.cleanup()


@pytest.mark.parametrize(
    "profile",
    [
        {"verify": False},
        {"verify": False, "allow_insecure": True},
        {"allow_insecure": True},
    ],
)
def test_boolean_verification_controls_are_retired(profile: dict[str, bool]) -> None:
    with pytest.raises(TransportSecurityError, match="verification_control_retired"):
        resolve_tls_profile(
            "SYNTHETIC_SERVICE",
            profile=profile,
            environ={},
        )


def test_configured_resolver_projects_agent_config() -> None:
    cfg = AgentConfig(TLS_SYSTEM_TRUST=True, TLS_TRUST_ENV=False)
    trust = resolve_configured_tls_profile("synthetic", config=cfg)

    assert trust.verify_enabled is True
    assert trust.trust_env is False
    assert trust.psycopg_kwargs()["sslmode"] == "verify-full"
    assert trust.pymongo_kwargs()["tls"] is True
    assert trust.redis_kwargs()["ssl_cert_reqs"] == "required"
    assert trust.redis_kwargs()["ssl_check_hostname"] is True


def test_redis_checkpoint_rejects_plaintext_transport() -> None:
    from agent_utilities.core.checkpoint.manager import RedisBackend

    with pytest.raises(ValueError, match="requires rediss"):
        RedisBackend("redis://cache.example.test:6379")


def test_custom_only_trust_requires_an_anchor() -> None:
    with pytest.raises(TransportSecurityError, match="tls_trust_anchor_missing"):
        resolve_tls_profile(
            "SYNTHETIC_SERVICE",
            profile={"system_trust": False},
            environ={},
        )


def test_external_graph_config_is_secret_ref_only() -> None:
    config = AgentConfig(
        EXTERNAL_GRAPH_CONNECTORS=[
            {
                "name": "domain-source",
                "source_alias": "domain-source",
                "backend": "neo4j",
                "connection_profile_ref": "secret://graphs/domain/connection",
                "tls_profile_ref": "vault://graphs/domain/tls",
            }
        ]
    )

    connector = config.external_graph_connectors[0]
    assert connector.backend == "neo4j"
    assert connector.require_approval is True
    assert connector.schema_drift_policy == "fail_closed"
    assert connector.ingest_page_size == 500
    assert connector.ingest_max_pages == 100
    assert connector.ingest_max_row_bytes == 1_048_576
    assert connector.ingest_max_total_bytes == 16_777_216
    assert connector.ingest_max_nesting_depth == 16
    assert connector.ingest_max_collection_items == 10_000
    assert connector.sync_mode == "auto"
    assert connector.reconcile_deletions is True
    assert connector.allow_empty_snapshot is False
    assert "endpoint" not in connector.model_dump()

    with pytest.raises(ValueError, match="runtime secret refs"):
        AgentConfig(
            EXTERNAL_GRAPH_CONNECTORS=[
                {
                    "name": "domain-source",
                    "source_alias": "domain-source",
                    "backend": "neo4j",
                    "connection_profile_ref": "https://graph.example.test",
                }
            ]
        )


@pytest.mark.parametrize("duplicate_field", ["name", "source_alias"])
def test_external_graph_config_rejects_duplicate_declaration_identities(
    duplicate_field,
) -> None:
    declarations = [
        {
            "name": name,
            "source_alias": f"{name}-alias",
            "backend": backend,
            "connection_profile_ref": f"secret://graphs/{name}/connection",
            **(
                {"mapping_policy_ref": f"secret://graphs/{name}/mapping"}
                if backend == "graphql"
                else {}
            ),
        }
        for name, backend in (("source-one", "neo4j"), ("source-two", "graphql"))
    ]
    declarations[1][duplicate_field] = declarations[0][duplicate_field]

    with pytest.raises(
        ValueError, match="names and source_alias values must be unique"
    ):
        AgentConfig(EXTERNAL_GRAPH_CONNECTORS=declarations)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ingest_page_size", 0),
        ("ingest_page_size", 1_001),
        ("ingest_page_size", True),
        ("ingest_max_pages", 0),
        ("ingest_max_pages", 1_001),
        ("ingest_max_records", True),
        ("ingest_max_row_bytes", 255),
        ("ingest_max_row_bytes", 8_388_609),
        ("ingest_max_total_bytes", 67_108_865),
        ("ingest_max_nesting_depth", 0),
        ("ingest_max_nesting_depth", True),
        ("ingest_max_collection_items", 0),
        ("ingest_max_collection_items", True),
        ("reconcile_deletions", "yes"),
        ("allow_empty_snapshot", 1),
        ("sync_mode", "incremental"),
    ],
)
def test_external_property_graph_sync_policy_is_bounded(field, value) -> None:
    declaration = {
        "name": "domain-source",
        "source_alias": "domain-source",
        "backend": "neo4j",
        "connection_profile_ref": "secret://graphs/domain/connection",
        field: value,
    }

    with pytest.raises(ValueError):
        AgentConfig(EXTERNAL_GRAPH_CONNECTORS=[declaration])


def test_external_property_graph_total_budget_must_cover_one_row() -> None:
    with pytest.raises(ValueError, match="cover one bounded row"):
        AgentConfig(
            EXTERNAL_GRAPH_CONNECTORS=[
                {
                    "name": "domain-source",
                    "source_alias": "domain-source",
                    "backend": "neo4j",
                    "connection_profile_ref": "secret://graphs/domain/connection",
                    "ingest_max_row_bytes": 1_024,
                    "ingest_max_total_bytes": 512,
                }
            ]
        )


def test_graphql_external_connector_uses_the_same_secret_ref_boundary() -> None:
    config = AgentConfig(
        EXTERNAL_GRAPH_CONNECTORS=[
            {
                "name": "schema-source",
                "source_alias": "schema-source",
                "backend": "graphql",
                "connection_profile_ref": "secret://graphs/schema/connection",
                "mapping_policy_ref": "secret://graphs/schema/mapping",
            }
        ]
    )

    assert config.external_graph_connectors[0].backend == "graphql"
    assert config.external_graph_connectors[0].discovery_max_depth == 6
    assert config.external_graph_connectors[0].ingest_max_records == 1_000

    generated = AgentConfig(
        EXTERNAL_GRAPH_CONNECTORS=[
            {
                "name": "schema-source",
                "source_alias": "schema-source",
                "backend": "graphql",
                "connection_profile_ref": "secret://graphs/schema/connection",
                "allow_introspection": True,
                "allow_empty_snapshot": True,
            }
        ]
    ).external_graph_connectors[0]
    assert generated.mapping_policy_ref is None
    assert generated.allow_introspection is True
    assert generated.allow_empty_snapshot is True

    with pytest.raises(ValueError, match="allow_introspection"):
        AgentConfig(
            EXTERNAL_GRAPH_CONNECTORS=[
                {
                    "name": "schema-source",
                    "source_alias": "schema-source",
                    "backend": "graphql",
                    "connection_profile_ref": "secret://graphs/schema/connection",
                }
            ]
        )

    with pytest.raises(ValueError, match="property graph sources"):
        AgentConfig(
            EXTERNAL_GRAPH_CONNECTORS=[
                {
                    "name": "schema-source",
                    "source_alias": "schema-source",
                    "backend": "graphql",
                    "connection_profile_ref": "secret://graphs/schema/connection",
                    "mapping_policy_ref": "secret://graphs/schema/mapping",
                    "semantic_mapping": True,
                }
            ]
        )


def test_agent_config_rejects_literal_durable_graph_connection_material() -> None:
    with pytest.raises(ValueError, match="secret reference"):
        AgentConfig(
            KG_CONNECTIONS=[
                {
                    "name": "source",
                    "backend": "neo4j",
                    "uri": "bolt://runtime-only",
                }
            ]
        )


def test_vector_connections_are_typed_and_secret_ref_only() -> None:
    config = AgentConfig(
        DATABASE_TYPE="qdrant",
        DB_HOST="vector.example.test",
        DB_PORT=7443,
        DB_USERNAME_REF="secret://vector/username",
        DB_PASSWORD_REF="secret://vector/password",
        QDRANT_API_KEY_REF="vault://vector/api-key",
        QDRANT_TLS_PROFILE="private-trust",
        QDRANT_TLS_PROFILE_REF="secret://vector/tls-profile",
        QDRANT_HTTP_ALLOWED_PRIVATE_HOSTS=["VECTOR.INTERNAL", "vector.internal"],
        POSTGRES_TLS_PROFILE="database-trust",
        POSTGRES_TLS_PROFILE_REF="secret://vector/postgres-tls",
    )

    assert config.vector_database_type == "qdrant"
    assert config.vector_db_port == 7443
    assert config.qdrant_http_allowed_private_hosts == ["vector.internal"]
    environment = tls_environment_from_config(config, base_environ={})
    assert environment["QDRANT_TLS_PROFILE"] == "private-trust"
    assert environment["QDRANT_TLS_PROFILE_REF"] == "secret://vector/tls-profile"
    assert environment["POSTGRES_TLS_PROFILE"] == "database-trust"
    assert environment["POSTGRES_TLS_PROFILE_REF"] == ("secret://vector/postgres-tls")

    with pytest.raises(ValueError, match="runtime-only material"):
        AgentConfig(QDRANT_API_KEY_REF="literal-api-key")
