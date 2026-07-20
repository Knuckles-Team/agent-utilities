from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from agent_utilities.observability.langfuse_trust import (
    LangfuseTrustError,
    configure_langfuse_trust,
    langfuse_parent_kg_ingestion_enabled,
    langfuse_provider_contract_ready,
    native_langfuse_mcp_config,
    prepare_langfuse_mcp_config,
    resolve_langfuse_credentials,
    resolve_langfuse_requests_transport,
    validate_ca_bundle,
)


def _ca_pair() -> str:
    now = datetime.now(UTC)
    root_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    root_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Synthetic Root")])
    root = (
        x509.CertificateBuilder()
        .subject_name(root_name)
        .issuer_name(root_name)
        .public_key(root_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
        .sign(root_key, hashes.SHA256())
    )
    intermediate_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    intermediate_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "Synthetic Intermediate")]
    )
    intermediate = (
        x509.CertificateBuilder()
        .subject_name(intermediate_name)
        .issuer_name(root_name)
        .public_key(intermediate_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=20))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(root_key, hashes.SHA256())
    )
    return b"".join(
        certificate.public_bytes(serialization.Encoding.PEM)
        for certificate in (intermediate, root)
    ).decode("ascii")


def _credential_refs() -> tuple[dict[str, str], dict[str, str]]:
    public_ref = "secret://observability/public-key"
    secret_ref = "secret://observability/secret-key"
    return (
        {
            "LANGFUSE_PUBLIC_KEY_REF": public_ref,
            "LANGFUSE_SECRET_KEY_REF": secret_ref,
        },
        {public_ref: "synthetic-public", secret_ref: "synthetic-" + "secret"},
    )


def _patch_current_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.find_spec",
        lambda name: (
            object()
            if name
            in {
                "langfuse_agent.mcp_server",
                "langfuse_agent.runtime_posture",
            }
            else None
        ),
    )
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.package_version",
        lambda _name: "1.0.3",
    )


def _client_pair() -> tuple[str, str]:
    now = datetime.now(UTC)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Synthetic Client")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
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


@pytest.mark.parametrize(
    ("reason", "category"),
    [
        ("langfuse_host_invalid", "host"),
        ("langfuse_credentials_missing", "credentials"),
        ("langfuse_credentials_invalid", "credentials"),
        ("langfuse_ca_bundle_invalid", "transport_security"),
        ("unexpected-sensitive-value", "configuration"),
    ],
)
def test_langfuse_trust_error_exposes_only_safe_category(
    reason: str, category: str
) -> None:
    error = LangfuseTrustError(reason)

    assert error.category == category
    assert "unexpected-sensitive-value" not in str(error)


def test_ca_bundle_accepts_chain_and_rejects_truncated_pem() -> None:
    bundle = _ca_pair()

    valid = validate_ca_bundle(bundle)
    invalid = validate_ca_bundle(bundle.split("-----END CERTIFICATE-----", 1)[0])

    assert valid.valid is True
    assert valid.certificate_count == 2
    assert invalid.valid is False
    assert invalid.reason == "invalid_pem"


@pytest.mark.parametrize(
    "sentinel",
    [
        "REDACTED",
        "[REDACTED]",
        "YOUR_LANGFUSE_PUBLIC_KEY",
        "{{ LANGFUSE_SECRET_KEY }}",
        "pk-lf-${LANGFUSE_PUBLIC_KEY}",
        "sk-" + "lf-" + "REDACTED",
        "********",
        "change-me",
    ],
)
def test_credentials_reject_redaction_and_template_sentinels(sentinel: str) -> None:
    references = {
        "LANGFUSE_PUBLIC_KEY_REF": "secret://observability/public-key",
        "LANGFUSE_SECRET_KEY_REF": "secret://observability/secret-key",
    }
    values = {
        "secret://observability/public-key": sentinel,
        "secret://observability/secret-key": "synthetic-secret",
    }

    with pytest.raises(
        LangfuseTrustError, match="langfuse_credentials_invalid"
    ) as caught:
        resolve_langfuse_credentials(
            environ=references,
            resolver=values.get,
        )
    assert sentinel not in str(caught.value)


def test_single_root_and_independent_root_store_are_valid() -> None:
    first = x509.load_pem_x509_certificates(_ca_pair().encode("ascii"))[-1]
    second = x509.load_pem_x509_certificates(_ca_pair().encode("ascii"))[-1]
    single = first.public_bytes(serialization.Encoding.PEM)
    unrelated = b"".join(
        certificate.public_bytes(serialization.Encoding.PEM)
        for certificate in (first, second)
    )

    single_status = validate_ca_bundle(single)
    store_status = validate_ca_bundle(unrelated)

    assert single_status.valid is True
    assert single_status.certificate_count == 1
    assert store_status.valid is True
    assert store_status.certificate_count == 2


def test_configure_projects_one_bundle_to_all_tls_runtimes(tmp_path) -> None:
    bundle_path = tmp_path / "bundle.pem"
    bundle_path.write_text(_ca_pair(), encoding="utf-8")
    environ = {"LANGFUSE_CA_BUNDLE": str(bundle_path)}

    status = configure_langfuse_trust(environ=environ)

    assert status.valid is True
    assert status.certificate_count == 2
    assert environ["REQUESTS_CA_BUNDLE"] == str(bundle_path)
    assert environ["SSL_CERT_FILE"] == str(bundle_path)
    assert "UV_NATIVE_TLS" not in environ
    assert str(bundle_path) not in repr(status)


def test_configure_accepts_standard_environment_trust_store(tmp_path) -> None:
    bundle_path = tmp_path / "platform-store.pem"
    bundle_path.write_text(_ca_pair() * 5, encoding="utf-8")
    environ = {"SSL_CERT_FILE": str(bundle_path)}

    status = configure_langfuse_trust(environ=environ)

    assert status.valid is True
    assert status.source == "environment_trust_store"
    assert status.certificate_count == 10
    assert environ["REQUESTS_CA_BUNDLE"] == str(bundle_path)


def test_materialized_child_trust_ignores_unavailable_parent_profile(tmp_path) -> None:
    bundle_path = tmp_path / "bundle.pem"
    bundle_path.write_text(_ca_pair(), encoding="utf-8")
    environ = {
        "LANGFUSE_TRUST_MATERIALIZED": "true",
        "LANGFUSE_TLS_PROFILE": "parent-only-profile",
        "TLS_PROFILES_REF": "env://PARENT_ONLY_TLS_CATALOG",
        "REQUESTS_CA_BUNDLE": str(bundle_path),
        "SSL_CERT_FILE": str(bundle_path),
    }
    parent_config = SimpleNamespace(langfuse_tls_profile="parent-only-profile")

    status = configure_langfuse_trust(
        environ=environ,
        agent_config=parent_config,
    )

    assert status.valid is True
    assert status.certificate_count == 2


def test_large_requests_ca_store_uses_platform_validation(tmp_path) -> None:
    store_path = tmp_path / "store.pem"
    store_path.write_text(_ca_pair() * 5, encoding="utf-8")
    environ = {"REQUESTS_CA_BUNDLE": str(store_path)}

    status = configure_langfuse_trust(environ=environ)

    assert status.valid is True
    assert status.source == "environment_trust_store"
    assert status.certificate_count == 10


def test_explicit_mcp_bundle_is_validated_before_child_start(tmp_path) -> None:
    incomplete_path = tmp_path / "incomplete.pem"
    incomplete_path.write_text(
        _ca_pair().split("-----END CERTIFICATE-----", 1)[0], encoding="utf-8"
    )

    credential_env, values = _credential_refs()
    with pytest.raises(LangfuseTrustError, match="langfuse_ca_bundle_invalid"):
        prepare_langfuse_mcp_config(
            {
                "command": "langfuse-mcp",
                "env": {
                    "LANGFUSE_HOST": "https://telemetry.example.test",
                    **credential_env,
                    "REQUESTS_CA_BUNDLE": str(incomplete_path),
                },
            },
            environ={},
            resolver=values.get,
        )


def test_mcp_launcher_placeholders_resolve_from_runtime_environment(tmp_path) -> None:
    bundle_path = tmp_path / "bundle.pem"
    bundle_path.write_text(_ca_pair(), encoding="utf-8")
    credential_env, values = _credential_refs()
    environ = {
        "LANGFUSE_HOST": "https://telemetry.example.test",
        **credential_env,
        "REQUESTS_CA_BUNDLE": str(bundle_path),
    }

    config = prepare_langfuse_mcp_config(
        {
            "command": "langfuse-mcp",
            "env": {
                "LANGFUSE_HOST": "${LANGFUSE_HOST}",
                "LANGFUSE_PUBLIC_KEY_REF": "${LANGFUSE_PUBLIC_KEY_REF}",
                "LANGFUSE_SECRET_KEY_REF": "${LANGFUSE_SECRET_KEY_REF}",
            },
        },
        environ=environ,
        resolver=values.get,
    )

    assert config["env"]["LANGFUSE_HOST"] == environ["LANGFUSE_HOST"]
    assert config["env"]["LANGFUSE_PUBLIC_KEY"] == "synthetic-public"
    assert config["env"]["LANGFUSE_SECRET_KEY"] == "synthetic-secret"


def test_mcp_launcher_projects_explicit_child_policy_flags() -> None:
    credential_env, values = _credential_refs()

    config = prepare_langfuse_mcp_config(
        {"command": "langfuse-mcp", "env": credential_env},
        environ={
            "LANGFUSE_CAPTURE_CONTENT": "false",
            "LANGFUSE_KG_AUTO_INGEST": "true",
        },
        agent_config=SimpleNamespace(
            langfuse_capture_content=True,
            langfuse_kg_auto_ingest=False,
        ),
        resolver=values.get,
    )

    assert config["env"]["LANGFUSE_CAPTURE_CONTENT"] == "false"
    assert config["env"]["LANGFUSE_KG_AUTO_INGEST"] == "false"
    assert langfuse_parent_kg_ingestion_enabled(config) is True


def test_mcp_launcher_rejects_forged_parent_ingestion_marker() -> None:
    credential_env, values = _credential_refs()

    with pytest.raises(LangfuseTrustError, match="langfuse_configuration_invalid"):
        prepare_langfuse_mcp_config(
            {
                "command": "langfuse-mcp",
                "_graphos_parent_kg_ingestion": True,
                "env": credential_env,
            },
            environ={"LANGFUSE_KG_AUTO_INGEST": "false"},
            resolver=values.get,
        )


def test_mcp_launcher_rejects_invalid_child_policy_flag() -> None:
    credential_env, values = _credential_refs()

    with pytest.raises(LangfuseTrustError, match="langfuse_configuration_invalid"):
        prepare_langfuse_mcp_config(
            {"command": "langfuse-mcp", "env": credential_env},
            environ={"LANGFUSE_CAPTURE_CONTENT": "sometimes"},
            resolver=values.get,
        )


def test_mcp_launcher_fails_closed_for_unresolved_host_or_credentials() -> None:
    with pytest.raises(LangfuseTrustError, match="langfuse_host_invalid"):
        prepare_langfuse_mcp_config(
            {
                "command": "langfuse-mcp",
                "env": {"LANGFUSE_HOST": "telemetry.example.test"},
            },
            environ={},
        )

    with pytest.raises(LangfuseTrustError, match="langfuse_credentials_missing"):
        prepare_langfuse_mcp_config(
            {
                "command": "langfuse-mcp",
                "env": {"LANGFUSE_HOST": "https://telemetry.example.test"},
            },
            environ={},
        )


def test_mcp_launcher_uses_canonical_default_host_with_credential_refs() -> None:
    credential_env, values = _credential_refs()

    config = prepare_langfuse_mcp_config(
        {"command": "langfuse-mcp", "env": credential_env},
        environ={},
        resolver=values.get,
    )

    assert config["env"]["LANGFUSE_HOST"] == "https://cloud.langfuse.com"


def test_mcp_launcher_rejects_noncanonical_host_input() -> None:
    with pytest.raises(LangfuseTrustError, match="langfuse_host_invalid"):
        prepare_langfuse_mcp_config(
            {
                "command": "langfuse-mcp",
                "env": {
                    "LANGFUSE_BASE_URL": "https://telemetry.example.test",
                },
            },
            environ={},
        )


@pytest.mark.parametrize(
    "host",
    [
        "http://telemetry.example.test",
        "http://10.0.0.8:3000",
        "http://192.168.1.8:3000",
        "http://localhost.example.test:3000",
        "https://user:password@telemetry.example.test",
        "https://telemetry.example.test?project=private",
    ],
)
def test_mcp_launcher_rejects_insecure_explicit_child_host(host) -> None:
    with pytest.raises(LangfuseTrustError, match="langfuse_host_invalid"):
        prepare_langfuse_mcp_config(
            {
                "command": "langfuse-mcp",
                "env": {"LANGFUSE_HOST": host},
            },
            environ={},
        )


def test_mcp_launcher_allows_tightly_bounded_loopback_http() -> None:
    credential_env, values = _credential_refs()
    config = prepare_langfuse_mcp_config(
        {
            "command": "langfuse-mcp",
            "env": {
                "LANGFUSE_HOST": "http://127.0.0.1:3000",
                **credential_env,
            },
        },
        environ={},
        resolver=values.get,
    )

    assert config["env"]["LANGFUSE_HOST"] == "http://127.0.0.1:3000"


def test_secret_ref_materializes_runtime_bundle_without_reporting_path(
    tmp_path,
) -> None:
    environ = {"LANGFUSE_CA_BUNDLE_REF": "vault://observability/ca-bundle"}

    status = configure_langfuse_trust(
        environ=environ,
        resolver=lambda _ref: _ca_pair(),
        destination_root=tmp_path,
    )

    assert status.valid is True
    assert status.source == "secret_ref"
    assert "vault://" not in repr(status)
    assert environ["REQUESTS_CA_BUNDLE"] == environ["SSL_CERT_FILE"]
    assert list((tmp_path / "transport-security").glob("tls-ca-*.pem"))


def test_native_mcp_config_is_in_memory_and_carries_complete_trust(
    tmp_path, monkeypatch
) -> None:
    bundle_path = tmp_path / "bundle.pem"
    bundle_path.write_text(_ca_pair(), encoding="utf-8")
    environ = {
        "LANGFUSE_MCP_ENABLED": "true",
        "LANGFUSE_HOST": "https://telemetry.example.test",
        "LANGFUSE_PUBLIC_KEY_REF": "env://TEST_LANGFUSE_PUBLIC",
        "LANGFUSE_SECRET_KEY_REF": "env://TEST_LANGFUSE_SECRET",
        "LANGFUSE_PERSISTENCE_HMAC_KEY_REF": "env://TEST_LANGFUSE_PERSISTENCE_HMAC",
        "TEST_LANGFUSE_PUBLIC": "synthetic-public",
        "TEST_LANGFUSE_SECRET": "synthetic-secret",
        "TEST_LANGFUSE_PERSISTENCE_HMAC": "synthetic-persistence-key-material-32",
        "REQUESTS_CA_BUNDLE": str(bundle_path),
    }
    _patch_current_provider(monkeypatch)

    config = native_langfuse_mcp_config(environ=environ)

    assert config is not None
    assert config["command"]
    assert config["args"] == ["-m", "langfuse_agent.mcp_server"]
    assert config["env"]["LANGFUSE_HOST"] == environ["LANGFUSE_HOST"]
    assert config["env"]["LANGFUSE_PUBLIC_KEY"] == "synthetic-public"
    assert "LANGFUSE_PERSISTENCE_HMAC_KEY_REF" not in config["env"]
    assert config["env"]["LANGFUSE_PERSISTENCE_HMAC_KEY"] == (
        "synthetic-persistence-key-material-32"
    )
    assert config["env"]["LANGFUSE_PERSISTENCE_HMAC_MATERIALIZED"] == "true"
    assert config["env"]["REQUESTS_CA_BUNDLE"] == str(bundle_path)
    assert "UV_NATIVE_TLS" not in config["env"]
    assert config["env"]["LANGFUSE_TRUST_MATERIALIZED"] == "true"
    assert config["timeout"] == 120


def test_native_mcp_honors_explicit_opt_out_even_when_credentials_exist(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.find_spec", lambda name: object()
    )
    environ = {
        "LANGFUSE_MCP_ENABLED": "false",
        "LANGFUSE_HOST": "https://telemetry.example.test",
        "LANGFUSE_PUBLIC_KEY_REF": "env://TEST_LANGFUSE_PUBLIC",
        "LANGFUSE_SECRET_KEY_REF": "env://TEST_LANGFUSE_SECRET",
    }

    assert native_langfuse_mcp_config(environ=environ) is None


def test_mcp_launcher_rejects_invalid_persistence_hmac_key_ref() -> None:
    credential_env, values = _credential_refs()

    with pytest.raises(
        LangfuseTrustError, match="langfuse_persistence_hmac_key_invalid"
    ):
        prepare_langfuse_mcp_config(
            {
                "command": "langfuse-mcp",
                "env": {
                    **credential_env,
                    "LANGFUSE_PERSISTENCE_HMAC_KEY_REF": "plaintext-key",
                },
            },
            environ={},
            resolver=values.get,
        )


def test_native_mcp_auto_enables_when_credentials_are_ready(monkeypatch) -> None:
    _patch_current_provider(monkeypatch)
    environ = {
        "LANGFUSE_PUBLIC_KEY_REF": "env://TEST_LANGFUSE_PUBLIC",
        "LANGFUSE_SECRET_KEY_REF": "env://TEST_LANGFUSE_SECRET",
        "TEST_LANGFUSE_PUBLIC": "synthetic-public",
        "TEST_LANGFUSE_SECRET": "synthetic-secret",
    }

    config = native_langfuse_mcp_config(environ=environ)

    assert config is not None
    assert config["env"]["LANGFUSE_HOST"] == "https://cloud.langfuse.com"


def test_explicit_environment_mapping_resolves_without_ambient_or_backend(
    tmp_path, monkeypatch
) -> None:
    bundle_path = tmp_path / "bundle.pem"
    bundle_path.write_text(_ca_pair(), encoding="utf-8")
    for name in (
        "TEST_LANGFUSE_PUBLIC",
        "TEST_LANGFUSE_SECRET",
        "TEST_LANGFUSE_PERSISTENCE_HMAC",
    ):
        monkeypatch.delenv(name, raising=False)
    durable_resolver = MagicMock()
    durable_resolver.side_effect = AssertionError(
        "env refs must not initialize a durable secret backend"
    )
    environ = {
        "LANGFUSE_HOST": "https://telemetry.example.test",
        "LANGFUSE_PUBLIC_KEY_REF": "env://TEST_LANGFUSE_PUBLIC",
        "LANGFUSE_SECRET_KEY_REF": "env://TEST_LANGFUSE_SECRET",
        "LANGFUSE_PERSISTENCE_HMAC_KEY_REF": "env://TEST_LANGFUSE_PERSISTENCE_HMAC",
        "TEST_LANGFUSE_PUBLIC": "synthetic-public",
        "TEST_LANGFUSE_SECRET": "synthetic-secret",
        "TEST_LANGFUSE_PERSISTENCE_HMAC": "synthetic-persistence-key-material-32",
        "REQUESTS_CA_BUNDLE": str(bundle_path),
    }

    config = prepare_langfuse_mcp_config(
        {"command": "langfuse-mcp"},
        environ=environ,
        resolver=durable_resolver,
    )

    durable_resolver.assert_not_called()
    assert config["env"]["LANGFUSE_PUBLIC_KEY"] == "synthetic-public"
    assert config["env"]["LANGFUSE_SECRET_KEY"] == "synthetic-secret"
    assert config["env"]["LANGFUSE_PERSISTENCE_HMAC_KEY"] == (
        "synthetic-persistence-key-material-32"
    )


def test_native_mcp_requires_installed_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.find_spec", lambda name: None
    )
    monkeypatch.setenv("TEST_LANGFUSE_PUBLIC", "synthetic-public")
    monkeypatch.setenv("TEST_LANGFUSE_SECRET", "synthetic-secret")
    environ = {
        "LANGFUSE_HOST": "https://telemetry.example.test",
        "LANGFUSE_PUBLIC_KEY_REF": "env://TEST_LANGFUSE_PUBLIC",
        "LANGFUSE_SECRET_KEY_REF": "env://TEST_LANGFUSE_SECRET",
    }

    assert native_langfuse_mcp_config(environ=environ) is None


def test_native_mcp_rejects_stale_provider_at_launcher_boundary(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.find_spec",
        lambda _name: object(),
    )
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.package_version",
        lambda _name: "0.29.0",
    )
    environ = {
        "LANGFUSE_PUBLIC_KEY_REF": "env://TEST_LANGFUSE_PUBLIC",
        "LANGFUSE_SECRET_KEY_REF": "env://TEST_LANGFUSE_SECRET",
        "TEST_LANGFUSE_PUBLIC": "synthetic-public",
        "TEST_LANGFUSE_SECRET": "synthetic-secret",
    }

    assert native_langfuse_mcp_config(environ=environ) is None


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.29.0", False),
        ("1.0.1", False),
        ("1.0.2", False),
        ("1.0.3", True),
        ("1.9.0", True),
        ("2.0.0", False),
        ("not-a-version", False),
    ],
)
def test_native_mcp_requires_current_provider_contract(
    monkeypatch, version: str, expected: bool
) -> None:
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.find_spec",
        lambda _name: object(),
    )
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.package_version",
        lambda _name: version,
    )

    assert langfuse_provider_contract_ready() is expected


def test_native_mcp_requires_runtime_posture_surface(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.find_spec",
        lambda name: object() if name == "langfuse_agent.mcp_server" else None,
    )
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.package_version",
        lambda _name: "1.0.3",
    )

    assert langfuse_provider_contract_ready() is False


def test_mcp_materializes_credential_and_tls_profile_refs(tmp_path) -> None:
    profile_ref = "vault://observability/tls-profile"
    public_ref = "vault://observability/public-key"
    secret_ref = "vault://observability/secret-key"
    values = {
        profile_ref: json.dumps({"ca_bundle_pem": _ca_pair()}),
        public_ref: "synthetic-public",
        secret_ref: "synthetic-" + "secret",
    }
    config = prepare_langfuse_mcp_config(
        {"command": "langfuse-mcp"},
        environ={
            "LANGFUSE_HOST": "https://telemetry.example.test",
            "LANGFUSE_PUBLIC_KEY_REF": public_ref,
            "LANGFUSE_SECRET_KEY_REF": secret_ref,
            "LANGFUSE_TLS_PROFILE_REF": profile_ref,
        },
        resolver=values.get,
        destination_root=tmp_path,
    )

    assert config["env"]["LANGFUSE_PUBLIC_KEY"] == "synthetic-public"
    assert config["env"]["LANGFUSE_SECRET_KEY"] == "synthetic-secret"
    assert "LANGFUSE_PUBLIC_KEY_REF" not in config["env"]
    assert "LANGFUSE_SECRET_KEY_REF" not in config["env"]
    assert "LANGFUSE_TLS_PROFILE_REF" not in config["env"]
    assert config["env"]["REQUESTS_CA_BUNDLE"] == config["env"]["SSL_CERT_FILE"]


def test_mcp_projects_ref_backed_mtls_into_requests_transport(tmp_path) -> None:
    credential_env, credential_values = _credential_refs()
    cert_pem, key_pem = _client_pair()
    cert_ref = "secret://observability/client-cert"
    key_ref = "secret://observability/client-key"
    values = {
        **credential_values,
        cert_ref: cert_pem,
        key_ref: key_pem,
    }

    config = prepare_langfuse_mcp_config(
        {
            "command": "langfuse-mcp",
            "env": {
                **credential_env,
                "LANGFUSE_CLIENT_CERT_REF": cert_ref,
                "LANGFUSE_CLIENT_KEY_REF": key_ref,
            },
        },
        environ={},
        resolver=values.get,
        destination_root=tmp_path / "parent",
    )

    child_env = config["env"]
    assert "LANGFUSE_CLIENT_CERT_REF" not in child_env
    assert "LANGFUSE_CLIENT_KEY_REF" not in child_env
    assert child_env["LANGFUSE_CLIENT_CERT"] == child_env["LANGFUSE_CLIENT_KEY"]
    request_kwargs = resolve_langfuse_requests_transport(
        environ=child_env,
        destination_root=tmp_path / "child",
    )
    assert request_kwargs["verify"] is True
    assert isinstance(request_kwargs["cert"], str)
