"""CLI credentials are resolved from references and never accepted in argv."""

from __future__ import annotations

import argparse
import sys

import pytest

from agent_utilities.agent.factory import create_agent_parser
from agent_utilities.mcp.server_factory import create_mcp_parser
from agent_utilities.security import cli as secret_cli
from agent_utilities.security.cli_secrets import (
    RuntimeSecretReferenceError,
    resolve_runtime_secret_reference,
    validate_runtime_secret_reference,
)


def _options(parser: argparse.ArgumentParser) -> set[str]:
    return {
        option
        for action in parser._actions  # noqa: SLF001 - inspect the public CLI contract
        for option in action.option_strings
    }


def test_runtime_reference_resolves_inherited_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_RUNTIME_CREDENTIAL", "runtime-only-value")
    assert (
        resolve_runtime_secret_reference("env://TEST_RUNTIME_CREDENTIAL")
        == "runtime-only-value"
    )


def test_runtime_reference_can_be_validated_without_resolution() -> None:
    assert (
        validate_runtime_secret_reference("secret://providers/runtime/token")
        == "secret://providers/runtime/token"
    )
    with pytest.raises(RuntimeSecretReferenceError):
        validate_runtime_secret_reference("literal-material")


@pytest.mark.parametrize(
    "reference",
    [
        "plaintext-secret",
        "sqlite://old/credential",
        "file:///private/credential",
        "env://MISSING VALUE",
        "vault://../credential",
    ],
)
def test_runtime_reference_rejects_non_current_or_unsafe_inputs(reference: str) -> None:
    with pytest.raises(RuntimeSecretReferenceError) as captured:
        resolve_runtime_secret_reference(reference)
    assert reference not in str(captured.value)


def test_secret_manager_set_accepts_only_value_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys, "argv", ["secret-manager", "set", "service/key", "literal"])
        secret_cli.main()

    writes: list[tuple[str, str]] = []

    class _Backend:
        pass

    class _Client:
        backend = _Backend()

        def set(self, key: str, value: str) -> None:
            writes.append((key, value))

    monkeypatch.setenv("TEST_SECRET_MANAGER_VALUE", "runtime-value")
    monkeypatch.setattr(secret_cli, "create_secrets_client", lambda _config: _Client())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "secret-manager",
            "set",
            "service/key",
            "--value-ref",
            "env://TEST_SECRET_MANAGER_VALUE",
        ],
    )

    secret_cli.main()

    assert writes == [("service/key", "runtime-value")]


def test_agent_cli_exposes_only_reference_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", "resolved-llm-key")
    monkeypatch.setenv("TEST_OTEL_HEADERS", "Authorization=runtime")
    monkeypatch.setenv("TEST_OTEL_PUBLIC", "resolved-otel-public")
    monkeypatch.setenv("TEST_OTEL_SECRET", "resolved-otel-key")
    parser = create_agent_parser()
    argv = [
        "--api-key-ref",
        "env://TEST_LLM_KEY",
        "--otel-headers-ref",
        "env://TEST_OTEL_HEADERS",
        "--otel-public-key-ref",
        "env://TEST_OTEL_PUBLIC",
        "--otel-secret-key-ref",
        "env://TEST_OTEL_SECRET",
    ]
    args = parser.parse_args(argv)

    assert args.api_key == "resolved-llm-key"
    assert args.otel_headers == "Authorization=runtime"
    assert args.otel_public_key == "resolved-otel-public"
    assert args.otel_secret_key == "resolved-otel-key"
    assert "resolved-llm-key" not in " ".join(argv)
    assert "resolved-otel-key" not in " ".join(argv)
    assert {
        "--api-key",
        "--otel-headers",
        "--otel-public-key",
        "--otel-secret-key",
    }.isdisjoint(_options(parser))


def test_mcp_cli_exposes_only_reference_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "TEST_OAUTH_SECRET": "oauth-runtime",
        "TEST_OIDC_SECRET": "oidc-runtime",
        "TEST_OPENAPI_PASSWORD": "openapi-password-runtime",
        "TEST_OPENAPI_SECRET": "openapi-oauth-runtime",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    parser = create_mcp_parser()
    argv = [
        "--oauth-upstream-client-secret-ref",
        "env://TEST_OAUTH_SECRET",
        "--oidc-client-secret-ref",
        "env://TEST_OIDC_SECRET",
        "--openapi-password-ref",
        "env://TEST_OPENAPI_PASSWORD",
        "--openapi-client-secret-ref",
        "env://TEST_OPENAPI_SECRET",
    ]
    args = parser.parse_args(argv)

    assert args.oauth_upstream_client_secret == values["TEST_OAUTH_SECRET"]
    assert args.oidc_client_secret == values["TEST_OIDC_SECRET"]
    assert args.openapi_password == values["TEST_OPENAPI_PASSWORD"]
    assert args.openapi_client_secret == values["TEST_OPENAPI_SECRET"]
    for value in values.values():
        assert value not in " ".join(argv)
    assert {
        "--token-secret",
        "--oauth-upstream-client-secret",
        "--oidc-client-secret",
        "--openapi-password",
        "--openapi-client-secret",
    }.isdisjoint(_options(parser))


def test_mcp_environment_defaults_resolve_secret_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_DEFAULT_OIDC_SECRET", "oidc-default-runtime")
    monkeypatch.setenv("TEST_DEFAULT_OAUTH_SECRET", "oauth-default-runtime")
    monkeypatch.setenv("TEST_DEFAULT_OPENAPI_PASSWORD", "openapi-password-runtime")
    monkeypatch.setenv("TEST_DEFAULT_OPENAPI_SECRET", "openapi-secret-runtime")
    monkeypatch.setenv(
        "OIDC_CLIENT_SECRET_REF", "env://TEST_DEFAULT_OIDC_SECRET"
    )
    monkeypatch.setenv(
        "OAUTH_UPSTREAM_CLIENT_SECRET_REF", "env://TEST_DEFAULT_OAUTH_SECRET"
    )
    monkeypatch.setenv(
        "OPENAPI_PASSWORD_REF", "env://TEST_DEFAULT_OPENAPI_PASSWORD"
    )
    monkeypatch.setenv(
        "OPENAPI_CLIENT_SECRET_REF", "env://TEST_DEFAULT_OPENAPI_SECRET"
    )

    args = create_mcp_parser().parse_args([])

    assert args.oidc_client_secret == "oidc-default-runtime"
    assert args.oauth_upstream_client_secret == "oauth-default-runtime"
    assert args.openapi_password == "openapi-password-runtime"
    assert args.openapi_client_secret == "openapi-secret-runtime"
