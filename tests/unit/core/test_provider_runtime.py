from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.core.config import AgentConfig
from agent_utilities.core.provider_runtime import (
    ProviderRuntimeError,
    get_provider_runtime_profile,
    prepare_provider_runtime_child_environment,
    prepare_resolved_provider_runtime_child_environment,
    resolve_provider_runtime_profile,
    resolve_selected_provider_runtime_profile,
)


def _config(*, enabled: bool = True) -> AgentConfig:
    return AgentConfig(
        PROVIDER_CONFIGS={
            "synthetic-provider": {
                "enabled": enabled,
                "endpoint_ref": "env://PROVIDER_ENDPOINT",
                "credential_refs": {"PROVIDER_TOKEN": "env://PROVIDER_TOKEN_REFERENCE"},
                "selector_refs": {"PROVIDER_SCOPE": "env://PROVIDER_SCOPE_REFERENCE"},
                "tls_profile_ref": "env://PROVIDER_TLS_REFERENCE",
            }
        }
    )


def test_provider_runtime_resolves_ephemerally_and_cleans_tls(monkeypatch) -> None:
    values = {
        "env://PROVIDER_ENDPOINT": "https://provider.example.test/api",
        "env://PROVIDER_TOKEN_REFERENCE": "synthetic-runtime-token",
        "env://PROVIDER_SCOPE_REFERENCE": "read-only",
    }
    monkeypatch.setattr(
        "agent_utilities.security.cli_secrets.resolve_runtime_secret_reference",
        lambda reference: values[reference],
    )
    cleaned: list[bool] = []
    trust = SimpleNamespace(cleanup=lambda: cleaned.append(True), verify_enabled=True)
    monkeypatch.setattr(
        "agent_utilities.core.transport_security.resolve_configured_tls_profile",
        lambda *_args, **_kwargs: trust,
    )

    with resolve_provider_runtime_profile(
        "synthetic-provider", config=_config()
    ) as runtime:
        assert runtime.endpoint == "https://provider.example.test/api"
        assert runtime.credentials == {"PROVIDER_TOKEN": "synthetic-runtime-token"}
        assert runtime.selectors == {"PROVIDER_SCOPE": "read-only"}
        assert repr(runtime) == "<ResolvedProviderRuntime redacted>"

    assert cleaned == [True]
    assert runtime.endpoint is None
    assert runtime.credentials == {}
    assert runtime.selectors == {}
    assert runtime.tls is None


def test_provider_child_environment_is_selected_and_reference_scoped(
    monkeypatch,
) -> None:
    values = {
        "env://PROVIDER_ENDPOINT": "https://provider.example.test/api",
        "env://PROVIDER_TOKEN_REFERENCE": "synthetic-runtime-token",
        "env://PROVIDER_SCOPE_REFERENCE": "read-only",
        "env://PROVIDER_TLS_REFERENCE": '{"system_trust":true}',
    }
    monkeypatch.setattr(
        "agent_utilities.security.cli_secrets.resolve_runtime_secret_reference",
        lambda reference: values[reference],
    )
    cleaned: list[bool] = []
    trust = SimpleNamespace(
        cleanup=lambda: cleaned.append(True),
        system_trust=True,
        trust_env=False,
        ca_bundle_path=None,
        ca_directory=None,
        client_cert_path=None,
        client_key_path=None,
        client_key_password=None,
        proxy_url=None,
        no_proxy=None,
    )
    monkeypatch.setattr(
        "agent_utilities.core.transport_security.resolve_configured_tls_profile",
        lambda *_args, **_kwargs: trust,
    )
    monkeypatch.setenv("VAULT_TOKEN", "unrelated-global-authority")

    prepared = prepare_provider_runtime_child_environment(
        "synthetic-provider", config=_config()
    )
    projected = dict(prepared.environment)

    assert projected["AGENT_PROVIDER_PROFILE"] == "synthetic-provider"
    assert (
        projected["AGENT_PROVIDER_RUNTIME_ENDPOINT"]
        == values["env://PROVIDER_ENDPOINT"]
    )
    assert (
        projected["AGENT_PROVIDER_RUNTIME_CREDENTIAL_00"]
        == values["env://PROVIDER_TOKEN_REFERENCE"]
    )
    assert (
        projected["AGENT_PROVIDER_RUNTIME_SELECTOR_00"]
        == values["env://PROVIDER_SCOPE_REFERENCE"]
    )
    assert "VAULT_TOKEN" not in projected
    assert set(
        AgentConfig(PROVIDER_CONFIGS=projected["PROVIDER_CONFIGS"]).provider_configs
    ) == {"synthetic-provider"}
    prepared.close()
    assert prepared.environment == {}
    assert cleaned == [True]


def test_provider_child_environment_rejects_process_oversized_value(
    monkeypatch,
) -> None:
    def resolve(reference: str) -> str:
        if reference == "env://PROVIDER_ENDPOINT":
            return "https://provider.example.test/api"
        if reference == "env://PROVIDER_TOKEN_REFERENCE":
            return "x" * 100_000
        return "read-only"

    cleaned: list[bool] = []
    monkeypatch.setattr(
        "agent_utilities.security.cli_secrets.resolve_runtime_secret_reference",
        resolve,
    )
    monkeypatch.setattr(
        "agent_utilities.core.transport_security.resolve_configured_tls_profile",
        lambda *_args, **_kwargs: SimpleNamespace(
            cleanup=lambda: cleaned.append(True),
            system_trust=True,
            trust_env=False,
            ca_bundle_path=None,
            ca_directory=None,
            client_cert_path=None,
            client_key_path=None,
            client_key_password=None,
            proxy_url=None,
            no_proxy=None,
        ),
    )

    with pytest.raises(
        ProviderRuntimeError, match="provider_child_environment_invalid"
    ):
        prepare_provider_runtime_child_environment(
            "synthetic-provider", config=_config()
        )

    assert cleaned == [True]


def test_already_resolved_runtime_projects_without_second_resolution(
    monkeypatch,
) -> None:
    runtime = SimpleNamespace(
        _closed=False,
        endpoint="https://provider.example.test/api",
        credentials={},
        selectors={"PROVIDER_SCOPE": "read-only"},
        tls=None,
        close=lambda: None,
    )

    prepared = prepare_resolved_provider_runtime_child_environment(
        "synthetic-provider",
        runtime,
    )

    projected = dict(prepared.environment)
    assert projected["AGENT_PROVIDER_RUNTIME_ENDPOINT"] == runtime.endpoint
    assert projected["AGENT_PROVIDER_RUNTIME_SELECTOR_00"] == "read-only"
    prepared.close()


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://provider.example.test/api",
        "https://user:password@provider.example.test/api",
        "https://provider.example.test/api?tenant=private",
        "https://provider.example.test/api#private",
        "ftp://provider.example.test/api",
    ],
)
def test_provider_runtime_rejects_insecure_or_credentialed_endpoints(
    monkeypatch, endpoint
) -> None:
    def resolve(reference: str) -> str:
        if reference == "env://PROVIDER_ENDPOINT":
            return endpoint
        return "synthetic-runtime-value"

    monkeypatch.setattr(
        "agent_utilities.security.cli_secrets.resolve_runtime_secret_reference",
        resolve,
    )

    with pytest.raises(ProviderRuntimeError, match="provider_endpoint_invalid"):
        resolve_provider_runtime_profile("synthetic-provider", config=_config())


def test_provider_runtime_allows_exact_loopback_http(monkeypatch) -> None:
    def resolve(reference: str) -> str:
        if reference == "env://PROVIDER_ENDPOINT":
            return "http://127.0.0.1:8080/api"
        return "synthetic-runtime-value"

    monkeypatch.setattr(
        "agent_utilities.security.cli_secrets.resolve_runtime_secret_reference",
        resolve,
    )
    monkeypatch.setattr(
        "agent_utilities.core.transport_security.resolve_configured_tls_profile",
        lambda *_args, **_kwargs: SimpleNamespace(cleanup=lambda: None),
    )

    with resolve_provider_runtime_profile(
        "synthetic-provider", config=_config()
    ) as runtime:
        assert runtime.endpoint == "http://127.0.0.1:8080/api"


def test_provider_runtime_rejects_disabled_missing_and_invalid_names() -> None:
    with pytest.raises(ProviderRuntimeError, match="provider_profile_disabled"):
        get_provider_runtime_profile(
            "synthetic-provider", config=_config(enabled=False)
        )
    with pytest.raises(ProviderRuntimeError, match="provider_profile_unavailable"):
        get_provider_runtime_profile("missing-provider", config=_config())
    with pytest.raises(ProviderRuntimeError, match="provider_profile_invalid"):
        get_provider_runtime_profile("INVALID", config=_config())


def test_provider_runtime_fails_closed_without_disclosing_resolved_value(
    monkeypatch,
) -> None:
    private_value = "private-runtime-material"
    monkeypatch.setattr(
        "agent_utilities.security.cli_secrets.resolve_runtime_secret_reference",
        lambda _reference: private_value + "\n",
    )

    with pytest.raises(ProviderRuntimeError) as caught:
        resolve_provider_runtime_profile("synthetic-provider", config=_config())

    assert private_value not in str(caught.value)


def test_selected_provider_runtime_requires_explicit_launcher_selection(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "agent_utilities.core.provider_runtime.resolve_provider_runtime_profile",
        lambda profile_name, **_kwargs: profile_name,
    )
    monkeypatch.setattr(
        "agent_utilities.core.config.setting",
        lambda name: "synthetic-provider" if name == "AGENT_PROVIDER_PROFILE" else None,
    )
    assert resolve_selected_provider_runtime_profile() == "synthetic-provider"

    monkeypatch.setattr("agent_utilities.core.config.setting", lambda _name: None)
    with pytest.raises(ProviderRuntimeError, match="provider_profile_not_selected"):
        resolve_selected_provider_runtime_profile()
