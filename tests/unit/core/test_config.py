"""CONCEPT:AU-OS.safety.doom-loop-detection"""

import os
import sys
from pathlib import Path

import pytest

from agent_utilities.core.config import (
    PRODUCTION_CERTIFICATION_SCENARIOS,
    AgentConfig,
)


def test_placement_control_loop_is_typed_and_opt_in(monkeypatch):
    monkeypatch.delenv("PLACEMENT_CONTROL_LOOP_ENABLED", raising=False)
    assert AgentConfig().placement_control_loop_enabled is False

    monkeypatch.setenv("PLACEMENT_CONTROL_LOOP_ENABLED", "true")
    assert AgentConfig().placement_control_loop_enabled is True


def test_dispatch_lease_config_is_bounded_by_recovery_objective():
    config = AgentConfig()
    assert config.agent_dispatch_renew_interval_s < config.agent_dispatch_claim_ttl_s
    assert config.agent_dispatch_claim_ttl_s <= 300.0

    with pytest.raises(ValueError):
        AgentConfig(AGENT_DISPATCH_CLAIM_TTL_S=301)

    with pytest.raises(ValueError):
        AgentConfig(AGENT_DISPATCH_RENEW_INTERVAL_S=31)


@pytest.mark.parametrize(
    "key",
    [
        "ENGINE_" + "MODE",
        "ENGINE_" + "ENDPOINT",
        "EPISTEMIC_GRAPH_" + "AUTOSTART",
        "EPISTEMIC_GRAPH_" + "ENCRYPTION_KEY",
        "GRAPH_SERVICE_" + "SOCKET",
        "GRAPH_SERVICE_TCP_" + "ADDR",
        "GRAPH_DIRECT_" + "EXECUTION",
        "GRAPH_" + "BACKEND",
        "GRAPH_" + "AUTHORITY",
        "MESSAGING_" + "REACTIONS",
        "PERMISSIONS_SIGNING_" + "KEY",
    ],
)
def test_agent_config_rejects_retired_configuration_keys(key, monkeypatch):
    monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="retired durable configuration"):
        AgentConfig(**{key: "retired"})

    monkeypatch.setenv(key, "retired")
    with pytest.raises(ValueError, match="retired durable configuration"):
        AgentConfig()


def test_agent_config_has_no_graph_authority_selector_fields():
    fields = AgentConfig.model_fields
    assert "graph_" + "backend" not in fields
    assert "graph_" + "authority" not in fields


def test_reactions_uses_the_current_typed_key(monkeypatch):
    monkeypatch.delenv("REACTIONS", raising=False)
    assert AgentConfig().reactions == "1"

    monkeypatch.setenv("REACTIONS", "0")
    assert AgentConfig().reactions == "0"


@pytest.mark.parametrize(
    "key",
    [
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_PUBLIC_KEY",
        "OTEL_EXPORTER_OTLP_SECRET_KEY",
    ],
)
def test_agent_config_rejects_raw_durable_otel_auth(key, monkeypatch):
    monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="retired durable configuration"):
        AgentConfig(**{key: "raw-material"})


@pytest.mark.parametrize(
    "key",
    [
        "OIDC_CLIENT_SECRET",
        "MCP_BASIC_AUTH_PASSWORD",
    ],
)
def test_agent_config_rejects_raw_durable_outbound_auth(key, monkeypatch):
    monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="retired durable configuration"):
        AgentConfig(**{key: "raw-material"})


def test_engine_encryption_config_accepts_only_external_runtime_references():
    environment = AgentConfig(
        EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF="env://TEST_ENGINE_DATA_KEY"
    )
    vault = AgentConfig(EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF="vault://engine/data-key")

    assert (
        environment.epistemic_graph_encryption_key_ref == "env://TEST_ENGINE_DATA_KEY"
    )
    assert vault.epistemic_graph_encryption_key_ref == "vault://engine/data-key"

    with pytest.raises(ValueError, match="external runtime secret ref"):
        AgentConfig(EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF="secret://engine/data-key")
    with pytest.raises(ValueError, match="retired durable configuration"):
        AgentConfig(EPISTEMIC_GRAPH_ENCRYPTION_KEY="raw-material")


def test_agent_config_types_complete_outbound_oidc_declaration():
    config = AgentConfig(
        MCP_CLIENT_AUTH="oidc-client-credentials",
        OIDC_CLIENT_ID="fleet-client",
        OIDC_CLIENT_SECRET_REF="env://TEST_FLEET_CLIENT_SECRET",
        OIDC_AUDIENCE="graph-api",
        OIDC_ISSUER="HTTPS://identity.example.test/issuer/",
        OIDC_SCOPE="graph.read graph.write",
    )

    assert config.mcp_client_auth == "oidc-client-credentials"
    assert config.oidc_client_id == "fleet-client"
    assert config.oidc_client_secret_ref == "env://TEST_FLEET_CLIENT_SECRET"
    assert config.oidc_audience == "graph-api"
    assert config.oidc_issuer == "https://identity.example.test/issuer"
    assert config.oidc_token_url is None
    assert config.oidc_scope == "graph.read graph.write"


def test_agent_config_types_exact_skill_certification_references(tmp_path):
    material = [
        str(tmp_path / name) for name in ("config", "profile", "spec", "promotion")
    ]
    command = [str(Path(sys.executable).absolute()), "--version"]
    config = AgentConfig(
        SKILL_CERT_RUNTIME_CONFIGURATION=material[0],
        SKILL_CERT_RUNTIME_PROFILE=material[1],
        SKILL_CERT_RELEASE_SPEC=material[2],
        SKILL_CERT_PROMOTION_EVIDENCE=material[3],
        SKILL_CERT_GRAPHOS_ENDPOINT="http://localhost:8000/mcp",
        SKILL_CERT_GRAPHOS_COMMAND=command,
        SKILL_VALIDATION_EVIDENCE_SIGNER_COMMAND=command,
        SKILL_VALIDATION_EVIDENCE_VERIFIER_COMMAND=command,
    )

    rendered = config.model_dump(by_alias=True)
    assert rendered["SKILL_CERT_RUNTIME_CONFIGURATION"] == material[0]
    assert rendered["SKILL_CERT_RUNTIME_PROFILE"] == material[1]
    assert rendered["SKILL_CERT_RELEASE_SPEC"] == material[2]
    assert rendered["SKILL_CERT_PROMOTION_EVIDENCE"] == material[3]
    assert rendered["SKILL_CERT_GRAPHOS_ENDPOINT"] == "http://localhost:8000/mcp"
    assert rendered["SKILL_CERT_GRAPHOS_COMMAND"] == command
    assert rendered["SKILL_VALIDATION_EVIDENCE_SIGNER_COMMAND"] == command
    assert rendered["SKILL_VALIDATION_EVIDENCE_VERIFIER_COMMAND"] == command
    assert rendered["SKILL_CERT_IDENTITY_AUTHORITY_MODE"] == (
        "ephemeral-https-loopback"
    )
    assert rendered["SKILL_CERT_IDENTITY_TOKEN_TTL_SECONDS"] == 300


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("SKILL_CERT_RUNTIME_CONFIGURATION", "relative/config.json"),
        ("SKILL_CERT_RUNTIME_PROFILE", "/bounded/../profile.json"),
        ("SKILL_CERT_GRAPHOS_ENDPOINT", "https://example.invalid/mcp"),
        ("SKILL_CERT_GRAPHOS_ENDPOINT", "http://localhost/mcp?expanded=true"),
        ("SKILL_CERT_GRAPHOS_COMMAND", ["graph-os"]),
        ("SKILL_VALIDATION_EVIDENCE_SIGNER_COMMAND", ["/bin/tool\narg"]),
        ("SKILL_CERT_IDENTITY_AUTHORITY_MODE", "external-oidc"),
        ("SKILL_CERT_IDENTITY_TOKEN_TTL_SECONDS", 179),
    ],
)
def test_agent_config_rejects_structurally_invalid_skill_certification_fields(
    field, value
):
    with pytest.raises(ValueError):
        AgentConfig(**{field: value})


def test_agent_config_types_complete_production_certification_contract(tmp_path):
    command = [str(Path(sys.executable).absolute()), "--version"]
    command_map = {scenario: command for scenario in PRODUCTION_CERTIFICATION_SCENARIOS}
    config = AgentConfig(
        CERTIFICATION_MODE="production",
        CERT_RELEASE_MANIFEST=str(tmp_path / "release.json"),
        CERT_ARTIFACTS_DIR=str(tmp_path / "artifacts"),
        CERT_HARDWARE_CLASS="capacity-standard",
        CERT_LOAD_COMMAND=[*command, "--report", "{report_file}"],
        CERT_METRICS_COMMAND=command,
        CERT_HOOK_COMMANDS=command_map,
        CERT_FAULT_ACTION_COMMANDS=command_map,
        CERT_FAULT_PROBE_COMMANDS=command_map,
        CERT_EVIDENCE_SIGNER_COMMAND=command,
        CERT_EVIDENCE_VERIFIER_COMMAND=command,
        CERT_PROMETHEUS_URL="HTTPS://metrics.example.test/prometheus/",
        CERT_PROMETHEUS_BEARER_TOKEN_REF="env://TEST_CERT_TOKEN",
        CERT_PROMETHEUS_TLS_PROFILE_REF="vault://tls/prometheus",
    )

    rendered = config.model_dump(by_alias=True)
    assert rendered["CERTIFICATION_MODE"] == "production"
    assert rendered["CERT_RELEASE_MANIFEST"] == str(tmp_path / "release.json")
    assert rendered["CERT_ARTIFACTS_DIR"] == str(tmp_path / "artifacts")
    assert rendered["CERT_HARDWARE_CLASS"] == "capacity-standard"
    assert tuple(rendered["CERT_HOOK_COMMANDS"]) == (PRODUCTION_CERTIFICATION_SCENARIOS)
    assert set(rendered["CERT_FAULT_ACTION_COMMANDS"]) == set(
        PRODUCTION_CERTIFICATION_SCENARIOS
    )
    assert set(rendered["CERT_FAULT_PROBE_COMMANDS"]) == set(
        PRODUCTION_CERTIFICATION_SCENARIOS
    )
    assert rendered["CERT_PROMETHEUS_URL"] == (
        "https://metrics.example.test/prometheus"
    )
    assert rendered["CERT_PROMETHEUS_BEARER_TOKEN_REF"] == ("env://TEST_CERT_TOKEN")
    assert rendered["CERT_PROMETHEUS_TLS_PROFILE_REF"] == ("vault://tls/prometheus")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("CERTIFICATION_MODE", "local"),
        ("CERT_RELEASE_MANIFEST", "relative/release.json"),
        ("CERT_ARTIFACTS_DIR", "/private/../artifacts"),
        ("CERT_HARDWARE_CLASS", "developer-laptop"),
        ("CERT_LOAD_COMMAND", ["python", "load.py"]),
        ("CERT_HOOK_COMMANDS", {"unknown-scenario": ["/bin/true"]}),
        ("CERT_PROMETHEUS_URL", "http://metrics.example.test"),
        ("CERT_PROMETHEUS_URL", "https://user:password@example.test"),
        ("CERT_PROMETHEUS_BEARER_TOKEN_REF", "raw-token"),
        ("CERT_PROMETHEUS_TLS_PROFILE_REF", "/private/ca.pem"),
        ("CERT_PROMETHEUS_TLS_PROFILE", "invalid profile"),
    ],
)
def test_agent_config_rejects_invalid_production_certification_fields(field, value):
    with pytest.raises(ValueError):
        AgentConfig(**{field: value})


_RETIRED_CERTIFICATION_KEYS = [
    "CERT_PROMETHEUS_BEARER_TOKEN_FILE",
    "CERT_HOOK_IDENTITY_TLS_TRACE",
    "CERT_HOOK_KILL_COMMIT_PHASE",
    "CERT_HOOK_WORKER_LOSS",
    "CERT_HOOK_RAFT_LEADER_LOSS",
    "CERT_HOOK_BROKER_LEADER_LOSS",
    "CERT_HOOK_NODE_LOSS",
    "CERT_HOOK_ZONE_ISOLATION",
    "CERT_HOOK_BROKER_REBALANCE",
    "CERT_HOOK_ONLINE_RESHARD",
    "CERT_HOOK_ATOMIC_RELEASE_CUTOVER",
    "CERT_HOOK_INDEX_MIGRATION",
    "CERT_HOOK_ONTOLOGY_MIGRATION",
    "CERT_HOOK_BACKUP_RESTORE",
    "CERT_HOOK_REGIONAL_RECOVERY",
    "CERT_HOOK_POLICY_DELETION",
    *[
        f"CERT_{operation}_{scenario.upper().replace('-', '_')}"
        for operation in ("ACTION", "PROBE")
        for scenario in PRODUCTION_CERTIFICATION_SCENARIOS
    ],
]


@pytest.mark.parametrize("key", _RETIRED_CERTIFICATION_KEYS)
def test_agent_config_rejects_retired_production_certification_keys(key):
    with pytest.raises(ValueError, match="retired durable configuration"):
        AgentConfig(**{key: "retired"})


def test_agent_config_accepts_one_complete_otel_reference_source():
    headers = AgentConfig(OTEL_EXPORTER_OTLP_HEADERS_REF="env://TEST_OTEL_HEADERS")
    assert headers.otel_exporter_otlp_headers_ref == "env://TEST_OTEL_HEADERS"

    pair = AgentConfig(
        OTEL_EXPORTER_OTLP_PUBLIC_KEY_REF="env://TEST_OTEL_PUBLIC",
        OTEL_EXPORTER_OTLP_SECRET_KEY_REF="env://TEST_OTEL_SECRET",
    )
    assert pair.otel_exporter_otlp_public_key_ref
    assert pair.otel_exporter_otlp_secret_key_ref


def test_agent_config_rejects_ambiguous_otel_reference_sources():
    with pytest.raises(ValueError, match="one OTLP authentication source"):
        AgentConfig(
            OTEL_EXPORTER_OTLP_HEADERS_REF="env://TEST_OTEL_HEADERS",
            OTEL_EXPORTER_OTLP_PUBLIC_KEY_REF="env://TEST_OTEL_PUBLIC",
            OTEL_EXPORTER_OTLP_SECRET_KEY_REF="env://TEST_OTEL_SECRET",
        )


@pytest.mark.concept("CONCEPT:AU-OS.safety.doom-loop-detection")
def test_agent_config_defaults():
    orig_host = os.environ.pop("HOST", None)
    orig_port = os.environ.pop("PORT", None)
    try:
        config = AgentConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.routing_strategy == "hybrid"
        assert config.tool_guard_mode == "strict"
    finally:
        if orig_host is not None:
            os.environ["HOST"] = orig_host
        if orig_port is not None:
            os.environ["PORT"] = orig_port


@pytest.mark.concept("CONCEPT:AU-OS.safety.doom-loop-detection")
def test_agent_config_overrides():
    os.environ["HOST"] = "1.2.3.4"
    os.environ["PORT"] = "8080"
    try:
        config = AgentConfig()
        assert config.host == "1.2.3.4"
        assert config.port == 8080
    finally:
        os.environ.pop("HOST", None)
        os.environ.pop("PORT", None)


def test_runtime_integration_settings_are_typed_and_normalized():
    config = AgentConfig(
        INFRA_INVENTORY_PATH="  synthetic-inventory.yml  ",
        FLEET_MCP_URL_TEMPLATE=("HTTPS://gateway.example.test/services/{server}/mcp/"),
        COMFYUI_URL="HTTPS://media.example.test/comfy/",
        SVD_URL="http://video.example.test:8188/api/",
    )

    assert config.infra_inventory_path == "synthetic-inventory.yml"
    assert config.fleet_mcp_url_template == (
        "https://gateway.example.test/services/{server}/mcp"
    )
    assert config.comfyui_url == "https://media.example.test/comfy"
    assert config.svd_url == "http://video.example.test:8188/api"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("INFRA_INVENTORY_PATH", "invalid\x00path"),
        ("FLEET_MCP_URL_TEMPLATE", "https://gateway.example.test/mcp"),
        (
            "FLEET_MCP_URL_TEMPLATE",
            "https://gateway.example.test/{{server}}/mcp",
        ),
        ("COMFYUI_URL", "file:///runtime/media.sock"),
        ("XTTS_URL", "https://user:password@example.test"),
        ("WHISPER_URL", "https://media.example.test/api?tenant=private"),
    ],
)
def test_runtime_integration_settings_reject_unsafe_formats(key, value):
    with pytest.raises(ValueError):
        AgentConfig(**{key: value})


@pytest.mark.concept("CONCEPT:AU-OS.safety.doom-loop-detection")
def test_agent_config_has_no_dotenv_source():
    assert AgentConfig.model_config.get("env_file") is None


@pytest.mark.concept("CONCEPT:AU-KG.query.vendor-agnostic-traversal")
def test_kg_loop_flags_default_off():
    # The golden-loop family is opt-in: every KG_GOLDEN_* flag now lives on
    # AgentConfig (off bare os.environ) and defaults to the conservative value.
    for k in (
        "KG_LOOP",
        "KG_LOOP_DISTILL",
        "KG_LOOP_BREADTH",
        "KG_LOOP_STANDARDIZE",
        "KG_GOLDEN_AUTO_MERGE",
    ):
        os.environ.pop(k, None)
    c = AgentConfig()
    assert c.kg_loop is False
    assert c.kg_golden_auto_merge is False
    assert c.kg_loop_interval == 3600.0
    assert c.kg_loop_topics == 5
    assert c.kg_golden_merge_threshold is None


@pytest.mark.concept("CONCEPT:AU-KG.query.vendor-agnostic-traversal")
def test_kg_loop_override_from_env():
    os.environ["KG_LOOP"] = "1"
    os.environ["KG_LOOP_TOPICS"] = "9"
    try:
        c = AgentConfig()
        assert c.kg_loop is True
        assert c.kg_loop_topics == 9
    finally:
        os.environ.pop("KG_LOOP", None)
        os.environ.pop("KG_LOOP_TOPICS", None)


@pytest.mark.concept("CONCEPT:EG-KG.storage.nonblocking-checkpoint")
def test_kg_dev_mode_default_off():
    # Production default: background daemons are on (dev mode off). This single
    # switch replaced the per-daemon KG_*_DAEMON env toggles.
    os.environ.pop("KG_DEV_MODE", None)
    assert AgentConfig().kg_dev_mode is False


@pytest.mark.concept("CONCEPT:EG-KG.storage.nonblocking-checkpoint")
def test_kg_dev_mode_override_from_env():
    os.environ["KG_DEV_MODE"] = "true"
    try:
        assert AgentConfig().kg_dev_mode is True
    finally:
        os.environ.pop("KG_DEV_MODE", None)


@pytest.mark.concept("CONCEPT:EG-KG.storage.nonblocking-checkpoint")
def test_kg_dev_mode_helper_reads_config(monkeypatch):
    # The engine's daemon gate reads the SAME typed config source of truth, so
    # all KG background daemons collapse behind this one switch.
    from agent_utilities.core import config as cfg_mod
    from agent_utilities.knowledge_graph.core import engine_tasks

    monkeypatch.setattr(cfg_mod.config, "kg_dev_mode", False, raising=False)
    assert engine_tasks._kg_dev_mode() is False
    monkeypatch.setattr(cfg_mod.config, "kg_dev_mode", True, raising=False)
    assert engine_tasks._kg_dev_mode() is True


@pytest.mark.concept("CONCEPT:AU-OS.safety.doom-loop-detection")
def test_tool_guard_mode_override():
    os.environ["TOOL_GUARD_MODE"] = "on"
    try:
        config = AgentConfig()
        assert config.tool_guard_mode == "on"
    finally:
        os.environ.pop("TOOL_GUARD_MODE", None)


@pytest.mark.concept("CONCEPT:AU-OS.safety.doom-loop-detection")
def test_tool_guard_mode_rejects_disabled_values(monkeypatch):
    monkeypatch.setenv("TOOL_GUARD_MODE", "off")
    with pytest.raises(ValueError):
        AgentConfig()


def test_permissions_signing_authority_accepts_only_runtime_reference() -> None:
    with pytest.raises(ValueError):
        AgentConfig(PERMISSIONS_SIGNING_KEY_REF="literal-material")

    config = AgentConfig(
        PERMISSIONS_SIGNING_KEY_REF="env://PERMISSIONS_RUNTIME_TEST_KEY"
    )
    assert config.permissions_signing_key_ref == "env://PERMISSIONS_RUNTIME_TEST_KEY"


def test_ontology_release_signer_is_typed_and_reference_only() -> None:
    with pytest.raises(ValueError, match="runtime-only material"):
        AgentConfig(ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF="literal-material")

    config = AgentConfig(
        ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF=(
            "env://ONTOLOGY_RELEASE_SIGNING_TEST_KEY"
        ),
        ONTOLOGY_RELEASE_TRUSTED_PUBLIC_KEYS="public-key-one,public-key-two",
    )
    assert config.ontology_release_signing_private_key_ref == (
        "env://ONTOLOGY_RELEASE_SIGNING_TEST_KEY"
    )
    assert config.ontology_release_trusted_public_keys == (
        "public-key-one,public-key-two"
    )


def test_mcp_fleet_secret_refs_accept_only_neutral_runtime_mappings() -> None:
    config = AgentConfig(
        MCP_FLEET_SECRET_REFS={
            "CHILD_ACCESS_TOKEN": "secret://fleet/child/access-token",
            "CHILD_CLIENT_SECRET": "vault://fleet/child/client-secret",
            "CHILD_SIGNING_KEY": "env://CHILD_SIGNING_KEY",
        }
    )

    assert config.mcp_fleet_secret_refs == {
        "CHILD_ACCESS_TOKEN": "secret://fleet/child/access-token",
        "CHILD_CLIENT_SECRET": "vault://fleet/child/client-secret",
        "CHILD_SIGNING_KEY": "env://CHILD_SIGNING_KEY",
    }


@pytest.mark.parametrize(
    "mapping",
    [
        {"lowercase_alias": "secret://fleet/child/token"},
        {" CHILD_TOKEN": "secret://fleet/child/token"},
        {"CHILD_TOKEN": "literal-secret"},
        {"CHILD_TOKEN": "https://secret.invalid/value"},
        {"CHILD_TOKEN": "env://lowercase_source"},
        {"CHILD_TOKEN": "secret://fleet/../token"},
    ],
)
def test_mcp_fleet_secret_refs_reject_invalid_aliases_and_references(mapping) -> None:
    with pytest.raises(ValueError):
        AgentConfig(MCP_FLEET_SECRET_REFS=mapping)


def test_mcp_fleet_secret_refs_parse_strict_json() -> None:
    config = AgentConfig(
        MCP_FLEET_SECRET_REFS=('{"CHILD_TOKEN":"secret://fleet/child/token"}')
    )
    assert config.mcp_fleet_secret_refs == {"CHILD_TOKEN": "secret://fleet/child/token"}

    with pytest.raises(ValueError):
        AgentConfig(
            MCP_FLEET_SECRET_REFS=(
                '{"CHILD_TOKEN":"secret://fleet/one",'
                '"CHILD_TOKEN":"secret://fleet/two"}'
            )
        )


def test_provider_configs_are_reference_only_and_native_vector_is_default() -> None:
    config = AgentConfig(
        PROVIDER_CONFIGS={
            "synthetic-provider": {
                "enabled": True,
                "endpoint_ref": "env://SYNTHETIC_PROVIDER_ENDPOINT",
                "credential_refs": {
                    "SYNTHETIC_PROVIDER_TOKEN": "secret://providers/token"
                },
                "selector_refs": {
                    "SYNTHETIC_PROVIDER_SCOPE": "vault://providers/scope"
                },
                "tls_profile_ref": "secret://providers/tls",
            }
        }
    )

    profile = config.provider_configs["synthetic-provider"]
    assert config.vector_database_type == "epistemic_graph"
    assert profile.endpoint_ref == "env://SYNTHETIC_PROVIDER_ENDPOINT"
    assert profile.credential_refs == {
        "SYNTHETIC_PROVIDER_TOKEN": "secret://providers/token"
    }
    assert profile.selector_refs == {
        "SYNTHETIC_PROVIDER_SCOPE": "vault://providers/scope"
    }


@pytest.mark.parametrize(
    "profile",
    [
        {"enabled": True},
        {
            "enabled": True,
            "endpoint_ref": "https://provider.invalid",
            "tls_profile": "system",
        },
        {
            "enabled": True,
            "endpoint_ref": "env://PROVIDER_ENDPOINT",
        },
        {
            "enabled": True,
            "credential_refs": {"provider_token": "secret://provider/token"},
        },
        {
            "enabled": True,
            "credential_refs": {"PROVIDER_TOKEN": "literal-material"},
        },
        {
            "enabled": True,
            "credential_refs": {"PROVIDER_VALUE": "secret://provider/value"},
            "selector_refs": {"PROVIDER_VALUE": "secret://provider/selector"},
        },
        {
            "enabled": True,
            "selector_refs": {"PROVIDER_SCOPE": "secret://provider/scope"},
            "tls_profile": "named",
            "tls_profile_ref": "secret://provider/tls",
        },
    ],
)
def test_provider_configs_reject_ambiguous_or_literal_runtime_material(
    profile,
) -> None:
    with pytest.raises(ValueError):
        AgentConfig(PROVIDER_CONFIGS={"synthetic-provider": profile})


def test_provider_configs_parse_strict_json_and_reject_duplicate_profiles() -> None:
    config = AgentConfig(
        PROVIDER_CONFIGS=(
            '{"synthetic-provider":{"enabled":true,'
            '"credential_refs":{"PROVIDER_TOKEN":"env://PROVIDER_TOKEN"}}}'
        )
    )
    assert config.provider_configs["synthetic-provider"].enabled is True

    with pytest.raises(ValueError):
        AgentConfig(
            PROVIDER_CONFIGS=(
                '{"synthetic-provider":{"enabled":false},'
                '"synthetic-provider":{"enabled":false}}'
            )
        )


@pytest.mark.concept("CONCEPT:AU-OS.safety.doom-loop-detection")
def test_sensitive_tool_patterns_defaults():
    config = AgentConfig()
    assert isinstance(config.sensitive_tool_patterns, list)
    assert r".*delete.*" in config.sensitive_tool_patterns
    assert r".*rm_.*" in config.sensitive_tool_patterns


@pytest.mark.concept("CONCEPT:AU-OS.safety.doom-loop-detection")
def test_agent_config_langfuse_host_is_canonical(monkeypatch):
    monkeypatch.setenv("LANGFUSE_HOST", "HTTPS://canonical.invalid/")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://source.invalid")
    monkeypatch.setenv("LANGFUSE_URL", "https://legacy.invalid")

    assert AgentConfig().langfuse_host == "https://canonical.invalid"


@pytest.mark.parametrize(
    "host",
    [
        "http://127.0.0.1:3000",
        "http://localhost:3000/langfuse",
        "http://[::1]:3000",
    ],
)
def test_agent_config_langfuse_host_allows_exact_loopback_http(host):
    assert AgentConfig(LANGFUSE_HOST=host).langfuse_host == host


@pytest.mark.parametrize(
    "host",
    [
        "http://langfuse.example.test",
        "http://192.0.2.8:3000",
        "http://198.51.100.8:3000",
        "http://localhost.example.test:3000",
        "ftp://langfuse.example.test",
        "https://user:password@langfuse.example.test",
        "https://langfuse.example.test?project=private",
        "https://langfuse.example.test/#fragment",
    ],
)
def test_agent_config_langfuse_host_rejects_insecure_or_noncanonical_urls(host):
    with pytest.raises(ValueError):
        AgentConfig(LANGFUSE_HOST=host)


@pytest.mark.concept("CONCEPT:AU-OS.safety.doom-loop-detection")
def test_agent_config_ignores_noncanonical_langfuse_host_inputs(monkeypatch):
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://source.invalid")
    monkeypatch.setenv("LANGFUSE_URL", "https://legacy.invalid")

    assert AgentConfig().langfuse_host == "https://cloud.langfuse.com"


@pytest.mark.concept("CONCEPT:AU-OS.safety.doom-loop-detection")
def test_resolve_langfuse_host_reads_only_canonical_input(monkeypatch):
    from agent_utilities.core.config import resolve_langfuse_host

    monkeypatch.setenv("LANGFUSE_HOST", "https://canonical.invalid")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://source.invalid")
    monkeypatch.setenv("LANGFUSE_URL", "https://legacy.invalid")
    assert resolve_langfuse_host() == "https://canonical.invalid"

    monkeypatch.delenv("LANGFUSE_HOST")
    assert resolve_langfuse_host() == "https://cloud.langfuse.com"


@pytest.mark.concept("CONCEPT:AU-OS.safety.doom-loop-detection")
def test_lazy_module_level_getattr():
    from agent_utilities.core.config import (
        DEFAULT_HOST,
        DEFAULT_LLM_PROVIDER,
        DEFAULT_PORT,
    )

    assert DEFAULT_HOST == "127.0.0.1"
    assert DEFAULT_PORT == 9000
    assert DEFAULT_LLM_PROVIDER == "openai" or DEFAULT_LLM_PROVIDER is not None


_AUTO_FLAG_ENV = (
    "AUTH_JWT_ISSUER",
    "AUTH_JWT_JWKS_URI",
    "KG_FUSEKI_PUBLISH",
    "KG_FUSEKI_ENDPOINT",
)


def _clear_auto_flag_env() -> None:
    for key in _AUTO_FLAG_ENV:
        os.environ.pop(key, None)


@pytest.mark.concept("CONCEPT:AU-OS.identity.authenticated-identity-enforcement")
def test_zero_infra_default_auto_enables_nothing():
    """No dependency configured keeps optional Fuseki publishing off."""
    _clear_auto_flag_env()
    try:
        config = AgentConfig()
        assert config.kg_fuseki_publish is False
    finally:
        _clear_auto_flag_env()


def test_plaintext_langfuse_environment_does_not_enable_integrations(
    monkeypatch,
):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "runtime-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "runtime-secret")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY_REF", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY_REF", raising=False)
    monkeypatch.delenv("LANGFUSE_MCP_ENABLED", raising=False)
    monkeypatch.delenv("KG_FAILURE_EVOLUTION", raising=False)
    config = AgentConfig()
    assert config.langfuse_mcp_enabled is False
    assert config.kg_failure_evolution is False
    assert config.langfuse_capture_content is False
    assert config.kg_agent_auto_apply is False
    assert config.kg_golden_auto_merge is False


def test_langfuse_dependency_auto_enable_remains_explicitly_opt_out(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY_REF", "env://TEST_LANGFUSE_PUBLIC")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY_REF", "env://TEST_LANGFUSE_SECRET")
    monkeypatch.setenv("LANGFUSE_MCP_ENABLED", "false")
    monkeypatch.setenv("KG_FAILURE_EVOLUTION", "false")
    config = AgentConfig()
    assert config.langfuse_mcp_enabled is False
    assert config.kg_failure_evolution is False


def test_langfuse_secret_refs_enable_metadata_mcp_and_evolution(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setenv(
        "LANGFUSE_PUBLIC_KEY_REF", "vault://observability/langfuse-public"
    )
    monkeypatch.setenv(
        "LANGFUSE_SECRET_KEY_REF", "vault://observability/langfuse-secret"
    )
    monkeypatch.setenv(
        "LANGFUSE_PERSISTENCE_HMAC_KEY_REF",
        "vault://observability/langfuse-persistence-hmac",
    )
    monkeypatch.setenv("LANGFUSE_KG_AUTO_INGEST", "true")
    monkeypatch.setenv("LANGFUSE_TLS_PROFILE_REF", "vault://observability/langfuse-tls")
    monkeypatch.delenv("LANGFUSE_MCP_ENABLED", raising=False)
    monkeypatch.delenv("KG_FAILURE_EVOLUTION", raising=False)

    config = AgentConfig()

    assert config.langfuse_mcp_enabled is True
    assert config.kg_failure_evolution is True
    assert config.langfuse_tls_profile_ref.endswith("langfuse-tls")
    assert config.langfuse_persistence_hmac_key_ref.endswith(
        "langfuse-persistence-hmac"
    )
    assert config.langfuse_kg_auto_ingest is True
    assert config.langfuse_capture_content is False


@pytest.mark.parametrize(
    "field",
    [
        "LANGFUSE_PUBLIC_KEY_REF",
        "LANGFUSE_SECRET_KEY_REF",
        "LANGFUSE_PERSISTENCE_HMAC_KEY_REF",
        "LANGFUSE_TLS_PROFILE_REF",
        "LANGFUSE_CA_BUNDLE_REF",
        "LANGFUSE_CLIENT_CERT_REF",
        "LANGFUSE_CLIENT_KEY_REF",
        "LANGFUSE_CLIENT_KEY_PASSWORD_REF",
        "LANGFUSE_PROXY_URL_REF",
    ],
)
def test_langfuse_runtime_fields_reject_legacy_sqlite_refs(field):
    with pytest.raises(ValueError, match="runtime-only material"):
        AgentConfig(**{field: "sqlite://legacy/material"})


def test_google_workspace_oauth_runtime_fields_are_typed_and_https_only(monkeypatch):
    monkeypatch.setenv("GOOGLE_WORKSPACE_OAUTH_CLIENT_ID", "synthetic-client.apps")
    monkeypatch.setenv(
        "GOOGLE_WORKSPACE_OAUTH_BROKER_URL", "https://oauth-broker.example.test/"
    )
    config = AgentConfig()
    assert config.google_workspace_oauth_client_id == "synthetic-client.apps"
    assert (
        config.google_workspace_oauth_broker_url == "https://oauth-broker.example.test"
    )

    monkeypatch.setenv(
        "GOOGLE_WORKSPACE_OAUTH_BROKER_URL", "http://oauth-broker.example.test"
    )
    with pytest.raises(ValueError, match="must use HTTPS"):
        AgentConfig()


def test_evolution_staging_root_is_runtime_typed_and_bounded(monkeypatch):
    monkeypatch.setenv("EVOLUTION_STAGING_ROOT", "runtime-staging")
    assert AgentConfig().evolution_staging_root == "runtime-staging"

    with pytest.raises(ValueError, match="EVOLUTION_STAGING_ROOT"):
        AgentConfig(EVOLUTION_STAGING_ROOT="invalid\x00root")


@pytest.mark.concept("CONCEPT:AU-KG.ontology.authoritative-tbox")
def test_fuseki_publish_auto_enables_when_endpoint_configured():
    """Configuring a Fuseki endpoint engages KG_FUSEKI_PUBLISH by default."""
    _clear_auto_flag_env()
    os.environ["KG_FUSEKI_ENDPOINT"] = "https://fuseki.example.test/ds"
    try:
        assert AgentConfig().kg_fuseki_publish is True
    finally:
        _clear_auto_flag_env()
