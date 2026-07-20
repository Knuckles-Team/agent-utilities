#!/usr/bin/python
"""Tests for the production profile guard (core/profile_guard.py)."""

import pytest

from agent_utilities.core.config import AgentConfig
from agent_utilities.core.profile_guard import (
    PROFILE_ENV_VAR,
    ProductionProfileError,
    assert_production_safe,
    collect_production_violations,
    is_production_profile,
)


def _make_config(**overrides):
    # Build a base config and override the relevant attributes after construction.
    # (pydantic-settings sources can shadow init kwargs; direct mutation is the
    # reliable way to pin the exact attribute values the guard inspects.)
    cfg = AgentConfig()
    # Hermetic baseline: the guard inspects these connection fields, and
    # pydantic-settings would otherwise source them from a leaked os.environ (test
    # order pollution — e.g. a DSN set by an earlier test). Reset to None unless a
    # test overrides them, so each scenario is pinned regardless of run order.
    for field in ("graph_db_connection_profile_ref", "graph_mirror_targets"):
        if field not in overrides:
            setattr(cfg, field, None)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _toy_config(**overrides):
    base = dict(
        graph_persistence_type="file",
        a2a_broker="in-memory",
        a2a_storage="in-memory",
        agent_bus_log_backend=None,
        # Keep this fixture focused on the 13 durability/observability failures;
        # identity bypasses have their own parameterized production tests below.
        auth_jwt_jwks_uri="https://identity.invalid/jwks",
        auth_jwt_issuer="https://identity.invalid/",
        auth_jwt_audience="agent-services",
        kg_policy_version="policy-v1",
        # kafka_bootstrap_servers now defaults to this deployment's real
        # in-cluster broker (config.py's kafka_bootstrap_servers field), not
        # None — pin it explicitly here so this fixture still represents the
        # genuinely-unconfigured "toy"/zero-infra scenario the guard tests
        # below exercise.
        kafka_bootstrap_servers=None,
        usage_db_backend="sqlite",
        usage_tracking_enabled=True,
        usage_content_retention="metadata",
        langfuse_capture_content=False,
        persistence_identity_hmac_key_ref=None,
        permissions_signing_key_ref=None,
        epistemic_graph_max_resident_graphs=0,
        epistemic_graph_lazy_open_page_size=0,
        epistemic_graph_max_nodes_per_graph=0,
        enable_otel=False,
        otel_exporter_otlp_endpoint=None,
    )
    base.update(overrides)
    return _make_config(**base)


def _prod_config(**overrides):
    base = dict(
        graph_persistence_type="postgresql",
        a2a_broker="epistemic_graph",
        a2a_storage="epistemic_graph",
        agent_bus_log_backend="engine",
        kafka_bootstrap_servers="redpanda-0:9092,redpanda-1:9092",
        usage_db_backend="postgres",
        usage_tracking_enabled=True,
        usage_content_retention="metadata",
        langfuse_capture_content=False,
        persistence_identity_hmac_key_ref="env://PERSISTENCE_IDENTITY_HMAC_KEY",
        permissions_signing_key_ref="env://AGENT_PERMISSION_AUTHORITY",
        epistemic_graph_encryption_key_ref="env://ENGINE_DATA_KEY",
        auth_jwt_jwks_uri="https://identity.invalid/jwks",
        auth_jwt_issuer="https://identity.invalid/",
        auth_jwt_audience="agent-services",
        kg_policy_version="policy-v1",
        epistemic_graph_max_resident_graphs=1024,
        epistemic_graph_lazy_open_page_size=4096,
        epistemic_graph_max_nodes_per_graph=250_000,
        enable_otel=True,
        otel_exporter_otlp_endpoint="https://telemetry.invalid",
    )
    base.update(overrides)
    return _make_config(**base)


def test_unset_kafka_ledger_is_flagged_in_prod():
    # Plan 08 Synergy 2: prod requires a distributed reactive ledger.
    cfg = _prod_config(kafka_bootstrap_servers=None)
    violations = collect_production_violations(cfg)
    assert any("kafka_bootstrap_servers" in v for v in violations)


def test_is_production_profile_detection():
    assert is_production_profile("prod")
    assert is_production_profile("PROD")
    assert is_production_profile(" production ")
    assert not is_production_profile("dev")
    assert not is_production_profile("")
    assert not is_production_profile(None)


def test_prod_profile_with_file_persistence_fails():
    cfg = _toy_config()
    with pytest.raises(ProductionProfileError) as exc:
        assert_production_safe(cfg, profile="prod")
    # The error must list every offending setting (persistence, broker, storage,
    # the unset reactive ledger, and production observability/safety gates).
    assert len(exc.value.offending) >= 11
    joined = "\n".join(exc.value.offending)
    assert "graph_persistence_type" in joined
    assert "a2a_broker" in joined
    assert "agent_bus_log_backend" in joined
    assert "kafka_bootstrap_servers" in joined
    assert "epistemic_graph_max_resident_graphs" in joined
    assert "epistemic_graph_lazy_open_page_size" in joined
    assert "a2a_storage" in joined
    assert "usage_db_backend" in joined
    assert "persistence_identity_hmac_key_ref" in joined
    assert "permissions_signing_key_ref" in joined
    assert "OpenTelemetry" in joined


def test_sqlite_persistence_also_fails_under_prod():
    cfg = _prod_config(graph_persistence_type="sqlite")
    with pytest.raises(ProductionProfileError) as exc:
        assert_production_safe(cfg, profile="prod")
    assert any("graph_persistence_type" in o for o in exc.value.offending)
    assert len(exc.value.offending) == 1


def test_proper_prod_config_passes():
    cfg = _prod_config()
    # Should not raise.
    assert_production_safe(cfg, profile="prod")
    assert collect_production_violations(cfg) == []


def test_kafka_agent_bus_is_rejected_for_million_agent_production():
    cfg = _prod_config(agent_bus_log_backend="kafka")
    violations = collect_production_violations(cfg)
    assert any("agent_bus_log_backend" in value for value in violations)


def test_local_production_engine_requires_external_encryption_key_reference():
    violations = collect_production_violations(
        _prod_config(epistemic_graph_encryption_key_ref=None)
    )
    assert any("epistemic_graph_encryption_key_ref" in value for value in violations)

    remote = _prod_config(
        epistemic_graph_encryption_key_ref=None,
        graph_service_endpoints=["tls://engine.invalid:9100"],
    )
    assert collect_production_violations(remote) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("auth_jwt_jwks_uri", None),
        ("auth_jwt_issuer", None),
        ("auth_jwt_audience", None),
        ("kg_policy_version", None),
        ("permissions_signing_key_ref", None),
    ],
)
def test_identity_configuration_gaps_fail_under_prod(field, value):
    violations = collect_production_violations(_prod_config(**{field: value}))
    assert any(field in item for item in violations)


def test_dev_profile_unaffected_even_with_toy_settings():
    cfg = _toy_config()
    # No raise under dev / default profiles, even though settings are toy.
    assert_production_safe(cfg, profile="dev")
    assert_production_safe(cfg, profile="")
    assert_production_safe(cfg, profile=None)


def test_default_profile_via_env_is_unaffected(monkeypatch):
    monkeypatch.delenv(PROFILE_ENV_VAR, raising=False)
    cfg = _toy_config()
    assert_production_safe(cfg)  # reads APP_PROFILE -> unset -> no-op


def test_env_var_prod_triggers_guard(monkeypatch):
    monkeypatch.setenv(PROFILE_ENV_VAR, "prod")
    cfg = _toy_config()
    with pytest.raises(ProductionProfileError):
        assert_production_safe(cfg)


def test_config_method_delegates(monkeypatch):
    monkeypatch.setenv(PROFILE_ENV_VAR, "prod")
    with pytest.raises(ProductionProfileError):
        _toy_config().assert_production_safe()
    # proper prod config passes through the method too
    _prod_config().assert_production_safe()


def test_collect_violations_is_profile_independent():
    # collect_* always evaluates rules regardless of APP_PROFILE.
    assert len(collect_production_violations(_toy_config())) >= 12
    assert collect_production_violations(_prod_config()) == []


def test_tool_guard_cannot_be_disabled_under_prod():
    violations = collect_production_violations(_prod_config(tool_guard_mode="off"))
    assert any("tool_guard_mode" in value for value in violations)


def test_external_projection_declaration_does_not_change_prod_authority():
    cfg = _prod_config(graph_mirror_targets=["age"])
    assert_production_safe(cfg, profile="prod")
    assert collect_production_violations(cfg) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epistemic_graph_max_resident_graphs", 0),
        ("epistemic_graph_lazy_open_page_size", 0),
        ("epistemic_graph_max_nodes_per_graph", 0),
    ],
)
def test_unbounded_graph_lifecycle_setting_fails_under_prod(field, value):
    cfg = _prod_config(**{field: value})
    violations = collect_production_violations(cfg)
    assert any(field in item for item in violations)


def test_per_process_usage_store_fails_under_prod():
    cfg = _prod_config(usage_db_backend="sqlite")
    violations = collect_production_violations(cfg)
    assert any("usage_db_backend" in item for item in violations)


def test_incomplete_otel_path_fails_under_prod():
    cfg = _prod_config(enable_otel=False, otel_exporter_otlp_endpoint=None)
    violations = collect_production_violations(cfg)
    assert any("OpenTelemetry" in item for item in violations)


def test_disabled_usage_tracking_fails_under_prod():
    cfg = _prod_config(usage_tracking_enabled=False)
    violations = collect_production_violations(cfg)
    assert any("usage_tracking_enabled" in item for item in violations)


def test_observability_content_capture_fails_under_prod():
    cfg = _prod_config(
        usage_content_retention="sanitized", langfuse_capture_content=True
    )
    violations = collect_production_violations(cfg)
    assert any("usage_content_retention" in item for item in violations)
    assert any("langfuse_capture_content" in item for item in violations)


def test_missing_persistence_identity_key_fails_under_prod():
    cfg = _prod_config(persistence_identity_hmac_key_ref=None)
    violations = collect_production_violations(cfg)
    assert any("persistence_identity_hmac_key_ref" in item for item in violations)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("debug", True),
        ("allowed_hosts", "*"),
        ("allowed_origins", "https://safe.invalid,*"),
        ("mcp_allowed_hosts", "*.invalid"),
        ("server_tls_certfile", "/runtime/server.pem"),
        ("mcp_tls_keyfile", "/runtime/server.key"),
        ("messaging_alert_intake_allow_remote", True),
    ],
)
def test_explicit_production_escape_hatches_are_fatal(field, value):
    violations = collect_production_violations(_prod_config(**{field: value}))
    marker = field.split("_tls_")[0].lower()
    assert any(marker in item.lower() for item in violations)


def test_non_loopback_listener_requires_exact_host_and_tls_boundary():
    violations = collect_production_violations(_prod_config(host="0.0.0.0"))
    assert any("allowed_hosts" in item for item in violations)
    assert any("direct TLS" in item for item in violations)

    cfg = _prod_config(
        host="0.0.0.0",
        allowed_hosts="graph.example.invalid",
        server_tls_terminated=True,
    )
    assert collect_production_violations(cfg) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("auth_jwt_jwks_uri", "http://identity.example.invalid/jwks"),
        ("mcp_url", "http://graph.example.invalid/mcp"),
        ("openai_base_url", "http://models.example.invalid/v1"),
        ("otel_exporter_otlp_endpoint", "http://telemetry.example.invalid:4318"),
    ],
)
def test_remote_plaintext_http_is_fatal_in_production(field, value):
    violations = collect_production_violations(_prod_config(**{field: value}))
    assert any(field in item and "plaintext HTTP" in item for item in violations)


def test_loopback_plaintext_http_remains_valid_for_process_local_transport():
    cfg = _prod_config(
        openai_base_url="http://127.0.0.1:8000/v1",
        mcp_url="http://localhost:9000/mcp",
    )
    assert collect_production_violations(cfg) == []


def test_alert_intake_is_tokenized_and_loopback_only_in_production():
    cfg = _prod_config(
        messaging_alert_intake_port=9100,
        messaging_alert_intake_host="0.0.0.0",
        messaging_alert_intake_token_ref=None,
    )
    violations = collect_production_violations(cfg)
    assert any("token reference" in item for item in violations)
    assert any("loopback" in item for item in violations)


def test_lazy_runtime_config_invokes_production_guard(monkeypatch):
    import importlib

    config_module = importlib.import_module("agent_utilities.core.config")

    class _UnsafeConfig:
        app_profile = "production"

        def assert_production_safe(self, *, profile=None):
            assert profile == "production"
            raise ProductionProfileError(["synthetic unsafe setting"], profile)

    monkeypatch.setattr(config_module, "_ensure_env_loaded", lambda: None)
    monkeypatch.setattr(config_module, "AgentConfig", _UnsafeConfig)
    config_module._LAZY_CACHE.clear()
    with pytest.raises(ProductionProfileError, match="synthetic unsafe setting"):
        config_module._init_lazy_config()
    assert "_config" not in config_module._LAZY_CACHE
