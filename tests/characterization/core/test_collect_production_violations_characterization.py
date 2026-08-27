#!/usr/bin/python
"""Characterization test for ``collect_production_violations`` (CX-AU-07).

Pins the OBSERVED behaviour of the pre-refactor function (CCN 98,
``agent_utilities/core/profile_guard.py``) before it is decomposed into
per-concern helper functions. This test must be green against the unmodified
function; if it is not, the test is wrong, not the code. It is not touched
again in the refactor commit.

Written against the exact append order of the current implementation,
including a deliberately toy/edge-case-heavy configuration so that the exact
list of violation strings — content AND order — is pinned, not just the
count. Order matters here because a refactor into helper functions grouped
by concern could silently interleave results differently from the original
top-to-bottom evaluation.
"""

from __future__ import annotations

from agent_utilities.core.config import AgentConfig
from agent_utilities.core.profile_guard import collect_production_violations


def _make_config(**overrides):
    cfg = AgentConfig()
    for field in (
        "graph_db_connection_profile_ref",
        "graph_mirror_targets",
        "graph_service_endpoints",
    ):
        if field not in overrides:
            setattr(cfg, field, None)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _fully_toy_config(**overrides):
    """Every rule violated at once, in the exact field-declaration order."""
    base = dict(
        graph_persistence_type="sqlite",
        auth_jwt_jwks_uri="",
        auth_jwt_issuer="",
        auth_jwt_audience="",
        kg_policy_version="",
        kg_loop_allow_host_validation=True,
        debug=True,
        tool_guard_mode="off",
        allowed_hosts="*",
        allowed_origins="*",
        cors_allow_credentials=True,
        server_tls_certfile="cert.pem",
        server_tls_keyfile="",
        server_tls_terminated=False,
        host="0.0.0.0",
        mcp_allowed_hosts="*",
        mcp_tls_certfile="",
        mcp_tls_keyfile="key.pem",
        messaging_alert_intake_allow_remote=True,
        messaging_alert_intake_port=9999,
        messaging_alert_intake_token_ref="",
        messaging_alert_intake_host="0.0.0.0",
        mcp_url="http://remote.invalid/mcp",
        a2a_broker="in-memory",
        agent_bus_log_backend="kafka",
        a2a_storage="in-memory",
        kafka_bootstrap_servers=None,
        epistemic_graph_max_resident_graphs=0,
        epistemic_graph_lazy_open_page_size=0,
        epistemic_graph_max_nodes_per_graph=0,
        graph_service_endpoints=None,
        epistemic_graph_encryption_key_ref="",
        usage_db_backend="sqlite",
        usage_tracking_enabled=False,
        usage_content_retention="full",
        langfuse_capture_content=True,
        langfuse_public_key_ref="pub-ref",
        langfuse_secret_key_ref=None,
        langfuse_mcp_enabled=True,
        langfuse_host="http://langfuse.invalid",
        persistence_identity_hmac_key_ref="",
        permissions_signing_key_ref="",
        secrets_backend="local-file",
        enable_otel=False,
        otel_exporter_otlp_endpoint="",
        # Pinned explicitly (rather than left at AgentConfig()'s own default,
        # which is itself environment-dependent -- see the hermeticity note
        # on _make_config in tests/scale/test_prod_profile_guard.py) so the
        # exact-order/content assertion below is deterministic regardless of
        # ambient env/session fixtures. The chat/embedding endpoint sweep is
        # separately pinned by its own dedicated tests below.
        chat_models=(),
        embedding_models=(),
        sparql_endpoints=(),
    )
    base.update(overrides)
    return _make_config(**base)


def _clean_prod_config(**overrides):
    base = dict(
        graph_persistence_type="postgresql",
        auth_jwt_jwks_uri="https://identity.invalid/jwks",
        auth_jwt_issuer="https://identity.invalid/",
        auth_jwt_audience="agent-services",
        kg_policy_version="policy-v1",
        kg_loop_allow_host_validation=False,
        debug=False,
        tool_guard_mode="strict",
        allowed_hosts="api.invalid",
        allowed_origins="https://ui.invalid",
        cors_allow_credentials=False,
        server_tls_certfile="cert.pem",
        server_tls_keyfile="key.pem",
        server_tls_terminated=False,
        host="0.0.0.0",
        mcp_allowed_hosts="mcp.invalid",
        mcp_tls_certfile="cert.pem",
        mcp_tls_keyfile="key.pem",
        messaging_alert_intake_allow_remote=False,
        messaging_alert_intake_port=None,
        mcp_url="https://mcp.invalid",
        fleet_mcp_url_template=None,
        openai_base_url=None,
        deepseek_base_url=None,
        vllm_base_url=None,
        kg_rerank_base_url=None,
        enable_otel=True,
        otel_exporter_otlp_endpoint="https://telemetry.invalid",
        eunomia_type="none",
        kg_fuseki_publish=False,
        sparql_endpoints=(),
        chat_models=(),
        embedding_models=(),
        a2a_broker="epistemic_graph",
        agent_bus_log_backend="engine",
        a2a_storage="epistemic_graph",
        kafka_bootstrap_servers="redpanda-0:9092",
        epistemic_graph_max_resident_graphs=1024,
        epistemic_graph_lazy_open_page_size=4096,
        epistemic_graph_max_nodes_per_graph=250_000,
        graph_service_endpoints=None,
        epistemic_graph_encryption_key_ref="env://ENGINE_DATA_KEY",
        usage_db_backend="postgres",
        usage_tracking_enabled=True,
        usage_content_retention="metadata",
        langfuse_capture_content=False,
        langfuse_public_key_ref=None,
        langfuse_secret_key_ref=None,
        langfuse_mcp_enabled=False,
        kg_failure_evolution=False,
        trace_export_enabled=False,
        persistence_identity_hmac_key_ref="env://PERSISTENCE_IDENTITY_HMAC_KEY",
        permissions_signing_key_ref="env://AGENT_PERMISSION_AUTHORITY",
    )
    base.update(overrides)
    return _make_config(**base)


def test_clean_production_config_has_zero_violations():
    assert collect_production_violations(_clean_prod_config()) == []


def test_fully_toy_config_violation_list_exact_order_and_content():
    cfg = _fully_toy_config()
    violations = collect_production_violations(cfg)
    expected = [
        "graph_persistence_type='sqlite' is single-host and non-shardable; "
        "use a distributed backend (e.g. 'postgresql').",
        "auth_jwt_jwks_uri is required for verified identity.",
        "auth_jwt_issuer is required for issuer binding.",
        "auth_jwt_audience is required for audience binding.",
        "kg_policy_version is required for policy pinning.",
        "kg_loop_allow_host_validation enables a development-only host "
        "capability and must be disabled in production.",
        "debug is enabled; production must not expose debug behavior.",
        "tool_guard_mode must be 'on' or 'strict'; production cannot disable "
        "the governed tool boundary.",
        "allowed_hosts contains a wildcard; production requires exact authorities.",
        "allowed_origins contains a wildcard; production requires exact origins.",
        "server TLS identity is incomplete; configure both certificate and key.",
        "a non-loopback REST listener requires direct TLS or an explicitly "
        "declared TLS-terminating ingress.",
        "mcp_allowed_hosts contains a wildcard; production requires exact authorities.",
        "MCP TLS identity is incomplete; configure both certificate and key.",
        "messaging_alert_intake_allow_remote is a plaintext remote-serving "
        "escape hatch and must be disabled in production.",
        "messaging alert intake requires a runtime token reference.",
        "messaging alert intake must bind to loopback in production.",
        "mcp_url uses plaintext HTTP to a remote service; use HTTPS or a "
        "loopback transport in production.",
        "a2a_broker must be 'epistemic_graph'; the native durable broker is "
        "the sole current FastA2A delivery plane.",
        "agent_bus_log_backend must be 'engine' in production; the Kafka "
        "AgentBus Kafka topology creates a consumer group per "
        "recipient, and graph fallback is not a scalable delivery plane.",
        "a2a_storage must be 'epistemic_graph'; native CAS-fenced records are "
        "the sole current FastA2A state plane.",
        "kafka_bootstrap_servers is unset; the reactive event ledger falls "
        "back to the single-process in-memory EventBus (no durability, no "
        "cross-node fan-out). Set it to a Redpanda/Kafka cluster for prod.",
        "epistemic_graph_max_resident_graphs must be positive in production; "
        "0 leaves the resident graph cache unbounded.",
        "epistemic_graph_lazy_open_page_size must be positive in production; "
        "0 performs an unbounded all-at-once lazy open.",
        "epistemic_graph_max_nodes_per_graph must be positive in production; "
        "0 leaves one graph's resident node set unbounded.",
        "epistemic_graph_encryption_key_ref is required for a packaged local "
        "production engine.",
        "usage_db_backend='sqlite' is not a shared production analytics "
        "store; use 'postgres' for tenant-aware usage/SLO data.",
        "usage_tracking_enabled is false; production requires tenant-aware "
        "usage and cost evidence.",
        "usage_content_retention must be 'metadata' in production; the "
        "analytics store cannot retain prompts, thinking text, or tool inputs.",
        "langfuse_capture_content is true; production traces must remain "
        "metadata-only.",
        "Langfuse credential configuration is partial; configure a runtime "
        "secret reference for each key.",
        "LANGFUSE_HOST must use HTTPS in production.",
        "persistence_identity_hmac_key_ref is unset; production durable "
        "identities require a secret-backed HMAC key reference.",
        "permissions_signing_key_ref is unset and no durable secret store is "
        "configured to self-provision a stable signing key; production agent "
        "identities require either a runtime secret reference or a durable "
        "engine/vault secrets backend.",
        "the production OpenTelemetry signal path is incomplete; set "
        "ENABLE_OTEL=true and inject OTEL_EXPORTER_OTLP_ENDPOINT at runtime.",
    ]
    assert violations == expected


def test_jwks_uri_remote_plaintext_http_is_flagged():
    cfg = _clean_prod_config(auth_jwt_jwks_uri="http://identity.invalid/jwks")
    violations = collect_production_violations(cfg)
    assert any("auth_jwt_jwks_uri uses plaintext HTTP" in v for v in violations)


def test_loopback_jwks_uri_over_http_is_not_flagged_as_remote():
    # _is_remote_plaintext_http only flags non-loopback HTTP hosts.
    cfg = _clean_prod_config(auth_jwt_jwks_uri="http://127.0.0.1:8000/jwks")
    violations = collect_production_violations(cfg)
    assert not any("plaintext HTTP" in v for v in violations)


def test_loopback_host_skips_rest_boundary_checks_even_without_allowlist():
    cfg = _clean_prod_config(host="127.0.0.1", allowed_hosts="")
    violations = collect_production_violations(cfg)
    assert not any("non-loopback REST listener" in v for v in violations)


class _FakeModel:
    def __init__(self, base_url, fallback=None):
        self.base_url = base_url
        self.fallback = fallback


def test_embedding_model_fallback_chain_depth_capped_at_two():
    # The original loop walks `current.fallback` while depth < 2, so a
    # third-level fallback is never inspected -- pin that boundary exactly.
    inner = _FakeModel("http://l3.invalid")
    mid = _FakeModel("http://l2.invalid", fallback=inner)
    outer = _FakeModel("http://l1.invalid", fallback=mid)
    cfg = _clean_prod_config(embedding_models=(outer,))
    violations = collect_production_violations(cfg)
    flagged = [v for v in violations if "embedding_models[0]" in v]
    assert len(flagged) == 2
    assert "embedding_models[0].base_url" in flagged[0]
    assert "embedding_models[0].fallback.base_url" in flagged[1]
    assert not any("l3.invalid" in v for v in violations)


def test_chat_model_remote_plaintext_flagged_by_index():
    model_a = _FakeModel("https://ok.invalid")
    model_b = _FakeModel("http://bad.invalid")
    cfg = _clean_prod_config(chat_models=(model_a, model_b))
    violations = collect_production_violations(cfg)
    assert any("chat_models[1].base_url" in v for v in violations)
    assert not any("chat_models[0].base_url" in v for v in violations)


def test_sparql_endpoints_indexed_in_violation_message():
    cfg = _clean_prod_config(
        sparql_endpoints=("https://ok.invalid", "http://bad.invalid")
    )
    violations = collect_production_violations(cfg)
    assert any("sparql_endpoints[1]" in v for v in violations)
    assert not any("sparql_endpoints[0]" in v for v in violations)


def test_otel_endpoint_only_checked_when_enable_otel_true():
    # enable_otel=False means otel_exporter_otlp_endpoint is never added to
    # endpoint_fields, so a remote-plaintext OTLP endpoint is NOT flagged by
    # the endpoint-plaintext check -- only by the always-on otel completeness
    # check at the end.
    cfg = _clean_prod_config(
        enable_otel=False, otel_exporter_otlp_endpoint="http://collector.invalid"
    )
    violations = collect_production_violations(cfg)
    assert not any(
        "otel_exporter_otlp_endpoint uses plaintext HTTP" in v for v in violations
    )
    assert any("OpenTelemetry signal path is incomplete" in v for v in violations)


def test_eunomia_remote_url_checked_only_when_type_remote():
    cfg = _clean_prod_config(
        eunomia_type="none", eunomia_remote_url="http://eunomia.invalid"
    )
    violations = collect_production_violations(cfg)
    assert not any("eunomia_remote_url" in v for v in violations)

    cfg2 = _clean_prod_config(
        eunomia_type="remote", eunomia_remote_url="http://eunomia.invalid"
    )
    violations2 = collect_production_violations(cfg2)
    assert any("eunomia_remote_url" in v for v in violations2)


def test_fuseki_endpoint_checked_only_when_publish_enabled():
    cfg = _clean_prod_config(
        kg_fuseki_publish=False, kg_fuseki_endpoint="http://fuseki.invalid"
    )
    violations = collect_production_violations(cfg)
    assert not any("kg_fuseki_endpoint" in v for v in violations)

    cfg2 = _clean_prod_config(
        kg_fuseki_publish=True, kg_fuseki_endpoint="http://fuseki.invalid"
    )
    violations2 = collect_production_violations(cfg2)
    assert any("kg_fuseki_endpoint" in v for v in violations2)


def test_messaging_alert_intake_port_none_skips_token_and_host_checks():
    cfg = _clean_prod_config(
        messaging_alert_intake_port=None,
        messaging_alert_intake_allow_remote=False,
    )
    violations = collect_production_violations(cfg)
    assert not any("messaging alert intake" in v for v in violations)


def test_local_engine_encryption_key_not_required_when_remote_endpoints_configured():
    # local_engine_managed is False when graph_service_endpoints is truthy, so
    # the encryption-key-ref requirement is skipped entirely.
    cfg = _clean_prod_config(
        graph_service_endpoints=["postgresql://remote/db"],
        epistemic_graph_encryption_key_ref="",
    )
    violations = collect_production_violations(cfg)
    assert not any("epistemic_graph_encryption_key_ref" in v for v in violations)


def test_permission_signing_key_self_provisioning_durable_backend_accepted():
    # No explicit permissions_signing_key_ref, but a durable secrets backend
    # (engine/vault) is accepted as self-provisioning -- observed bug/feature:
    # 'local-file' (anything outside {engine, vault}) is NOT accepted.
    cfg = _clean_prod_config(permissions_signing_key_ref="", secrets_backend="vault")
    violations = collect_production_violations(cfg)
    assert not any("permissions_signing_key_ref" in v for v in violations)

    cfg2 = _clean_prod_config(
        permissions_signing_key_ref="", secrets_backend="local-file"
    )
    violations2 = collect_production_violations(cfg2)
    assert any("permissions_signing_key_ref" in v for v in violations2)
