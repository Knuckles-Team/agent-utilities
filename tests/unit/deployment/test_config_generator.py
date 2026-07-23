"""Tests for the full-deployment config generator / validator.

Covers complete generation per profile, secret redaction, the grouped reference,
the doctor, and the live-path through the ``graph_configure`` MCP actions.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.deployment import (
    PROFILES,
    config_doctor,
    config_reference,
    generate_config,
    write_config,
)


# ── generation ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("profile", PROFILES)
def test_generate_config_is_complete(profile):
    cfg = generate_config(profile)
    # The whole AgentConfig surface is present (>=250 fields).
    assert len(cfg) >= 250
    assert "GRAPH_" + "BACKEND" not in cfg
    assert "GRAPH_" + "AUTHORITY" not in cfg


def test_generate_config_profile_presets():
    tiny = generate_config("tiny")
    snp = generate_config("single-node-prod")
    ent = generate_config("enterprise")
    assert tiny["DEPLOYMENT_PROFILE"] == "tiny"
    assert snp["DEPLOYMENT_PROFILE"] == "single-node-prod"
    assert ent["DEPLOYMENT_PROFILE"] == "enterprise"
    assert not tiny.get("GRAPH_DB_CONNECTION_PROFILE_REF")  # zero-infra
    assert "GRAPH_DB_CONNECTION_PROFILE_REF" in snp
    assert snp["GRAPH_SERVICE_ENDPOINTS"] == []
    assert "STATE_DB_URI" in ent
    assert ent["TASK_QUEUE_BACKEND"] == "kafka"
    assert ent["GRAPH_SERVICE_ENDPOINTS"] == []
    for production in (snp, ent):
        assert "KG_BRAIN_ENFORCE" not in production
        assert "KG_GRAPH_SESSION_REQUIRED" not in production
        assert production["AUTH_JWT_JWKS_URI"] is None
        assert production["AUTH_JWT_ISSUER"] is None
        assert production["AUTH_JWT_AUDIENCE"] is None
        assert production["KG_POLICY_VERSION"] == "baseline-v1"
        assert production["EPISTEMIC_GRAPH_MAX_RESIDENT_GRAPHS"] == 1024
        assert production["EPISTEMIC_GRAPH_LAZY_OPEN_PAGE_SIZE"] == 4096
        assert production["EPISTEMIC_GRAPH_MAX_NODES_PER_GRAPH"] == 250000
        assert production["ENABLE_OTEL"] is True
        assert production["LANGFUSE_MCP_ENABLED"] is True
        assert production["TRACE_EXPORT_ENABLED"] is True
        assert production["KG_LOOP"] is True
        assert production["KG_OPTIMIZATION_ENABLED"] is True
        assert production["KG_FAILURE_EVOLUTION"] is True
        assert production["KG_LOOP_MINE_DISCOVERY"] is True
        assert production["KG_LOOP_BELIEF_REVISION"] is True
        assert production["KG_LOOP_INSIGHT_VALIDATION"] is True
        assert production["KG_LOOP_TRACE_MINING"] is True
        assert production["LANGFUSE_CAPTURE_CONTENT"] is False
        assert production["KG_FAILURE_REGRESSION_DATASET"] is False
        assert production["KG_GOLDEN_AUTO_MERGE"] is False
        assert production["KG_AGENT_AUTO_APPLY"] is False
        assert production["KG_LOOP_AUTO_DEVELOP"] is False
        assert production["KG_LOOP_ALLOW_HOST_VALIDATION"] is False
        assert production["KG_INSIGHT_AUTONOMY"] is False


def test_generate_config_and_reference_include_exact_skill_certification_fields():
    expected = {
        "SKILL_CERT_RUNTIME_CONFIGURATION",
        "SKILL_CERT_RUNTIME_PROFILE",
        "SKILL_CERT_RELEASE_SPEC",
        "SKILL_CERT_PROMOTION_EVIDENCE",
        "SKILL_CERT_GRAPHOS_ENDPOINT",
        "SKILL_CERT_GRAPHOS_COMMAND",
        "SKILL_VALIDATION_EVIDENCE_SIGNER_COMMAND",
        "SKILL_VALIDATION_EVIDENCE_VERIFIER_COMMAND",
        "SKILL_CERT_IDENTITY_AUTHORITY_MODE",
        "SKILL_CERT_IDENTITY_TOKEN_TTL_SECONDS",
    }
    generated = generate_config("tiny")
    assert expected <= set(generated)
    for field in expected - {
        "SKILL_CERT_IDENTITY_AUTHORITY_MODE",
        "SKILL_CERT_IDENTITY_TOKEN_TTL_SECONDS",
    }:
        assert generated[field] in (None, [])
    assert generated["SKILL_CERT_IDENTITY_AUTHORITY_MODE"] == (
        "ephemeral-https-loopback"
    )
    assert generated["SKILL_CERT_IDENTITY_TOKEN_TTL_SECONDS"] == 300

    sections = config_reference()
    certification = next(
        section
        for section in sections
        if section["section"] == "Exact skill certification deployment references"
    )
    assert {field["env"] for field in certification["fields"]} == expected


def test_generate_config_and_reference_include_production_certification_fields():
    expected = {
        "CERTIFICATION_MODE",
        "CERT_RELEASE_MANIFEST",
        "CERT_ARTIFACTS_DIR",
        "CERT_HARDWARE_CLASS",
        "CERT_LOAD_COMMAND",
        "CERT_METRICS_COMMAND",
        "CERT_HOOK_COMMANDS",
        "CERT_FAULT_ACTION_COMMANDS",
        "CERT_FAULT_PROBE_COMMANDS",
        "CERT_EVIDENCE_SIGNER_COMMAND",
        "CERT_EVIDENCE_VERIFIER_COMMAND",
        "CERT_PROMETHEUS_URL",
        "CERT_PROMETHEUS_BEARER_TOKEN_REF",
        "CERT_PROMETHEUS_TLS_PROFILE",
        "CERT_PROMETHEUS_TLS_PROFILE_REF",
    }
    generated = generate_config("tiny")
    assert expected <= set(generated)
    assert generated["CERTIFICATION_MODE"] == "disabled"
    for field in expected - {"CERTIFICATION_MODE"}:
        assert generated[field] in (None, [], {})

    certification = next(
        section
        for section in config_reference()
        if section["section"] == "Production certification runtime"
    )
    assert {field["env"] for field in certification["fields"]} == expected


@pytest.mark.parametrize("profile", PROFILES)
def test_generate_config_roundtrips_through_agent_config(profile):
    from agent_utilities.core.config import AgentConfig

    generated = generate_config(profile)
    parsed = AgentConfig(**generated)
    assert parsed.secrets_backend == generated["SECRETS_BACKEND"]
    assert (
        parsed.model_dump(by_alias=True)["SECRETS_BACKEND"]
        == generated["SECRETS_BACKEND"]
    )


def test_generate_config_redacts_populated_secrets():
    from agent_utilities.deployment.config_generator import _is_secret

    cfg = generate_config("enterprise")
    # No populated credential survives (None/"" are acceptable placeholders).
    for key, val in cfg.items():
        if _is_secret(key):
            assert val in (None, ""), f"{key} not redacted: {val!r}"
    # Non-credential config keys with 'SECRET' in the name are NOT clobbered.
    assert cfg["SECRETS_BACKEND"] == "vault"
    assert cfg["SECRETS_VAULT_URL"]  # preset URL preserved


def test_generate_config_ignores_live_settings_and_sanitizes_nested_material(
    monkeypatch,
):
    private_material = "private-nested-material"
    private_endpoint = "https://private-host.invalid/v1"
    monkeypatch.setenv(
        "CHAT_MODELS",
        json.dumps(
            [
                {
                    "id": "ambient-model",
                    "provider": "openai",
                    "base_url": private_endpoint,
                    "api_key": private_material,
                    "headers": {"Authorization": private_material},
                }
            ]
        ),
    )
    from agent_utilities.deployment.config_generator import (
        _sanitize_generated_value,
    )

    generated = generate_config("tiny")
    sanitized = _sanitize_generated_value(
        {
            "models": [
                {
                    "api_key": "env://RAW_MODEL_API_KEY",
                    "api_key_ref": "env://MODEL_API_KEY",
                    "oauth2": {"client_secret": "env://MODEL_CLIENT_SECRET"},
                    "headers": {"Authorization": private_material},
                    "headers_ref": "env://MODEL_HEADERS",
                }
            ]
        }
    )
    rendered = json.dumps({"generated": generated, "sanitized": sanitized})

    assert private_material not in rendered
    assert private_endpoint not in rendered
    assert sanitized["models"][0]["api_key"] == ""
    assert sanitized["models"][0]["api_key_ref"] == "env://MODEL_API_KEY"
    assert sanitized["models"][0]["headers"] == {}
    assert sanitized["models"][0]["headers_ref"] == "env://MODEL_HEADERS"
    assert (
        sanitized["models"][0]["oauth2"]["client_secret"] == "env://MODEL_CLIENT_SECRET"
    )


def test_generate_config_unknown_profile():
    with pytest.raises(ValueError):
        generate_config("bogus")


def test_write_config_roundtrips(tmp_path):
    out = tmp_path / "config.json"
    res = write_config("single-node-prod", out)
    assert res["status"] == "success"
    assert res["destination"] == "explicit"
    assert "path" not in res
    assert str(tmp_path) not in json.dumps(res)
    assert res["keys"] >= 250
    loaded = json.loads(out.read_text())
    assert (
        loaded["GRAPH_DB_CONNECTION_PROFILE_REF"]
        == generate_config("single-node-prod")["GRAPH_DB_CONNECTION_PROFILE_REF"]
    )


def test_write_config_default_uses_active_config_root_without_returning_path(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "active-root"
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(root))

    result = write_config("tiny")

    assert (root / "config.json").is_file()
    assert result["destination"] == "xdg"
    assert str(root) not in json.dumps(result)


# Codex registration is covered in test_codex_registration.py.
# ── reference ──────────────────────────────────────────────────────────────
def test_config_reference_groups_all_fields():
    ref = config_reference()
    sections = {s["section"] for s in ref}
    assert len(sections) >= 5  # multiple subsystems detected
    total = sum(len(s["fields"]) for s in ref)
    assert total >= 250  # every field appears exactly once
    # A known field is grouped and carries env + type.
    flat = {f["env"]: f for s in ref for f in s["fields"]}
    assert "GRAPH_MIRROR_TARGETS" in flat
    assert flat["GRAPH_MIRROR_TARGETS"]["type"]
    assert "GRAPH_" + "BACKEND" not in flat
    assert "GRAPH_" + "AUTHORITY" not in flat


def test_config_reference_marks_secrets():
    flat = {f["env"]: f for s in config_reference() for f in s["fields"]}
    assert flat["OPENAI_API_KEY"]["secret"] is True
    assert flat["OPENAI_API_KEY"]["default"] == "***"


def test_config_reference_includes_memento_privacy_policy():
    flat = {f["env"]: f for s in config_reference() for f in s["fields"]}
    assert flat["MEMENTO_RAW_RETENTION_ENABLED"]["default"] is False
    assert "MEMENTO_RAW_RETENTION_POLICY" in flat
    assert "MEMENTO_RAW_ENCRYPTION_KEY_REF" in flat


def test_config_reference_exposes_placement_control_opt_in():
    flat = {f["env"]: f for s in config_reference() for f in s["fields"]}
    assert flat["PLACEMENT_CONTROL_LOOP_ENABLED"]["default"] is False


def test_config_reference_exposes_dispatch_lease_recovery_controls():
    flat = {f["env"]: f for s in config_reference() for f in s["fields"]}
    assert flat["AGENT_DISPATCH_CLAIM_TTL_S"]["default"] == 120.0
    assert flat["AGENT_DISPATCH_RENEW_INTERVAL_S"]["default"] == 30.0


# ── doctor ─────────────────────────────────────────────────────────────────
def test_doctor_tiny_healthy(tmp_path):
    out = tmp_path / "c.json"
    write_config("tiny", out)
    rep = config_doctor("tiny", out)
    assert rep["status"] == "success"
    assert rep["healthy"] is True
    lease = next(c for c in rep["checks"] if c["check"] == "dispatch_lease_recovery")
    assert lease["ok"] is True
    assert lease["renew_interval_seconds"] < lease["claim_ttl_seconds"] <= 300.0


def test_secret_reference_inventory_covers_nested_and_secret_scheme(monkeypatch):
    from agent_utilities.deployment.config_generator import _unresolved_secret_refs

    class _Secrets:
        def resolve_ref(self, reference):
            return "resolved" if reference.endswith("RESOLVED") else None

    monkeypatch.setattr(
        "agent_utilities.security.secrets_client.create_secrets_client",
        lambda: _Secrets(),
    )
    monkeypatch.setenv("TEST_RESOLVED", "runtime-material")
    unresolved = _unresolved_secret_refs(
        {
            "top": "env://TEST_RESOLVED",
            "models": [
                {"oauth2": {"client_secret": "env://TEST_NESTED_MISSING"}},
                {"profile": "secret://runtime/profile"},
            ],
        }
    )

    assert unresolved == ["unresolved", "unresolved"]
    assert all(marker == "unresolved" for marker in unresolved)


def test_env_only_secret_inventory_never_constructs_durable_backend(monkeypatch):
    from agent_utilities.deployment.config_generator import _unresolved_secret_refs

    calls = 0

    def forbidden_backend():
        nonlocal calls
        calls += 1
        raise AssertionError("env-only inventory must not construct a backend")

    monkeypatch.setattr(
        "agent_utilities.security.secrets_client.create_secrets_client",
        forbidden_backend,
    )
    monkeypatch.setenv("TEST_ENV_ONLY_RESOLVED", "runtime-material")

    unresolved = _unresolved_secret_refs(
        {
            "resolved": "env://TEST_ENV_ONLY_RESOLVED",
            "missing": "env://TEST_ENV_ONLY_MISSING",
        }
    )

    assert unresolved == ["unresolved"]
    assert calls == 0


# ── secret_reference_scheme_counts (SECRETS_BACKEND scheme/backend gate) ────


def test_secret_reference_scheme_counts_buckets_by_scheme():
    from agent_utilities.deployment.config_generator import (
        secret_reference_scheme_counts,
    )

    counts = secret_reference_scheme_counts(
        {
            "top": "vault://apps/gitlab/token",
            "nested": [
                {"oauth2": {"client_secret": "secret://identity/client-secret"}},
                {"other": "env://SOME_TOKEN"},
            ],
            "duplicate": "vault://apps/gitlab/token",
            "not_a_ref": "just a plain string",
            "number": 42,
        }
    )

    assert counts == {"env": 1, "vault": 1, "secret": 1}


def test_secret_reference_scheme_counts_empty_when_no_references():
    from agent_utilities.deployment.config_generator import (
        secret_reference_scheme_counts,
    )

    assert secret_reference_scheme_counts({"a": "b", "c": [1, 2, {"d": "e"}]}) == {
        "env": 0,
        "vault": 0,
        "secret": 0,
    }


def test_secret_reference_scheme_counts_counts_repeated_scheme():
    from agent_utilities.deployment.config_generator import (
        secret_reference_scheme_counts,
    )

    counts = secret_reference_scheme_counts(
        {
            "a": "vault://apps/one/A",
            "b": "vault://apps/two/B",
            "c": "vault://apps/three/C",
        }
    )

    assert counts == {"env": 0, "vault": 3, "secret": 0}


def test_candidate_doctor_never_uses_ambient_secret_backend(
    tmp_path,
    monkeypatch,
):
    private_detail = "backend-private-detail"
    calls = 0

    def forbidden_backend():
        nonlocal calls
        calls += 1
        raise RuntimeError(private_detail)

    monkeypatch.setattr(
        "agent_utilities.security.secrets_client.create_secrets_client",
        forbidden_backend,
    )
    out = tmp_path / "c.json"
    candidate = generate_config("tiny")
    candidate["LANGFUSE_SECRET_KEY_REF"] = "secret://runtime/profile"
    out.write_text(json.dumps(candidate), encoding="utf-8")

    report = config_doctor("tiny", out)
    check = next(item for item in report["checks"] if item["check"] == "secret_refs")
    rendered = json.dumps(report, sort_keys=True)

    assert report["healthy"] is False
    assert check["ok"] is False
    assert check["redacted"] is True
    assert check["unresolved_count"] == 1
    assert "evaluation_error" not in check
    assert calls == 0
    assert private_detail not in rendered


def test_candidate_doctor_does_not_resolve_env_refs_from_ambient_process(
    tmp_path,
    monkeypatch,
):
    candidate = generate_config("tiny")
    candidate["LANGFUSE_SECRET_KEY_REF"] = "env://CANDIDATE_ONLY_SECRET"
    out = tmp_path / "c.json"
    out.write_text(json.dumps(candidate), encoding="utf-8")
    monkeypatch.setenv("CANDIDATE_ONLY_SECRET", "ambient-private-material")

    report = config_doctor("tiny", out)
    check = next(item for item in report["checks"] if item["check"] == "secret_refs")
    rendered = json.dumps(report, sort_keys=True)

    assert check["ok"] is False
    assert check["unresolved_count"] == 1
    assert "ambient-private-material" not in rendered
    assert "CANDIDATE_ONLY_SECRET" not in rendered


def test_doctor_consumes_explicit_deployment_identity(tmp_path):
    cfg = generate_config("single-node-prod")
    cfg["GRAPH_DB_CONNECTION_PROFILE_REF"] = "env://SYNTHETIC_GRAPH_PROFILE"
    out = tmp_path / "c.json"
    out.write_text(json.dumps(cfg))
    out.chmod(0o600)

    rep = config_doctor(config_path=out)

    assert rep["profile"] == "single-node-prod"
    identity = next(c for c in rep["checks"] if c["check"] == "deployment_profile")
    assert identity == {
        "check": "deployment_profile",
        "profile": "single-node-prod",
        "source": "configuration",
        "ok": True,
    }


def test_doctor_does_not_guess_enterprise_from_production_posture(tmp_path):
    cfg = generate_config("single-node-prod")
    cfg.pop("DEPLOYMENT_PROFILE")
    out = tmp_path / "c.json"
    out.write_text(json.dumps(cfg))
    out.chmod(0o600)

    rep = config_doctor(config_path=out)

    assert rep["status"] == "error"
    assert rep["healthy"] is False
    assert rep["error"] == "deployment_profile_required"
    assert "profile" not in rep


def test_doctor_enterprise_flags_missing(tmp_path):
    # An enterprise config with the required DSN/auth blanked must be unhealthy.
    cfg = generate_config("enterprise")
    cfg["GRAPH_DB_CONNECTION_PROFILE_REF"] = ""
    cfg["STATE_DB_URI"] = ""
    cfg["AUTH_JWT_JWKS_URI"] = ""
    out = tmp_path / "c.json"
    out.write_text(json.dumps(cfg))
    out.chmod(0o600)
    rep = config_doctor("enterprise", out)
    assert rep["healthy"] is False
    req = next(c for c in rep["checks"] if c["check"] == "required_keys")
    assert "GRAPH_DB_CONNECTION_PROFILE_REF" in req["missing"]


def test_doctor_accepts_single_node_local_engine_topology(tmp_path):
    cfg = generate_config("single-node-prod")
    cfg["GRAPH_DB_CONNECTION_PROFILE_REF"] = "env://SYNTHETIC_GRAPH_PROFILE"
    cfg["AUTH_JWT_JWKS_URI"] = "https://identity.example.test/jwks"
    cfg["AUTH_JWT_ISSUER"] = "https://identity.example.test"
    cfg["AUTH_JWT_AUDIENCE"] = "graph-api"
    assert cfg["GRAPH_SERVICE_ENDPOINTS"] == []
    out = tmp_path / "c.json"
    out.write_text(json.dumps(cfg))
    out.chmod(0o600)

    rep = config_doctor("single-node-prod", out)
    required = next(c for c in rep["checks"] if c["check"] == "required_keys")

    assert required["ok"] is True
    assert "GRAPH_SERVICE_ENDPOINTS" not in required["missing"]


def test_doctor_reports_missing_outbound_oidc_audience_without_values(tmp_path):
    cfg = generate_config("tiny")
    cfg.update(
        {
            "MCP_CLIENT_AUTH": "oidc-client-credentials",
            "OIDC_CLIENT_ID": "fleet-client",
            "OIDC_CLIENT_SECRET_REF": "env://TEST_FLEET_CLIENT_SECRET",
            "OIDC_TOKEN_URL": "https://identity.example.test/token",
            "OIDC_AUDIENCE": None,
        }
    )
    out = tmp_path / "c.json"
    out.write_text(json.dumps(cfg))

    rep = config_doctor("tiny", out)
    check = next(c for c in rep["checks"] if c["check"] == "outbound_mcp_auth")

    assert check == {
        "check": "outbound_mcp_auth",
        "ok": False,
        "mode": "oidc-client-credentials",
        "missing": ["OIDC_AUDIENCE"],
        "redacted": True,
    }


def test_doctor_reports_redacted_propose_only_observability_readiness(tmp_path):
    cfg = generate_config("enterprise")
    cfg["KG_OPTIMIZATION_ENABLED"] = False
    cfg["LANGFUSE_CAPTURE_CONTENT"] = True
    out = tmp_path / "c.json"
    out.write_text(json.dumps(cfg))
    out.chmod(0o600)

    rep = config_doctor("enterprise", out)
    check = next(c for c in rep["checks"] if c["check"] == "propose_only_observability")
    assert check["ok"] is False
    assert check["redacted"] is True
    assert set(check["mismatched"]) == {
        "KG_OPTIMIZATION_ENABLED",
        "LANGFUSE_CAPTURE_CONTENT",
    }
    assert "actual" not in check and "expected" not in check


def test_doctor_unreadable_config(tmp_path):
    rep = config_doctor("tiny", tmp_path / "does-not-exist.json")
    assert rep["status"] == "error"
    assert rep["healthy"] is False
    assert rep["error"] == "configuration_source_unreadable"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission contract")
def test_candidate_doctor_rejects_nonprivate_production_source(
    tmp_path,
):
    cfg = generate_config("single-node-prod")
    out = tmp_path / "private-candidate.json"
    out.write_text(json.dumps(cfg), encoding="utf-8")
    out.chmod(0o640)

    report = config_doctor("single-node-prod", out)
    rendered = json.dumps(report, sort_keys=True)

    assert report["status"] == "error"
    assert report["healthy"] is False
    assert report["error"] == "configuration_source_unreadable"
    assert report["error_class"] == "ConfigurationSourceError"
    assert str(tmp_path) not in rendered


def test_doctor_returns_structured_schema_error(tmp_path):
    cfg = generate_config("single-node-prod")
    cfg["SECRETS_BACKEND"] = "retired-backend"
    out = tmp_path / "c.json"
    out.write_text(json.dumps(cfg))
    out.chmod(0o600)

    rep = config_doctor("single-node-prod", out)

    assert rep == {
        "status": "error",
        "healthy": False,
        "error": "configuration_schema_invalid",
        "error_class": "ValidationError",
        "checks": [{"check": "schema", "ok": False}],
    }


def test_doctor_rejects_partial_memento_raw_retention_config(tmp_path):
    cfg = generate_config("tiny")
    cfg["MEMENTO_RAW_RETENTION_ENABLED"] = True
    out = tmp_path / "c.json"
    out.write_text(json.dumps(cfg))

    rep = config_doctor("tiny", out)
    check = next(c for c in rep["checks"] if c["check"] == "memento_raw_retention")
    assert check["ok"] is False
    assert set(check["issues"]) == {
        "approved_policy_required",
        "encryption_key_reference_required",
    }


# ── live path: graph_configure MCP actions ─────────────────────────────────
class _MockMCP:
    def __init__(self):
        self.funcs = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.funcs[fn.__name__] = fn
            return fn

        return deco

    def custom_route(self, *a, **k):
        def deco(fn):
            self.funcs[fn.__name__] = fn
            return fn

        return deco


@pytest.fixture
def registered_tools():
    mock_mcp = _MockMCP()
    engine = MagicMock()
    engine.backend = MagicMock()
    engine.backend.read_only = False
    with patch(
        "agent_utilities.mcp.server_factory.create_mcp_server",
        return_value=(None, mock_mcp, []),
    ):
        with patch("agent_utilities.mcp.kg_server._get_engine", return_value=engine):
            from agent_utilities.mcp.kg_server import _build_server

            _build_server()
    return mock_mcp.funcs


@pytest.mark.asyncio
async def test_graph_configure_generate_config_uses_fixed_xdg_path(
    registered_tools, tmp_path, monkeypatch
):
    from agent_utilities.mcp import kg_server

    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path))
    raw = await kg_server._execute_tool(
        "graph_configure",
        action="generate_config",
        config_key="single-node-prod",
        config_value=json.dumps({}),
    )
    res = json.loads(raw)
    assert res["status"] == "success"
    assert res["destination"] == "xdg"
    assert "path" not in res
    assert (tmp_path / "config.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action, option", [("generate_config", "out"), ("config_doctor", "config")]
)
async def test_graph_configure_rejects_remote_filesystem_paths(
    registered_tools,
    tmp_path,
    action,
    option,
):
    from agent_utilities.mcp import kg_server

    candidate = tmp_path / "private-candidate.json"
    raw = await kg_server._execute_tool(
        "graph_configure",
        action=action,
        config_key="tiny",
        config_value=json.dumps({option: str(candidate)}),
    )
    result = json.loads(raw)

    assert result == {"error": "remote_path_not_allowed"}
    assert str(candidate) not in raw
    assert not candidate.exists()


@pytest.mark.asyncio
async def test_graph_configure_config_reference_live_path(registered_tools):
    from agent_utilities.mcp import kg_server

    raw = await kg_server._execute_tool("graph_configure", action="config_reference")
    ref = json.loads(raw)
    assert isinstance(ref, list) and ref and "section" in ref[0]
