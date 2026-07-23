"""Tests for the holistic `agent-utilities doctor` health sweep.

The doctor is an aggregator of existing checks; tests assert the contract (report
shape, status precedence, fix routing, defensiveness) with the underlying checks
monkeypatched, plus a live-path through the graph_configure MCP action.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.deployment import doctor as D


def _ok(name):
    return lambda **kw: D._result(name, "ok", "fine")


def test_run_doctor_all_ok(monkeypatch):
    monkeypatch.setattr(D, "CHECKS", {n: _ok(n) for n in ("a", "b", "c")})
    rep = D.run_doctor()
    assert rep["status"] == "healthy"
    assert len(rep["checks"]) == 3
    assert rep["summary"] == "All checks passed."


def test_run_doctor_loads_agent_config_before_checks(monkeypatch):
    observed = {"loaded": False, "checked": False}

    def load_config():
        observed["loaded"] = True

    def check():
        observed["checked"] = observed["loaded"]
        return D._result("config", "ok", "fine")

    monkeypatch.setattr("agent_utilities.core.config.load_config", load_config)
    monkeypatch.setattr(D, "CHECKS", {"config": check})

    rep = D.run_doctor()

    assert rep["status"] == "healthy"
    assert observed == {"loaded": True, "checked": True}


def test_run_doctor_redacts_configuration_load_failure(monkeypatch):
    def load_config():
        raise ValueError("private configuration detail")

    monkeypatch.setattr("agent_utilities.core.config.load_config", load_config)
    monkeypatch.setattr(
        D, "CHECKS", {"config": lambda: D._result("config", "ok", "fine")}
    )

    rep = D.run_doctor()

    assert rep["status"] == "unhealthy"
    assert rep["checks"][0]["status"] == "error"
    assert "ValueError" in rep["checks"][0]["detail"]
    assert "private configuration detail" not in json.dumps(rep, sort_keys=True)


def test_graph_authority_doctor_rejects_external_primary(monkeypatch):
    from agent_utilities.knowledge_graph import backends

    class ExternalBackend:
        pass

    monkeypatch.setattr(backends, "get_active_backend", lambda: ExternalBackend())

    result = D._check_graph_authority()

    assert result["name"] == "graph_authority"
    assert result["status"] == "fail"
    assert result["data"] == {"authority_current": False}


def test_graph_authority_doctor_accepts_fixed_fanout_authority(monkeypatch):
    from agent_utilities.knowledge_graph import backends
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.backends.fanout_backend import FanOutBackend

    authority = object.__new__(EpistemicGraphBackend)
    monkeypatch.setattr(
        EpistemicGraphBackend,
        "health_check",
        lambda _self: True,
        raising=False,
    )
    fanout = object.__new__(FanOutBackend)
    fanout._authority = authority
    fanout._mirrors = {"projection": object()}
    monkeypatch.setattr(backends, "get_active_backend", lambda: fanout)

    result = D._check_graph_authority()

    assert result["status"] == "ok"
    assert result["data"] == {
        "authority_current": True,
        "projection_count": 1,
    }


def test_unified_install_doctor_never_reports_local_roots(monkeypatch, tmp_path):
    from agent_utilities.core import paths, unified_install
    from agent_utilities.core.provider_materialization import build_asset_manifest

    monkeypatch.setattr(paths, "skills_dir", lambda: tmp_path / "skills")
    monkeypatch.setattr(paths, "ontology_dir", lambda: tmp_path / "ontologies")
    monkeypatch.setattr(
        unified_install, "unified_prompts_dir", lambda: tmp_path / "prompts"
    )
    sources = {}
    for leg in ("skills", "prompts", "ontologies"):
        source = tmp_path / "sources" / leg
        source.mkdir(parents=True)
        if leg == "skills":
            skill = source / "skill"
            skill.mkdir()
            (skill / "SKILL.md").write_text("body", encoding="utf-8")
        elif leg == "prompts":
            (source / "prompt.json").write_text("{}", encoding="utf-8")
        else:
            (source / "ontology.ttl").write_text("# ttl", encoding="utf-8")
        sources[leg] = (source, "a" * 64, build_asset_manifest(source, leg=leg))
    monkeypatch.setattr(
        "agent_utilities.core.providers.provider_registrations", lambda _group: ()
    )
    monkeypatch.setattr(unified_install, "own_provider_asset", lambda leg: sources[leg])

    result = D._check_unified_install()
    rendered = json.dumps(result, sort_keys=True)

    assert "roots" not in (result.get("data") or {})
    assert set(result["data"]["roots_ready"]) == {
        "skills",
        "prompts",
        "ontologies",
    }
    assert str(tmp_path) not in rendered


def test_unified_install_doctor_warns_on_stale_and_unmarked_provider_roots(
    monkeypatch, tmp_path
):
    from agent_utilities.core import paths, unified_install
    from agent_utilities.core.provider_materialization import (
        build_asset_manifest,
        inactive_marker,
        write_managed_provider_marker,
    )

    roots = {
        "skills": tmp_path / "skills",
        "prompts": tmp_path / "prompts",
        "ontologies": tmp_path / "ontologies",
    }
    sources = {}
    for leg, root in roots.items():
        source = tmp_path / "sources" / leg
        source.mkdir(parents=True)
        if leg == "skills":
            skill = source / "skill"
            skill.mkdir()
            (skill / "SKILL.md").write_text("body", encoding="utf-8")
        elif leg == "prompts":
            (source / "prompt.json").write_text("{}", encoding="utf-8")
        else:
            (source / "ontology.ttl").write_text("# ttl", encoding="utf-8")
        manifest = build_asset_manifest(source, leg=leg)
        sources[leg] = (source, "a" * 64, manifest)
        root.mkdir(parents=True, exist_ok=True)
        unified_install._materialize_provider(
            root=root,
            provider="agent-utilities",
            leg=leg,
            registration="a" * 64,
            source=source,
            manifest=manifest,
        )
    stale = roots["prompts"] / "synthetic-removed-provider"
    stale.mkdir()
    write_managed_provider_marker(
        stale,
        inactive_marker(provider=stale.name, leg="prompts", registration="b" * 64),
    )
    unmanaged = roots["skills"] / "unmarked-provider" / "synthetic-skill"
    unmanaged.mkdir(parents=True)
    (unmanaged / "SKILL.md").write_text("synthetic", encoding="utf-8")
    local = roots["skills"] / "operator-skill"
    local.mkdir()
    (local / "SKILL.md").write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(paths, "skills_dir", lambda: roots["skills"])
    monkeypatch.setattr(paths, "ontology_dir", lambda: roots["ontologies"])
    monkeypatch.setattr(
        unified_install, "unified_prompts_dir", lambda: roots["prompts"]
    )
    monkeypatch.setattr(
        "agent_utilities.core.providers.provider_registrations", lambda _group: ()
    )
    monkeypatch.setattr(unified_install, "own_provider_asset", lambda leg: sources[leg])

    result = D._check_unified_install()

    assert result["status"] == "warn"
    assert result["data"]["managed_ready"] is False
    assert result["data"]["stale_managed"] == 1
    assert result["data"]["unmanaged_nested"] == 1
    assert str(tmp_path) not in json.dumps(result, sort_keys=True)


def test_unified_install_doctor_fails_closed_on_unresolved_provider(
    monkeypatch, tmp_path
):
    from agent_utilities.core import paths, providers, unified_install
    from agent_utilities.core.provider_materialization import build_asset_manifest

    roots = {
        "skills": tmp_path / "skills",
        "prompts": tmp_path / "prompts",
        "ontologies": tmp_path / "ontologies",
    }
    monkeypatch.setattr(paths, "skills_dir", lambda: roots["skills"])
    monkeypatch.setattr(paths, "ontology_dir", lambda: roots["ontologies"])
    monkeypatch.setattr(
        unified_install, "unified_prompts_dir", lambda: roots["prompts"]
    )
    unresolved = providers.ProviderRegistration(
        name="private-provider",
        group=providers.SKILL_PROVIDER_GROUP,
        target="private.skills",
        owner_name="private-distribution",
        owner_version="1",
        digest="a" * 64,
        source_root=None,
        owned_paths=frozenset(),
    )
    monkeypatch.setattr(
        providers,
        "provider_registrations",
        lambda group: (unresolved,) if group == providers.SKILL_PROVIDER_GROUP else (),
    )
    sources = {}
    for leg in roots:
        source = tmp_path / "sources" / leg
        source.mkdir(parents=True)
        if leg == "skills":
            skill = source / "skill"
            skill.mkdir()
            (skill / "SKILL.md").write_text("body", encoding="utf-8")
        elif leg == "prompts":
            (source / "prompt.json").write_text("{}", encoding="utf-8")
        else:
            (source / "ontology.ttl").write_text("# ttl", encoding="utf-8")
        sources[leg] = (source, "b" * 64, build_asset_manifest(source, leg=leg))
    monkeypatch.setattr(unified_install, "own_provider_asset", lambda leg: sources[leg])

    result = D._check_unified_install()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "fail"
    assert result["data"]["unresolved"] == 1
    assert result["data"]["redacted"] is True
    assert "private-provider" not in rendered
    assert "private-distribution" not in rendered
    assert str(tmp_path) not in rendered


def test_run_doctor_worst_status_wins(monkeypatch):
    monkeypatch.setattr(
        D,
        "CHECKS",
        {
            "a": _ok("a"),
            "b": lambda **kw: D._result("b", "warn", "meh"),
            "c": lambda **kw: D._result(
                "c", "fail", "broken", remediation="do x", skill="s"
            ),
        },
    )
    rep = D.run_doctor()
    assert rep["status"] == "unhealthy"  # a fail dominates
    assert rep["counts"]["fail"] == 1 and rep["counts"]["warn"] == 1


def test_run_doctor_skip_is_not_unhealthy(monkeypatch):
    monkeypatch.setattr(
        D, "CHECKS", {"a": _ok("a"), "b": lambda **kw: D._result("b", "skip", "n/a")}
    )
    assert D.run_doctor()["status"] == "healthy"


def test_run_doctor_check_exception_is_contained(monkeypatch):
    def _boom(**kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(D, "CHECKS", {"a": _ok("a"), "b": _boom})
    rep = D.run_doctor()
    # The crashing check becomes an 'error' result, doctor still completes.
    b = next(c for c in rep["checks"] if c["name"] == "b")
    assert b["status"] == "error" and "RuntimeError" in b["detail"]
    assert "kaboom" not in b["detail"]
    assert rep["status"] == "unhealthy"


def test_run_doctor_only_filter(monkeypatch):
    monkeypatch.setattr(D, "CHECKS", {n: _ok(n) for n in ("a", "b", "c")})
    rep = D.run_doctor(["b"])
    assert [c["name"] for c in rep["checks"]] == ["b"]


def test_run_doctor_forwards_live_only_to_live_capability_checks(monkeypatch):
    seen = {}

    def live_check(name):
        def check(*, live=False):
            seen[name] = live
            return D._result(name, "ok", "fine")

        return check

    monkeypatch.setattr(
        D,
        "CHECKS",
        {
            "mcp_fleet": live_check("mcp_fleet"),
            "langfuse": live_check("langfuse"),
            "native_optimizer": live_check("native_optimizer"),
            "hooks": lambda: D._result("hooks", "ok", "fine"),
        },
    )

    rep = D.run_doctor(live=True)

    assert rep["status"] == "healthy"
    assert seen == {
        "mcp_fleet": True,
        "langfuse": True,
        "native_optimizer": True,
    }


def test_run_doctor_fix_runs_autofix_then_reruns(monkeypatch):
    state = {"fixed": False}

    def flaky(**kw):
        return D._result(
            "hooks",
            "ok" if state["fixed"] else "warn",
            "hooks",
            auto_fixable=True,
        )

    monkeypatch.setattr(D, "CHECKS", {"hooks": flaky})

    def fake_fix(name):
        state["fixed"] = True
        return {"fixed": name, "result": "ok"}

    monkeypatch.setattr(D, "_auto_fix", fake_fix)
    rep = D.run_doctor(fix=True)
    assert rep["fixes"] and rep["fixes"][0]["fixed"] == "hooks"
    # After the fix the re-run flipped the check to ok → overall healthy.
    assert rep["checks"][0]["status"] == "ok"
    assert rep["status"] == "healthy"


def test_individual_checks_never_raise():
    # Every real check must return a dict with the contract keys, never raise,
    # even with nothing deployed.
    for name, fn in D.CHECKS.items():
        res = fn(live=False) if name == "mcp_fleet" else fn()
        assert set(res) >= {"name", "status", "detail"}
        assert res["status"] in ("ok", "warn", "fail", "skip", "error")


def test_permission_governance_doctor_reports_missing_reference_redacted(
    monkeypatch,
):
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: SimpleNamespace(
            permissions_signing_key_ref=None,
            agent_policies_path=None,
            app_profile="dev",
        ),
    )

    result = D._check_permission_governance()

    assert result["status"] == "warn"
    assert result["data"]["redacted"] is True
    assert result["data"]["signing_reference_configured"] is False


def test_permission_governance_doctor_verifies_authority(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: SimpleNamespace(
            permissions_signing_key_ref="env://DOCTOR_PERMISSION_KEY",
            agent_policies_path=None,
            app_profile="dev",
        ),
    )
    monkeypatch.setattr(
        "agent_utilities.security.cli_secrets.resolve_runtime_secret_reference",
        lambda _reference: "doctor-signing-authority-material-32b",
    )

    result = D._check_permission_governance()

    assert result["status"] == "ok"
    assert result["data"]["identity_verified"] is True
    assert result["data"]["policy_count"] == 5


def test_ontology_release_signing_doctor_reports_missing_reference_redacted(
    monkeypatch,
):
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: SimpleNamespace(
            ontology_release_signing_private_key_ref=None,
            app_profile="dev",
        ),
    )

    result = D._check_ontology_release_signing()

    assert result["status"] == "warn"
    assert result["data"]["redacted"] is True
    assert result["data"]["signing_reference_configured"] is False


def test_ontology_release_signing_doctor_verifies_stable_authority(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: SimpleNamespace(
            ontology_release_signing_private_key_ref="env://ONTOLOGY_RELEASE_KEY",
            app_profile="dev",
        ),
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.ontology_integrity.ReleaseSigner.from_runtime",
        lambda: SimpleNamespace(public_key="stable-public-key"),
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.ontology_integrity.release_trusted_public_keys",
        lambda: ("stable-public-key",),
    )

    result = D._check_ontology_release_signing()

    assert result["status"] == "ok"
    assert result["data"]["signing_authority_ready"] is True
    assert result["data"]["signer_public_key_trusted"] is True


def test_outbound_auth_doctor_reports_redacted_readiness(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.mcp.client_credentials.outbound_auth_configuration_status",
        lambda: {
            "mode": "oidc-client-credentials",
            "ready": True,
            "missing": (),
            "invalid": (),
            "redacted": True,
        },
    )

    result = D._check_outbound_auth()

    assert result["status"] == "ok"
    assert result["data"] == {
        "mode": "oidc-client-credentials",
        "ready": True,
        "redacted": True,
    }


def test_outbound_auth_doctor_fails_closed_on_missing_audience(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.mcp.client_credentials.outbound_auth_configuration_status",
        lambda: {
            "mode": "oidc-client-credentials",
            "ready": False,
            "missing": ("OIDC_AUDIENCE",),
            "invalid": (),
            "redacted": True,
        },
    )

    result = D._check_outbound_auth()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "fail"
    assert result["data"] == {
        "mode": "oidc-client-credentials",
        "ready": False,
        "missing_count": 1,
        "invalid_count": 0,
        "redacted": True,
    }
    assert "env://" not in rendered
    assert "secret://" not in rendered


def test_graph_identity_doctor_validates_token_ref_without_exposing_it(monkeypatch):
    cfg = SimpleNamespace(
        kg_auth_token_ref="secret://graph/private-token",
        kg_identity_oauth2=None,
    )
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: cfg)
    monkeypatch.setattr(
        "agent_utilities.security.cli_secrets.resolve_runtime_secret_reference",
        lambda _ref: "opaque-runtime-token",
    )

    result = D._check_graph_identity()
    rendered = json.dumps(result, sort_keys=True)
    assert result["status"] == "ok"
    assert result["data"] == {
        "mode": "token_ref",
        "ready": True,
        "redacted": True,
    }
    assert "secret://" not in rendered
    assert "opaque-runtime-token" not in rendered


def test_graph_identity_doctor_rejects_ambiguous_or_missing_source(monkeypatch):
    for cfg in (
        SimpleNamespace(
            kg_auth_token_ref=None,
            kg_identity_oauth2=None,
            deployment_profile="single-node-prod",
            graph_service_endpoints=[],
        ),
        SimpleNamespace(
            kg_auth_token_ref="secret://graph/token",
            kg_identity_oauth2={"client_secret": "secret://graph/client"},
            deployment_profile="tiny",
            graph_service_endpoints=[],
        ),
    ):
        monkeypatch.setattr(
            "agent_utilities.core.config.AgentConfig", lambda cfg=cfg: cfg
        )
        result = D._check_graph_identity()
        assert result["status"] == "fail"
        assert result["data"] == {"ready": False, "redacted": True}


def test_graph_identity_doctor_accepts_private_tiny_authority(monkeypatch):
    cfg = SimpleNamespace(
        kg_auth_token_ref=None,
        kg_identity_oauth2=None,
        deployment_profile="tiny",
        graph_service_endpoints=[],
    )
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: cfg)

    result = D._check_graph_identity()

    assert result["status"] == "ok"
    assert result["data"] == {
        "mode": "ephemeral_local",
        "ready": True,
        "redacted": True,
    }


@pytest.mark.parametrize(
    ("bundle_ready", "tls_profile", "tls_ready", "expected_status"),
    [
        (True, '{"system_trust":true,"trust_env":true}', True, "warn"),
        (False, '{"system_trust":true,"trust_env":true}', True, "fail"),
        (True, "malformed-trust-material", False, "fail"),
    ],
)
def test_transport_security_doctor_returns_only_redacted_readiness(
    monkeypatch, bundle_ready, tls_profile, tls_ready, expected_status
):
    connector = SimpleNamespace(
        name="domain-source",
        source_alias="domain-source",
        backend="neo4j",
        connection_profile_ref="secret://graphs/domain/connection",
        mapping_policy_ref=None,
        tls_profile_ref="secret://graphs/domain/tls",
        auth_profile_ref=None,
        ingest_page_size=250,
        ingest_max_pages=12,
        sync_mode="snapshot",
        reconcile_deletions=True,
        allow_empty_snapshot=False,
        semantic_mapping=False,
        require_approval=True,
        schema_drift_policy="fail_closed",
    )
    cfg = SimpleNamespace(
        tls_profile=None,
        tls_profile_ref=None,
        tls_profiles_ref=None,
        tls_ca_bundle_ref=None,
        tls_client_cert_ref=None,
        tls_client_key_ref=None,
        tls_client_key_password_ref=None,
        tls_proxy_url_ref=None,
        tls_system_trust=True,
        tls_trust_env=True,
        engine_tls_profile=None,
        engine_tls_profile_ref=None,
        graph_service_endpoints=[],
        external_graph_connectors=[connector],
    )
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: cfg)

    def resolver(ref: str) -> str:
        if ref.endswith("/connection"):
            return json.dumps({"uri": "neo4j+s://graph.example.test"})
        return tls_profile

    monkeypatch.setattr(
        "agent_utilities.security.secrets_client.create_secrets_client",
        lambda: SimpleNamespace(resolve_ref=resolver),
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.connector_manifest_gate.precheck_source",
        lambda source: {
            "checked": True,
            "ok": bundle_ready,
            "connector": "native-source-connectors",
        },
    )

    result = D._check_transport_security()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == expected_status
    assert result["data"]["external_graph_connectors"][0]["refs_ready"] == {
        "connection": True,
        "mapping": None,
        "tls": tls_ready,
        "auth": None,
        "variables": None,
    }
    assert (
        result["data"]["external_graph_connectors"][0]["mapping_lifecycle"]
        == "unavailable"
    )
    assert (
        result["data"]["external_graph_connectors"][0]["mapping_policy_drift"]
        == "unknown"
    )
    assert (
        result["data"]["external_graph_connectors"][0]["capability_bundle_ready"]
        is bundle_ready
    )
    assert result["data"]["external_graph_connectors"][0]["sync_policy"] == {
        "allow_empty_snapshot": False,
        "max_collection_items": 10_000,
        "max_nesting_depth": 16,
        "max_pages": 12,
        "max_row_bytes": 1_048_576,
        "max_total_bytes": 16_777_216,
        "page_size": 250,
        "reconcile_deletions": True,
        "sync_mode": "snapshot",
    }
    assert result["data"]["external_graph_source_aliases_unique"] is True
    assert result["data"]["external_graph_connection_names_unique"] is True
    assert "neo4j+s://" not in rendered
    assert "secret://" not in rendered
    assert "graph.example.test" not in rendered


def test_transport_security_doctor_resolves_env_profile_without_secret_backend(
    monkeypatch,
):
    monkeypatch.setenv(
        "TEST_TLS_PROFILE_CATALOG",
        json.dumps(
            {
                "profiles": {
                    "private-service": {
                        "system_trust": True,
                        "trust_env": True,
                    }
                }
            }
        ),
    )
    cfg = SimpleNamespace(
        tls_profile="private-service",
        tls_profile_ref=None,
        tls_profiles_ref="env://TEST_TLS_PROFILE_CATALOG",
        tls_ca_bundle_ref=None,
        tls_client_cert_ref=None,
        tls_client_key_ref=None,
        tls_client_key_password_ref=None,
        tls_proxy_url_ref=None,
        tls_system_trust=True,
        tls_trust_env=True,
        engine_tls_profile=None,
        engine_tls_profile_ref=None,
        graph_service_endpoints=[],
        external_graph_connectors=[],
    )
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: cfg)
    with patch(
        "agent_utilities.security.secrets_client.create_secrets_client",
        side_effect=AssertionError("env references must not open a secret backend"),
    ) as create_client:
        result = D._check_transport_security()

    create_client.assert_not_called()
    assert result["status"] == "ok"
    assert result["data"]["tls"]["verify_enabled"] is True


@pytest.mark.parametrize("invalid_ref", ["connection", "mapping", "auth"])
@pytest.mark.parametrize("format_mode", ["missing", "unknown"])
def test_transport_doctor_rejects_missing_or_unknown_graphql_runtime_formats(
    monkeypatch, invalid_ref, format_mode
):
    connector = SimpleNamespace(
        name="schema-source",
        source_alias="schema-source",
        backend="graphql",
        connection_profile_ref="secret://schema/connection",
        mapping_policy_ref="secret://schema/mapping",
        tls_profile_ref="secret://schema/tls",
        auth_profile_ref="secret://schema/auth",
        variables_ref=None,
        allow_introspection=False,
        allow_empty_snapshot=False,
        semantic_mapping=False,
        require_approval=True,
        schema_drift_policy="fail_closed",
    )
    cfg = SimpleNamespace(
        tls_profile=None,
        tls_profile_ref=None,
        tls_profiles_ref=None,
        tls_ca_bundle_ref=None,
        tls_client_cert_ref=None,
        tls_client_key_ref=None,
        tls_client_key_password_ref=None,
        tls_proxy_url_ref=None,
        tls_system_trust=True,
        tls_trust_env=True,
        engine_tls_profile=None,
        engine_tls_profile_ref=None,
        graph_service_endpoints=[],
        external_graph_connectors=[connector],
    )
    documents = {
        "connection": {
            "profile_format": "graphql-connection/v1",
            "endpoint": "https://source.example.test/graphql",
        },
        "mapping": {
            "profile_format": "graphql-document-policy/v1",
            "discovery": {"enabled": True},
            "operations": {},
        },
        "auth": {
            "profile_format": "graphql-auth/v1",
            "headers": {},
        },
        "tls": {"system_trust": True, "trust_env": True},
    }
    if format_mode == "missing":
        documents[invalid_ref].pop("profile_format")
    else:
        documents[invalid_ref]["profile_format"] = "unsupported/v2"

    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: cfg)
    secrets_client = SimpleNamespace(
        resolve_ref=lambda ref: json.dumps(documents[ref.rsplit("/", 1)[-1]])
    )
    monkeypatch.setattr(
        "agent_utilities.security.secrets_client.create_secrets_client",
        lambda: secrets_client,
    )

    result = D._check_transport_security()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "fail"
    readiness = result["data"]["external_graph_connectors"][0]["refs_ready"]
    assert readiness[invalid_ref] is False
    assert readiness["runtime_contract"] is False
    assert "source.example.test" not in rendered
    assert "secret://" not in rendered


@pytest.mark.parametrize(
    ("client_id", "broker_url", "expected"),
    [
        (None, None, "skip"),
        ("synthetic-client", None, "fail"),
        (None, "https://oauth-broker.example.test", "fail"),
        ("synthetic-client", "https://oauth-broker.example.test", "ok"),
    ],
)
def test_google_workspace_oauth_doctor_is_redacted(
    monkeypatch, client_id, broker_url, expected
):
    cfg = SimpleNamespace(
        google_workspace_oauth_client_id=client_id,
        google_workspace_oauth_broker_url=broker_url,
    )
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: cfg)

    result = D._check_google_workspace_oauth()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == expected
    assert "synthetic-client" not in rendered
    assert "oauth-broker.example.test" not in rendered


def test_engine_doctor_never_returns_endpoint_or_socket_material(monkeypatch):
    cfg = SimpleNamespace()
    endpoint = "unix:///private/machine/path/engine.sock"
    resolved = SimpleNamespace(
        mode="remote",
        endpoint=endpoint,
        autostart_allowed=False,
        idle_shutdown_secs=0,
    )
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: cfg)
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.shard_topology.default_graph_name",
        lambda _cfg: "default",
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.shard_topology.shard_topology_status",
        lambda *_args, **_kwargs: {
            "mode": "single",
            "endpoints": [{"endpoint": endpoint, "reachable": True}],
        },
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.engine_resolver.resolve_engine",
        lambda *_args, **_kwargs: resolved,
    )

    result = D._check_engine()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "ok"
    assert result["data"]["reachable_endpoint_count"] == 1
    assert result["data"]["durable_encryption"] == {
        "ready": True,
        "source": "remote_managed",
        "material_exposed": False,
    }
    assert result["data"]["redacted"] is True
    assert endpoint not in rendered
    assert "/private/machine" not in rendered


def _skill_certification_config(
    root: Path, *, complete: bool = True
) -> SimpleNamespace:
    endpoint = "http://localhost:8000/mcp"
    material = {
        name: root / name
        for name in ("config.json", "profile.json", "spec.json", "promotion.json")
    }
    commands = {}
    for name in ("graph-os", "evidence-signer", "evidence-verifier"):
        executable = root / name
        executable.write_text("synthetic executable", encoding="utf-8")
        executable.chmod(0o700)
        commands[name] = [str(executable)]
    if complete:
        for path in material.values():
            path.write_text("{}", encoding="utf-8")
    return SimpleNamespace(
        skill_cert_runtime_configuration=str(material["config.json"]),
        skill_cert_runtime_profile=(
            str(material["profile.json"]) if complete else None
        ),
        skill_cert_release_spec=str(material["spec.json"]) if complete else None,
        skill_cert_promotion_evidence=(
            str(material["promotion.json"]) if complete else None
        ),
        skill_cert_graphos_endpoint=endpoint if complete else None,
        skill_cert_graphos_command=commands["graph-os"] if complete else [],
        skill_validation_evidence_signer_command=(
            commands["evidence-signer"] if complete else []
        ),
        skill_validation_evidence_verifier_command=(
            commands["evidence-verifier"] if complete else []
        ),
        skill_cert_identity_authority_mode="ephemeral-https-loopback",
        skill_cert_identity_token_ttl_seconds=300,
        mcp_url=endpoint,
    )


def test_skill_certification_doctor_skips_only_when_wholly_unconfigured(
    monkeypatch,
):
    from agent_utilities.core import config as config_module

    cfg = SimpleNamespace(
        skill_cert_runtime_configuration=None,
        skill_cert_runtime_profile=None,
        skill_cert_release_spec=None,
        skill_cert_promotion_evidence=None,
        skill_cert_graphos_endpoint=None,
        skill_cert_graphos_command=[],
        skill_validation_evidence_signer_command=[],
        skill_validation_evidence_verifier_command=[],
        skill_cert_identity_authority_mode="ephemeral-https-loopback",
        skill_cert_identity_token_ttl_seconds=300,
    )
    monkeypatch.setattr(config_module, "AgentConfig", lambda: cfg)

    result = D._check_skill_certification()

    assert result["status"] == "skip"
    assert result["data"] == {
        "configured_count": 0,
        "required_count": 8,
        "ready": False,
        "redacted": True,
    }


def test_skill_certification_doctor_rejects_partial_configuration(
    tmp_path, monkeypatch
):
    from agent_utilities.core import config as config_module

    cfg = _skill_certification_config(tmp_path, complete=False)
    monkeypatch.setattr(config_module, "AgentConfig", lambda: cfg)

    result = D._check_skill_certification()

    assert result["status"] == "fail"
    assert result["data"]["configured_count"] == 1
    assert result["data"]["redacted"] is True
    assert str(tmp_path) not in json.dumps(result, sort_keys=True)


def test_skill_certification_doctor_proves_regular_material_and_commands(
    tmp_path, monkeypatch
):
    from agent_utilities.core import config as config_module
    from agent_utilities.core import paths
    from agent_utilities.deployment import skill_validation_assets as assets

    cfg = _skill_certification_config(tmp_path)
    monkeypatch.setattr(config_module, "AgentConfig", lambda: cfg)
    monkeypatch.setattr(paths, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(
        assets,
        "_configuration_proof",
        lambda _payload: {"digest": "sha256:" + "a" * 64},
    )
    monkeypatch.setattr(assets, "_validate_profile", lambda *_args, **_kwargs: None)

    result = D._check_skill_certification()

    assert result["status"] == "ok"
    assert result["data"] == {
        "configured_count": 8,
        "required_count": 8,
        "regular_input_count": 4,
        "command_count": 3,
        "identity_authority_mode": "ephemeral-https-loopback",
        "identity_authority_lifecycle_owned": True,
        "identity_authority_tls_verification_required": True,
        "identity_authority_renewable_credentials_required": True,
        "ready": True,
        "redacted": True,
    }
    assert str(tmp_path) not in json.dumps(result, sort_keys=True)


def test_skill_certification_doctor_fails_closed_without_exposing_material(
    tmp_path, monkeypatch
):
    from agent_utilities.core import config as config_module

    cfg = _skill_certification_config(tmp_path)
    Path(cfg.skill_cert_promotion_evidence).unlink()
    monkeypatch.setattr(config_module, "AgentConfig", lambda: cfg)

    result = D._check_skill_certification()

    assert result["status"] == "fail"
    assert result["data"]["configured_count"] == 8
    assert result["data"]["redacted"] is True
    rendered = json.dumps(result, sort_keys=True)
    assert str(tmp_path) not in rendered
    assert "promotion.json" not in rendered


def _signed_release_fixture(*, matrix_digest: str) -> dict:
    from scripts.release import check_compatibility

    unsigned = {
        "apiVersion": "graphos.io/v1",
        "kind": "ReleaseManifest",
        "manifestState": "signed-release",
        "releaseId": "release-doctor-fixture",
        "matrixDigest": matrix_digest,
        "configurationDigest": "sha256:" + "1" * 64,
        "protocolSchemas": {},
        "components": {},
        "migrationPlanDigest": "sha256:" + "2" * 64,
        "certificationDigests": {},
        "exactGateEvidence": {},
        "evidence": {},
    }
    return {
        **unsigned,
        "signature": {
            "scheme": "fixture-signature",
            "subjectDigest": check_compatibility.canonical_digest(unsigned),
            "bundleDigest": "sha256:" + "3" * 64,
            "signerIdentityDigest": "sha256:" + "4" * 64,
            "value": "fixture-signature-value",
            "verifierEnv": "RELEASE_VERIFIER_COMMAND",
        },
    }


def _production_certification_config(
    root: Path,
    *,
    state: str = "complete",
    release_document: dict | None = None,
) -> SimpleNamespace:
    release = root / "release.json"
    artifacts = root / "artifacts"
    executable = root / "certification-command"
    from scripts.release import check_compatibility

    matrix_path = Path(__file__).resolve().parents[3] / (
        "deploy/release/compatibility-matrix.yml"
    )
    if release_document is None:
        release_document = _signed_release_fixture(
            matrix_digest=check_compatibility.file_digest(matrix_path)
        )
    release.write_text(json.dumps(release_document), encoding="utf-8")
    artifacts.mkdir(mode=0o700)
    artifacts.chmod(0o700)
    executable.write_text("synthetic executable", encoding="utf-8")
    executable.chmod(0o700)
    command = [str(executable)]
    load_command = [str(executable), "--report", "{report_file}"]
    from agent_utilities.core.config import PRODUCTION_CERTIFICATION_SCENARIOS

    command_map = {scenario: command for scenario in PRODUCTION_CERTIFICATION_SCENARIOS}
    if state == "disabled":
        return SimpleNamespace(
            certification_mode="disabled",
            cert_release_manifest=None,
            cert_artifacts_dir=None,
            cert_hardware_class=None,
            cert_load_command=[],
            cert_metrics_command=[],
            cert_hook_commands={},
            cert_fault_action_commands={},
            cert_fault_probe_commands={},
            cert_evidence_signer_command=[],
            cert_evidence_verifier_command=[],
            cert_prometheus_url=None,
            cert_prometheus_bearer_token_ref=None,
            cert_prometheus_tls_profile=None,
            cert_prometheus_tls_profile_ref=None,
        )
    if state == "partial":
        return SimpleNamespace(
            certification_mode="disabled",
            cert_release_manifest=str(release),
            cert_artifacts_dir=None,
            cert_hardware_class=None,
            cert_load_command=[],
            cert_metrics_command=[],
            cert_hook_commands={},
            cert_fault_action_commands={},
            cert_fault_probe_commands={},
            cert_evidence_signer_command=[],
            cert_evidence_verifier_command=[],
            cert_prometheus_url=None,
            cert_prometheus_bearer_token_ref=None,
            cert_prometheus_tls_profile=None,
            cert_prometheus_tls_profile_ref=None,
        )
    return SimpleNamespace(
        certification_mode="production",
        cert_release_manifest=str(release),
        cert_artifacts_dir=str(artifacts),
        cert_hardware_class="capacity-standard",
        cert_load_command=load_command,
        cert_metrics_command=command,
        cert_hook_commands=command_map,
        cert_fault_action_commands=command_map,
        cert_fault_probe_commands=command_map,
        cert_evidence_signer_command=command,
        cert_evidence_verifier_command=command,
        cert_prometheus_url="https://metrics.example.test",
        cert_prometheus_bearer_token_ref=None,
        cert_prometheus_tls_profile="prometheus-production",
        cert_prometheus_tls_profile_ref=None,
    )


def _patch_valid_release_verification(monkeypatch) -> MagicMock:
    from scripts.release import check_compatibility

    def verify(manifest, _matrix, **kwargs):
        assert kwargs["verify_signatures"] is True
        assert kwargs["manifest_path"].name == "release.json"
        assert (
            check_compatibility.file_digest(kwargs["matrix_path"])
            == manifest["matrixDigest"]
        )
        check_compatibility._validate_manifest_signature(manifest)
        return {"ok": True, "signaturesVerified": True}

    verifier = MagicMock(side_effect=verify)
    monkeypatch.setattr(check_compatibility, "verify_release_manifest", verifier)
    return verifier


def test_production_certification_doctor_skips_only_when_wholly_unconfigured(
    tmp_path, monkeypatch
):
    from agent_utilities.core import config as config_module

    cfg = _production_certification_config(tmp_path, state="disabled")
    monkeypatch.setattr(config_module, "AgentConfig", lambda: cfg)

    result = D._check_production_certification()

    assert result["status"] == "skip"
    assert result["data"] == {
        "configured_count": 0,
        "required_count": 13,
        "scenario_count": 0,
        "command_count": 0,
        "bearer_auth_configured": False,
        "ready": False,
        "redacted": True,
    }


def test_production_certification_doctor_rejects_partial_configuration(
    tmp_path, monkeypatch
):
    from agent_utilities.core import config as config_module

    cfg = _production_certification_config(tmp_path, state="partial")
    monkeypatch.setattr(config_module, "AgentConfig", lambda: cfg)

    result = D._check_production_certification()

    assert result["status"] == "fail"
    assert result["data"]["configured_count"] == 1
    assert result["data"]["redacted"] is True
    assert str(tmp_path) not in json.dumps(result, sort_keys=True)


def test_production_certification_doctor_proves_complete_redacted_authority(
    tmp_path, monkeypatch
):
    from agent_utilities.core import config as config_module
    from agent_utilities.core import transport_security

    cfg = _production_certification_config(tmp_path)
    trust = SimpleNamespace(verify_enabled=True, cleanup=lambda: None)
    monkeypatch.setattr(config_module, "AgentConfig", lambda: cfg)
    monkeypatch.setattr(
        transport_security,
        "resolve_configured_tls_profile",
        lambda *_args, **_kwargs: trust,
    )
    verifier = _patch_valid_release_verification(monkeypatch)

    result = D._check_production_certification()

    assert result["status"] == "ok"
    assert result["data"] == {
        "configured_count": 13,
        "required_count": 13,
        "scenario_count": 15,
        "command_count": 49,
        "bearer_auth_configured": False,
        "prometheus_tls_verification_required": True,
        "ready": True,
        "redacted": True,
    }
    assert str(tmp_path) not in json.dumps(result, sort_keys=True)
    verifier.assert_called_once()


@pytest.mark.parametrize("release_state", ("invalid", "unsigned", "wrong-matrix"))
def test_production_certification_doctor_rejects_untrusted_release_redacted(
    tmp_path, monkeypatch, release_state
):
    from agent_utilities.core import config as config_module
    from scripts.release import check_compatibility

    matrix_path = Path(__file__).resolve().parents[3] / (
        "deploy/release/compatibility-matrix.yml"
    )
    document = _signed_release_fixture(
        matrix_digest=check_compatibility.file_digest(matrix_path)
    )
    if release_state == "invalid":
        document = {}
    elif release_state == "unsigned":
        document.pop("signature")
        document["manifestState"] = "unsigned-local-binder"
    else:
        document["matrixDigest"] = "sha256:" + "9" * 64
        unsigned = {key: value for key, value in document.items() if key != "signature"}
        document["signature"]["subjectDigest"] = check_compatibility.canonical_digest(
            unsigned
        )
    cfg = _production_certification_config(
        tmp_path,
        release_document=document,
    )
    monkeypatch.setattr(config_module, "AgentConfig", lambda: cfg)

    result = D._check_production_certification()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "fail"
    assert result["data"]["redacted"] is True
    assert str(tmp_path) not in rendered
    if document.get("matrixDigest"):
        assert document["matrixDigest"] not in rendered
    assert "fixture-signature-value" not in rendered


def test_production_certification_doctor_rejects_nonempty_artifacts_redacted(
    tmp_path, monkeypatch
):
    from agent_utilities.core import config as config_module

    cfg = _production_certification_config(tmp_path)
    (Path(cfg.cert_artifacts_dir) / "private.json").write_text(
        "private", encoding="utf-8"
    )
    monkeypatch.setattr(config_module, "AgentConfig", lambda: cfg)
    _patch_valid_release_verification(monkeypatch)

    result = D._check_production_certification()

    assert result["status"] == "fail"
    assert result["data"]["redacted"] is True
    rendered = json.dumps(result, sort_keys=True)
    assert str(tmp_path) not in rendered
    assert "private.json" not in rendered


@pytest.mark.parametrize(("ready", "expected"), [(True, "warn"), (False, "fail")])
def test_engine_doctor_reports_local_encryption_readiness_without_material(
    monkeypatch, ready, expected
):
    cfg = SimpleNamespace(
        graph_raft_group_endpoints={},
        epistemic_graph_encryption_key_ref="env://PRIVATE_ENGINE_DATA_KEY",
    )
    resolved = SimpleNamespace(
        mode="autostart",
        autostart_allowed=True,
        idle_shutdown_secs=60,
    )
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: cfg)
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.shard_topology.default_graph_name",
        lambda _cfg: "default",
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.shard_topology.shard_topology_status",
        lambda *_args, **_kwargs: {
            "mode": "single",
            "endpoints": [{"endpoint": "private-runtime-endpoint", "reachable": False}],
        },
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.engine_resolver.resolve_engine",
        lambda *_args, **_kwargs: resolved,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.graph_compute.engine_encryption_readiness",
        lambda *_args, **_kwargs: {
            "ready": ready,
            "source": "explicit_reference",
            "material_exposed": False,
        },
    )

    result = D._check_engine()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == expected
    assert result["data"]["durable_encryption"]["ready"] is ready
    assert result["data"]["durable_encryption"]["material_exposed"] is False
    assert "PRIVATE_ENGINE_DATA_KEY" not in rendered
    assert "private-runtime-endpoint" not in rendered


def test_engine_request_context_doctor_reports_current_only_contract():
    result = D._check_engine_request_context()

    assert result["status"] == "ok"
    assert result["data"] == {
        "verified_context_required": True,
        "legacy_protocol_available": False,
        "unauthenticated_transport_available": False,
    }


def test_secret_doctor_reports_counts_not_reference_names(monkeypatch):
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: object())
    monkeypatch.setattr(
        "agent_utilities.core.config.runtime_secret_source_status",
        lambda: {
            "state": "ready",
            "present": True,
            "valid": True,
            "referenced_count": 2,
            "matched_count": 0,
            "projected_count": 0,
            "overridden_count": 0,
        },
    )
    monkeypatch.setattr(
        "agent_utilities.deployment.config_generator._unresolved_secret_refs",
        lambda _cfg: ["PRIVATE_TOKEN", "CUSTOMER_CA_PATH"],
    )

    result = D._check_secrets()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "fail"
    assert result["data"] == {
        "runtime_source": {
            "state": "ready",
            "present": True,
            "valid": True,
            "referenced_count": 2,
            "matched_count": 0,
            "projected_count": 0,
            "overridden_count": 0,
        },
        "unresolved_count": 2,
        "redacted": True,
    }
    assert "PRIVATE_TOKEN" not in rendered
    assert "CUSTOMER_CA_PATH" not in rendered


def test_secret_doctor_fails_closed_when_backend_evaluation_raises(monkeypatch):
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: object())
    monkeypatch.setattr(
        "agent_utilities.deployment.config_generator._unresolved_secret_refs",
        lambda _cfg: (_ for _ in ()).throw(RuntimeError("private detail")),
    )

    result = D._check_secrets()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "fail"
    assert result["data"]["redacted"] is True
    assert "private detail" not in rendered


def _patch_secrets_backend_scan(monkeypatch, *, backend, counts):
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: SimpleNamespace(secrets_backend=backend),
    )
    monkeypatch.setattr(
        "agent_utilities.deployment.config_generator.secret_reference_scheme_counts",
        lambda _cfg: counts,
    )


def test_secrets_backend_doctor_fails_when_engine_backend_has_vault_refs(
    monkeypatch,
):
    _patch_secrets_backend_scan(
        monkeypatch, backend="engine", counts={"env": 0, "vault": 2, "secret": 0}
    )

    result = D._check_secrets_backend()

    assert result["status"] == "fail"
    assert result["data"] == {
        "configured_backend": "engine",
        "vault_scheme_ref_count": 2,
        "secret_scheme_ref_count": 0,
        "redacted": True,
    }
    assert "SECRETS_BACKEND=vault" in result["remediation"]


def test_secrets_backend_doctor_passes_when_vault_backend_has_vault_refs(
    monkeypatch,
):
    _patch_secrets_backend_scan(
        monkeypatch, backend="vault", counts={"env": 0, "vault": 3, "secret": 0}
    )

    result = D._check_secrets_backend()

    assert result["status"] == "ok"
    assert result["data"]["configured_backend"] == "vault"


def test_secrets_backend_doctor_passes_when_engine_backend_has_only_secret_refs(
    monkeypatch,
):
    _patch_secrets_backend_scan(
        monkeypatch, backend="engine", counts={"env": 1, "vault": 0, "secret": 2}
    )

    result = D._check_secrets_backend()

    assert result["status"] == "ok"


def test_secrets_backend_doctor_warns_when_vault_backend_has_secret_scheme_refs(
    monkeypatch,
):
    _patch_secrets_backend_scan(
        monkeypatch, backend="vault", counts={"env": 0, "vault": 0, "secret": 1}
    )

    result = D._check_secrets_backend()

    assert result["status"] == "warn"
    assert result["data"]["secret_scheme_ref_count"] == 1


def test_secrets_backend_doctor_prefers_fail_when_both_directions_present(
    monkeypatch,
):
    """A vault:// mismatch is the more severe finding; it wins over the warn."""
    _patch_secrets_backend_scan(
        monkeypatch, backend="engine", counts={"env": 0, "vault": 1, "secret": 5}
    )

    result = D._check_secrets_backend()

    assert result["status"] == "fail"


def test_secrets_backend_doctor_ok_with_no_references_at_all(monkeypatch):
    _patch_secrets_backend_scan(
        monkeypatch, backend="engine", counts={"env": 0, "vault": 0, "secret": 0}
    )

    result = D._check_secrets_backend()

    assert result["status"] == "ok"


def test_secrets_backend_doctor_fails_closed_when_scan_raises(monkeypatch):
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: object())
    monkeypatch.setattr(
        "agent_utilities.deployment.config_generator.secret_reference_scheme_counts",
        lambda _cfg: (_ for _ in ()).throw(RuntimeError("private detail")),
    )

    result = D._check_secrets_backend()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "fail"
    assert result["data"]["redacted"] is True
    assert "private detail" not in rendered


def test_secrets_backend_doctor_never_leaks_reference_paths(monkeypatch):
    """Only counts/backend name may appear — never the reference path/value."""
    _patch_secrets_backend_scan(
        monkeypatch, backend="engine", counts={"env": 0, "vault": 1, "secret": 0}
    )

    result = D._check_secrets_backend()
    rendered = json.dumps(result, sort_keys=True)

    assert "apps/" not in rendered
    assert result["data"]["redacted"] is True


@pytest.mark.parametrize("selection", [[], ["unknown"], ["config", "config"]])
def test_doctor_rejects_empty_unknown_or_duplicate_selection(selection):
    result = D.run_doctor(selection)
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "unhealthy"
    assert result["counts"] == {"error": 1}
    assert result["checks"][0]["name"] == "selection"
    assert result["checks"][0]["data"] == {"redacted": True}
    assert "unknown" not in rendered


def test_mcp_fleet_doctor_returns_only_aggregate_readiness(tmp_path, monkeypatch):
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text('{"mcpServers": {}}', encoding="utf-8")
    private_endpoint = "https://customer-internal.example.test/mcp"
    report = {
        "total": 2,
        "ok": ["public-server"],
        "invalid": {"private-server": private_endpoint},
        "unreachable": {},
        "missing_from_config": ["private-mcp.example.test"],
        "passed": False,
    }
    fake_validator = SimpleNamespace(
        validate=lambda *_args, **_kwargs: report,
        caddy_hosts=lambda: {},
    )
    monkeypatch.setattr(
        "agent_utilities.core.workspace.get_mcp_config_path",
        lambda: str(config_path),
    )
    monkeypatch.setattr(D.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(D.importlib, "import_module", lambda _name: fake_validator)

    result = D._check_mcp_fleet(live=False)
    rendered = json.dumps(result, sort_keys=True)

    assert result["data"]["invalid_count"] == 1
    assert result["data"]["missing_route_count"] == 1
    assert result["data"]["redacted"] is True
    assert private_endpoint not in rendered
    assert "private-server" not in rendered


def test_mcp_fleet_secret_doctor_reports_only_aggregate_readiness(monkeypatch):
    direct_material = "direct-runtime-material"
    mapped_material = "mapped-runtime-material"
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: SimpleNamespace(
            mcp_fleet_secret_refs={
                "DIRECT_ALIAS": "secret://fleet/direct",
                "MAPPED_ALIAS": "secret://fleet/mapped",
            }
        ),
    )
    monkeypatch.setattr(
        "agent_utilities.core.config.setting",
        lambda alias: direct_material if alias == "DIRECT_ALIAS" else None,
    )
    monkeypatch.setattr(
        "agent_utilities.security.cli_secrets.resolve_runtime_secret_reference",
        lambda _reference: mapped_material,
    )

    result = D._check_mcp_fleet_secrets()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "ok"
    assert result["data"] == {
        "configured_alias_count": 2,
        "direct_alias_count": 1,
        "mapped_alias_count": 1,
        "unresolved_alias_count": 0,
        "redacted": True,
    }
    assert "DIRECT_ALIAS" not in rendered
    assert "MAPPED_ALIAS" not in rendered
    assert direct_material not in rendered
    assert mapped_material not in rendered


def test_mcp_fleet_secret_doctor_fails_closed_and_redacted(monkeypatch):
    private_reference = "secret://fleet/private/value"
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: SimpleNamespace(
            mcp_fleet_secret_refs={"PRIVATE_ALIAS": private_reference}
        ),
    )
    monkeypatch.setattr("agent_utilities.core.config.setting", lambda _alias: None)
    monkeypatch.setattr(
        "agent_utilities.security.cli_secrets.resolve_runtime_secret_reference",
        lambda _reference: (_ for _ in ()).throw(RuntimeError("private detail")),
    )

    result = D._check_mcp_fleet_secrets()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "fail"
    assert result["data"]["unresolved_alias_count"] == 1
    assert "PRIVATE_ALIAS" not in rendered
    assert private_reference not in rendered
    assert "private detail" not in rendered


def test_provider_profile_doctor_reports_only_aggregate_readiness(monkeypatch):
    profiles = {
        "provider-one": SimpleNamespace(enabled=True),
        "provider-two": SimpleNamespace(enabled=False),
    }
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: SimpleNamespace(provider_configs=profiles),
    )
    monkeypatch.setattr(
        "agent_utilities.core.provider_runtime.prepare_provider_runtime_child_environment",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )

    result = D._check_provider_profiles()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "ok"
    assert result["data"] == {
        "configured_count": 2,
        "enabled_count": 1,
        "disabled_count": 1,
        "ready_count": 1,
        "invalid_count": 0,
        "redacted": True,
    }
    assert "provider-one" not in rendered
    assert "provider-two" not in rendered


def test_provider_profile_doctor_fails_closed_without_runtime_details(monkeypatch):
    private_detail = "private-provider-runtime-detail"
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig",
        lambda: SimpleNamespace(
            provider_configs={"private-provider": SimpleNamespace(enabled=True)}
        ),
    )
    monkeypatch.setattr(
        "agent_utilities.core.provider_runtime.prepare_provider_runtime_child_environment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(private_detail)),
    )

    result = D._check_provider_profiles()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "fail"
    assert result["data"]["invalid_count"] == 1
    assert "private-provider" not in rendered
    assert private_detail not in rendered


def test_runtime_integrations_doctor_reports_only_redacted_readiness(
    tmp_path, monkeypatch
):
    inventory = tmp_path / "private-inventory.yml"
    inventory.write_text("all: {}", encoding="utf-8")
    private_template = "https://private-gateway.example.test/{server}/mcp"
    private_media_endpoint = "https://private-media.example.test/api"
    cfg = SimpleNamespace(
        infra_inventory_path=str(inventory),
        fleet_mcp_url_template=private_template,
        comfyui_url=private_media_endpoint,
        xtts_url=None,
        openai_tts_url=None,
        whisper_url=None,
        faster_whisper_url=None,
        flux_url=None,
        sd35_url=None,
        hunyuan_url=None,
        svd_url=None,
    )
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: cfg)

    result = D._check_runtime_integrations()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "ok"
    assert result["data"]["configured_category_count"] == 3
    assert result["data"]["ready_category_count"] == 3
    assert result["data"]["inventory_format_ready"] is True
    assert result["data"]["media_endpoint_configured_count"] == 1
    assert result["data"]["network_probed"] is False
    assert result["data"]["redacted"] is True
    assert str(inventory) not in rendered
    assert private_template not in rendered
    assert private_media_endpoint not in rendered


def test_runtime_integrations_doctor_warns_without_emitting_missing_path(monkeypatch):
    missing_path = "synthetic-missing-inventory.yml"
    cfg = SimpleNamespace(
        infra_inventory_path=missing_path,
        fleet_mcp_url_template=None,
        comfyui_url=None,
        xtts_url=None,
        openai_tts_url=None,
        whisper_url=None,
        faster_whisper_url=None,
        flux_url=None,
        sd35_url=None,
        hunyuan_url=None,
        svd_url=None,
    )
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: cfg)

    result = D._check_runtime_integrations()

    assert result["status"] == "warn"
    assert result["data"]["inventory_file_ready"] is False
    assert result["data"]["inventory_format_ready"] is False
    assert missing_path not in json.dumps(result, sort_keys=True)


def test_runtime_integrations_doctor_redacts_invalid_config_value(monkeypatch):
    private_value = "https://private-runtime.example.test?identity=hidden"

    def invalid_config():
        raise ValueError(private_value)

    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", invalid_config)

    result = D._check_runtime_integrations()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "fail"
    assert result["data"] == {"ready": False, "redacted": True}
    assert private_value not in rendered
    assert "identity=hidden" not in rendered


def test_skills_check_requires_exact_suite_without_persisting_local_path(
    tmp_path, monkeypatch
):
    from agent_utilities.core import paths, providers
    from agent_utilities.core.provider_materialization import build_asset_manifest

    source = tmp_path / "source"
    (source / "graph-query-and-explanation").mkdir(parents=True)
    (source / "graph-query-and-explanation" / "SKILL.md").write_text(
        "synthetic", encoding="utf-8"
    )
    xdg = tmp_path / "xdg"
    xdg.mkdir()
    manifest = build_asset_manifest(source, leg="skills")
    monkeypatch.setattr(paths, "skills_dir", lambda: xdg)
    monkeypatch.setattr(providers, "current_provider_assets", lambda _group: ())
    monkeypatch.setattr(
        providers,
        "_own_skill_assets",
        lambda: ("agent-utilities", "a" * 64, source, manifest),
    )

    result = D._check_skills()

    assert result["status"] == "warn"
    assert "graph-ingestion-and-integration" in result["data"]["missing"]
    assert str(tmp_path) not in json.dumps(result)


def test_skills_check_accepts_the_complete_consolidated_suite(tmp_path, monkeypatch):
    from agent_utilities.core import paths, providers
    from agent_utilities.core.provider_materialization import build_asset_manifest
    from agent_utilities.skills import BUNDLED_SKILLS

    source = tmp_path / "source"
    for name in BUNDLED_SKILLS:
        skill_dir = source / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("synthetic", encoding="utf-8")
    xdg = tmp_path / "xdg"
    xdg.mkdir()
    manifest = build_asset_manifest(source, leg="skills")
    monkeypatch.setattr(paths, "skills_dir", lambda: xdg)
    monkeypatch.setattr(providers, "current_provider_assets", lambda _group: ())
    monkeypatch.setattr(
        providers,
        "_own_skill_assets",
        lambda: ("agent-utilities", "a" * 64, source, manifest),
    )

    result = D._check_skills()

    assert result["status"] == "ok"
    assert result["data"]["required"] == 10
    assert str(tmp_path) not in json.dumps(result)


def test_evolution_staging_doctor_reports_readiness_without_path(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    cfg = SimpleNamespace(evolution_staging_root=str(tmp_path))
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: cfg)

    result = D._check_evolution_staging()

    assert result["status"] == "ok"
    assert result["data"] == {"configured": True, "usable": True}
    assert str(tmp_path) not in json.dumps(result, sort_keys=True)


def test_observability_doctor_derives_redacted_langfuse_otlp(monkeypatch):
    monkeypatch.setenv("ENABLE_OTEL", "true")
    monkeypatch.setenv("APP_PROFILE", "dev")
    monkeypatch.setenv("LANGFUSE_HOST", "https://telemetry.example.test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY_REF", "env://TEST_LANGFUSE_PUBLIC")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY_REF", "env://TEST_LANGFUSE_SECRET")
    monkeypatch.setenv("TEST_LANGFUSE_PUBLIC", "synthetic-public")
    monkeypatch.setenv("TEST_LANGFUSE_SECRET", "synthetic-secret")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    result = D._check_observability()
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "ok"
    assert result["data"]["endpoint_derived_from_langfuse"] is True
    assert result["data"]["metadata_only"] is True
    assert result["data"]["redacted"] is True
    assert "synthetic-public" not in rendered
    assert "synthetic-secret" not in rendered
    assert "telemetry.example.test" not in rendered


def _langfuse_doctor_config(**overrides):
    values = {
        "langfuse_public_key_ref": "env://TEST_LANGFUSE_PUBLIC",
        "langfuse_secret_key_ref": "env://TEST_LANGFUSE_SECRET",
        "langfuse_tls_profile": None,
        "langfuse_tls_profile_ref": None,
        "langfuse_host": "https://telemetry.example.test",
        "langfuse_persistence_hmac_key_ref": None,
        "langfuse_kg_auto_ingest": False,
        "langfuse_mcp_enabled": False,
        "kg_failure_evolution": True,
        "trace_export_enabled": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_langfuse_static_dependencies(monkeypatch, cfg, *, launcher=None):
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: cfg)
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.langfuse_credentials_configured",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.resolve_langfuse_credentials",
        lambda **_kwargs: ("synthetic-public", "synthetic-secret"),
    )
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.configure_langfuse_trust",
        lambda **_kwargs: SimpleNamespace(valid=True, configured=False),
    )
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.resolve_langfuse_persistence_hmac_key",
        lambda **_kwargs: "synthetic-persistence-key-material-32",
    )
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.langfuse_provider_contract_ready",
        lambda: launcher is not None,
    )


def test_langfuse_doctor_accepts_direct_key_pair_without_refs(monkeypatch):
    """langfuse-agent's own client reads LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY
    directly (no _REF suffix). The doctor must not report the integration as
    unconfigured/incomplete when only that direct pair is set."""
    cfg = _langfuse_doctor_config(
        langfuse_public_key_ref=None,
        langfuse_secret_key_ref=None,
        kg_failure_evolution=False,
        trace_export_enabled=False,
    )
    _patch_langfuse_static_dependencies(monkeypatch, cfg, launcher=None)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-synthetic")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-synthetic")

    result = D._check_langfuse()

    # Without the direct-key fallback this would report "skip" (integration not
    # configured) or "fail" (incomplete pair) even though real credentials are
    # set. No integration is enabled here, so a fully-recognized pair still
    # resolves through to the "ready but disabled" warn -- proving the direct
    # LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY pair was recognized and resolved.
    assert result["status"] == "warn"
    assert result["data"]["credential_pair_configured"] is True
    assert result["data"]["credential_material_ready"] is True


def test_langfuse_launcher_is_required_only_for_mcp(monkeypatch):
    cfg = _langfuse_doctor_config(langfuse_mcp_enabled=False)
    _patch_langfuse_static_dependencies(monkeypatch, cfg, launcher=None)

    result = D._check_langfuse()

    assert result["status"] == "ok"
    assert result["data"]["mcp_launcher_required"] is False
    assert result["data"]["mcp_launcher_available"] is False
    assert result["data"]["live_probed"] is False


def test_langfuse_doctor_rejects_placeholder_material_locally(monkeypatch):
    from agent_utilities.observability.langfuse_trust import LangfuseTrustError

    cfg = _langfuse_doctor_config()
    _patch_langfuse_static_dependencies(monkeypatch, cfg, launcher=None)
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.resolve_langfuse_credentials",
        MagicMock(side_effect=LangfuseTrustError("langfuse_credentials_invalid")),
    )

    result = D._check_langfuse()

    assert result["status"] == "fail"
    assert result["data"]["credential_material_ready"] is False
    assert result["data"]["error_code"] == "langfuse_credentials_invalid"
    assert "placeholder" not in json.dumps(result, sort_keys=True).lower()


def test_langfuse_mcp_fails_closed_without_launcher(monkeypatch):
    cfg = _langfuse_doctor_config(langfuse_mcp_enabled=True)
    _patch_langfuse_static_dependencies(monkeypatch, cfg, launcher=None)

    result = D._check_langfuse()

    assert result["status"] == "fail"
    assert result["data"]["mcp_launcher_required"] is True
    assert result["data"]["error_code"] == "langfuse_mcp_provider_contract_unavailable"


def test_langfuse_persistence_requires_dedicated_key_ref(monkeypatch):
    cfg = _langfuse_doctor_config(langfuse_kg_auto_ingest=True)
    _patch_langfuse_static_dependencies(monkeypatch, cfg, launcher=None)

    result = D._check_langfuse()

    assert result["status"] == "fail"
    assert result["data"]["persistence_enabled"] is True
    assert result["data"]["persistence_key_ref_configured"] is False


def test_langfuse_persistence_ref_must_resolve(monkeypatch):
    cfg = _langfuse_doctor_config(
        langfuse_kg_auto_ingest=True,
        langfuse_persistence_hmac_key_ref="secret://observability/persistence-hmac",
    )
    _patch_langfuse_static_dependencies(monkeypatch, cfg, launcher=None)
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.resolve_langfuse_persistence_hmac_key",
        MagicMock(side_effect=RuntimeError("private provider detail")),
    )

    result = D._check_langfuse()

    assert result["status"] == "fail"
    assert result["data"]["persistence_key_ready"] is False
    assert "private provider detail" not in json.dumps(result, sort_keys=True)


def test_langfuse_live_check_requires_operational_proof(monkeypatch):
    cfg = _langfuse_doctor_config(langfuse_mcp_enabled=True)
    _patch_langfuse_static_dependencies(monkeypatch, cfg, launcher="available")
    monkeypatch.setattr(
        D,
        "_probe_langfuse_live",
        lambda _cfg: {
            "live_probed": True,
            "api_reachable": True,
            "mcp_visible": True,
            "trace_round_trip": False,
            "error_code": "trace_round_trip_failed",
            "redacted": True,
        },
    )

    result = D._check_langfuse(live=True)

    assert result["status"] == "fail"
    assert result["data"]["live_probed"] is True
    assert result["data"]["trace_round_trip"] is False
    assert "synthetic-public" not in json.dumps(result, sort_keys=True)
    assert "synthetic-secret" not in json.dumps(result, sort_keys=True)


def test_langfuse_live_check_accepts_api_mcp_and_trace_proof(monkeypatch):
    cfg = _langfuse_doctor_config(langfuse_mcp_enabled=True)
    _patch_langfuse_static_dependencies(monkeypatch, cfg, launcher="available")
    monkeypatch.setattr(
        D,
        "_probe_langfuse_live",
        lambda _cfg: {
            "live_probed": True,
            "api_reachable": True,
            "mcp_visible": True,
            "trace_round_trip": True,
            "redacted": True,
        },
    )

    result = D._check_langfuse(live=True)

    assert result["status"] == "ok"
    assert result["data"]["api_reachable"] is True
    assert result["data"]["mcp_visible"] is True
    assert result["data"]["trace_round_trip"] is True


def _patch_langfuse_mcp_probe(monkeypatch, responses):
    calls = []

    class Runtime:
        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            payload, is_error = responses[len(calls) - 1]
            return SimpleNamespace(
                isError=is_error,
                structuredContent=payload,
                content=[],
            )

    class Multiplexer:
        def __init__(self, _path):
            self._catalog = {}
            self.tool_to_server = {}
            self.children = {}

        async def mount_child(self, _server):
            self.tool_to_server = {
                "langfuse_observability": (
                    "langfuse-mcp",
                    "langfuse_observability",
                )
            }
            self.children = {"langfuse-mcp": Runtime()}

        async def aclose(self):
            return None

    monkeypatch.setattr(
        "agent_utilities.mcp.multiplexer.MCPMultiplexer",
        Multiplexer,
    )
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.native_langfuse_mcp_config",
        lambda **_kwargs: {"command": "synthetic"},
    )
    return calls


def test_langfuse_mcp_live_probe_invokes_posture_and_bounded_trace_list(monkeypatch):
    calls = _patch_langfuse_mcp_probe(
        monkeypatch,
        [
            (
                {
                    "content_capture_enabled": False,
                    "metadata_only": True,
                },
                False,
            ),
            ({"data": [], "meta": {"page": 1}}, False),
        ],
    )

    assert D._probe_langfuse_mcp_visibility(SimpleNamespace()) is True
    assert calls == [
        ("langfuse_observability", {"action": "runtime_posture"}),
        (
            "langfuse_observability",
            {
                "action": "trace_list",
                "page": 1,
                "limit": 1,
                "fields": "core",
            },
        ),
    ]


def test_langfuse_mcp_live_probe_attests_runtime_materialized_child(monkeypatch):
    from agent_utilities.mcp import multiplexer as multiplexer_module

    observed: dict[str, bool] = {}

    class Runtime:
        async def call_tool(self, name, arguments):
            if arguments["action"] == "runtime_posture":
                payload = {
                    "content_capture_enabled": False,
                    "metadata_only": True,
                }
            else:
                payload = {"data": []}
            return SimpleNamespace(
                isError=False,
                structuredContent=payload,
                content=[],
            )

        async def aclose(self):
            return None

    async def mount_child(self, server_name):
        child = self._catalog[server_name]
        observed["attested"] = multiplexer_module._runtime_materialized(child)
        self.tool_to_server = {
            "langfuse_observability": (
                "langfuse-mcp",
                "langfuse_observability",
            )
        }
        self.children = {"langfuse-mcp": Runtime()}

    monkeypatch.setattr(multiplexer_module.MCPMultiplexer, "mount_child", mount_child)
    monkeypatch.setattr(
        "agent_utilities.observability.langfuse_trust.native_langfuse_mcp_config",
        lambda **_kwargs: {
            "command": "synthetic",
            "env": {"LANGFUSE_SECRET_KEY": "runtime-only"},
        },
    )

    assert D._probe_langfuse_mcp_visibility(SimpleNamespace()) is True
    assert observed == {"attested": True}


@pytest.mark.parametrize(
    "responses",
    [
        [
            (
                {
                    "content_capture_enabled": True,
                    "metadata_only": False,
                },
                False,
            )
        ],
        [
            (
                {
                    "content_capture_enabled": False,
                    "metadata_only": True,
                },
                False,
            ),
            ({"data": []}, True),
        ],
        [
            (
                {
                    "content_capture_enabled": False,
                    "metadata_only": True,
                },
                False,
            ),
            ({"unexpected": []}, False),
        ],
    ],
)
def test_langfuse_mcp_live_probe_fails_closed(monkeypatch, responses):
    _patch_langfuse_mcp_probe(monkeypatch, responses)

    assert D._probe_langfuse_mcp_visibility(SimpleNamespace()) is False


def test_native_optimizer_live_check_requires_completed_job(monkeypatch):
    cfg = SimpleNamespace(kg_optimization_enabled=True)
    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: cfg)
    monkeypatch.setattr(
        D,
        "_probe_native_optimizer_live",
        lambda: {
            "live_probed": True,
            "operational": False,
            "error_code": "native_execution_failed",
            "privacy_safe_payload": True,
        },
    )

    result = D._check_native_optimizer(live=True)

    assert result["status"] == "fail"
    assert result["data"]["live_probed"] is True
    assert result["data"]["operational"] is False


def test_native_optimizer_live_probe_sends_only_opaque_program_payload(monkeypatch):
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    captured = {}

    class Engine:
        def optimize_program(self, request):
            captured.update(request)
            return {"status": "proposed", "result": {}}

    monkeypatch.setattr(
        GraphComputeEngine, "get_active", classmethod(lambda cls: Engine())
    )

    result = D._probe_native_optimizer_live()
    rendered = json.dumps(captured, sort_keys=True)

    assert result == {
        "live_probed": True,
        "operational": True,
        "privacy_safe_payload": True,
    }
    assert "synthetic-capability-probe" not in rendered
    assert "synthetic-capability-result" not in rendered
    assert captured["corpus"]["privacy"]["raw_pii_persisted"] is False
    assert captured["corpus"]["privacy"]["local_identifiers_persisted"] is False


# ── live path: graph_configure system_doctor MCP action ────────────────────
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
async def test_graph_configure_system_doctor_live_path(registered_tools, monkeypatch):
    from agent_utilities.mcp import kg_server

    monkeypatch.setattr(D, "CHECKS", {"a": _ok("a")})
    raw = await kg_server._execute_tool(
        "graph_configure",
        action="system_doctor",
        config_value=json.dumps({"only": ["a"]}),
    )
    rep = json.loads(raw)
    assert rep["status"] == "healthy"
    assert rep["checks"][0]["name"] == "a"
