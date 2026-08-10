"""Tests for named, explicitly-reviewable genesis k8s deployment-input profiles.

CONCEPT:AU-OS.deployment.genesis-environment-profiles

Covers: the three shipped defaults (dev/test/prod) load and validate; the
operator extension mechanism (a new profile name with zero code change);
fail-loud behavior for an unknown profile, a missing/empty secret reference, a
non-scheme secret ref, a secret-shaped key leaking into ``configuration.env``,
an unjustified writable path, the prod digest-pin gate, the universal
mcp-tools-list functional-check requirement, and the identity/secrets
cross-check. See ``.specify/design/genesis-environment-profiles/design.md``.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from agent_utilities.deployment.genesis_environments import (
    BUILTIN_ENVIRONMENTS_DIR,
    EnvironmentProfileError,
    MissingSecretReferenceError,
    _profile_from_mapping,
    list_environment_profiles,
    load_environment_profile,
    profile_summary,
    validate_environment_profile,
)


def _raw(name: str) -> dict:
    path = BUILTIN_ENVIRONMENTS_DIR / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ── defaults ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["dev", "test", "prod"])
def test_default_profile_loads_and_validates(name):
    profile = load_environment_profile(name)
    assert profile.environment.name == name
    assert profile.target.orchestrator == "kubernetes"
    # Every secret is a reference, never a literal-looking value.
    for secret in profile.secrets.required:
        assert "://" in secret.ref
    # The universal functional-check requirement.
    assert any(
        fc.kind == "mcp-tools-list" for fc in profile.validation.functional_checks
    )


def test_list_environment_profiles_finds_all_three():
    catalog = list_environment_profiles()
    assert {"dev", "test", "prod"} <= set(catalog)
    assert all(path.suffix == ".yaml" for path in catalog.values())


def test_prod_is_digest_pinned():
    profile = load_environment_profile("prod")
    assert profile.release.tag_policy == "digest-pinned"
    assert profile.release.revision


def test_profile_summary_has_no_secret_values_only_refs():
    profile = load_environment_profile("prod")
    summary = profile_summary(profile)
    for secret in summary["secrets"]["required"]:
        assert secret["ref"].split("://", 1)[0] in {
            "env",
            "vault",
            "secret",
            "k8s-secret",
        }


# ── extension mechanism (no code change for a new named profile) ─────────
def test_extension_directory_adds_a_new_profile_with_zero_code_change(
    tmp_path, monkeypatch
):
    ext_dir = tmp_path / "environments"
    ext_dir.mkdir()
    uat = copy.deepcopy(_raw("dev"))
    uat["environment"]["name"] = "uat"
    (ext_dir / "uat.yaml").write_text(yaml.safe_dump(uat), encoding="utf-8")

    monkeypatch.setattr(
        "agent_utilities.deployment.genesis_environments._extension_dir",
        lambda: ext_dir,
    )

    catalog = list_environment_profiles()
    assert "uat" in catalog
    profile = load_environment_profile("uat")
    assert profile.environment.name == "uat"


def test_extension_directory_overrides_a_builtin_by_name(tmp_path, monkeypatch):
    ext_dir = tmp_path / "environments"
    ext_dir.mkdir()
    custom_dev = copy.deepcopy(_raw("dev"))
    custom_dev["target"]["namespace"] = "operator-custom-namespace"
    (ext_dir / "dev.yaml").write_text(yaml.safe_dump(custom_dev), encoding="utf-8")

    monkeypatch.setattr(
        "agent_utilities.deployment.genesis_environments._extension_dir",
        lambda: ext_dir,
    )

    profile = load_environment_profile("dev")
    assert profile.target.namespace == "operator-custom-namespace"


# ── fail loud: unknown profile ────────────────────────────────────────────
def test_unknown_profile_names_what_was_found():
    with pytest.raises(EnvironmentProfileError) as exc:
        load_environment_profile("does-not-exist")
    message = str(exc.value)
    assert "does-not-exist" in message
    assert "dev" in message and "test" in message and "prod" in message


# ── fail loud: we do not infer credentials ────────────────────────────────
def test_missing_secret_ref_raises_missing_secret_error():
    raw = _raw("dev")
    raw["secrets"]["required"][0]["ref"] = ""
    with pytest.raises(MissingSecretReferenceError) as exc:
        _profile_from_mapping(raw, source=BUILTIN_ENVIRONMENTS_DIR / "dev.yaml")
    assert "graph-os-secrets" in str(exc.value)
    assert "do not infer credentials" in str(exc.value)


def test_secret_ref_must_use_a_recognized_scheme():
    raw = _raw("dev")
    raw["secrets"]["required"][0]["ref"] = "a-literal-looking-value"
    profile = _profile_from_mapping(raw, source=BUILTIN_ENVIRONMENTS_DIR / "dev.yaml")
    with pytest.raises(
        EnvironmentProfileError, match="not a recognized reference scheme"
    ):
        validate_environment_profile(profile)


@pytest.mark.parametrize(
    "ref",
    [
        "env://GRAPH_OS_SECRET",
        "vault://apps/graph-os/oidc-client-secret",
        "secret://graph-os/oidc",
        "k8s-secret://platform/graph-os-secrets",
    ],
)
def test_every_recognized_scheme_is_accepted(ref):
    raw = _raw("dev")
    raw["secrets"]["required"][0]["ref"] = ref
    profile = _profile_from_mapping(raw, source=BUILTIN_ENVIRONMENTS_DIR / "dev.yaml")
    validate_environment_profile(profile)  # must not raise


# ── fail loud: configuration must stay non-secret ─────────────────────────
def test_secret_shaped_key_in_configuration_env_is_rejected():
    raw = _raw("dev")
    raw["configuration"]["env"]["DB_PASSWORD"] = "should-not-be-here"
    profile = _profile_from_mapping(raw, source=BUILTIN_ENVIRONMENTS_DIR / "dev.yaml")
    with pytest.raises(EnvironmentProfileError, match="secret-shaped key"):
        validate_environment_profile(profile)


# ── fail loud: identity must point at a declared secret ───────────────────
def test_identity_client_secret_ref_must_name_a_declared_secret():
    raw = _raw("prod")
    raw["identity"]["client_secret_ref"] = "no-such-secret"
    profile = _profile_from_mapping(raw, source=BUILTIN_ENVIRONMENTS_DIR / "prod.yaml")
    with pytest.raises(MissingSecretReferenceError, match="no-such-secret"):
        validate_environment_profile(profile)


# ── fail loud: filesystem writable paths must be justified ────────────────
def test_writable_path_without_reason_is_rejected():
    raw = _raw("dev")
    raw["filesystem"]["writable_paths"][0]["reason"] = "   "
    profile = _profile_from_mapping(raw, source=BUILTIN_ENVIRONMENTS_DIR / "dev.yaml")
    with pytest.raises(EnvironmentProfileError, match="no reason"):
        validate_environment_profile(profile)


def test_writable_path_with_unknown_medium_is_rejected():
    raw = _raw("dev")
    raw["filesystem"]["writable_paths"][0]["medium"] = "nfs"
    profile = _profile_from_mapping(raw, source=BUILTIN_ENVIRONMENTS_DIR / "dev.yaml")
    with pytest.raises(EnvironmentProfileError, match="unrecognized medium"):
        validate_environment_profile(profile)


# ── fail loud: prod tier release discipline ────────────────────────────────
def test_prod_tier_rejects_floating_tag():
    raw = _raw("prod")
    raw["release"]["tag_policy"] = "floating-tag"
    profile = _profile_from_mapping(raw, source=BUILTIN_ENVIRONMENTS_DIR / "prod.yaml")
    with pytest.raises(EnvironmentProfileError, match="digest-pinned"):
        validate_environment_profile(profile)


def test_prod_tier_requires_a_revision_even_when_digest_pinned():
    raw = _raw("prod")
    raw["release"]["revision"] = None
    profile = _profile_from_mapping(raw, source=BUILTIN_ENVIRONMENTS_DIR / "prod.yaml")
    with pytest.raises(EnvironmentProfileError, match="revision is empty"):
        validate_environment_profile(profile)


# ── fail loud: the standing MCP-liveness rule applies to every tier ───────
def test_every_tier_requires_an_mcp_tools_list_functional_check():
    raw = _raw("dev")
    raw["validation"]["functional_checks"] = [
        {"kind": "http-endpoint", "target": "/health", "expected": "200 OK"}
    ]
    profile = _profile_from_mapping(raw, source=BUILTIN_ENVIRONMENTS_DIR / "dev.yaml")
    with pytest.raises(EnvironmentProfileError, match="mcp-tools-list"):
        validate_environment_profile(profile)


# ── fail loud: closed schema — every field required, no unknown fields ────
def test_missing_required_field_is_rejected():
    raw = _raw("dev")
    del raw["target"]["namespace"]
    with pytest.raises(EnvironmentProfileError, match="missing required key"):
        _profile_from_mapping(raw, source=BUILTIN_ENVIRONMENTS_DIR / "dev.yaml")


def test_unrecognized_field_is_rejected():
    raw = _raw("dev")
    raw["target"]["unexpected_field"] = "surprise"
    with pytest.raises(EnvironmentProfileError, match="unrecognized key"):
        _profile_from_mapping(raw, source=BUILTIN_ENVIRONMENTS_DIR / "dev.yaml")


def test_missing_top_level_section_is_rejected():
    raw = _raw("dev")
    del raw["secrets"]
    with pytest.raises(EnvironmentProfileError, match="missing required key"):
        _profile_from_mapping(raw, source=BUILTIN_ENVIRONMENTS_DIR / "dev.yaml")


# ── fail loud: target/identity cross-checked against genesis.yaml ─────────
def test_target_orchestrator_must_be_a_genesis_yaml_run_plan_value():
    raw = _raw("dev")
    raw["target"]["orchestrator"] = "not-a-real-orchestrator"
    profile = _profile_from_mapping(raw, source=BUILTIN_ENVIRONMENTS_DIR / "dev.yaml")
    with pytest.raises(EnvironmentProfileError, match="run_plan.orchestrators"):
        validate_environment_profile(profile)


def test_identity_idp_must_be_a_genesis_yaml_run_plan_value():
    raw = _raw("dev")
    raw["identity"]["idp"] = "not-a-real-idp"
    profile = _profile_from_mapping(raw, source=BUILTIN_ENVIRONMENTS_DIR / "dev.yaml")
    with pytest.raises(EnvironmentProfileError, match="run_plan.idp"):
        validate_environment_profile(profile)
