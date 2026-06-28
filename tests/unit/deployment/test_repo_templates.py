"""Tests for the standard private-repo set + generalized CI templates.

CONCEPT:OS-5.74 / CONCEPT:OS-5.75. Covers the TEMPLATING + the provisioning PLAN +
profile scaling — no network / no live repo create. The load-bearing invariant is
that NOTHING here carries operator-specific environment data (the whole point).
"""

from __future__ import annotations

import pytest

from agent_utilities.deployment import (
    CI_TEMPLATES,
    PROFILE_REPO_SETS,
    STANDARD_REPOS,
    provision_plan,
    render_skeleton,
    runner_plan,
    standard_repos,
)
from agent_utilities.deployment.repo_templates import (
    PLACEHOLDER_TOKENS,
    PROFILES,
    ci_enabled,
    manifest_summary,
    referenced_tokens,
    render,
)


# ── profile scaling ─────────────────────────────────────────────────────────
def test_profile_repo_sets_scale_monotonically():
    tiny = set(PROFILE_REPO_SETS["tiny"])
    snp = set(PROFILE_REPO_SETS["single-node-prod"])
    ent = set(PROFILE_REPO_SETS["enterprise"])
    # Each larger profile is a superset of the smaller one (degrades cleanly).
    assert tiny < snp < ent
    # Tiny is minimal & local; enterprise alone gets the shared CI templates repo.
    assert tiny == {"inventory", "config"}
    assert "pipelines" in ent and "pipelines" not in snp


@pytest.mark.parametrize("profile", PROFILES)
def test_standard_repos_resolve_for_every_profile(profile):
    repos = standard_repos(profile)
    assert repos and all(r.skeleton for r in repos)
    assert [r.key for r in repos] == list(PROFILE_REPO_SETS[profile])


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        standard_repos("galactic")
    with pytest.raises(ValueError):
        provision_plan("galactic")


# ── the plan ────────────────────────────────────────────────────────────────
def test_tiny_plan_is_local_no_ci_no_runners():
    plan = provision_plan("tiny")
    assert plan["git_mode"] == "local"
    assert plan["git_host"] == "local"  # tiny forces local even if gitlab passed
    assert plan["ci"]["enabled"] is False
    assert plan["runners"]["register"] is False
    assert all(r["ci"] is False for r in plan["repos"])


def test_enterprise_plan_full_with_ci_and_runners():
    plan = provision_plan("enterprise", git_host="gitlab", namespace="acme")
    keys = [r["key"] for r in plan["repos"]]
    assert keys == list(PROFILE_REPO_SETS["enterprise"])
    assert plan["ci"]["enabled"] is True
    assert plan["ci"]["templates_repo"] == "acme/pipelines"
    assert plan["runners"]["count"] >= 1 and plan["runners"]["scope"] == "group"
    # All repos private; names namespaced.
    assert all(r["visibility"] == "private" for r in plan["repos"])
    assert all(r["name"].startswith("acme/") for r in plan["repos"])


def test_plan_is_idempotent_skips_existing():
    plan = provision_plan(
        "single-node-prod",
        namespace="acme",
        existing_repos=["inventory", "acme/config"],
    )
    actions = {r["key"]: r["action"] for r in plan["repos"]}
    assert actions["inventory"] == "skip"
    assert actions["config"] == "skip"
    assert actions["networks"] == "create"
    assert plan["idempotent"] is True


def test_runner_plan_scales():
    assert runner_plan("tiny")["count"] == 0
    assert runner_plan("single-node-prod")["count"] >= 1
    assert (
        runner_plan("enterprise")["count"] >= runner_plan("single-node-prod")["count"]
    )


def test_ci_enabled_matches_profiles():
    assert ci_enabled("tiny") is False
    assert ci_enabled("single-node-prod") is True
    assert ci_enabled("enterprise") is True


# ── templating ──────────────────────────────────────────────────────────────
def test_render_substitutes_known_tokens_only():
    out = render(
        "repo=${GIT_NAMESPACE}/x tag=${RUNNER_TAG} keep=${UNKNOWN}",
        {
            "GIT_NAMESPACE": "acme",
            "RUNNER_TAG": "shell-1",
        },
    )
    assert "acme/x" in out and "shell-1" in out
    # Unknown tokens are preserved, never corrupted.
    assert "${UNKNOWN}" in out


def test_render_skeleton_resolves_ci_templates():
    pipelines = next(r for r in STANDARD_REPOS if r.key == "pipelines")
    rendered = render_skeleton(
        pipelines,
        {
            "CI_TEMPLATES_PROJECT": "acme/pipelines",
            "RUNNER_TAG": "docker-1",
            "REGISTRY": "registry.example",
        },
    )
    deploy = rendered["service-deploy.yml"]
    assert "acme/pipelines" in deploy and "docker-1" in deploy
    # No leftover placeholders for the tokens we supplied.
    assert "${CI_TEMPLATES_PROJECT}" not in deploy
    assert "${RUNNER_TAG}" not in deploy


# ── the load-bearing invariant: NO operator-specific env in the repo ────────
_FORBIDDEN = (
    # Real operator hostnames / runner tags / project paths that must NEVER appear.
    "Shell-R820",
    "homelab/pipelines/gitlab-pipelines",
    "R820",
    "RW710",
    "10.0.0.",
    "100.64.",
    "100.65.",
    ".arpa",
)


def test_no_operator_specifics_in_any_skeleton_or_ci_template():
    blobs = []
    for repo in STANDARD_REPOS:
        blobs.extend(repo.skeleton.values())
    blobs.extend(CI_TEMPLATES.values())
    for blob in blobs:
        for needle in _FORBIDDEN:
            assert needle not in blob, (
                f"operator-specific {needle!r} leaked into a template"
            )


def test_ci_templates_use_tokens_not_hardcoded_runner_tags():
    # The generalized templates must reference ${RUNNER_TAG}, not a literal tag.
    deploy = CI_TEMPLATES["service-deploy.yml"]
    assert "${RUNNER_TAG}" in deploy
    assert "RUNNER_TAG" in referenced_tokens(deploy)


# GitLab-provided / template-internal CI variables (NOT operator placeholders).
_KNOWN_CI_VARS = {
    "CI_COMMIT_SHORT_SHA",
    "PACKAGE_NAME",
    "STACK_NAME",
    "K8S_NAMESPACE",
    "HELM_CHART",
}


def test_all_referenced_tokens_are_declared_or_ci_vars():
    # Every ${...} in a skeleton is either a declared operator placeholder or a
    # GitLab/template CI variable — never an undeclared operator-specific value.
    allowed = set(PLACEHOLDER_TOKENS) | _KNOWN_CI_VARS
    for repo in STANDARD_REPOS:
        for content in repo.skeleton.values():
            assert referenced_tokens(content) <= allowed, repo.key


def test_ci_templates_reference_operator_placeholders():
    # The shared CI templates carry the deploy-time tokens (runner tag + registry +
    # templates project) — proving operator specifics are tokens, not literals.
    used: set[str] = set()
    for content in CI_TEMPLATES.values():
        used |= referenced_tokens(content)
    assert {"RUNNER_TAG", "CI_TEMPLATES_PROJECT", "REGISTRY"} <= used


def test_secrets_repo_is_references_only():
    secrets = next(r for r in STANDARD_REPOS if r.key == "secrets-config")
    text = "\n".join(secrets.skeleton.values()).lower()
    assert "never commit plaintext" in text
    assert "vault://" in text or "engine://__secrets__" in text


def test_manifest_summary_is_generator_friendly():
    summary = manifest_summary()
    assert summary["concept"] == "OS-5.74"
    assert set(summary["per_profile"]) == set(PROFILES)
    assert summary["ci_templates"] == sorted(CI_TEMPLATES)
