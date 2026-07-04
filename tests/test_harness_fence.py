#!/usr/bin/python
"""Tests for the governance-derived Claude Code permission fence.

CONCEPT:AU-OS.deployment.governance-derived-claude-code — Governance-derived Claude Code permission-fence generator
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_utilities.claude_harness import claude_fence as cf
from agent_utilities.orchestration.action_policy import ActionPolicy


def _policy_with_rule(tmp_path: Path, rules: list[dict]) -> ActionPolicy:
    """An ActionPolicy backed by a temp YAML carrying ``rules``."""
    import yaml

    path = tmp_path / "policy.yml"
    path.write_text(yaml.safe_dump({"version": 1, "rules": rules}), encoding="utf-8")
    return ActionPolicy(policy_path=str(path))


def test_forbidden_tier_maps_to_deny(tmp_path: Path):
    policy = _policy_with_rule(
        tmp_path, [{"kind": "deploy_service", "target": "*", "tier": "forbidden"}]
    )
    rules = cf.derive_deny_allow_ask(policy)
    # deploy_service maps to docker patterns; forbidden -> they must be denied.
    assert "Bash(docker:*)" in rules.deny
    assert "Bash(docker:*)" not in rules.ask
    assert "Bash(docker:*)" not in rules.allow


def test_approval_tier_maps_to_ask(tmp_path: Path):
    policy = _policy_with_rule(
        tmp_path,
        [{"kind": "restart_service", "target": "*", "tier": "approval_required"}],
    )
    rules = cf.derive_deny_allow_ask(policy)
    assert "Bash(systemctl:*)" in rules.ask


def test_deny_outranks_allow_dedup():
    # A pattern that appears in both buckets resolves to deny only.
    rules = cf.FenceRules(
        deny=["Bash(rm -rf:*)"], ask=["Bash(rm -rf:*)"], allow=["Bash(rm -rf:*)"]
    )
    assert rules.deny == ["Bash(rm -rf:*)"]
    assert "Bash(rm -rf:*)" not in rules.ask
    assert "Bash(rm -rf:*)" not in rules.allow


def test_default_mode_is_accept_edits_never_bypass():
    settings = cf.build_settings_dict()
    assert settings["permissions"]["defaultMode"] == "acceptEdits"
    # bypassPermissions must never appear anywhere in the generated artifact.
    assert "bypassPermissions" not in json.dumps(settings)


def test_irreversible_baseline_denied():
    deny = cf.build_settings_dict()["permissions"]["deny"]
    for pat in (
        "Bash(rm -rf:*)",
        "Bash(git push --force:*)",
        "Bash(git reset --hard:*)",
        "Bash(curl:*)",
    ):
        assert pat in deny


def test_secret_globs_in_claudeignore_and_deny():
    ignore = cf.build_claudeignore_text()
    assert ".env" in ignore
    assert "*.pem" in ignore
    deny = cf.build_settings_dict()["permissions"]["deny"]
    assert "Read(.env)" in deny
    assert "Edit(.env)" in deny


def test_is_secret_path_uses_suffixes():
    assert cf.is_secret_path(".env")
    assert cf.is_secret_path("config/prod.pem")
    assert cf.is_secret_path("db.token")  # _SECRET_SUFFIXES reuse
    assert not cf.is_secret_path("src/main.py")


def test_write_fence_idempotent_and_preserves_operator_allow(tmp_path: Path):
    target = tmp_path / ".claude"
    # First write.
    r1 = cf.write_fence(target)
    assert r1["changed"] is True
    settings = json.loads((target / "settings.json").read_text())
    # Operator adds a custom allow entry by hand.
    settings["permissions"]["allow"].append("Bash(my-custom-tool:*)")
    (target / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    # Second write merges, keeps the custom entry, and (policy unchanged) the
    # third run is a clean no-op.
    cf.write_fence(target)
    merged = json.loads((target / "settings.json").read_text())
    assert "Bash(my-custom-tool:*)" in merged["permissions"]["allow"]
    r3 = cf.write_fence(target)
    assert r3["changed"] is False


def test_claudeignore_written_to_parent(tmp_path: Path):
    target = tmp_path / ".claude"
    cf.write_fence(target)
    assert (tmp_path / ".claudeignore").exists()


def test_dry_run_does_not_write(tmp_path: Path):
    target = tmp_path / ".claude"
    res = cf.write_fence(target, dry_run=True)
    assert res["dry_run"] is True
    assert not (target / "settings.json").exists()
    assert "settings" in res
