"""Config migration: retired-key stripping + doctor auto-migration (WS-A)."""

from __future__ import annotations

import json
from pathlib import Path

from agent_utilities.core.config import (
    plaintext_secret_keys,
    retired_configuration_keys,
    strip_retired_configuration_keys,
)
from agent_utilities.deployment.config_generator import (
    config_doctor,
    migrate_config_file,
    unknown_configuration_keys,
)


def test_strip_retired_removes_only_retired() -> None:
    mapping = {
        "ENGINE_MODE": "external",  # retired
        "OIDC_CLIENT_SECRET": "x",  # retired (durable secret)
        "WORKSPACE_PATH": "keep-me",  # a real field alias
        "CAMUNDA_URL": "http://c",  # unknown, not retired -> kept here
    }
    cleaned, removed = strip_retired_configuration_keys(mapping)
    assert removed == ["ENGINE_MODE", "OIDC_CLIENT_SECRET"]
    assert "ENGINE_MODE" not in cleaned and "OIDC_CLIENT_SECRET" not in cleaned
    assert cleaned["WORKSPACE_PATH"] == "keep-me"
    assert cleaned["CAMUNDA_URL"] == "http://c"
    assert "ENGINE_MODE" in retired_configuration_keys()


def test_migrate_config_file_strips_retired_and_backs_up(tmp_path: Path) -> None:
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"ENGINE_MODE": "external", "WORKSPACE_PATH": "a"}))
    report = migrate_config_file(p)
    assert report["status"] == "migrated"
    assert report["removed"] == ["ENGINE_MODE"]
    assert Path(report["backup"]).exists()
    on_disk = json.loads(p.read_text())
    assert "ENGINE_MODE" not in on_disk and on_disk["WORKSPACE_PATH"] == "a"


def test_migrate_config_file_reports_but_keeps_unknown_by_default(
    tmp_path: Path,
) -> None:
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"ENGINE_MODE": "external", "CAMUNDA_URL": "http://c"}))
    report = migrate_config_file(p)  # strip_unknown defaults False
    assert "CAMUNDA_URL" in report["unknown_present"]
    assert "CAMUNDA_URL" in json.loads(p.read_text())  # connector key preserved


def test_migrate_config_file_aggressive_strips_unknown(tmp_path: Path) -> None:
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"WORKSPACE_PATH": "a", "CAMUNDA_URL": "http://c"}))
    report = migrate_config_file(p, strip_unknown=True)
    assert report["status"] == "migrated"
    assert "CAMUNDA_URL" in report["unknown_removed"]
    assert "CAMUNDA_URL" not in json.loads(p.read_text())


def test_config_doctor_flags_retired_and_migrate_fixes(tmp_path: Path) -> None:
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"ENGINE_MODE": "external", "WORKSPACE_PATH": "a"}))
    flagged = config_doctor(profile="tiny", config_path=p)
    assert flagged["status"] == "needs_migration"
    assert flagged["healthy"] is False
    # migrate=True removes them before validation
    fixed = config_doctor(profile="tiny", config_path=p, migrate=True)
    assert fixed["status"] != "needs_migration"
    assert "ENGINE_MODE" not in json.loads(p.read_text())


def test_unknown_configuration_keys_excludes_fields_and_retired() -> None:
    unknown = unknown_configuration_keys(
        {"WORKSPACE_PATH": "a", "ENGINE_MODE": "x", "CAMUNDA_URL": "http://c"}
    )
    assert unknown == ["CAMUNDA_URL"]


def test_canonicalize_passes_through_connector_keys() -> None:
    """The strict XDG canonicalizer preserves dynamic config.setting() keys
    (connector/service config) instead of rejecting them — aligning with the
    documented 'config.json drives setting()' mechanism. Retired keys are still
    rejected downstream; ambiguous keys still raise."""
    import pytest

    from agent_utilities.core.config import _canonicalize_xdg_configuration

    out = _canonicalize_xdg_configuration(
        {"CAMUNDA_URL": "http://c", "WORKSPACE_PATH": "/x"}
    )
    assert out["CAMUNDA_URL"] == "http://c"  # dynamic key preserved
    assert "WORKSPACE_PATH" in out  # known field canonicalized

    with pytest.raises(Exception):  # genuine ambiguity still rejected
        _canonicalize_xdg_configuration({"camunda_url": "a", "CAMUNDA_URL": "b"})


def test_plaintext_secret_keys_flags_inline_credentials() -> None:
    offenders = plaintext_secret_keys(
        {
            "GITLAB_TOKEN": "ghp_plaintext",  # credential suffix, not a *_REF
            "LANGFUSE_SECRET_KEY": "sk-live",  # credential suffix
            "PASSWORD": "hunter2",  # sensitive mapping key
            "GITLAB_TOKEN_REF": "vault://apps/gitlab#token",  # a *_REF -> allowed
            "WORKSPACE_PATH": "/x",  # ordinary field
            "TIMEOUT_SECONDS": 30,  # ordinary field
        }
    )
    assert offenders == ["GITLAB_TOKEN", "LANGFUSE_SECRET_KEY", "PASSWORD"]


def test_plaintext_secret_keys_empty_when_all_refs() -> None:
    assert (
        plaintext_secret_keys(
            {
                "OIDC_CLIENT_SECRET_REF": "vault://apps/x#s",
                "WORKSPACE_PATH": "/x",
                "GITLAB_TOKEN": "",  # empty placeholder is not a live secret
            }
        )
        == []
    )


def test_config_doctor_flags_plaintext_secret(tmp_path: Path) -> None:
    """The doctor must NAME an inline plaintext secret (the load path would else
    reject the whole config with an opaque DurableSecretError)."""
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"GITLAB_TOKEN": "ghp_plaintext", "WORKSPACE_PATH": "a"}))
    report = config_doctor(profile="tiny", config_path=p)
    assert report["status"] == "needs_migration"
    assert report["healthy"] is False
    check = next(c for c in report["checks"] if c["check"] == "durable_secret_policy")
    assert check["keys"] == ["GITLAB_TOKEN"]
    assert "_REF" in check["remediation"]


def test_migrate_config_file_reports_plaintext_secrets(tmp_path: Path) -> None:
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"TWENTY_TOKEN": "tok", "WORKSPACE_PATH": "a"}))
    report = migrate_config_file(p)
    assert "TWENTY_TOKEN" in report["plaintext_secrets"]
    # reported, never stripped or moved — the value stays put for a human-gated move
    assert "TWENTY_TOKEN" in json.loads(p.read_text())
