"""Config migration: retired-key stripping + doctor auto-migration (WS-A)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_utilities.core.config import (
    AgentConfig,
    ConfigurationSourceError,
    _staged_xdg_document,
    plaintext_secret_keys,
    retired_configuration_keys,
    strip_retired_configuration_keys,
)
from agent_utilities.deployment.config_generator import (
    config_doctor,
    migrate_config_file,
    unknown_configuration_keys,
)

_RETIRED_ENGINE_KEY = "_".join(("ENGINE", "MODE"))
_RETIRED_MESSAGING_TRIGGER = "MESSAGING_VENDOR_TRIGGER"
_RETIRED_MESSAGING_MODEL = "MESSAGING_VENDOR_MODEL"


@pytest.fixture(autouse=True)
def governed_retired_configuration_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    config_dir = tmp_path / "operator-config"
    policy = config_dir / "governance" / "retired-configuration-keys.json"
    policy.parent.mkdir(parents=True)
    policy.write_text(
        json.dumps(
            {
                "version": "test-v1",
                "renames": {
                    _RETIRED_MESSAGING_TRIGGER: "MESSAGING_MODEL_TRIGGER",
                    _RETIRED_MESSAGING_MODEL: "MESSAGING_ADDRESSED_MODEL",
                    "MESSAGING_LOCAL_MODEL": "MESSAGING_DEFAULT_MODEL",
                },
            }
        )
    )
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(config_dir))
    return policy


def test_strip_retired_removes_only_retired() -> None:
    mapping = {
        _RETIRED_ENGINE_KEY: "external",  # retired
        "OIDC_CLIENT_SECRET": "x",  # retired (durable secret)
        "WORKSPACE_PATH": "keep-me",  # a real field alias
        "CAMUNDA_URL": "http://c",  # unknown, not retired -> kept here
    }
    cleaned, removed = strip_retired_configuration_keys(mapping)
    assert removed == [_RETIRED_ENGINE_KEY, "OIDC_CLIENT_SECRET"]
    assert _RETIRED_ENGINE_KEY not in cleaned and "OIDC_CLIENT_SECRET" not in cleaned
    assert cleaned["WORKSPACE_PATH"] == "keep-me"
    assert cleaned["CAMUNDA_URL"] == "http://c"
    assert _RETIRED_ENGINE_KEY in retired_configuration_keys()


def test_migrate_config_file_strips_retired_and_backs_up(tmp_path: Path) -> None:
    p = tmp_path / "config.json"
    p.write_text(json.dumps({_RETIRED_ENGINE_KEY: "external", "WORKSPACE_PATH": "a"}))
    report = migrate_config_file(p)
    assert report["status"] == "migrated"
    assert report["removed"] == [_RETIRED_ENGINE_KEY]
    assert Path(report["backup"]).exists()
    on_disk = json.loads(p.read_text())
    assert _RETIRED_ENGINE_KEY not in on_disk and on_disk["WORKSPACE_PATH"] == "a"


def test_migrate_config_file_atomically_renames_messaging_selectors(
    tmp_path: Path,
) -> None:
    p = tmp_path / "config.json"
    p.write_text(
        json.dumps(
            {
                _RETIRED_MESSAGING_TRIGGER: "/model",
                _RETIRED_MESSAGING_MODEL: "addressed-model",
                "MESSAGING_LOCAL_MODEL": "default-model",
            }
        )
    )

    report = migrate_config_file(p, backup=False)

    assert report["status"] == "migrated"
    assert len(report["renamed"]) == 3
    assert json.loads(p.read_text()) == {
        "MESSAGING_MODEL_TRIGGER": "/model",
        "MESSAGING_ADDRESSED_MODEL": "addressed-model",
        "MESSAGING_DEFAULT_MODEL": "default-model",
    }
    assert not list(tmp_path.glob(".config-*.tmp"))


def test_migrate_config_file_preserves_non_selector_model_keys(tmp_path: Path) -> None:
    p = tmp_path / "config.json"
    original = json.dumps({"MESSAGING_VOICE_MODEL": "speech-model"})
    p.write_text(original)

    report = migrate_config_file(p, backup=False)

    assert report["status"] == "ok"
    assert p.read_text() == original


def test_migrate_config_file_does_not_guess_unlisted_provider_keys(
    tmp_path: Path,
) -> None:
    p = tmp_path / "config.json"
    original = json.dumps({"MESSAGING_UNLISTED_MODEL": "model-v1"})
    p.write_text(original)

    report = migrate_config_file(p, backup=False)

    assert report["status"] == "ok"
    assert p.read_text() == original


def test_migrate_config_file_rejects_messaging_key_conflict_without_writing(
    tmp_path: Path,
) -> None:
    p = tmp_path / "config.json"
    original = json.dumps(
        {
            _RETIRED_MESSAGING_MODEL: "old-selector",
            "MESSAGING_ADDRESSED_MODEL": "new-selector",
        }
    )
    p.write_text(original)

    report = migrate_config_file(p, backup=False)

    assert report == {
        "status": "error",
        "error": "messaging_model_migration_conflict",
        "path": str(p),
    }
    assert p.read_text() == original


def test_xdg_load_migrates_persisted_messaging_selectors(tmp_path: Path) -> None:
    p = tmp_path / "config.json"
    p.write_text(json.dumps({_RETIRED_MESSAGING_MODEL: "addressed-model"}))
    p.chmod(0o600)

    staged = _staged_xdg_document(p, strict=True)

    assert staged["MESSAGING_ADDRESSED_MODEL"] == "addressed-model"
    assert json.loads(p.read_text()) == {"MESSAGING_ADDRESSED_MODEL": "addressed-model"}


def test_xdg_load_rejects_invalid_renamed_document_without_writing(
    tmp_path: Path,
) -> None:
    p = tmp_path / "config.json"
    original = json.dumps(
        {
            _RETIRED_MESSAGING_MODEL: "addressed-model",
            "MESSAGING_INTAKE_ENABLED": "not-a-boolean",
        }
    )
    p.write_text(original)
    p.chmod(0o600)

    with pytest.raises(ConfigurationSourceError):
        _staged_xdg_document(p, strict=True)

    assert p.read_text() == original


def test_xdg_load_rejects_invalid_governance_catalog_without_writing(
    tmp_path: Path, governed_retired_configuration_catalog: Path
) -> None:
    governed_retired_configuration_catalog.write_text(
        json.dumps(
            {
                "version": "test-v2",
                "renames": {
                    _RETIRED_MESSAGING_MODEL: "MESSAGING_VOICE_MODEL",
                },
            }
        )
    )
    p = tmp_path / "config.json"
    original = json.dumps({_RETIRED_MESSAGING_MODEL: "addressed-model"})
    p.write_text(original)
    p.chmod(0o600)

    with pytest.raises(ConfigurationSourceError) as raised:
        _staged_xdg_document(p, strict=True)

    assert raised.value.source_type == "governance"
    assert raised.value.error_class == "RetiredConfigurationCatalogError"
    assert p.read_text() == original


def test_migrate_config_file_rejects_invalid_renamed_document_without_writing(
    tmp_path: Path,
) -> None:
    p = tmp_path / "config.json"
    original = json.dumps(
        {
            _RETIRED_MESSAGING_MODEL: "addressed-model",
            "MESSAGING_INTAKE_ENABLED": "not-a-boolean",
        }
    )
    p.write_text(original)

    report = migrate_config_file(p, backup=False)

    assert report["status"] == "error"
    assert report["error"] == "messaging_model_migration_invalid"
    assert p.read_text() == original


def test_legacy_messaging_environment_input_fails_with_neutral_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_RETIRED_MESSAGING_MODEL, "addressed-model")

    with pytest.raises(ValueError, match="MESSAGING_ADDRESSED_MODEL"):
        AgentConfig()


def test_migrate_config_file_reports_but_keeps_unknown_by_default(
    tmp_path: Path,
) -> None:
    p = tmp_path / "config.json"
    p.write_text(
        json.dumps({_RETIRED_ENGINE_KEY: "external", "CAMUNDA_URL": "http://c"})
    )
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
    p.write_text(json.dumps({_RETIRED_ENGINE_KEY: "external", "WORKSPACE_PATH": "a"}))
    flagged = config_doctor(profile="tiny", config_path=p)
    assert flagged["status"] == "needs_migration"
    assert flagged["healthy"] is False
    # migrate=True removes them before validation
    fixed = config_doctor(profile="tiny", config_path=p, migrate=True)
    assert fixed["status"] != "needs_migration"
    assert _RETIRED_ENGINE_KEY not in json.loads(p.read_text())


def test_unknown_configuration_keys_excludes_fields_and_retired() -> None:
    unknown = unknown_configuration_keys(
        {"WORKSPACE_PATH": "a", _RETIRED_ENGINE_KEY: "x", "CAMUNDA_URL": "http://c"}
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

    # Assert the EXACT error: a bare `Exception` would pass on any failure at
    # all, including an unrelated import or type error, so it would not actually
    # prove that genuine key ambiguity is what gets rejected.
    with pytest.raises(ConfigurationSourceError, match="AmbiguousKeyError"):
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
