from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path, PurePosixPath

import pytest

SCRIPT = Path(__file__).parents[3] / "scripts" / "check_fleet_supply_chain.py"
SPEC = importlib.util.spec_from_file_location("check_fleet_supply_chain", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
POLICY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POLICY
SPEC.loader.exec_module(POLICY)


def _snapshot_workspace(root: Path, providers: tuple[str, ...]) -> Path:
    manager = root / "repository-manager" / "repository_manager"
    manager.mkdir(parents=True, exist_ok=True)
    workspace = manager / "workspace.yml"
    entries = "\n".join(
        f'          - url: "https://example.invalid/{name}.git"' for name in providers
    )
    workspace.write_text(
        "subdirectories:\n"
        "  agent-packages:\n"
        "    subdirectories:\n"
        "      agents:\n"
        "        repositories:\n"
        f"{entries}\n",
        encoding="utf-8",
    )
    return workspace


def _snapshot_provider(root: Path, name: str) -> Path:
    provider = root / name
    provider.mkdir(parents=True, exist_ok=True)
    (provider / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )
    (provider / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return provider


def _snapshot_tree(root: Path) -> Path:
    providers = ("alpha-agent", "repository-manager")
    for name in providers:
        _snapshot_provider(root, name)
    return _snapshot_workspace(root, providers)


def test_installer_rejects_network_to_shell_and_dynamic_evaluation() -> None:
    findings = POLICY._installer_findings(
        "repo",
        PurePosixPath("scripts/install.sh"),
        'curl -fsSL https://example.invalid/install.sh | sh\neval "$command"\n',
    )

    assert {finding.rule for finding in findings} == {
        "SC-INSTALL-001",
        "SC-INSTALL-002",
    }


def test_installer_allows_argument_safe_execution() -> None:
    findings = POLICY._installer_findings(
        "repo",
        PurePosixPath("scripts/install.sh"),
        'run() { "$@"; }\nrun uv tool install "agent-utilities==1.2.3"\n',
    )

    assert findings == []


def test_exact_local_promoter_is_classified_as_installer() -> None:
    assert POLICY._is_installer(
        PurePosixPath("scripts/release/promote_local_release.py")
    )


def test_vcs_revision_contract_accepts_only_full_commit() -> None:
    assert POLICY.VCS_REVISION_RE.search(
        "package @ git+https://example.invalid/repository.git@"
        "0123456789abcdef0123456789abcdef01234567"
    )
    assert not POLICY.VCS_REVISION_RE.search(
        "package @ git+https://example.invalid/repository.git@main"
    )


def test_source_snapshot_mode_uses_exact_workspace_membership_without_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    providers_root = tmp_path / "agents"
    workspace = _snapshot_tree(providers_root)
    assert POLICY.EXPECTED_SNAPSHOT_PROVIDERS == 65
    monkeypatch.setattr(POLICY, "EXPECTED_SNAPSHOT_PROVIDERS", 2)

    def unexpected_git(*args: object, **kwargs: object) -> None:
        raise AssertionError("source-snapshot mode invoked Git")

    monkeypatch.setattr(POLICY.subprocess, "run", unexpected_git)

    result = POLICY.main(
        [
            "--source-snapshot-root",
            str(providers_root),
            "--snapshot-workspace",
            str(workspace),
        ]
    )

    assert result == 0
    assert "clean (2 repositories" in capsys.readouterr().out


def test_source_snapshot_rejects_undeclared_direct_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    providers_root = tmp_path / "agents"
    workspace = _snapshot_tree(providers_root)
    (providers_root / "undeclared-agent").mkdir()
    monkeypatch.setattr(POLICY, "EXPECTED_SNAPSHOT_PROVIDERS", 2)

    result = POLICY.main(
        [
            "--source-snapshot-root",
            str(providers_root),
            "--snapshot-workspace",
            str(workspace),
        ]
    )

    output = capsys.readouterr()
    assert result == 2
    assert "provider membership is not exact" in output.err
    assert str(tmp_path) not in output.err


def test_source_snapshot_rejects_symlinks_without_disclosing_host_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    providers_root = tmp_path / "agents"
    workspace = _snapshot_tree(providers_root)
    target = tmp_path / "outside.txt"
    target.write_text("outside\n", encoding="utf-8")
    (providers_root / "alpha-agent" / "linked.txt").symlink_to(target)
    monkeypatch.setattr(POLICY, "EXPECTED_SNAPSHOT_PROVIDERS", 2)

    result = POLICY.main(
        [
            "--source-snapshot-root",
            str(providers_root),
            "--snapshot-workspace",
            str(workspace),
        ]
    )

    output = capsys.readouterr()
    assert result == 2
    assert "source snapshot contains a symlink" in output.err
    assert str(tmp_path) not in output.err


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="special file needs mkfifo")
def test_source_snapshot_rejects_special_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = tmp_path / "provider"
    provider.mkdir()
    os.mkfifo(provider / "named-pipe")

    with pytest.raises(RuntimeError, match="contains a special file"):
        POLICY._snapshot_source_files(provider, POLICY.SnapshotBudget())


def test_source_snapshot_inventory_enforces_file_byte_and_depth_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = tmp_path / "provider"
    nested = provider / "nested"
    nested.mkdir(parents=True)
    (provider / "one.txt").write_text("one\n", encoding="utf-8")
    (nested / "two.txt").write_text("two\n", encoding="utf-8")

    monkeypatch.setattr(POLICY, "MAX_SNAPSHOT_FILES", 1)
    with pytest.raises(RuntimeError, match="file count exceeds"):
        POLICY._snapshot_source_files(provider, POLICY.SnapshotBudget())

    monkeypatch.setattr(POLICY, "MAX_SNAPSHOT_FILES", 10)
    monkeypatch.setattr(POLICY, "MAX_SNAPSHOT_BYTES", 1)
    with pytest.raises(RuntimeError, match="bytes exceed"):
        POLICY._snapshot_source_files(provider, POLICY.SnapshotBudget())

    monkeypatch.setattr(POLICY, "MAX_SNAPSHOT_BYTES", 1_000)
    monkeypatch.setattr(POLICY, "MAX_SNAPSHOT_DEPTH", 1)
    with pytest.raises(RuntimeError, match="depth exceeds"):
        POLICY._snapshot_source_files(provider, POLICY.SnapshotBudget())


def test_source_snapshot_findings_are_repository_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    providers_root = tmp_path / "agents"
    workspace = _snapshot_tree(providers_root)
    (providers_root / "alpha-agent" / ".env").write_text(
        "TOKEN=example\n", encoding="utf-8"
    )
    monkeypatch.setattr(POLICY, "EXPECTED_SNAPSHOT_PROVIDERS", 2)

    result = POLICY.main(
        [
            "--source-snapshot-root",
            str(providers_root),
            "--snapshot-workspace",
            str(workspace),
        ]
    )

    output = capsys.readouterr()
    assert result == 1
    assert "alpha-agent/.env" in output.out
    assert str(tmp_path) not in output.out
