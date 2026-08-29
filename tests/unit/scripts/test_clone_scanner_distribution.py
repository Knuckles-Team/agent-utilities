"""Distribution contracts for the source-checkout clone-scanner surface."""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _load_build_backend():
    spec = importlib.util.spec_from_file_location(
        "_clone_scanner_distribution_build_backend", ROOT / "build_backend.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_clone_scanner_contract_and_workflow_use_the_same_pins() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        document = tomllib.load(stream)
    profile = document["tool"]["agent_utilities"]["clone_scanners"]

    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    pre_commit = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")

    assert profile["dupehound_version"] == "0.1.2"
    assert profile["jscpd_version"] == "5.0.16"
    assert "entry: python3 scripts/check_dupehound.py" in pre_commit
    assert "stages: [pre-commit]" in pre_commit
    assert "scripts/check_duplication.py enforce --base-ref" in pre_commit
    assert "stages: [manual, pre-push]" in pre_commit
    assert "cargo install dupehound --version 0.1.2 --locked" in workflow
    assert "npm install --global jscpd@5.0.16" in workflow
    assert "needs: [gates, clone-scanners]" in workflow
    assert "fetch-depth: 0" in workflow
    assert "github.event.pull_request.base.sha" in workflow
    assert "github.event.pull_request.head.sha" in workflow
    assert "github.event.before" in workflow
    assert 'git cat-file -e "$base^{commit}"' in workflow


def test_source_distribution_carries_the_clone_scanner_surface() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    required = (
        ".pre-commit-config.yaml",
        ".cccc.toml",
        ".kiss/kiss.toml",
        ".github/workflows/release.yml",
        "scripts/_clone_scanner_config.py",
        "scripts/_git_subprocess_env.py",
        "scripts/check_dupehound.py",
        "scripts/check_duplication.py",
    )

    assert all(f"include {path}" in manifest for path in required)


def test_runtime_wheel_excludes_source_checkout_clone_scanners() -> None:
    backend = _load_build_backend()
    source_only = (
        "scripts/_clone_scanner_config.py",
        "scripts/_git_subprocess_env.py",
        "scripts/check_dupehound.py",
        "scripts/check_duplication.py",
    )

    assert all(not backend._retain_runtime_member(path) for path in source_only)
    assert backend._retain_runtime_member("scripts/release/check_release_wheel.py")
