"""Distribution contracts for the source-checkout clone-scanner surface."""

from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
_EXACT_VERSION = re.compile(r"\d+\.\d+\.\d+")
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _load_build_backend():
    spec = importlib.util.spec_from_file_location(
        "_clone_scanner_distribution_build_backend", ROOT / "build_backend.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _hooks() -> dict[str, dict]:
    config = yaml.safe_load(
        (ROOT / ".config" / "pre-commit.yaml").read_text(encoding="utf-8")
    )
    return {hook["id"]: hook for repo in config["repos"] for hook in repo["hooks"]}


def test_scanner_versions_are_pinned_once_and_installed_from_that_pin() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        profile = tomllib.load(stream)["tool"]["agent_utilities"]["clone_scanners"]
    installer = (ROOT / "scripts" / "install_scanners.sh").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    for key in ("dupehound_version", "jscpd_version"):
        assert _EXACT_VERSION.fullmatch(profile[key]), key
        assert f"pin {key}" in installer
    # CI provisions through the same installer, never a second copy of the pins.
    assert "scripts/install_scanners.sh" in workflow
    for key in ("dupehound_version", "jscpd_version"):
        assert profile[key] not in workflow


def test_clone_gates_run_at_their_declared_stages() -> None:
    hooks = _hooks()

    dupehound = hooks["clone-dupehound-changed-functions"]
    assert "scripts/check_dupehound.py" in dupehound["entry"]
    assert dupehound["stages"] == ["pre-commit"]
    jscpd = hooks["clone-jscpd-diff"]
    assert "scripts/check_duplication.py enforce --base-ref" in jscpd["entry"]
    assert jscpd["stages"] == ["manual"]


def test_release_clone_range_is_the_immutable_candidate() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "needs: [gates, clone-scanners]" in workflow
    assert "fetch-depth: 0" in workflow
    assert "github.event.pull_request.base.sha" in workflow
    assert "github.event.pull_request.head.sha" in workflow
    assert "github.event.before" in workflow
    assert 'git cat-file -e "$base^{commit}"' in workflow


def test_every_sibling_path_source_is_pinned_in_the_siblings_lock() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        sources = tomllib.load(stream)["tool"]["uv"]["sources"]
    declared = {
        name
        for name, entry in sources.items()
        if isinstance(entry, dict)
        and str(entry.get("path", "")).startswith(".uv-workspace-siblings/")
    }
    pinned = {}
    for line in (ROOT / "scripts" / "siblings.lock").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, url, commit, when = line.split()
        assert url.startswith("https://"), name
        assert _COMMIT.fullmatch(commit), name
        assert when in {"always", "engine"}, name
        pinned[name] = when

    assert declared <= set(pinned)
    # CI materializes the same pins through the bootstrap, not a copied SHA.
    action = (
        ROOT / ".github" / "actions" / "checkout-agent-connector-sdk" / "action.yml"
    ).read_text(encoding="utf-8")
    assert "scripts/bootstrap.sh --siblings-only" in action


def test_source_distribution_carries_the_clone_scanner_surface() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    required = (
        ".config/pre-commit.yaml",
        ".config/codespell.ignore",
        ".cccc.toml",
        ".kiss/kiss.toml",
        ".github/workflows/release.yml",
        "scripts/_clone_scanner_config.py",
        "scripts/_gate_skip.py",
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
