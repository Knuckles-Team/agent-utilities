"""CONCEPT:AU-OS.deployment.workspace-venv-reconciler (D-VS-8) — the budgeted
prune path exercised end-to-end against a REAL `uv` workspace with a real venv.

Unit tests (``tests/unit/deployment/test_venv_sync.py``) prove ``plan_prune``'s
candidate-selection logic and mock the apply step. This test proves the whole
thing for real: a genuine ``uv sync`` builds a genuine virtualenv, a genuine
``uv pip install`` adds a package the lock does not know about, and
``prune()`` removes exactly that package via a genuine ``uv pip uninstall`` —
while the real workspace member survives untouched. No network access is
required (every package here is a local, dependency-free path).

This is the live counterpart to the finding that motivated D-VS-8's fix: a
manual run on 2026-07-31 showed `sync(workspace, allow_uninstalls=1)` leaves
an extraneous package completely untouched, because `--inexact` (forced,
unconditionally, into every sanctioned `uv sync`) means uv's own dry-run plan
never proposes removing it in the first place. `prune()` exists precisely
because that path can never work.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from agent_utilities.deployment.venv_sync import VenvSyncError, Workspace, prune

pytestmark = pytest.mark.integration

_UV = shutil.which("uv")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _local_package(root: Path, name: str, module: str, value: int) -> Path:
    pkg_dir = root / name
    _write(
        pkg_dir / "pyproject.toml",
        f'[project]\nname = "{name}"\nversion = "0.1.0"\n'
        'requires-python = ">=3.9"\ndependencies = []\n\n'
        '[build-system]\nrequires = ["setuptools>=61"]\n'
        'build-backend = "setuptools.build_meta"\n',
    )
    _write(pkg_dir / module / "__init__.py", f"VALUE = {value}\n")
    return pkg_dir


@pytest.mark.skipif(_UV is None, reason="uv is not on PATH")
def test_prune_removes_an_extraneous_package_from_a_real_venv(
    tmp_path: Path,
) -> None:
    ws_root = tmp_path / "ws"
    _write(
        ws_root / "pyproject.toml",
        '[project]\nname = "root-project"\nversion = "0.1.0"\n'
        'requires-python = ">=3.9"\ndependencies = ["alpha"]\n\n'
        '[tool.uv.workspace]\nmembers = ["pkgs/*"]\n\n'
        '[tool.uv.sources]\nalpha = { workspace = true }\n',
    )
    _local_package(ws_root / "pkgs", "alpha", "alpha", value=1)
    extraneous_pkg_dir = _local_package(
        tmp_path / "extraneous", "extraneous-pkg", "extraneous_pkg", value=2
    )

    # A genuine `uv sync` builds a genuine venv from a genuine lock.
    subprocess.run(
        [_UV, "sync"], cwd=ws_root, check=True, capture_output=True, text=True
    )
    workspace = Workspace.discover(ws_root, uv=_UV)
    site = workspace.site_packages()
    assert site is not None
    assert any(site.glob("alpha-*.dist-info"))

    # A genuine `uv pip install` adds a package the lock does not know about.
    python = workspace.venv / "bin" / "python"
    subprocess.run(
        [_UV, "pip", "install", "--python", str(python), str(extraneous_pkg_dir)],
        cwd=ws_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert any(site.glob("extraneous_pkg-*.dist-info"))

    # A zero budget must refuse, exactly like sync()'s own default.
    with pytest.raises(VenvSyncError):
        prune(workspace, allow_uninstalls=0, ignore_activity=True)
    assert any(site.glob("extraneous_pkg-*.dist-info")), (
        "a refused prune must not have touched anything"
    )

    # A real budget removes exactly the extraneous package, for real.
    outcome = prune(workspace, allow_uninstalls=1, ignore_activity=True)
    assert outcome.applied is True
    assert [c.name for c in outcome.plan.candidates] == ["extraneous-pkg"]
    assert not any(site.glob("extraneous_pkg-*.dist-info")), (
        "extraneous-pkg should be gone from the real venv"
    )
    assert any(site.glob("alpha-*.dist-info")), (
        "the real workspace member must survive untouched"
    )
