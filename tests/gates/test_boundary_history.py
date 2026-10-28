"""Check the historical evidence guard against a hermetic Git history."""

import subprocess
from pathlib import Path

import pytest

from scripts.check_boundary_history import (
    BACKENDS,
    BASE,
    OIDC,
    check_deployment,
    check_toolkit,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def history(tmp_path: Path) -> tuple[Path, str, str, str, str]:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Test Author")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    for name in (BACKENDS, OIDC, BASE):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("initial\n")
    _git(tmp_path, "add", BACKENDS, OIDC, BASE)
    _git(tmp_path, "commit", "-qm", "original")
    original = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/gate.py").write_text("pass\n")
    _git(tmp_path, "add", "scripts/gate.py")
    _git(tmp_path, "commit", "-qm", "gate")
    gate = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "rm", "-q", BACKENDS)
    _git(tmp_path, "commit", "-qm", "remove backends")
    backends_cut = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "rm", "-q", OIDC, BASE)
    _git(tmp_path, "commit", "-qm", "remove oidc and base")
    cut = _git(tmp_path, "rev-parse", "HEAD")
    return tmp_path, original, gate, backends_cut, cut


@pytest.mark.spec("AU-BOUNDARY-R002.1")
def test_deployment_history_requires_original_and_later_deletions(
    history: tuple[Path, str, str, str, str],
) -> None:
    root, original, _, backends_cut, cut = history
    check_deployment(root, original, backends_cut, cut, "HEAD")
    with pytest.raises(ValueError, match="original deployment evidence"):
        check_deployment(root, backends_cut, backends_cut, cut, "HEAD")
    with pytest.raises(ValueError, match="deployment deletion evidence"):
        check_deployment(root, original, backends_cut, backends_cut, "HEAD")


@pytest.mark.spec("AU-BOUNDARY-R010.1")
def test_toolkit_history_requires_gate_only_change_and_later_deletion(
    history: tuple[Path, str, str, str, str],
) -> None:
    root, original, gate, _, cut = history
    check_toolkit(root, gate, cut, "HEAD")
    with pytest.raises(ValueError, match="gate-only"):
        check_toolkit(root, original, cut, "HEAD")
    with pytest.raises(ValueError, match="toolkit history evidence"):
        check_toolkit(root, gate, gate, "HEAD")
