"""Deletion census for AU-BOUNDARY-R002.2 (and future R001/R010-family
siblings sharing this module).

AU-BOUNDARY-R002 retires AU's deployment and skill-certification tooling in
favor of graph-os. R002.1 found that the PR which moved `R002` to LANDED
never actually deleted `agent_utilities/deployment/backends.py` (or
`certification_oidc.py`); both files were still present at `origin/main`.
R002.2 is the bite-sized row that deletes `backends.py` for real and
repoints its one production importer (the CLI's `deploy-plan` console
command, which collided with graph-os's own deployment-planner entry point
and is removed outright per R002's "AU removes the console scripts that
... target the moved modules" text) and its dedicated unit test.

This gate asserts the file is gone and that no importer -- production or
test -- references it anymore, so the row cannot silently regress to the
R002.1 state (text says deleted, file still on disk).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "agent_utilities"
TESTS_ROOT = REPO_ROOT / "tests"

DELETED_MODULE_PATH = PACKAGE_ROOT / "deployment" / "backends.py"


def _files_importing(needle: str, roots: tuple[Path, ...]) -> frozenset[str]:
    """Relative (posix, from REPO_ROOT) paths of files whose import
    statements reference `needle` (a dotted module substring)."""
    hits: set[str] = set()
    for root in roots:
        for path in root.rglob("*.py"):
            try:
                source = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                if any(needle in name for name in names):
                    hits.add(path.relative_to(REPO_ROOT).as_posix())
                    break
    return frozenset(hits)


@pytest.mark.spec("AU-BOUNDARY-R002.2")
def test_deployment_backends_deleted() -> None:
    """`agent_utilities/deployment/backends.py` is gone and no importer --
    production or test -- still references `agent_utilities.deployment.backends`."""
    assert not DELETED_MODULE_PATH.exists(), (
        f"{DELETED_MODULE_PATH} still present; AU-BOUNDARY-R002.2 requires it deleted"
    )

    importers = _files_importing(
        "agent_utilities.deployment.backends", (PACKAGE_ROOT, TESTS_ROOT)
    )
    assert importers == frozenset(), (
        "no importer of agent_utilities.deployment.backends may remain, found: "
        f"{sorted(importers)}"
    )
"""Deletion census for the AU-BOUNDARY-R001 gateway-retirement children.

``AU-BOUNDARY-R001`` retires AU's duplicated ``agent_utilities/gateway/``
module once graph-os serves equivalent routes and widgets. Rows
``AU-BOUNDARY-R001.2`` through ``.8`` each delete exactly one AU-only file
from that package and repoint or delete its importers. This module is the
shared acceptance gate those child rows name: one regression per file,
asserting the file is gone from disk and that no production import of its
dotted module path remains anywhere in the repository (test-only references
inside this gate file itself are excluded from the scan).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "agent_utilities"


_SELF_PATH = "tests/gates/test_boundary_r001_r010_deletion_census.py"

# Actual Python import statement shapes that would still bind the deleted
# module -- deliberately narrower than a bare substring match so prose
# mentions in docstrings/comments (including this file's own) are not
# mistaken for a live importer.
_IMPORT_PATTERNS: tuple[str, ...] = (
    r"^\s*from agent_utilities\.gateway\.api import ",
    r"^\s*from agent_utilities\.gateway import api(\s|,|$)",
    r"^\s*import agent_utilities\.gateway\.api(\s|$)",
)


def _grep_importers(pattern: str) -> list[str]:
    """Return every ``path:line`` hit for ``pattern`` outside this gate
    file."""
    raw = subprocess.run(
        ["git", "grep", "-n", "-E", pattern, "--", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return [
        line
        for line in raw.splitlines()
        if not line.split(":", 1)[0].endswith(_SELF_PATH)
    ]


@pytest.mark.spec("AU-BOUNDARY-R001.2")
def test_gateway_api_deleted() -> None:
    """``agent_utilities/gateway/api.py`` is gone and no production import of
    ``agent_utilities.gateway.api`` (nor ``dashboard_router``'s mount)
    remains."""
    deleted_path = PACKAGE_ROOT / "gateway" / "api.py"
    assert not deleted_path.exists(), f"{deleted_path} should have been deleted"

    for pattern in _IMPORT_PATTERNS:
        importers = _grep_importers(pattern)
        assert importers == [], f"stale importers of agent_utilities.gateway.api: {importers}"
