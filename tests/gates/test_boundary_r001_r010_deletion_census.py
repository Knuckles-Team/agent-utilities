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
