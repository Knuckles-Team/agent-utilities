"""Records of what the boundary gate commits actually shipped.

``AU-BOUNDARY-R001``, ``R003`` and ``R004`` were marked LANDED after commits
that only added the ownership/coverage gate scripts. These tests pin that
history: neither commit deleted any retirement target.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PR23_MERGE = "aa5eba73e0f7"
GATE_COMMIT = "2456745c1e4d"
GATE_SCRIPTS = {
    "scripts/boundary_inventory.py",
    "scripts/check_boundary_coverage.py",
    "scripts/check_boundary_owners.py",
}


def _name_status(*args: str) -> dict[str, str]:
    """Map changed path to its status letter for a git diff/show invocation."""
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip("boundary gate commit not available in this clone")
    changes: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        status, _, path = line.partition("\t")
        if path:
            changes[path] = status[0]
    return changes


def _assert_gate_only(changes: dict[str, str], protected: tuple[str, ...]) -> None:
    assert GATE_SCRIPTS <= set(changes)
    assert "D" not in set(changes.values())
    touched = [p for p in changes if p.startswith(protected)]
    assert touched == []


@pytest.mark.spec("AU-BOUNDARY-R001.1")
def test_pr23_shipped_gate_scripts_not_gateway_deletion() -> None:
    changes = _name_status("diff", "--name-status", f"{PR23_MERGE}^1", PR23_MERGE)
    _assert_gate_only(changes, ("agent_utilities/gateway/",))


@pytest.mark.spec("AU-BOUNDARY-R003.1")
def test_gate_commit_did_not_delete_mcp_host() -> None:
    changes = _name_status("show", "--name-status", "--format=", GATE_COMMIT)
    _assert_gate_only(
        changes,
        (
            "agent_utilities/mcp/kg_server.py",
            "agent_utilities/mcp/_graphos_action_manifest.py",
            "agent_utilities/sdd/watcher.py",
        ),
    )


@pytest.mark.spec("AU-BOUNDARY-R004.1")
def test_gate_commit_did_not_delete_multiplexer_family() -> None:
    changes = _name_status("show", "--name-status", "--format=", GATE_COMMIT)
    _assert_gate_only(
        changes,
        (
            "agent_utilities/mcp/multiplexer.py",
            "agent_utilities/mcp/shared_multiplexer.py",
        ),
    )
