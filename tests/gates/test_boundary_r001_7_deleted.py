"""AU-BOUNDARY-R001.7: `agent_utilities/gateway/widgets/genius_agent.py` is gone.

Deliberately a standalone file (not the shared
``test_boundary_r001_r010_deletion_census.py``) so this one row's lane never
conflicts with sibling deletion-row lanes editing that shared census file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DELETED_MODULE = (
    REPO_ROOT / "agent_utilities" / "gateway" / "widgets" / "genius_agent.py"
)
MODULE_DOTTED_PATH = "agent_utilities.gateway.widgets.genius_agent"


@pytest.mark.spec("AU-BOUNDARY-R001.7")
def test_gateway_widgets_genius_agent_file_deleted() -> None:
    assert not DELETED_MODULE.exists(), (
        f"{DELETED_MODULE} still exists; AU-BOUNDARY-R001.7 requires it deleted."
    )


@pytest.mark.spec("AU-BOUNDARY-R001.7")
def test_gateway_widgets_genius_agent_no_importers_remain() -> None:
    result = subprocess.run(
        [
            "git",
            "grep",
            "-nE",
            r"gateway\.widgets\.genius_agent|widgets\.genius_agent|widgets import genius_agent",
            "--",
            "agent_utilities",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # git grep exits 1 when no matches are found; anything else is a real error.
    assert result.returncode in (0, 1), f"git grep failed unexpectedly: {result.stderr}"
    assert result.returncode == 1 and not result.stdout.strip(), (
        "Found remaining production references to the deleted "
        f"{MODULE_DOTTED_PATH} module:\n{result.stdout}"
    )
