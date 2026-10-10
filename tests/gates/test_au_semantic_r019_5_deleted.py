"""AU-SEMANTIC-R019.5: governance/relational_authority.py deletion gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.gates._deletion_support import files_importing

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.spec("AU-SEMANTIC-R019.5")
def test_relational_authority_module_is_deleted() -> None:
    assert not (
        REPO_ROOT / "agent_utilities" / "governance" / "relational_authority.py"
    ).exists()
    found = files_importing(
        [REPO_ROOT / "agent_utilities", REPO_ROOT / "scripts"],
        ("governance.relational_authority", "governance import relational_authority"),
        REPO_ROOT,
    )
    assert found == frozenset(), f"importers remain: {sorted(found)}"
