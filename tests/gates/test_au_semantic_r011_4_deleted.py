"""AU-SEMANTIC-R011.4: the local Postgres queue backend is deleted."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.gates._deletion_support import files_importing

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "agent_utilities/knowledge_graph/core/postgres_queue_backend.py"


@pytest.mark.spec("AU-SEMANTIC-R011.4")
def test_postgres_queue_backend_is_deleted_and_unimported():
    assert not MODULE.exists()
    hits = files_importing(
        [REPO_ROOT / "agent_utilities", REPO_ROOT / "scripts"],
        ("postgres_queue_backend",),
        REPO_ROOT,
    )
    assert hits == frozenset()
