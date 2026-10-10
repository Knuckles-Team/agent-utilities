"""AU-SEMANTIC-R011.3: the local Kafka queue backend is deleted."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.gates._deletion_support import files_importing

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "agent_utilities/knowledge_graph/core/kafka_queue_backend.py"


@pytest.mark.spec("AU-SEMANTIC-R011.3")
def test_kafka_queue_backend_deleted_and_unimported() -> None:
    assert not MODULE.exists()
    hits = files_importing(
        [REPO_ROOT / "agent_utilities", REPO_ROOT / "scripts"],
        ("kafka_queue_backend",),
        REPO_ROOT,
    )
    assert not hits, sorted(hits)
