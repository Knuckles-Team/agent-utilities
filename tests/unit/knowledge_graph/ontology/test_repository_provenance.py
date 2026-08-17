"""Unit tests for RepositorySnapshot/Branch/Tag/ChangeEvent (U-47, W01) —
additive Git identity/history nodes anchored to the EXISTING ``commit:<sha>``
node id convention ``enrichment/git_history.py`` already writes.

@pytest.mark.concept("AU-KG.ontology.repository-provenance-snapshot")
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ontology.repository_provenance import (
    POINTS_AT,
    SNAPSHOT_OF,
    Branch,
    ChangeEvent,
    RepositorySnapshot,
    Tag,
    branch_id,
    change_event_id,
    repository_snapshot_id,
    tag_id,
)

pytestmark = pytest.mark.concept("AU-KG.ontology.repository-provenance-snapshot")


def test_repository_snapshot_id_is_deterministic():
    a = repository_snapshot_id("agent-utilities", "abc123")
    b = repository_snapshot_id("agent-utilities", "abc123")
    assert a == b
    assert a != repository_snapshot_id("agent-utilities", "def456")


def test_repository_snapshot_links_to_existing_commit_node_id():
    snap = RepositorySnapshot(
        repo_id="agent-utilities", commit_sha="abc123", ref="main"
    )
    entities, relationships = snap.to_graph_slice()
    assert entities[0]["id"] == snap.snapshot_id
    assert entities[0]["node_type"] == "RepositorySnapshot"
    assert relationships == [
        {
            "source": snap.snapshot_id,
            "target": "commit:abc123",
            "relationship": SNAPSHOT_OF,
        }
    ]


def test_branch_points_at_head_commit():
    branch = Branch(repo_id="agent-utilities", name="main", head_commit_sha="abc123")
    entities, relationships = branch.to_graph_slice()
    assert entities[0]["id"] == branch_id("agent-utilities", "main")
    assert relationships[0]["relationship"] == POINTS_AT
    assert relationships[0]["target"] == "commit:abc123"


def test_tag_points_at_commit():
    tag = Tag(
        repo_id="agent-utilities",
        name="v1.0.0",
        commit_sha="def456",
        annotation="release",
    )
    entities, relationships = tag.to_graph_slice()
    assert entities[0]["id"] == tag_id("agent-utilities", "v1.0.0")
    assert entities[0]["annotation"] == "release"
    assert relationships[0]["target"] == "commit:def456"


def test_change_event_id_deterministic_per_kind():
    a = change_event_id("agent-utilities", "abc123", "commit")
    b = change_event_id("agent-utilities", "abc123", "commit")
    c = change_event_id("agent-utilities", "abc123", "branch_move")
    assert a == b
    assert a != c


def test_change_event_to_node_carries_subject_when_present():
    event = ChangeEvent(
        repo_id="agent-utilities",
        commit_sha="abc123",
        kind="commit",
        occurred_at="2026-08-16T00:00:00Z",
        subject_id="file:foo.py",
    )
    node = event.to_node()
    assert node["node_type"] == "ChangeEvent"
    assert node["subject_id"] == "file:foo.py"


def test_change_event_to_node_omits_subject_when_absent():
    event = ChangeEvent(
        repo_id="agent-utilities",
        commit_sha="abc123",
        kind="tag_cut",
        occurred_at="2026-08-16T00:00:00Z",
    )
    assert "subject_id" not in event.to_node()
