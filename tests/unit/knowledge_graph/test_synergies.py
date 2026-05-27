#!/usr/bin/python
"""Unit tests for KG version branching, semantic compaction, and ontological guardrails."""

import pytest
from unittest.mock import MagicMock, patch, ANY
from typing import Any

from agent_utilities.knowledge_graph.core.kg_versioning import (
    KGVersionEngine,
    KGTransaction,
    KGCommit,
    SpeculativeGraphBrancher,
)
from agent_utilities.knowledge_graph.memory.memory_compaction import SemanticCompactor
from agent_utilities.security.tool_guard import check_ontological_guardrails


def test_speculative_graph_brancher():
    """Test creating a speculative branch, making changes, and merging with conflict checking."""
    main_engine = KGVersionEngine()
    main_state = {
        "nodes": {
            "node1": {"name": "Node 1", "type": "Test"},
            "node2": {"name": "Node 2", "type": "Test"},
        },
        "edges": [
            ("node1", "node2", "RELATED_TO")
        ],
    }

    brancher = SpeculativeGraphBrancher(main_engine, main_state)

    # 1. Create speculative branch
    branch_state = brancher.create_branch("branch-1")
    assert "node1" in branch_state["nodes"]

    # 2. Mutate speculative branch
    branch_state["nodes"]["node3"] = {"name": "Node 3", "type": "Test"}
    branch_state["edges"].append(("node2", "node3", "LINKS_TO"))

    # 3. Merge branch
    commit = brancher.merge_branch("branch-1")
    assert commit is not None
    assert "node3" in main_state["nodes"]
    assert ("node2", "node3", "LINKS_TO") in main_state["edges"]


def test_speculative_graph_brancher_conflict():
    """Test that a conflict is raised when a modified node was deleted in the main state."""
    main_engine = KGVersionEngine()
    main_state = {
        "nodes": {
            "node1": {"name": "Node 1", "type": "Test"},
        },
        "edges": [],
    }

    brancher = SpeculativeGraphBrancher(main_engine, main_state)
    branch_state = brancher.create_branch("branch-1")

    # Modify node1 in branch
    branch_state["nodes"]["node1"]["name"] = "Node 1 Modified"

    # Concurrently delete node1 in main_state
    main_nodes: Any = main_state["nodes"]
    main_nodes.pop("node1")

    # Try to merge branch - should raise ValueError/conflict
    with pytest.raises(ValueError, match="Merge Conflict"):
        brancher.merge_branch("branch-1")


def test_semantic_compactor():
    """Test that SemanticCompactor correctly aggregates and deletes raw traces."""
    mock_engine = MagicMock()
    # Mock return rows representing trace nodes
    mock_rows = [
        {"id": "process:1", "state": "completed", "tokens_used": 100},
        {"id": "process:2", "state": "completed", "tokens_used": 200},
        {"id": "process:3", "state": "failed", "tokens_used": 50},
    ]

    # Mocking rows returned by database execution
    mock_res = MagicMock()
    mock_res.rows = mock_rows
    mock_engine.backend.execute.return_value = mock_res

    compactor = SemanticCompactor(mock_engine)

    # If threshold is 2, it should perform compaction (since we have 3 nodes)
    deleted_count = compactor.compact_traces(agent_id="agent-123", threshold=2)

    assert deleted_count == 3

    # Check that it executed queries to merge the summary, link it, and delete the processes
    mock_engine.backend.execute.assert_any_call(
        ANY,
        {
            "summary_id": "summary:agent:agent-123:3_compacted",
            "name": "Compacted Trace Summary for Agent agent-123",
            "compacted_count": 3,
            "total_tokens": 350,
            "agent_id": "agent-123",
        }
    )
    mock_engine.backend.execute.assert_any_call(
        ANY,
        {"pid": "process:1"}
    )


def test_ontological_guardrails_no_targets():
    """Test that check_ontological_guardrails returns False when there are no target args."""
    res = check_ontological_guardrails("some_tool", {"arg1": "val1"})
    assert res is False


def test_ontological_guardrails_fallback():
    """Test fallback keywords check in check_ontological_guardrails."""
    res = check_ontological_guardrails("run_command", {"filepath": "/etc/passwd"})
    assert res is True


def test_ontological_guardrails_kg_integration():
    """Test check_ontological_guardrails with full KG integration when engine is present."""
    mock_engine = MagicMock()
    # Mock a networkx-like graph with a SecurityPolicyNode
    mock_graph = MagicMock()
    mock_graph.nodes.return_value = [
        ("policy1", {"type": "SecurityPolicyNode", "target": "restricted_dir", "name": "Restricted Directory Policy"})
    ]
    mock_engine.graph = mock_graph

    # Match target
    res = check_ontological_guardrails(
        "write_file",
        {"directory": "/home/user/restricted_dir/file.txt"},
        engine=mock_engine
    )
    assert res is True

    # No match target
    res_no_match = check_ontological_guardrails(
        "write_file",
        {"directory": "/home/user/allowed_dir/file.txt"},
        engine=mock_engine
    )
    assert res_no_match is False
