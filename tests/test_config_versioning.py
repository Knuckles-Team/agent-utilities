"""Tests for AgentConfigVersionManager (CONCEPT:AHE-3.13).

@pytest.mark.concept("AHE-3.13")
"""

import pytest

from agent_utilities.observability.config_versioning import (
    AgentConfigSnapshot,
    AgentConfigVersionManager,
    ConfigDiff,
)


@pytest.fixture
def manager() -> AgentConfigVersionManager:
    return AgentConfigVersionManager()


@pytest.fixture
def populated_manager() -> AgentConfigVersionManager:
    mgr = AgentConfigVersionManager()
    mgr.create_version(
        "agent-a",
        {
            "model_name": "gpt-4o",
            "instruction": "Be helpful",
            "tools_config": {"web_search": True},
        },
        author="alice",
        change_summary="Initial config",
    )
    mgr.create_version(
        "agent-a",
        {
            "model_name": "gpt-4o-mini",
            "instruction": "Be concise",
            "tools_config": {"web_search": True, "calculator": True},
        },
        author="bob",
        change_summary="Switched to mini model",
    )
    return mgr


# ---------------------------------------------------------------------------
# Snapshot model
# ---------------------------------------------------------------------------


class TestAgentConfigSnapshot:
    def test_auto_id(self):
        s = AgentConfigSnapshot(agent_name="test", version_number=3)
        assert s.id == "config_v:test:v3"

    def test_timestamp_auto(self):
        s = AgentConfigSnapshot(agent_name="test")
        assert s.timestamp > 0


# ---------------------------------------------------------------------------
# Version creation
# ---------------------------------------------------------------------------


class TestVersionCreation:
    def test_first_version(self, manager):
        v = manager.create_version(
            "agent-a",
            {
                "model_name": "gpt-4o",
                "instruction": "test",
            },
        )
        assert v.version_number == 1
        assert v.parent_version_id == ""
        assert v.agent_name == "agent-a"
        assert v.model_name == "gpt-4o"

    def test_sequential_numbering(self, manager):
        manager.create_version("agent-a", {"model_name": "m1"})
        v2 = manager.create_version("agent-a", {"model_name": "m2"})
        assert v2.version_number == 2
        assert v2.parent_version_id != ""

    def test_author_and_summary(self, manager):
        v = manager.create_version(
            "agent-a", {}, author="alice", change_summary="Initial setup"
        )
        assert v.created_by == "alice"
        assert v.change_summary == "Initial setup"


# ---------------------------------------------------------------------------
# Version history
# ---------------------------------------------------------------------------


class TestVersionHistory:
    def test_newest_first(self, populated_manager):
        history = populated_manager.get_version_history("agent-a")
        assert len(history) == 2
        assert history[0].version_number == 2  # Newest first
        assert history[1].version_number == 1

    def test_limit(self, populated_manager):
        history = populated_manager.get_version_history("agent-a", limit=1)
        assert len(history) == 1
        assert history[0].version_number == 2

    def test_unknown_agent(self, manager):
        history = manager.get_version_history("nonexistent")
        assert len(history) == 0


# ---------------------------------------------------------------------------
# Get specific version
# ---------------------------------------------------------------------------


class TestGetVersion:
    def test_get_by_number(self, populated_manager):
        v = populated_manager.get_version("agent-a", 1)
        assert v is not None
        assert v.model_name == "gpt-4o"

    def test_get_nonexistent(self, populated_manager):
        v = populated_manager.get_version("agent-a", 99)
        assert v is None

    def test_get_latest(self, populated_manager):
        latest = populated_manager.get_latest("agent-a")
        assert latest is not None
        assert latest.version_number == 2


# ---------------------------------------------------------------------------
# Diff versions
# ---------------------------------------------------------------------------


class TestDiffVersions:
    def test_diff_basic(self, populated_manager):
        diff = populated_manager.diff_versions("agent-a", 1, 2)
        assert diff.from_version == 1
        assert diff.to_version == 2
        assert "model_name" in diff.changes
        assert diff.changes["model_name"]["from"] == "gpt-4o"
        assert diff.changes["model_name"]["to"] == "gpt-4o-mini"

    def test_diff_includes_instruction(self, populated_manager):
        diff = populated_manager.diff_versions("agent-a", 1, 2)
        assert "instruction" in diff.changes

    def test_diff_summary_human_readable(self, populated_manager):
        diff = populated_manager.diff_versions("agent-a", 1, 2)
        assert "model:" in diff.summary
        assert "→" in diff.summary

    def test_diff_no_changes(self, manager):
        manager.create_version("a", {"model_name": "m1"})
        manager.create_version("a", {"model_name": "m1"})
        diff = manager.diff_versions("a", 1, 2)
        assert diff.summary == "No changes"

    def test_diff_missing_version(self, populated_manager):
        diff = populated_manager.diff_versions("agent-a", 1, 99)
        assert "not found" in diff.summary


# ---------------------------------------------------------------------------
# Rollback (forward-only)
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rollback_creates_new_version(self, populated_manager):
        v3 = populated_manager.rollback_to_version("agent-a", 1)
        assert v3.version_number == 3  # New version, not destructive
        assert v3.model_name == "gpt-4o"  # Copied from v1
        assert "Rollback to version 1" in v3.change_summary

    def test_rollback_preserves_history(self, populated_manager):
        populated_manager.rollback_to_version("agent-a", 1)
        history = populated_manager.get_version_history("agent-a")
        assert len(history) == 3  # v1, v2, v3 (rollback)

    def test_rollback_nonexistent_raises(self, populated_manager):
        with pytest.raises(ValueError, match="not found"):
            populated_manager.rollback_to_version("agent-a", 99)

    def test_rollback_copies_all_fields(self, manager):
        manager.create_version(
            "a",
            {
                "model_name": "m1",
                "instruction": "i1",
                "tools_config": {"t": 1},
                "guardrail_config": {"g": 2},
                "mcp_servers": ["s1"],
            },
        )
        manager.create_version("a", {"model_name": "m2"})
        v3 = manager.rollback_to_version("a", 1)
        assert v3.model_name == "m1"
        assert v3.instruction == "i1"
        assert v3.tools_config == {"t": 1}
        assert v3.guardrail_config == {"g": 2}
        assert v3.mcp_servers == ["s1"]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_structure(self, populated_manager):
        s = populated_manager.summary()
        assert s["agents_tracked"] == 1
        assert s["total_versions"] == 2
        assert "agent-a" in s["agents"]

    def test_summary_empty(self, manager):
        s = manager.summary()
        assert s["agents_tracked"] == 0
        assert s["total_versions"] == 0


# ---------------------------------------------------------------------------
# Multi-agent isolation
# ---------------------------------------------------------------------------


class TestMultiAgent:
    def test_separate_version_chains(self, manager):
        manager.create_version("agent-a", {"model_name": "m1"})
        manager.create_version("agent-b", {"model_name": "m2"})
        manager.create_version("agent-a", {"model_name": "m3"})

        assert manager.get_latest("agent-a").version_number == 2
        assert manager.get_latest("agent-b").version_number == 1
