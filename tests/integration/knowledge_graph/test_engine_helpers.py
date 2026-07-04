"""Tests for IntelligenceGraphEngine KG-first helpers.

CONCEPT:AU-KG.query.object-graph-mapper — Identity Management
CONCEPT:AU-KG.query.object-graph-mapper — Prompt Management
CONCEPT:AU-KG.query.object-graph-mapper — Granular Resource Queries
CONCEPT:AU-KG.query.object-graph-mapper — Workspace Reload
"""

import pytest

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.models.knowledge_graph import (
    RegistryEdgeType,
)


@pytest.fixture(autouse=True)
def _reset_active_engine():
    """Ensure each test starts with a clean singleton."""
    IntelligenceGraphEngine._ACTIVE_ENGINE = None
    yield
    IntelligenceGraphEngine._ACTIVE_ENGINE = None


@pytest.fixture
def engine():
    import uuid

    unique_name = f"test_graph_{uuid.uuid4().hex}"
    GraphComputeEngine(graph_name=unique_name, backend_type="rust")
    return IntelligenceGraphEngine(db_path=":memory:")


# ─────────────────────────────────────────────────────────────────────
#  CONCEPT:AU-KG.query.object-graph-mapper — Identity Management
# ─────────────────────────────────────────────────────────────────────


class TestIdentityManagement:
    """Tests for get/add/update_agent_identity (CONCEPT:AU-KG.query.object-graph-mapper)."""

    def test_get_identity_empty_graph(self, engine: IntelligenceGraphEngine):
        """Returns a default identity when graph is empty."""
        identity = engine.get_agent_identity()
        assert identity["name"] == "Agent"
        assert identity["content"] == ""

    def test_add_identity(self, engine: IntelligenceGraphEngine):
        """Creates a new identity node in the graph."""
        result = engine.add_agent_identity(
            {
                "name": "TestBot",
                "description": "A test bot",
                "content": "You are TestBot",
            }
        )
        assert result["name"] == "TestBot"
        assert "id" in result
        assert result["id"] in engine.graph

    def test_get_identity_after_add(self, engine: IntelligenceGraphEngine):
        """Can retrieve the identity after adding it."""
        engine.add_agent_identity(
            {
                "name": "TestBot",
                "description": "A test bot",
                "content": "You are TestBot",
            }
        )
        identity = engine.get_agent_identity()
        assert identity["name"] == "TestBot"

    def test_update_identity(self, engine: IntelligenceGraphEngine):
        """Updates an existing identity in the graph."""
        engine.add_agent_identity(
            {
                "name": "Original",
                "description": "v1",
                "content": "Original prompt",
            }
        )
        engine.update_agent_identity(
            {
                "name": "Updated",
                "description": "v2",
            }
        )
        identity = engine.get_agent_identity()
        assert identity["name"] == "Updated"
        assert identity["description"] == "v2"

    def test_update_identity_creates_if_missing(self, engine: IntelligenceGraphEngine):
        """update_agent_identity creates a node if none exists."""
        engine.update_agent_identity(
            {
                "name": "NewBot",
                "content": "New content",
            }
        )
        identity = engine.get_agent_identity()
        assert identity["name"] == "NewBot"


# ─────────────────────────────────────────────────────────────────────
#  CONCEPT:AU-KG.query.object-graph-mapper — Prompt Management
# ─────────────────────────────────────────────────────────────────────


class TestPromptManagement:
    """Tests for prompt CRUD and versioning (CONCEPT:AU-KG.query.object-graph-mapper)."""

    def test_add_prompt(self, engine: IntelligenceGraphEngine):
        """Creates a new prompt node."""
        result = engine.add_prompt(
            content="You are a researcher.",
            name="research-prompt",
            author="user",
            description="Research specialist",
        )
        assert result["name"] == "research-prompt"
        assert result["content"] == "You are a researcher."
        assert result["version_number"] == 1
        assert result["id"].startswith("prompt:")

    def test_get_prompt(self, engine: IntelligenceGraphEngine):
        """Retrieves a prompt by ID."""
        created = engine.add_prompt(content="Test", name="test")
        retrieved = engine.get_prompt(created["id"])
        assert retrieved is not None
        assert retrieved["id"] == created["id"]

    def test_get_prompt_not_found(self, engine: IntelligenceGraphEngine):
        """Returns None for non-existent prompt."""
        assert engine.get_prompt("nonexistent") is None

    def test_get_all_prompts(self, engine: IntelligenceGraphEngine):
        """Lists all prompts."""
        engine.add_prompt(content="A", name="prompt-a")
        engine.add_prompt(content="B", name="prompt-b")
        prompts = engine.get_all_prompts()
        assert len(prompts) >= 2

    def test_get_prompts_list_alias(self, engine: IntelligenceGraphEngine):
        """get_prompts_list is an alias for get_all_prompts."""
        engine.add_prompt(content="A", name="alias-test")
        assert engine.get_prompts_list() == engine.get_all_prompts()


class TestPromptVersioning:
    """Tests for prompt versioning and rollback (CONCEPT:AU-KG.query.object-graph-mapper)."""

    def test_update_creates_new_version(self, engine: IntelligenceGraphEngine):
        """update_prompt creates a SUPERSEDES link."""
        v1 = engine.add_prompt(content="v1", name="versioned")
        v2 = engine.update_prompt(v1["id"], content="v2")
        assert v2["id"] != v1["id"]
        assert v2["content"] == "v2"
        assert v2["parent_id"] == v1["id"]

        # Check edge exists. The graph engine canonicalizes the relationship
        # under ``rel_type`` (the uppercased relationship-type slot).
        has_supersedes = any(
            edata.get("rel_type") == RegistryEdgeType.SUPERSEDES.name
            for _, _, edata in engine.graph.out_edges(v2["id"], data=True)
        )
        assert has_supersedes

    def test_update_nonexistent_raises(self, engine: IntelligenceGraphEngine):
        """update_prompt raises ValueError for missing prompt."""
        with pytest.raises(ValueError, match="not found"):
            engine.update_prompt("nonexistent", content="test")

    def test_version_history(self, engine: IntelligenceGraphEngine):
        """get_prompt_versions walks the SUPERSEDES chain."""
        v1 = engine.add_prompt(content="v1", name="history-test")
        _v2 = engine.update_prompt(v1["id"], content="v2")  # noqa: F841

        versions = engine.get_prompt_versions(v1["id"])
        assert len(versions) >= 1

    def test_rollback_creates_new_version(self, engine: IntelligenceGraphEngine):
        """Rollback creates a forward version copying old content."""
        v1 = engine.add_prompt(content="original", name="rollback-test")
        v2 = engine.update_prompt(v1["id"], content="changed")
        v3 = engine.rollback_prompt(v2["id"], v1["id"])

        assert v3["id"] != v1["id"]
        assert v3["id"] != v2["id"]
        assert v3["content"] == "original"  # Content restored
        assert v3["author"] == "rollback"

    def test_rollback_nonexistent_raises(self, engine: IntelligenceGraphEngine):
        """Rollback to non-existent version raises ValueError."""
        v1 = engine.add_prompt(content="test", name="rollback-err")
        with pytest.raises(ValueError, match="not found"):
            engine.rollback_prompt(v1["id"], "nonexistent")


# ─────────────────────────────────────────────────────────────────────
#  CONCEPT:AU-KG.query.object-graph-mapper — Granular Resource Queries
# ─────────────────────────────────────────────────────────────────────


class TestGranularResourceQueries:
    """Tests for get_skills, get_tools, toggle_resource (CONCEPT:AU-KG.query.object-graph-mapper)."""

    def test_get_skills_empty(self, engine: IntelligenceGraphEngine):
        """Returns empty list when no skills are in the graph."""
        assert engine.get_skills() == []

    def test_get_skills_from_graph(self, engine: IntelligenceGraphEngine):
        """Finds skill-type nodes in the in-memory graph."""
        engine.graph.add_node(
            "skill:code-enhancer",
            type="skill",
            name="code-enhancer",
            description="Code analysis",
        )
        skills = engine.get_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "code-enhancer"
        assert skills[0]["type"] == "skill"

    def test_get_skills_by_resource_type(self, engine: IntelligenceGraphEngine):
        """Also finds AGENT_SKILL resource_type nodes."""
        engine.graph.add_node(
            "res:web-search",
            type="callable_resource",
            resource_type="agent_skill",
            name="web-search",
            description="Web search capability",
        )
        skills = engine.get_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "web-search"

    def test_get_tools_empty(self, engine: IntelligenceGraphEngine):
        """Returns empty list when no tools are in the graph."""
        assert engine.get_tools() == []

    def test_get_tools_from_graph(self, engine: IntelligenceGraphEngine):
        """Finds mcp_tool-type nodes."""
        engine.graph.add_node(
            "tool:jira-search",
            resource_type="mcp_tool",
            name="jira-search",
            description="Search Jira issues",
            endpoint="atlassian-mcp",
        )
        tools = engine.get_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "jira-search"
        assert tools[0]["type"] == "mcp_tool"

    def test_get_skills_sorted(self, engine: IntelligenceGraphEngine):
        """Skills are returned sorted alphabetically."""
        engine.graph.add_node("s2", type="skill", name="zeta")
        engine.graph.add_node("s1", type="skill", name="alpha")
        skills = engine.get_skills()
        assert skills[0]["name"] == "alpha"
        assert skills[1]["name"] == "zeta"

    def test_toggle_resource(self, engine: IntelligenceGraphEngine):
        """toggle_resource flips the enabled flag."""
        engine.graph.add_node("tool:x", type="tool", name="x", enabled=True)
        result = engine.toggle_resource("tool:x")
        assert result["enabled"] is False
        assert engine.graph.nodes["tool:x"]["enabled"] is False

        # Toggle back
        result = engine.toggle_resource("tool:x")
        assert result["enabled"] is True

    def test_toggle_resource_default_enabled(self, engine: IntelligenceGraphEngine):
        """Nodes without explicit enabled flag are treated as enabled."""
        engine.graph.add_node("tool:y", type="tool", name="y")
        result = engine.toggle_resource("tool:y")
        assert result["enabled"] is False

    def test_toggle_nonexistent_raises(self, engine: IntelligenceGraphEngine):
        """toggle_resource raises ValueError for missing resource."""
        with pytest.raises(ValueError, match="not found"):
            engine.toggle_resource("nonexistent")


# ─────────────────────────────────────────────────────────────────────
#  CONCEPT:AU-KG.query.object-graph-mapper — Workspace Reload
# ─────────────────────────────────────────────────────────────────────


class TestWorkspaceReload:
    """Tests for reload_from_workspace (CONCEPT:AU-KG.query.object-graph-mapper)."""

    def test_reload_returns_summary(self, engine: IntelligenceGraphEngine):
        """reload_from_workspace returns a change summary dict."""
        changes = engine.reload_from_workspace()
        assert "identity_changed" in changes
        assert "prompts_updated" in changes
        assert "tools_synced" in changes
        assert "cron_tasks_refreshed" in changes

    def test_reload_counts_existing_resources(self, engine: IntelligenceGraphEngine):
        """Reload summary reflects existing resources."""
        engine.graph.add_node("tool:a", resource_type="mcp_tool", name="a")
        engine.graph.add_node("tool:b", resource_type="mcp_tool", name="b")
        changes = engine.reload_from_workspace()
        assert changes["tools_synced"] == 2
