"""Tests for IntelligenceGraphEngine KG-first helpers.

CONCEPT:AU-KG.query.object-graph-mapper — Identity Management
CONCEPT:AU-KG.query.object-graph-mapper — Prompt Management
CONCEPT:AU-KG.query.object-graph-mapper — Granular Resource Queries
CONCEPT:AU-KG.query.object-graph-mapper — Workspace Reload
"""

from typing import Any

import pytest

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
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
def engine(isolate_graph_compute_engine):
    """An IntelligenceGraphEngine bound to its OWN uniquely-named test graph.

    Two prior shapes of this fixture both broke:

    1. A throwaway ``GraphComputeEngine(graph_name=<locally invented uuid>,
       ...)`` constructed and discarded before building the real engine became
       the process-owned singleton, bound to a graph name that did NOT match
       the ambient ``GraphSession``'s declared graph -- every
       ``engine.backend.execute()`` then failed with ``PermissionError: A
       graph-scoped view cannot retarget the verified GraphSession``
       (``graph_compute.py._send`` rejects any explicit ``graph`` that
       disagrees with ``session.graph``).
    2. Dropping that line entirely and just calling
       ``IntelligenceGraphEngine(db_path=":memory:")`` resolves, via
       ``create_backend`` -> bare ``EpistemicGraphBackend()`` ->
       ``resolve_routing_graph(None)``, onto the SHARED tenant-default graph
       (``tenant__<tenant>____commons__``) -- every test in this 26-test file
       then wrote to the SAME graph sequentially and collided:
       ``RuntimeError: durable graph registration failed: STALE_FENCE``.

    The fix (same shape as
    ``tests/unit/knowledge_graph/test_topological_analogy.py::base_graph``):
    retarget a per-test ``GraphSession`` at an explicit, uniquely-named graph
    derived from the autouse ``isolate_graph_compute_engine`` fixture's own
    test graph name, then construct the backend for THAT graph name inside
    that session -- explicit and ambient agree, so neither failure mode
    applies.
    """
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.session import GraphSession, use_session

    graph_name = f"{isolate_graph_compute_engine}_engine_helpers"
    session = GraphSession.from_ambient().with_graph(graph_name)
    with use_session(session):
        backend = EpistemicGraphBackend(graph_name=graph_name)
        yield IntelligenceGraphEngine(backend=backend)


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

        # Check edge exists. kg_adapter.update_prompt's own self.graph.add_edge
        # call writes the edge property under the canonical ``relationship``
        # key with the raw (lowercase) RegistryEdgeType value, but when a
        # backend is configured (as here) update_prompt ALSO dispatches
        # through self.link_nodes -- which upper-cases the relationship
        # value before its own write (the converged relationship-type
        # convention every link_nodes call site uses, D-GS4-1/D-GS7-1) --
        # so the value actually observable on the compute-leg edge is
        # upper-cased, not the enum's lowercase .value.
        has_supersedes = any(
            edata.get("relationship") == RegistryEdgeType.SUPERSEDES.value.upper()
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
            node_type="skill",
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
            node_type="callable_resource",
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
        engine.graph.add_node("s2", node_type="skill", name="zeta")
        engine.graph.add_node("s1", node_type="skill", name="alpha")
        skills = engine.get_skills()
        assert skills[0]["name"] == "alpha"
        assert skills[1]["name"] == "zeta"

    def test_toggle_resource(self, engine: IntelligenceGraphEngine):
        """toggle_resource flips the enabled flag."""
        engine.graph.add_node("tool:x", node_type="tool", name="x", enabled=True)
        result = engine.toggle_resource("tool:x")
        assert result["enabled"] is False
        assert engine.graph.nodes["tool:x"]["enabled"] is False

        # Toggle back
        result = engine.toggle_resource("tool:x")
        assert result["enabled"] is True

    def test_toggle_resource_default_enabled(self, engine: IntelligenceGraphEngine):
        """Nodes without explicit enabled flag are treated as enabled."""
        engine.graph.add_node("tool:y", node_type="tool", name="y")
        result = engine.toggle_resource("tool:y")
        assert result["enabled"] is False

    def test_toggle_nonexistent_raises(self, engine: IntelligenceGraphEngine):
        """toggle_resource raises ValueError for missing resource."""
        with pytest.raises(ValueError, match="not found"):
            engine.toggle_resource("nonexistent")


# ─────────────────────────────────────────────────────────────────────
#  CONCEPT:AU-KG.query.object-graph-mapper — MCP Server Catalog
# ─────────────────────────────────────────────────────────────────────


class TestMCPServerCatalog:
    """Tests for get_registered_mcp_servers (CONCEPT:AU-KG.query.object-graph-mapper).

    Root cause this covers: agent-webui's Prompts Registry/Skills views
    already read Prompt/Skill nodes straight from the KG, but the MCP
    Servers surface fell back to a static ``mcp_config.json`` file that was
    never populated with the discovered fleet catalog (D-W5WR-4/D-WD-7
    follow-up) -- it always returned zero servers regardless of how many
    real ``:MCPServer`` nodes the KG held. This proves the KG-authority
    path a registration/catalog-audit consumer can call.

    BUG-042 note: this method used to be named ``get_all_mcp_servers`` --
    an ambiguous name that invited exactly the two-sources-of-truth defect
    BUG-042 records, once a second, unrelated worker independently sourced
    the WebUI's MCP-servers-to-dispatch panel from
    ``agent_utilities.mcp.shared_multiplexer`` (live per-session
    dispatchable truth). Renamed to ``get_registered_mcp_servers`` to name
    exactly what it answers -- "what is registered in the KG" -- which is a
    real, distinct question from "what is dispatchable right now", never a
    competing answer to the same question. See the method's docstring for
    the full naming contract. ``test_get_registered_mcp_servers_never_claims_live_dispatchability``
    below is the anti-divergence guard: it fails if this method's output
    schema ever grows a field that would let it masquerade as the
    dispatchable-truth source.
    """

    def test_get_registered_mcp_servers_empty(self, engine: IntelligenceGraphEngine):
        """Returns empty list when no MCPServer nodes are in the graph."""
        assert engine.get_registered_mcp_servers() == []

    def test_get_registered_mcp_servers_from_backend(
        self, engine: IntelligenceGraphEngine
    ):
        """Finds MCPServer nodes written through the real backend."""
        engine._upsert_node(
            "MCPServer",
            "mcp_server_servicenow-api",
            {
                "name": "servicenow-api",
                "synonyms": ["servicenow"],
                "disabled": False,
            },
        )
        servers = engine.get_registered_mcp_servers()
        assert len(servers) == 1
        assert servers[0]["name"] == "servicenow-api"
        assert servers[0]["disabled"] is False
        assert servers[0]["type"] == "mcp_server"
        assert servers[0]["tool_count"] == 0

    def test_get_registered_mcp_servers_counts_served_tools(
        self, engine: IntelligenceGraphEngine
    ):
        """A server's tool_count reflects its outgoing SERVES edges."""
        engine._upsert_node(
            "MCPServer", "mcp_server_gitlab-mcp", {"name": "gitlab-mcp", "disabled": False}
        )
        engine._upsert_node(
            "Tool",
            "tool_gitlab-mcp_gitlab_issues",
            {"name": "gitlab_issues"},
        )
        engine.link_nodes(
            "mcp_server_gitlab-mcp", "tool_gitlab-mcp_gitlab_issues", "SERVES"
        )
        servers = engine.get_registered_mcp_servers()
        assert len(servers) == 1
        assert servers[0]["tool_count"] == 1

    def test_get_registered_mcp_servers_sorted(self, engine: IntelligenceGraphEngine):
        """Servers are returned sorted alphabetically by name."""
        engine._upsert_node("MCPServer", "mcp_server_zeta", {"name": "zeta"})
        engine._upsert_node("MCPServer", "mcp_server_alpha", {"name": "alpha"})
        servers = engine.get_registered_mcp_servers()
        assert [s["name"] for s in servers] == ["alpha", "zeta"]

    # ── BUG-042 anti-divergence guard ──────────────────────────────────
    #
    # The multiplexer (``agent_utilities.mcp.shared_multiplexer.list_catalog``)
    # and this method are BOTH allowed to exist because they answer
    # genuinely different questions (registered vs. dispatchable). What
    # must never happen again is one of them silently growing a field that
    # lets a caller treat it as an answer to the OTHER question -- that is
    # exactly how the original defect could recur even after the WebUI was
    # correctly rewired to the multiplexer (GOC-60-W03/W04a): a future edit
    # to this method could reintroduce an ``available``/``dispatchable``
    # field and a future caller could reasonably read that as live status.
    # ``_REGISTRATION_ONLY_SCHEMA`` positively enumerates the registration
    # truth's own contract fields; ``_LIVE_DISPATCH_ONLY_FIELDS`` are the
    # multiplexer's own vocabulary (``mcp/shared_multiplexer.py``'s
    # ``list_catalog`` row shape: ``server``/``tool_count``/``enabled_count``/
    # ``process_running``/``probed``/``available``) that must never appear
    # on a registration-truth row.

    _LIVE_DISPATCH_ONLY_FIELDS = frozenset(
        {
            "available",
            "dispatchable",
            "dispatchable_tools",
            "mounted",
            "process_running",
            "probed",
            "enabled_count",
            "pending",
            "stale",
        }
    )

    @staticmethod
    def _assert_never_claims_live_dispatchability(
        rows: list[dict[str, Any]],
    ) -> None:
        for row in rows:
            leaked = set(row) & TestMCPServerCatalog._LIVE_DISPATCH_ONLY_FIELDS
            assert not leaked, (
                "get_registered_mcp_servers row leaked live-dispatch field(s) "
                f"{sorted(leaked)}; that truth belongs ONLY to "
                "agent_utilities.mcp.shared_multiplexer.list_catalog (BUG-042) "
                f"-- got row {row!r}"
            )

    def test_get_registered_mcp_servers_never_claims_live_dispatchability(
        self, engine: IntelligenceGraphEngine
    ):
        """BUG-042 regression guard, positive case: a real registered server
        never grows a live-dispatch field."""
        engine._upsert_node(
            "MCPServer",
            "mcp_server_declared-only",
            {"name": "declared-only-server", "synonyms": [], "disabled": False},
        )
        servers = engine.get_registered_mcp_servers()
        assert len(servers) == 1
        self._assert_never_claims_live_dispatchability(servers)

    def test_anti_divergence_guard_catches_a_conflated_row(self):
        """Proves the guard above is not vacuous: known-bad input --
        a row shaped like the BUG-042 regression (a registration-truth row
        that ALSO claims live dispatchability, the exact ambiguity the
        rename + docstring contract forbid) -- must fail the check."""
        conflated_row = {
            "id": "mcp_server_x",
            "name": "x",
            "type": "mcp_server",
            "available": True,  # <- the live-dispatch claim that must never appear here
        }
        with pytest.raises(AssertionError, match="leaked live-dispatch field"):
            self._assert_never_claims_live_dispatchability([conflated_row])

    def test_get_all_mcp_servers_ambiguous_name_is_not_reintroduced(
        self, engine: IntelligenceGraphEngine
    ):
        """BUG-042: the ambiguous pre-fix name must never come back, not even
        as a back-compat alias -- No Legacy applies, and a second name for
        the same method would re-open the "which one is authoritative"
        confusion the rename exists to close."""
        assert not hasattr(type(engine), "get_all_mcp_servers")


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
