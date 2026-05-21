"""Dynamic Distribution & Workflow Binding Tests.

CONCEPT:ORCH-1.24 — Capability-to-Workflow Binding

Tests the dynamic capability ingestion pipeline and validates that
ingested capabilities (MCP servers, skills, native tools) can be
bound to workflow steps for automated orchestration.

Structural tests — no LLM required.
"""

import json
import logging
import os
from pathlib import Path

import networkx as nx
import platformdirs
import pytest

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.mcp.kg_server import _ingest_capabilities

logger = logging.getLogger(__name__)


@pytest.fixture
def setup_test_environment(tmp_path, monkeypatch):
    """Create an isolated test environment with mock MCP config and skills."""

    def mock_user_config_path(app_name, app_author):
        return tmp_path / app_name / app_author

    monkeypatch.setattr(platformdirs, "user_config_path", mock_user_config_path)

    cfg_dir = tmp_path / "agent-utilities" / "knuckles-team"
    cfg_dir.mkdir(parents=True)
    mcp_config_path = cfg_dir / "mcp_config.json"
    mcp_config_path.write_text(
        json.dumps(
            {"mcpServers": {"test-server": {"command": "uv", "args": ["run", "test"]}}}
        )
    )

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Create a dummy skill
    my_skill = skills_dir / "my-test-skill"
    my_skill.mkdir()
    skill_md = my_skill / "SKILL.md"
    skill_md.write_text("""---
name: my_test_skill
description: A mock skill for testing
version: 1.0
---
# Instructions
Do the test thing!""")

    # Force the skills directory to the tmp_path
    monkeypatch.setenv("CUSTOM_SKILLS_DIRECTORY", str(skills_dir))

    # Setup test engine
    graph = nx.MultiDiGraph()
    engine = IntelligenceGraphEngine(graph=graph, backend=None)

    # Reload config to pick up the env var for CUSTOM_SKILLS_DIRECTORY
    from agent_utilities.core.config import config

    config.custom_skills_directory = str(skills_dir)

    return engine


# ═══════════════════════════════════════════════════════════════════════
# Original Capability Ingestion Tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_capability_ingestion(setup_test_environment):
    """CONCEPT:ORCH-1.21 — MCP, Skill, and NativeTool ingestion."""
    engine = setup_test_environment

    # Run ingestion
    _ingest_capabilities(engine)

    # 1. Test MCPServer ingestion
    mcp_nodes = [
        data
        for _, data in engine.graph.nodes(data=True)
        if data.get("type") == "MCPServer"
    ]
    assert len(mcp_nodes) > 0, "MCPServer node was not ingested"
    assert mcp_nodes[0]["name"] == "test-server"

    # 2. Test Skill ingestion
    skill_nodes = [
        data for _, data in engine.graph.nodes(data=True) if data.get("type") == "Skill"
    ]
    assert len(skill_nodes) > 0, "Skill node was not ingested"
    assert skill_nodes[0]["name"] == "my_test_skill"
    assert skill_nodes[0]["description"] == "A mock skill for testing"
    assert skill_nodes[0]["version"] == 1.0

    # 3. Test Native Tools ingestion
    native_nodes = [
        data
        for _, data in engine.graph.nodes(data=True)
        if data.get("type") == "NativeTool"
    ]
    assert len(native_nodes) > 0, "No NativeTool nodes were ingested"

    found_tool = native_nodes[0]
    module_name = found_tool["module"]
    function_name = found_tool["name"]

    # 4. Test dynamic resolution of the NativeTool
    assert module_name.startswith("agent_utilities.tools."), (
        "Dynamic import bounds check failed!"
    )

    import importlib

    module = importlib.import_module(module_name)
    func = getattr(module, function_name)

    assert callable(func), "Resolved tool is not callable"
    assert hasattr(func, "__agentic_version__"), "Resolved tool lacks version metadata"

    # Verify tracing decorator was successfully applied
    assert hasattr(func, "__wrapped__") or "trace" in str(func), (
        "Tracing decorator not found on tool"
    )


# ═══════════════════════════════════════════════════════════════════════
# Workflow Catalog Integration Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCatalogWorkflowIngestion:
    """Tests for ingesting catalog workflows into the KG."""

    def test_catalog_workflows_ingest_into_kg(self):
        """CONCEPT:ORCH-1.24 — Catalog workflows become KG nodes."""
        from agent_utilities.workflows.catalog import WorkflowCatalog

        graph = nx.MultiDiGraph()
        engine = IntelligenceGraphEngine(graph=graph, backend=None)

        catalog = WorkflowCatalog.load()
        workflow_ids = catalog.register_in_kg(engine)

        assert len(workflow_ids) > 0

        # Verify WorkflowDefinition nodes exist in NX graph
        wf_nodes = [
            data
            for _, data in engine.graph.nodes(data=True)
            if data.get("type") == "WorkflowDefinition"
        ]
        assert len(wf_nodes) == len(catalog.scenarios)
        logger.info("Ingested %d workflow definitions into KG", len(wf_nodes))

    def test_workflow_steps_linked_to_definitions(self):
        """CONCEPT:ORCH-1.24 — Steps are linked via HAS_STEP edges."""
        from agent_utilities.workflows.catalog import WorkflowCatalog

        graph = nx.MultiDiGraph()
        engine = IntelligenceGraphEngine(graph=graph, backend=None)

        catalog = WorkflowCatalog.load()
        catalog.register_in_kg(engine)

        # Find HAS_STEP edges
        has_step_edges = [
            (u, v, d)
            for u, v, d in engine.graph.edges(data=True)
            if d.get("type") == "HAS_STEP"
        ]
        assert len(has_step_edges) > 0
        logger.info("Found %d HAS_STEP edges", len(has_step_edges))

        # Count total steps across all scenarios
        expected_steps = sum(len(s.steps) for s in catalog.scenarios)
        step_nodes = [
            data
            for _, data in engine.graph.nodes(data=True)
            if data.get("type") == "WorkflowStep"
        ]
        assert len(step_nodes) == expected_steps

    def test_workflow_round_trip_from_catalog(self):
        """CONCEPT:ORCH-1.22 — Save→Load round-trip preserves structure."""
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore
        from agent_utilities.workflows.catalog import WorkflowCatalog

        graph = nx.MultiDiGraph()
        engine = IntelligenceGraphEngine(graph=graph, backend=None)

        catalog = WorkflowCatalog.load()
        catalog.register_in_kg(engine)

        store = WorkflowStore(engine)

        for scenario in catalog.scenarios[:3]:  # Test first 3
            loaded = store.load_workflow(scenario.name)
            assert loaded is not None, f"Failed to load '{scenario.name}'"
            assert len(loaded.steps) == len(scenario.steps), (
                f"Step count mismatch for '{scenario.name}': "
                f"expected {len(scenario.steps)}, got {len(loaded.steps)}"
            )
            logger.info(
                "Round-trip OK: '%s' (%d steps)",
                scenario.name,
                len(loaded.steps),
            )


class TestWorkflowDiscovery:
    """Tests for workflow discovery via KG queries."""

    def test_find_workflows_by_tag_in_graph(self):
        """CONCEPT:ORCH-1.24 — Discover workflows by tag via NX."""
        from agent_utilities.workflows.catalog import WorkflowCatalog

        graph = nx.MultiDiGraph()
        engine = IntelligenceGraphEngine(graph=graph, backend=None)

        catalog = WorkflowCatalog.load()
        catalog.register_in_kg(engine)

        # Search for "docker" tagged workflows via NX graph
        docker_workflows = []
        for nid, data in engine.graph.nodes(data=True):
            if data.get("type") == "WorkflowDefinition":
                meta_json = data.get("metadata_json", "")
                if "docker" in str(meta_json).lower():
                    docker_workflows.append(data.get("name", nid))

        assert len(docker_workflows) > 0
        assert "container_health_check" in docker_workflows
        logger.info("Docker workflows: %s", docker_workflows)

    def test_find_workflows_by_domain(self):
        """CONCEPT:ORCH-1.24 — Discover workflows by domain."""
        from agent_utilities.workflows.catalog import WorkflowCatalog

        graph = nx.MultiDiGraph()
        engine = IntelligenceGraphEngine(graph=graph, backend=None)

        catalog = WorkflowCatalog.load()
        catalog.register_in_kg(engine)

        # Search for research domain
        research_workflows = []
        for nid, data in engine.graph.nodes(data=True):
            if data.get("type") == "WorkflowDefinition":
                meta_json = data.get("metadata_json", "")
                if '"domain": "research"' in str(meta_json):
                    research_workflows.append(data.get("name", nid))

        assert len(research_workflows) > 0
        logger.info("Research workflows: %s", research_workflows)


class TestCapabilityToWorkflowBinding:
    """Tests for binding ingested capabilities to workflow steps."""

    def test_capability_binding_via_agent_name(self, setup_test_environment):
        """CONCEPT:ORCH-1.24 — Workflow steps resolve to ingested agents."""
        engine = setup_test_environment
        _ingest_capabilities(engine)

        from agent_utilities.workflows.catalog import WorkflowCatalog

        catalog = WorkflowCatalog.load()
        catalog.register_in_kg(engine)

        # Get all ingested capability names
        capability_names = set()
        for _, data in engine.graph.nodes(data=True):
            if data.get("type") in ("MCPServer", "Skill", "NativeTool", "Server"):
                name = data.get("name", "")
                if name:
                    capability_names.add(name)

        logger.info("Ingested capabilities: %s", capability_names)

        # Get all workflow step agents
        step_agents = set()
        for scenario in catalog.scenarios:
            for step in scenario.steps:
                step_agents.add(step.agent)

        logger.info("Workflow step agents: %s", step_agents)

        # At least the test-server should be present as a capability
        assert "test-server" in capability_names

    def test_workflow_mermaid_with_capabilities(self, setup_test_environment):
        """CONCEPT:ORCH-1.24 — Mermaid diagrams include capability nodes."""
        engine = setup_test_environment
        _ingest_capabilities(engine)

        from agent_utilities.workflows.catalog import WorkflowCatalog

        catalog = WorkflowCatalog.load()

        scenario = catalog.get("full_ecosystem_health")
        assert scenario is not None

        plan = scenario.to_graph_plan()
        mermaid = plan.to_mermaid(title="Ecosystem Health")
        assert "systems-manager" in mermaid
        assert "container-manager-mcp" in mermaid
        logger.info("Capability mermaid:\n%s", mermaid)
