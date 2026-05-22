"""Integration tests for KG-Driven Graph Factory (CONCEPT:ORCH-1.4).

Tests the full lifecycle from KG template ingestion → graph materialization
→ step execution → provenance tracking.

All tests use in-memory NetworkX graphs (no LadybugDB) for isolation.
Set AGENT_UTILITIES_TESTING=true to avoid engine startup.
"""

from __future__ import annotations

import os

import networkx as nx

os.environ["AGENT_UTILITIES_TESTING"] = "true"

from agent_utilities.graph.kg_graph_factory import (
    KGGraphResult,
    KGMaterializedStep,
    _resolve_prompt_from_kg,
    _resolve_tools_from_kg,
    _topological_sort,
    build_pydantic_graph_from_kg,
)
from agent_utilities.models.knowledge_graph import (
    AgentTemplateNode,
    RegistryNodeType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockEngine:
    """Minimal mock of IntelligenceGraphEngine for testing.

    Uses an in-memory NetworkX graph and a mock backend.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.backend = MockBackend(self.graph)

    def search(self, query: str, top_k: int = 10, node_types=None):
        """Hybrid search stub — returns nodes from the NX graph."""
        results = []
        for nid, data in self.graph.nodes(data=True):
            if node_types and data.get("type") not in [
                nt.lower().replace(" ", "_") for nt in node_types
            ]:
                continue
            results.append({"id": nid, **data})
        return results[:top_k]

    def find_agent_for_tool(self, word):
        return set()

    def search_hybrid(self, query, top_k=5):
        return []

    def find_relevant_policies(self, query):
        return []

    def find_relevant_processes(self, query):
        return []


class MockBackend:
    """Mock Cypher backend that queries the NetworkX graph."""

    def __init__(self, graph: nx.DiGraph):
        self._graph = graph

    def execute(self, query: str, params: dict):
        """Simplified Cypher query execution using NX graph."""
        results = []

        # Handle AgentTemplate queries
        if "AgentTemplate" in query and "RETURN" in query:
            for nid, data in self._graph.nodes(data=True):
                if data.get("type") == "agent_template":
                    if "WHERE" in query and "IN $ids" in query:
                        if nid not in params.get("ids", []):
                            continue
                    results.append(
                        {
                            "id": nid,
                            "name": data.get("name", ""),
                            "role": data.get("role", ""),
                            "system_prompt_id": data.get("system_prompt_id", ""),
                            "toolset_ids": data.get("toolset_ids", []),
                            "model_preference": data.get("model_preference", ""),
                            "execution_tier": data.get("execution_tier", "standard"),
                            "step_order": data.get("step_order", 0),
                            "is_parallel": data.get("is_parallel", False),
                            "max_retries": data.get("max_retries", 2),
                            "description": data.get("description", ""),
                            "source": nid,
                            "target": "",
                        }
                    )

        # Handle Prompt queries
        elif "Prompt" in query and "$pid" in query:
            pid = params.get("pid", "")
            if self._graph.has_node(pid):
                data = self._graph.nodes[pid]
                results.append(
                    {
                        "prompt": data.get("system_prompt", data.get("content", "")),
                        "id": pid,
                    }
                )

        # Handle DEPENDS_ON queries
        elif "DEPENDS_ON" in query:
            ids = params.get("ids", [])
            for src, tgt, edata in self._graph.edges(data=True):
                if edata.get("type") == "depends_on" and src in ids and tgt in ids:
                    results.append({"source": src, "target": tgt})

        # Handle Tool queries
        elif "Tool" in query and "$ids" in query:
            ids = params.get("ids", [])
            for nid, data in self._graph.nodes(data=True):
                if data.get("type") == "tool" and nid in ids:
                    results.append({"name": data.get("name", ""), "server": ""})

        return results


def _build_mock_engine_with_templates() -> MockEngine:
    """Build a mock engine with 3 AgentTemplate nodes in sequential order."""
    engine = MockEngine()
    g = engine.graph

    # Create prompt nodes
    g.add_node(
        "prompt:researcher",
        type="prompt",
        system_prompt="You are a research specialist.",
    )
    g.add_node(
        "prompt:coder",
        type="prompt",
        system_prompt="You are an expert Python programmer.",
    )

    # Create tool nodes
    g.add_node("tool:web_search", type="tool", name="web_search")
    g.add_node("tool:code_exec", type="tool", name="code_executor")

    # Create AgentTemplate nodes
    g.add_node(
        "at:researcher",
        type="agent_template",
        name="Researcher",
        role="researcher",
        system_prompt_id="prompt:researcher",
        toolset_ids=["tool:web_search"],
        model_preference="gpt-4o-mini",
        execution_tier="lite",
        step_order=0,
        is_parallel=False,
        max_retries=2,
        description="Research specialist for web queries",
    )

    g.add_node(
        "at:coder",
        type="agent_template",
        name="Coder",
        role="coder",
        system_prompt_id="prompt:coder",
        toolset_ids=["tool:code_exec"],
        model_preference="gpt-4o",
        execution_tier="standard",
        step_order=1,
        is_parallel=False,
        max_retries=2,
        description="Python programming specialist",
    )

    g.add_node(
        "at:reviewer",
        type="agent_template",
        name="Reviewer",
        role="reviewer",
        system_prompt_id="",
        toolset_ids=[],
        model_preference="",
        execution_tier="standard",
        step_order=2,
        is_parallel=False,
        max_retries=1,
        description="Code review specialist",
    )

    # Create DEPENDS_ON edges (researcher → coder → reviewer)
    g.add_edge("at:researcher", "at:coder", type="depends_on", weight=1.0)
    g.add_edge("at:coder", "at:reviewer", type="depends_on", weight=1.0)

    return engine


def _build_mock_engine_parallel() -> MockEngine:
    """Build a mock engine with 2 parallel AgentTemplate nodes + 1 joiner."""
    engine = MockEngine()
    g = engine.graph

    # Create parallel templates
    g.add_node(
        "at:web_researcher",
        type="agent_template",
        name="Web Researcher",
        role="web_researcher",
        system_prompt_id="",
        toolset_ids=[],
        model_preference="",
        step_order=0,
        is_parallel=True,
        description="Web research specialist",
    )
    g.add_node(
        "at:doc_researcher",
        type="agent_template",
        name="Doc Researcher",
        role="doc_researcher",
        system_prompt_id="",
        toolset_ids=[],
        model_preference="",
        step_order=0,
        is_parallel=True,
        description="Documentation research specialist",
    )
    g.add_node(
        "at:synthesizer",
        type="agent_template",
        name="Synthesizer",
        role="synthesizer",
        system_prompt_id="",
        toolset_ids=[],
        model_preference="",
        step_order=1,
        is_parallel=False,
        description="Synthesizes results from parallel research",
    )

    # DEPENDS_ON: both researchers → synthesizer
    g.add_edge("at:web_researcher", "at:synthesizer", type="depends_on", weight=1.0)
    g.add_edge("at:doc_researcher", "at:synthesizer", type="depends_on", weight=1.0)

    return engine


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestBuildSimpleSequentialGraph:
    """Test building a sequential graph from KG templates."""

    def test_sequential_graph_materialization(self):
        """Test that 3 sequential templates produce a valid KGGraphResult."""
        engine = _build_mock_engine_with_templates()
        result = build_pydantic_graph_from_kg(
            query="Build a REST API",
            engine=engine,  # type: ignore
            top_k=10,
        )

        assert isinstance(result, KGGraphResult)
        assert result.graph is not None
        assert result.entry_node_id != ""
        assert len(result.specialist_configs) == 3
        assert "researcher" in result.specialist_configs
        assert "coder" in result.specialist_configs
        assert "reviewer" in result.specialist_configs

    def test_sequential_ordering(self):
        """Test that templates are ordered by DEPENDS_ON topology."""
        engine = _build_mock_engine_with_templates()
        result = build_pydantic_graph_from_kg(
            query="Build an API",
            engine=engine,  # type: ignore
        )

        # Entry should be the researcher (step_order=0, no dependencies)
        assert result.entry_node_id == "at:researcher"

    def test_provenance_tracking(self):
        """Test that KG provenance is captured for each step."""
        engine = _build_mock_engine_with_templates()
        result = build_pydantic_graph_from_kg(
            query="Implement feature",
            engine=engine,  # type: ignore
        )

        assert len(result.kg_provenance) == 3
        for prov in result.kg_provenance:
            assert "type" in prov
            assert prov["type"] == "agent_template"
            assert "node_id" in prov
            assert "role" in prov


class TestBuildParallelGraph:
    """Test building a graph with parallel fan-out."""

    def test_parallel_graph_materialization(self):
        """Test that parallel templates produce correct specialist configs."""
        engine = _build_mock_engine_parallel()
        result = build_pydantic_graph_from_kg(
            query="Research a topic comprehensively",
            engine=engine,  # type: ignore
        )

        assert isinstance(result, KGGraphResult)
        assert len(result.specialist_configs) == 3
        assert "web_researcher" in result.specialist_configs
        assert "doc_researcher" in result.specialist_configs
        assert "synthesizer" in result.specialist_configs


class TestPromptResolutionFromKG:
    """Test prompt resolution from KG nodes."""

    def test_prompt_resolved_via_kg(self):
        """Test that system prompts are resolved from KG Prompt nodes."""
        engine = _build_mock_engine_with_templates()
        result = build_pydantic_graph_from_kg(
            query="Code review",
            engine=engine,  # type: ignore
        )

        # Check researcher's resolved prompt
        researcher_cfg = result.specialist_configs.get("researcher", {})
        assert "research specialist" in researcher_cfg.get("system_prompt", "").lower()

    def test_prompt_resolution_helper(self):
        """Test the _resolve_prompt_from_kg helper directly."""
        engine = _build_mock_engine_with_templates()
        prompt, pid = _resolve_prompt_from_kg(engine, "prompt:researcher")  # type: ignore
        assert prompt == "You are a research specialist."
        assert pid == "prompt:researcher"

    def test_prompt_missing_returns_fallback(self):
        """Test graceful fallback when prompt doesn't exist in KG.

        The load_specialized_prompts fallback produces a generic helper
        prompt, which _resolve_prompt_from_kg should accept.
        """
        engine = MockEngine()
        prompt, pid = _resolve_prompt_from_kg(engine, "nonexistent")  # type: ignore
        # Fallback generates a generic prompt string via load_specialized_prompts
        assert "nonexistent" in prompt or prompt == ""


class TestToolResolutionFromKG:
    """Test tool resolution from KG nodes."""

    def test_tools_resolved_via_kg(self):
        """Test that tools are resolved from KG Tool nodes."""
        engine = _build_mock_engine_with_templates()
        result = build_pydantic_graph_from_kg(
            query="Search and code",
            engine=engine,  # type: ignore
        )

        researcher_cfg = result.specialist_configs.get("researcher", {})
        assert "web_search" in researcher_cfg.get("tools", [])

    def test_tool_resolution_helper(self):
        """Test the _resolve_tools_from_kg helper directly."""
        engine = _build_mock_engine_with_templates()
        tools = _resolve_tools_from_kg(engine, ["tool:web_search", "tool:code_exec"])  # type: ignore
        assert "web_search" in tools
        assert "code_executor" in tools

    def test_empty_toolset_returns_empty(self):
        """Test empty toolset returns empty list."""
        engine = MockEngine()
        tools = _resolve_tools_from_kg(engine, [])  # type: ignore
        assert tools == []


class TestFallbackWithoutTemplates:
    """Test fallback behavior when no AgentTemplate nodes exist."""

    def test_empty_engine_returns_result(self):
        """Test that an empty engine still produces a result (via fallback)."""
        engine = MockEngine()
        result = build_pydantic_graph_from_kg(
            query="Hello world",
            engine=engine,  # type: ignore
        )

        # Should return a result, possibly empty or from team composer fallback
        assert isinstance(result, KGGraphResult)
        assert result.graph is not None

    def test_none_engine_returns_result(self):
        """Test that None engine produces a result."""
        result = build_pydantic_graph_from_kg(
            query="Test query",
            engine=None,
        )

        assert isinstance(result, KGGraphResult)


class TestKGProvenanceTracking:
    """Test KG provenance tracking through the full pipeline."""

    def test_provenance_includes_all_nodes(self):
        """Test that provenance records exist for every materialized step."""
        engine = _build_mock_engine_with_templates()
        result = build_pydantic_graph_from_kg(
            query="Full pipeline",
            engine=engine,  # type: ignore
        )

        node_ids = {p["node_id"] for p in result.kg_provenance}
        assert "at:researcher" in node_ids
        assert "at:coder" in node_ids
        assert "at:reviewer" in node_ids

    def test_provenance_includes_tool_counts(self):
        """Test that provenance records include tool counts."""
        engine = _build_mock_engine_with_templates()
        result = build_pydantic_graph_from_kg(
            query="Pipeline test",
            engine=engine,  # type: ignore
        )

        researcher_prov = next(
            (p for p in result.kg_provenance if p["role"] == "researcher"), None
        )
        assert researcher_prov is not None
        assert researcher_prov["tool_count"] == 1


class TestObservabilityEventsEmitted:
    """Test that KG_BRIDGE trace events are properly emitted."""

    def test_phase_map_includes_kg_bridge(self):
        """Test that all KG_BRIDGE events are in the phase map."""
        from agent_utilities.graph.config_helpers import _PHASE_MAP

        kg_events = [
            "kg_query_start",
            "kg_query_complete",
            "kg_template_resolved",
            "kg_prompt_injected",
            "kg_topology_materialized",
        ]

        for event in kg_events:
            assert event in _PHASE_MAP, f"Missing KG_BRIDGE event: {event}"
            assert _PHASE_MAP[event] == "KG_BRIDGE"


class TestTopologicalSort:
    """Test the topological sort helper."""

    def test_sorts_by_step_order_without_edges(self):
        """Test that templates are sorted by step_order when no edges."""
        templates = [
            {"id": "c", "step_order": 2},
            {"id": "a", "step_order": 0},
            {"id": "b", "step_order": 1},
        ]
        sorted_t = _topological_sort(templates, [])
        assert [t["id"] for t in sorted_t] == ["a", "b", "c"]

    def test_sorts_by_dependency_edges(self):
        """Test that templates follow DEPENDS_ON order."""
        templates = [
            {"id": "c", "step_order": 0},
            {"id": "a", "step_order": 0},
            {"id": "b", "step_order": 0},
        ]
        edges = [("a", "b"), ("b", "c")]
        sorted_t = _topological_sort(templates, edges)
        assert [t["id"] for t in sorted_t] == ["a", "b", "c"]

    def test_handles_fan_out_topology(self):
        """Test sorting with fan-out (1 → N) structure."""
        templates = [
            {"id": "root", "step_order": 0},
            {"id": "branch_a", "step_order": 1},
            {"id": "branch_b", "step_order": 1},
        ]
        edges = [("root", "branch_a"), ("root", "branch_b")]
        sorted_t = _topological_sort(templates, edges)
        assert sorted_t[0]["id"] == "root"
        # Both branches should come after root
        remaining = {t["id"] for t in sorted_t[1:]}
        assert remaining == {"branch_a", "branch_b"}


class TestAgentTemplateNodeModel:
    """Test the AgentTemplateNode Pydantic model."""

    def test_default_values(self):
        """Test that AgentTemplateNode has correct defaults."""
        node = AgentTemplateNode(
            id="test-1",
            name="Test Template",
        )
        assert node.type == RegistryNodeType.AGENT_TEMPLATE
        assert node.role == ""
        assert node.execution_tier == "standard"
        assert node.step_order == 0
        assert node.is_parallel is False
        assert node.max_retries == 2

    def test_full_initialization(self):
        """Test full initialization with all fields."""
        node = AgentTemplateNode(
            id="at:python",
            name="Python Specialist",
            role="python_programmer",
            system_prompt_id="prompt:python",
            toolset_ids=["tool:code_exec", "tool:lint"],
            model_preference="gpt-4o",
            execution_tier="super",
            step_order=1,
            is_parallel=False,
            max_retries=3,
        )
        assert node.role == "python_programmer"
        assert len(node.toolset_ids) == 2
        assert node.execution_tier == "super"


class TestKGMaterializedStepModel:
    """Test the KGMaterializedStep Pydantic model."""

    def test_step_initialization(self):
        """Test basic step initialization."""
        step = KGMaterializedStep(
            step_id="s1",
            role="researcher",
            system_prompt="You research.",
            tool_names=["web_search"],
            is_terminal=False,
            next_step_ids=["s2"],
        )
        assert step.step_id == "s1"
        assert step.role == "researcher"
        assert step.tool_names == ["web_search"]
        assert not step.is_terminal

    def test_terminal_step(self):
        """Test terminal step configuration."""
        step = KGMaterializedStep(
            step_id="s3",
            role="finalizer",
            is_terminal=True,
        )
        assert step.is_terminal
        assert step.next_step_ids == []
