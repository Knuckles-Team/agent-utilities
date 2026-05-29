#!/usr/bin/python
"""Tests for CONCEPT:KG-2.3 — Graph Integrity Validator."""

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
import pytest

from agent_utilities.knowledge_graph.security.graph_validator import (
    EDGE_TYPE_ALIASES,
    NODE_TYPE_ALIASES,
    GraphValidator,
)


@pytest.fixture
def mock_engine():
    """Create a minimal mock engine with a graph compute engine."""

    class MockEngine:
        def __init__(self):
            self.graph = GraphComputeEngine(backend_type="rust")
            if self.graph._client:
                self.graph._client.clear()

    return MockEngine()


@pytest.fixture
def populated_engine(mock_engine):
    """Engine with a small healthy graph."""
    g = mock_engine.graph
    g.add_node(
        "agent:genius",
        type="agent",
        name="genius-agent",
        description="Primary orchestrator",
        importance_score=0.9,
    )
    g.add_node(
        "tool:search",
        type="tool",
        name="search-tool",
        description="Web search tool",
        importance_score=0.7,
    )
    g.add_node(
        "skill:browser",
        type="skill",
        name="browser-skill",
        description="Browser automation",
        importance_score=0.5,
    )
    g.add_edge("agent:genius", "tool:search", type="provides", weight=1.0)
    g.add_edge("agent:genius", "skill:browser", type="has_skill", weight=0.8)
    g.add_edge("tool:search", "skill:browser", type="related_to", weight=0.5)
    return mock_engine


class TestTier1AutoFix:
    """Tier 1: Auto-fix tests (silent corrections)."""

    def test_normalizes_node_type_aliases(self, mock_engine):
        """LLM type aliases should be normalized to canonical types."""
        mock_engine.graph.add_node("n1", type="func", name="my_func")
        mock_engine.graph.add_node("n2", type="service", name="my_service")

        validator = GraphValidator(mock_engine)
        report = validator.validate()

        assert mock_engine.graph.nodes["n1"]["type"] == "symbol"
        assert mock_engine.graph.nodes["n2"]["type"] == "agent"
        assert len([f for f in report.tier1_fixes if f.category == "type_alias"]) == 2

    def test_clamps_importance_score(self, mock_engine):
        """Scores outside [0, 1] should be clamped."""
        mock_engine.graph.add_node("n1", type="agent", name="a", importance_score=1.5)
        mock_engine.graph.add_node("n2", type="tool", name="b", importance_score=-0.3)

        validator = GraphValidator(mock_engine)
        report = validator.validate()

        assert mock_engine.graph.nodes["n1"]["importance_score"] == 1.0
        assert mock_engine.graph.nodes["n2"]["importance_score"] == 0.0
        assert len([f for f in report.tier1_fixes if f.category == "score_clamp"]) == 2

    def test_sets_missing_name(self, mock_engine):
        """Nodes without names should get their ID as name."""
        mock_engine.graph.add_node("node-xyz", type="file", name="")

        validator = GraphValidator(mock_engine)
        report = validator.validate()

        assert mock_engine.graph.nodes["node-xyz"]["name"] == "node-xyz"
        assert any(f.category == "missing_name" for f in report.tier1_fixes)

    def test_normalizes_edge_aliases(self, mock_engine):
        """LLM edge type aliases should be normalized."""
        mock_engine.graph.add_node("a", type="agent", name="a")
        mock_engine.graph.add_node("b", type="agent", name="b")
        mock_engine.graph.add_edge("a", "b", type="extends")

        validator = GraphValidator(mock_engine)
        validator.validate()

        edge_data = mock_engine.graph.edges["a", "b", 0]
        assert edge_data["type"] == "inherits_from"

    def test_clamps_edge_weight(self, mock_engine):
        """Edge weights outside [0, 10] should be clamped."""
        mock_engine.graph.add_node("a", type="agent", name="a")
        mock_engine.graph.add_node("b", type="tool", name="b")
        mock_engine.graph.add_edge("a", "b", type="provides", weight=15.0)

        validator = GraphValidator(mock_engine)
        validator.validate()

        assert mock_engine.graph.edges["a", "b", 0]["weight"] == 10.0


class TestTier2Integrity:
    """Tier 2: Referential integrity tests."""

    def test_detects_missing_node_type(self, mock_engine):
        """Nodes without type should be flagged."""
        mock_engine.graph.add_node("orphan", name="no-type")

        validator = GraphValidator(mock_engine)
        report = validator.validate()

        assert any(v.category == "missing_type" for v in report.tier2_violations)

    def test_detects_untyped_edges(self, mock_engine):
        """Edges without type should be flagged."""
        mock_engine.graph.add_node("a", type="agent", name="a")
        mock_engine.graph.add_node("b", type="tool", name="b")
        mock_engine.graph.add_edge("a", "b")  # No type

        validator = GraphValidator(mock_engine)
        report = validator.validate()

        assert any(v.category == "untyped_edge" for v in report.tier2_violations)


class TestTier3Quality:
    """Tier 3: Quality check tests."""

    def test_detects_orphan_nodes(self, mock_engine):
        """Nodes with no edges should be flagged as orphans."""
        mock_engine.graph.add_node(
            "lonely", type="file", name="lonely.py", description="A lonely file"
        )

        validator = GraphValidator(mock_engine)
        report = validator.validate()

        assert any(w.category == "orphan_node" for w in report.tier3_warnings)

    def test_detects_self_referencing_edges(self, mock_engine):
        """Self-referencing edges should be flagged."""
        mock_engine.graph.add_node("loop", type="agent", name="loop")
        mock_engine.graph.add_edge("loop", "loop", type="depends_on")

        validator = GraphValidator(mock_engine)
        report = validator.validate()

        assert any(w.category == "self_reference" for w in report.tier3_warnings)

    def test_detects_generic_descriptions(self, mock_engine):
        """Generic placeholder descriptions should be flagged."""
        mock_engine.graph.add_node(
            "generic", type="tool", name="tool", description="TODO"
        )
        mock_engine.graph.add_node(
            "generic2", type="tool", name="tool2", description="placeholder"
        )

        validator = GraphValidator(mock_engine)
        report = validator.validate()

        generic_warnings = [
            w for w in report.tier3_warnings if w.category == "generic_description"
        ]
        assert len(generic_warnings) >= 2


class TestTier4Fatal:
    """Tier 4: Fatal check tests."""

    def test_detects_empty_graph(self, mock_engine):
        """Empty graph should trigger a fatal error."""
        validator = GraphValidator(mock_engine)
        report = validator.validate()

        assert any(f.category == "empty_graph" for f in report.tier4_fatal)
        assert not report.is_healthy

    def test_healthy_graph_passes(self, populated_engine):
        """A well-formed graph should pass all checks."""
        validator = GraphValidator(populated_engine)
        report = validator.validate()

        assert report.is_healthy
        assert len(report.tier4_fatal) == 0


class TestValidationReport:
    """Test report properties and summary."""

    def test_report_summary(self, populated_engine):
        validator = GraphValidator(populated_engine)
        report = validator.validate()

        summary = report.summary()
        assert "Graph Validation Report" in summary
        assert "HEALTHY" in summary

    def test_report_issue_count(self, mock_engine):
        """Empty graph should have at least one issue."""
        validator = GraphValidator(mock_engine)
        report = validator.validate()

        assert report.issue_count > 0

    def test_alias_maps_complete(self):
        """Verify alias maps have reasonable coverage."""
        assert len(NODE_TYPE_ALIASES) > 10
        assert len(EDGE_TYPE_ALIASES) > 10
        # All canonical values should be lowercase
        for v in NODE_TYPE_ALIASES.values():
            assert v == v.lower()
        for v in EDGE_TYPE_ALIASES.values():
            assert v == v.lower()
