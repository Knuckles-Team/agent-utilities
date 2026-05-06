"""Unit tests for the Constitution & Prompt Policy Ingestor (CONCEPT:KG-2.2).

Tests the policy ingestion pipeline from constitutions, prompt JSONs,
unified ingestion, policy querying, and prompt rendering.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.policy_ingestor import (
    PolicyIngestor,
    _determine_category,
    _extract_prompt_rules,
    _is_normative,
    parse_constitution_md,
)

# ── Sample constitution fixtures ─────────────────────────────────────

SAMPLE_CONSTITUTION = textwrap.dedent("""\
    # Project Constitution - test-project

    ## Vision & Mission
    **test-project** is a sample project for testing.

    ## Core Principles
    ### Guiding Principles
    - **Protocol-native**: Communicate via open standards.
    - **Type safety**: Use strict type checking everywhere.
    - **No framework lock-in**: Avoid opinionated frameworks.

    ### Normative Statements
    - You MUST use Pydantic for all data models.
    - You MUST NOT use global mutable state.
    - All code MUST be type-hinted.

    ## Governance
    - **Unified Registry**: The KG is the single source of truth.
    - Changes to this constitution MUST be approved via PR.

    ## Quality Gates
    ### Testing
    - All features MUST be implemented with corresponding Pytests.
    - Integration tests MUST pass before merge.
    ### Verification Loop
    - After any code change, `pre-commit run --all-files` MUST be executed.
    ### Prohibited Uses
    - Do NOT use for UI development.
    - Do NOT use for SaaS integrations.

    ## Tech Stack & Standards
    - **Language**: Python 3.11+
    - **Framework**: Pydantic AI
""")

SAMPLE_PROMPT_JSON = {
    "task": "python_programmer",
    "instructions": {
        "core_directive": "Write clean Python code.",
        "capabilities": {
            "testing": [
                "Test-driven development with pytest as default",
                "Fixtures for test data management and cleanup",
            ],
        },
        "responsibilities": [
            "**Idiomatic Development**: Follow PEP 8 and Pythonic patterns.",
        ],
    },
    "rules": [
        "Follow PEP 8 style guide",
        "Use pathlib for filesystem operations",
        "Prefer composition over inheritance",
    ],
}

SAMPLE_PROMPT_JSON_CATEGORIZED = {
    "task": "architect",
    "rules": {
        "quality_gates": [
            "All functions have type hints",
            "Docstrings on public APIs",
            "No mutable default arguments",
        ],
        "constraints": [
            "Do not use global state",
            "No framework lock-in",
        ],
    },
}


# ── Parser tests ─────────────────────────────────────────────────────


class TestConstitutionParser:
    """Tests for the constitution markdown parser."""

    def test_parse_extracts_project_name(self):
        parsed = parse_constitution_md(SAMPLE_CONSTITUTION)
        assert parsed["project_name"] == "test-project"

    def test_parse_extracts_policies(self):
        parsed = parse_constitution_md(SAMPLE_CONSTITUTION)
        assert len(parsed["policies"]) > 0

    def test_parse_categorizes_principles(self):
        parsed = parse_constitution_md(SAMPLE_CONSTITUTION)
        principles = [p for p in parsed["policies"] if p["category"] == "principle"]
        assert len(principles) >= 3  # Protocol-native, Type safety, No lock-in

    def test_parse_categorizes_normative(self):
        parsed = parse_constitution_md(SAMPLE_CONSTITUTION)
        normative = [p for p in parsed["policies"] if p["category"] == "normative"]
        assert len(normative) >= 3  # MUST use Pydantic, MUST NOT global, MUST type-hint

    def test_parse_categorizes_quality_gates(self):
        parsed = parse_constitution_md(SAMPLE_CONSTITUTION)
        qg = [p for p in parsed["policies"] if p["category"] == "quality_gate"]
        assert len(qg) >= 3  # Pytest, integration, pre-commit

    def test_parse_categorizes_constraints(self):
        parsed = parse_constitution_md(SAMPLE_CONSTITUTION)
        constraints = [p for p in parsed["policies"] if p["category"] == "constraint"]
        assert len(constraints) >= 2  # No UI, no SaaS

    def test_parse_detects_normative_flag(self):
        parsed = parse_constitution_md(SAMPLE_CONSTITUTION)
        normative_policies = [p for p in parsed["policies"] if p["is_normative"]]
        assert len(normative_policies) >= 5  # Multiple MUST statements

    def test_parse_categorizes_tech_stack(self):
        parsed = parse_constitution_md(SAMPLE_CONSTITUTION)
        tech = [p for p in parsed["policies"] if p["category"] == "tech_stack"]
        assert len(tech) >= 2  # Language, Framework


class TestHelperFunctions:
    """Tests for parser helper functions."""

    def test_determine_category_principles(self):
        assert (
            _determine_category("Core Principles", "Guiding Principles") == "principle"
        )

    def test_determine_category_normative(self):
        assert (
            _determine_category("Core Principles", "Normative Statements")
            == "normative"
        )

    def test_determine_category_quality_gates(self):
        assert _determine_category("Quality Gates", "Testing") == "quality_gate"

    def test_determine_category_governance(self):
        assert _determine_category("Governance", "") == "governance"

    def test_is_normative_detects_must(self):
        assert _is_normative("You MUST use Pydantic")

    def test_is_normative_detects_must_not(self):
        assert _is_normative("You MUST NOT use global state")

    def test_is_normative_rejects_plain(self):
        assert not _is_normative("Prefer composition over inheritance")


class TestPromptRuleExtraction:
    """Tests for extracting rules from prompt JSON files."""

    def test_extracts_rules_simple_list(self):
        """Only reads from the explicit 'rules' key — not capabilities etc."""
        rules = _extract_prompt_rules(SAMPLE_PROMPT_JSON)
        assert len(rules) == 3
        assert all(r["category"] == "prompt_rule" for r in rules)
        assert any("PEP 8" in r["statement"] for r in rules)

    def test_extracts_rules_categorized_dict(self):
        """Categorized dict format: rules.quality_gates, rules.constraints."""
        rules = _extract_prompt_rules(SAMPLE_PROMPT_JSON_CATEGORIZED)
        quality = [r for r in rules if r["category"] == "quality_gates"]
        constraints = [r for r in rules if r["category"] == "constraints"]
        assert len(quality) == 3
        assert len(constraints) == 2

    def test_ignores_capabilities_and_other_fields(self):
        """Capabilities, responsibilities, and core_directive are NOT extracted."""
        rules = _extract_prompt_rules(SAMPLE_PROMPT_JSON)
        # Should only have the 3 items from the 'rules' key
        assert len(rules) == 3
        # No capability/responsibility items should leak in
        assert not any("Fixtures" in r["statement"] for r in rules)
        assert not any("Idiomatic" in r["statement"] for r in rules)

    def test_empty_prompt_returns_nothing(self):
        rules = _extract_prompt_rules({"task": "empty"})
        assert len(rules) == 0

    def test_no_rules_key_returns_nothing(self):
        """Prompt with instructions but no rules key → nothing extracted."""
        rules = _extract_prompt_rules(
            {
                "task": "test",
                "instructions": {
                    "core_directive": "Do stuff",
                    "capabilities": {"a": ["b"]},
                },
            }
        )
        assert len(rules) == 0


# ── Integration tests (with mock engine) ──────────────────────────────


class MockHybridRetriever:
    embed_model = None


class MockEngine:
    """Minimal mock for IntelligenceGraphEngine."""

    def __init__(self):
        import networkx as nx

        self.graph = nx.MultiDiGraph()
        self.backend = None
        self.hybrid_retriever = MockHybridRetriever()

    def link_nodes(self, src, tgt, edge_type, metadata=None):
        self.graph.add_edge(src, tgt, type=edge_type, **(metadata or {}))

    def _serialize_node(self, node, label=None):
        return node.model_dump()

    def _upsert_node(self, label, node_id, data):
        pass


class TestPolicyIngestor:
    """Integration tests for the PolicyIngestor."""

    @pytest.fixture
    def engine(self):
        return MockEngine()

    @pytest.fixture
    def ingestor(self, engine):
        return PolicyIngestor(engine)

    def test_ingest_constitution_from_file(self, ingestor, engine, tmp_path):
        """Test ingesting a constitution.md from a workspace."""
        # Create .specify/memory/constitution.md
        specify_dir = tmp_path / ".specify" / "memory"
        specify_dir.mkdir(parents=True)
        (specify_dir / "constitution.md").write_text(SAMPLE_CONSTITUTION)

        stats = ingestor.ingest_constitution(str(tmp_path))

        assert stats["policies_ingested"] > 0
        assert stats["edges_created"] > 0

        # Verify policy nodes were created
        policy_nodes = [
            n for n, d in engine.graph.nodes(data=True) if d.get("type") == "policy"
        ]
        assert len(policy_nodes) == stats["policies_ingested"]

    def test_ingest_constitution_creates_project_node(self, ingestor, engine, tmp_path):
        """Test that a project anchor node is created."""
        specify_dir = tmp_path / ".specify" / "memory"
        specify_dir.mkdir(parents=True)
        (specify_dir / "constitution.md").write_text(SAMPLE_CONSTITUTION)

        ingestor.ingest_constitution(str(tmp_path))

        project_nodes = [
            n
            for n, d in engine.graph.nodes(data=True)
            if d.get("type") == "software_project"
        ]
        assert len(project_nodes) == 1

    def test_ingest_constitution_normative_priority(self, ingestor, engine, tmp_path):
        """Test that normative (MUST) policies get higher priority."""
        specify_dir = tmp_path / ".specify" / "memory"
        specify_dir.mkdir(parents=True)
        (specify_dir / "constitution.md").write_text(SAMPLE_CONSTITUTION)

        ingestor.ingest_constitution(str(tmp_path))

        policy_data = [
            d for _, d in engine.graph.nodes(data=True) if d.get("type") == "policy"
        ]

        normative = [
            d for d in policy_data if d.get("metadata", {}).get("is_normative")
        ]
        non_normative = [
            d for d in policy_data if not d.get("metadata", {}).get("is_normative")
        ]

        if normative and non_normative:
            avg_norm_priority = sum(d.get("priority", 0) for d in normative) / len(
                normative
            )
            avg_non_priority = sum(d.get("priority", 0) for d in non_normative) / len(
                non_normative
            )
            assert avg_norm_priority > avg_non_priority

    def test_ingest_constitution_missing_file(self, ingestor, engine, tmp_path):
        """Test graceful handling when no constitution exists."""
        stats = ingestor.ingest_constitution(str(tmp_path))
        assert stats["policies_ingested"] == 0

    def test_ingest_constitution_alternative_locations(
        self, ingestor, engine, tmp_path
    ):
        """Test that CONSTITUTION.md in project root is also found."""
        (tmp_path / "CONSTITUTION.md").write_text(SAMPLE_CONSTITUTION)
        stats = ingestor.ingest_constitution(str(tmp_path))
        assert stats["policies_ingested"] > 0

    def test_ingest_prompt_rules(self, ingestor, engine, tmp_path):
        """Test ingesting rules from prompt JSON files."""
        # Create a prompts directory with a test JSON
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "test_agent.json").write_text(json.dumps(SAMPLE_PROMPT_JSON))

        stats = ingestor.ingest_prompt_rules(prompts_dir=str(prompts_dir))

        assert stats["prompts_scanned"] == 1
        # Only the 3 items from the 'rules' key
        assert stats["policies_ingested"] == 3

    def test_ingest_all_combines_sources(self, ingestor, engine, tmp_path):
        """Test the unified ingestion of all sources."""
        # Constitution
        specify_dir = tmp_path / ".specify" / "memory"
        specify_dir.mkdir(parents=True)
        (specify_dir / "constitution.md").write_text(SAMPLE_CONSTITUTION)

        stats = ingestor.ingest_all(str(tmp_path))

        assert "constitution" in stats
        assert "prompts" in stats
        assert stats["constitution"]["policies_ingested"] > 0

    def test_query_policies_by_project(self, ingestor, engine, tmp_path):
        """Test querying policies filtered by project name."""
        specify_dir = tmp_path / ".specify" / "memory"
        specify_dir.mkdir(parents=True)
        (specify_dir / "constitution.md").write_text(SAMPLE_CONSTITUTION)

        ingestor.ingest_constitution(str(tmp_path))

        results = ingestor.query_policies_for_context(project_name="test-project")
        assert len(results) > 0
        assert all("test-project" in r["applies_to"] for r in results)

    def test_query_policies_normative_only(self, ingestor, engine, tmp_path):
        """Test filtering for normative-only policies."""
        specify_dir = tmp_path / ".specify" / "memory"
        specify_dir.mkdir(parents=True)
        (specify_dir / "constitution.md").write_text(SAMPLE_CONSTITUTION)

        ingestor.ingest_constitution(str(tmp_path))

        results = ingestor.query_policies_for_context(
            project_name="test-project",
            include_normative_only=True,
        )
        assert len(results) > 0
        assert all(r["is_normative"] for r in results)

    def test_query_policies_by_category(self, ingestor, engine, tmp_path):
        """Test filtering by policy category."""
        specify_dir = tmp_path / ".specify" / "memory"
        specify_dir.mkdir(parents=True)
        (specify_dir / "constitution.md").write_text(SAMPLE_CONSTITUTION)

        ingestor.ingest_constitution(str(tmp_path))

        results = ingestor.query_policies_for_context(
            project_name="test-project",
            category="quality_gate",
        )
        assert len(results) > 0
        assert all(r["category"] == "quality_gate" for r in results)

    def test_render_policies_for_prompt(self, ingestor, engine, tmp_path):
        """Test rendering policies as markdown for prompt injection."""
        specify_dir = tmp_path / ".specify" / "memory"
        specify_dir.mkdir(parents=True)
        (specify_dir / "constitution.md").write_text(SAMPLE_CONSTITUTION)

        ingestor.ingest_constitution(str(tmp_path))

        results = ingestor.query_policies_for_context(project_name="test-project")
        rendered = ingestor.render_policies_for_prompt(results)

        assert "Active Policies" in rendered
        assert "Project Constitution" in rendered
        assert "[MUST]" in rendered

    def test_render_policies_empty(self, ingestor):
        """Test rendering with no policies."""
        rendered = ingestor.render_policies_for_prompt([])
        assert rendered == ""

    def test_real_constitution_parse(self):
        """Test parsing the actual agent-utilities constitution if it exists."""
        const_path = Path(
            "/home/apps/workspace/agent-packages/agent-utilities/.specify/memory/constitution.md"
        )
        if not const_path.exists():
            pytest.skip("agent-utilities constitution not found")

        content = const_path.read_text()
        parsed = parse_constitution_md(content)

        assert parsed["project_name"] == "agent-utilities"
        assert len(parsed["policies"]) > 10  # Real constitution has many policies

        # Verify categories
        categories = {p["category"] for p in parsed["policies"]}
        assert "principle" in categories
        assert "normative" in categories
        assert "quality_gate" in categories
