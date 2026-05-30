from __future__ import annotations

"""Unit tests for the Engineering Rules Engine (CONCEPT:KG-2.2).

Tests the rule ingestor's markdown parser, node creation, conflict
resolution, and prompt rendering capabilities.
"""


import pytest

from agent_utilities.knowledge_graph.security.rule_ingestor import (
    RuleIngestor,
    _extract_list_items,
    _parse_frontmatter,
    _split_by_h2,
    parse_mini_markdown,
)
from agent_utilities.models.knowledge_graph import (
    EngineeringRuleNode,
    RegistryEdgeType,
    RegistryNodeType,
    RuleBookNode,
)
from agent_utilities.prompting.structured import (
    EngineeringRulesSection,
    PromptInstructions,
    StructuredPrompt,
    TriggerRule,
)

# ── Sample markdown fixtures ─────────────────────────────────────────

SAMPLE_MINI_MD = """\
# Clean Architecture: Mini

## When to use

Use when building or extending systems where long-term modularity, testability, and independence from frameworks, databases, and external services matter more than short-term speed.

## Primary bias to correct

Frameworks, databases, and HTTP are not the architecture.

## Decision rules

- Structure code so that source-code dependency direction always points inward.
- Put business logic in plain-language domain objects with no framework imports.
- Express external contracts (DB, HTTP, queues) as interfaces defined by the domain.

## Trigger rules

- When adding a new framework dependency, verify it stays at the outer ring.
- When touching domain logic, check that no infrastructure import has crept in.

## Final checklist

- Dependency direction inward?
- Domain free of framework imports?
- Ports tested with fakes?
"""

SAMPLE_MINI_MD_WITH_FRONTMATTER = """\
---
name: "Clean Architecture"
description: "Dependency management and architectural boundaries for maintainable systems"
author: "Robert C. Martin"
version: "2.1.0"
tier: "mini"
---
# Clean Architecture: Mini

## When to use

Use when building or extending systems where long-term modularity matters.

## Primary bias to correct

Frameworks are not the architecture.

## Decision rules

- Structure code so that dependency direction always points inward.
- Put business logic in plain-language domain objects.

## Trigger rules

- When adding a new framework dependency, verify it stays at the outer ring.

## Final checklist

- Dependency direction inward?
"""

SAMPLE_NANO_MD = """\
# Refactoring: Nano

## When to use

Use when improving structure without changing behavior.

## Primary bias to correct

Do not mix redesign with refactoring.

## Decision rules

- Preserve observable behavior.
- Use small, verifiable steps.
- Get a safety net first.
"""


# ── Markdown parser tests ────────────────────────────────────────────


class TestMarkdownParser:
    """Tests for the mini/nano markdown parser."""

    def test_split_by_h2_extracts_all_sections(self):
        sections = _split_by_h2(SAMPLE_MINI_MD)
        assert "When to use" in sections
        assert "Primary bias to correct" in sections
        assert "Decision rules" in sections
        assert "Trigger rules" in sections
        assert "Final checklist" in sections

    def test_extract_list_items_basic(self):
        text = "- First item\n- Second item\n- Third item"
        items = _extract_list_items(text)
        assert len(items) == 3
        assert items[0] == "First item"
        assert items[2] == "Third item"

    def test_extract_list_items_multiline(self):
        text = "- First item that\n  continues here\n- Second item"
        items = _extract_list_items(text)
        assert len(items) == 2
        assert "continues here" in items[0]

    def test_parse_mini_markdown_full_structure(self):
        parsed = parse_mini_markdown(SAMPLE_MINI_MD, "clean-architecture", "mini")
        assert parsed.book_slug == "clean-architecture"
        assert parsed.tier == "mini"
        assert "Clean Architecture" in parsed.title
        assert "modularity" in parsed.when_to_use
        assert "not the architecture" in parsed.primary_bias
        assert len(parsed.decision_rules) == 3
        assert len(parsed.trigger_rules) == 2
        assert len(parsed.checklist_items) == 3

    def test_parse_nano_markdown(self):
        parsed = parse_mini_markdown(SAMPLE_NANO_MD, "refactoring", "nano")
        assert parsed.tier == "nano"
        assert len(parsed.decision_rules) == 3
        assert len(parsed.trigger_rules) == 0
        assert "observable behavior" in parsed.decision_rules[0].lower()

    def test_parse_frontmatter_extracts_fields(self):
        fm, body = _parse_frontmatter(SAMPLE_MINI_MD_WITH_FRONTMATTER)
        assert fm["name"] == "Clean Architecture"
        assert fm["version"] == "2.1.0"
        assert fm["author"] == "Robert C. Martin"
        assert fm["tier"] == "mini"
        assert "# Clean Architecture" in body

    def test_parse_frontmatter_absent(self):
        fm, body = _parse_frontmatter(SAMPLE_MINI_MD)
        assert fm == {}
        assert body == SAMPLE_MINI_MD

    def test_frontmatter_version_propagates_to_parsed(self):
        parsed = parse_mini_markdown(
            SAMPLE_MINI_MD_WITH_FRONTMATTER, "clean-architecture", "mini"
        )
        assert parsed.frontmatter_version == "2.1.0"
        assert parsed.frontmatter_name == "Clean Architecture"
        assert parsed.frontmatter_author == "Robert C. Martin"
        assert parsed.frontmatter_description.startswith("Dependency management")

    def test_frontmatter_still_parses_body(self):
        """Frontmatter doesn't interfere with H2 section parsing."""
        parsed = parse_mini_markdown(
            SAMPLE_MINI_MD_WITH_FRONTMATTER, "clean-architecture", "mini"
        )
        assert len(parsed.decision_rules) == 2
        assert len(parsed.trigger_rules) == 1
        assert len(parsed.checklist_items) == 1
        assert "Clean Architecture" in parsed.title


# ── Pydantic model tests ─────────────────────────────────────────────


class TestEngineeringRuleModels:
    """Tests for the KG Pydantic models."""

    def test_engineering_rule_node_creation(self):
        node = EngineeringRuleNode(
            id="rule:test-001",
            name="Test Rule",
            principle_id="test-rule-001",
            statement="Always test your code",
            tier="mini",
            rule_class="decision-changing",
            bias_corrected="skipping tests",
            task_relevance_tags=["code-quality", "testing"],
            source_book_id="book:clean-code",
        )
        assert node.type == RegistryNodeType.ENGINEERING_RULE
        assert node.tier == "mini"
        assert node.efficacy_score == 0.5
        assert node.conflict_weight == 0.5
        assert node.version == "1.0.0"

    def test_engineering_rule_inherits_principle(self):
        """EngineeringRuleNode should inherit PrincipleNode fields."""
        node = EngineeringRuleNode(
            id="rule:inherit-test",
            name="Inherit Test",
            principle_id="inherit-001",
            statement="Test inheritance",
            strength=0.8,
            scope_node_ids=["concept:testing"],
        )
        assert node.strength == 0.8
        assert "concept:testing" in node.scope_node_ids

    def test_rule_book_node_creation(self):
        node = RuleBookNode(
            id="book:clean-architecture",
            name="Clean Architecture",
            book_id="clean-architecture",
            author="Robert C. Martin",
            domain_tags=["architecture", "boundaries"],
            primary_bias="Frameworks are not the architecture",
        )
        assert node.type == RegistryNodeType.RULE_BOOK
        assert node.book_id == "clean-architecture"
        assert "architecture" in node.domain_tags

    def test_engineering_rule_tier_validation(self):
        """Tier must be one of full, mini, nano."""
        with pytest.raises((ValueError, TypeError)):
            EngineeringRuleNode(
                id="rule:bad-tier",
                name="Bad Tier",
                principle_id="bad-001",
                statement="Bad",
                tier="invalid",  # type: ignore
            )

    def test_engineering_rule_class_validation(self):
        """Rule class must be from the PROCESS.md taxonomy."""
        with pytest.raises((ValueError, TypeError)):
            EngineeringRuleNode(
                id="rule:bad-class",
                name="Bad Class",
                principle_id="bad-002",
                statement="Bad",
                rule_class="invented-class",  # type: ignore
            )


# ── StructuredPrompt integration tests ────────────────────────────────


class TestStructuredPromptRules:
    """Tests for the StructuredPrompt engineering rules integration."""

    def test_engineering_rules_section_creation(self):
        section = EngineeringRulesSection(
            always_on=["clean-code.nano"],
            on_demand={"refactoring": ["refactoring.mini"]},
            trigger_rules=[
                TriggerRule(when="adding external dependency", apply="release-it.mini"),
            ],
            context_budget="mini",
        )
        assert len(section.always_on) == 1
        assert "refactoring" in section.on_demand
        assert len(section.trigger_rules) == 1

    def test_engineering_rules_section_render(self):
        section = EngineeringRulesSection(
            always_on=["clean-code.nano"],
            on_demand={"refactoring": ["refactoring.mini"]},
            trigger_rules=[
                TriggerRule(when="adding external dependency", apply="release-it.mini"),
            ],
        )
        rendered = section.render_section()
        assert "ALWAYS-ON RULES" in rendered
        assert "clean-code.nano" in rendered
        assert "ON-DEMAND RULES" in rendered
        assert "Refactoring" in rendered
        assert "TRIGGER RULES" in rendered
        assert "release-it.mini" in rendered

    def test_structured_prompt_with_rules_key(self):
        """Test the 'rules' key for extracted engineering guidance."""
        prompt = StructuredPrompt(
            task="python_programmer",
            instructions=PromptInstructions(
                core_directive="Write clean Python code.",
            ),
            rules=[
                "Follow PEP 8 and Pythonic patterns",
                "Use type hints on all public APIs",
                "Write tests before implementation (TDD)",
            ],
        )
        rendered = prompt.render()
        assert "ENGINEERING RULES" in rendered
        assert "PEP 8" in rendered
        assert "TDD" in rendered

    def test_structured_prompt_full_render_with_engineering_rules(self):
        """Test full render with both instructions and engineering_rules."""
        prompt = StructuredPrompt(
            task="architect",
            instructions=PromptInstructions(
                core_directive="Design clean, modular architectures.",
            ),
            engineering_rules=EngineeringRulesSection(
                always_on=["clean-architecture.nano"],
                trigger_rules=[
                    TriggerRule(
                        when="adding framework dependency",
                        apply="clean-architecture.mini#trigger-rules",
                    ),
                ],
            ),
            rules=["Dependency direction always points inward"],
        )
        rendered = prompt.render()
        assert "Design clean" in rendered
        assert "ALWAYS-ON RULES" in rendered
        assert "ENGINEERING RULES" in rendered
        assert "Dependency direction" in rendered

    def test_structured_prompt_json_roundtrip_with_rules(self):
        """Test that engineering_rules survives JSON serialization."""
        prompt = StructuredPrompt(
            task="test",
            engineering_rules=EngineeringRulesSection(
                always_on=["clean-code.nano"],
                context_budget="nano",
            ),
            rules=["Test rule 1"],
        )
        json_str = prompt.model_dump_json()
        restored = StructuredPrompt.model_validate_json(json_str)
        assert restored.engineering_rules is not None
        assert restored.engineering_rules.always_on == ["clean-code.nano"]
        assert restored.rules == ["Test rule 1"]


# ── RuleIngestor integration tests (with mock engine) ─────────────────


class MockHybridRetriever:
    embed_model = None


class MockBackend:
    pass


class MockEngine:
    """Minimal mock for IntelligenceGraphEngine."""

    def __init__(self):
        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )

        self.graph = GraphComputeEngine(backend_type="rust")
        self.backend = None
        self.hybrid_retriever = MockHybridRetriever()

    def link_nodes(self, src, tgt, edge_type, metadata=None):
        self.graph.add_edge(src, tgt, type=edge_type, **(metadata or {}))

    def _serialize_node(self, node, label=None):
        return node.model_dump()

    def _upsert_node(self, label, node_id, data):
        pass


class TestRuleIngestor:
    """Integration tests for rule ingestion into a mock KG."""

    @pytest.fixture
    def engine(self):
        return MockEngine()

    @pytest.fixture
    def ingestor(self, engine):
        return RuleIngestor(engine)

    def test_ingest_single_parsed_ruleset(self, ingestor, engine):
        """Test ingesting a parsed mini.md into the KG."""
        parsed = parse_mini_markdown(SAMPLE_MINI_MD, "clean-architecture", "mini")
        book_node = RuleBookNode(
            id="book:clean-architecture",
            name="Clean Architecture",
            book_id="clean-architecture",
            author="Robert C. Martin",
            domain_tags=["architecture"],
        )
        engine.graph.add_node(book_node.id, **book_node.model_dump())

        count = ingestor._ingest_rules_from_parsed(
            parsed,
            book_node.id,
            {"domain_tags": ["architecture"]},
            "2026-05-01T00:00:00Z",
            "1.0.0",
        )

        # 3 decision + 2 trigger + 3 checklist = 8 rules
        assert count == 8

        # Verify nodes were created
        rule_nodes = [
            n
            for n, d in engine.graph.nodes(data=True)
            if d.get("type") == "engineering_rule"
        ]
        assert len(rule_nodes) == 8

        # Verify edges
        derived_edges = [
            (u, v)
            for u, v, d in engine.graph.edges(data=True)
            if d.get("type") == RegistryEdgeType.WAS_DERIVED_FROM
        ]
        assert len(derived_edges) == 8

    def test_query_rules_for_task(self, ingestor, engine):
        """Test querying rules by task tags."""
        parsed = parse_mini_markdown(SAMPLE_MINI_MD, "clean-architecture", "mini")
        book_node = RuleBookNode(
            id="book:clean-architecture",
            name="Clean Architecture",
            book_id="clean-architecture",
        )
        engine.graph.add_node(book_node.id, **book_node.model_dump())

        ingestor._ingest_rules_from_parsed(
            parsed,
            book_node.id,
            {"domain_tags": ["architecture", "boundaries"]},
            "2026-05-01T00:00:00Z",
            "1.0.0",
        )

        # Query for architecture tasks
        results = ingestor.query_rules_for_task(
            task_tags=["architecture"],
            tier="mini",
        )
        assert len(results) > 0
        assert all(r["tier"] == "mini" for r in results)

    def test_render_rules_for_prompt(self, ingestor, engine):
        """Test rendering rules as markdown for prompt injection."""
        parsed = parse_mini_markdown(SAMPLE_MINI_MD, "clean-architecture", "mini")
        book_node = RuleBookNode(
            id="book:clean-architecture",
            name="Clean Architecture",
            book_id="clean-architecture",
        )
        engine.graph.add_node(book_node.id, **book_node.model_dump())

        ingestor._ingest_rules_from_parsed(
            parsed,
            book_node.id,
            {"domain_tags": ["architecture"]},
            "2026-05-01T00:00:00Z",
            "1.0.0",
        )

        results = ingestor.query_rules_for_task(
            task_tags=["architecture"],
            tier="mini",
        )
        rendered = ingestor.render_rules_for_prompt(results)
        assert "Engineering Rules" in rendered
        assert "Clean Architecture" in rendered

    def test_conflict_resolution(self, ingestor, engine):
        """Test that conflict resolution keeps the higher-weighted rule."""
        # Create two conflicting rules
        node_a = EngineeringRuleNode(
            id="rule:a",
            name="Rule A",
            principle_id="a-001",
            statement="Functions should be very short",
            conflict_weight=0.8,
            task_relevance_tags=["code-quality"],
        )
        node_b = EngineeringRuleNode(
            id="rule:b",
            name="Rule B",
            principle_id="b-001",
            statement="Functions should be as long as needed",
            conflict_weight=0.6,
            task_relevance_tags=["code-quality"],
        )

        engine.graph.add_node(node_a.id, **node_a.model_dump())
        engine.graph.add_node(node_b.id, **node_b.model_dump())

        # Wire conflict edge
        engine.link_nodes(
            node_a.id,
            node_b.id,
            RegistryEdgeType.CONFLICTS_WITH,
        )

        # Simulate query results
        rules = [
            {"id": "rule:a", "statement": "Short functions", "score": 0.8},
            {"id": "rule:b", "statement": "Long functions", "score": 0.7},
        ]

        resolved = ingestor.resolve_conflicts(rules)
        # Rule A (weight=0.8) should win over Rule B (weight=0.6)
        assert len(resolved) == 1
        assert resolved[0]["id"] == "rule:a"
