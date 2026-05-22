#!/usr/bin/python
from __future__ import annotations

"""Tests for Chat Search, Agents MD, and Engineering Patterns facade modules.

CONCEPT:KG-2.1

Validates the facade modules that were referenced in overview.md but
previously existed only as inline implementations scattered across
other modules.
"""


import textwrap
from pathlib import Path

import pytest

# ========================================================================
# KG-2.13 — Chat Search Facade
# ========================================================================


@pytest.mark.concept("KG-2.13")
class TestChatSearchFacade:
    """Tests for agent_utilities.knowledge_graph.chat_search."""

    def test_imports_from_facade(self) -> None:
        from agent_utilities.knowledge_graph.retrieval.chat_search import (
            ChatRecallMessage,
            ChatRecallResult,
            ChatRecallResults,
            ChatSearchResult,
            search_chat_history,
            search_sessions,
        )

        assert ChatRecallMessage is not None
        assert ChatRecallResult is not None
        assert ChatRecallResults is not None
        assert ChatSearchResult is not None
        assert callable(search_chat_history)
        assert callable(search_sessions)

    def test_chat_search_result_dataclass(self) -> None:
        from agent_utilities.knowledge_graph.retrieval.chat_search import (
            ChatSearchResult,
        )

        result = ChatSearchResult(
            session_id="test-123",
            session_title="Test Session",
            snippet="Hello world",
            relevance=0.85,
            last_activity="2026-05-07T00:00:00",
            match_count=3,
        )
        assert result.session_id == "test-123"
        assert result.relevance == 0.85
        assert result.match_count == 3

    def test_search_sessions_no_engine(self) -> None:
        """search_sessions should gracefully return empty without a KG engine."""
        from agent_utilities.knowledge_graph.retrieval.chat_search import (
            search_sessions,
        )

        results = search_sessions("test query")
        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_sessions_empty_query(self) -> None:
        from agent_utilities.knowledge_graph.retrieval.chat_search import (
            search_sessions,
        )

        results = search_sessions("")
        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_chat_history_no_engine(self) -> None:
        from agent_utilities.knowledge_graph.retrieval.chat_search import (
            search_chat_history,
        )

        results = search_chat_history("kubernetes")
        assert results.query == "kubernetes"
        assert results.total_matches == 0

    def test_all_exports(self) -> None:
        import agent_utilities.knowledge_graph.retrieval.chat_search as mod

        assert hasattr(mod, "__all__")
        assert "ChatSearchResult" in mod.__all__
        assert "search_sessions" in mod.__all__
        assert "search_chat_history" in mod.__all__


# ========================================================================
# KG-2.14 — Agents MD Facade
# ========================================================================


@pytest.mark.concept("KG-2.14")
class TestAgentsMdFacade:
    """Tests for agent_utilities.knowledge_graph.agents_md."""

    def test_imports_from_facade(self) -> None:
        from agent_utilities.knowledge_graph.core.agents_md import (
            extract_project_metadata,
            find_agents_md,
            inject_project_context,
            load_agents_md,
        )

        assert callable(load_agents_md)
        assert callable(find_agents_md)
        assert callable(inject_project_context)
        assert callable(extract_project_metadata)

    def test_load_agents_md_found(self, tmp_path: Path) -> None:
        from agent_utilities.knowledge_graph.core.agents_md import load_agents_md

        (tmp_path / "AGENTS.md").write_text("# My Project\n\nRules here")
        content = load_agents_md(tmp_path)
        assert content is not None
        assert "My Project" in content
        assert "Rules here" in content

    def test_load_agents_md_not_found(self, tmp_path: Path) -> None:
        from agent_utilities.knowledge_graph.core.agents_md import load_agents_md

        content = load_agents_md(tmp_path)
        assert content is None

    def test_load_agents_md_multiple_files(self, tmp_path: Path) -> None:
        from agent_utilities.knowledge_graph.core.agents_md import load_agents_md

        (tmp_path / "AGENTS.md").write_text("# Main rules")
        (tmp_path / "AGENTS.local.md").write_text("# Local overrides")

        content = load_agents_md(tmp_path)
        assert content is not None
        assert "Main rules" in content
        assert "Local overrides" in content

    def test_load_agents_md_dot_agents_dir(self, tmp_path: Path) -> None:
        from agent_utilities.knowledge_graph.core.agents_md import load_agents_md

        agents_dir = tmp_path / ".agents"
        agents_dir.mkdir()
        (agents_dir / "AGENTS.md").write_text("# Nested rules")

        content = load_agents_md(tmp_path)
        assert content is not None
        assert "Nested rules" in content

    def test_load_agents_md_invalid_path(self) -> None:
        from agent_utilities.knowledge_graph.core.agents_md import load_agents_md

        content = load_agents_md("/nonexistent/path/xyz")
        assert content is None

    def test_find_agents_md_walks_up(self, tmp_path: Path) -> None:
        from agent_utilities.knowledge_graph.core.agents_md import find_agents_md

        (tmp_path / "AGENTS.md").write_text("# Root rules")
        nested = tmp_path / "src" / "module"
        nested.mkdir(parents=True)

        found = find_agents_md(nested)
        assert found is not None
        assert found.name == "AGENTS.md"
        assert found.parent == tmp_path

    def test_find_agents_md_not_found(self, tmp_path: Path) -> None:
        from agent_utilities.knowledge_graph.core.agents_md import find_agents_md

        nested = tmp_path / "empty" / "dir"
        nested.mkdir(parents=True)

        found = find_agents_md(nested)
        # May or may not find depending on filesystem — at minimum shouldn't crash
        assert found is None or found.is_file()

    def test_inject_project_context(self, tmp_path: Path) -> None:
        from agent_utilities.knowledge_graph.core.agents_md import (
            inject_project_context,
        )

        (tmp_path / "AGENTS.md").write_text("# Rules\n- Use pytest")
        prompt = "You are a helpful assistant."
        enriched = inject_project_context(prompt, tmp_path)
        assert "You are a helpful assistant." in enriched
        assert "AGENTS.md (Project Rules & Memory)" in enriched
        assert "Use pytest" in enriched

    def test_inject_project_context_with_memory(self, tmp_path: Path) -> None:
        from agent_utilities.knowledge_graph.core.agents_md import (
            inject_project_context,
        )

        (tmp_path / "AGENTS.md").write_text("# Rules")
        (tmp_path / "MEMORY.md").write_text("# Learned stuff")
        enriched = inject_project_context("Base prompt", tmp_path)
        assert "MEMORY.md (Learned Context)" in enriched

    def test_inject_project_context_no_agents(self, tmp_path: Path) -> None:
        from agent_utilities.knowledge_graph.core.agents_md import (
            inject_project_context,
        )

        enriched = inject_project_context("Base prompt", tmp_path)
        assert enriched == "Base prompt"

    def test_extract_project_metadata(self, tmp_path: Path) -> None:
        from agent_utilities.knowledge_graph.core.agents_md import (
            extract_project_metadata,
        )

        (tmp_path / "AGENTS.md").write_text(
            textwrap.dedent("""\
                # Project: TestApp

                ## Build Commands
                - Build: `make build`

                ## Test Commands
                - Run all: `pytest`
                - Single: `pytest tests/test_foo.py`

                ## Style Guidelines
                - Use ruff
            """)
        )
        meta = extract_project_metadata(tmp_path)
        assert "build_commands" in meta
        assert "test_commands" in meta
        assert "style_guidelines" in meta
        assert "make build" in meta["build_commands"]

    def test_extract_project_metadata_empty(self, tmp_path: Path) -> None:
        from agent_utilities.knowledge_graph.core.agents_md import (
            extract_project_metadata,
        )

        meta = extract_project_metadata(tmp_path)
        assert meta == {}


# ========================================================================
# AHE-3.14 — Engineering Patterns Facade
# ========================================================================


@pytest.mark.concept("AHE-3.14")
class TestEngineeringPatternsFacade:
    """Tests for agent_utilities.harness.engineering."""

    def test_imports_from_facade(self) -> None:
        from agent_utilities.harness.engineering import (
            EngineeringPatternOrchestrator,
            PatternResult,
            PatternType,
        )

        assert EngineeringPatternOrchestrator is not None
        assert PatternResult is not None
        assert PatternType is not None

    def test_pattern_type_enum(self) -> None:
        from agent_utilities.harness.engineering import PatternType

        assert PatternType.TDD == "tdd"
        assert PatternType.FIRST_RUN_TESTS == "first_run_tests"
        assert PatternType.MANUAL_TESTING == "manual_testing"
        assert PatternType.CODE_WALKTHROUGH == "code_walkthrough"
        assert PatternType.INTERACTIVE_EXPLANATION == "interactive_explanation"

    def test_pattern_result_defaults(self) -> None:
        from agent_utilities.harness.engineering import PatternResult, PatternType

        result = PatternResult(pattern=PatternType.TDD)
        assert result.success is True
        assert result.output == ""
        assert result.artifacts == []
        assert result.error is None

    def test_pattern_result_failure(self) -> None:
        from agent_utilities.harness.engineering import PatternResult, PatternType

        result = PatternResult(
            pattern=PatternType.TDD,
            success=False,
            error="spec not found",
        )
        assert result.success is False
        assert result.error == "spec not found"

    def test_orchestrator_creation(self, tmp_path: Path) -> None:
        from agent_utilities.harness.engineering import (
            EngineeringPatternOrchestrator,
        )

        orch = EngineeringPatternOrchestrator(str(tmp_path))
        assert orch.workspace_path == str(tmp_path)

    def test_list_available_patterns(self, tmp_path: Path) -> None:
        from agent_utilities.harness.engineering import (
            EngineeringPatternOrchestrator,
        )

        orch = EngineeringPatternOrchestrator(str(tmp_path))
        patterns = orch.list_available_patterns()
        assert len(patterns) == 5
        names = {p["name"] for p in patterns}
        assert "tdd" in names
        assert "first_run_tests" in names
        assert "manual_testing" in names

    @pytest.mark.asyncio()
    async def test_execute_tdd_requires_spec(self, tmp_path: Path) -> None:
        from agent_utilities.harness.engineering import (
            EngineeringPatternOrchestrator,
            PatternType,
        )

        orch = EngineeringPatternOrchestrator(str(tmp_path))
        result = await orch.execute(PatternType.TDD)
        assert result.success is False
        assert "spec_id" in (result.error or "")

    def test_all_exports(self) -> None:
        import agent_utilities.harness.engineering as mod

        assert hasattr(mod, "__all__")
        assert "EngineeringPatternOrchestrator" in mod.__all__
        assert "PatternResult" in mod.__all__
        assert "PatternType" in mod.__all__
