"""Tests for CONCEPT:KG-2.12 — KG Source Resolver.

Validates:
- ResolvedSource model
- Paper and codebase resolution from KG
- Filesystem materialization
- Graceful behavior when no KG is available
- Cleanup
"""

from pathlib import Path
from unittest.mock import MagicMock

import networkx as nx
import pytest

from agent_utilities.knowledge_graph.source_resolver import (
    DEFAULT_RESOLVE_DIR,
    KGSourceResolver,
    ResolvedSource,
)


@pytest.mark.concept("KG-2.12")
class TestResolvedSource:
    """Test ResolvedSource model."""

    def test_model_defaults(self):
        src = ResolvedSource(source_id="test:1", name="Test")
        assert src.source_type == "research"
        assert src.file_path == ""
        assert src.relevance_score == 0.0
        assert src.authors == []

    def test_model_full(self):
        src = ResolvedSource(
            source_id="article:scholarx:2406-12345",
            name="Test Paper",
            source_type="research",
            file_path="/tmp/test.md",
            relevance_score=4.5,
            authors=["Alice", "Bob"],
            domains=["orchestration", "memory"],
            content_preview="This paper proposes...",
        )
        assert src.source_id == "article:scholarx:2406-12345"
        assert len(src.authors) == 2


@pytest.mark.concept("KG-2.12")
class TestKGSourceResolverNoEngine:
    """Test graceful behavior when no KG engine is available."""

    def test_is_available_false(self):
        resolver = KGSourceResolver(engine=None)
        assert not resolver.is_available()

    def test_resolve_papers_empty(self):
        resolver = KGSourceResolver(engine=None)
        result = resolver.resolve_papers("test query")
        assert result == []

    def test_resolve_codebases_empty(self):
        resolver = KGSourceResolver(engine=None)
        result = resolver.resolve_codebases("test query")
        assert result == []

    def test_resolve_any_empty(self):
        resolver = KGSourceResolver(engine=None)
        result = resolver.resolve_any("test query")
        assert result == []


@pytest.mark.concept("KG-2.12")
class TestKGSourceResolverWithEngine:
    """Test resolution with a mocked KG engine."""

    def setup_method(self):
        self.graph = nx.MultiDiGraph()
        self.engine = MagicMock()
        self.engine.graph = self.graph
        self.engine.hybrid_retriever = None

    def _add_article(self, article_id, title, content, tags=None, importance=0.8):
        from agent_utilities.models.knowledge_graph import RegistryNodeType
        self.graph.add_node(article_id, **{
            "type": RegistryNodeType.ARTICLE,
            "name": title,
            "description": content[:200],
            "content": content,
            "tags": tags or [],
            "importance_score": importance,
            "timestamp": "2026-05-07T00:00:00Z",
        })

    def _add_kb(self, kb_id, name, source_type="directory"):
        from agent_utilities.models.knowledge_graph import RegistryNodeType
        self.graph.add_node(kb_id, **{
            "type": RegistryNodeType.KNOWLEDGE_BASE,
            "name": name,
            "description": f"Knowledge base: {name}",
            "content": f"Content of {name} knowledge base with code examples",
            "source_type": source_type,
            "importance_score": 0.7,
        })

    def test_resolve_papers(self, tmp_path):
        self._add_article(
            "article:test:1",
            "Multi-Agent Planning",
            "A paper about multi-agent planning and orchestration systems.",
            tags=["orchestration", "planning"],
        )
        self._add_article(
            "article:test:2",
            "Protein Folding Study",
            "A study on protein folding mechanisms.",
            tags=["biology"],
        )

        resolver = KGSourceResolver(engine=self.engine, resolve_dir=str(tmp_path))
        results = resolver.resolve_papers("multi-agent planning")

        assert len(results) >= 1
        assert results[0].name == "Multi-Agent Planning"
        assert Path(results[0].file_path).exists()

    def test_resolve_codebases(self, tmp_path):
        self._add_kb("kb:agent-utilities", "agent-utilities", source_type="directory")

        resolver = KGSourceResolver(engine=self.engine, resolve_dir=str(tmp_path))
        results = resolver.resolve_codebases("agent utilities")

        assert len(results) >= 1
        assert results[0].source_type == "codebase"

    def test_resolve_any_combined(self, tmp_path):
        self._add_article(
            "article:test:1",
            "Knowledge Graph Research",
            "Research on knowledge graph systems and memory.",
            tags=["knowledge graph"],
        )
        self._add_kb("kb:test-project", "test-project", source_type="directory")

        resolver = KGSourceResolver(engine=self.engine, resolve_dir=str(tmp_path))
        results = resolver.resolve_any("knowledge graph")

        # Should return at least the article
        assert len(results) >= 1

    def test_materialization_creates_files(self, tmp_path):
        self._add_article(
            "article:test:mat",
            "Materialization Test",
            "Content that should be written to disk.",
            tags=["test"],
        )

        resolver = KGSourceResolver(engine=self.engine, resolve_dir=str(tmp_path))
        results = resolver.resolve_papers("materialization test")

        assert len(results) == 1
        file_path = Path(results[0].file_path)
        assert file_path.exists()
        content = file_path.read_text()
        assert "Materialization Test" in content
        assert "Content that should be written to disk" in content

    def test_cleanup(self, tmp_path):
        self._add_article(
            "article:test:cleanup",
            "Cleanup Test",
            "Content to be cleaned up.",
        )

        resolver = KGSourceResolver(engine=self.engine, resolve_dir=str(tmp_path))
        resolver.resolve_papers("cleanup test")

        # Verify file exists
        assert len(list(tmp_path.glob("*.md"))) > 0

        # Cleanup
        removed = resolver.cleanup()
        assert removed > 0
        assert len(list(tmp_path.glob("*.md"))) == 0

    def test_no_content_skipped(self, tmp_path):
        from agent_utilities.models.knowledge_graph import RegistryNodeType
        self.graph.add_node("article:empty", **{
            "type": RegistryNodeType.ARTICLE,
            "name": "Empty Article",
            "description": "",
            "content": "",
            "tags": [],
        })

        resolver = KGSourceResolver(engine=self.engine, resolve_dir=str(tmp_path))
        results = resolver.resolve_papers("empty")
        # Should not materialize empty content
        assert len(results) == 0


@pytest.mark.concept("KG-2.12")
class TestDefaultResolveDir:
    """Test default resolve directory."""

    def test_default_dir_path(self):
        assert ".scholarx" in DEFAULT_RESOLVE_DIR
        assert "analysis" in DEFAULT_RESOLVE_DIR
