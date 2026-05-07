"""Tests for CONCEPT:KG-2.11 — Research Artifact Generator.

Validates:
- ResearchArtifact and DigestArtifact models
- Paper artifact generation from KG nodes
- Contribution and method extraction
- Application mapping to concept IDs
- Digest generation and rendering
"""

import time
from unittest.mock import MagicMock

import networkx as nx
import pytest

from agent_utilities.knowledge_graph.research_artifacts import (
    DigestArtifact,
    ResearchArtifact,
    ResearchArtifactGenerator,
)


@pytest.mark.concept("KG-2.11")
class TestResearchArtifactModel:
    """Test ResearchArtifact model."""

    def test_defaults(self):
        art = ResearchArtifact(article_id="test:1", title="Test")
        assert art.summary == ""
        assert art.key_contributions == []
        assert art.methods == []
        assert art.importance_score == 0.0

    def test_full_model(self):
        art = ResearchArtifact(
            article_id="article:test:1",
            title="Test Paper",
            summary="A great paper",
            key_contributions=["Novel approach"],
            methods=["transformer", "RLHF"],
            potential_applications=["Could enhance ORCH-1.0"],
            tags=["orchestration"],
            authors=["Alice"],
            importance_score=0.9,
        )
        assert art.article_id == "article:test:1"
        assert len(art.methods) == 2


@pytest.mark.concept("KG-2.11")
class TestDigestArtifactModel:
    """Test DigestArtifact model."""

    def test_defaults(self):
        digest = DigestArtifact()
        assert digest.period == "daily"
        assert digest.paper_count == 0
        assert digest.markdown == ""


@pytest.mark.concept("KG-2.11")
class TestArtifactGeneratorNoEngine:
    """Test generator behavior without KG engine."""

    def test_generate_without_engine(self):
        gen = ResearchArtifactGenerator(engine=None)
        art = gen.generate_paper_artifact("article:test:1")
        assert art.title == "Unknown"


@pytest.mark.concept("KG-2.11")
class TestArtifactGeneratorWithEngine:
    """Test generator with mocked KG engine."""

    def setup_method(self):
        self.graph = nx.MultiDiGraph()
        self.engine = MagicMock()
        self.engine.graph = self.graph
        self.gen = ResearchArtifactGenerator(engine=self.engine)

    def _add_article(self, article_id, title, content, tags=None):
        from agent_utilities.models.knowledge_graph import RegistryNodeType
        self.graph.add_node(article_id, **{
            "type": RegistryNodeType.ARTICLE,
            "name": title,
            "description": content[:200],
            "content": content,
            "tags": tags or [],
            "importance_score": 0.8,
            "timestamp": "2026-05-07T00:00:00Z",
        })

    def test_generate_paper_artifact(self):
        self._add_article(
            "article:test:1",
            "Multi-Agent Knowledge Graph Memory",
            "We propose a novel multi-agent system that uses a knowledge graph "
            "for persistent memory. We introduce a new transformer-based "
            "architecture with contrastive learning for improved retrieval. "
            "Our approach outperforms existing baselines on standard benchmarks.",
            tags=["orchestration", "memory"],
        )

        art = self.gen.generate_paper_artifact("article:test:1")
        assert art.title == "Multi-Agent Knowledge Graph Memory"
        assert art.importance_score == 0.8
        assert len(art.key_contributions) > 0
        assert "transformer" in art.methods
        assert len(art.potential_applications) > 0

    def test_generate_artifact_applications(self):
        self._add_article(
            "article:test:apps",
            "Tool Use Safety",
            "We study safety mechanisms for tool use in language models with "
            "evaluation of guardrail effectiveness and reward shaping.",
            tags=["safety", "tools"],
        )

        art = self.gen.generate_paper_artifact("article:test:apps")
        # Should map to existing concept IDs
        app_text = " ".join(art.potential_applications)
        assert any(
            concept in app_text
            for concept in ["OS-5", "ECO-4", "AHE-3", "ORCH-1", "KG-2"]
        )

    def test_generate_digest(self):
        self._add_article("article:d1", "Paper Alpha",
                         "Research on multi-agent planning systems.",
                         tags=["orchestration"])
        self._add_article("article:d2", "Paper Beta",
                         "Study on memory retrieval and knowledge graphs.",
                         tags=["memory", "orchestration"])

        digest = self.gen.generate_digest(
            paper_ids=["article:d1", "article:d2"],
            period="daily",
        )

        assert digest.paper_count == 2
        assert digest.period == "daily"
        assert len(digest.top_papers) == 2
        assert "orchestration" in digest.domain_distribution
        assert "Research Digest" in digest.markdown

    def test_digest_emerging_themes(self):
        self._add_article("article:t1", "Paper 1",
                         "Content about transformers.", tags=["reasoning"])
        self._add_article("article:t2", "Paper 2",
                         "More reasoning content.", tags=["reasoning"])
        self._add_article("article:t3", "Paper 3",
                         "Unrelated biology paper.", tags=["biology"])

        digest = self.gen.generate_digest(
            paper_ids=["article:t1", "article:t2", "article:t3"],
            period="weekly",
        )

        # "reasoning" appears 2+ times → emerging theme
        assert "reasoning" in digest.emerging_themes

    def test_save_digest(self, tmp_path):
        digest = DigestArtifact(
            period="daily",
            paper_count=1,
            markdown="# Test Digest\n\nTest content.",
        )

        path = self.gen.save_digest(digest, output_dir=tmp_path)
        assert "digest_daily_" in path
        assert (tmp_path / path.split("/")[-1]).exists()


@pytest.mark.concept("KG-2.11")
class TestContributionExtraction:
    """Test internal extraction methods."""

    def setup_method(self):
        self.gen = ResearchArtifactGenerator(engine=None)

    def test_extract_contributions(self):
        text = (
            "We propose a novel framework for multi-agent coordination. "
            "We introduce a new attention mechanism for improved planning. "
            "The weather is nice today."
        )
        contributions = self.gen._extract_contributions(text)
        assert len(contributions) >= 2
        assert any("propose" in c.lower() for c in contributions)

    def test_extract_methods(self):
        text = "Our system uses a transformer architecture with RLHF and contrastive learning."
        methods = self.gen._extract_methods(text)
        assert "transformer" in methods
        assert "RLHF" in methods
        assert "contrastive" in methods

    def test_suggest_experiments(self):
        text = "We demonstrate improvement over baseline on standard benchmarks with ablation studies."
        experiments = self.gen._suggest_experiments(text, None)
        assert len(experiments) >= 1
