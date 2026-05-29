"""Tests for CONCEPT:KG-2.6 — Research Pipeline Runner.

Validates:
- PipelineConfig defaults and customization
- Relevance scoring against the 9-domain taxonomy
- Tiered ingestion (relevant, marginal, skipped)
- Deduplication via _is_paper_known
- Digest generation
- Pipeline report structure
"""

from unittest.mock import MagicMock

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
import pytest

from agent_utilities.automation.research_pipeline import (
    DEFAULT_CATEGORIES,
    RELEVANCE_TAXONOMY,
    IngestedPaperRecord,
    PipelineConfig,
    PipelineReport,
    ResearchPipelineRunner,
)


@pytest.mark.concept("KG-2.11")
class TestPipelineConfig:
    """Test PipelineConfig defaults and validation."""

    def test_default_categories(self):
        cfg = PipelineConfig()
        assert "cs.AI" in cfg.categories
        assert "cs.MA" in cfg.categories
        assert len(cfg.categories) == len(DEFAULT_CATEGORIES)

    def test_default_thresholds(self):
        cfg = PipelineConfig()
        assert cfg.relevant_threshold == 3.0
        assert cfg.marginal_threshold == 1.0

    def test_custom_config(self):
        cfg = PipelineConfig(
            categories=["cs.AI"],
            relevant_threshold=5.0,
            marginal_threshold=2.0,
            max_papers_per_run=10,
        )
        assert cfg.categories == ["cs.AI"]
        assert cfg.relevant_threshold == 5.0
        assert cfg.max_papers_per_run == 10

    def test_storage_dir_default(self):
        cfg = PipelineConfig()
        assert ".scholarx" in cfg.storage_dir
        assert "papers" in cfg.storage_dir

    def test_lookback_hours(self):
        cfg = PipelineConfig(lookback_hours=48)
        assert cfg.lookback_hours == 48


@pytest.mark.concept("KG-2.11")
class TestRelevanceScoring:
    """Test paper relevance scoring."""

    def setup_method(self):
        self.runner = ResearchPipelineRunner()

    def test_high_relevance_multi_agent(self):
        score, domains = self.runner.score_paper(
            title="Multi-Agent Orchestration for Knowledge Graph Construction",
            abstract="We propose a novel multi-agent system with knowledge graph "
            "integration and memory-based retrieval for planning tasks.",
        )
        assert score >= 3.0
        assert "orchestration" in domains
        assert "memory" in domains

    def test_low_relevance_unrelated(self):
        score, domains = self.runner.score_paper(
            title="Protein Folding via Monte Carlo Simulation",
            abstract="We study the thermodynamics of protein folding using "
            "traditional Monte Carlo methods on amino acid sequences.",
        )
        assert score < 1.0

    def test_marginal_relevance(self):
        score, domains = self.runner.score_paper(
            title="Efficient Inference for Large Language Models",
            abstract="We present a method for faster inference using reward "
            "shaping techniques.",
        )
        assert score >= 1.0
        assert len(domains) >= 1

    def test_extra_keywords_boost(self):
        score_base, _ = self.runner.score_paper(
            title="A new framework for data processing",
            abstract="Processes structured data efficiently.",
        )
        score_boosted, _ = self.runner.score_paper(
            title="A new framework for data processing",
            abstract="Processes structured data efficiently.",
            extra_keywords=["data processing", "framework"],
        )
        assert score_boosted >= score_base

    def test_taxonomy_domains_present(self):
        assert "orchestration" in RELEVANCE_TAXONOMY
        assert "memory" in RELEVANCE_TAXONOMY
        assert "reasoning" in RELEVANCE_TAXONOMY
        assert "learning" in RELEVANCE_TAXONOMY
        assert "tools" in RELEVANCE_TAXONOMY
        assert "security" in RELEVANCE_TAXONOMY
        assert "evaluation" in RELEVANCE_TAXONOMY
        assert "ontology" in RELEVANCE_TAXONOMY
        assert "communication" in RELEVANCE_TAXONOMY


@pytest.mark.concept("KG-2.11")
class TestPaperIngestion:
    """Test paper ingestion methods."""

    def setup_method(self):
        self.graph = GraphComputeEngine(backend_type="rust")
        self.engine = MagicMock()
        self.engine.graph = self.graph
        self.engine.backend = None
        self.runner = ResearchPipelineRunner(engine=self.engine)

    @pytest.mark.asyncio
    async def test_ingest_paper_full(self):
        article_id = await self.runner.ingest_paper_full(
            paper_id="2406.12345",
            title="Test Paper Full",
            abstract="Abstract of the paper",
            authors=["Author One", "Author Two"],
            source_url="https://arxiv.org/abs/2406.12345",
            relevance_score=4.5,
            domains=["orchestration", "memory"],
        )
        assert article_id == "article:scholarx:2406.12345"
        assert article_id in self.graph.nodes

    @pytest.mark.asyncio
    async def test_ingest_paper_marginal(self):
        article_id = await self.runner.ingest_paper_marginal(
            paper_id="2406.99999",
            title="Test Paper Marginal",
            abstract="Some abstract",
            authors=["Author Three"],
        )
        assert article_id == "article:scholarx:2406.99999"
        assert article_id in self.graph.nodes
        # Marginal papers have lower importance
        node_data = self.graph.nodes[article_id]
        assert node_data.get("importance_score", 1.0) <= 0.5

    @pytest.mark.asyncio
    async def test_deduplication(self):
        await self.runner.ingest_paper_full(
            paper_id="2406.12345",
            title="First Time",
            abstract="First ingestion",
            authors=["Author"],
        )
        assert self.runner._is_paper_known("2406.12345")
        assert not self.runner._is_paper_known("2406.99999")


@pytest.mark.concept("KG-2.11")
class TestPipelineExecution:
    """Test full pipeline execution."""

    def setup_method(self):
        self.graph = GraphComputeEngine(backend_type="rust")
        self.engine = MagicMock()
        self.engine.graph = self.graph
        self.engine.backend = None
        self.runner = ResearchPipelineRunner(
            engine=self.engine,
            config=PipelineConfig(run_owl_cycle=False),
        )

    @pytest.mark.asyncio
    async def test_run_with_preloaded_papers(self):
        papers = [
            {
                "id": "2406.00001",
                "title": "Multi-Agent Orchestration with Knowledge Graph Memory",
                "abstract": "We propose a multi-agent system with knowledge graph "
                "and memory retrieval for planning coordination tasks.",
                "authors": ["Alice"],
                "url": "https://arxiv.org/abs/2406.00001",
            },
            {
                "id": "2406.00002",
                "title": "Improving Evaluation Benchmarks",
                "abstract": "A new evaluation benchmark for tool use assessment.",
                "authors": ["Bob"],
                "url": "https://arxiv.org/abs/2406.00002",
            },
            {
                "id": "2406.00003",
                "title": "Gardening Tips for Spring",
                "abstract": "How to plant roses and tomatoes in your backyard.",
                "authors": ["Charlie"],
                "url": "",
            },
        ]

        report = await self.runner.run_daily_pipeline(papers=papers)
        assert isinstance(report, PipelineReport)
        assert report.papers_discovered == 3
        assert report.papers_relevant >= 1
        assert report.papers_skipped >= 1
        assert len(report.records) == 3

    @pytest.mark.asyncio
    async def test_empty_pipeline(self):
        report = await self.runner.run_daily_pipeline(papers=[])
        assert report.papers_discovered == 0
        assert report.papers_relevant == 0

    @pytest.mark.asyncio
    async def test_dedup_in_pipeline(self):
        # Pre-seed a paper
        await self.runner.ingest_paper_full(
            paper_id="2406.00001",
            title="Already Known",
            abstract="Already in KG",
            authors=["Known Author"],
        )

        papers = [
            {
                "id": "2406.00001",
                "title": "Already Known",
                "abstract": "Already in KG",
                "authors": [],
                "url": "",
            },
        ]
        report = await self.runner.run_daily_pipeline(papers=papers)
        assert report.papers_already_known == 1


@pytest.mark.concept("KG-2.11")
class TestDigestGeneration:
    """Test digest markdown generation."""

    def test_generate_digest(self):
        runner = ResearchPipelineRunner()
        report = PipelineReport(
            papers_discovered=5,
            papers_relevant=2,
            papers_marginal=1,
            papers_skipped=2,
            records=[
                IngestedPaperRecord(
                    paper_id="1",
                    title="Paper A",
                    tier="relevant",
                    relevance_score=4.5,
                    domains_matched=["orchestration"],
                ),
                IngestedPaperRecord(
                    paper_id="2",
                    title="Paper B",
                    tier="marginal",
                    relevance_score=1.5,
                ),
                IngestedPaperRecord(
                    paper_id="3",
                    title="Paper C",
                    tier="skipped",
                    relevance_score=0.2,
                ),
            ],
        )

        digest = runner.generate_digest(report)
        assert "Research Digest" in digest
        assert "Paper A" in digest
        assert "Fully Ingested" in digest
        assert "Abstract-Only" in digest

    def test_pipeline_report_model(self):
        report = PipelineReport()
        assert report.run_id.startswith("run:")
        assert report.papers_discovered == 0
        assert isinstance(report.records, list)


@pytest.mark.concept("KG-2.11")
class TestWatchlists:
    """Test KG watchlist loading."""

    def test_load_empty_watchlists(self):
        runner = ResearchPipelineRunner()
        watchlists = runner._load_watchlists_from_kg()
        assert watchlists == []

    def test_load_watchlists_from_kg(self):
        graph = GraphComputeEngine(backend_type="rust")
        graph.add_node(
            "policy:wl1",
            type="policy",
            policy_type="research_watchlist",
            name="Agent Safety",
            keywords=["alignment", "safety"],
        )

        engine = MagicMock()
        engine.graph = graph
        runner = ResearchPipelineRunner(engine=engine)
        watchlists = runner._load_watchlists_from_kg()
        assert len(watchlists) == 1
        assert "alignment" in watchlists[0]["keywords"]
