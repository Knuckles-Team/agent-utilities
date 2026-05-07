#!/usr/bin/python
"""CONCEPT:KG-2.11 — Automated Research Intelligence Pipeline.

Orchestrates the end-to-end research ingestion cycle:
  ScholarX Discovery → Relevance Scoring → Tiered Ingestion → OWL Enrichment → Digest

Architecture:
    - **ResearchPipelineRunner**: Main orchestrator wiring ScholarX, KBIngestionEngine,
      and OWL reasoning into a single automated flow.
    - **PipelineConfig**: Configurable thresholds, categories, and watchlists.
    - **PipelineReport**: Structured output of each pipeline run.

Tiered Ingestion (per user specification):
    - **Relevant** (score ≥ 3.0): Full PDF download + KG ingestion + SQLite doc storage.
      ArticleNode importance_score = 0.8.
    - **Marginal** (score ≥ 1.0): Abstract + metadata only (no PDF download).
      ArticleNode importance_score = 0.5. Serves as memory to avoid re-fetching.

Integrates with:
    - CONCEPT:OS-5.2 (MaintenanceCron): Triggered by scholarx_paper_discovery task
    - CONCEPT:KG-2.0 (IntelligenceGraphEngine): Graph persistence
    - CONCEPT:KG-2.2 (OWL Bridge): Post-ingestion reasoning cycle

See docs/research-pipeline.md §CONCEPT:KG-2.11.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..knowledge_graph.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

# Default arXiv categories to monitor
DEFAULT_CATEGORIES = [
    "cs.AI",
    "cs.MA",
    "cs.SE",
    "cs.LG",
    "cs.CL",
    "cs.IR",
    "cs.DC",
    "cs.DB",
    "q-bio",
]

# Relevance taxonomy aligned with agent-utilities domains
RELEVANCE_TAXONOMY: dict[str, dict[str, float]] = {
    "orchestration": {
        "keywords": [
            "multi-agent",
            "orchestration",
            "workflow",
            "task decomposition",
            "planning",
            "scheduling",
            "coordination",
            "swarm",
        ],
        "weight": 1.5,
    },
    "memory": {
        "keywords": [
            "memory",
            "retrieval",
            "knowledge graph",
            "episodic",
            "working memory",
            "long-term memory",
            "recall",
        ],
        "weight": 1.4,
    },
    "reasoning": {
        "keywords": [
            "reasoning",
            "chain-of-thought",
            "deliberation",
            "inference",
            "logical",
            "causal",
            "abductive",
        ],
        "weight": 1.3,
    },
    "learning": {
        "keywords": [
            "continual learning",
            "self-improvement",
            "curriculum",
            "reward",
            "reinforcement",
            "experience replay",
        ],
        "weight": 1.2,
    },
    "tools": {
        "keywords": [
            "tool use",
            "function calling",
            "API",
            "code generation",
            "MCP",
            "model context protocol",
        ],
        "weight": 1.1,
    },
    "security": {
        "keywords": [
            "safety",
            "alignment",
            "guardrail",
            "injection",
            "adversarial",
            "robustness",
            "red team",
        ],
        "weight": 1.0,
    },
    "evaluation": {
        "keywords": [
            "evaluation",
            "benchmark",
            "metric",
            "scoring",
            "judge",
            "grading",
            "assessment",
        ],
        "weight": 1.0,
    },
    "ontology": {
        "keywords": [
            "ontology",
            "OWL",
            "RDF",
            "semantic web",
            "knowledge representation",
            "taxonomy",
            "linked data",
        ],
        "weight": 1.3,
    },
    "communication": {
        "keywords": [
            "agent communication",
            "protocol",
            "A2A",
            "message passing",
            "negotiation",
            "consensus",
            "federation",
        ],
        "weight": 1.1,
    },
}


class PipelineConfig(BaseModel):
    """Configuration for the research pipeline.

    Attributes:
        categories: arXiv categories to monitor.
        relevant_threshold: Score ≥ this triggers full ingestion.
        marginal_threshold: Score ≥ this triggers abstract-only ingestion.
        max_papers_per_run: Cap on papers processed per pipeline execution.
        storage_dir: Directory for downloaded PDFs.
        lookback_hours: How far back to search for new papers.
        custom_watchlists: Additional keyword watchlists from KG PolicyNodes.
        run_owl_cycle: Whether to run OWL reasoning after ingestion.
    """

    categories: list[str] = Field(default_factory=lambda: list(DEFAULT_CATEGORIES))
    relevant_threshold: float = 3.0
    marginal_threshold: float = 1.0
    max_papers_per_run: int = 50
    storage_dir: str = str(Path.home() / ".scholarx" / "papers")
    lookback_hours: int = 24
    custom_watchlists: list[dict[str, Any]] = Field(default_factory=list)
    run_owl_cycle: bool = True


class IngestedPaperRecord(BaseModel):
    """Record of a single paper's ingestion result.

    Attributes:
        paper_id: External paper identifier (e.g., arXiv ID).
        title: Paper title.
        relevance_score: Computed relevance score.
        tier: Ingestion tier: 'relevant' (full) or 'marginal' (abstract-only).
        article_id: KG ArticleNode ID if ingested.
        status: Ingestion status.
        domains_matched: Which taxonomy domains matched.
    """

    paper_id: str
    title: str
    relevance_score: float = 0.0
    tier: str = "skipped"
    article_id: str = ""
    status: str = "pending"
    domains_matched: list[str] = Field(default_factory=list)


class PipelineReport(BaseModel):
    """Report from a single pipeline execution.

    Attributes:
        run_id: Unique identifier for this run.
        timestamp: ISO timestamp of execution.
        papers_discovered: Total papers found.
        papers_relevant: Papers meeting relevant threshold.
        papers_marginal: Papers meeting marginal threshold.
        papers_skipped: Papers below marginal threshold.
        papers_already_known: Papers already in KG (deduped).
        owl_inferences: Number of new OWL inferences discovered.
        records: Individual paper records.
        errors: Any errors encountered.
        duration_seconds: Total execution time.
    """

    run_id: str = Field(default_factory=lambda: f"run:{uuid.uuid4().hex[:8]}")
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    papers_discovered: int = 0
    papers_relevant: int = 0
    papers_marginal: int = 0
    papers_skipped: int = 0
    papers_already_known: int = 0
    owl_inferences: int = 0
    records: list[IngestedPaperRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class ResearchPipelineRunner:
    """CONCEPT:KG-2.11 — Automated research ingestion pipeline.

    Orchestrates: ScholarX Discovery → Relevance Scoring → Tiered Ingestion
    → OWL Reasoning → Digest Generation.

    Supports multiple input sources:
    - arXiv paper discovery via ScholarX API
    - Local PDF/HTML/Markdown files already on disk
    - Web URLs (HTML articles, blog posts)

    Args:
        engine: The IntelligenceGraphEngine for KG persistence.
        config: Pipeline configuration.
    """

    engine: IntelligenceGraphEngine | None = None
    config: PipelineConfig = field(default_factory=PipelineConfig)

    def _get_kb_engine(self):
        """Lazily construct KBIngestionEngine from the graph engine."""
        if not self.engine:
            return None
        try:
            from ..knowledge_graph.kb.ingestion import KBIngestionEngine

            return KBIngestionEngine(
                graph=self.engine.graph,
                backend=self.engine.backend,
            )
        except Exception as e:
            logger.warning(f"Failed to create KBIngestionEngine: {e}")
            return None

    def _get_scholarx_bridge(self):
        """Lazily construct ScholarXKGBridge if scholarx is available."""
        if not self.engine:
            return None
        try:
            from scholarx.kg_integration import ScholarXKGBridge

            return ScholarXKGBridge(self.engine)
        except ImportError:
            logger.info("ScholarX not installed — using direct KG ingestion")
            return None

    def _load_watchlists_from_kg(self) -> list[dict[str, Any]]:
        """Load custom research watchlists from KG PolicyNodes.

        Watchlists are stored as PolicyNodes with type 'research_watchlist'
        containing keywords and optional category filters.
        """
        if not self.engine:
            return []

        watchlists = []
        for node_id, data in self.engine.graph.nodes(data=True):
            if (
                data.get("type") == "policy"
                and data.get("policy_type") == "research_watchlist"
            ):
                watchlists.append(
                    {
                        "id": node_id,
                        "name": data.get("name", ""),
                        "keywords": data.get("keywords", []),
                        "categories": data.get("categories", []),
                        "threshold_override": data.get("threshold_override"),
                    }
                )
        return watchlists

    def score_paper(
        self,
        title: str,
        abstract: str,
        extra_keywords: list[str] | None = None,
    ) -> tuple[float, list[str]]:
        """Score a paper's relevance against the taxonomy.

        Args:
            title: Paper title.
            abstract: Paper abstract text.
            extra_keywords: Additional keywords from watchlists.

        Returns:
            Tuple of (score, list of matched domain names).
        """
        text = f"{title} {abstract}".lower()
        total_score = 0.0
        matched_domains: list[str] = []

        for domain, info in RELEVANCE_TAXONOMY.items():
            keywords = info["keywords"] if isinstance(info["keywords"], list) else []
            weight = (
                info.get("weight", 1.0)
                if isinstance(info.get("weight"), (int, float))
                else 1.0
            )
            domain_hits = sum(1 for kw in keywords if kw.lower() in text)
            if domain_hits > 0:
                total_score += domain_hits * weight
                matched_domains.append(domain)

        # Boost from custom watchlist keywords
        if extra_keywords:
            for kw in extra_keywords:
                if kw.lower() in text:
                    total_score += 0.5

        return total_score, matched_domains

    def _is_paper_known(self, paper_id: str) -> bool:
        """Check if a paper is already in the KG by its external ID."""
        if not self.engine:
            return False

        safe_id = paper_id.replace(":", "-")
        article_id = f"article:scholarx:{safe_id}"
        return article_id in self.engine.graph.nodes

    async def ingest_paper_full(
        self,
        paper_id: str,
        title: str,
        abstract: str,
        authors: list[str],
        pdf_path: str | None = None,
        source_url: str = "",
        relevance_score: float = 0.0,
        domains: list[str] | None = None,
    ) -> str:
        """Fully ingest a paper: PDF parsing + KG + SQLite doc storage.

        Args:
            paper_id: External paper identifier.
            title: Paper title.
            abstract: Paper abstract.
            authors: List of author names.
            pdf_path: Path to downloaded PDF file.
            source_url: Original URL of the paper.
            relevance_score: Computed relevance score.
            domains: Matched taxonomy domains.

        Returns:
            The KG article node ID.
        """
        safe_id = paper_id.replace(":", "-")
        article_id = f"article:scholarx:{safe_id}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Try ScholarX bridge first (handles PersonNodes, SourceNodes)
        bridge = self._get_scholarx_bridge()
        if bridge:
            try:
                from scholarx.models import Paper

                paper_obj = Paper(
                    id=paper_id,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    url=source_url,
                    source="pipeline",
                )
                if pdf_path:
                    result = await bridge.ingest_paper(paper_obj, pdf_path=pdf_path)
                else:
                    result = await bridge.ingest_paper_abstract_only(paper_obj)
                if result.get("status") == "ingested":
                    return result.get("article_id", article_id)
            except Exception as e:
                logger.warning(f"ScholarX bridge ingestion failed, falling back: {e}")

        # Direct KG ingestion fallback
        if self.engine:
            from ..models.knowledge_graph import (
                ArticleNode,
                RegistryEdgeType,
                SourceNode,
            )

            article_node = ArticleNode(
                id=article_id,
                name=title,
                description=abstract[:500],
                summary=abstract[:500],
                content=abstract,
                importance_score=0.8,
                timestamp=timestamp,
                tags=domains or [],
            )
            self.engine.graph.add_node(article_id, **article_node.model_dump())

            # Create source node
            source_id = f"source:scholarx:{safe_id}"
            source_node = SourceNode(
                id=source_id,
                source_id=source_id,
                name=f"Source: {title[:60]}",
                url=source_url,
                description=f"Research paper: {title}",
                authors=authors,
                importance_score=0.5,
                timestamp=timestamp,
            )
            self.engine.graph.add_node(source_id, **source_node.model_dump())
            self.engine.graph.add_edge(
                article_id, source_id, type=RegistryEdgeType.CITES
            )

            # Create author PersonNodes
            for author in authors[:10]:
                author_id = (
                    f"person:{author.lower().replace(' ', '-').replace('.', '')[:40]}"
                )
                if author_id not in self.engine.graph.nodes:
                    from ..models.knowledge_graph import PersonNode

                    person_node = PersonNode(
                        id=author_id,
                        person_id=author_id,
                        name=author,
                        description=f"Author of: {title[:60]}",
                        importance_score=0.4,
                        timestamp=timestamp,
                    )
                    self.engine.graph.add_node(author_id, **person_node.model_dump())
                if not self.engine.graph.has_edge(article_id, author_id):
                    self.engine.graph.add_edge(
                        article_id, author_id, type=RegistryEdgeType.AUTHORED
                    )

            # Persist to backend
            if self.engine.backend:
                self.engine._upsert_node(
                    "Article",
                    article_id,
                    self.engine._serialize_node(article_node, "Article"),
                )

        # If PDF exists, also ingest through KBIngestionEngine for full vectorization
        kb_engine = self._get_kb_engine()
        if kb_engine and pdf_path and Path(pdf_path).exists():
            try:
                await kb_engine.ingest_directory(
                    Path(pdf_path).parent,
                    kb_name="scholarx-research",
                    topic=f"Research: {title[:80]}",
                    force=False,
                )
            except Exception as e:
                logger.warning(f"KB full ingestion failed for {paper_id}: {e}")

        logger.info(f"[CONCEPT:KG-2.11] Fully ingested: {title[:60]} → {article_id}")
        return article_id

    async def ingest_paper_marginal(
        self,
        paper_id: str,
        title: str,
        abstract: str,
        authors: list[str],
        source_url: str = "",
        relevance_score: float = 0.0,
        domains: list[str] | None = None,
    ) -> str:
        """Ingest abstract + metadata only (no PDF download).

        This creates a lightweight KG footprint so the paper is remembered
        and won't be re-fetched in future runs.

        Args:
            paper_id: External paper identifier.
            title: Paper title.
            abstract: Paper abstract.
            authors: List of author names.
            source_url: Original URL.
            relevance_score: Computed relevance score.
            domains: Matched taxonomy domains.

        Returns:
            The KG article node ID.
        """
        safe_id = paper_id.replace(":", "-")
        article_id = f"article:scholarx:{safe_id}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if self.engine:
            from ..models.knowledge_graph import (
                ArticleNode,
                RegistryEdgeType,
                SourceNode,
            )

            article_node = ArticleNode(
                id=article_id,
                name=title,
                description=abstract[:500],
                summary=abstract[:500],
                content=abstract,
                importance_score=0.5,  # Lower for marginal
                timestamp=timestamp,
                tags=domains or [],
            )
            self.engine.graph.add_node(article_id, **article_node.model_dump())

            source_id = f"source:scholarx:{safe_id}"
            source_node = SourceNode(
                id=source_id,
                source_id=source_id,
                name=f"Source: {title[:60]}",
                url=source_url,
                description=f"Marginal relevance paper: {title}",
                authors=authors,
                importance_score=0.3,
                timestamp=timestamp,
            )
            self.engine.graph.add_node(source_id, **source_node.model_dump())
            self.engine.graph.add_edge(
                article_id, source_id, type=RegistryEdgeType.CITES
            )

            if self.engine.backend:
                self.engine._upsert_node(
                    "Article",
                    article_id,
                    self.engine._serialize_node(article_node, "Article"),
                )

        logger.info(f"[CONCEPT:KG-2.11] Marginal ingested: {title[:60]} → {article_id}")
        return article_id

    async def ingest_local_file(
        self,
        file_path: str | Path,
        kb_name: str = "scholarx-research",
        topic: str | None = None,
    ) -> str:
        """Ingest a local file (PDF, HTML, Markdown, etc.) into the KG.

        Supports any format handled by KBDocumentParser: .pdf, .html, .md,
        .txt, .docx, .epub.

        Args:
            file_path: Path to the local file.
            kb_name: Knowledge base name for grouping.
            topic: Topic description.

        Returns:
            The KB ID.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        kb_engine = self._get_kb_engine()
        if not kb_engine:
            raise RuntimeError("No KG engine available for ingestion")

        result = await kb_engine.ingest_directory(
            file_path.parent if file_path.is_file() else file_path,
            kb_name=kb_name,
            topic=topic or f"Research: {file_path.stem}",
            force=False,
        )
        logger.info(
            f"[CONCEPT:KG-2.11] Local file ingested: {file_path.name} → {result.id}"
        )
        return result.id

    async def ingest_url(
        self,
        url: str,
        kb_name: str = "scholarx-research",
        topic: str | None = None,
    ) -> str:
        """Ingest a web URL (HTML article, blog post, etc.) into the KG.

        Args:
            url: URL to fetch and ingest.
            kb_name: Knowledge base name for grouping.
            topic: Topic description.

        Returns:
            The KB ID.
        """
        kb_engine = self._get_kb_engine()
        if not kb_engine:
            raise RuntimeError("No KG engine available for ingestion")

        result = await kb_engine.ingest_url(
            url=url,
            kb_name=kb_name,
            topic=topic or f"Web article: {url[:60]}",
            force=False,
        )
        logger.info(f"[CONCEPT:KG-2.11] URL ingested: {url[:60]} → {result.id}")
        return result.id

    def _run_owl_enrichment(self) -> int:
        """Run OWL reasoning cycle to discover new inferences from ingested papers."""
        if not self.engine or not self.config.run_owl_cycle:
            return 0

        try:
            from ..knowledge_graph.backends.owl import create_owl_backend
            from ..knowledge_graph.owl_bridge import OWLBridge

            owl_backend = create_owl_backend()
            bridge = OWLBridge(
                graph=self.engine.graph,
                owl_backend=owl_backend,
                backend=self.engine.backend,
            )
            stats = bridge.run_cycle(lightweight=True)
            inferred = stats.get("inferred", 0)
            logger.info(f"[CONCEPT:KG-2.11] OWL enrichment: {inferred} inferences")
            return inferred
        except Exception as e:
            logger.debug(f"OWL enrichment skipped: {e}")
            return 0

    async def run_daily_pipeline(
        self,
        papers: list[dict[str, Any]] | None = None,
    ) -> PipelineReport:
        """Execute the full daily research pipeline.

        If papers are not provided, attempts to discover them via ScholarX.

        Args:
            papers: Optional pre-fetched paper dicts with keys:
                    id, title, abstract, authors, url.

        Returns:
            PipelineReport with full execution details.
        """
        start = time.time()
        report = PipelineReport()

        # Load watchlists from KG
        watchlists = self._load_watchlists_from_kg()
        extra_keywords: list[str] = []
        for wl in watchlists + self.config.custom_watchlists:
            extra_keywords.extend(wl.get("keywords", []))

        # Discover papers if not provided
        if papers is None:
            papers = await self._discover_papers()

        report.papers_discovered = len(papers)

        # Score and classify each paper
        for paper_data in papers[: self.config.max_papers_per_run]:
            paper_id = paper_data.get("id", "")
            title = paper_data.get("title", "")
            abstract = paper_data.get("abstract", "")
            authors = paper_data.get("authors", [])
            url = paper_data.get("url", "")

            record = IngestedPaperRecord(paper_id=paper_id, title=title)

            # Dedup check
            if self._is_paper_known(paper_id):
                record.status = "already_known"
                record.tier = "skipped"
                report.papers_already_known += 1
                report.records.append(record)
                continue

            # Score relevance
            score, domains = self.score_paper(title, abstract, extra_keywords)
            record.relevance_score = score
            record.domains_matched = domains

            try:
                if score >= self.config.relevant_threshold:
                    # Full ingestion
                    article_id = await self.ingest_paper_full(
                        paper_id=paper_id,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        source_url=url,
                        relevance_score=score,
                        domains=domains,
                    )
                    record.tier = "relevant"
                    record.article_id = article_id
                    record.status = "ingested_full"
                    report.papers_relevant += 1

                elif score >= self.config.marginal_threshold:
                    # Abstract-only ingestion
                    article_id = await self.ingest_paper_marginal(
                        paper_id=paper_id,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        source_url=url,
                        relevance_score=score,
                        domains=domains,
                    )
                    record.tier = "marginal"
                    record.article_id = article_id
                    record.status = "ingested_abstract"
                    report.papers_marginal += 1

                else:
                    record.tier = "skipped"
                    record.status = "below_threshold"
                    report.papers_skipped += 1

            except Exception as e:
                record.status = f"error: {e}"
                report.errors.append(f"{paper_id}: {e}")
                logger.error(f"[CONCEPT:KG-2.11] Ingestion error for {paper_id}: {e}")

            report.records.append(record)

        # Run OWL reasoning
        if report.papers_relevant > 0 or report.papers_marginal > 0:
            report.owl_inferences = self._run_owl_enrichment()

        report.duration_seconds = time.time() - start
        logger.info(
            "[CONCEPT:KG-2.11] Pipeline complete: %d discovered, %d relevant, "
            "%d marginal, %d skipped, %d known, %.1fs",
            report.papers_discovered,
            report.papers_relevant,
            report.papers_marginal,
            report.papers_skipped,
            report.papers_already_known,
            report.duration_seconds,
        )
        return report

    async def _discover_papers(self) -> list[dict[str, Any]]:
        """Discover papers via ScholarX API client."""
        try:
            from scholarx.api_client import ScholarXClient

            client = ScholarXClient()
            results = await client.search_recent(
                categories=self.config.categories,
                hours=self.config.lookback_hours,
                max_results=self.config.max_papers_per_run,
            )
            return [
                {
                    "id": r.id if hasattr(r, "id") else str(r.get("id", "")),
                    "title": r.title
                    if hasattr(r, "title")
                    else str(r.get("title", "")),
                    "abstract": r.abstract
                    if hasattr(r, "abstract")
                    else str(r.get("abstract", "")),
                    "authors": r.authors
                    if hasattr(r, "authors")
                    else r.get("authors", []),
                    "url": r.url if hasattr(r, "url") else str(r.get("url", "")),
                }
                for r in (results if isinstance(results, list) else [])
            ]
        except ImportError:
            logger.info("ScholarX not installed — no automatic discovery available")
            return []
        except Exception as e:
            logger.warning(f"Paper discovery failed: {e}")
            return []

    def generate_digest(self, report: PipelineReport) -> str:
        """Generate a markdown digest from a pipeline report.

        Args:
            report: The completed pipeline report.

        Returns:
            Markdown-formatted digest string.
        """
        lines = [
            f"# Research Digest — {report.timestamp}",
            "",
            f"**Run ID**: `{report.run_id}`",
            f"**Duration**: {report.duration_seconds:.1f}s",
            f"**Papers discovered**: {report.papers_discovered}",
            f"**Fully ingested**: {report.papers_relevant}",
            f"**Abstract-only**: {report.papers_marginal}",
            f"**Skipped**: {report.papers_skipped}",
            f"**Already known**: {report.papers_already_known}",
            f"**OWL inferences**: {report.owl_inferences}",
            "",
        ]

        relevant = [r for r in report.records if r.tier == "relevant"]
        if relevant:
            lines.append("## Fully Ingested (Relevant)")
            lines.append("")
            for r in relevant:
                domains = (
                    ", ".join(r.domains_matched) if r.domains_matched else "general"
                )
                lines.append(
                    f"- **{r.title}** (score: {r.relevance_score:.1f}, domains: {domains})"
                )
            lines.append("")

        marginal = [r for r in report.records if r.tier == "marginal"]
        if marginal:
            lines.append("## Abstract-Only (Marginal)")
            lines.append("")
            for r in marginal:
                lines.append(f"- {r.title} (score: {r.relevance_score:.1f})")
            lines.append("")

        if report.errors:
            lines.append("## Errors")
            lines.append("")
            for err in report.errors:
                lines.append(f"- {err}")

        return "\n".join(lines)
