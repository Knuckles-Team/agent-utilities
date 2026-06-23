#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:KG-2.6 — Automated Research Intelligence Pipeline.

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

See docs/pillars/2_epistemic_knowledge_graph.md §CONCEPT:KG-2.6
"""


import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

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
    # Quantitative finance (CONCEPT:EE-037) — microstructure, trading, pricing.
    "q-fin.TR",
    "q-fin.PM",
    "q-fin.ST",
    "q-fin.CP",
    "q-fin.RM",
]

# Relevance taxonomy aligned with agent-utilities domains
RELEVANCE_TAXONOMY: dict[str, dict[str, Any]] = {
    "trading": {
        "keywords": [
            "market microstructure",
            "order flow",
            "limit order book",
            "statistical arbitrage",
            "market making",
            "optimal execution",
            "realized volatility",
            "kelly criterion",
            "regime switching",
            "deflated sharpe",
            "backtest overfit",
            "alpha signal",
            "price impact",
        ],
        "weight": 1.4,
    },
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


# World-model relevance taxonomy (CONCEPT:KG-2.116) — the news/finance/tech sibling
# of RELEVANCE_TAXONOMY, used by the FreshRSS world-model gate (WorldModelPipelineRunner)
# instead of the research-paper profile. Keyed by domain → {keywords, weight}, scored by
# the SAME ``score_text`` function. The ``companies`` set is seeded here and augmented at
# runtime from existing KG OrganizationNodes so ingestion biases toward what we already model.
WORLD_MODEL_TAXONOMY: dict[str, dict[str, Any]] = {
    "macro_economy": {
        "keywords": [
            "inflation",
            "interest rate",
            "federal reserve",
            "central bank",
            "gdp",
            "recession",
            "unemployment",
            "tariff",
            "supply chain",
            "monetary policy",
        ],
        "weight": 1.3,
    },
    "markets_finance": {
        "keywords": [
            "earnings",
            "stock",
            "equities",
            "bond yield",
            "ipo",
            "merger",
            "acquisition",
            "guidance",
            "dividend",
            "valuation",
            "crypto",
            "bitcoin",
            "commodities",
        ],
        "weight": 1.4,
    },
    "technology": {
        "keywords": [
            "artificial intelligence",
            "llm",
            "semiconductor",
            "chip",
            "gpu",
            "cloud",
            "data center",
            "open source",
            "model release",
            "product launch",
            "robotics",
        ],
        "weight": 1.3,
    },
    "cybersecurity": {
        "keywords": [
            "breach",
            "vulnerability",
            "cve",
            "ransomware",
            "exploit",
            "zero-day",
            "threat actor",
            "data leak",
            "malware",
        ],
        "weight": 1.3,
    },
    "geopolitics": {
        "keywords": [
            "sanctions",
            "election",
            "treaty",
            "conflict",
            "regulation",
            "antitrust",
            "trade war",
            "export control",
            "diplomacy",
        ],
        "weight": 1.1,
    },
    "science": {
        "keywords": [
            "breakthrough",
            "space",
            "nasa",
            "energy",
            "fusion",
            "biotech",
            "climate",
            "vaccine",
            "quantum",
        ],
        "weight": 1.1,
    },
    "companies": {
        # seeded; live set augmented from KG OrganizationNodes at runtime.
        "keywords": [
            "nvidia",
            "openai",
            "anthropic",
            "microsoft",
            "apple",
            "google",
            "amazon",
            "meta",
            "tesla",
        ],
        "weight": 1.2,
    },
}


# Named relevance profiles (CONCEPT:KG-2.116). One scorer, two taxonomies: the research
# pipeline uses ``research``; the FreshRSS world-model gate uses ``world_model``.
RELEVANCE_PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "research": RELEVANCE_TAXONOMY,
    "world_model": WORLD_MODEL_TAXONOMY,
}


def score_text(
    title: str,
    abstract: str,
    taxonomy: dict[str, dict[str, Any]],
    extra_keywords: list[str] | None = None,
) -> tuple[float, list[str]]:
    """Profile-agnostic keyword relevance score (CONCEPT:KG-2.116).

    The shared scorer behind both ``ResearchPipelineRunner.score_paper`` (research
    taxonomy) and ``WorldModelPipelineRunner`` (world-model taxonomy): sum of
    ``domain_hits * weight`` over the lowercased ``title + abstract``, plus +0.5 per
    matched ``extra_keywords`` entry. Returns ``(score, matched_domains)``.
    """
    text = f"{title} {abstract}".lower()
    total_score = 0.0
    matched_domains: list[str] = []

    for domain, info in taxonomy.items():
        keywords = info["keywords"] if isinstance(info["keywords"], list) else []
        weight = (
            info.get("weight", 1.0)
            if isinstance(info.get("weight"), int | float)
            else 1.0
        )
        domain_hits = sum(1 for kw in keywords if kw.lower() in text)
        if domain_hits > 0:
            total_score += domain_hits * weight
            matched_domains.append(domain)

    if extra_keywords:
        for kw in extra_keywords:
            if kw.lower() in text:
                total_score += 0.5

    return total_score, matched_domains


def concept_novelty(engine: Any, text: str, *, holder: Any = None) -> float | None:
    """Matcher-driven novelty in [0,1] (1.0 = fully novel); ``None`` if unknown.

    A cheap (no-LLM) ``ConceptMatcher`` cosine probe of ``text`` against the ecosystem
    ``Concept`` registry — the shared "relevant-enough-vs-existing-KG" signal behind both
    the research pipeline (CONCEPT:KG-2.75) and the world-model gate (CONCEPT:KG-2.116).
    When a ``holder`` object is given, the concept index + embed fn are cached on it
    (``_novelty_index`` / ``_novelty_embed_fn``) for the run. Best-effort: returns ``None``
    (no demotion) when no engine / no concepts / embedder is unavailable.
    """
    try:
        from agent_utilities.knowledge_graph.assimilation.concept_matcher import (
            ConceptMatcher,
            _build_concept_index,
        )
        from agent_utilities.knowledge_graph.assimilation.gap_analysis import (
            _CONCEPT_TYPES,
            _collect_rich,
        )

        if engine is None:
            return None
        idx = getattr(holder, "_novelty_index", None) if holder is not None else None
        if idx is None:
            concepts = _collect_rich(engine, _CONCEPT_TYPES)
            idx = _build_concept_index(concepts) if concepts else ()
            if holder is not None:
                holder._novelty_index = idx  # cache for the run
        if not idx or not idx[1]:  # no concept vectors
            return None
        concept_by_key, concept_vecs, concept_text = idx
        embed_fn = (
            getattr(holder, "_novelty_embed_fn", None) if holder is not None else None
        )
        if embed_fn is None:
            from agent_utilities.knowledge_graph.enrichment.semantic import (
                make_embed_fn,
            )

            embed_fn = make_embed_fn()
            if holder is not None:
                holder._novelty_embed_fn = embed_fn
        vec = embed_fn([text])[0]
        if not vec or len(vec) < 2:
            return None
        fm = ConceptMatcher(use_llm=False).match_feature(
            "feature:probe",
            {"name": text[:120], "summary": text, "embedding": vec},
            concept_by_key=concept_by_key,
            concept_vecs=concept_vecs,
            concept_text=concept_text,
            feature_vec=vec,
        )
        return fm.novelty_score
    except Exception:  # noqa: BLE001 — best-effort; never block discovery
        return None


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
    novelty: float | None = None  # matcher novelty (1.0 = novel); None if unknown
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
    """CONCEPT:KG-2.6 — Automated research ingestion pipeline.

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
        return score_text(title, abstract, RELEVANCE_TAXONOMY, extra_keywords)

    def _paper_novelty(self, title: str, abstract: str) -> float | None:
        """Matcher-driven novelty in [0,1] (1.0 = fully novel); ``None`` if unknown.

        A cheap (no-LLM) ConceptMatcher cosine probe of the paper against the
        ecosystem ``Concept`` registry. Drives tiering: a paper we ALREADY have
        (low novelty / already-covered) is demoted to memory-only ingestion so we
        don't spend a full PDF ingest re-acquiring a built capability. Best-effort:
        returns ``None`` (no demotion) when no engine / no concepts / embedder is
        unavailable, so discovery never blocks on it. The authoritative
        covered/related verdict is still written later by the assimilate matcher.
        (CONCEPT:KG-2.75)
        """
        return concept_novelty(self.engine, f"{title} — {abstract}", holder=self)

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

        # When invoked with only an id + a PDF (a research cohort, CONCEPT:KG-2.194),
        # title/abstract arrive empty — extract them from the PDF so the Article node
        # carries real TEXT. ConceptMatcher builds its match text from name/abstract/
        # content and embeds it on-the-fly; an empty node has no text to recall against
        # and is scored as spuriously "novel". So fill content from the PDF here.
        if pdf_path and not (title.strip() or abstract.strip()):
            try:
                from ..knowledge_graph.extraction.readers import read_any

                full = (read_any(str(pdf_path)) or "").strip()
                if full:
                    if not title.strip():
                        first = next(
                            (ln.strip() for ln in full.splitlines() if ln.strip()), ""
                        )
                        title = first[:300] or paper_id
                    if not abstract.strip():
                        abstract = full[:6000]
            except Exception as e:  # noqa: BLE001 — best-effort; fall back to id-only
                logger.warning(f"cohort PDF text extraction failed for {paper_id}: {e}")

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
                    source="arxiv",
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

        logger.info(f"[CONCEPT:KG-2.6] Fully ingested: {title[:60]} → {article_id}")
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

        logger.info(f"[CONCEPT:KG-2.6] Marginal ingested: {title[:60]} → {article_id}")
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
            f"[CONCEPT:KG-2.6] Local file ingested: {file_path.name} → {result.id}"
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
        logger.info(f"[CONCEPT:KG-2.6] URL ingested: {url[:60]} → {result.id}")
        return result.id

    def _run_owl_enrichment(self) -> int:
        """Run OWL reasoning cycle to discover new inferences from ingested papers."""
        if not self.engine or not self.config.run_owl_cycle:
            return 0

        try:
            from ..knowledge_graph.backends.owl import create_owl_backend
            from ..knowledge_graph.core.owl_bridge import OWLBridge

            owl_backend = create_owl_backend()
            bridge = OWLBridge(
                graph=self.engine.graph,
                owl_backend=owl_backend,
                backend=self.engine.backend,
            )
            stats = bridge.run_cycle(lightweight=True)
            inferred = stats.get("inferred", 0)
            logger.info(f"[CONCEPT:KG-2.6] OWL enrichment: {inferred} inferences")
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

        # Score + ingest every paper CONCURRENTLY (CONCEPT:KG-2.174 — generalized
        # cross-lane parallelization). Each paper's full/abstract ingest (the heavy
        # PDF + LLM work) is independent, so they fan out under a bounded semaphore
        # sized to the ingest worker count; vLLM batches server-side, turning an
        # N-paper run from N sequential ingests into ~N/concurrency. The per-paper
        # helper is pure (returns its record, mutates no shared state), so counters
        # are tallied race-free after the gather.
        import asyncio

        from ..knowledge_graph.core.engine_tasks import compute_ingest_worker_count

        sem = asyncio.Semaphore(max(1, compute_ingest_worker_count()))

        async def _process_one(paper_data: dict[str, Any]) -> IngestedPaperRecord:
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
                return record

            # Score relevance (cheap keyword prefilter) then let the ConceptMatcher
            # drive the tier by NOVELTY: a paper we already have (low novelty) is
            # demoted from a full PDF ingest to memory-only. (CONCEPT:KG-2.75)
            score, domains = self.score_paper(title, abstract, extra_keywords)
            novelty = self._paper_novelty(title, abstract)
            record.relevance_score = score
            record.domains_matched = domains
            if novelty is not None:
                record.novelty = novelty
                if novelty < 0.25 and score >= self.config.relevant_threshold:
                    score = self.config.marginal_threshold  # already-built → memory

            try:
                if score >= self.config.relevant_threshold:
                    async with sem:  # bound the heavy full-PDF + LLM ingest
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
                elif score >= self.config.marginal_threshold:
                    async with sem:
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
                else:
                    record.tier = "skipped"
                    record.status = "below_threshold"
            except Exception as e:
                record.status = f"error: {e}"
                logger.error(f"[CONCEPT:KG-2.6] Ingestion error for {paper_id}: {e}")
            return record

        records = await asyncio.gather(
            *(_process_one(p) for p in papers[: self.config.max_papers_per_run])
        )

        # Tally counters race-free from the returned records (the helper is pure).
        for record in records:
            report.records.append(record)
            status = record.status
            if status == "already_known":
                report.papers_already_known += 1
            elif status == "ingested_full":
                report.papers_relevant += 1
            elif status == "ingested_abstract":
                report.papers_marginal += 1
            elif status == "below_threshold":
                report.papers_skipped += 1
            elif status.startswith("error: "):
                report.errors.append(f"{record.paper_id}: {status[len('error: ') :]}")

        # Run OWL reasoning
        if report.papers_relevant > 0 or report.papers_marginal > 0:
            report.owl_inferences = self._run_owl_enrichment()

        report.duration_seconds = time.time() - start
        logger.info(
            "[CONCEPT:KG-2.6] Pipeline complete: %d discovered, %d relevant, "
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
