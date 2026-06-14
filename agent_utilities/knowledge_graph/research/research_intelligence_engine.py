"""Research Intelligence Engine — Synthesized Research Facade.

CONCEPT:KG-2.6/2.33/2.39 — Research Intelligence Engine

Provides a single entry point for all research intelligence capabilities:
- Automated paper discovery and ingestion pipeline (KG-2.7)
- Citation graph traversal with session management (KG-2.33)
- Orchestrated research-to-KG cycles (KG-2.39)

All sub-modules remain as separate files; this engine provides a unified
API with lazy initialization for minimal startup overhead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class ResearchIntelligenceEngine:
    """Synthesized research intelligence engine.

    CONCEPT:KG-2.6/2.33/2.39 — Research Intelligence Engine

    Unifies paper discovery, citation traversal, and research-to-KG
    orchestration into a single facade.

    Usage::

        engine = ResearchIntelligenceEngine(kg_engine)

        # Quick paper scoring
        score, domains = engine.score_paper("Spectral Clustering", "...")

        # Full research cycle
        report = await engine.run_research_cycle()

        # Citation traversal
        edges = engine.traverse_citations("arXiv:2605.04970")

    Args:
        engine: The IntelligenceGraphEngine for KG persistence.
    """

    def __init__(self, engine: IntelligenceGraphEngine | None = None) -> None:
        self._engine = engine
        self._pipeline: Any = None
        self._orchestrator: Any = None
        self._initialized = False

    def _lazy_init(self) -> None:
        """Lazily initialize sub-components."""
        if self._initialized:
            return
        self._initialized = True

        try:
            from ...automation.research_pipeline import (
                PipelineConfig,
                ResearchPipelineRunner,
            )

            self._pipeline = ResearchPipelineRunner(
                engine=self._engine, config=PipelineConfig()
            )
        except Exception as e:
            logger.debug("ResearchPipelineRunner not available: %s", e)

        try:
            from ..orchestration.research_orchestrator import ResearchOrchestrator

            self._orchestrator = ResearchOrchestrator(engine=self._engine)
        except Exception as e:
            logger.debug("ResearchOrchestrator not available: %s", e)

    # --- Pipeline API (KG-2.7) ---

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
        self._lazy_init()
        if not self._pipeline:
            return 0.0, []
        return self._pipeline.score_paper(title, abstract, extra_keywords)

    async def ingest_paper(
        self,
        paper_id: str,
        title: str,
        abstract: str,
        authors: list[str],
        pdf_path: str | None = None,
        source_url: str = "",
    ) -> str:
        """Fully ingest a paper into the KG.

        Args:
            paper_id: External paper identifier.
            title: Paper title.
            abstract: Paper abstract.
            authors: List of author names.
            pdf_path: Optional path to downloaded PDF.
            source_url: Original URL.

        Returns:
            The KG article node ID.
        """
        self._lazy_init()
        if not self._pipeline:
            raise RuntimeError("ResearchPipelineRunner not available")
        return await self._pipeline.ingest_paper_full(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            pdf_path=pdf_path,
            source_url=source_url,
        )

    async def ingest_local_file(
        self,
        file_path: str,
        kb_name: str = "scholarx-research",
        topic: str | None = None,
    ) -> str:
        """Ingest a local file into the KG."""
        self._lazy_init()
        if not self._pipeline:
            raise RuntimeError("ResearchPipelineRunner not available")
        return await self._pipeline.ingest_local_file(file_path, kb_name, topic)

    async def ingest_url(
        self,
        url: str,
        kb_name: str = "scholarx-research",
        topic: str | None = None,
    ) -> str:
        """Ingest a web URL into the KG."""
        self._lazy_init()
        if not self._pipeline:
            raise RuntimeError("ResearchPipelineRunner not available")
        return await self._pipeline.ingest_url(url, kb_name, topic)

    # --- Citation API (KG-2.33) ---

    def traverse_citations(
        self,
        paper_id: str,
        max_depth: int = 2,
        max_per_level: int = 5,
    ) -> list[Any]:
        """Traverse the citation graph for a paper.

        Args:
            paper_id: Starting paper ID.
            max_depth: Maximum traversal depth.
            max_per_level: Max citations per paper.

        Returns:
            List of CitationEdgeNode instances.
        """
        from ..orchestration.research_subagent import CitationGraphWalker

        walker = CitationGraphWalker()
        return walker.get_citations(paper_id, max_depth, max_per_level)

    def create_research_session(
        self,
        query: str,
        token_budget_max: int = 190_000,
    ) -> Any:
        """Create an isolated research session.

        Args:
            query: Research query.
            token_budget_max: Token budget for the session.

        Returns:
            ResearchSubagent instance.
        """
        from ..orchestration.research_subagent import ResearchSubagent

        return ResearchSubagent(query=query, token_budget_max=token_budget_max)

    # --- Orchestration API (KG-2.39) ---

    async def _run_unified_cycle(
        self, papers: list[dict[str, Any]] | None, *, synthesize: bool
    ) -> dict[str, Any]:
        """Run the single canonical research-intelligence cycle (CONCEPT:KG-2.77).

        The golden loop IS the research-pipeline runner: its intake stage discovers
        + ingests + LLM-extracts papers, and its assimilate stage matches them
        against the ecosystem via the ConceptMatcher. This facade no longer owns a
        parallel cycle — it delegates here so the daemon, MCP, and this facade all
        run ONE cycle (No-Legacy). ``run_one_cycle`` is sync; off-thread it so an
        async caller's loop is never blocked.
        """
        import asyncio

        from .golden_loop import GoldenLoopController

        if self._engine is None:
            return {"intake_papers": None, "errors": ["no engine"], "skipped": True}
        return await asyncio.to_thread(
            GoldenLoopController(self._engine).run_one_cycle,
            discover=(papers is None),
            papers=papers,
            breadth=False,
            synthesize=synthesize,
        )

    async def run_research_cycle(
        self,
        query: str = "",  # noqa: ARG002 — focus-query biasing of intake is a TODO
        papers: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Execute the unified research-to-KG cycle (intake → match → synthesize).

        Delegates to the single canonical cycle (CONCEPT:KG-2.77).
        """
        return await self._run_unified_cycle(papers, synthesize=True)

    async def run_daily_pipeline(
        self,
        papers: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Execute the unified cycle's intake + matcher (no synthesis).

        Delegates to the single canonical cycle (CONCEPT:KG-2.77).
        """
        return await self._run_unified_cycle(papers, synthesize=False)

    def can_run_cycle(self) -> bool:
        """Check if enough time has elapsed since the last orchestration cycle."""
        self._lazy_init()
        if not self._orchestrator:
            return True
        return self._orchestrator.can_run_cycle()
