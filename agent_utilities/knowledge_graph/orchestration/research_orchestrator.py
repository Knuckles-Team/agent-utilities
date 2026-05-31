#!/usr/bin/python
from __future__ import annotations

"""Research Orchestration Loop.

CONCEPT:KG-2.6 — Research Orchestration Integration

Connects the ResearchSubagent (KG-2.33) to the ResearchPipelineRunner
(KG-2.11) and the unified RAG-KG retriever (KG-2.38) for automated,
scheduled research loops.

Architecture::

    Daily Schedule (OS-5.2 Cron)
           ↓
    ResearchOrchestrator.run_research_cycle()
           ↓
    ┌─────────────────────────────────────────────┐
    │ 1. Discover papers via ResearchPipelineRunner│
    │ 2. Create ResearchSubagent session           │
    │ 3. Traverse citation graphs (KG-2.33)        │
    │ 4. Score & ingest via pipeline (KG-2.11)     │
    │ 5. Build similarity edges (KG-2.36)          │
    │ 6. Refresh cluster index (KG-2.34)           │
    │ 7. Persist session + provenance to KG        │
    └─────────────────────────────────────────────┘

Designed for integration with:
- MCP tool interface (ECO-4.0): ``run_research_cycle`` as an MCP tool
- MaintenanceCron (OS-5.2): As a scheduled daily task
- universal-skills research-scanner: As the discovery backend
"""


import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_utilities.knowledge_graph.memory import (
    AutoSimilarityLinker,
)
from agent_utilities.knowledge_graph.orchestration.research_subagent import (
    ResearchSubagent,
)
from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
    KGNativeRetrievalRetriever,
)
from agent_utilities.models.knowledge_graph import (
    RegistryEdge,
    RegistryNode,
    ResearchSessionNode,
)

if TYPE_CHECKING:
    from agent_utilities.automation.research_pipeline import (
        PipelineReport,
        ResearchPipelineRunner,
    )
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationConfig:
    """Configuration for the research orchestration loop.

    Attributes:
        max_papers_per_cycle: Maximum papers to process per cycle.
        citation_depth: Maximum citation graph traversal depth.
        citation_breadth: Maximum citations to follow per paper.
        enable_similarity_linking: Auto-create similarity edges post-ingestion.
        enable_cluster_refresh: Rebuild spectral cluster index post-ingestion.
        token_budget_max: Token budget for the research subagent session.
        min_relevance_score: Minimum relevance score for citation traversal.
        cycle_interval_hours: Minimum hours between automated cycles.
    """

    max_papers_per_cycle: int = 50
    citation_depth: int = 2
    citation_breadth: int = 5
    enable_similarity_linking: bool = True
    enable_cluster_refresh: bool = True
    token_budget_max: int = 190_000
    min_relevance_score: float = 3.0
    cycle_interval_hours: int = 24


@dataclass
class OrchestrationReport:
    """Report from a single orchestration cycle.

    Attributes:
        cycle_id: Unique identifier for this cycle.
        timestamp: ISO timestamp.
        papers_discovered: Papers found in discovery phase.
        papers_ingested: Papers that passed relevance thresholds.
        citations_traversed: Total citation edges traversed.
        findings_recorded: Research findings recorded.
        similarity_edges_created: Auto-similarity edges created.
        clusters_built: Spectral clusters (re)built.
        pipeline_report: Underlying pipeline report.
        session_node: The finalized research session node.
        duration_seconds: Total execution time.
        errors: Any errors encountered.
    """

    cycle_id: str = ""
    timestamp: str = ""
    papers_discovered: int = 0
    papers_ingested: int = 0
    citations_traversed: int = 0
    findings_recorded: int = 0
    similarity_edges_created: int = 0
    clusters_built: int = 0
    pipeline_report: PipelineReport | None = None
    session_node: ResearchSessionNode | None = None
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class ResearchOrchestrator:
    """Automated research orchestration loop.

    CONCEPT:KG-2.6 — Research Orchestration Integration

    Connects the ResearchSubagent, ResearchPipelineRunner, and
    KGNativeRetrievalRetriever into a cohesive cycle for autonomous,
    scheduled research ingestion and KG enrichment.

    Example::

        orchestrator = ResearchOrchestrator(engine)
        report = await orchestrator.run_research_cycle()
        print(f"Discovered {report.papers_discovered} papers, "
              f"created {report.similarity_edges_created} edges")

    MCP Integration::

        # Register as MCP tool in universal-skills
        @mcp_tool
        async def run_research_cycle(query: str = ""):
            orchestrator = ResearchOrchestrator(engine)
            return await orchestrator.run_research_cycle(query=query)
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine | None = None,
        config: OrchestrationConfig | None = None,
    ):
        """Initialize the research orchestrator.

        Args:
            engine: The KG engine for graph access and persistence.
            config: Orchestration configuration.
        """
        self.engine = engine
        self.config = config or OrchestrationConfig()
        self._last_cycle_time: float = 0.0

        # Sub-components
        self._similarity_linker = AutoSimilarityLinker()
        self._unified_retriever = KGNativeRetrievalRetriever(engine=engine)

    def _create_pipeline_runner(self) -> ResearchPipelineRunner:
        """Create a configured ResearchPipelineRunner."""
        from agent_utilities.automation.research_pipeline import (
            PipelineConfig,
            ResearchPipelineRunner,
        )

        pipeline_config = PipelineConfig(
            max_papers_per_run=self.config.max_papers_per_cycle,
        )
        return ResearchPipelineRunner(
            engine=self.engine,
            config=pipeline_config,
        )

    def can_run_cycle(self) -> bool:
        """Check if enough time has elapsed since the last cycle.

        Returns:
            True if the minimum interval has passed.
        """
        if self._last_cycle_time == 0.0:
            return True
        elapsed_hours = (time.time() - self._last_cycle_time) / 3600.0
        return elapsed_hours >= self.config.cycle_interval_hours

    async def run_research_cycle(
        self,
        query: str = "",
        papers: list[dict[str, Any]] | None = None,
    ) -> OrchestrationReport:
        """Execute a full research-to-KG orchestration cycle.

        Phases:
        1. **Discovery**: Find new papers via pipeline runner.
        2. **Subagent Session**: Create isolated research context.
        3. **Citation Traversal**: Walk citation graphs for high-relevance papers.
        4. **Pipeline Ingestion**: Score and ingest via tiered pipeline.
        5. **Similarity Linking**: Auto-create SIMILAR_TO edges.
        6. **Cluster Refresh**: Rebuild spectral cluster index.
        7. **Persist**: Save session + provenance to KG.

        Args:
            query: Optional focus query to bias discovery.
            papers: Optional pre-fetched papers. If None, discovers automatically.

        Returns:
            OrchestrationReport with full cycle details.
        """
        start_time = time.time()
        cycle_id = f"orch_{uuid.uuid4().hex[:8]}"
        report = OrchestrationReport(
            cycle_id=cycle_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        # Create subagent session
        subagent = ResearchSubagent(
            query=query or "daily research discovery",
            session_id=cycle_id,
            token_budget_max=self.config.token_budget_max,
        )

        try:
            # Phase 1: Discovery via pipeline
            pipeline = self._create_pipeline_runner()
            pipeline_report = await pipeline.run_daily_pipeline(papers=papers)
            report.pipeline_report = pipeline_report
            report.papers_discovered = pipeline_report.papers_discovered
            report.papers_ingested = (
                pipeline_report.papers_relevant + pipeline_report.papers_marginal
            )

            # Phase 2: Citation traversal for high-relevance papers
            if pipeline_report.records:
                for record in pipeline_report.records:
                    if (
                        record.relevance_score >= self.config.min_relevance_score
                        and record.paper_id
                    ):
                        try:
                            edges = subagent.traverse_citations(
                                paper_id=record.paper_id,
                                max_depth=self.config.citation_depth,
                                max_per_level=self.config.citation_breadth,
                            )
                            report.citations_traversed += len(edges)

                            # Record finding per paper
                            subagent.add_finding(
                                claim=f"Traversed {len(edges)} citations for: {record.title[:80]}",
                                confidence=0.9,
                                _source_paper_id=record.paper_id,
                            )
                            report.findings_recorded += 1
                        except Exception as e:
                            err = (
                                f"Citation traversal failed for {record.paper_id}: {e}"
                            )
                            report.errors.append(err)
                            logger.warning(err)

            # Phase 3: Auto-similarity linking
            if self.config.enable_similarity_linking and self.engine:
                try:
                    all_kg_nodes = self._get_recent_nodes()
                    for i, node in enumerate(all_kg_nodes):
                        if node.embedding:
                            predecessors = all_kg_nodes[:i]
                            new_edges = self._similarity_linker.link_new_node(
                                new_node=node,
                                existing_nodes=predecessors,
                            )
                            report.similarity_edges_created += len(new_edges)

                            # Persist edges to KG
                            registry_edges = self._similarity_linker.to_registry_edges(
                                new_edges
                            )
                            for edge in registry_edges:
                                self._persist_edge(edge)
                except Exception as e:
                    err = f"Similarity linking failed: {e}"
                    report.errors.append(err)
                    logger.warning(err)

            # Phase 4: Cluster index refresh
            if self.config.enable_cluster_refresh and self.engine:
                try:
                    all_nodes = self._get_recent_nodes()
                    n_clusters = self._unified_retriever.build_cluster_index(
                        all_nodes, domain="research"
                    )
                    report.clusters_built = n_clusters
                except Exception as e:
                    err = f"Cluster refresh failed: {e}"
                    report.errors.append(err)
                    logger.warning(err)

            # Phase 5: Finalize session
            session_node = subagent.finalize()
            report.session_node = session_node

            # Persist session and provenance
            if self.engine:
                self._persist_session(subagent)

        except Exception as e:
            report.errors.append(f"Cycle failed: {e}")
            logger.error("Research orchestration cycle failed: %s", e)

        report.duration_seconds = time.time() - start_time
        self._last_cycle_time = time.time()

        logger.info(
            "[CONCEPT:KG-2.6] Orchestration cycle %s complete: "
            "%d papers, %d ingested, %d citations, %d similarity edges, "
            "%d clusters, %.1fs",
            cycle_id,
            report.papers_discovered,
            report.papers_ingested,
            report.citations_traversed,
            report.similarity_edges_created,
            report.clusters_built,
            report.duration_seconds,
        )

        return report

    def _get_recent_nodes(self, limit: int = 500) -> list[RegistryNode]:
        """Get recent KG nodes for similarity/clustering.

        Fetches nodes from the KG engine with embeddings,
        prioritizing recently created ones.
        """
        if not self.engine:
            return []

        nodes: list[RegistryNode] = []
        for node_id, data in list(self.engine.graph.nodes(data=True))[-limit:]:
            try:
                embedding = data.get("embedding")
                if embedding and isinstance(embedding, list):
                    node = RegistryNode(  # type: ignore
                        id=str(node_id),
                        name=data.get("name", str(node_id)),
                        description=data.get("description", ""),
                        embedding=embedding,
                    )
                    nodes.append(node)
            except Exception:
                continue  # nosec

        return nodes

    def _persist_session(self, subagent: ResearchSubagent) -> None:
        """Persist the research session and provenance edges to the KG."""
        if not self.engine:
            return

        # Persist all nodes
        for node in subagent.get_all_nodes():
            try:
                self.engine.graph.add_node(node.id, **node.model_dump())
            except Exception as e:
                logger.debug("Failed to persist node %s: %s", node.id, e)

        # Persist provenance edges
        for edge in subagent.get_provenance_edges():
            self._persist_edge(edge)

    def _persist_edge(self, edge: RegistryEdge) -> None:
        """Persist a single edge to the KG."""
        if not self.engine:
            return

        try:
            self.engine.graph.add_edge(
                edge.source,
                edge.target,
                type=edge.type,
                weight=edge.weight,
                **(edge.metadata or {}),
            )
        except Exception as e:
            logger.debug(
                "Failed to persist edge %s→%s: %s", edge.source, edge.target, e
            )
