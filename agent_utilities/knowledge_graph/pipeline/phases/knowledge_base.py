#!/usr/bin/python
"""Phase 13: Knowledge Base Sync.

Runs after Phase 12 (Sync) and handles automatic ingestion of enabled
skill-graphs and any KB updates. Controlled by PipelineConfig:
  - enable_knowledge_base: bool (default True)
  - kb_auto_ingest_skill_graphs: bool (default False — on-demand only)
"""

import logging
import time
from typing import Any

from ..types import PhaseResult, PipelineContext, PipelinePhase

logger = logging.getLogger(__name__)


async def execute_knowledge_base(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Phase 13: KB Sync — auto-ingest enabled skill-graphs if configured."""
    if not ctx.config.enable_knowledge_base:
        return {"status": "skipped", "reason": "knowledge base disabled"}

    start = time.time()
    kb_results = []

    # Auto-ingest enabled skill-graphs (if configured)
    if ctx.config.kb_auto_ingest_skill_graphs:
        try:
            from skill_graphs import get_skill_graphs_path
        except ImportError:
            logger.debug("skill_graphs package not available, skipping auto-ingest")
            return {"status": "skipped", "reason": "skill_graphs not installed"}

        enabled_paths = get_skill_graphs_path()
        if not enabled_paths:
            return {
                "status": "skipped",
                "reason": "no enabled skill-graphs found",
                "kb_count": 0,
            }

        from ...kb.ingestion import KBIngestionEngine

        engine = KBIngestionEngine(
            graph=ctx.nx_graph,
            backend=ctx.backend,
            chunk_size=ctx.config.kb_chunk_size,
        )

        for graph_path in enabled_paths:
            try:
                logger.info(f"Auto-ingesting skill-graph: {graph_path}")
                meta = await engine.ingest_skill_graph(graph_path, force=False)
                kb_results.append(
                    {
                        "name": meta.name,
                        "articles": meta.article_count,
                        "sources": meta.source_count,
                        "status": meta.status,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to ingest skill-graph {graph_path}: {e}")
                kb_results.append(
                    {"path": str(graph_path), "status": "error", "error": str(e)}
                )

    duration = (time.time() - start) * 1000
    return {
        "status": "complete",
        "kbs_processed": len(kb_results),
        "kb_results": kb_results,
        "duration_ms": duration,
        "auto_ingest": ctx.config.kb_auto_ingest_skill_graphs,
    }


knowledge_base_phase = PipelinePhase(
    name="knowledge_base",
    deps=["sync"],
    execute_fn=execute_knowledge_base,
)
