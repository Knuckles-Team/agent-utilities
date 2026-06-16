#!/usr/bin/python
"""Phase 13: Knowledge Base Sync.

Runs after Phase 12 (Sync) and auto-ingests the packaged knowledge — ALL
skill-graphs and the universal-skills workflow corpus — into the KG so it is
queryable out of the box. Delta-skipped (only the first run is heavy). Controlled
by PipelineConfig (all default-on; disable via KG_AUTO_INGEST_SKILLS=false):
  - enable_knowledge_base: bool (default True)
  - kb_auto_ingest_skill_graphs: bool (default True — ALL graphs)
  - kb_auto_ingest_universal_skills: bool (default True — workflow corpus)
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

    # Auto-ingest ALL discovered skill-graphs (default-on). default_enabled=True so the
    # whole packaged library is ingested, not only env-enabled graphs. ingest is
    # delta-skipped (force=False) so only the first run is heavy.
    if ctx.config.kb_auto_ingest_skill_graphs:
        try:
            from skill_graphs import get_skill_graphs_path

            enabled_paths = get_skill_graphs_path(default_enabled=True)
        except ImportError:
            logger.debug("skill_graphs package not available, skipping auto-ingest")
            enabled_paths = []

        if enabled_paths:
            from ...kb.ingestion import KBIngestionEngine

            engine = KBIngestionEngine(
                graph=ctx.graph,
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

    # Auto-ingest the universal-skills workflow corpus (default-on, best-effort) so the
    # dispatchable WorkflowDefinition DAGs are discoverable by graph-os out of the box.
    if ctx.config.kb_auto_ingest_universal_skills:
        try:
            from ...ingestion.skill_workflow_ingest import ingest_skill_workflows

            wf = ingest_skill_workflows(ctx.graph)
            kb_results.append(
                {"name": "universal-skills/workflows", "status": "complete", **wf}
            )
        except Exception as e:
            logger.warning(f"universal-skills workflow auto-ingest failed: {e}")
            kb_results.append(
                {"name": "universal-skills/workflows", "status": "error", "error": str(e)}
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
