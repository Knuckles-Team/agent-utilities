#!/usr/bin/python
"""Phase 13: Knowledge Base Sync.

Runs after Phase 12 (Sync) and natively auto-ingests ALL packaged knowledge into
the KG so it is queryable out of the box — all three skill types at document-grade
enrichment: skill-graphs, atomic skills, and workflow DAGs. Delta-skipped (only the
first run is heavy). Controlled by PipelineConfig (all default-on; disable via
KG_AUTO_INGEST_SKILLS=false):
  - enable_knowledge_base: bool (default True)
  - kb_auto_ingest_skill_graphs: bool (default True — ALL graphs, document-grade)
  - kb_auto_ingest_universal_skills: bool (default True — atomic skills + workflows)
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
    # whole packaged library is ingested, not only env-enabled graphs. We ingest each
    # graph's reference/ corpus through the **document** content type — the SAME
    # full-enrichment path documents/books take (chunk → embed → Document/Chunk objects
    # + concept + fact extraction), not the lossy curated-Article KB path — so skills
    # become semantically searchable and concept-linked. Delta-skipped (force=False),
    # so only the first run is heavy; embedder calls are now bounded (KG-2.48) so a
    # flaky GPU degrades to no-vectors instead of hanging.
    if ctx.config.kb_auto_ingest_skill_graphs:
        from pathlib import Path as _Path

        try:
            from skill_graphs import get_skill_graphs_path

            enabled_paths = get_skill_graphs_path(default_enabled=True)
        except ImportError:
            logger.debug("skill_graphs package not available, skipping auto-ingest")
            enabled_paths = []

        from ...core.engine import IntelligenceGraphEngine

        kg_engine = IntelligenceGraphEngine.get_active()
        if enabled_paths and kg_engine is not None:
            from ...ingestion.engine import (
                ContentType,
                IngestionEngine,
                IngestionManifest,
            )

            ie = IngestionEngine(kg_engine=kg_engine)
            for graph_path in enabled_paths:
                ref = _Path(graph_path) / "reference"
                target = ref if ref.is_dir() else _Path(graph_path)
                try:
                    logger.info(
                        f"Auto-ingesting skill-graph (document-grade): {graph_path}"
                    )
                    res = await ie.ingest(
                        IngestionManifest(
                            content_type=ContentType.DOCUMENT,
                            source_uri=str(target),
                            metadata={"chunk_objects": True},
                        )
                    )
                    kb_results.append(
                        {
                            "name": _Path(graph_path).name,
                            "status": res.status,
                            "nodes": res.nodes_created,
                            "edges": res.edges_created,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to ingest skill-graph {graph_path}: {e}")
                    kb_results.append(
                        {"path": str(graph_path), "status": "error", "error": str(e)}
                    )

    # Auto-ingest the universal-skills corpus (default-on, best-effort): the ATOMIC
    # SKILLS (full enrichment, same as documents — chunk+embed+concept+fact + skill_type)
    # AND the workflow DAGs. So all three skill types — atomic skill / skill-graph /
    # workflow — land in the KG natively, cross-linked by concept.
    if ctx.config.kb_auto_ingest_universal_skills:
        from ...core.engine import IntelligenceGraphEngine

        us_engine = IntelligenceGraphEngine.get_active()

        # (a) atomic skills — every SKILL.md that is NOT a workflow.
        try:
            from universal_skills.skill_utilities import get_universal_skills_path

            atomic = [
                p for p in get_universal_skills_path() if "/workflows/" not in str(p)
            ]
        except ImportError:
            atomic = []
        if atomic and us_engine is not None:
            from ...ingestion.engine import (
                ContentType,
                IngestionEngine,
                IngestionManifest,
            )

            sie = IngestionEngine(kg_engine=us_engine)
            ingested = 0
            for sd in atomic:
                try:
                    res = await sie.ingest(
                        IngestionManifest(
                            content_type=ContentType.SKILL, source_uri=str(sd)
                        )
                    )
                    ingested += res.status == "success"
                except Exception as e:  # noqa: BLE001 — one bad skill must not abort
                    logger.debug("atomic skill ingest failed for %s: %s", sd, e)
            kb_results.append(
                {
                    "name": "universal-skills/atomic",
                    "status": "complete",
                    "skills": ingested,
                    "scanned": len(atomic),
                }
            )

        # (b) workflow DAGs.
        try:
            from ...ingestion.skill_workflow_ingest import ingest_skill_workflows

            if us_engine is not None:
                wf = ingest_skill_workflows(us_engine)
                kb_results.append(
                    {"name": "universal-skills/workflows", "status": "complete", **wf}
                )
        except Exception as e:
            logger.warning(f"universal-skills workflow auto-ingest failed: {e}")
            kb_results.append(
                {
                    "name": "universal-skills/workflows",
                    "status": "error",
                    "error": str(e),
                }
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
