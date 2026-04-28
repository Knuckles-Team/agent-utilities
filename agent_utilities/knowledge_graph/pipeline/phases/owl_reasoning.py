#!/usr/bin/python
"""OWL Reasoning Pipeline Phase.

Phase 13 — Runs after sync, before knowledge_base.
Promotes stable LPG nodes to OWL individuals, runs HermiT/Stardog
reasoning, and downfeeds inferred facts back into the NX graph.
"""

import logging
from pathlib import Path
from typing import Any

from ..types import PhaseResult, PipelineContext, PipelinePhase

logger = logging.getLogger(__name__)

# Default ontology path: bundled alongside the knowledge_graph package
_DEFAULT_ONTOLOGY = str(Path(__file__).parent.parent.parent / "ontology.ttl")


try:
    from ...backends.owl import create_owl_backend
    from ...owl_bridge import OWLBridge

    OWL_SUPPORT = True
except ImportError:
    OWL_SUPPORT = False


async def execute_owl_reasoning(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Phase 13: OWL Reasoning — promote → reason → downfeed."""
    if not ctx.config.enable_owl_reasoning:
        return {"status": "skipped", "reason": "OWL reasoning disabled"}

    if not OWL_SUPPORT:
        return {
            "status": "skipped",
            "reason": "OWL dependencies not installed (agent-utilities[owl] required)",
        }

    # Resolve ontology path
    ontology_path = ctx.config.owl_ontology_path or _DEFAULT_ONTOLOGY
    if not Path(ontology_path).exists():
        return {
            "status": "error",
            "reason": f"Ontology file not found: {ontology_path}",
        }

    # Create OWL backend
    try:
        owl_backend = create_owl_backend(
            backend_type=ctx.config.owl_backend,
            ontology_path=ontology_path,
        )
    except Exception as e:
        logger.error("Failed to create OWL backend: %s", e)
        return {"status": "error", "reason": str(e)}

    # Create bridge and run cycle
    bridge = OWLBridge(
        graph=ctx.nx_graph,
        owl_backend=owl_backend,
        backend=ctx.backend,
        importance_threshold=ctx.config.owl_promotion_importance_threshold,
        recency_days=ctx.config.owl_promotion_recency_days,
    )

    try:
        stats = bridge.run_cycle()
    except Exception as e:
        logger.error("OWL reasoning cycle failed: %s", e)
        return {"status": "error", "reason": str(e)}
    finally:
        owl_backend.close()

    return {
        "status": "completed",
        "promoted_nodes": stats.get("promoted_nodes", 0),
        "promoted_edges": stats.get("promoted_edges", 0),
        "inferred": stats.get("inferred", 0),
        "downfed": stats.get("downfed", 0),
    }


owl_reasoning_phase = PipelinePhase(
    name="owl_reasoning",
    deps=["sync"],
    execute_fn=execute_owl_reasoning,
)
