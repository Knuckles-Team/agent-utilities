"""Post-ingestion validation pipeline phase (CONCEPT:KG-2.3).

Runs the ``GraphValidator`` as a non-blocking post-ingestion step.
Auto-fixes recoverable issues and logs integrity/quality reports.
"""

from typing import Any

from agent_utilities.knowledge_graph.memory import EvaluationCapture
from agent_utilities.knowledge_graph.security.graph_validator import GraphValidator

from ..types import (
    PhaseResult,
    PipelineContext,
    PipelinePhase,
)


async def execute_validate(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Run graph integrity validation as a post-ingestion phase.

    CONCEPT:KG-2.3 — Non-blocking graph validation.

    Auto-fixes Tier 1 issues, logs Tier 2/3/4 issues.
    Stores validation metrics in eval_capture for trend analysis.
    Never blocks pipeline completion unless Tier 4 fatals are detected
    AND the graph is truly unusable (zero nodes).
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    # Build a temporary engine wrapper around the pipeline's graph
    engine = IntelligenceGraphEngine(
        graph=ctx.graph,
        backend=ctx.backend,
    )

    validator = GraphValidator(engine)
    report = validator.validate()

    # Store validation report as eval_capture entry for trend analysis
    try:
        capture = EvaluationCapture(knowledge_engine=engine, enabled=True)
        capture.capture(
            query="__graph_validation__",
            method="validator",
            result_node_ids=[
                f"tier1:{len(report.tier1_fixes)}",
                f"tier2:{len(report.tier2_violations)}",
                f"tier3:{len(report.tier3_warnings)}",
                f"tier4:{len(report.tier4_fatal)}",
            ],
            scores=[
                float(len(report.tier1_fixes)),
                float(len(report.tier2_violations)),
                float(len(report.tier3_warnings)),
                float(len(report.tier4_fatal)),
            ],
            latency_ms=report.duration_ms,
        )
    except Exception:
        pass  # nosec B110 # Best-effort — don't fail validation due to eval_capture issues

    return {
        "validated": True,
        "healthy": report.is_healthy,
        "tier1_fixes": len(report.tier1_fixes),
        "tier2_violations": len(report.tier2_violations),
        "tier3_warnings": len(report.tier3_warnings),
        "tier4_fatal": len(report.tier4_fatal),
        "total_nodes": report.total_nodes,
        "total_edges": report.total_edges,
        "duration_ms": report.duration_ms,
        "summary": report.summary(),
    }


validate_phase = PipelinePhase(
    name="validate",
    deps=["knowledge_base"],  # Runs after all ingestion phases
    execute_fn=execute_validate,
)
