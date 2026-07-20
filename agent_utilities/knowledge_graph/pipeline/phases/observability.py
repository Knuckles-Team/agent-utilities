from ..types import PipelineContext, PipelinePhase


async def _execute_experience_distillation(ctx: PipelineContext, deps: dict) -> dict:
    """Phase 16: Experience Distillation.

    Parses local logs and extracts reasoning traces into the Knowledge Graph.
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Executing Phase 16: Experience Distillation (Synchronous)")

    # In a real implementation, this would parse .system_generated/logs
    # and map them to ReasoningTrace and OutcomeEvaluation nodes.
    # For now, it represents the hook into the pipeline.

    return {"status": "distilled", "traces_extracted": 0}


async def _execute_decision_evolution(ctx: PipelineContext, deps: dict) -> dict:
    """Phase 17: Decision Evolution.

    Uses kg_analogy_search to detect patterns in OutcomeEvaluation nodes
    and proposes HypothesisNodes.

    NOTE: As per design, this is deferred to an async/background process
    to prevent blocking the main ingestion pipeline.
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Deferred Phase 17: Decision Evolution to async GraphMaintainer")

    # Mark as deferred so the GraphMaintainer knows it needs to be run
    return {"status": "deferred_to_maintainer"}


experience_distillation_phase = PipelinePhase(
    name="experience_distillation",
    deps=["validate"],  # Runs after validation
    execute_fn=_execute_experience_distillation,
)

decision_evolution_phase = PipelinePhase(
    name="decision_evolution",
    deps=["experience_distillation"],
    execute_fn=_execute_decision_evolution,
)
