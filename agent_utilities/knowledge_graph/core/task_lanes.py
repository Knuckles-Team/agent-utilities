"""Functional task lanes — fair, observable scheduling + per-LLM routing (CONCEPT:ORCH-1.75).

The KG task queue used ONE strict priority order across ALL task types, so a slow/backed-up
type could head-of-line-block another: codebase ingestion sat at 75-pending / 0-processed
while loop_cycle/research churned, and the only way to see it was a manual metrics dump.

Lanes partition the queue by FUNCTIONAL DOMAIN. Each lane is:
  * **serviced fairly** — the worker claims round-robin across lanes, so no lane starves
    another (within a lane, the existing priority buckets still order work);
  * **routed to its own LLM** — ``model_role`` resolves to a vLLM model via create_model's
    role routing (ORCH-1.27), so e.g. heavy research doesn't contend with latency-sensitive
    chat on one model;
  * **independently observable** — :func:`lane congestion metrics` expose per-lane depth /
    oldest-pending age / in-flight so congestion is visible *before* it starves work.

Collapse-to-1: a lane whose model is unhealthy folds its LLM work onto a healthy lane's model
(``healthy_model_role``) — degrade, never stall.
"""

from __future__ import annotations

# lane -> {task_types it owns, the model role its LLM work routes to}.
TASK_LANES: dict[str, dict] = {
    # user-facing, latency-sensitive — must never queue behind heavy ingestion/research
    "queries": {
        "task_types": frozenset({"conversation", "kg_memory"}),
        "model_role": "generator",
    },
    # the heavy bulk lane that was being starved
    "ingestion": {
        "task_types": frozenset(
            {"codebase", "document", "content_url", "diff", "skill_workflows"}
        ),
        "model_role": "lite",
    },
    "research": {
        "task_types": frozenset(
            {"research_paper_fetch", "background_research", "deep_extract"}
        ),
        "model_role": "learner",
    },
    # scheduled maintenance / orchestration / analysis (loop_cycle lives here as scheduled_job)
    "maint": {
        "task_types": frozenset(
            {
                "scheduled_job",
                "deep_analysis",
                "relevance_sweep",
                "fleet_event_triage",
                "deploy_watch",
                "synthesize",
            }
        ),
        "model_role": "judge",
    },
}

LANE_NAMES: tuple[str, ...] = tuple(TASK_LANES)
DEFAULT_LANE = "maint"

_TYPE_TO_LANE: dict[str, str] = {
    t: lane for lane, cfg in TASK_LANES.items() for t in cfg["task_types"]
}


def lane_for_task_type(task_type: str | None) -> str:
    """The functional lane that owns a task type (``DEFAULT_LANE`` for unmapped types)."""
    return _TYPE_TO_LANE.get(task_type or "", DEFAULT_LANE)


def lane_model_role(lane: str) -> str:
    """The model role a lane routes its LLM work to (for create_model role resolution)."""
    return TASK_LANES.get(lane, TASK_LANES[DEFAULT_LANE])["model_role"]
