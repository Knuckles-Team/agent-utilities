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
    # CONCEPT:ORCH-1.77 — external connector delta syncs (gitlab/servicenow/atlassian/egeria/…)
    # as their OWN lane: the */20m fleet sweep enqueues one connector_sync task per connector,
    # so they fan out in parallel here instead of one slow connector blocking the rest inline,
    # and never contend with codebase/document indexing in the ingestion lane.
    "connectors": {
        # connector_sync = external delta syncs; feed_sweep = the on-demand RSS/
        # FreshRSS sweep run OFF the request path (it fetches+gates+enqueues the
        # per-article worldview/research tasks, so it must not ride the 300s MCP
        # call). Both are sweep PRODUCERS, kept off the ingestion lane. (KG-2.121)
        "task_types": frozenset({"connector_sync", "feed_sweep"}),
        "model_role": "lite",
    },
    "research": {
        # CONCEPT:KG-2.172 — the research-cohort barrier gate (cohort_synthesize) lives
        # HERE, not the best-effort maint lane: under heavy cohort ingestion the maint
        # floor-cap starved the gate so the matrix never synthesized. Research is a real
        # throughput lane, so the gate is reliably claimed once its members drain.
        "task_types": frozenset(
            {"research_paper_fetch", "background_research", "cohort_synthesize"}
        ),
        "model_role": "learner",
    },
    # CONCEPT:KG-2.121 — the WORLDVIEW stream: relevance-gated news/world-event
    # articles (feed_ingest) build the world model. Its OWN lane so it drains in
    # parallel with — and never head-of-line-blocks behind — research-paper fetch
    # (which feeds agent-utilities-evolution) or the heavy codebase backlog. The
    # world-model gate is the router that splits feed items into research vs here.
    "worldview": {
        "task_types": frozenset({"feed_ingest"}),
        "model_role": "lite",
    },
    # CONCEPT:ORCH-1.76 — the LLM-extraction / deep-analysis phase is its OWN lane (heavy model
    # work), kept off the file-ingestion lane so a slow extraction can't head-of-line-block a
    # fast codebase/diff index. (Vectorization already runs in its own _embed_backfill_thread,
    # off the queue entirely; connector-sync + enrichment become lanes once they are task types.)
    "extraction": {
        "task_types": frozenset({"deep_extract", "deep_analysis"}),
        "model_role": "learner",
    },
    # scheduled maintenance / orchestration (loop_cycle lives here as scheduled_job)
    "maint": {
        "task_types": frozenset(
            {
                "scheduled_job",
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

# CONCEPT:ORCH-1.82 — "best-effort" lanes: low-value, high-volume periodic work
# (the maint interval ticks) that must be GUARANTEED its minimum coverage but must
# NEVER expand into spare workers. Without this, a backlog of cheap scheduled-job
# ticks (one per due-minute, per schedule) crowds the worker pool — the rotation
# offers maint most often because it has the most pending — and starves the
# throughput lanes (ingestion/worldview/research) of all but their minimum. The
# AdmissionPolicy caps a best-effort lane at its per-lane floor (see
# :meth:`AdmissionPolicy.decide`). Pairs with the scheduler's stale-tick collapse
# (CONCEPT:OS-5.53), which keeps the backlog itself small.
BEST_EFFORT_LANES: frozenset[str] = frozenset({"maint"})

_TYPE_TO_LANE: dict[str, str] = {
    t: lane for lane, cfg in TASK_LANES.items() for t in cfg["task_types"]
}


def lane_for_task_type(task_type: str | None) -> str:
    """The functional lane that owns a task type (``DEFAULT_LANE`` for unmapped types)."""
    return _TYPE_TO_LANE.get(task_type or "", DEFAULT_LANE)


def lane_task_types(lane: str) -> list[str]:
    """The task types a lane owns, in deterministic order — for per-TYPE fair claiming WITHIN
    the lane (CONCEPT:ORCH-1.76) so a fast type (diff/document) isn't stuck behind a slow one
    (a big codebase batch) sharing the lane."""
    return sorted(TASK_LANES.get(lane, {}).get("task_types", ()))


def lane_model_role(lane: str) -> str:
    """The model role a lane routes its LLM work to (for create_model role resolution)."""
    return TASK_LANES.get(lane, TASK_LANES[DEFAULT_LANE])["model_role"]
