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
            {
                "codebase",
                "document",
                "content_url",
                "diff",
                "skill_workflows",
                # CONCEPT:KG-2.272 — remote chat/session bundle uploads land in
                # the usage store off the request path; a bulk write, so it rides
                # the ingestion lane (not the best-effort maint lane).
                "session_upload",
            }
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
        # connector_drain = ONE paginated page of a chunked full-corpus drain (CONCEPT:KG-2.301):
        # a single ``source_sync(full)`` of a large source (freshrss's ~11k backlog) fans out as a
        # self-continuing chain of these, draining the whole corpus under this lane's background
        # priority + the GB10 capacity guard so it can't time out or OOM.
        "task_types": frozenset({"connector_sync", "feed_sweep", "connector_drain"}),
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
    # CONCEPT:KG-2.153 — OWL capability-card backfill is its OWN throughput lane,
    # NOT a maint interval tick. It used to ride the ``maint`` lane as a generic
    # ``scheduled_job`` and so was capped at the best-effort floor (1 worker, shared
    # round-robin with ~17 other maint ticks), which left ~85k Code symbols
    # un-carded (codebase card coverage ~0.0006). As its own lane it drains the
    # ``cards_pending`` backlog in parallel with — but, being a normal
    # (non-best-effort) lane still bounded by the per-lane min / hot-spare
    # reservation AND the shared background-throttle semaphore, never starving — the
    # control plane (isolated on ``__control__``, KG-2.148) or the latency-sensitive
    # ``queries`` lane. ``lite`` model role: card summaries are a structured
    # extraction the lite model handles fast (the default in ``_tick_enrichment``).
    "enrichment": {
        "task_types": frozenset({"enrichment_backfill"}),
        "model_role": "lite",
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

# CONCEPT:KG-2.289 — INTERACTIVE lanes: latency-sensitive work that must ALWAYS
# have a free host worker, even when ingestion saturates the pool. The scheduler's
# AdmissionPolicy reserves a worker floor that NON-interactive lanes
# (codebase/document/connector/maint) can never claim, so an interactive call is
# serviceable even under a heavy bulk ingest. ``queries`` owns conversation /
# kg_memory (the on-pool half of MCP/chat); the off-pool MCP server stays
# responsive because the host is never 100%-consumed by ingestion workers.
INTERACTIVE_LANES: frozenset[str] = frozenset({"queries"})

# CONCEPT:KG-2.286 — per-lane SOFT execution timeout (seconds). A claimed task
# whose execution exceeds its lane's bound is cancelled and routed through the
# existing KG-2.113 retry→backoff→dead_letter machinery, so a hung connector or
# maint tick frees its worker FAST instead of pinning it until the reaper's 2h
# absolute runtime cap. Auto-sized per lane from its expected envelope — the live
# tail evidence is: connectors p50=16ms but one hung 456s (→180s bound catches it
# with ~10000x p50 headroom); maint p50=30s but one hung 761s (→600s bound); the
# heavy ingestion lane is generous (codebase p95 is legitimately minutes, and the
# big-repo split (KG-2.287) shrinks each unit anyway). No env knob: the bound is a
# deterministic function of the lane, and the reaper's absolute cap is the backstop.
LANE_SOFT_TIMEOUT_SEC: dict[str, float] = {
    "queries": 120.0,
    "connectors": 180.0,
    "worldview": 300.0,
    "maint": 600.0,
    "research": 1800.0,
    "extraction": 1800.0,
    "ingestion": 3600.0,
}
# Default bound for an unmapped lane: generous, below the reaper's 2h hard cap.
DEFAULT_SOFT_TIMEOUT_SEC: float = 1800.0


def lane_soft_timeout(lane: str | None) -> float:
    """The soft execution-timeout bound (seconds) for ``lane`` (CONCEPT:KG-2.286)."""
    return LANE_SOFT_TIMEOUT_SEC.get(lane or "", DEFAULT_SOFT_TIMEOUT_SEC)


def task_soft_timeout(task_type: str | None) -> float:
    """The soft execution-timeout bound (seconds) for a task TYPE (CONCEPT:KG-2.286).

    Resolves the type's lane and returns that lane's bound, so the cancel-and-retry
    watchdog is sized by functional domain without a per-type table to drift.
    """
    return lane_soft_timeout(lane_for_task_type(task_type))


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
