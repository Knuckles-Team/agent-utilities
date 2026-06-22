# Recipe — Unified scheduling, the priority queue, and the ScholarX RSS research feed

The gateway daemon runs **one** intelligent scheduler (CONCEPT:OS-5.44). Every
recurring job — the `deploy/schedules.yml` entries, the former fixed-interval
maintenance ticks, the self-evolution `loop_cycle`, and the ScholarX RSS research
feed — is a durable `:Schedule` node. The single scheduler tick evaluates them and
**enqueues** a `scheduled_job` `:Task` onto one hardened priority+scheduled queue
(CONCEPT:KG-2.113) that the worker pool drains. Nothing recurring runs inline in the
scheduler thread anymore.

## The queue (CONCEPT:KG-2.113)

A `:Task` carries:

- **`prio_bucket`** — discrete priority `0` (critical) … `3` (background). Workers
  claim the lowest non-empty bucket first (the L1 graph interpreter strips
  `ORDER BY`, so priority is N equality queries, not a sort). `prioritize_task`
  and the `prio`/`priority` arguments set it.
- **`scheduled` + eta** — delayed execution. A task with a future eta waits as
  `status='scheduled'` until the per-minute `promotion_sweep` makes it `pending`.
  Application-level **retry/backoff** reuses this lane (a failure reschedules with
  exponential backoff); a task that exhausts `max_attempts` becomes a
  `dead_letter` (distinct from the reaper's crash-requeue).
- **`blocked` + `depends_on`** — dependency gating. A blocked task is promoted once
  every dependency has `completed`; a terminally-failed dependency cancels it.

## Controlling schedules (two surfaces)

```bash
# MCP
graph_schedules action=list
graph_schedules action=disable name=research_feed
graph_schedules action=prioritize name=loop_cycle priority=1
graph_schedules action=set_interval name=research_feed interval_s=900
graph_schedules action=run_now name=enrichment

# REST (the auto-mounted twin)
curl -s localhost:8080/graph/schedules -d '{"action":"list"}'
```

`deploy/schedules.yml` is the **seed** (desired state); the `:Schedule` node holds
live last-run / next-run / failure-backoff and survives restart and leader-failover.

## The ScholarX RSS research feed (CONCEPT:KG-2.114)

A default-on `research_feed` schedule (`KG_RESEARCH_FEED`, cadence
`KG_RESEARCH_FEED_INTERVAL`, default 30 min) enqueues
`LoopController.run_rss_feed_screen`, which:

1. reads the arXiv **RSS feed** (`get_recent_papers(days=1)` — cheap title+abstract);
2. **skips already-examined items** via a `DeltaManifest` seen-set keyed by arXiv id
   (every graded item, including rejects, is recorded so it is never re-graded);
3. **grades** each new item — keyword taxonomy (`score_paper`) plus a ConceptMatcher
   novelty probe (`_paper_novelty`); on a GPU/embedder outage the novelty probe
   returns `None` and grading degrades to keyword-only rather than failing;
4. enqueues a **`research_paper_fetch`** task for items at/above the relevance
   threshold, with `prio_bucket` derived from the grade — so the highest-graded
   papers are fetched and ingested **first** (priority = queue reordering). The
   fetch task downloads the full paper and ingests it via
   `ResearchPipelineRunner.ingest_paper_full`. Marginal items get a cheap
   abstract-only ingest inline.

Enable/disable or retune it like any schedule via `graph_schedules`.

## Duplicate-tick safety (coalesce + collapse)

A scheduled job is an *interval tick*, not a backlog item — running a stale missed
tick adds no value. Two mechanisms keep the queue from accumulating duplicates:

- **Coalesce (per-schedule, at enqueue):** a tick is not enqueued while a prior
  tick for the same schedule is still un-consumed (CONCEPT:OS-5.44).
- **Collapse (self-healing, at each tick):** `collapse_stale_ticks` cancels any
  schedule's *active* duplicate ticks down to ≤1, recovering from a backlog that
  pre-dates the coalescer or a window where its probe failed (CONCEPT:OS-5.53).
  `running` ticks are never touched.

This pairs with the **best-effort lane cap** (CONCEPT:ORCH-1.82): the `maint` lane
is capped at its floor coverage so a tick backlog can never crowd the throughput
lanes. See [Ingestion Throughput](../architecture/ingestion_throughput.md).

## Verifying end-to-end

```bash
# 1. trigger the feed now and watch the queue
graph_schedules action=run_now name=research_feed
graph_query "MATCH (t:Task {type:'research_paper_fetch'}) RETURN t.id, t.prio_bucket, t.status"

# 2. re-run: already-seen items are skipped (seen_skipped > 0)
graph_schedules action=run_now name=research_feed

# 3. the ingested papers land as Documents
graph_query "MATCH (a) WHERE a.id STARTS WITH 'article:scholarx:' RETURN count(a)"
```

See also: [Delta-based ingestion](delta-ingestion.md),
[the gateway daemon map](../architecture/gateway_daemon.md),
[the Loop engine](../guides/loop-engine.md).
