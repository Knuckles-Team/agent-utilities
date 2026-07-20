# Recipe — Unified scheduling, the priority queue, and the ScholarX RSS research feed

The gateway daemon runs **one** intelligent scheduler (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent). Every
recurring job — the `deploy/schedules.yml` entries, the former fixed-interval
maintenance ticks, the self-evolution `loop_cycle`, and the ScholarX RSS research
feed — is a durable `:Schedule` node. The single scheduler tick evaluates them and
**enqueues** a `scheduled_job` WorkItem onto the native priority+scheduled queue
(CONCEPT:AU-KG.ingest.hardened-priority-scheduled-task) that the worker pool drains. Nothing recurring runs inline in the
scheduler thread anymore.

## The queue (CONCEPT:AU-KG.ingest.hardened-priority-scheduled-task)

A WorkItem carries:

- **`prio_bucket`** — discrete priority `0` (critical) … `3` (background). Workers
  claim the lowest non-empty bucket first (the native graph interpreter strips
  `ORDER BY`, so priority is N equality queries, not a sort). `prioritize_task`
  and the `prio`/`priority` arguments set it.
- **`ready` + `next_retry_at`** — native delayed availability. There is no
  Python promotion writer. Application-level retry/backoff is committed by
  `CommitWorkItemResult`; exhausted attempts become `dead_letter`.
- **`submitted` + `depends_on`** — native dependency gating. Atomic parent
  commit releases the child when `dep_count` reaches zero.
- **lease epoch + fencing token** — `ClaimWorkItem`, renew, commit, cancel, and
  defer are the only lifecycle mutation family. Missing native verbs fail closed.

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

## The ScholarX RSS research feed (CONCEPT:AU-KG.research.scholarx-rss-research-feed)

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
   abstract-only ingest inline. Both tiers commit the complete paper/source/
   pseudonymous-author topology through one engine-native `ChangeEnvelope`; the
   durable fetch task carries only non-reversible author references. A full PDF's
   Document/Chunk projection uses the same native boundary and never stores the
   local file path.

Enable/disable or retune it like any schedule via `graph_schedules`.

## Duplicate-tick safety (coalesce + collapse)

A scheduled job is an *interval tick*, not a backlog item — running a stale missed
tick adds no value. Two mechanisms keep the queue from accumulating duplicates:

- **Coalesce (per-schedule, at enqueue):** a tick is not enqueued while a prior
  tick for the same schedule is still un-consumed (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent).
- **Collapse (self-healing, at each tick):** `collapse_stale_ticks` cancels any
  schedule's *active* duplicate ticks down to ≤1, recovering from a backlog that
  pre-dates the coalescer or a window where its probe failed (CONCEPT:AU-OS.state.stale-tick-collapse).
  `running` ticks are never touched.

This pairs with the **best-effort lane cap** (CONCEPT:AU-ORCH.scheduling.low-value-high-volume): the `maint` lane
is capped at its floor coverage so a tick backlog can never crowd the throughput
lanes. See [Ingestion Throughput](../architecture/ingestion_throughput.md).

## Verifying end-to-end

```bash
# 1. trigger the feed now and watch the queue
graph_schedules action=run_now name=research_feed
graph_query "MATCH (w:WorkItem {fairness_group:'research_paper_fetch'}) RETURN w.id, w.prio_bucket, w.status"

# 2. re-run: already-seen items are skipped (seen_skipped > 0)
graph_schedules action=run_now name=research_feed

# 3. the ingested papers land as Documents
graph_query "MATCH (a) WHERE a.id STARTS WITH 'article:scholarx:' RETURN count(a)"
```

See also: [Delta-based ingestion](delta-ingestion.md),
[the gateway daemon map](../architecture/gateway_daemon.md),
[the Loop engine](../guides/loop-engine.md).
