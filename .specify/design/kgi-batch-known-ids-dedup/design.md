# Design Document: Batch dedup is ONE engine round-trip, never N per-item checks

> `agent_utilities/automation/worldmodel_pipeline.py:87-95` (`run_gated_ingest`),
> `agent_utilities/automation/worldmodel_pipeline.py:266-275` (`_batch_known_ids`).

CONCEPT:AU-KG.ingest.instead

## Decision — `graph.has_batch` once per sweep, not `has_node` once per item

The marker text reads `CONCEPT:AU-KG.ingest.instead` because it sits mid-sentence
("... in ONE engine round-trip (CONCEPT:AU-KG.ingest.instead) instead of N
items × ~4 per-item `has_node` round-trips") — the machine triage flagged the
bare id as a likely generic-noun retire candidate, but reading the site shows
it names a real, load-bearing decision: this is the world-model review
plane's dedup check, and the review plane is explicitly called out as "the
50k/hr hot path" — its known-check must be O(1) round-trips, not O(N).

**The rejected alternative, named directly in the code**: checking each
drained document individually — N items × roughly 4 per-item `has_node`
round-trips (one per candidate node-id key a document might already occupy).
At 50k items/hour that is tens of thousands of extra engine round-trips
purely for existence checks, before any actual ingestion work happens.

**The design chosen**: `_batch_known_ids` (`worldmodel_pipeline.py:266`)
collects every candidate node-id key across the WHOLE batch up front, then
asks the engine once via `graph.has_batch`, mapping the present keys back to
their owning doc ids. `run_gated_ingest` calls this exactly once per sweep
(`worldmodel_pipeline.py:91`), before any per-item scoring/tiering begins.

**Graceful degradation preserves correctness over speed when the fast path is
unavailable**: `_batch_known_ids` returns `None` when the engine lacks bulk
existence support (no `has_batch`), and each item then falls back to
per-item `_is_known` — the old N-round-trip behavior is kept as a working
fallback rather than being deleted, so an engine backend without batch
support still functions correctly, just without the throughput win.

## Risk Assessment

- **Blast Radius**: `agent_utilities/automation/worldmodel_pipeline.py`
  (`WorldModelPipelineRunner.run_gated_ingest` and `_batch_known_ids`).
- **Backward Compatible**: Yes — the per-item fallback path is preserved
  verbatim for engines without `has_batch`.
- **Breaking Changes**: None.
- **Known weak point**: the batch-vs-per-item behavior is a silent runtime
  branch keyed on engine capability (`has_batch` presence) — there is no
  visible signal in the sweep's report distinguishing "ran the fast batched
  path" from "fell back to N round-trips", so a capability regression on the
  engine side (e.g. `has_batch` silently disappearing after a backend
  migration) degrades throughput without raising any alarm.
