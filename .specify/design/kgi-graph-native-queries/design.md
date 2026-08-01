# Design Document: First-class graph nodes enable queries and aggregations an opaque trace store can't do

> `agent_utilities/harness/trace_analytics.py` (trace/observability queries),
> `agent_utilities/knowledge_graph/extraction/fact_extractor.py` (corroborated
> fact-confidence aggregation) — two independent realizations of the same
> underlying advantage: because the relevant data is a first-class typed KG
> node/edge rather than an opaque row, the engine can graph-reason over it.

CONCEPT:AU-KG.ingest.observability-queries-opik-cannot

## Decision 1 — traces, scores, and generations are typed KG nodes, so the engine graph-reasons over them

`trace_analytics.py:1-19`.

**The rejected alternative, named directly in the module docstring's framing**:
leaving traces/online-scores/assertion-verdicts/generations/prompt-versions
as opaque ClickHouse rows — queryable by an observability tool like Opik, but
not graph-reasonable (no traversal, no cross-entity join through typed
relationships).

**The design chosen**: because these are all first-class KG nodes, three
queries surface what an opaque store cannot: `trace_rootcause` (every FAILED
assertion / low online-score with its trace's agent, grouped by agent —
"what is failing and where does it come from"), `prompt_regression` (mean
online-score per prompt version via `GenerationNode.prompt_version_id` →
trace → scores), and `failure_cluster` (failing traces clustered by which
assertion failed, surfacing systemic breaks shared across ≥N agents — the
pile-attack triage signal). All three read via `backend.execute(<cypher>)`,
the SAME path the eval corpus uses, matching on the canonical `node_type`
property — no separate query engine. Every query degrades to empty results
when no backend/query is available, rather than erroring.

## Decision 2 — repeated fact mentions corroborate into one higher-confidence edge, not N separate low-confidence edges

`fact_extractor.py:591-614` (`aggregate_confidence`, `persist_facts`).

**The rejected alternative**: averaging repeated mentions of the same
`(subject, predicate, object)` triple down, or persisting each mention as a
separate edge. Either loses the signal that independent corroboration should
INCREASE confidence, not dilute it.

**The design chosen**: `aggregate_confidence` combines per-mention
confidences via the product-complement formula `1 − ∏(1 − cᵢ)` — two
independent 0.5-confidence mentions combine to 0.75, three to 0.875,
reflecting that independent weak corroboration reinforces rather than
averages. `persist_facts` then merges repeated mentions of the same triple
into ONE edge: `confidence` is the product-complement aggregate,
`support_count` is the number of corroborating mentions, and `weight` is set
to that count so well-supported edges outrank singleton mentions when
ranked — populating fields the engine's `EdgeData` already carries, not a
new schema. Aggregation happens client-side over the in-batch facts (already
resident), costing no extra engine round-trips; already-known duplicates
(`is_duplicate`) are skipped entirely.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/trace_analytics.py`,
  `agent_utilities/knowledge_graph/extraction/fact_extractor.py`.
- **Backward Compatible**: Yes — both are additive query/aggregation
  utilities over existing typed data; neither changes the underlying schema.
- **Breaking Changes**: None.
- **Known weak point**: the product-complement confidence formula assumes
  mention INDEPENDENCE — if multiple mentions of the same fact trace back to
  the same underlying source (e.g. the same document re-ingested, or several
  extraction passes over correlated text), the aggregate confidence is
  inflated beyond what genuinely independent corroboration would justify;
  nothing in `aggregate_confidence` detects or discounts correlated mentions.
