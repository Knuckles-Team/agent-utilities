# Design Document: Per-agent token usage is a native engine tsdb range query, not a Python re-scan

> `agent_utilities/mcp/tools/write_ingest_tools.py:1568-1585` (the `series`
> action), `agent_utilities/observability/token_tracker.py` (`query_token_series`).

CONCEPT:AU-KG.ingest.per-agent-token-usage

## Decision — `series` reads the engine's native tsdb window, never re-scans usage records in Python

`write_ingest_tools.py:1575-1585`.

**The problem**: per-agent token usage over time is a time-windowed
aggregation query — exactly the kind of workload that gets slow and
memory-heavy if implemented as "load all usage rows for the time range into
Python, then bucket/sum them" as the corpus of recorded usage events grows.

**The rejected alternative, named directly in the code comment**: a "Python
re-scan" — pulling raw usage rows and aggregating them in application code.
Rejected because it does not scale with usage-history volume and duplicates
aggregation logic the engine's own timeseries store already provides
natively.

**The design chosen**: `query_token_series` reads the engine's NATIVE
range/window tsdb query — `from_date`/`to_date` (epoch seconds) bound the
window, `model` selects the bucket field (default `total_tokens`), `limit`
carries the window/bucket size in seconds (`0` = raw points, no bucketing),
and `agent` is the series key. The aggregation (bucketing/summing over the
window) happens inside the engine's timeseries store, not in the MCP tool's
Python process — the tool is a thin parameter-validation + query-dispatch
layer over it.

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/tools/write_ingest_tools.py`
  (`series` action), `agent_utilities/observability/token_tracker.py`.
- **Backward Compatible**: Yes — an additive read-only query action.
- **Breaking Changes**: None.
- **Known weak point**: because aggregation is delegated entirely to the
  engine's native tsdb, this tool's own behavior is only as capable as
  whatever bucketing/window semantics the engine's timeseries store exposes
  — extending the query shape (e.g. a new aggregation function) requires an
  engine-side change, not just a tool-side one.
