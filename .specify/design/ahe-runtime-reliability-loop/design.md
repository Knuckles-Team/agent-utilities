# Design Document: A fifth gap-discovery track for runtime events that never became a "run," gated conservative by design

CONCEPT:AU-AHE.harness.runtime-reliability-loop

> `agent_utilities/knowledge_graph/research/runtime_reliability.py`.

## Decision — aggregate persisted `:RuntimeSignal` events into the SAME canonical `:Gap`, but respond with recommendation-only or already-safe heals, never speculative prod mutation

The module docstring (`runtime_reliability.py:4-19`) states the gap this
closes: the existing reward/failure-analyzer flywheel keys off AGENT-RUN
quality and LLM spans, but four runtime signals
(`engine_latency`/`listener_restart`/`retrieval_degraded`/`delegation_over_budget`)
are runtime events that "often never became a *run* at all" — invisible to
every existing discovery track. `runtime_reliability_analyzer` drains
buffered signals, aggregates by `(kind, subject)` over a window, and for a
pattern crossing threshold opens the SAME canonical `:Gap` every other track
uses (`submit_gap`), deduped against already-open gaps.

**The rejected alternative is what existed before: a flywheel blind to
runtime infrastructure failures that never manifest as a scored agent run.**
A second, independent decision governs the response once a pattern IS
detected: the docstring states the division of labor is "deliberately
conservative (recommendation-only or already-safe; NO speculative
auto-mutation of prod)." `runtime_reconciler` handles recognized classes
with a fixed, safe disposition per kind — `listener_restart` is *already*
auto-healed by the messaging supervisor, so it just records a resolved heal
(a closed-loop annotation, not open work); `engine_latency`/
`retrieval_degraded` open a RECOMMENDATION gap ("consider batching/caching",
"review the retrieval budget") rather than mutating any config. **The
rejected alternative here is auto-remediating detected runtime patterns
directly** — e.g. automatically adjusting a retrieval budget or restarting a
component the analyzer itself decided was degraded — which would give a
pattern-detection heuristic unsupervised write access to production
behavior. Unrecognized classes (`delegation_over_budget` and any future
unknown kind) are filed as open "investigate" work for the SDD flywheel
rather than guessed at.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/research/runtime_reliability.py`,
  `agent_utilities/observability/runtime_signals.py`,
  `agent_utilities/knowledge_graph/research/gaps.py` (`submit_gap`),
  `agent_utilities/orchestration/agent_runner.py`,
  `agent_utilities/knowledge_graph/core/engine_breaker.py`.
- **Backward Compatible**: Yes — additive discovery track; existing tracks
  and the gap lifecycle are unmodified.
- **Known weak point**: the fixed per-kind disposition table
  (`runtime_reconciler`) is a closed enumeration — a new runtime-signal kind
  added elsewhere without a corresponding disposition falls into the
  "unrecognized, file as investigate" path even if its correct handling is
  actually well understood, until someone updates the reconciler.
