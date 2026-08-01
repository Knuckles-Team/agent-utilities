# Design Document: Specialist outputs compete for broadcast, they are not accepted equally

CONCEPT:AU-ORCH.execution.global-workspace-attention ·
CONCEPT:AU-ORCH.execution.consensus-aggregation

> `agent_utilities/graph/workspace_attention.py` (primary — module docstring
> lines 4-27, `WorkspaceAttention` class from line 116). The consensus pointer
> is a delegation into `agent_utilities/graph/coordination.py`'s
> `aggregate_scores()`.

## Decision — an always-on Global-Workspace-Theory attention mechanism scores, ranks, and filters specialist outputs before they reach the KG

`CONCEPT:AU-ORCH.execution.global-workspace-attention`

The module docstring states the mechanism directly: "Inspired by Global
Workspace Theory (GWT), adaptive_agent_router submit proposals that are
scored and ranked before integration into the final response"
(`workspace_attention.py:4-19`). Each specialist's output is wrapped as a
`Proposal` with a tri-score — relevance (embedding cosine similarity to the
query), confidence (self-reported, parsed from the output), and track record
(read from the persistent self-model) — combined into one composite score
(`0.5 * relevance + 0.3 * track_record + 0.2 * confidence`,
`WorkspaceAttention` docstring, line 133). `select_winners()` (242-270) keeps
only the top-`max_broadcast_slots` proposals; `broadcast_to_kg()` (286-340)
persists just the winners as `ProposalNode`s, which
`get_attention_score()` (370-410) later reads back as a specialist's runtime
standing.

**The rejected alternative** — accepting every specialist's output equally —
is named directly in the class docstring: "Instead of accepting all
specialist outputs equally, this mechanism scores each output by relevance,
confidence, and track record, then selects the top-K for integration into
the final response" (`workspace_attention.py:116-123`). Accept-all loses on
two counts the code is built around: (1) low-relevance or low-confidence
noise from a specialist would dilute the final synthesis undifferentiated
from a strong contribution, and (2) there would be no persistent
win/lose signal to write back to the KG, so the self-model's track-record
scoring (used by the very next `collect_proposals()` call) would have
nothing to learn from. The mechanism is deliberately always-on rather than
conditional — "Cost: ~50ms per query... Always-on for consistent quality
improvement" (lines 18-19) — because gating it behind a heuristic ("only
filter when uncertain") would reintroduce exactly the inconsistency the
design exists to remove.

A second, narrower decision lives in the same module and is worth recording
here rather than treating as incidental: the write side (`broadcast_to_kg`)
and read side (`get_attention_score`) **must** operate on the same engine
instance for the GWT loop to reinforce itself. The module carries dedicated
process-wide telemetry (`_GwtTelemetry`, lines 51-72) specifically to catch
the silent failure mode where they don't: `suspected_engine_mismatch` flips
true when proposals have been broadcast and reads keep missing with zero
hits (`workspace_attention_telemetry()`, 75-91; `_maybe_flag_engine_mismatch`,
412-436), warning once or raising under `AGENT_UTILITIES_GWT_STRICT`.

### Pointer — `CONCEPT:AU-ORCH.execution.consensus-aggregation`

Grounded at `workspace_attention.py:262-282` (`WorkspaceAttention.select_winners()`
/ `consensus_score()`). The decision: computing a consensus figure over the
winning proposals' composite scores is done via a **named operator**
(mean/median/max/min/log_pool) delegated to `aggregate_scores()` in
`graph/coordination.py`, rather than reimplementing the statistics inline.
The comment at the call site is explicit: "named-aggregation consensus over
winners' scores, via the coordination layer's aggregation registry (STRATEGY
synergy #2)" (line 262-263), and `consensus_score()`'s own docstring repeats
it: "Delegates to the coordination layer's aggregation registry so winner
consensus, coordination aggregation, and selection share one taxonomy"
(274-278).

**Rejected alternative**: a local `statistics.mean(...)` (or equivalent)
call inline in `workspace_attention.py`. It loses because it would create a
second, independent aggregation implementation alongside
`CoordinationLayer.aggregate()` (used for coordinated multi-agent outputs
elsewhere) — two call sites that could silently diverge, e.g. one supporting
`log_pool` (the geometric mean used for combining independent probabilities)
and the other not. Sharing one registry means GWT winner consensus and
general multi-agent coordination aggregation are provably the same
operation, not two that happen to compute the same thing today.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/workspace_attention.py`,
  `agent_utilities/graph/coordination.py` (`aggregate_scores` import).
- **Backward Compatible**: Yes.
- **Known weak point**: the engine-identity coupling between the write side
  (`broadcast_to_kg`) and the read side (`get_attention_score`) is only
  detected after the fact via the miss-counting telemetry — a fresh
  mismatch produces silent zero-score reads for at least
  `_MISMATCH_WARN_AFTER_MISSES` (3) reads before a warning (or exception
  under strict mode) ever fires.
