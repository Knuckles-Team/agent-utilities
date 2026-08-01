# Design Document: Subagent interaction is a formalized four-tier taxonomy, with pattern-selection confidence learned from recorded outcomes

CONCEPT:AU-ORCH.execution.active-subagent-lifecycle

> `agent_utilities/graph/subagent_patterns.py` — `SubagentPattern`,
> `SubagentPatternRouter.select_pattern`, `_adjust_from_history`,
> `record_outcome`, `get_infrastructure_mapping`.

## Decision — four named interaction patterns, each mapped onto EXISTING infrastructure, with a router that self-adjusts its confidence from recorded historical outcomes

The module formalizes subagent interaction as one of four tiers, ordered by
increasing coordination cost (`subagent_patterns.py:4-19`):
`INLINE_TOOL` (single specialist, direct tool call) → `FAN_OUT` (parallel
dispatch + aggregation) → `AGENT_POOL` (persistent pool with advisory
messaging) → `TEAMS` (cross-agent A2A collaboration). Each pattern maps to
infrastructure that already existed independently before this taxonomy:
`INLINE_TOOL` → the single-specialist `executor.py` path, `FAN_OUT` →
`SwarmPresetEngine`, `AGENT_POOL` → `Council`, `TEAMS` → `A2AClient`. The
taxonomy's job is not to build new execution machinery — it's to give one
consistent name and selection policy to four mechanisms that already existed
as separate, disconnected code paths.

`SubagentPatternRouter.select_pattern` picks a pattern from task features
(complexity, parallelizability, collaboration need, specialist count) and
returns a `SubagentPatternDecision` with a `confidence` score and free-text
`reasoning`. Every decision is persisted to the KG as a
`SubagentPatternDecision` node (`_persist_decision`), and — this is the part
that earns its own concept rather than being a static rules table —
`_adjust_from_history` (`subagent_patterns.py:300-345`) blends the pattern's
base confidence with its **actual historical success rate** recorded from
past `record_outcome` calls: `adjusted = 0.7 * base_confidence + 0.3 *
historical_rate`, gated on a minimum sample size of 3 prior decisions for
that pattern before history is trusted at all. It tries a Cypher aggregate
query against the KG backend first and falls back to an O(N) NetworkX scan of
in-memory graph nodes if that's unavailable or errors — the same decision
made resilient to which backend is live.

**The rejected alternative is a static, hand-tuned confidence per pattern
that never changes.** That was the obvious simpler design — pick sensible
starting weights and ship them — and it loses because a pattern's real-world
reliability depends on the specific fleet/task mix a given deployment
actually sees, which a fixed number can't capture. The 70/30 blend with a
sample-size floor is the compromise: confident in the prior for
under-sampled patterns (avoids overreacting to 1-2 data points), increasingly
led by observed outcomes as evidence accumulates. `record_outcome`
(`subagent_patterns.py:410-424`) is explicit that a KG write failure while
recording an outcome only drops that ONE outcome from future
`_adjust_from_history` stats — it never falsely marks a decision as
successful, keeping the learning signal honest even when the KG write path
degrades.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/subagent_patterns.py` only —
  consumers (executor.py, SwarmPresetEngine, Council, A2AClient) are read
  through the mapping, not modified by this module.
- **Backward Compatible**: Yes — a fresh deployment with zero recorded
  outcomes uses pure base confidence (sample size never reaches 3), so
  behavior degrades gracefully to the pre-learning static baseline.
- **Known weak point**: the `_adjust_from_history` Cypher-first / NX-fallback
  split means the confidence-adjustment logic exists in two independently
  maintained implementations that must stay behaviorally identical (same
  blend formula, same sample-size gate) — a change made to one path and
  forgotten in the other would silently make pattern selection backend-
  dependent.
