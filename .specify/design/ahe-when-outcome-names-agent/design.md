# Design Document: Tag an eval case by opaque agent reference when its outcome names the agent, for real per-agent attribution

CONCEPT:AU-AHE.harness.when-outcome-names-agent

> `agent_utilities/knowledge_graph/adaptation/feedback.py:446-476` (primary
> — this is where the rule and its rationale are actually stated; the id
> also appears more loosely on `agent_utilities/harness/memorydata/bakeoff.py:44`
> and `agent_utilities/harness/memorydata/samples.py:4` as the same
> "tag by identity so downstream code can pool correctly" idea applied to
> bake-off cells and sample families).

## Decision — when an action outcome identifies the agent that produced it, tag the eval case with an opaque agent reference, not just its trace signature

The comment at the decision site (`feedback.py:446-449`) states the rule
directly: "when the outcome names the agent that produced it, tag the eval
case with an opaque agent reference so the per-agent trainset
(`build_agent_trainset`) can pool THIS agent's real metrics for its own
native program optimization (attribution by agent, not just trace
signature)." `record_action_outcome` conditionally appends an
`agent_ref:<opaque>` tag (via `opaque_program_reference`) whenever an
`agent_id` is available, and `agent_eval_cases` later slices the corpus by
that same opaque reference to build the per-agent attribution
`harden_agent_prompt` optimizes against.

**The rejected alternative, named directly in the comment, is attributing
eval cases only by trace signature** — grouping cases by which run produced
them rather than by which agent produced them. Trace-signature grouping
can't answer "how has THIS agent specifically performed across many runs,"
which is exactly what per-agent prompt hardening needs: a trainset scoped to
one agent's own accumulated executions. The reference is deliberately opaque
(`opaque_program_reference`, the same indirection pattern used elsewhere in
this codebase for proposal/component references) rather than the raw
`agent_id`, keeping the tag consistent with how other identity references
are handled in this corpus.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/adaptation/feedback.py`,
  `agent_utilities/harness/evolve_agent.py` (`harden_agent_prompt`'s
  `build_agent_trainset`), `agent_utilities/harness/eval_corpus.py`.
- **Backward Compatible**: Yes — tagging is conditional on `agent_id` being
  available; an outcome that doesn't name its agent is tagged exactly as
  before.
- **Known weak point**: tagging is best-effort inside a broad
  `except Exception` (`feedback.py:463-464`, `logger.debug` only) — a
  failure to tag an eval case with its agent reference is silently swallowed
  rather than surfaced, so a systematic tagging failure for one agent could
  go unnoticed and quietly starve that agent's per-agent trainset.
