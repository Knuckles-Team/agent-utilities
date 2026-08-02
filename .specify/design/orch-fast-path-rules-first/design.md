# Design Document: The fast-path classifier is rules-first and structural — a turn escalates only on a concrete signal, never because it failed to match a greeting

CONCEPT:AU-ORCH.routing.original-rule-was-far

> Realised by `agent_utilities/graph/routing/strategies/fast_path.py:1-22`
> (module docstring), `:63-89` (`needs_full_orchestration`) and `:120-141`
> (`is_trivial_query`). Introduced by commit `b794e6af`
> ("perf(orchestration): chat execution profile + non-blocking reply path"),
> which describes it as *"ORCH-1.63 widened fast-path classifier."*

## Decision — invert the default: take the fast path unless a structural signal demands the full graph

The concept id is awkward because it was slugified from the first line of the
module's own explanation, but the decision it names is precise and the module
states it against its predecessor directly.

The original rule admitted a turn to the fast path only if it was *"≤6 words
AND starting with a fixed greeting prefix"* (`fast_path.py:10-16`). That is an
allow-list of one shape. The docstring names the consequence: *"a normal simple
question ... did NOT qualify and paid for the full graph."* The classifier was
nominally an optimization but fired almost never, because real user turns are
rarely six-word greetings.

The replacement inverts the polarity. `needs_full_orchestration` (`:63-89`)
escalates only on a **concrete structural signal**: a slash command, more than
40 words, or multiple clauses. Everything else takes the fast path. The
distinguishing property is that all three signals are properties of the text's
*structure*, observable without knowing anything about the domain.

**A second alternative was tried and deleted, and its deletion is the more
interesting half of this decision.** An `_ESCALATION_KEYWORDS` list — a
hardcoded domain vocabulary that escalated any turn mentioning a known
capability — was removed, with the reason recorded at `fast_path.py:46-51`:
*"an unbounded word list is the wrong gate — it both missed real capabilities
... and could not name the fleet."* A keyword list fails in both directions at
once. It misses capabilities nobody thought to add, and it can never enumerate
a fleet that grows at runtime, so it degrades silently as the system it guards
expands. Structural signals do not have that failure mode: "more than 40 words"
means the same thing next year.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/routing/strategies/fast_path.py`;
  every turn's routing decision passes through it.
- **Backward Compatible**: No, deliberately — far more turns take the fast path
  than before. That is the entire purpose.
- **Known weak point**: the failure mode moved rather than disappeared. The old
  rule was too conservative (expensive, always correct); the new one is
  permissive, so a *short, single-clause* turn that genuinely needs tools now
  takes the fast path and gets a worse answer. The 40-word and multi-clause
  thresholds are unvalidated constants, and nothing currently measures fast-path
  turns that should have escalated.
