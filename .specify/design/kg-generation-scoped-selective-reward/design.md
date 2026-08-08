# Design Document: Learned reward is erased by PROVENANCE — when an id's embedding changes generation — not by age, and not by rebuilding the index

CONCEPT:AU-KG.memory.generation-scoped-selective-reward

> Realised by `agent_utilities/knowledge_graph/retrieval/capability_index.py:143-157`
> (`_REWARD_REGEN_DISTANCE` and its rationale comment), `:546-564` (the erasure
> branch inside `CapabilityIndex.add()`) and `:1086`
> (`selective_erase_rewards`). Introduced by commit `192108c3`; the full
> trade-off analysis is in
> `.specify/reports/memory-2606.26294-comparative-analysis-2026-06-28.md` §4-5.

## Decision — treat "the thing this id refers to has been replaced" as the trigger for forgetting its reward

`CapabilityIndex.add()` is an upsert: calling it with an existing id replaces
the stored embedding. The reward EMA attached to that id was left untouched by
that replacement.

The introducing commit names the gap precisely:

> *"The memory maintenance quadrant forgot learned reward only by AGE
> (`decay_rewards`) or idle/max-age reapers — never by PROVENANCE ... the
> in-place `CapabilityIndex.add()` upsert replaced the vector but kept the
> stale reward EMA."*

The failure this produces is specific and bad: a capability is rewritten into
something substantively different, keeps its id, and inherits the reputation
the *old* content earned. The index then confidently ranks a new, unproven
capability highly on evidence that no longer applies to it.

The fix adds a provenance trigger. On upsert, cosine distance between the new
and stored embedding is compared to `_REWARD_REGEN_DISTANCE = 0.25`; beyond
that, the id is treated as a new generation and its reward EMA is reset to the
neutral prior (`:546-564`).

**Three alternatives were considered and rejected, and the analysis report
tabulates why:**

- **Age-based decay (`decay_rewards`, the existing mechanism).** Rejected as
  answering a different question — it forgets by *age, not provenance*. A
  capability rewritten five minutes ago still has fresh, high reward; time-based
  decay cannot see the rewrite at all.
- **Full index rebuild.** Rejected as *"over-erases"* — it *"nukes all
  reward"*. It does fix the stale-reward case, by discarding every id's learned
  reward including the overwhelming majority that were not rewritten and whose
  reputation is still valid. The correct scope is the ids whose provenance
  actually changed.
- **RQGM's full epoch / frozen-evaluator / ε-best-belief machinery.**
  Explicitly *"Rejected"* in the report — it *"re-implements GM base-model
  search we intentionally don't do"*, i.e. it solves this problem as a
  by-product of a much larger apparatus this system has deliberately chosen not
  to adopt.

Resetting to the **neutral prior** rather than to zero is the last piece: a new
generation should be unproven, not penalised, or it could never earn its way
back into the rankings it needs to be selected for.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/retrieval/capability_index.py`.
- **Backward Compatible**: Yes — additive on the upsert path.
- **Known weak point**: `_REWARD_REGEN_DISTANCE = 0.25` is a single global
  threshold standing in for "is this semantically the same thing". A
  substantive rewrite that happens to stay lexically close keeps reward it no
  longer deserves; a cosmetic reindex that crosses the threshold discards
  reward it did deserve. The constant is documented but not tuned per domain,
  and nothing measures the false-positive/false-negative rate of the trigger.
