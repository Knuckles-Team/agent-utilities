# Design Document: Memory trust is trained from recall→usage telemetry with a Bayesian-smoothed score, not a raw helpful/total ratio

CONCEPT:AU-KG.retrieval.evidence-weighted-memory

> `agent_utilities/knowledge_graph/retrieval/retrieval_quality.py:484-559`
> (`bayesian_trust`, `LineageRecord`, `UsageTelemetry`).

## Decision — trust is `(helpful + prior·weight) / (total + weight)`, smoothed toward a neutral prior, not `helpful / total`

`retrieval_quality.py:484-489` (assimilated from memory-os,
`ClaudioDrews/memory-os@a4ca094`) names the gap directly: the existing
retrieval-quality gate above this section SCORES a retrieval, "but nothing
trains those scores." This closes the loop with usage telemetry (did a
recalled memory actually get used?) plus a Bayesian trust score derived from
it, plus a generation-lineage record linking an answer back to the memory ids
it was grounded on. `bayesian_trust`'s docstring states the rejected
alternative directly: the smoothed form "avoids the divide-by-zero /
overconfident swings of a raw helpful/total ratio." A raw ratio is undefined
at `total=0` and swings to a confident 0 or 1 after a single observation (one
use out of one recall reads as "100% trusted," which is not a warranted
conclusion from one data point); the Bayesian prior (`TRUST_PRIOR=0.5`,
`TRUST_WEIGHT=2.0` pseudo-counts) keeps an unseen fact at a neutral prior and
requires accumulated evidence to move meaningfully away from it — "a fact
retrieved many times and usually used trends toward 1.0; one retrieved but
never used trends toward 0.0."

**`UsageTelemetry` splits recall and usage into two separately-recorded
counts** rather than inferring trust from retrieval rank or a single
"was this helpful" signal: `record_recall` logs that ids were surfaced,
`record_usage` logs that an id actually informed the answer, and `trust(id)`
derives the Bayesian score from the two counts together — with
`flush_to_engine` persisting the trained score onto nodes (the previously
unused `store_memory(trust_score=...)` field) so it survives restarts rather
than resetting to the prior every process start. `LineageRecord`/
`build_lineage` closes the audit half: a stable content-hash over the
retrieved id set plus the query, so a specific generation's grounding is
independently verifiable rather than trusted from the trust score alone.

## Risk Assessment

- **Blast Radius**: `retrieval_quality.py` (`UsageTelemetry`,
  `bayesian_trust`, `LineageRecord`), consumed by `hybrid_retriever.py`'s
  `record_recall`/`record_answer_usage` (see
  `.specify/design/kgr-batched-neighborhood-prefetch-pointers/design.md`'s
  recall-feedback pointer).
- **Backward Compatible**: Yes — `trust_score` is a new, previously-unused
  node field; nodes without it read as "at the prior" via `bayesian_trust`'s
  default arguments.
- **Known weak point**: `UsageTelemetry` is in-process by default —
  `flush_to_engine` must be called explicitly to persist; a process that
  accumulates recall/usage counts and crashes before flushing loses that
  cycle's training signal entirely, with no separate durability guarantee.
