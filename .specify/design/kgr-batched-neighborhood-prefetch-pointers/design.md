# Design Document: Recall telemetry and the trivial-turn/fallback-cascade gate — two pointers off the batched-neighborhood-prefetch decision

CONCEPT:AU-KG.retrieval.recall-feedback ·
CONCEPT:AU-KG.retrieval.triviality-gate

> Primary decision (`AU-KG.retrieval.batched-neighborhood-prefetch`,
> `AU-KG.retrieval.fail-closed-grounding-contract`'s sibling
> `CONCEPT:AU-KG.retrieval.fail-closed-grounding-contract`) is already
> documented at
> `.specify/design/retrieval-batched-neighborhood-prefetch/design.md`. This
> document covers the two pointer concepts that decision's cluster names but
> does not itself contain, both inside `agent_utilities/knowledge_graph/
> retrieval/hybrid_retriever.py`'s `plan_and_retrieve`, plus
> `hyde_planner.py`.

## Pointer — `CONCEPT:AU-KG.retrieval.recall-feedback`

`hybrid_retriever.py:1439-1444`. On the live retrieval path,
`plan_and_retrieve` records that the returned node ids were RECALLED
(`self.usage_telemetry.record_recall(...)`) — the first half of the
recall→usage loop `CONCEPT:AU-KG.retrieval.evidence-weighted-memory`'s
`UsageTelemetry`/`bayesian_trust` closes. The usage half (which recalled
nodes actually *informed* the answer) is closed separately by the generation
step calling `record_answer_usage`, so trust is trained from the two counts
together. **The rejected alternative**: computing a static, un-trained
relevance signal per node forever — recording only recall (never usage) would
mean a node retrieved often but never actually used could not be
distinguished from one that is genuinely useful; the split recall/usage
telemetry is what makes that distinction measurable at all.

## Pointer — `CONCEPT:AU-KG.retrieval.triviality-gate`

`hybrid_retriever.py:1380-1390,1432-1437`, `hyde_planner.py:39-90`. Two
related but distinct gates share this id:

1. **The social-closer / triviality gate** (`hyde_planner.py:39-90`,
   `is_trivial_query`) — assimilated from memory-os
   (`scripts/context_enhancer.py:586`). A message that is an exact social
   closer ("ok", "thanks", "bye", …), under 6 ASCII chars, or emoji/symbol-only
   carries no retrievable intent, so `plan_and_retrieve` skips the whole HyDE
   plan + retrieval for it (`hybrid_retriever.py:1381`), saving a planner LLM
   call and embedding work per turn. **The rejected alternative**: running the
   full retrieval pipeline on every turn including "ok" and "thanks" — which
   is not just wasted cost but can surface irrelevant context for a turn that
   has no informational content to ground.
2. **The 4-level fallback cascade** (`hybrid_retriever.py:1432-1437,1484`) —
   when the vector path yields nothing (embeddings/HNSW unavailable or
   offline), the retriever degrades: hybrid → dense-only (already inside
   `retrieve_hybrid`) → lexical keyword scan → backend `CONTAINS` scan. **The
   rejected alternative**: returning an empty result the moment the primary
   vector path is unavailable. The tiered degrade means "a query always
   returns *something* instead of an empty result," at the cost of weaker
   relevance ranking on the degraded tiers — a deliberate availability-over-
   precision trade for exactly the failure mode (embedder/HNSW down) where the
   alternative is a hard stop.

## Risk Assessment

- **Blast Radius**: `hybrid_retriever.py` (`plan_and_retrieve`,
  `_lexical_fallback`), `hyde_planner.py`, `retrieval_quality.py`
  (`UsageTelemetry`).
- **Backward Compatible**: Yes — both are additive gates inside an existing
  method; neither changes the retrieval contract for a non-trivial query on a
  healthy vector path.
- **Known weak point**: the social-closer list (`SOCIAL_CLOSERS`) is a fixed,
  hand-curated set — a genuinely informational turn that happens to be short
  and colloquial ("no wifi") risks being misclassified as trivial and skipping
  retrieval it should have run.
