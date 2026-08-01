# Design Document: Reproducible retrieval evaluation needs a frozen corpus AND a way to mine the hard negatives that expose a retriever's real weaknesses

CONCEPT:AU-KG.retrieval.fixed-corpus-evaluation ·
CONCEPT:AU-KG.retrieval.hard-negative-mining

> `agent_utilities/knowledge_graph/retrieval/evaluation_corpus.py` (primary
> decision), `agent_utilities/knowledge_graph/retrieval/
> hard_negative_miner.py` (pointer). Both cite
> `docs/pillars/2_epistemic_knowledge_graph/KG-2.3-Graph_Integrity_And_Retrieval.md`
> and BrowseComp-Plus (arXiv:2508.06600).

## Decision — retrieval scope can be frozen to a curated, immutable document set, so a benchmark result is reproducible rather than a moving target

`evaluation_corpus.py:4-9` states the motivating claim directly, citing
BrowseComp-Plus: "fixed corpora are essential for fair, reproducible
evaluation of deep-research agents." `EvaluationCorpus` names a set of
document ids (with optional query-answer pairs); `CorpusManager` gives it
CRUD plus **freeze semantics that make a corpus immutable**; and
`HybridRetriever.retrieve_hybrid()` accepts a `corpus_id` to restrict search
scope to exactly that frozen set.

**The rejected alternative** is evaluating retrieval quality against the
live, continuously-ingesting KG: a benchmark run today and the same benchmark
run next week are implicitly measuring different corpora as new documents
land, so a score change cannot be cleanly attributed to a retrieval-algorithm
change versus a corpus change. Freezing the evaluation scope makes the
distinction possible — the same `corpus_id`, re-run, is a controlled
comparison.

## Pointer — `CONCEPT:AU-KG.retrieval.hard-negative-mining`

`hard_negative_miner.py:4-50`. Mines *hard negatives* — documents that match
a sub-query but not the full query — from the SAME query-decomposition →
multi-retrieval → filter pipeline BrowseComp-Plus specifies:
`HardNegativeMiner` reuses `HybridRetriever`'s existing `_decompose_query()`
to break a complex query into sub-queries, fetches results per sub-query, and
flags documents that surface for a sub-query but not the composed one as
`HardNegative` records (with the triggering sub-query and score attached).
**What this concretely adds over the frozen corpus alone**: a fixed corpus
makes a benchmark score reproducible, but says nothing about *why* the score
is what it is. Hard negatives are exactly the failure cases that look
plausible enough to fool a retriever on a partial signal — they are the
calibration data a retriever precision fix would target, and mining them
automatically from decomposition (rather than hand-curating adversarial
examples) is what keeps the process cheap enough to run continuously.
Deliberately gated behind `KG_ENABLE_HARD_NEGATIVE_MINING` (default `false`)
— **the rejected alternative here is running it unconditionally**: mining is
extra retrieval work per query (one retrieval per sub-query, on top of the
composed query), so it is opt-in rather than a tax on every retrieval call.

## Risk Assessment

- **Blast Radius**: `evaluation_corpus.py`, `hard_negative_miner.py`,
  `hybrid_retriever.py` (`retrieve_hybrid`'s `corpus_id`/`hard_negatives`
  parameters, `_decompose_query`).
- **Backward Compatible**: Yes — `corpus_id=None` (unscoped) and
  `KG_ENABLE_HARD_NEGATIVE_MINING=false` are both the pre-existing behavior;
  both features are additive opt-ins.
- **Known weak point**: hard-negative mining's signal quality depends
  entirely on `_decompose_query()`'s decomposition being a faithful breakdown
  of the composed query — a poor decomposition produces sub-queries whose
  "false positives" are not actually representative hard negatives for the
  real query.
