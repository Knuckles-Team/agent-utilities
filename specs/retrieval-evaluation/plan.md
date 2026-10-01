# AU-RETRIEVAL-001 — Design

## Existing wiring and authority

Reuse `agent_utilities/knowledge_graph/retrieval/evaluation_corpus.py` for frozen query/doc sets, `hierarchical_document_retriever.py` for hierarchy, `context_compiler.py` for agent context, and `agent_utilities/harness/citation_tracker.py` for citation scoring. The current `HybridRetriever.retrieve_hybrid()` is a comparison input while production retrieval authority migrates to EG; do not add another durable search index in AU. The actual served EG hybrid baseline uses the public EG client and disposable engine, with same snapshot and actor scope. Offline lexical results must be labelled surrogate evidence.

## Corpus and metrics

Use a checked-in fixture manifest with `corpus_version`, document IDs, revision/content SHA-256, tree edges, gold span IDs, query IDs, split and visibility labels. Validate no train/test leakage by document family. Freeze before measurement. An optional external dataset adapter first proves redistribution and evaluation rights; missing license or download records `UNAVAILABLE` and leaves the core suite runnable. For each baseline, persist method/version, fixture digest, query-level rankings, visible candidates, token budget, answer citations, elapsed time and update bytes/time. Compute recall and nDCG from the same gold labels, paired bootstrap confidence intervals from held-out queries, and stale/moved citation rates.

## Integration and failure boundaries

CLI/harness evaluation → corpus loader → tenant-filtered candidate provider → each baseline → common metric scorer → report. Use EG's public generated client for served retrieval and EG visibility policy; AU receives typed search results and compiles answers. Denied or missing visibility, stale document hash, unavailable EG, malformed tree, missing gold span or mismatched corpus digest fail explicitly. Never silently substitute BM25 for a served hybrid result. A planned trained selector needs its own model artifact, training split, license and revision before comparison; lexical heuristics must retain their own label.

## Decision and quality

Predeclare a held-out adoption criterion of Recall@1 improvement whose paired 95% confidence interval lower bound exceeds zero, no Recall@3 or nDCG regression, zero cross-tenant citations, and p95 latency/update-cost within explicitly published budgets. If sample size cannot support confidence, report inconclusive. KISS favors one common scorer and existing retrieval ports. Run focused evaluation tests, `python3 scripts/check_dupehound.py`, `python3 scripts/check_duplication.py diff` (jscpd), `python3 scripts/check_complexity_staged.py` (CCCC cyclomatic 10/cognitive 15 caps), and full AU tests. Record exact merged revision and served receipt; no invented green metrics.
