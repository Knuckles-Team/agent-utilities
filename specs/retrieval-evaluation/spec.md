# EH-652 — Reproducible structure-aware retrieval evaluation

**Owner:** agent-utilities for evaluation, answer citations and adoption decision. **Status:** SPECIFIED; acceptance NOT_AUDITED. EG owns production graph, hybrid/vector search and visibility enforcement. This spec does not authorize a second AU retrieval engine.

## User outcome

A contributor can run a public, reproducible comparison of constrained section-leaf selection, hierarchical document retrieval and served hybrid retrieval, then decide whether any structure-aware method should be adopted. A tiny lexical experiment or inaccessible third-party corpus cannot support a production improvement claim.

## Requirements

| ID | Required behavior | Acceptance |
|---|---|---|
| FR-1 | Commit a synthetic, licensed public fixture corpus with versioned documents, section tree, moved-section revisions, gold evidence spans, queries and visibility labels; freeze content hashes and splits. | Fresh fork reproduces corpus and metrics without private data or SearchTome access. Optional external datasets require documented license and pinned digest. |
| FR-2 | Compare at least lexical BM25, constrained ToC/section selection, hierarchical retrieval and actual served EG hybrid/vector+graph under identical visible candidate sets and token budgets. | Report per-query Recall@1/3, nDCG, citation validity, p50/p95 latency and index/update cost; name any unavailable baseline separately. |
| FR-3 | Measure cited answer correctness and moved/stale evidence behavior through AU's existing `CitationTracker` and context compiler, with document version/snapshot carried through. | No answer cites an unseen, stale or wrong-tenant span; citation errors are counted, not dropped. |
| FR-4 | Evaluate structure-aware adoption on a held-out set with a predeclared non-regression threshold and confidence interval; separate lexical selector from a trained selector. | Adopt only with a statistically defensible gain and no visibility/citation regression; otherwise retain current behavior. |

The test matrix in [test-spec.md](test-spec.md) covers each requirement. A synthetic fixture is mandatory; SearchTome is optional and can never be a required public build input.
