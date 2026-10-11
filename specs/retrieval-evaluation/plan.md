# AU-RETRIEVAL — Design

## Existing wiring and authority

Reuse `agent_utilities/knowledge_graph/retrieval/evaluation_corpus.py` for frozen query/doc sets, `hierarchical_document_retriever.py` for hierarchy, `context_compiler.py` for agent context, and `agent_utilities/harness/citation_tracker.py` for citation scoring. The current `HybridRetriever.retrieve_hybrid()` is a comparison input while production retrieval authority migrates to EG. Do not add another durable search index in AU. The actual served EG hybrid baseline uses the public EG client and disposable engine, with same snapshot and actor scope. Offline lexical results must be labelled surrogate evidence.

## Corpus and metrics

Use a checked-in fixture manifest with `corpus_version`, document IDs, revision/content SHA-256, tree edges, gold span IDs, query IDs, split and visibility labels. Validate no train/test leakage by document family. Freeze before measurement. An optional external dataset adapter first proves redistribution and evaluation rights; missing license or download records `UNAVAILABLE` and leaves the core suite runnable. For each baseline, persist method/version, fixture digest, query-level rankings, visible candidates, token budget, answer citations, elapsed time and update bytes/time. Compute recall and nDCG from the same gold labels, paired bootstrap confidence intervals from held-out queries, and stale/moved citation rates.

## Integration and failure boundaries

CLI/harness evaluation → corpus loader → tenant-filtered candidate provider → each baseline → common metric scorer → report. Use EG's public generated client for served retrieval and EG visibility policy. AU receives typed search results and compiles answers. Denied or missing visibility, stale document hash, unavailable EG, malformed tree, missing gold span or mismatched corpus digest fail explicitly. Never silently substitute BM25 for a served hybrid result. A planned trained selector needs its own model artifact, training split, license and revision before comparison. Lexical heuristics must retain their own label.

## Decision and quality

Predeclare a held-out adoption criterion of Recall@1 improvement whose paired 95% confidence interval lower bound exceeds zero, no Recall@3 or nDCG regression, zero cross-tenant citations, and p95 latency/update-cost within explicitly published budgets. If sample size cannot support confidence, report inconclusive. KISS favors one common scorer and existing retrieval ports. Run focused evaluation tests, `python3 scripts/check_dupehound.py`, `python3 scripts/check_duplication.py diff` (jscpd), `python3 scripts/check_complexity_staged.py` (CCCC cyclomatic 10/cognitive 15 caps), and full AU tests. Record exact merged revision and served receipt; no invented green metrics.

## Iterative memory expansion (AU-RETRIEVAL-R003)

Data flow: `served memory retrieval -> retrieve_hybrid_budgeted (knowledge_graph/retrieval/hybrid_retriever.py:1898) -> memory expansion (cue -> tag -> content) -> answer or RetrievalAbstention -> retrieval trace`.

Types: `ExpansionBudget`, `ExpansionPath` and `RetrievalAbstention`. The hybrid retriever and the token-budget cut are reused. The expansion loop and its call site are new.

Failure: an exhausted budget abstains with the partial path. An unreachable graph returns the existing typed unavailable result.

Decision: a declared budget bounds cost and makes the stop reason auditable. The rejected alternative is an unbounded walk with a score cut-off, which has no cost ceiling.

References:
- arXiv:2606.06036 — Ji et al., "Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents" (2026)

## Insufficient evidence and strict fallback (AU-RETRIEVAL-R004)

Data flow: `served executable_rag action -> _analysis_action_executable_rag (mcp/tools/analysis_tools.py:2280) -> HybridRetriever.retrieve_executable (hybrid_retriever.py:1917) -> ExecutableRagProgram.run (executable_rag.py:236) -> _retrieve_with_repair (executable_rag.py:198) -> answer step (executable_rag.py:256) -> answer or RetrievalAbstention -> StepTrace`.

Types: `EvidenceSufficiency`, `FallbackPolicy`, and the `RetrievalAbstention` of R003.4. Failure: `strict` abstains after `max_repairs` rounds.

Existing code: `knowledge_graph/retrieval/executable_rag.py:87`, `executable_rag.py:198`, `executable_rag.py:236`, `hybrid_retriever.py:1917`, `hybrid_retriever.py:1957`, `mcp/tools/analysis_tools.py:2294`. Reachable today: yes, for repair; no, for a typed verdict or strict abstention.

Reused: the k boost and mode fallback. New: the typed verdict, the policy and the abstention. Decision: the policy is explicit per call. The rejected alternative is always abstaining, which loses useful partial answers.

## Calibrated confidence margin for stopping (AU-RETRIEVAL-R005)

Data flow: `served query -> engine_query.py:1509 -> IterativeQueryExpander.run (iterative_expansion.py:168) -> IterativeStopper.update (adaptive_stopping.py:151) -> StopDecision(reason, margin) -> trace`. The reconstruction path calls `update` at `active_reconstruction.py:282` from `entity_context.py:153`.

Types: `RoundConfidence` and the `margin` field on `StopDecision`. Failure: a missing confidence keeps the repeat-only rule. The stop reason is traced.

Existing code: `adaptive_stopping.py:39`, `adaptive_stopping.py:80`, `adaptive_stopping.py:151`, `iterative_expansion.py:171`, `iterative_expansion.py:229`, `active_reconstruction.py:237`, `active_reconstruction.py:282`, `engine_query.py:1509`. Reachable today: yes for the repeat rule; no for the margin, because no caller passes a confidence.

Decision: a fixed calibration table keeps the rule training-free. The rejected alternative is a learned stopping head, which needs training data and a model artifact.

References:
- arXiv:2606.13814 — Kieback et al., "TASR: Training-Free Adaptive Stopping for Iterative Retrieval" (2026)

## Prior-versus-evidence arbitration (AU-RETRIEVAL-R006)

Data flow: `served executable_rag action -> retrieve_executable (hybrid_retriever.py:1917) -> ExecutableRagProgram.run answer step (executable_rag.py:256) -> arbiter -> chosen candidate or RetrievalAbstention -> trace`.

Types: `ArbitrationInput`, `ArbitrationResult`, and the classes `parametric` and `retrieved` added to `EvidenceClass` (`api/agent_control_contracts.py:73`). Failure: a missing prior probe skips arbitration and records that.

Existing code: `agent_control_contracts.py:73`, `search_synthesis/shortcut_risks.py:42`, `executable_rag.py:256`, `hybrid_retriever.py:1917`. Reachable today: no. Search terms with no hit: `arbitrat`, `parametric`, `trust margin`, `closed-book`, `prior_answer`. The arbiter is new.

Decision: a fixed margin keeps the choice training-free and auditable. The rejected alternative is always trusting retrieval, which fails when retrieval returns a wrong passage.

References:
- arXiv:2606.08397 — Xu et al., "TrustMargin: Training-Free Arbitration between Parametric Memory and Retrieved Evidence in Large Language Models" (2026)
