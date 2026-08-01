# Design Document: Chunk selection and reranking are pluggable and dependency-optional by default — heavy models are an auto-detected upgrade, never a hard import

CONCEPT:AU-KG.retrieval.unset-dependency-free ·
CONCEPT:AU-KG.retrieval.pack-retrieval-signals ·
CONCEPT:AU-KG.retrieval.retrieval-quality-assessment

> `agent_utilities/knowledge_graph/retrieval/score_gate.py` (dual-score
> fusion), `agent_utilities/knowledge_graph/retrieval/
> neural_reranker.py` (optional cross-encoder), `agent_utilities/core/
> config.py:4522-4526` (`KG_RERANK_MODEL`), `agent_utilities/knowledge_graph/
> retrieval/autocut.py`, `agent_utilities/knowledge_graph/retrieval/
> reasoning_reranker.py`.

## Decision — the dependency-free lexical scorer is the ALWAYS-available default; a heavy neural backend is auto-detected, never a required import

`CONCEPT:AU-KG.retrieval.unset-dependency-free`

`neural_reranker.py:17-26` states the pattern directly: "The heavy model
dependency (`sentence-transformers`/`torch`) is OPTIONAL. Importing this
module must never import torch or sentence-transformers; the model library is
touched only lazily, on first scoring or via `is_available`." Wiring code
calls `build_rerank_scorer`, which auto-detects — neural cross-encoder when a
model is injected or the library is importable, else the always-available
lexical proxy — and **no environment is read** to decide this: "the model
name and batch size come from arguments/constants," so the fallback is a
structural property of the import graph, not a runtime config toggle to get
wrong. `config.py:4522-4526` names the operator-facing half:
`KG_RERANK_MODEL` set → score via a remote vLLM `/v1/rerank` endpoint,
"consistent with embeddings/LLM on vLLM"; **unset → the dependency-free
lexical scorer** (or opt-in local neural via `KG_RERANK_LOCAL_NEURAL`).

**The rejected alternative** is a hard dependency on the neural stack: a
serving image that must ship `torch`/`sentence-transformers` (or a remote
reranker endpoint) just to rerank at all. That would make the lean serving
profile impossible and turn every reranking call into a hard failure the
moment the optional dependency or endpoint is unavailable, instead of a
graceful, silent-by-design step down in ranking fidelity. `score_gate.py:4-33`
generalizes this pattern one level further: `ScoreGate` fuses TWO
complementary signals (fast bi-encoder vector score + slower cross-encoder
reranker score) via per-component z-standardization rather than choosing
between them, and — critically — "missing cross-encoder scores fall back to
the bi-encoder score, so the gate degrades gracefully to single-signal
behavior when reranking has not run." The gate itself never requires the
optional signal to be present.

### Pointer — `CONCEPT:AU-KG.retrieval.pack-retrieval-signals`

`autocut.py:6-18`. The single-signal precursor `ScoreGate` explicitly
generalizes: instead of forcing the caller to pick a fixed `top_k`, `autocut`
trims the ranked list at the largest *relative* score drop (the "knee"),
mirroring gbrain's `autocut.ts`. **The rejected alternative, stated
directly**: "instead of forcing the caller to pick a `top_k`" — a fixed cutoff
either truncates a strong result set too early or pads a weak one with noise.
Autocut is conservative by construction: it never trims below `min_results`,
and a flat score distribution (no clear knee) returns the full set rather than
guessing a cut.

### Pointer — `CONCEPT:AU-KG.retrieval.retrieval-quality-assessment`

`reasoning_reranker.py:4-89`. The reranking-stage application of the same
dependency-optional `RerankScorer` protocol `unset-dependency-free` defines:
a second-stage reranker that reorders an over-fetched candidate pool by
query-relevance before it is capped to the context window, distilled from the
MemReranker research (calibrated `[0,1]` five-level scores, instruction-
awareness, prior-blended ranking). The default `LexicalRelevanceScorer` is,
again, "deterministic and dependency-free (no model, no network) so it is
always available and unit-testable" — with a distilled cross-encoder droppable
in later "behind the same protocol without touching the retrieval path." The
protocol is what lets `score_gate.py`, `neural_reranker.py`, and
`reasoning_reranker.py` share one pluggable scorer contract instead of each
inventing its own.

## Risk Assessment

- **Blast Radius**: `score_gate.py`, `neural_reranker.py`,
  `reasoning_reranker.py`, `autocut.py`, `config.py` (`KG_RERANK_MODEL`/
  `KG_RERANK_BASE_URL`/`KG_RERANK_LOCAL_NEURAL`).
- **Backward Compatible**: Yes — every heavier scorer is additive and
  auto-detected; the lexical path is the unconditional fallback.
- **Known weak point**: the lexical scorers (token containment/Jaccard/bigram
  overlap) are a *proxy* for relevance, not a learned signal — on queries
  where lexical overlap and true relevance diverge (paraphrase-heavy
  queries), the dependency-free default is measurably weaker than the neural
  path it silently substitutes for when the heavy stack isn't installed.
