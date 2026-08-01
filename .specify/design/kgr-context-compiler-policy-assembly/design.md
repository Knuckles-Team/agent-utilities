# Design Document: ContextCompiler replaces ad-hoc "flatten hits by score" with one six-axis selection/assembly layer

CONCEPT:AU-KG.retrieval.context-compiler ·
CONCEPT:AU-KG.retrieval.context-compiler-kv-seam ·
CONCEPT:AU-KG.retrieval.self-correcting-second-pass

> `agent_utilities/knowledge_graph/retrieval/context_compiler.py` (the
> compiler), `agent_utilities/core/contextual_model.py` (the KV-cache seam +
> per-process breaker), `agent_utilities/knowledge_graph/retrieval/
> hybrid_retriever.py:1414-1430` (the self-correcting second pass).

## Decision — six axes, one benchmarkable pass, instead of every caller hand-rolling its own "top hits, flattened" context block

`CONCEPT:AU-KG.retrieval.context-compiler`

`context_compiler.py:6-16` names the gap and the status quo it replaces
directly: "Every caller that needs to hand an LLM 'the relevant context' for a
query today does the same ad-hoc thing: run a retrieval call, then flatten the
hits into a flat text block sorted by raw similarity score" (citing
`query_tools.py`'s `graph_search` formatter and `context_builder.py`'s
single-node string concatenation as the two existing instances of this
pattern). **That path optimizes for exactly one axis — relevance — and drops
everything else on the floor**: near-duplicate hits crowd out coverage,
low-confidence claims ride alongside well-evidenced ones with no signal,
nothing checks visibility, nothing caps to a token budget, nothing explains
why an item was included or dropped.

`ContextCompiler` is the rejected-status-quo's replacement: it optimizes
**relevance, diversity (MMR), evidence quality (epistemic columns),
freshness (bi-temporal decay), token cost (`RetrievalBudgetManager`), and
policy (the same permissioning gate the live read path uses)** into one
`ContextBundle` — selected items with per-axis scores, a flat citation list,
a proof graph, and a `decisions` log recording every selection/rejection with
its scores. That log is explicitly "the observable, benchmarkable half of the
contract: same candidates + same session ⇒ same bundle" — a property an
ad-hoc per-caller flattening function has no reason to have. The module is
explicit about its own boundary: it is a SELECTION/ASSEMBLY/OPTIMIZATION
layer — it calls `search_hybrid`/`HybridRetriever` for retrieval and
`permissioning.enforce` for gating rather than reimplementing either, so the
six-axis logic is additive to the existing retrieval and policy stacks, not a
competing implementation of them.

### Pointer — `CONCEPT:AU-KG.retrieval.context-compiler-kv-seam` (Seam 6)

`context_compiler.py:65-70`, `contextual_model.py:161-174,629,921,161`. The
compiled bundle can be routed through the SAME shared, content-addressed
KV-cache layer the engine's `/kv` HTTP surface exposes for LMCache/vLLM
token-block reuse, via `kv_backend=` on `ContextCompiler.compile`. The
`_InProcessBundleCache` (`contextual_model.py:171`) is a bounded, TTL-expiring
(`_DEFAULT_BUNDLE_CACHE_TTL_S=300`), thread-safe deployment-default cache,
lazily constructed once behind a lock so a burst of concurrent first calls
builds exactly one instance rather than a stampede of redundant compiles.
What this pointer concretely adds over the base compiler: a compiled bundle is
expensive (it runs retrieval + six-axis scoring), and repeated compiles for
the same effective query/session within the TTL window are pure waste — the
seam makes that cost payable once, not once per caller.

### Pointer — `CONCEPT:AU-KG.retrieval.self-correcting-second-pass`

`hybrid_retriever.py:1414-1430`. Inside `plan_and_retrieve`, a second,
deeper-threshold retrieval pass — "fire only when the quality gate measured a
failure" (`gate_failed = report is not None and not
getattr(report, "gate_passed", True)`) — re-runs the same queries at
`threshold_for_mode("deep")` and merges the results back in. **The rejected
alternative is the obvious default**: always running a second pass (or never
running one). Always-on doubles retrieval cost on every call, including the
large majority that already passed the quality gate on the first pass;
never-on leaves a genuinely low-quality first retrieval uncorrected. Gating
the second pass strictly on the measured quality-gate outcome is what makes
this a *self-correcting* pass rather than a blanket "retrieve twice" policy —
also exposed as the `self_correct` flag on the served `graph_search` MCP tool
(`query_tools.py:1188-1191`).

## Risk Assessment

- **Blast Radius**: `context_compiler.py`, `contextual_model.py`,
  `hybrid_retriever.py`, `mcp/tools/query_tools.py` (the `graph_search` and
  `mode="compiled"` surface), `observability/gateway_metrics.py`.
- **Backward Compatible**: Yes — `context_compiler.py` is additive
  (`search`/`search_hybrid` are unchanged); `kv_backend=` and `self_correct=`
  are both opt-in parameters that default to the prior behavior.
- **Known weak point**: the six-axis scoring is itself a set of tunable
  weights/thresholds (MMR lambda, recency half-life, budget policy) — the
  `decisions` log makes a bad selection *diagnosable*, but nothing in this
  design automatically tunes those weights from the log; that remains a
  human/eval loop, not a closed feedback loop yet.
