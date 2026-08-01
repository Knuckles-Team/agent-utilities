# Design Document: Code-intelligence answers are deterministic, embedder-free reads over the resolved :Code graph, cited by file:line — never grep-then-synthesize

CONCEPT:AU-KG.retrieval.synthesized-cited-answer ·
CONCEPT:AU-KG.retrieval.architecture-report ·
CONCEPT:AU-KG.retrieval.every-usage-published-symbol ·
CONCEPT:AU-KG.retrieval.god-nodes-communities ·
CONCEPT:AU-KG.retrieval.structural-analytics

> `agent_utilities/knowledge_graph/retrieval/code_context.py` (primary
> decision — already has its own doc at
> `.specify/design/retrieval-latency-optimization/design.md`, which this
> supplements),`agent_utilities/knowledge_graph/retrieval/code_metrics.py`,
> `agent_utilities/mcp/kg_server.py`, `agent_utilities/mcp/tools/
> analysis_tools.py`.

## Decision (restated) — one grounded, `file:line`-cited explanation composed from already-built code-intelligence primitives

`CONCEPT:AU-KG.retrieval.synthesized-cited-answer`

`code_context.py:4-13` (KG-2.135): `code_context(query, intent)` composes the
typed/scope-resolved call graph, embedder-free similar-code, code↔service
routes, git change-coupling, `CONCEPT:` markers, and ingested docs into one
answer so an agent learns "how an area works / where a symbol is used / what a
change impacts" by *querying the KG*, not by grep-then-read across the tree.
It is deterministic and embedder-free (pure Cypher, so it answers even when
the remote embedder is unavailable), degrades gracefully (unenriched sections
come back empty rather than guessed), and resolves usages cross-repo by
symbol name rather than node id. This decision already has a full write-up at
`.specify/design/retrieval-latency-optimization/design.md`; this document
exists to give a home to the three sibling actions below, which are
`analysis_tools.py`/`kg_server.py` REST twins of the SAME family — each a
genuinely distinct code-intelligence view, built the same way (durable engine
reads, never a one-shot scratch computation).

### Pointer — `CONCEPT:AU-KG.retrieval.architecture-report`

`code_metrics.py:1-22` (the `arch_report` action), `kg_server.py:1906`. The
regenerable architecture report — a `GRAPH_REPORT.md` analog assembled as
Markdown + metrics from the SAME durable graph the other actions read, rather
than a static, hand-maintained document that drifts from the code the moment
either changes.

### Pointer — `CONCEPT:AU-KG.retrieval.every-usage-published-symbol`

`code_context.py:332,483`, `kg_server.py:1871` (the `cross_repo_usages`
action). Aggregates every usage of a published symbol across the WHOLE
ingested fleet by symbol *name*, not node id — `run_agent`'s callers, for
example, span agent-utilities, the frameworks, and the agents in one query.
**The rejected alternative** the fleet-aggregation solves is what code
search tools default to: usages scoped to one repo at a time, requiring a
caller to already know which repos might reference a symbol before searching
each in turn.

### Pointer — `CONCEPT:AU-KG.retrieval.god-nodes-communities` and `CONCEPT:AU-KG.retrieval.structural-analytics`

`code_metrics.py:1-22` (module decision), `analysis_tools.py:2296-2304` (the
`code_metrics` dispatcher action — the SAME decision under a second concept
id at its call site). Graphify-style structural analytics — god nodes
(degree hubs), Louvain communities (the engine's ephemeral detector,
KG-2.58), surprising cross-community connections, language/relation/
confidence distributions — scoped to the resolved `:Code` call/inheritance
subgraph. **The rejected alternative is stated directly in the code**: "Reuses
the durable resolved graph — not a one-shot NetworkX notebook." Graphify's
original approach re-builds its analysis graph from scratch per invocation in
an ad-hoc notebook; this decision instead reuses the engine's own ephemeral
Louvain projection (no tenant load, no persistence churn), carries the
resolver's per-edge confidence through every view (Graphify's
EXTRACTED/INFERRED/AMBIGUOUS analog), and produces a regenerable,
`file:line`-cited report node instead of a static snapshot file — the same
"regenerable over durable state, not a static artifact" principle the
architecture-report pointer above applies.

## Risk Assessment

- **Blast Radius**: `code_context.py`, `code_metrics.py`, `kg_server.py`,
  `analysis_tools.py`.
- **Backward Compatible**: Yes — each action is additive over the resolved
  `:Code` graph; none mutate it.
- **Known weak point**: every action here degrades gracefully when enrichment
  hasn't run (empty sections, not errors) — good for availability, but a
  caller cannot distinguish "nothing relevant exists" from "the enrichment
  sweep hasn't reached this repo yet" without checking ingestion status
  separately.
