# Design Document: The engine computes; Python orchestrates

> Backfilled under the concept-lineage rule (CONCEPT:AU-OS.governance.concept-lineage-parent-doc).
> One conviction expressed in two places: the unified query plan, and the
> RDF/SHACL projection.

CONCEPT:AU-KG.compute.graph-compute-engine · CONCEPT:AU-KG.compute.native-sparql-owl-shacl

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-ORCH.sandbox.compiled-orchestration-kernel` | the same "compiled kernel, thin Python" conviction on the orchestration side; declared in the same module header | 0.60 | ORCH |
| `AU-KG.sharding.tenant-partitioned-sharding-hrw` | engine-authoritative placement — the same refusal to hold authority in Python; declared in the same header | 0.50 | KG |
| `AU-KG.compute.numpy-scipy-drop` | the numeric expression of the same conviction | 0.40 | KG |

### Extension Analysis

- **Primary Extension Point**: `knowledge_graph/core/graph_compute.py`
  (`GraphComputeEngine`) and `knowledge_graph/core/owl_bridge.py`.
- **Extension Strategy**: augment.
- **New Concept Required?**: No.

## Decision 1 — computation happens in the engine, in one costed round-trip

`CONCEPT:AU-KG.compute.graph-compute-engine` — `knowledge_graph/core/graph_compute.py:1,2567`.

`query_unified` (`graph_compute.py:2560`) submits an ordered list of
externally-tagged `Op` dicts — `Scan`/`Filter`/`Traverse`/`Rank`/`RankText`/
`FuseRrf`/`AsOf`/`Limit` — over a shared `RowSet`, and the ENGINE sequences
the whole thing over one off-lock snapshot: filter via DataFusion, traverse
via petgraph BFS, rank via the native ANN, reciprocal-rank fusion in-plan.

**The rejected alternative**, named directly in the docstring: "the old
hand-orchestrated Python pipeline of siloed round-trips." That design worked,
and it lost because each leg saw a DIFFERENT snapshot — a filter, a traverse,
and a vector rank issued as three separate calls are three different points
in time against a graph that can mutate between them — and because the cost
model lived nowhere: Python chose an execution order with no statistics, so a
selective filter could run after an expensive traverse instead of before it.

The consequence, accepted deliberately: `query_unified` **requires** an
engine built with the `query` feature — a build without it raises a clear
engine error, with no O(N) Python fallback path. The native ANN primitive
(`semantic_search`) remains as a bounded operational fallback for simpler
queries, not a general substitute for the unified plan.

**What breaks if violated**: a caller that reassembles filter/traverse/rank
as separate Python-orchestrated round-trips (bypassing `query_unified`)
reintroduces the exact snapshot-inconsistency and unordered-execution problem
this decision retired — results computed against different points in time,
with no cost-based ordering.

## Decision 2 — SPARQL, OWL and SHACL run over the engine projection

`CONCEPT:AU-KG.compute.native-sparql-owl-shacl` — `knowledge_graph/pipeline/phases/shacl_gate.py:114`.

The SHACL validation gate sources the data it validates from the ENGINE's own
RDF projection first — one N-Triples round-trip via `get_rdf`
(`shacl_gate.py:114`, `build_data_graph`) — rather than materializing the
whole property graph into an in-memory `rdflib.Graph` and validating there.

**The rejected alternative — an rdflib-side materialization — is still
present as a degraded fallback** (`shacl_gate.py:121-124`, per-node iteration
of the LPG when no engine is reachable), and its continued existence is what
makes this a real decision rather than an implementation detail: that
fallback path promotes opaque property-graph ids straight into RDF `URIRef`s
without independently guaranteeing IRI legality — exactly the class of bug
that motivates keeping the AUTHORITATIVE validation path inside the engine's
own RDF projection instead.

**What breaks if violated**: relying on the rdflib fallback as the primary
path (rather than the engine projection) reintroduces the IRI-legality risk
the engine-native path exists to avoid, and validates against a
point-in-time Python-side materialization rather than the live graph.

### The pointers to this decision — the same conviction, restated once each

`tokio-service-layer` (`graph_compute.py:3` header) — the Tokio-first service
layer the compute client itself is built on; declared in the same module
header as the `graph-compute-engine` concept, not a separate design.
`graph-builder` (`graph/builder.py:516,524`) — the agent registry graph
persists ENGINE-ONLY, with no Python-side twin; the same refusal to hold
authority in Python, applied to one specific graph rather than the query
layer generally.

## C4 Context Diagram

```mermaid
C4Context
    title The engine computes; Python orchestrates

    System_Boundary(b1, "agent-utilities") {
        System(client, "GraphComputeEngine.query_unified", "Ordered Op-plan over one off-lock snapshot")
        System(shacl, "shacl_gate.build_data_graph", "Sources validation data from engine RDF projection")
        System(builder, "graph/builder registry graph", "Persists engine-only, no Python twin")
    }
    System_Ext(engine, "epistemic-graph engine", "DataFusion filter, petgraph traverse, native ANN, RRF fusion")

    Rel(client, engine, "one costed round-trip per query")
    Rel(shacl, engine, "get_rdf: one N-Triples round-trip")
    Rel(builder, engine, "authoritative persistence, no Python copy")
```

## Data Flow

1. **ORCH**: none directly; the compiled orchestration kernel is the sibling
   decision on the ORCH side.
2. **KG**: every retrieval, traversal and validation path that needs a
   cross-modal or RDF-projected view of the graph.
3. **AHE**: none.
4. **ECO**: MCP `graph_query`/`graph_search` reach the engine through
   `query_unified`.
5. **OS**: none directly.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/core/graph_compute.py`,
  `knowledge_graph/core/owl_bridge.py`,
  `knowledge_graph/retrieval/hybrid_retriever.py`,
  `knowledge_graph/pipeline/phases/shacl_gate.py`, `graph/builder.py`.
- **Backward Compatible**: Yes at the Python API surface; the engine
  `query` feature requirement is the real contract change.
- **Breaking Changes**: an engine built without the `query` feature loses
  unified planning and degrades to the native ANN primitive for simpler
  queries.
- **Known hazard**: the rdflib SPARQL fallback in `shacl_gate.py` is a
  genuinely different code path with a genuinely different IRI-safety story
  than the engine-native projection, and it is the least-exercised path here
  (only reached when no engine is reachable at all).
