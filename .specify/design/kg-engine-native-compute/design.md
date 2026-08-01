# Design Document: The engine computes; Python orchestrates

> One conviction expressed in four places — the compute client, the query plan,
> the transport, and the RDF/SHACL projection. Backfilled under the
> concept-lineage rule (CONCEPT:AU-OS.governance.concept-lineage-parent-doc):
> five sibling `AU-KG.compute` markers are instances of it.

CONCEPT:AU-KG.compute.graph-compute-engine ·
CONCEPT:AU-KG.compute.native-sparql-owl-shacl

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-ORCH.sandbox.compiled-orchestration-kernel` | the same "compiled kernel, thin Python" conviction on the orchestration side; declared in the same module header | 0.60 | ORCH |
| `AU-KG.sharding.tenant-partitioned-sharding-hrw` | engine-authoritative placement — the same refusal to hold authority in Python; also declared in this header | 0.50 | KG |
| `AU-KG.compute.numpy-scipy-drop` | the numeric expression of the same conviction | 0.40 | KG |

### Extension Analysis

- **Primary Extension Point**: `knowledge_graph/core/graph_compute.py`
  (`GraphComputeEngine`) and `knowledge_graph/core/owl_bridge.py`.
- **Extension Strategy**: augment.
- **New Concept Required?**: No new ones.

## Decision 1 — computation happens in the engine, in one costed round-trip

`CONCEPT:AU-KG.compute.graph-compute-engine`

`query_unified` submits an ordered list of externally-tagged `Op` dicts —
`Scan`/`Filter`/`Traverse`/`Rank`/`RankText`/`FuseRrf`/`AsOf`/`Limit` — over a
shared `RowSet`, and the engine sequences the whole thing over **one off-lock
snapshot**: filter via DataFusion, traverse via petgraph BFS, rank via the native
ANN, reciprocal-rank fusion in-plan.

**The rejected alternative is named in the code**: "the old hand-orchestrated
Python pipeline of siloed round-trips". That design worked. It lost because each
leg saw a *different* snapshot — a filter, a traverse and a vector rank issued
separately are three points in time — and because the cost model lived nowhere:
Python chose an order with no statistics, so a selective filter could run after
an expensive traverse.

The consequence, accepted: retrieval now **requires** an engine built with the
`query` feature, with the native ANN primitive (`semantic_search`) as a bounded
operational fallback rather than a Python path. `docs/architecture/vector_index_lifecycle.md`
states the invariant plainly — the vector neighbourhood is *always* computed by
the engine, never by Python.

**Transport follows from the same conviction.** The client demultiplexes many
in-flight requests by request id over the ONE authenticated process transport;
an auxiliary connection pool was rejected as redundant *and* as a security
regression, because it would pin one caller's authority into long-lived clients.
The two markers that carried that reasoning (`when-exposes`, `when-exposes-native`
— ids slugified out of the sentence "when the engine exposes the native…") are
retired; the reasoning is here.

Server-side entity resolution (`ResolveCandidates` escalation in
`assimilation/dedup.py`) is the same pattern once more: candidate generation runs
where the data is.

## Decision 2 — SPARQL, OWL and SHACL run over the engine projection

`CONCEPT:AU-KG.compute.native-sparql-owl-shacl`

The SHACL gate validates against the live graph via one N-Triples round-trip
(`get_rdf`) over the engine projection, rather than materialising the graph into
an rdflib in-memory store and validating there.

**The rejected alternative — an rdflib-side store — is still present as a
degraded path**, and its existence is what makes this a decision rather than an
implementation: the fallback promoted opaque property-graph ids straight into
`URIRef`s without guaranteeing IRI legality, which is exactly the class of bug
that motivated moving the authoritative path into the engine.

## What the pointers to these decisions are

- `tokio-service-layer` — the Tokio-first service layer the compute client is
  built on; declared in the same module header as the engine concept itself.
- `graph-builder` — the agent registry graph persists engine-only, with no
  Python-side twin. The same refusal to keep authority in Python, applied to one
  graph.

Four further markers in this cluster carried no name: `kg-2` and (in the
neighbouring world-model cluster) `kg-3` are bare citations of the retired
`KG-2.NN` numbering; `vector` is a single generic noun whose marker text
(`…compute.vector/214/215`) shows the author was writing a citation list, not a
name; `when-exposes`/`when-exposes-native` are sentence fragments. The decisions
they sat on are recorded above; the ids are retired.

## Data Flow

1. **ORCH**: none directly; the orchestration kernel is the sibling decision.
2. **KG**: every retrieval, traversal and validation path.
3. **AHE**: none.
4. **ECO**: MCP `graph_query`/`graph_search` reach the engine through this client.
5. **OS**: one authenticated transport per process — authority is per-request,
   never pinned into a pooled connection.

## Risk Assessment

- **Blast Radius**: `graph_compute.py`, `owl_bridge.py`,
  `retrieval/hybrid_retriever.py`, `retrieval/context_compiler.py`,
  `retrieval/engine_capability_search.py`, `pipeline/phases/shacl_gate.py`,
  `assimilation/dedup.py`, `enrichment/pipeline.py`, `graph/builder.py`.
- **Backward Compatible**: Yes at the Python API; the engine feature requirement
  is the real contract change.
- **Breaking Changes**: an engine built without the `query` feature loses unified
  planning and degrades to the native ANN primitive.
- **Known hazard**: the rdflib SPARQL fallback is a genuinely different code path
  with a genuinely different IRI-safety story. It is retained for the no-engine
  case and is the least-exercised path here.
