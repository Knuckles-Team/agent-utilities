# OWL/RDF Layer — always-on, local, fast (CONCEPT:KG-2.7)

OWL/RDF is a **core, always-on** layer, not an enterprise add-on. It works
identically over *any* configured LPG storage backend (epistemic-graph, LadybugDB,
pg-age/AGE, Neo4j, FalkorDB): it consumes the bundled ontologies, **infers new
relationships**, **back-feeds them durably into the LPG store**, validates writes
with SHACL, and answers **SPARQL from a local endpoint with zero external
dependencies**. Apache Jena Fuseki / Stardog are an *optional* enterprise
scale-out (federation, a durable triplestore) — never required, and never on the
critical path for the zero-dep "tiny" / Raspberry-Pi profile.

## Why local-first

The fast/light inference substrate already exists: the Rust `epistemic-graph`
engine ships a native OWL-RL reasoner (`reasoning.rs` / `RunDatalogReasoning`:
subclass, subproperty, transitive, symmetric, inverse, domain/range,
property-chains) plus VF2 pattern matching and a bulk `GetTriples` RDF-export op.
So inference and SPARQL materialization run **in-process over UDS/MessagePack** —
no triplestore deployment, no network hop.

## Architecture

```mermaid
graph TB
    ING["Ingest / write path"] --> SH["SHACL gate\n(governance + value-type shapes)\nCONCEPT:KG-2.39"]
    SH --> LPG["LPG store\n(epistemic_graph | ladybug | pg-age/AGE | neo4j | falkordb)"]

    subgraph "OWL/RDF layer — always-on, local"
        TBOX["Bundled ontologies (TBox)\n30× ontology*.ttl\nloaded at startup"]
        REASON["epistemic-graph reasoner\nRunDatalogReasoning (OWL-RL)\nCONCEPT:KG-2.17"]
        BRIDGE["OWLBridge\npromote → reason → downfeed"]
        SPARQL["Local SPARQL endpoint\nGET/POST {gateway}/api/sparql"]
    end

    LPG --> BRIDGE
    TBOX --> REASON
    BRIDGE --> REASON
    REASON -->|inferred triples| BACKFEED["Durable back-feed\nlink_nodes(inferred=true)"]
    BACKFEED --> LPG

    LPG -->|GetTriples bulk export\n(fast path)| MAT["rdflib materialization"]
    MAT --> SPARQL
    BRIDGE -. optional .-> FUSEKI["Jena Fuseki / Stardog\n(enterprise scale-out)"]
```

## The cycle (OWLBridge)

1. **Promote** — stable LPG nodes/edges become OWL individuals/assertions.
2. **Reason** — the engine's OWL-RL rules (sourced from the loaded TBox) infer new
   triples (transitive closure, inverses, subclass, domain/range, property chains).
3. **Downfeed (durable)** — inferred edges are written **back into the LPG store**
   via the active engine's `link_nodes`, provenance-tagged
   (`inferred=true`, `inferred_from=owl_reasoner`, `inference_type`). This is
   synchronous and idempotent (it previously used an asyncio queue that silently
   no-op'd without a running event loop, so inferred triples never persisted).

## Local SPARQL

`{gateway}/api/sparql` (GET `?query=` or POST `{"query": …}`) is served by
`OWLBridge.query_sparql` and returns W3C SPARQL-JSON. Materialization uses a
**fast path**: the engine's `GetTriples` op exports the whole graph as
`[subject, predicate, object]` triples in **one call** (edges → `(s, rel, o)`,
node type → `(id, rdf:type, label)`, scalar props → `(id, prop, literal)`), which
feeds rdflib's mature SPARQL engine — rather than reimplementing SPARQL in Rust or
making per-node round-trips. It falls back to per-node iteration on engines without
`GetTriples`, and works **without `owlready2`** (the rdflib path needs only rdflib).

## SHACL validation

The pre-commit SHACL gate (`pipeline/phases/shacl_gate.py`, on by default) validates
materialized writes against the bundled `governance.shapes.ttl` **and** value-type
generated shapes (`ValueType.to_shacl()`, CONCEPT:KG-2.39) — so value-type
constraints (EmailAddress, Percentage, …) are enforced alongside governance rules.
Violating nodes are quarantined, not silently dropped.

## Deployment posture

| Profile | OWL reasoning | SPARQL | Triplestore |
|---|---|---|---|
| **tiny (Pi-3, zero-dep)** | ✅ local (engine OWL-RL) | ✅ local `/api/sparql` | none |
| single-node prod | ✅ local | ✅ local | optional |
| enterprise | ✅ local | ✅ local | + Jena Fuseki / Stardog (federation), `KG_FUSEKI_PUBLISH=1` |

## Reasoning *as* the research engine — one ontology over the whole ecosystem

agent-utilities maps the **entire ecosystem — `agent-packages/agents/*` + `services/*` +
enterprise systems + research papers — into ONE ontology-driven knowledge graph**
(the canonical ArchiMate upper ontology, KG-2.9; `ecosystem_topology`; Egeria SoR).
OWL/RDF reasoning is not a post-processing add-on here: it is the **engine** of research
and workflow execution. Its value is *extrapolating relationships that did not exist
before reasoning* — transitive/symmetric/inverse/domain-range/property-chain closures and
subClassOf/equivalentClass — **across the whole ecosystem at once**, so a research concept
can be inferred to relate to a deployed service or an agent capability, not siloed.

Every long-running objective (a **Loop**, KG-2.78 — research / develop / skill) runs the
**`OntologyReasoningDriver`** (KG-2.79) each cycle: it promotes the loop's working set +
the surrounding ecosystem subgraph, runs `OWLBridge.run_cycle` (promote → reason →
downfeed), and **harvests the newly-inferred cross-domain relationships back as fresh
research topics** — a closed extrapolation loop. This replaced the old one-shot enrichment
that ran reasoning and never consumed the inferences.

**Agent-Native Research Artifacts (ARA, KG-2.80)** are the OWL-native output: a 4-layer
artifact (`/logic` claims, `/src` code specs, `/trace` exploration DAG with dead-ends and
pivots, `/evidence` raw outputs) whose layers are **first-class ontology classes + typed
object-properties** (`research_artifact`/`claim`/`code_spec`/`evidence`/`exploration_node`;
`contains`/`grounded_in`/`implemented_by`). `grounded_in` is transitive with a `supports`
inverse, so reasoning chains a claim → evidence → ecosystem code/service automatically —
which is why we extrapolate cross-domain links from the *first* compiled artifact rather
than only "at critical mass". The ARA Compiler grounds each claim to the ecosystem it
touches; the ARA Seal verifies it (L1 = SHACL + interface conformance + OWL consistency,
L2 = rigor, L3 = exec-reproducibility with `/evidence` withheld via markings, KG-2.46) and
emits a signed `seal_certificate`. Both surfaces are exposed identically — the
`research_artifact` MCP tool and `POST {prefix}/research/*` REST — over one shared service.

## Key modules

- `knowledge_graph/core/owl_bridge.py` — promote/reason/downfeed + `query_sparql`;
  ARA forensic-edge characteristics (transitive `grounded_in`, `grounded_in`↔`supports`).
- `knowledge_graph/research/ara/` — `reasoning_driver` (reasoning-as-engine, KG-2.79),
  `artifact`/`compiler`/`seal`/`exploration`/`live_manager`/`service` (ARA, KG-2.80).
- `knowledge_graph/research/loops.py` — the `Loop` long-running-objective unit (KG-2.78).
- `gateway/research_api.py` — granular `{prefix}/research/*` typed routes (single SoT).
- `knowledge_graph/backends/owl/` — local `owlready2` backend + Stardog.
- `knowledge_graph/backends/sparql/jena_fuseki_backend.py` — optional Fuseki tier.
- `gateway/graph_api.py` — `{prefix}/sparql` route + cached bridge.
- `core/graph_compute.py::get_triples()` — bulk RDF export (engine `GetTriples`).
- `epistemic-graph/src/reasoning.rs`, `src/server.rs` (`GetTriples`) — Rust substrate.
- `core/ontology_publisher.py` — bundled-ontology collection + optional Fuseki push.
