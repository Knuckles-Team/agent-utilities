# OWL/RDF Layer — always-on, local, fast (CONCEPT:AU-KG.query.vendor-agnostic-traversal)

OWL/RDF is a **core, always-on** layer, not an enterprise add-on. It works
identically over *any* configured LPG storage backend (epistemic-graph, LadybugDB,
pg-age/AGE, Neo4j, FalkorDB): it consumes the bundled ontologies, **infers new
relationships**, **back-feeds them durably into the LPG store**, validates writes
with SHACL, and answers **SPARQL from a local endpoint with no external
triplestore**. Apache Jena Fuseki / Stardog are an *optional* enterprise
scale-out (federation, a durable triplestore) — never required, and never on the
critical path for the engine-only `tiny` profile.

## Engine-native semantic web (CONCEPT:AU-KG.compute.native-sparql-owl-shacl)

SPARQL, OWL DL reasoning, and SHACL are served by the **engine's native RDF
surface** (`client.rdf.*`), not a Python rdflib/owlready2/pyshacl stack:

- **SPARQL 1.1** — `client.rdf.sparql(query)` runs over the LIVE engine graph (the
  RDF dataset maps onto the same property graph: a resource object → a typed edge, a
  literal → a typed property cell preserving xsd datatype/`@lang`, `rdf:type` → the
  engine `type` label). No rdflib materialization.
- **OWL 2 (EL⁺/RL) reasoning** — `client.rdf.owl_reason(ontology, target_class)`
  classifies the OWL axioms in the graph (plus passed Turtle) and materializes
  **confidence/decay-weighted entailments** (inferred subclass edges + class
  memberships), read-only. This is the memory-inference primitive.
- **SHACL** — `client.rdf.validate_shacl(shapes, data_graph)` runs the engine's native
  Rust SHACL validator against explicit Turtle shapes and data.

This native RDF/SPARQL/OWL surface is **pure-Rust** (oxrdf/oxttl/spargebra — no
native C deps) and ships in the single full engine artifact required by
`epistemic-graph[full]>=2.23.1,<3.0.0`. The `EngineResolver` (OS-5.63) auto-starts
that artifact on demand, so the engine's semantic surface is always available,
including on a Raspberry Pi 4+. Optional Python RDF/OWL tooling is not a compatibility
path for a missing engine and is kept out of the serving plane (the `serving` extra
does not pull `[owl]`).

## Architecture

```mermaid
graph TB
    ING["Ingest / write path"] --> SH["SHACL gate\n(governance + value-type shapes)\nCONCEPT:AU-KG.ontology.value-type-shacl-load"]
    SH --> LPG["LPG store\n(epistemic_graph | ladybug | pg-age/AGE | neo4j | falkordb)"]

    subgraph "OWL/RDF layer — engine-native (CONCEPT:AU-KG.compute.native-sparql-owl-shacl)"
        TBOX["Bundled ontologies (TBox)\n+ pack object-property axioms\n(emitted as Turtle)"]
        REASON["engine OWL 2 reasoner\nclient.rdf.owl_reason\n(EL⁺/RL, confidence-weighted)"]
        BRIDGE["OWLBridge\npromote → reason → downfeed"]
        SPARQL["SPARQL endpoint\nGET/POST {gateway}/api/sparql\nclient.rdf.sparql (live graph)"]
    end

    LPG --> BRIDGE
    TBOX --> REASON
    BRIDGE --> REASON
    REASON -->|inferred triples| BACKFEED["Durable back-feed\nlink_nodes(inferred=true)"]
    BACKFEED --> LPG

    LPG -->|client.rdf.sparql\n(live engine graph)| SPARQL
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
`OWLBridge.query_sparql` and returns W3C SPARQL-JSON. By default it dispatches to
the engine's native `client.rdf.sparql` (via `GraphComputeEngine.sparql`), which runs
the SPARQL 1.1 query **over the live engine graph** in one round-trip — no rdflib
materialization or no-engine compatibility path.

## SHACL validation

The SHACL gate (`pipeline/phases/shacl_gate.py`, on by default) validates writes
against the bundled `governance.shapes.ttl` **and** value-type generated shapes
(`ValueType.to_shacl()`, CONCEPT:AU-KG.ontology.value-type-shacl-load). The data graph
is sourced from the engine's RDF projection and submitted to the native SHACL validator
(CONCEPT:AU-KG.compute.native-sparql-owl-shacl).
Violating nodes are quarantined, not silently dropped.

## Deployment posture

| Profile | OWL reasoning | SPARQL | Triplestore |
|---|---|---|---|
| **tiny (Pi 4+, engine-only)** | ✅ engine-native (`client.rdf.owl_reason`) | ✅ engine-native `client.rdf.sparql` | none |
| single-node prod | ✅ engine-native | ✅ engine-native | optional |
| enterprise | ✅ engine-native | ✅ engine-native | + Jena Fuseki / Stardog (federation), `KG_FUSEKI_PUBLISH=1` |

## Reasoning *as* the research engine — one ontology over the whole ecosystem

agent-utilities maps the **entire ecosystem — `agent-packages/agents/*` + `services/*` +
enterprise systems + research papers — into ONE ontology-driven knowledge graph**
(the canonical ArchiMate upper ontology, KG-2.9; `ecosystem_topology`; Egeria SoR).
OWL/RDF reasoning is not a post-processing add-on here: it is the **engine** of research
and workflow execution. Its value is *extrapolating relationships that did not exist
before reasoning* — transitive/symmetric/inverse/domain-range/property-chain closures and
subClassOf/equivalentClass — **across the whole ecosystem at once**, so a research concept
can be inferred to relate to a deployed service or an agent capability, not siloed.

Every long-running objective is a **Loop** (AU-KG.research.these-properties-carry) — kind `research`, `develop`, or
`skill` — and the **one** `LoopController` (formerly the "golden loop") advances every
active Loop through a single hot path: research loops acquire sources + reason, `develop`
loops run act→validate (their `validation_cmd`), `skill` loops execute their skill /
skill-workflow. There is no separate goal-runner or research-runner — the goal system is a
thin adapter onto `LoopController.run_loop`. The single entrypoint is the **`graph_loops`**
MCP tool (`submit` / `list` / `run` / `drive` / `cancel`); `submit_loop` is the shared
creation path for goals, research topics, failure gaps and skill executions.

**One persistence model.** Goal lifecycle is *not* a separate SQLite/Postgres `goals`
table and is not writable on the Loop projection. The **KG Loop node** (a develop
`Concept`) retains definition and observability; its deterministic engine-native
`WorkItem` is the sole authority for status, claim, lease, retry, checkpoint, and terminal
outcome. `/goals` REST, `graph_goals`, dispatch, and restart recovery render or transition
that same WorkItem under its native fence, so a goal can never be double-driven by a
second lifecycle owner.

**Durable checkpointing is cross-cutting, not goal-specific.** `LoopController.run_loop`
drives one Loop of any kind to completion under the WorkItem's renewable lease. Each
completed iteration advances its opaque `checkpoint_id` through a tenant/owner/epoch/
fencing-token compare-and-set; a reclaimed expired lease resumes from that reference.
The matching idempotency key and fenced terminal commit make redelivery safe, while a
fleet pause/kill signal checkpoints and yields corrigibly (SAFE-1.5). No checkpoint
sidecar or state-store selector participates.

The research path runs the **`OntologyReasoningDriver`** (AU-KG.research.best-effort-lightweight-never) each cycle: it promotes
the loop's working set + the surrounding ecosystem subgraph, runs `OWLBridge.run_cycle`
(promote → reason → downfeed), and **harvests the newly-inferred cross-domain relationships
back as fresh research topics** — a closed extrapolation loop. This replaced the old
one-shot enrichment that ran reasoning and never consumed the inferences.

**Agent-Native Research Artifacts (ARA, AU-KG.ontology.verified-by-implemented-by)** are the OWL-native output: a 4-layer
artifact (`/logic` claims, `/src` code specs, `/trace` exploration DAG with dead-ends and
pivots, `/evidence` raw outputs) whose layers are **first-class ontology classes + typed
object-properties** (`research_artifact`/`claim`/`code_spec`/`evidence`/`exploration_node`;
`contains`/`grounded_in`/`implemented_by`). `grounded_in` is transitive with a `supports`
inverse, so reasoning chains a claim → evidence → ecosystem code/service automatically —
which is why we extrapolate cross-domain links from the *first* compiled artifact rather
than only "at critical mass". The ARA Compiler grounds each claim to the ecosystem it
touches; the ARA Seal verifies it (L1 = SHACL + interface conformance + OWL consistency,
L2 = rigor, L3 = exec-reproducibility with `/evidence` withheld via markings, AU-KG.ontology.redact-object-materialize-restricted) and
emits a signed `seal_certificate`. Both surfaces are exposed identically — the
`research_artifact` MCP tool and `POST {prefix}/research/*` REST — over one shared service.

## Key modules

- `knowledge_graph/core/owl_bridge.py` — promote/reason/downfeed + `query_sparql`;
  ARA forensic-edge characteristics (transitive `grounded_in`, `grounded_in`↔`supports`).
- `knowledge_graph/research/ara/` — `reasoning_driver` (reasoning-as-engine, AU-KG.research.best-effort-lightweight-never),
  `artifact`/`compiler`/`seal`/`exploration`/`live_manager`/`service` (ARA, AU-KG.ontology.verified-by-implemented-by).
- `knowledge_graph/research/loops.py` — the `Loop` long-running-objective unit +
  `submit_loop`/`active_loops`/`mark_loop_status` (AU-KG.research.these-properties-carry).
- `knowledge_graph/research/loop_controller.py` — the one `LoopController` advancing all
  Loop kinds (research stages + develop act→validate + skill execution).
- `gateway/research_api.py` — granular `{prefix}/research/*` typed routes (single SoT);
  `graph_loops` MCP tool — the single entrypoint for long-running objectives.
- `knowledge_graph/backends/owl/` — `owlready2` backend + Stardog (full-DL last-resort fallback only).
- `knowledge_graph/backends/sparql/jena_fuseki_backend.py` — optional Fuseki tier.
- `gateway/graph_api.py` — `{prefix}/sparql` route + cached bridge.
- `core/graph_compute.py::sparql()/owl_reason()/add_triples()/get_rdf()` — the engine-native RDF surface (CONCEPT:AU-KG.compute.native-sparql-owl-shacl): `client.rdf.sparql`/`owl_reason`/`add_triples`/`GetRdf`.
- `epistemic-graph` `crates/eg-rdf` (`rdf`/`sparql`/`owl` features, pure-Rust oxrdf/oxttl/spargebra) — the native RDF/SPARQL/OWL substrate (`client.rdf.*`).
- `core/ontology_publisher.py` — bundled-ontology collection + optional Fuseki push.
