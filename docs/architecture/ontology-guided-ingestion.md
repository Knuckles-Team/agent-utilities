# Ontology-Guided Ingestion & Entity Resolution

> Exceeds the `sift-kg` document→knowledge-graph pipeline while synergizing with our
> OWL/RDF ontology. Concepts: **KG-2.255** (ontology-guided extraction), **KG-2.256**
> (direction repair), **KG-2.257** (confidence/support-count weighting), **AHE-3.70**
> (dedup-ladder extensions + variant split), **KG-2.258** (community summarization),
> **KG-2.259** (schema discovery), **KG-2.260** (engine ResolveCandidates op).
> Comparative analysis: `workspace/reports/sift-kg-comparative-analysis-2026-06-26.md`.

This page documents how the ingestion/extraction path was upgraded so that extraction
is **driven by the OWL ontology** (not free-form), edges are **oriented + corroboration-
weighted**, entities are **resolved with transliteration/singularization + a variant
split**, communities are **summarized into queryable reports**, and the ontology
**self-extends** from the corpus — with a native Rust engine op as the scale tier.

## Where each piece lives

| Concept | What | File |
|---|---|---|
| KG-2.255 | OWL TBox → extraction schema, injected into the LLM prompt | `extraction/extraction_schema.py`, `fact_extractor.py`, `ingestion/engine.py:793` |
| KG-2.256 | Relation-direction repair via `rdfs:domain/range` | `extraction/direction_repair.py` |
| KG-2.257 | Product-complement confidence + support-count edge weight | `fact_extractor.py:persist_facts` |
| AHE-3.70 | Transliteration + singularization + version-variant split | `assimilation/entity_resolution.py`, `assimilation/dedup.py` |
| KG-2.258 | GraphRAG community summarization phase | `pipeline/phases/community_reports.py` |
| KG-2.259 | Ontology-aware schema discovery → `.ttl` proposals | `extraction/schema_discovery.py`, `mcp/tools/ontology_tools.py` |
| KG-2.260 | Native `ResolveCandidates` engine op + escalation | `epistemic-graph` `algorithms.rs`/`protocol.rs`/`graph_ops.rs`, `core/graph_compute.py` |

## End-to-end ingestion flow

```mermaid
flowchart TD
    DOC[Document / connector payload] --> ENR["_enrich_text seam<br/>engine.py:853"]
    ENR --> XFG["_extract_facts_into_graph<br/>engine.py:793"]

    subgraph SCHEMA["KG-2.255 ontology-guided extraction"]
        TTL[("ontology_*.ttl<br/>OWL TBox")] --> ES["load_extraction_schema(source_type)<br/>extraction_schema.py"]
        ES --> SCH["ExtractionSchema<br/>classes + rdfs:domain/range + skos"]
    end

    XFG -->|source_type| ES
    SCH -->|prompt_block injected| EF["extract_facts(schema=…)<br/>fact_extractor.py:440"]
    EF --> FACTS["ExtractedFacts<br/>(s)-[p]->(o) + confidence"]

    FACTS --> GND["ground_facts<br/>ontology_grounding.py"]
    GND --> REP["KG-2.256 repair_direction<br/>direction_repair.py"]
    REP -->|reversed→swap| PERSIST
    REP -->|domain/range violation| SHACL["SHACL contradiction shape<br/>KG-2.251/2.252"]

    PERSIST["KG-2.257 persist_facts<br/>group by (s,p,o)"] --> EDGE["one edge<br/>weight=support_count<br/>confidence=1−∏(1−cᵢ)"]
    EDGE --> ENGINE[("epistemic-graph<br/>EdgeData.weight/confidence")]

    style SCHEMA fill:#eef
    style SHACL fill:#fee
```

Key change: grounding + direction-repair now run **before** persist (extract → ground+repair
→ persist → annotate), so edges land oriented and node `ontology_type` annotations match the
persisted orientation. `schema=None` (non-prose content, or rdflib absent on the lean serving
plane per KG-2.242) falls back to the unchanged free-vocab path — no regression.

## Entity resolution: ladder + variant split + engine escalation

```mermaid
flowchart TD
    IN["entities (id, name)"] --> NORM["normalize_name (AHE-3.70)<br/>transliterate + singularize"]
    NORM --> LADDER

    subgraph LADDER["AHE-3.69/3.70 deterministic ladder"]
        EXACT["exact canonical-key match"] --> ENT["Shannon-entropy gate"]
        ENT --> LSH["MinHash + LSH + Jaccard≥0.9"]
        LSH --> VAR["version-variant split<br/>detect_version_variant"]
    end

    VAR -->|same_as| MERGE["merge_pairs → SUPERSEDES"]
    VAR -->|version variant| VLINK["variants → VARIANT_OF"]
    LADDER -->|residual ids| ESC

    subgraph ESC["KG-2.260 engine escalation (capability-gated)"]
        RC["GraphComputeEngine.resolve_candidates<br/>→ engine ResolveCandidates op"]
        RC --> ANN["all-pairs cosine ≥ sim_threshold"]
        ANN --> CL["union-find clusters<br/>(same-type ≥ merge_threshold)"]
        CL -->|same_as| MERGE
        CL -->|cross-type| VLINK
    end

    DEDUP["dedup_features<br/>assimilation/dedup.py"] --> LADDER
    DEDUP --> ESC
```

The native engine op (`epistemic-graph` `algorithms::resolve_candidates`) is **read/propose
only** — it returns `MergeProposal{canonical, members, score, kind}` and never mutates; the
Python side decides what to apply via `BatchUpdate`. It is the scale tier the ladder's
*residual* escalates into, replacing an O(N²) client-side embedding pass.

## Community summarization + schema discovery

```mermaid
flowchart LR
    subgraph G["KG-2.258 GraphRAG (pipeline phase)"]
        COMM["communities phase<br/>(native Louvain tag)"] --> CR["community_reports phase"]
        CR -->|per community| LLM1["lite LLM theme+summary"]
        LLM1 --> CRN["CommunityReport nodes<br/>+ PART_OF_COMMUNITY"]
        CRN --> GLOB["level-1 global report"]
        CRN --> QRY["graph_query / graph_search"]
    end

    subgraph B["KG-2.259 schema discovery"]
        SAMP["sample documents"] --> LLM2["LLM proposes types"]
        LLM2 --> DIFF["diff vs live ontology<br/>(schema + synonyms)"]
        DIFF -->|missing| PROP[".ttl proposal<br/>RESERVE-PENDING"]
        PROP --> EVO["concept reservation +<br/>evolution pipeline (human/SHACL-gated)"]
        EVO -.lands in.-> TTL[("ontology_*.ttl")]
    end

    OD["ontology_derive<br/>action=discover_extensions<br/>(MCP + REST)"] --> SAMP
```

Community reports become first-class nodes, so global-theme questions answer from
report-grounded nodes through the **existing** `graph_query`/`graph_search` surface — no new
store. Schema discovery never auto-merges a `.ttl` (a new top-level ontology file is a build
break); it emits a *proposal* with `RESERVE-PENDING` placeholders for the evolution loop.

## Why this exceeds sift-kg

- **Schema source.** sift-kg injects a flat YAML schema; we inject the **formal OWL TBox**
  (`owl:Class` + `rdfs:domain/range` + skos labels) and keep OWL reasoning + post-hoc
  grounding downstream — generation-time guidance *and* reasoning.
- **Direction repair** reuses `reasoning.rs infer_domain_range` (no new engine op) and routes
  violations into the existing contradiction/SHACL machinery (KG-2.251/2.252).
- **Resolution** runs the deterministic ladder (AHE-3.69) extended with transliteration +
  singularization + a variant split, and escalates to a **native Rust** clustering op rather
  than sift-kg's per-pair LLM/networkx resolution.
- **Community reports** are queryable graph nodes (GraphRAG), not a static narrative file.
- **Discovery** proposes **ontology extensions** into the evolution pipeline, closing the
  loop sift-kg's flat YAML cannot.

## Verification

Unit suites (all green): `test_extraction_schema.py`, `test_direction_repair.py`,
`test_persist_facts_aggregation.py`, `test_entity_resolution_variants.py`,
`test_assimilation_dedup.py`, `test_community_reports.py`, `test_schema_discovery.py`; Rust
`algorithms::resolve_candidates_tests` (4). Live E2E (per the ingestion-validation protocol):
restart graph-os → `source_sync(source=<domain corpus>, mode=delta)` → verify edges carry
canonical OWL types + `support_count`/`weight`, direction satisfies domain/range,
CommunityReport nodes answer a global-theme query, and re-run shows `skipped_unchanged>0`.

**Human-gated (deferred):** engine rebuild + image push + R820 swarm redeploy to serve the
`ResolveCandidates` op live; B-proposed ontology classes await review + concept reservation.
