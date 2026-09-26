# Ontology-Guided Ingestion & Entity Resolution

> Exceeds the `sift-kg` document→knowledge-graph pipeline while synergizing with our
> OWL/RDF ontology. Concepts: **AU-KG.retrieval.mmr-diversification** (ontology-guided extraction), **AU-KG.enrichment.direction-repair**
> (direction repair), **AU-KG.ingest.observability-queries-opik-cannot** (confidence/support-count weighting), **AU-AHE.assimilation.transliteration-singularization-extend-ahe**
> (dedup-ladder extensions + variant split), **AU-KG.enrichment.community-reports** (community summarization),
> **AU-KG.ontology.do-not-auto-merge** (schema discovery), **AU-KG.compute.when-exposes-native** (engine ResolveCandidates op).
> Comparative analysis: `workspace/reports/sift-kg-comparative-analysis-2026-06-26.md`.

This page documents how the ingestion/extraction path was upgraded so that extraction
is **driven by the OWL ontology** (not free-form), edges are **oriented + corroboration-
weighted**, entities are **resolved with transliteration/singularization + a variant
split**, communities are **summarized into queryable reports**, and the ontology
**self-extends** from the corpus — with a native Rust engine op as the scale tier.

## Where each piece lives

| Concept | What | File |
|---|---|---|
| AU-KG.retrieval.mmr-diversification | OWL TBox → extraction schema, injected into the LLM prompt | `extraction/extraction_schema.py`, `fact_extractor.py`, `ingestion/engine.py:793` |
| AU-KG.enrichment.direction-repair | Relation-direction repair via `rdfs:domain/range` | `extraction/direction_repair.py` |
| AU-KG.ingest.observability-queries-opik-cannot | Product-complement confidence + support-count edge weight | `fact_extractor.py:persist_facts` |
| AU-AHE.assimilation.transliteration-singularization-extend-ahe | Transliteration + singularization + version-variant split | `assimilation/entity_resolution.py`, `assimilation/dedup.py` |
| AU-KG.enrichment.community-reports | GraphRAG community summarization phase | `pipeline/phases/community_reports.py` |
| AU-KG.ontology.do-not-auto-merge | Ontology-aware schema discovery → `.ttl` proposals | `extraction/schema_discovery.py`, `mcp/tools/ontology_tools.py` |
| AU-KG.compute.when-exposes-native | Native `ResolveCandidates` engine op + escalation | `epistemic-graph` `algorithms.rs`/`protocol.rs`/`graph_ops.rs`, `core/graph_compute.py` |

## End-to-end ingestion flow

<div class="admonition architecture" markdown>
<p class="admonition-title">End-to-end ingestion: schema-guided extraction, grounded, repaired, persisted</p>

A document/connector payload passes through `_enrich_text` (`engine.py:853`)
into `_extract_facts_into_graph` (`engine.py:793`), which passes the
`source_type` to `load_extraction_schema` (`extraction_schema.py`). That
loads the OWL TBox (`ontology_*.ttl`) into an `ExtractionSchema` (classes
+ `rdfs:domain`/`range` + SKOS), which is injected as a `prompt_block`
into `extract_facts(schema=…)` (`fact_extractor.py:440`), producing
`ExtractedFacts` — `(s)-[p]->(o)` triples with confidence.

Facts flow through `ground_facts` (`ontology_grounding.py`) then
`repair_direction` (`direction_repair.py`): a reversed edge is swapped
before persisting; a domain/range violation instead raises a SHACL
contradiction shape (KG-2.251/2.252).

Grounded, repaired facts reach `persist_facts` (grouped by `(s,p,o)`),
which writes one edge per group with `weight=support_count` and
`confidence=1−∏(1−cᵢ)` into the epistemic-graph engine
(`EdgeData.weight`/`confidence`).
</div>

Key change: grounding + direction-repair now run **before** persist (extract → ground+repair
→ persist → annotate), so edges land oriented and node `ontology_type` annotations match the
persisted orientation. `schema=None` (non-prose content, or rdflib absent on the lean serving
plane per KG-2.242) falls back to the unchanged free-vocab path — no regression.

## Entity resolution: ladder + variant split + engine escalation

<div class="admonition architecture" markdown>
<p class="admonition-title">A deterministic ladder, escalating its residual to the engine</p>

Entities `(id, name)` are normalized (`normalize_name`: transliterate +
singularize) and fed into the deterministic ladder, seeded also by
`dedup_features` (`assimilation/dedup.py`): exact canonical-key match ->
Shannon-entropy gate -> MinHash + LSH + Jaccard >= 0.9 -> version-variant
split (`detect_version_variant`).

The ladder's output either merges as `same_as` (`merge_pairs` ->
`SUPERSEDES`) or links as a version variant (`VARIANT_OF`); its residual
ids escalate to the capability-gated engine tier, also seeded by
`dedup_features`: `GraphComputeEngine.resolve_candidates` (the engine's
`ResolveCandidates` op) computes all-pairs cosine similarity at or above
`sim_threshold`, then union-find clusters same-type pairs at or above
`merge_threshold` — same-type clusters merge as `same_as`, cross-type
clusters link as `VARIANT_OF`.
</div>

The native engine op (`epistemic-graph` `algorithms::resolve_candidates`) is **read/propose
only** — it returns `MergeProposal{canonical, members, score, kind}` and never mutates; the
Python side decides what to apply via `BatchUpdate`. It is the scale tier the ladder's
*residual* escalates into, replacing an O(N²) client-side embedding pass.

## Community summarization + schema discovery

<div class="admonition architecture" markdown>
<p class="admonition-title">Two pipelines: community summarization and schema discovery</p>

**Community summarization (GraphRAG pipeline phase).** The communities
phase (native Louvain tag) feeds the community_reports phase, which asks
a lite LLM for a theme+summary per community, producing `CommunityReport`
nodes (+ `PART_OF_COMMUNITY`). Those reports feed both a level-1 global
report and `graph_query`/`graph_search`.

**Schema discovery.** `ontology_derive action=discover_extensions`
(MCP + REST) samples documents; an LLM proposes types from the sample;
those are diffed against the live ontology (schema + synonyms); anything
missing becomes a `.ttl` proposal (`RESERVE-PENDING`), which goes through
concept reservation + the evolution pipeline (human/SHACL-gated) before
landing in `ontology_*.ttl`.
</div>

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
- **Resolution** runs the deterministic ladder (AU-AHE.assimilation.merge-entities) extended with transliteration +
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

**Human-gated (deferred):** engine rebuild, image publication, and an
orchestrator-managed redeploy are required to serve the `ResolveCandidates` op;
B-proposed ontology classes await review and concept reservation.
