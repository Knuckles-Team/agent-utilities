# Design Document: Deterministic, zero-LLM social/text entity extraction

CONCEPT:AU-KG.ingest.deterministic-social-entity-mining

> `agent_utilities/knowledge_graph/enrichment/extractors/social.py`
> (`extract_structured_entities`, `to_kg_rows`, `resolve_known_tools`),
> `agent_utilities/knowledge_graph/ontology.ttl` (the entity classes this
> extractor writes), pinned by
> `tests/unit/knowledge_graph/enrichment/test_social_extractor.py`.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| staged deterministic-write → LLM-enrich pipeline (`knowledge_graph/ingestion/staged_pipeline.py`) | the two-stage ingestion pipeline this extractor is a deterministic-stage contributor to | high | KG |

### Extension Analysis

- **Primary Extension Point**: the staged pipeline's deterministic-write
  stage, which previously had nothing real to extract for social/text-
  platform content beyond structural writes.
- **Extension Strategy**: augment — give the existing deterministic stage a
  concrete extractor for one content shape, not a new pipeline stage.
- **New Concept Required?**: Yes — the extractor is a distinct, reusable
  primitive even though it slots into an existing stage.

## Problem

A source platform (X/Twitter, Mastodon, and API-version-compatible peers)
already ships its own structured entity metadata — hashtags, @-mentions,
outbound URLs — inline in an already-fetched record. Sending every such
record through an LLM to re-derive facts the platform already handed over is
wasted cost and an extra network round trip for zero new information.

## Decision

Mine the platform's own structured metadata directly, deterministically, with
no LLM call: `extract_structured_entities` is schema-defensive extraction
from a record's nested `entities`-shaped metadata, tolerant of the exact
upstream API-version differences a raw payload carries (a v1.1 `legacy`
wrapper vs a v2 top-level `entities`; `tag` vs `text` for hashtags;
`screen_name` vs `username` for mentions; `expanded_url` vs `url` for links)
via the shared dotted-path digger, so a connector needs no per-shape
branching of its own. `resolve_known_tools` is a curated, exact +
registered-suffix lookup layered on top. This is the free-first stage a
connector should run BEFORE any LLM-based enrichment of the same record —
cheaper, deterministic, reproducible, and auditable — and it gives the
existing staged deterministic-write → LLM-enrich pipeline something real to
extract for this content shape rather than only doing structural writes.

## Wire-First

`to_kg_rows` projects the extracted entities onto the ontology classes in
`agent_utilities/knowledge_graph/ontology.ttl`; pinned by
`tests/unit/knowledge_graph/enrichment/test_social_extractor.py`.
