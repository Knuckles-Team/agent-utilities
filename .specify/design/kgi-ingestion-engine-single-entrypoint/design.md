# Design Document: One `IngestionEngine`, dispatching by `ContentType`, is the single entrypoint for all data ingestion into the KG

CONCEPT:AU-KG.ingest.ingestion-engine

> `agent_utilities/knowledge_graph/ingestion/__init__.py:1-8` (the package
> statement), `agent_utilities/knowledge_graph/ingestion/engine.py:1-6,192-199,673-680`
> (the `IngestionEngine` class and `ContentType` enum this entrypoint wraps).

## Decision — a single `IngestionEngine`, with content-typed `@adaptor` methods keyed by `ContentType`, is the one path all content enters the KG through

The package docstring states the architecture directly: **"Single entrypoint
for all data ingestion into the Knowledge Graph. Content-typed adaptors
handle codebase, document, social, SPARQL, skill, MCP server, policy, event
stream, and prompt ingestion."** (`__init__.py:3-7`). `engine.py:673-680`
reiterates the same commitment at the class itself: **"All content enters
the KG through this engine. Each `ContentType` maps 1:1 to a registered
`@adaptor` method."**

**The rejected alternative is per-content-type ingestion engines/entrypoints**
— a codebase ingester, a document ingester, an event-stream ingester, each
its own top-level class/module with its own conventions for hashing,
delta-detection, provenance stamping, and enrichment. That is the natural
outcome of adding ingestion support incrementally without a unifying
contract, and it is exactly what this decision forecloses: nine distinct
content types (`CODEBASE`/`DOCUMENT`/`SOCIAL`/`SPARQL`/`SKILL`/`MCP_SERVER`/
`POLICY`/`EVENT_STREAM`/`PROMPT`, `engine.py:192-199`) all flow through one
class, so cross-cutting concerns — the unified enrichment layer
(`AU-KG.ingest.deterministic-extraction-default`), content-hash delta-skip,
and priority-scoped background ingestion — are implemented ONCE on the
engine rather than N times per content type. `ContentType.classify()`
(`engine.py:216-220`) is the single path/URL → `ContentType` mapping every
caller (the MCP `graph_ingest` wrapper included) shares, rather than each
caller guessing or hardcoding the type.

The `__init__.py` re-export surface (`ChangeEnvelope`/`Operation`,
`ContentType`/`IngestionEngine`/`IngestionManifest`/`IngestionResult`,
`Artifact`/`Fragment`/`FragmentKind`) is the concrete evidence of "single
entrypoint": every caller importing from `knowledge_graph.ingestion` gets
the whole ingestion vocabulary from one package, not from N per-adaptor
submodules each with its own public surface.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ingestion/__init__.py`,
  `knowledge_graph/ingestion/engine.py` (the entire `IngestionEngine`
  class and every registered `@adaptor`).
- **Backward Compatible**: Yes — this documents the existing, foundational
  architecture.
- **Breaking Changes**: None.
- **Known weak point**: because every content type funnels through one
  class, a defect or performance regression in a shared stage (e.g. the
  unified enrichment pass, the content-hash registry) affects ALL content
  types simultaneously rather than being isolated to one adaptor — the
  concentration that buys consistency also concentrates blast radius.
