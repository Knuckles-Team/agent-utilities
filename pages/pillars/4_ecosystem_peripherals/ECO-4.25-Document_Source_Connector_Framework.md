# Document-Source Connector Framework (CONCEPT:AU-ECO.connector.document-source-framework)

## Overview
A `load` / `poll` / `slim` connector abstraction for ingesting external **documents**
(websites, filesystems, databases, SaaS apps) into the Knowledge Graph as first-class
`Document` + `Chunk` ontology objects. Ports the Onyx/Danswer connector surface
(`LoadConnector`/`PollConnector`/`SlimConnector`) onto agent-utilities' semantic core:
documents ingested this way inherit OWL semantics, bitemporal slicing, reified
`HAS_CHUNK`/`CHUNK_OF` links, and entailment-aware ACLs (AU-KG.ontology.redact-object-materialize-restricted) — a strict superset
of a flat vector index.

## Implementation Details
- **Source Code**: `agent_utilities/protocols/source_connectors/base.py`
  (`SourceDocument`, `ExternalAccess`, `LoadConnector`, `PollConnector`, `SlimConnector`,
  `PermSyncConnector`)
- **Reference connectors**: `connectors/web.py`, `connectors/filesystem.py`,
  `connectors/rest.py`, `connectors/database.py` (Postgres via `UniversalConnector` —
  the proven native path; other dialects route through `mcp_tool` instead),
  `connectors/mcp_package.py` (ECO-4.29 fleet adapter), `connectors/mcp_tool.py`
  (AU-KG.ingest.mcp-tool-connector universal MCP-tool source — sql-mcp/objectstore-mcp/servicenow/… as
  declarative, checkpointed sources), `connectors/git_markdown.py`
  (AU-ECO.connector.git-markdown-revision-connector — a git working tree, revision-scoped: content is
  read via `git show <sha>:<path>` at an exact commit, never the live working
  tree, and `git diff --name-status` between two revisions is the incremental
  change feed. A genuine native connector, not an `mcp_tool` preset, because
  git has real revision semantics a filesystem walk cannot express — see the
  module's own docstring for the full preset-vs-connector reasoning. Ships
  two domain-pack presets, `au-pillars`/`au-skills`, proving several
  structurally different static markdown corpora can be mounted at once with
  epistemic-graph authoritative across them.)
- **Ingestion**: `ContentType.CONNECTOR` adaptor in `knowledge_graph/ingestion/engine.py`;
  facade `kg.ontology.run_connector(...)`; MCP tool `source_connector`; REST `/connector/*`.
- **Source routing**: bulk hot-path extraction → native `database` (Postgres);
  harvest/poll workloads (tables, buckets, SaaS records) → `mcp_tool` via the
  owning fleet server (per-page MCP overhead is negligible vs chunk/embed/enrich
  cost); no fleet server → generic `rest`/`web`/`filesystem`.
- **Pillar**: ECO
