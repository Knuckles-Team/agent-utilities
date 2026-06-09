# Document-Source Connector Framework (CONCEPT:ECO-4.25)

## Overview
A `load` / `poll` / `slim` connector abstraction for ingesting external **documents**
(websites, filesystems, databases, SaaS apps) into the Knowledge Graph as first-class
`Document` + `Chunk` ontology objects. Ports the Onyx/Danswer connector surface
(`LoadConnector`/`PollConnector`/`SlimConnector`) onto agent-utilities' semantic core:
documents ingested this way inherit OWL semantics, bitemporal slicing, reified
`HAS_CHUNK`/`CHUNK_OF` links, and entailment-aware ACLs (KG-2.46) — a strict superset
of a flat vector index.

## Implementation Details
- **Source Code**: `agent_utilities/protocols/source_connectors/base.py`
  (`SourceDocument`, `ExternalAccess`, `LoadConnector`, `PollConnector`, `SlimConnector`,
  `PermSyncConnector`)
- **Reference connectors**: `connectors/web.py`, `connectors/filesystem.py`,
  `connectors/rest.py`, `connectors/database.py` (Postgres/MySQL-MariaDB/MSSQL/Oracle/
  SQLite/Mongo via `UniversalConnector`), `connectors/mcp_package.py` (ECO-4.29 fleet adapter)
- **Ingestion**: `ContentType.CONNECTOR` adaptor in `knowledge_graph/ingestion/engine.py`;
  facade `kg.ontology.run_connector(...)`; MCP tool `source_connector`; REST `/connector/*`.
- **Pillar**: ECO
