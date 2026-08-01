# Design Document: ETL and connector-sync lineage reuse the EXISTING provenance ontology — zero new node/edge types — and recording is always best-effort

CONCEPT:AU-KG.ontology.kg-3

> `agent_utilities/knowledge_graph/etl/lineage.py`,
> `agent_utilities/mcp/tools/ontology_tools.py:1361`.

## Decision — every `graph_etl` run and connector sync records its lineage trail using the SAME provenance node/edge types every other provenance record already uses

`lineage.py:4-19` states the reuse decision directly: "Reuses the existing
provenance ontology — **NO new node/edge types**": a run is a
`PROVENANCE_AGENT` node (`kind="etl_run"`) with `source`/`sink`/`direction`/
`nodes`/`edges`/`status`/`at` properties; `source`/`sink` systems are
`PROVENANCE_AGENT` marker nodes using the SAME `urn:source:`/`urn:sink:`
scheme the Stardog named-graph partitioning and `sparql_ingestor` already use;
`WAS_DERIVED_FROM` edges chain `sink → run → source` so a graph walk
reconstructs the flow. Lineage recording is explicitly best-effort: "a
failure to record never fails the ETL run itself."

The MACHINE TRIAGE TOOL flagged this id "retire" for the same id-shape reason
as `kg-2` — reading the site shows it is the concept the module's own
docstring names for its lineage-recording decision, not a bare legacy
citation.

**The rejected alternative is a lineage-specific node/edge type family** — a
`LineageRun`/`LineageEdge` schema purpose-built for ETL provenance. Reusing
the general provenance ontology instead means every existing provenance
consumer (impact analysis, "where did this data originate" queries) already
knows how to walk an ETL lineage trail without learning a second vocabulary —
the cost is that an ETL run's specific facts (`source`/`sink`/`direction`/
counts) live as properties on a generically-typed node rather than a
purpose-fitted schema.

The connector-sync extension (`record_connector_sync_activity`,
`record_connector_sync_claim`, `lineage.py:22-35`) applies the SAME
discipline at a different granularity: ONE `PROVENANCE_ACTIVITY` node per
sync run — "never one per ingested row" — with each synced row linked to it
atomically via the row's own write. The summary claim uses "the same
lightweight, directly-verified claim-persistence path
`orchestration.agent_dispatch_worker` already uses ... **never** the governed
mining-flywheel lifecycle, which is reserved for INFERRED findings needing a
confidence floor and review" — a deliberate choice that a directly-observed
sync summary ("source X said N records as of T") is not treated as an
inference requiring review, because it isn't one.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/etl/lineage.py`,
  `mcp/tools/ontology_tools.py` (lineage query surface), every `graph_etl`/
  connector-sync run.
- **Backward Compatible**: Yes — additive recording; best-effort means a
  lineage-recording bug degrades to missing lineage, never a failed sync.
- **Known weak point**: best-effort recording means a lineage gap is silent —
  an operator running an impact-analysis query has no signal distinguishing
  "no lineage exists because nothing flowed" from "lineage recording failed
  for this run."
