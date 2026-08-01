# Design Document: One ETL entrypoint composes existing machinery, rather than a bespoke pipeline per source/sink pair

CONCEPT:AU-KG.ontology.one-source

> `agent_utilities/knowledge_graph/etl/pipeline.py` (`run_etl`).

## Decision — `run_etl(source=X, sink=Y)` is a thin orchestrator over the KG's already-existing bidirectional machinery, not a new transport

`pipeline.py:4-30` states the shape directly: "a thin orchestrator that
collapses the KG's existing bidirectional machinery into a single 'move data
between systems' entrypoint. It writes no transport of its own." Inbound
extract+transform+load reuses `core.source_sync.sync_source` (the registered
extractor/hydrator per source) with the ontology layer (interfaces/links/OWL
bridge) as the transform — "the KG is the canonical hub." Outbound load
dispatches by sink kind: a `WritebackSink` domain routes through the existing
`run_writeback` (dry-run-first + `ProposalQueue`, never a direct unguarded
write to a system of record); a graph-store sink routes through
`stardog_sync.push_to_stardog` (partitioned into `urn:source:<system>` named
graphs) or `migration.copy_graph`. Every run is recorded via the existing
`lineage` module for impact analysis, and `run_etl` "stays pure (no MCP/registry
import)" — the caller resolves `sink_backend`, so this module never imports the
MCP tool registry it is itself exposed through.

**The rejected alternative is a bespoke pipeline per source×sink pair** — the
natural growth path without this decision: a `servicenow_to_leanix.py`, a
`leanix_to_stardog.py`, each hand-wiring its own extract/transform/load and its
own lineage recording. `one-source` collapses that N×M surface to one call
shape (`run_etl(source=, sink=)`, either side omittable for a one-directional
run) that composes the SAME extractor/writeback/graph-store primitives every
other part of the system already uses — a new source or sink is a registration
in the existing registries, not a new pipeline module.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/etl/pipeline.py`,
  `knowledge_graph/etl/result.py` (`EtlResult` wire contract),
  `knowledge_graph/etl/lineage.py`.
- **Backward Compatible**: Yes — composes existing paths without changing
  their individual contracts.
- **Known weak point**: `_step_result` projects arbitrary internal step-result
  dicts onto the strict `EtlResult` schema by field-name intersection
  (`pipeline.py:40-57`) — a connector that renames an internal field silently
  moves that value into `details` instead of a typed `EtlResult` field, with no
  error raised.
