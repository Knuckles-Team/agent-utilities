# AU-BASELINE-001 — Baseline ingestion and served ontology content

Status: BUILDING. Owner: agent-utilities.
Cross-repo IDs: SDK-CONNECTOR-CONTROL (ontology resource helper), SDK-SOURCE-INGEST (source ingest boundary),
EG-DECISION-ENGINE-R030 (decision-based ingestion lane routing).
Public provenance: the pull request that adds this spec.

## Purpose and user stories

A fresh Graph OS store holds no skills, prompts or code.
Retrieval then refuses chat because the semantic index is empty.
An operator expects a new deployment to ground itself without a manual sync step.

- As an operator, I deploy Graph OS on an empty store.
  The daemon role ingests the grounding corpus in the background.
- As an operator, I watch that ingestion through the existing task surface.
- As a client, I read each fleet package's ontologies and SHACL shapes as MCP resources.

## Requirements and acceptance

The [requirements table](requirements.md) defines each requirement ID and its closing proof.

- FR-001 (`AU-BASELINE-R001`): The daemon role starts one baseline thread after its daemons start.
  The thread enqueues durable WorkItems and ends. Serving never waits on it.
  A launch or planning failure logs one error and never stops startup.
- FR-002 (`AU-BASELINE-R002`): The baseline covers the prompt library, the configured skill providers and workspace code.
  It never calls the external connector sweep.
- FR-003 (`AU-BASELINE-R003`): Every baseline leg is content-hash delta.
  A repeat boot re-enqueues the legs, and unchanged content costs one hash check.
  A target with a live WorkItem is not enqueued twice.
- FR-004 (`AU-BASELINE-R004`): AU orders ingestion work by the EG stage sequence.
  Source commit work drains first, graph projection and lexical work next, vector work last.
  The tier names match the EG `SemanticQueueClass` wire values.
- FR-005 (`AU-BASELINE-R005`): Every server that the fleet factory builds serves each installed ontology provider.
  Ontology files appear as `ontology://<provider>/<file>.ttl`. Shape files appear as `shapes://<provider>/<file>.ttl`.
- SC-001: On an empty store with a daemon-role process, the task surface lists baseline WorkItems within one minute of boot.
- SC-002: A second boot over unchanged content enqueues the same legs and writes no changed nodes.
- SC-003: `resources/list` on a factory-built fleet server returns one `ontology://` entry per packaged ontology file.

## Scope and interfaces

Owned behavior:

- `agent_utilities.knowledge_graph.ingestion.baseline_ingest`: plan, enqueue and the background start.
- `agent_utilities.knowledge_graph.core.semantic_tiers`: the stage-to-class table and task routing.
- `agent_utilities.mcp.content_resources`: ontology provider registration on factory-built servers.
- Configuration: `KG_BASELINE_INGEST`, `KG_BASELINE_SKILL_PROVIDERS`, `KG_BASELINE_CODEBASES`, `KG_BASELINE_MAX_CODEBASES`.

Dependencies:

- The connector SDK `register_ontology_resources` helper (SDK-CONNECTOR-CONTROL).
- The EG `SemanticQueueClass` vocabulary.

Non-goals:

- Copying external source data. The virtual-graph direction keeps external records at their source.
- Driving the EG `SemanticIndex` stage queue. That method admits SQL-sourced bindings only.
  EG-DECISION-ENGINE-R030 specifies the routing decision that a later slice consumes.
- Changing the serving-role deployment. A client-role process runs no baseline by design.

## Traceability

| Requirement | Design section | Test ID | Evidence |
|---|---|---|---|
| AU-BASELINE-R001 | [plan.md#live-integration-path](plan.md#live-integration-path) | T-001, T-002 | PENDING |
| AU-BASELINE-R002 | [plan.md#architecture](plan.md#architecture) | T-003, T-004 | PENDING |
| AU-BASELINE-R003 | [plan.md#architecture](plan.md#architecture) | T-005 | PENDING |
| AU-BASELINE-R004 | [plan.md#interfaces-and-data-model](plan.md#interfaces-and-data-model) | T-006 | PENDING |
| AU-BASELINE-R005 | [plan.md#interfaces-and-data-model](plan.md#interfaces-and-data-model) | T-007 | PENDING |

## Open questions

- The live serving profile runs as a client with `KG_DEV_MODE=true`. No daemon-role process exists there.
  The operator decides between a daemon-role deployment and the unified sidecar profile.
- The task surface shows baseline WorkItems by job ID prefix `baseline-`. A dedicated filter is a later slice.
