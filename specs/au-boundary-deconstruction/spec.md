# AU boundary deconstruction

**ID:** AU-BOUNDARY-001
**Owner:** agent-utilities
**Delivery:** SPECIFIED; implementation and acceptance must be checked against an exact merged commit.
**Program items:** EH-476–EH-516, except EH-470–EH-475 (AU-SEMANTIC-001), EH-513 (AU-CONTEXT-001). The numbered AUD-01–AUD-33 cuts map one to one to EH-476–EH-516 where present. EH-514 owns the final boundary gate.

## Outcome

Agent Utilities exposes one agent execution application API. It no longer hosts a second graph engine, public gateway, connector transport, durable data store, or development governance service. A contributor can implement any cut using this spec and the checked-in source tree without private workspace context.

## Ownership and scope

| Cut | Items | Remove from AU after replacement is wired | Authority and retained AU behavior |
|---|---|---|---|
| Served shell | EH-476–EH-479, EH-488–EH-492, EH-515 | `gateway/`, `deployment/`, legacy `mcp/kg_server.py` and `mcp/tools/`, multiplexer family, `server/`, A2A/ACP/AG-UI hosts, served messaging adapters, duplicate console scripts, non-API imports by clients | graph-os hosts public REST/MCP/A2A, fleet, deployment, messaging; AU retains execution methods in `agent_utilities.api` |
| Connector lifecycle | EH-480–EH-486, EH-499–EH-501 | connector toolkit, `protocols/source_connectors/`, source polling/transport, vendor extractors and sinks, certification | agent-connector-sdk owns source access, cursors, pack authoring and write-back; AU submits typed agent claims only |
| Graph and semantic state | EH-487, EH-493–EH-498, EH-502–EH-510, EH-516 | second EG projection, graph engine/session facade, durable work/queue, tenancy, graph analytics, deterministic ingestion, ontology/SHACL, graph DTO copies, legacy SPARQL setup, retrieval engines, schema-drift authority and durable memory | epistemic-graph owns durable records, types, schema, reasoning, retrieval and memory; AU keeps policy/context compilation over generated client ports |
| Cross-cutting state | EH-511–EH-512 | AU SQLite usage facts and `governance/` merge/lane tooling | EG stores usage facts; repository-manager owns repository/lane governance; AU emits usage events |
| Boundary certification | EH-514 | undocumented or duplicated owner paths, stale scripts and callers | `architecture/component-registry.yml` and import/script gates match actual owner decisions |

EH-513 is a finance-specific AU cut with its own spec. Cuts may be grouped in PRs only when each ownership edge has a live caller and no dual authority remains.

The [deliverable matrix](cutover-matrix.md) defines the concrete source cut, target behavior and acceptance focus for every EH-476–EH-516 item.

## Requirements

1. Preserve observable behavior through the new owner and retire each replaced AU entry point in the same coherent change. Do not leave a forwarding alias or dual write as a permanent compatibility layer.
2. `graph-os` and frontends call AU only through `agent_utilities.api` for agent execution. AU application code never imports public transport, source vendor SDK, or persistence implementation.
3. A migration includes an explicit old-operation → new-operation inventory. Each old operation has a tested new owner or a reviewed deletion reason. Authorization, tenant scope, idempotency, error code and receipt parity are checked on the real entry point.
4. Replace AU graph DTOs with generated EG types from one pinned contract digest. Reject incompatible generations before side effects; do not hand-copy a wire model or digest algorithm.
5. Migrate durable usage and memory through tenant-scoped EG methods; no local SQLite fallback. Old untrusted records are quarantined until an authorized migration can attest source and tenant.
6. Remove obsolete dependencies, extras, console scripts and tests importing deleted modules only after callers are rewired and positive/negative tests exist at the destination.
7. A generated owner-manifest check rejects new AU code under destination-owned paths and duplicate script owners. It must run with public checkout fixtures and no live service.

## Acceptance

All mapped cuts are individually evidenced at exact merged revisions, public entry points have served or wiring parity, the forbidden-import and owner-manifest gates pass, and whole-repository quality gates are green. A source branch marked built does not count as accepted.
