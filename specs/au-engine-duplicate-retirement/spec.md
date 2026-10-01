# AU-RETIRE-001 — Retire duplicate graph engine authority in AU

**Owner:** agent-utilities. **Status:** SPECIFIED; acceptance NOT_AUDITED. **Related public owner:** epistemic-graph owns durable graph execution and native OWL/SPARQL/compute. This AU spec owns only AU deletion, caller migration, and parity proof.

## Outcome and scope

An agent can execute the same authorized graph task through a typed epistemic-graph client while AU contains no competing graph engine, OWL/SPARQL interpreter, or graph compute authority. Retain AU model-driven planning, agent workflow, context compilation and policy; move deterministic graph operations to the engine. This covers the previously unclassified long tail as well as obvious duplicates: classify every AU `knowledge_graph/` module by actual behavior before deleting it. The related [boundary cutover spec](../au-boundary-deconstruction/spec.md) owns per-module AU-BOUNDARY-R012, AU-BOUNDARY-R018–AU-BOUNDARY-R035 cuts; AU-RETIRE-R001 owns the cross-cutting audit and no-dual-authority verdict.

## Requirements

| ID | Observable requirement | Acceptance |
|---|---|---|
| FR-1 | Produce a tracked inventory of each AU graph implementation and live caller, with one of retain-agent, migrate-engine, migrate-SDK, migrate-GraphOS or delete-unused, plus replacement API and decision evidence. | All tracked `knowledge_graph/` Python modules have one disposition; unknowns block deletion. |
| FR-2 | For each duplicate graph operation, invoke one pinned generated EG API from AU's existing application port; preserve tenant, actor, graph, revision, time budget, result shape and typed error semantics. | Positive and denial tests compare old/new behavior on the same synthetic graph before old code is removed. |
| FR-3 | Delete AU graph compute, OWL/SPARQL and deterministic reasoning paths only after the replacement is reachable from a real agent request. | Import, script and dependency census finds no remaining AU graph authority or fallback; served caller proof identifies the route. |
| FR-4 | Reject incompatible client digest or denied scope before any write; do not silently route to an AU fallback on EG outage. | Negative tests show zero side effects and stable failure codes. |

## Design and test traceability

| Requirement | Design | Tests |
|---|---|---|
| FR-1 | [Inventory and migration](plan.md#inventory-and-migration) | RETIRE-1 |
| FR-2 | [Interfaces and data flow](plan.md#interfaces-and-data-flow) | RETIRE-2, RETIRE-3 |
| FR-3 | [Cutover](plan.md#cutover) | RETIRE-4 |
| FR-4 | [Failure and security](plan.md#failure-and-security) | RETIRE-5 |

The per-directory outcome of this audit is published as the [deletion and relocation inventory](../au-boundary-deconstruction/coverage.md#deletion-and-relocation-inventory).

Success means one graph authority, no AU copy, and exact-revision parity evidence. Creating this spec alone does not establish that result.

Requirement IDs are defined in [requirements.md](requirements.md); delivery state per ID is in `status.json`.
