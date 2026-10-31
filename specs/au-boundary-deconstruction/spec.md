# MERISTEM-BOUNDARY — AU boundary deconstruction

**ID:** MERISTEM-BOUNDARY
**Owner:** meristem
**Delivery:** SPECIFIED; implementation and acceptance must be checked against an exact merged commit.
**Requirements:** MERISTEM-BOUNDARY-R001–MERISTEM-BOUNDARY-R041. Every requirement ID is defined in [requirements.md](requirements.md); the delivery state and evidence of each are recorded in [status.json](status.json). MERISTEM-BOUNDARY-R038 owns the final boundary gate.

## Outcome

Meristem exposes one agent execution application API. It no longer hosts a second graph engine, public gateway, connector transport, durable data store, or development governance service. A contributor can implement any cut using only this spec and the checked-in source tree.

## Ownership and scope

| Cut | Requirements | Remove from AU after replacement is wired | Authority and retained AU behavior |
|---|---|---|---|
| Served shell | MERISTEM-BOUNDARY-R002, MERISTEM-BOUNDARY-R014–MERISTEM-BOUNDARY-R017 | `deployment/`, A2A/ACP/AG-UI hosts, served messaging adapters, duplicate console scripts, non-API imports by clients | arbora hosts public REST/MCP/A2A, fleet, deployment and messaging (ARBORA-DEPLOY, ARBORA-FLEET). AU keeps the agent harness. AU reaches arbora through `meristem.arbora_client`. |
| Connector lifecycle | MERISTEM-BOUNDARY-R005–MERISTEM-BOUNDARY-R011, MERISTEM-BOUNDARY-R024–MERISTEM-BOUNDARY-R026, MERISTEM-BOUNDARY-R049 | connector toolkit, `protocols/source_connectors/`, source polling/transport, vendor extractors and sinks, certification | agent-connector-sdk owns source access, cursors, pack authoring and write-back; AU submits typed agent claims only |
| Graph and semantic state | MERISTEM-BOUNDARY-R012, MERISTEM-BOUNDARY-R018–MERISTEM-BOUNDARY-R023, MERISTEM-BOUNDARY-R027–MERISTEM-BOUNDARY-R035, MERISTEM-BOUNDARY-R040 | second EG projection, graph engine/session facade, durable work/queue, tenancy, graph analytics, deterministic ingestion, ontology/SHACL, graph DTO copies, legacy SPARQL setup, retrieval engines, schema-drift authority and durable memory | medulla owns durable records, types, schema, reasoning, retrieval and memory; AU keeps policy/context compilation over generated client ports |
| Cross-cutting state | MERISTEM-BOUNDARY-R036–MERISTEM-BOUNDARY-R037 | AU SQLite usage facts and `governance/` merge-queue and concept-reservation tooling | EG stores usage facts. Repository-manager owns repository and worktree governance. AU emits usage events |
| Boundary certification | MERISTEM-BOUNDARY-R038 | undocumented or duplicated owner paths, stale scripts and callers | `architecture/component-registry.yml` and import/script gates match actual owner decisions |

MERISTEM-BOUNDARY-R041 removes the finance modules that produced fabricated data; the remaining finance cut, MERISTEM-CONTEXT-R007, is specified in [`agent-context-and-finance`](../agent-context-and-finance/spec.md). Cuts may be grouped in PRs only when each ownership edge has a live caller and no dual authority remains.

The [deliverable matrix](cutover-matrix.md) defines the concrete source cut, target behavior and acceptance focus for every requirement, and the [deletion and relocation inventory](plan.md#deletion-and-relocation-inventory) maps every AU package directory to the requirement that removes, relocates or keeps it.

Rows MERISTEM-BOUNDARY-R051 onward record the cut contracts that the harness holds as consumer of medulla decisions.

## Requirements

1. Preserve observable behavior through the new owner and retire each replaced AU entry point in the same coherent change. Do not leave a forwarding alias or dual write as a permanent compatibility layer.
2. `arbora` and frontends call AU only through `meristem.api` for agent execution. AU application code never imports public transport, source vendor SDK, or persistence implementation.
3. A migration includes an explicit old-operation → new-operation inventory. Each old operation has a tested new owner or a reviewed deletion reason. Authorization, tenant scope, idempotency, error code and receipt parity are checked on the real entry point.
4. Replace AU graph DTOs with generated EG types from one pinned contract digest. Reject incompatible generations before side effects; do not hand-copy a wire model or digest algorithm.
5. Migrate durable usage and memory through tenant-scoped EG methods; no local SQLite fallback. Old untrusted records are quarantined until an authorized migration can attest source and tenant.
6. Remove obsolete dependencies, extras, console scripts and tests importing deleted modules only after callers are rewired and positive/negative tests exist at the destination.
7. A generated owner-manifest check rejects new AU code under destination-owned paths and duplicate script owners. It must run with public checkout fixtures and no live service.

## Acceptance

All mapped cuts are individually evidenced at exact merged revisions, public entry points have served or wiring parity, the forbidden-import and owner-manifest gates pass, and whole-repository quality gates are green. A source branch marked built does not count as accepted.

Requirement IDs are defined in [requirements.md](requirements.md); delivery state per ID is in `status.json`.

## Retained run scaffold

MERISTEM-BOUNDARY-R043.4 and its children retain the model-edited, per-run scaffold
from historical MERISTEM-HARNESS-R014.1–R014.5. See the dated migration in
[plan.md](plan.md#amendment-2026-10-18-retain-the-run-scaffold-after-the-boundary-cut)
for immutable source provenance and the separate SDK, EG and Arbora read
obligations. These are implementation requirements, not delivery claims.

## Shared child execution explorer

R043.14, R043.15, R043.16 and R043.17 specify execution-only increments
for the shared explorer. Existing verified flat execution and completed-text
history do not establish nested conversations, active child messaging or durable
child transcript availability. Admission, targets and storage remain upstream.
