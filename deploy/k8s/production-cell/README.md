# GraphOS production-cell topology authority

Status: **renderable production candidate; not deployed, live, or 1M-certified**.

The YAML in this directory is a source template. The only path that can produce
a deployment candidate is `scripts/release/render_production_cell.py` with a
signed release manifest and a separately retained, measured topology input. The
committed `production-input.example.json` is schema-shaped synthetic data and
must never be treated as host inventory. The renderer refuses absent, stale,
malformed, unverified, secret-bearing, or capacity-inconsistent input before it
creates an output directory.

## Required topology bindings

The input binds one measured ResourcePool to the canonical AU production-cell
identities:

| Plane | Identity | Contract |
| --- | --- | --- |
| gateway (API + MCP) | `graphos-front` / `graphos-control` | `graph-os`; `agent_utilities_gateway_in_flight_requests` |
| dispatch | `graphos-dispatch-worker` / `graphos-cell` | `agent-dispatch-worker`; `agent_utilities_dispatch_queue_depth` |
| ingest | `graphos-ingest-worker` / `graphos-cell` | `kg-ingest-worker`; `agent_utilities_kg_ingest_queue_depth` |
| mining | `graphos-analytics-worker` / `graphos-cell` | `graph-os-analytics-worker`; fixed until a bounded signal exists |
| engine client | `epistemic-graph-coordinator` / `graphos-cell` | TLS RPC `9101` |
| engine peer | `epistemic-graph-raft` / `graphos-cell` | StatefulSet peer identity; Raft `9100`; metrics `9102` |

The three engine members use retained per-member storage, `OnDelete` activation,
quorum-preserving PDB settings, explicit drain/preStop timing, and verified TLS
service discovery. Workload HPAs use the exact external metric plus a
`graphos_workload` selector emitted by the renderer. Mining remains fixed rather
than inventing an unbounded metric.

The input also carries current/rollback digests for every workload and the
engine, OIDC discovery assertions, retained ConfigMap/Secret/session-store and
action-audit references, and separate rollout/rollback evidence references.
`topology-contract.yaml` and `release-pins.yaml` preserve bounded digests and
references in the rendered output; they do not contain credential material.

## Cross-repository integration boundary

All three production renderers consume the versioned
`services/epistemic-graph/k8s/production/engine-identity-contract.v1.json`.
Its digest is a required input/evidence binding, and root validation must prove
that every emitted Service, port, TLS server name, discovery identity, and
engine namespace matches it. The historical `epistemic-graph`/
`epistemic-graph-peer` names are explicit `migration-only` aliases with
`live_authority: false`; they cannot be silently mixed into a rendered or live
topology.

`deploy/k8s/production-cell/` does not deploy or claim live/1M evidence. Inventory
measurements, retained authority existence, server-side diff, rollout health,
session continuity, and metric freshness remain operator/root validation inputs.
