# Relational control-plane authority and placement

This document is the operator-facing contract for where a fact is allowed to
be written. Its machine-readable source is
[`agent_utilities/governance/relational_authority.json`](../../agent_utilities/governance/relational_authority.json),
and its fail-closed gate is
[`scripts/security/check_relational_authority.py`](../../scripts/security/check_relational_authority.py).
The JSON map is the source of truth; this page explains the boundary and the
reasoning behind it. A new writer, lifecycle field, or event must update the
map and pass the gate in the same change.

## One authority per fact

The control plane is intentionally split by fact type. A projection, cache,
mirror, or read model may copy a bounded reference, but it cannot become a
second writer for the authority field.

| Fact or payload | Sole authority | Permitted elsewhere |
|---|---|---|
| Tenant, principal, membership, role, entitlement, registry identity, lifecycle, immutable version, release, policy, approval, quota, idempotency, outbox and audit references | PostgreSQL transactional control plane | Safe GraphOS references, cache entries carrying source revision/digest |
| Semantic claims, classifications, relationships, evidence and enterprise knowledge | GraphOS / epistemic graph | Relational opaque IDs/digests used by an authorized control-plane decision |
| Work claims, leases, fencing, retries and active execution state | Native durable `WorkItem` authority | Relational run summaries and scheduler references; GraphOS lineage |
| Source/document/package bytes and large immutable artifacts | Artifact/object store or the source system | Relational media/size/digest/location; GraphOS evidence references |
| Embeddings and ANN index payloads | Tenant/physical-graph vector store | Model, dimension, chunk and index references |
| Raw metrics, logs, traces and high-cardinality observations | Observability backend | Bounded SLO/usage summaries and trace IDs |
| Secret, OAuth, refresh-token and private-key material | Approved secret/token provider | Opaque reference, status, expiry and rotation metadata only |

PostgreSQL is therefore a transactional *control* plane, not a second graph,
blob store, vector database, telemetry backend or vault. GraphOS is the
semantic knowledge authority, not the registry lifecycle writer. The native
WorkItem protocol is the only lease/fence authority; a relational `runs` or
graph node may point at a WorkItem but cannot reproduce its mutable state.

## Lifecycle and event rules

The `authority_placement.records` section assigns each canonical lifecycle
field exactly once. The `authority_placement.events` section binds safe event
metadata to the same owner. Event payloads carry IDs, versions, digests and
bounded references—not raw content or credentials. The gate rejects:

- an omitted or unknown authority store, record, lifecycle field, or event;
- a duplicate record, lifecycle field, event, or event payload field;
- a lifecycle field assigned to a different store than the closed contract;
- an event whose writer differs from its lifecycle-field owner;
- secret-bearing table columns or event fields (`password`, private keys,
  client secrets, access/refresh tokens, API-key values, bearer tokens,
  authorization headers, cookie values, and equivalent names);
- schema drift between the concrete relational DDL and the declared table map;
- a read model that is not explicitly write-forbidden.

Opaque references such as `secret_ref`, `auth_profile_ref`, key fingerprints,
digests and rotation timestamps are metadata and remain allowed. The value
behind an opaque reference must be resolved only by the owning provider at the
shortest practical lifetime.

## Projection and replay boundary

The PostgreSQL transaction that changes a control-plane record also writes one
bounded, immutable outbox event. A projector may retry that event and may
publish a safe GraphOS relationship, cache entry, or observability summary.
It acknowledges only after the target reflects the source authority and
revision. Projection failure is visible as lag/degraded state; it never edits
the source record or invents an empty healthy result.

GraphOS observations flowing toward a control-plane decision are evidence,
not reverse synchronization. The observation retains the explicit connection,
physical graph, graph revision, query/template digest, policy decision and
expiry. A steward or policy workflow must promote it into a new PostgreSQL
version; arbitrary graph writes cannot mutate registry identity or policy.

Deletion is a lifecycle tombstone: block new resolutions, emit generation-bound
tombstones, wait for permitted consumers, revoke provider references, and
retain the audit/provenance needed for reproducibility and legal hold. Removing
a projection never removes the authoritative object or its source bytes.

## Verification

Run the bounded gate from the repository root:

```bash
python3 scripts/security/check_relational_authority.py
```

The focused authority tests plant a duplicate-writer record, a secret-bearing
column, and a secret-bearing event. Each must fail closed. Root/orchestrator
validation additionally owns the normal test, lint, formatting, hook and
release-gate evidence; implementation tracks do not claim that evidence.

The contract is additive to the concrete three-domain map in
[`relational-authority.md`](relational-authority.md). That older page documents
the existing fleet-catalog, usage and state schemas; this page extends the
same source with the complete cross-system placement decision from the
program's authority/placement feedback.
