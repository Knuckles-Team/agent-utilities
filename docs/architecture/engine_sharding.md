# Authoritative engine placement and sharding

Production GraphOS uses one Epistemic Graph cluster per cell. The cluster owns
placement, fencing and routing epochs; Agent Utilities consumes that authority and
must not invent a second client-side placement map.

## Production topology

The production cell runs three Epistemic Graph members. Every configured MultiRaft
group is replicated across those three members, and a tenant graph resolves to one
group through the authoritative placement catalog. The group is the ordering and
transaction boundary. Cross-group writes use the engine's durable coordinator and
retained prepare/decision records.

```text
verified request
  -> stable cell coordinator
  -> placement catalog (graph, tenant, epoch)
  -> MultiRaft group
  -> three replicated members
```

`EPISTEMIC_GRAPH_RAFT_GROUPS` establishes the group ring. An explicit catalog
assignment overrides the ring. Placement responses include an epoch; a stale request
is redirected or rejected so a move cannot silently write to its former owner.
Agent Utilities obtains this route through the engine `PlacementRoute` RPC and
re-resolves after a stale-epoch response.

`ResolvedEngine` and the middleware-minted `GraphSession` retain the returned group
and catalog epoch. The Python client surfaces a structured `StaleRouteError`; the
GraphSession-routed transport refreshes placement and retries once with the original
idempotency key. Engine-native change envelopes are rebound to the refreshed epoch
and group fence before that retry.

The production templates deliberately configure one
`GRAPH_SERVICE_ENDPOINTS` value: the stable coordinator Service. That Service selects
StatefulSet pod index zero because analytics jobs and admin mutation recovery are
durable non-Raft ledgers today. Load-balancing requests across separate copies of
those ledgers would split authority. MultiRaft still replicates graph state across all
three members; loss of the coordinator pod is recovered by StatefulSet recreation and
reattachment of its retained claim. The certification campaign measures this recovery
time rather than claiming instantaneous failover.

If a non-production topology exposes groups through distinct client endpoints,
configure a JSON `GRAPH_RAFT_GROUP_ENDPOINTS` map keyed by group id. An endpointless
catalog result in a multi-endpoint deployment without that map fails closed. It never
falls back to an invented group-to-host rule.

## Tenant and graph resolution

The named graph remains the isolation key. A verified `ActorContext` supplies the
tenant, scopes and policy version, while an explicit graph name selects the logical
graph within that policy boundary. The engine enforces row-level policy and rejects a
stale placement epoch or fencing token.

Production invariants are:

- every request carries verified identity and tenant context;
- placement is returned by the engine, not reconstructed from an endpoint list;
- all clients in a cell use the same coordinator authority;
- a catalog move preserves graph identity, durable rows, audit chain and replay state;
- projection, indexing and reasoning advance only after authoritative state is
  durable.

There is no client-side placement mode. Multiple configured endpoints are ordered
coordinator contacts; they are never interpreted as a hash ring. If the engine
authority or group-to-endpoint topology is unavailable, the request fails closed.

## Online movement and rebalancing

Two related engine mechanisms move load without changing the client contract:

1. MultiRaft placement can repoint a graph to another group under a per-tenant
   migration fence. Because groups share the authoritative durable graph registry,
   the move checkpoints, changes the catalog route and resumes at a new epoch.
2. A durable multi-shard redb deployment can execute snapshot-plus-delta resharding.
   It bulk-copies a stable snapshot while writes continue, quiesces only for the delta
   and catalog flip, then purges the old copy after the new route is durable.

The rebalance planner is deterministic and bounded. Execution applies one graph move
at a time and resolves current placement before each move, so a stale plan cannot
blindly overwrite newer placement. Operators must use the governed
`engine_resharding` admin surface with `kg:admin`; direct catalog mutation is not an
operational procedure.

Before and after a move, prove:

- quorum and checkpoint health;
- no active incompatible release, ontology or index migration;
- the placement epoch advanced exactly once;
- acknowledged writes, auxiliary authority and audit continuity survived;
- projection/index cursors do not lead authoritative state;
- tenant policy, deletion and stale-fence rejection remain effective.

Online reshard and rebalance are mandatory scenarios in the exact-release
certification campaign.

## Capacity and observability

The engine StatefulSet is not HPA-scaled. Adding or removing a voting member is a
quorum operation; changing group count or tenant placement is a governed resharding
operation. Stateless frontends and standalone dispatch, ingest and analytics workers
scale independently from p99 latency, queue depth, consumer lag and authoritative job
backlog.

Production quorum health comes from ready StatefulSet members plus native
`epistemic_graph_*` Raft, WAL and checkpoint metrics. The
`agent_utilities_engine_shard_up{endpoint}` gauge describes configured coordinator
endpoints; with one stable coordinator it must not be interpreted as a Raft member
count.

See [capacity planning](../scaling/capacity_model.md), the
[production cell runbook](../operations/production-cell-runbook.md), and
[disaster recovery](../operations/disaster-recovery.md).
