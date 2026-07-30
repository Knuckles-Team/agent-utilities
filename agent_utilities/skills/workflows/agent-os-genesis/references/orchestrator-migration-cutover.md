# Orchestrator migration and cutover

Migrate one dependency-closed application unit at a time. Genesis owns the target
substrate; `agent-utilities-deployment` owns application render, migration, and
verification.

## Before change

- inventory the source artifact, configuration, secret references, identity, DNS,
  certificates, networks, volumes, database/data format, queues, and external
  dependencies;
- capture traffic, error, latency, and data-integrity baselines;
- build and validate target artifacts without applying them;
- prove backup and restore;
- define freeze/dual-write semantics and rollback boundary;
- lower DNS TTL only when DNS cutover is selected;
- require explicit approval for data mutation and traffic cutover.

## Sequence

1. Provision/attach the target substrate.
2. Deploy stateful dependencies without sending production traffic.
3. Quiesce or checkpoint source writers according to the data contract.
4. Migrate through supported logical export/import or replication; do not copy live
   database files between incompatible runtimes.
5. Verify counts, hashes, constraints, graph queries, and application-level reads.
6. Deploy application at zero/limited traffic and run internal health/auth/tool tests.
7. Canary user traffic and compare traces, outputs, cost, and latency.
8. Cut over the declared routing layer.
9. Stop or fence the old writer; verify with it unavailable.
10. Observe through the rollback window, then retire old resources separately.

Preserve canonical issuer/audience and external service names where possible. A
post-cutover authorization failure may be stale DNS or a gateway token route, not a
bad credential.

## Rollback

Rollback requires an exact prior artifact, compatible data state, routing action, and
owner. If writes reached the new system, execute the declared reverse replication or
forward-fix plan; never point the old binary at a newer incompatible store.

## Exit gates

User entrypoint, Graph-OS, model execution, selected tool, connector, engine
transaction, and trace must all pass through the target with the source disabled.
Verify persistence, identity, isolation, background checkpoint/resume, alerts, and a
second idempotent apply.
