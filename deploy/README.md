# Deploying multi-tenant graph-os over streamable-HTTP

One image (`graph-os`), one env contract, two profiles — cloud (k8s) and homelab
(Swarm) — differing only in replica counts and placement. This is the deployment
side of the segmentation/sharing/audit stack: **OS-5.14** (served identity),
**AU-KG.sharding.tenant-partitioned-sharding-hrw** (tenant→named-graph→shard), **AU-KG.compute.data-is-private-its** (org→user sharing + commons),
**AU-KG.backend.concept-2** (Postgres RLS), and **AU-OS.safety.ontological-guardrail/5.11** (tenant-scoped fleet + audit).

## Topology

```
 OIDC (Keycloak) ── JWT(org_id→tenant_id, sub→actor_id)
        │
 clients → LB/Ingress → FRONT (stateless streamable-HTTP, autoscaled, role=client)
                              │  ActorContext{tenant_id, actor_id, roles}
                              ▼
                    ENGINE shards (AU-KG.sharding.tenant-partitioned-sharding-hrw HRW: tenant graph → one shard, role=host)
                              │  write-through
                              ▼
            Postgres: pg-age L3 (RLS) + STATE_DB_URI (sessions/checkpoints)
```

## Files
- `k8s/graphos.yaml` — Namespace, ConfigMap/Secret, front Deployment+Service+HPA+Ingress, engine StatefulSet (headless).
- `swarm/graphos.stack.yml` — the same image downscaled: front + one engine + external pg-age.
- `postgres/tenant_rls.sql` — idempotent DB-level tenant isolation (apply once, as table owner).

## Quick start

**Cloud (k8s):**
```sh
kubectl apply -f deploy/k8s/graphos.yaml          # edit image/host/secrets first
kubectl exec -n graphos deploy/graphos-front -- \
  graph-os --help                                  # sanity
psql "$GRAPH_DB_URI" -f deploy/postgres/tenant_rls.sql
```

**Homelab (Swarm):**
```sh
docker node update --label-add graphos_engine=true <engine-node>
docker stack deploy -c deploy/swarm/graphos.stack.yml graphos
psql "$GRAPH_DB_URI" -f deploy/postgres/tenant_rls.sql
```

## The env contract (both profiles)

| Variable | Tier | Purpose |
|---|---|---|
| `TRANSPORT=streamable-http`, `HOST`, `PORT` | front | network MCP surface |
| `AUTH_JWT_JWKS_URI` / `_ISSUER` / `_AUDIENCE` | front | **required** — served profile refuses to start without JWKS (fail-loud, not fail-open) |
| `KG_AUTH_REQUIRED` / `KG_BRAIN_ENFORCE` | front | auth + tenant enforcement (auto-on for network transports; pin to override) |
| `KG_DAEMON_ROLE` | front=`client`, engine=`host` | front pods never own an L1 engine |
| `KG_DEFAULT_GRAPH` | both | the **commons** graph; tenants route to `tenant__<slug>__<this>` |
| `GRAPH_SERVICE_ENDPOINTS` | front | engine shard list; HRW routes each tenant graph to one shard |
| `GRAPH_DB_URI` | both | L3 pg-age (apply RLS here) |
| `STATE_DB_URI` | both | central sessions/goals/durable_checkpoints (AU-OS.state.unified-durable-state-externalization) — lets any pod resume any tenant's goal |

Dev escape hatch: `KG_SERVED_PROFILE=0` serves a network transport **without**
enforced identity (local only).

## Scaling

- **Front** is stateless → scale horizontally (HPA on CPU; Swarm `replicas`).
- **Engine** shards are the partition unit: add an endpoint to `GRAPH_SERVICE_ENDPOINTS`
  and a StatefulSet/Swarm replica. HRW keeps key movement minimal; a graph whose
  HRW winner changes must be moved with the snapshot tooling (AU-KG.sharding.tenant-partitioned-sharding-hrw is not
  auto-rebalancing by design).
- **State/L3** is the one stateful dependency: use managed/HA Postgres in cloud,
  the existing `kg-backbone_pg-age` in the homelab.

## What enforces isolation (defense in depth)
1. **Identity** — `ActorIdentityMiddleware` mints `tenant_id`/`actor_id` from the JWT; the served profile blocks unauthenticated HTTP.
2. **Physical** — AU-KG.sharding.tenant-partitioned-sharding-hrw routes each org to its own named graph → shard.
3. **Logical** — AU-KG.compute.data-is-private-its owner/scope predicate (private-by-default) + KG-2.6 tenant scoping in every guarded read.
4. **Database** — AU-KG.backend.concept-2 Postgres RLS (`app.tenant_id` GUC) under all of it.
5. **Audit** — every RunTrace/session/correlation carrier is stamped tenant+actor; the fleet plane is tenant-scoped.
