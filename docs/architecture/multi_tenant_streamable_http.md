# Multi-Tenant graph-os over Streamable-HTTP

Serving `graph-os` as a **streamable-HTTP MCP surface for thousands of clients**:
hierarchical **org → user** isolation, **private-by-default** memory with an
explicit **commons / markings** sharing path, full **tenant-stamped audit**, and an
**elastic per-tenant engine pool** — all opt-in, so single-tenant/local behaviour
is byte-for-byte unchanged when the flags are off.

Concepts: **OS-5.14** (served identity), **AU-KG.sharding.tenant-partitioned-sharding-hrw** (tenant→named-graph→shard),
**AU-KG.compute.data-is-private-its** (org→user sharing + commons), **AU-KG.backend.concept-2** (Postgres RLS), **AU-KG.sharding.elastic-over-kg-shard**
(engine pool), **AU-OS.safety.ontological-guardrail/5.11** (tenant-scoped fleet + audit). See also
[engine_sharding](engine_sharding.md), [company_brain_runtime](company_brain_runtime.md),
[state_externalization](state_externalization.md).

---

## Topology

One image (`graph-os`), three stateless tiers + central durable state. The cloud
(k8s) and homelab (Swarm) profiles differ only in replica counts and placement —
see [`deploy/`](../../deploy/README.md).

```mermaid
flowchart TD
    KC[Keycloak / OIDC] -.->|JWT: org_id→tenant_id, sub→actor_id| C[clients]
    C -->|Bearer JWT| LB[Load Balancer / Ingress]
    LB --> F["FRONT TIER<br/>stateless streamable-HTTP + gateway<br/>KG_DAEMON_ROLE=client"]
    F -->|ActorContext tenant_id, actor_id, roles| R["Tenant Router<br/>AU-KG.sharding.tenant-partitioned-sharding-hrw HRW + AU-KG.sharding.elastic-over-kg-shard warm pool"]
    R --> E1["ENGINE shard 1<br/>tenant graphs · role=host"]
    R --> E2[ENGINE shard N]
    R --> CM["COMMONS engine<br/>shared default graph · read-mostly"]
    E1 -->|fan-out| L3
    E2 --> L3
    CM --> L3
    F --> ST
    subgraph DURABLE [Central state + optional mirror]
      L3[("Postgres / pg-age mirror<br/>optional write-only fan-out · RLS by tenant_id")]
      ST[("STATE_DB_URI<br/>sessions · goals · durable_checkpoints")]
    end
```

## The five isolation layers (defense in depth)

```mermaid
flowchart LR
    A["1 · Identity<br/>OS-5.14 served JWT"] --> B["2 · Physical<br/>AU-KG.sharding.tenant-partitioned-sharding-hrw named graph per org"]
    B --> C["3 · Logical<br/>KG-2.6 tenant scope + AU-KG.compute.data-is-private-its owner/scope"]
    C --> D["4 · Database<br/>AU-KG.backend.concept-2 Postgres RLS app.tenant_id"]
    D --> E["5 · Audit<br/>OS-5.11 tenant+actor stamped"]
```

1. **Identity (OS-5.14).** `ActorIdentityMiddleware` mints `ActorContext{tenant_id,
   actor_id, roles}` from a validated JWT (`org_id→tenant_id`, `sub→actor_id`). The
   **served-security profile** (`apply_served_security_profile`) refuses to serve a
   network transport without `AUTH_JWT_JWKS_URI` (fail-loud, not fail-open) and turns
   on `KG_AUTH_REQUIRED` + `KG_BRAIN_ENFORCE`, so unauthenticated HTTP is rejected and
   the privileged `SYSTEM_ACTOR` fallback is unreachable over the network.
2. **Physical (AU-KG.sharding.tenant-partitioned-sharding-hrw + AU-KG.compute.data-is-private-its).** Under enforcement, each org routes to its own
   named graph `tenant__<slug>__<base>` — **even on a single engine endpoint** (HRW
   over one endpoint is the identity). Cross-org data is physically separate.
3. **Logical (KG-2.6 + AU-KG.compute.data-is-private-its).** On a shared graph, `scope()` injects
   `n.tenant_id = <org>` (the simple, parseable predicate) and a Python-side
   `visible()` filter applies private-by-default owner/scope. Applied at the
   `query_cypher` MCP read chokepoint and `facade.query`.
4. **Database (AU-KG.backend.concept-2).** Postgres Row-Level Security keyed on the per-session GUC
   `app.tenant_id` filters rows beneath everything else; `WITH CHECK` blocks
   cross-tenant writes. Apply [`deploy/postgres/tenant_rls.sql`](../../deploy/postgres/tenant_rls.sql).
5. **Audit (AU-OS.safety.ontological-guardrail/5.11).** Every `RunTrace`, session, and correlation carrier is
   stamped `tenant_id`+`actor_id`+`correlation_id`; `/api/fleet/*` is tenant-scoped
   (an org admin sees its own org; a platform admin sees the fleet).

## Hierarchical org → user + commons sharing (AU-KG.compute.data-is-private-its)

The **default graph is the commons.** Data is **private to its owner by default**;
sharing is explicit — by **where** it is placed (promote into the commons graph) or
by **how** it is placed (a mandatory marking).

```mermaid
flowchart TD
    W[guarded write] -->|stamp tenant_id, _owner_id, _shared_scope=private| P[private to owner]
    P -->|graph_share action=org| O["org-shared<br/>visible to the org"]
    P -->|graph_share action=commons| K["commons graph<br/>cross-org readable"]
    P -->|graph_share action=mark| M["marking<br/>role-gated, cross-org"]
    O -->|graph_share action=private| P
```

A reader sees: **own** (`_owner_id == me`) ∪ **org/commons-shared**
(`_shared_scope ∈ {org, commons}`) ∪ **unowned** (legacy/system) ∪ the **commons
graph**. Privileged (`admin`/`system`) actors are unrestricted.

Verbs (MCP tool `graph_share` / `POST /graph/share`):

| action | effect | mechanism |
|---|---|---|
| `org` | visible to the owner's org | in-place `_shared_scope='org'` |
| `commons` | cross-org readable | copy node into the commons graph |
| `mark` | role-gated cross-org | mandatory marking (AU-KG.ontology.redact-object-materialize-restricted) |
| `private` | restrict back to owner | `_shared_scope='private'` |

## Elastic per-tenant engine pool (AU-KG.sharding.elastic-over-kg-shard)

`GRAPH_SERVICE_ENDPOINTS` fixes the shard set; the pool is the *elastic* layer within
a process: a bounded **warm set** of per-tenant engine clients (LRU), **hydrate on
miss**, and (when `KG_ENGINE_POOL_DROP_ON_EVICT` is set and a pg-age mirror holds the data) an
engine-side **per-graph unload** to reclaim memory on eviction. Disabled by default
(`KG_ENGINE_POOL_SIZE=0` → per-use construction, today's behaviour).

## Configuration

| Flag | Default | Purpose |
|---|---|---|
| `KG_SERVED_PROFILE` | on | served fail-closed profile for network transports (`0` opts out, dev only) |
| `AUTH_JWT_JWKS_URI` / `_ISSUER` / `_AUDIENCE` | — | OIDC identity; **required** for the served profile |
| `KG_AUTH_REQUIRED` | off | reject unauthenticated HTTP (auto-on under served profile) |
| `KG_BRAIN_ENFORCE` | off | tenant scope + ACL + owner/scope enforcement (auto-on under served profile) |
| `KG_ACL_DEFAULT_ALLOW` | off | deny-on-missing-ACL when enforcing (fail-closed) |
| `KG_DEFAULT_GRAPH` | `__bus__` | the commons graph; tenants route to `tenant__<slug>__<this>` |
| `GRAPH_SERVICE_ENDPOINTS` | one socket | engine shard set (HRW routing) |
| `GRAPH_DB_URI` / `STATE_DB_URI` | — | pg-age mirror (apply RLS) / central session+checkpoint store |
| `KG_ENGINE_POOL_SIZE` | `0` | warm per-tenant engines (elastic pool); `0` = per-use |
| `KG_ENGINE_POOL_DROP_ON_EVICT` | off | unload the tenant graph from the engine on eviction (needs a pg-age mirror) |

## Tracking clients & their agents

"Which agents did client X spawn?" is a tenant-scoped query: the run-wide
`correlation_id` (OS-5.11) links every spawned agent's `RunTrace`, each stamped
`tenant_id`/`actor_id`; `/api/fleet/*` filters by the caller's tenant. External
side-effects carry `x-tenant-id`/`x-actor-id`/`x-correlation-id` so off-box writes
remain joinable to the originating client.

## Verification

Unit + integration: `tests/unit/knowledge_graph/test_tenant_sharing.py`,
`test_tenant_engine_pool.py`, `test_tenant_request_isolation.py`,
`test_fleet_supervisory.py`, `test_postgresql_backend.py`,
`tests/unit/core/test_request_identity.py`. Live: per-tenant named-graph isolation
verified against a running engine; Postgres RLS (isolation + commons + admin-bypass
+ `WITH CHECK`) verified against Postgres 16 with `deploy/postgres/tenant_rls.sql`.
