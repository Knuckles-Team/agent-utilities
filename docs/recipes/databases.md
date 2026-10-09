# Recipe: pg-age database environment

A beginner's, copy-paste guide to standing up the durable **Postgres** tier
agent-utilities is designed around — Apache AGE + pgvector + ParadeDB
(`pg_search`) — so graph relationships are backfilled into AGE alongside the
engine's own authoritative store.

> **The short version:** almost all of this already exists in the framework. This
> recipe wires it together from AgentConfig, runtime secret references, and
> `graph_configure` calls — there is no separate provisioning CLI: a Postgres
> instance is stood up by hand (docker compose, or a managed instance the operator
> already have), then registered as a mirror connection and reconciled.

External SPARQL triplestore federation (publishing/querying a triplestore like
Stardog) is owned by the epistemic-graph engine rather than this repository.
Agent Utilities does not push ontologies to an external triplestore.

---

## The loop the operator're building

```
agent-utilities graph ──promote──▶ ontology (OWL/RDF, KG-2.6)
        │                                  │
        │                                  └─ built-in /api/sparql (zero infra)
        │                                       └─ optional local Jena Fuseki
        ▼
   reconcile (KG-2.7) ──▶ Postgres / Apache AGE  (durable graph + pgvector + BM25)
```

- **Attach / query** ontology packs → EG GraphSchema + the gateway SPARQL endpoint.
- **Backfill relationships into pg-age** → the fanout backend's explicit
  `reconcile()` operation, driven through `graph_configure`.

### "Am I backfilling into pg-age today?"

Probably **not yet**. The zero-infra default is the engine alone — the one
authority (compute + cache + semantic + durable persistence), no mirrors. The operator
start projecting into AGE once the operator set `GRAPH_MIRROR_TARGETS` and `GRAPH_DB_CONNECTION_PROFILE_REF` +
`GRAPH_PG_AGE=1`. **This recipe flips that on.**

---

## Step 0 — Runtime connection references

Agent Utilities resolves database and TLS documents through `SecretsClient`.
Durable AgentConfig contains references only:

```json
{
  "SECRETS_BACKEND": "vault",
  "GRAPH_DB_CONNECTION_PROFILE_REF": "secret://graph/primary-mirror-profile",
  "KG_CONNECTIONS": [
    {
      "name": "pg-age-mirror",
      "backend": "age",
      "role": "mirror",
      "connection_profile_ref": "secret://graph/pg-age-mirror-profile"
    }
  ],
  "TLS_PROFILES_REF": "secret://tls/profile-catalog"
}
```

The referenced documents contain endpoints, database names, credentials, and TLS
selection and are resolved only inside the process. Do not copy them into the
repository, launcher configuration, logs, or reports. Use the
**`agent-utilities-deployment`** skill to configure the chosen secret backend, then
validate without revealing resolved material:

```bash
agent-utilities-doctor --only secrets graph_connections transport_security
```

---

## Step 1 — Postgres: AGE + pgvector + pg_search

The operator have two modes; **the operator can use both** across environments.

### Mode 1 — A Postgres this repository control (combined image)

The `services/pg-age/compose.yml` stack references a combined image with all three
extensions. The matching local build is **`docker/pg-age-full`**:

```bash
docker compose -f docker/pg-age-full.compose.yml up -d --build
```

This image preloads `shared_preload_libraries=pg_search,pg_cron,pg_stat_statements,age`
and the init SQL (`docker/pg-age-init/01-extensions.sql`) creates the `age` graph,
`vector`, and (guarded) `pg_search` extensions plus the `kg_embeddings` table.

> **Build note:** AGE and ParadeDB must agree on the Postgres *major*. The
> The Dockerfile pins `PG_MAJOR`, the ParadeDB manifest digest, and `AGE_REV`.
> Review and update those immutable inputs together when upgrading PostgreSQL.
> before building. If no compatible pair exists, run **two** Postgres instances
> (AGE+pgvector via `docker/pg-age`, ParadeDB separately) and give each its own
> DSN.

Lightweight alternative (AGE + pgvector, **no** BM25): `docker/pg-age.compose.yml`.

### Mode 2 — An existing / managed Postgres (connect-only)

If the operator can't replace the image (e.g. a managed RDS), point at it and let
`graph_configure(action="add_connection")` register it; `CREATE EXTENSION` runs
for whatever the instance permits.

`age` and `pg_search` need **superuser + `shared_preload_libraries`**; on a locked
managed instance they may be unavailable — `pgvector` usually works everywhere.

### Register the connection

```
graph_configure(action="add_connection", config_key="pg-age-mirror",
  config_value='{"backend":"age","connection_profile_ref":"secret://graph/pg-age-mirror-profile","role":"mirror"}')
```

### Verify

```
graph_configure(action="list_connections")
graph_configure(action="mirror_status")
```

---

## Step 2 — Fan the engine out into the mirror

With the connection registered (Step 1) and `GRAPH_MIRROR_TARGETS` naming it:

```
graph_configure(action="reconcile", config_key="pg-age-mirror")
```

This backfills the existing working graph into AGE. From here on, every KG
write — including each `source_sync` of LeanIX/ServiceNow — fans out into the
mirror via the durable outbox. The fan-out is **off the write-ack path**
(CONCEPT:AU-KG.backend.authority-has-already-acked): the authority commit
returns immediately and the mirror enqueue is a non-blocking hand-off to a
bounded in-memory ring that a persister thread drains into the durable outbox —
so a slow/locked mirror outbox never throttles ingestion. On a sustained burst
the ring overflows to a synchronous durable-outbox append (loud, reconcilable,
never dropped).

All of the above is reachable identically over REST (`POST /graph/configure`).

---

## Step 3 — Confirm the backfill into pg-age

```bash
# After running the graph for a while:
python -c "import json; from agent_utilities.knowledge_graph import backends as B; print(json.dumps(B.get_mirror_build_status(), indent=2))"
```

Read AGE directly to prove relationships landed:

```sql
LOAD 'age'; SET search_path = ag_catalog, "$user", public;
SELECT * FROM cypher('agent_graph', $$ MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 5 $$) AS (n agtype, r agtype, m agtype);
```

---

## Optional — publish the ontology to an external SPARQL endpoint

The operator already serve SPARQL locally — the gateway mounts `GET/POST /api/sparql`
(`SPARQLEndpoint`, KG-2.6), materialized from the operator's live graph + OWL bridge with
**zero extra infrastructure**:

```bash
curl 'http://localhost:9000/api/sparql?query=SELECT%20?s%20WHERE%20{?s%20?p%20?o}%20LIMIT%205'
```

Ontology distribution to an external triplestore is owned by the
epistemic-graph engine, not this repository.

---

## Surfaces (everything above)

| Capability | MCP | REST |
|---|---|---|
| Register a mirror / connection | `graph_configure(action=add_connection)` | `POST /graph/configure` |
| Backfill / reconcile a mirror | `graph_configure(action=reconcile)` | `POST /graph/configure` |
| Inspect mirror / connection health | `graph_configure(action=mirror_status\|list_connections)` | `POST /graph/configure` |

## Reference

- Backends & selection: [docs/architecture/graph_backends_architecture.md](../architecture/graph_backends_architecture.md)
- OWL/RDF + SPARQL: [docs/architecture/owl_rdf_layer.md](../architecture/owl_rdf_layer.md)
- KG-as-ETL hub (connectors, `graph_etl`, lineage): [docs/architecture/kg_etl_hub.md](../architecture/kg_etl_hub.md)
- Other recipes: [tiny](tiny.md) · [single-node-prod](single-node-prod.md) · [enterprise](enterprise.md)
- **Next:** [Delta-based ingestion via the backends](delta-ingestion.md) — turn the backend the operator just wired into an incremental, content-hash-deduped, background-swept ingestion store.
