# Recipe: Stardog + pg-age database environments

A beginner's, copy-paste guide to standing up the two database environments
agent-utilities is designed around — **prod** (push your ontology to **Stardog**,
host it over SPARQL, consume it back) and **dev** (host SPARQL **locally**, no
Stardog) — with a durable **Postgres** carrying Apache AGE + pgvector + ParadeDB
(`pg_search`) so graph relationships are backfilled into AGE.

> **The short version:** almost all of this already exists in the framework. This
> recipe wires it together from AgentConfig and runtime secret references, in one command:
> `setup-databases` (CLI), the `graph_configure` MCP action `setup_databases`, or
> the `database-environment-setup` skill.

---

## The loop you're building

```
agent-utilities graph ──promote──▶ ontology (OWL/RDF, KG-2.6)
        │                                  │
        │                                  ├─ prod ─▶ Stardog ──SPARQL──▶ your system
        │                                  └─ dev  ─▶ built-in /api/sparql (zero infra)
        │                                                    └─ optional local Jena Fuseki
        ▼
   reconcile (KG-2.7) ──▶ Postgres / Apache AGE  (durable graph + pgvector + BM25)
```

- **Push / host / consume** the ontology → `OntologyPublisher` +
  the gateway SPARQL endpoint.
- **Backfill relationships into pg-age** → the fanout backend's explicit
  `reconcile()` operation.

### "Am I backfilling into pg-age today?"

Probably **not yet**. The zero-infra default is the engine alone — the one
authority (compute + cache + semantic + durable persistence), no mirrors. You
start projecting into AGE once you set `GRAPH_MIRROR_TARGETS` and `GRAPH_DB_CONNECTION_PROFILE_REF` +
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
      "name": "reasoning-store",
      "backend": "stardog",
      "role": "mirror",
      "connection_profile_ref": "secret://graph/reasoning-store-profile"
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

You have two modes; **you can use both** across environments.

### Mode 1 — A Postgres we control (combined image)

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
> DSN; the provisioner supports that.

Lightweight alternative (AGE + pgvector, **no** BM25): `docker/pg-age.compose.yml`.

### Mode 2 — An existing / managed Postgres (connect-only)

If you can't replace the image (e.g. a managed RDS), point at it and let the
provisioner `CREATE EXTENSION` what's permitted:

```bash
setup-databases --verify --connection-profile-ref "$GRAPH_DB_CONNECTION_PROFILE_REF"
```

`age` and `pg_search` need **superuser + `shared_preload_libraries`**; on a locked
managed instance they may be unavailable. The verifier reports exactly which are
missing instead of failing silently — `pgvector` usually works everywhere.

### Verify

```bash
setup-databases --verify --connection-profile-ref "$GRAPH_DB_CONNECTION_PROFILE_REF"
# → {"status":"success","extensions":{"age":true,"vector":true,"pg_search":true},"ready":true}
```

---

## Step 2 — Prod recipe (Stardog)

With `STARDOG_*` set (Step 0) and Postgres up (Step 1):

```bash
setup-databases --profile prod --postgres-mode managed_image \
  --connection-profile-ref "$GRAPH_DB_CONNECTION_PROFILE_REF"
```

This (1) verifies Postgres, (2) wires `GRAPH_DB_CONNECTION_PROFILE_REF` +
`GRAPH_PG_AGE=1` + `GRAPH_MIRROR_TARGETS` so the engine authority fans writes
out into the AGE projection,
(3) **pushes the bundled ontology to Stardog**
(`OntologyPublisher.push_to_stardog`), (3b) **registers Stardog as a live data
mirror** so instance data replicates continuously (see Step 2b), (4) reconciles the
working graph into AGE *and* backfills the Stardog mirror, and (5) smoke-tests a
SPARQL `SELECT` against Stardog.

**Consume it** from your system against Stardog's SPARQL endpoint
through the configured Stardog connection profile — reasoning included, since the
Stardog OWL backend answers queries with inference on.

---

## Step 2b — Populate Stardog with your DATA (not just the ontology)

Pushing the ontology (Step 2) loads the **TBox** (schema). To also get your
**instance data** — the LeanIX fact sheets, ServiceNow TRM requests, etc. that land
in the KG as nodes/edges — Stardog is a first-class **SPARQL data backend**
(`StardogSparqlBackend`, distinct from the OWL *reasoning* backend). Data is
partitioned into `urn:source:<system>` **named graphs** so each source is a slice
you can push, query, or re-ingest on its own.

**Continuous (live mirror).** `setup-databases --profile prod` registers Stardog as
a `role="mirror"` connection by default, so every KG
write — including each `source_sync` of LeanIX/ServiceNow — fans out into Stardog
via the durable outbox. The fan-out is **off the write-ack path** (CONCEPT:AU-KG.backend.authority-has-already-acked):
the authority commit returns immediately and the mirror enqueue is a non-blocking
hand-off to a bounded in-memory ring that a persister thread drains into the durable
outbox — so a slow/locked mirror outbox never throttles ingestion. On a sustained
burst the ring overflows to a synchronous durable-outbox append (loud, reconcilable,
never dropped). Backfill what's already there with `reconcile` (or `setup`'s
Step 4). Opt out with `--no-mirror-data` if you only want the ontology.

```bash
# Register the mirror by hand (idempotent) + backfill the existing graph:
python -c "import json; from agent_utilities.knowledge_graph.setup import register_stardog_mirror; print(json.dumps(register_stardog_mirror(), indent=2))"
```

**On-demand (explicit push / pull / query)** via `graph_configure`:

```jsonc
// Push a subset — only LeanIX + ServiceNow — into their named graphs:
{"action":"push_to_stardog","config_value":"{\"sources\":[\"leanix\",\"servicenow\"]}"}

// Query one source's slice (SELECT/ASK/CONSTRUCT/UPDATE):
{"action":"stardog_sparql","config_value":"{\"query\":\"SELECT ?s ?p ?o WHERE { GRAPH <urn:source:servicenow> { ?s ?p ?o } } LIMIT 25\"}"}

// Pull a named graph back into the KG:
{"action":"pull_from_stardog","config_value":"{\"source\":\"leanix\"}"}
```

All three are reachable identically over REST (`POST /graph/configure`).

---

## Step 3 — Dev recipe (no Stardog)

You already serve SPARQL locally — the gateway mounts `GET/POST /api/sparql`
(`SPARQLEndpoint`, KG-2.6), materialized from your live graph + OWL bridge with
**zero extra infrastructure**.

```bash
setup-databases --profile dev --postgres-mode managed_image \
  --connection-profile-ref "$GRAPH_DB_CONNECTION_PROFILE_REF"
# consume at:  curl 'http://localhost:9000/api/sparql?query=SELECT%20?s%20WHERE%20{?s%20?p%20?o}%20LIMIT%205'
```

**Optional upgrade — local Jena Fuseki** (full SPARQL 1.1 parity with prod):

```bash
docker compose -f docker/jena_fuseki.compose.yml up -d
setup-databases --profile dev --sparql-target fuseki --connection-profile-ref "$GRAPH_DB_CONNECTION_PROFILE_REF"
```

---

## Step 4 — Confirm the backfill into pg-age

```bash
# After running the graph for a while:
python -c "import json; from agent_utilities.knowledge_graph.setup import backfill_to_age; print(json.dumps(backfill_to_age(), indent=2))"
# → {"status":"success","reconcile":{"nodes":N,"edges":M,"nodes_missing":0,...},"consistent":true}
```

Read AGE directly to prove relationships landed:

```sql
LOAD 'age'; SET search_path = ag_catalog, "$user", public;
SELECT * FROM cypher('agent_graph', $$ MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 5 $$) AS (n agtype, r agtype, m agtype);
```

---

## Surfaces (everything above, three ways)

| Surface | How |
|---|---|
| **CLI** | `setup-databases --profile {dev,prod} --postgres-mode {managed_image,existing} [--connection-profile-ref ...] [--verify]` |
| **MCP** | `graph_configure(action="setup_databases", config_key="prod", config_value='{"postgres_mode":"managed_image","connection_profile_ref":"secret://graph-connections/primary"}')`; `action="verify_databases"` |
| **REST** | `POST /graph/configure` with `{"action":"setup_databases","config_key":"prod","config_value":"{...}"}` |
| **Skill** | `database-environment-setup` (selects a deployment mode and resolves AgentConfig connection references) |

## Reference

- Backends & selection: [docs/architecture/graph_backends_architecture.md](../architecture/graph_backends_architecture.md)
- OWL/RDF + SPARQL: [docs/architecture/owl_rdf_layer.md](../architecture/owl_rdf_layer.md)
- KG-as-ETL hub (Stardog data backend, `graph_etl`, lineage): [docs/architecture/kg_etl_hub.md](../architecture/kg_etl_hub.md)
- Other recipes: [tiny](tiny.md) · [single-node-prod](single-node-prod.md) · [enterprise](enterprise.md)
- **Next:** [Delta-based ingestion via the backends](delta-ingestion.md) — turn the backend you just wired into an incremental, content-hash-deduped, background-swept ingestion store.
