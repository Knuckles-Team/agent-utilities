# Recipe: Stardog + pg-age database environments

A beginner's, copy-paste guide to standing up the two database environments
agent-utilities is designed around — **prod** (push your ontology to **Stardog**,
host it over SPARQL, consume it back) and **dev** (host SPARQL **locally**, no
Stardog) — with a durable **Postgres** carrying Apache AGE + pgvector + ParadeDB
(`pg_search`) so graph relationships are backfilled into AGE.

> **The short version:** almost all of this already exists in the framework. This
> recipe wires it together from your `.env` or OpenBao/Vault, in one command:
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
- **Backfill relationships into pg-age** → the fanout backend's
  `reconcile_to_durable()`.

### "Am I backfilling into pg-age today?"

Probably **not yet**. The zero-infra default is `GRAPH_BACKEND=epistemic_graph` —
the engine is the one authority (compute + cache + semantic + durable
persistence), no mirrors. You start mirroring into AGE once you set
`GRAPH_BACKEND=fanout` + `GRAPH_MIRROR_TARGETS` and `GRAPH_DB_URI` +
`GRAPH_PG_AGE=1`. **This recipe flips that on.**

---

## Step 0 — Credentials (OpenBao/Vault, `.env` fallback)

agent-utilities resolves secrets through one `SecretsClient`; nothing here reads
raw environment variables directly. Pick a source:

**A. OpenBao / HashiCorp Vault (recommended).** Point the secrets backend at your
vault and store DSNs/credentials as KV entries; reference them with `vault://`:

```bash
# config.json or .env
SECRETS_BACKEND=vault
SECRETS_VAULT_URL=https://vault.your.domain:8200
SECRETS_VAULT_MOUNT=secret
VAULT_AUTH_METHOD=approle        # or token / oidc / kubernetes
VAULT_ROLE=agent-utilities
VAULT_PATH_PREFIX=agents/db/

# Then values can be vault refs, resolved by SecretsClient:
GRAPH_DB_URI=vault://agents/db/pg_age#dsn
STARDOG_PASSWORD=vault://agents/db/stardog#password
```

Use the **`secret-vault-manager`** skill to unseal/seed those paths.

**B. Local `.env` (laptops / quick start).** Drop a `.env` next to where you run
the gateway:

```bash
GRAPH_DB_URI=postgresql://agent:agent@localhost:5432/agent_kg
STARDOG_ENDPOINT=http://localhost:5820
STARDOG_DATABASE=agent_kg
STARDOG_USER=admin
STARDOG_PASSWORD=changeme
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
> Dockerfile pins `PG_MAJOR` / `AGE_BRANCH` — verify they match your ParadeDB tag
> before building. If no compatible pair exists, run **two** Postgres instances
> (AGE+pgvector via `docker/pg-age`, ParadeDB separately) and give each its own
> DSN; the provisioner supports that.

Lightweight alternative (AGE + pgvector, **no** BM25): `docker/pg-age.compose.yml`.

### Mode 2 — An existing / managed Postgres (connect-only)

If you can't replace the image (e.g. a managed RDS), point at it and let the
provisioner `CREATE EXTENSION` what's permitted:

```bash
setup-databases --verify --dsn "$GRAPH_DB_URI"
```

`age` and `pg_search` need **superuser + `shared_preload_libraries`**; on a locked
managed instance they may be unavailable. The verifier reports exactly which are
missing instead of failing silently — `pgvector` usually works everywhere.

### Verify

```bash
setup-databases --verify --dsn postgresql://agent:agent@localhost:5432/agent_kg
# → {"status":"success","extensions":{"age":true,"vector":true,"pg_search":true},"ready":true}
```

---

## Step 2 — Prod recipe (Stardog)

With `STARDOG_*` set (Step 0) and Postgres up (Step 1):

```bash
setup-databases --profile prod --postgres-mode managed_image \
  --dsn "$GRAPH_DB_URI"
```

This (1) verifies Postgres, (2) wires `GRAPH_DB_URI`+`GRAPH_PG_AGE=1`+`GRAPH_BACKEND=fanout`
(+`GRAPH_MIRROR_TARGETS`) so the engine authority fans writes out into the AGE mirror,
(3) **pushes the bundled ontology to Stardog**
(`OntologyPublisher.push_to_stardog`), (3b) **registers Stardog as a live data
mirror** so instance data replicates continuously (see Step 2b), (4) reconciles the
working graph into AGE *and* backfills the Stardog mirror, and (5) smoke-tests a
SPARQL `SELECT` against Stardog.

**Consume it** from your system against Stardog's SPARQL endpoint
(`$STARDOG_ENDPOINT/$STARDOG_DATABASE/query`) — reasoning included, since the
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
a `role="mirror"` connection by default, so under `GRAPH_BACKEND=fanout` every KG
write — including each `source_sync` of LeanIX/ServiceNow — fans out into Stardog
via the durable outbox. Backfill what's already there with `reconcile` (or `setup`'s
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
  --dsn "$GRAPH_DB_URI"
# consume at:  curl 'http://localhost:9000/api/sparql?query=SELECT%20?s%20WHERE%20{?s%20?p%20?o}%20LIMIT%205'
```

**Optional upgrade — local Jena Fuseki** (full SPARQL 1.1 parity with prod):

```bash
docker compose -f docker/jena_fuseki.compose.yml up -d
setup-databases --profile dev --sparql-target fuseki --dsn "$GRAPH_DB_URI"
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
| **CLI** | `setup-databases --profile {dev,prod} --postgres-mode {managed_image,existing} [--dsn ...] [--verify]` |
| **MCP** | `graph_configure(action="setup_databases", config_key="prod", config_value='{"postgres_mode":"managed_image","dsn":"..."}')`; `action="verify_databases"` |
| **REST** | `POST /graph/configure` with `{"action":"setup_databases","config_key":"prod","config_value":"{...}"}` |
| **Skill** | `database-environment-setup` (prompts for env + Postgres mode, resolves OpenBao/`.env`) |

## Reference

- Backends & selection: [docs/architecture/graph_backends_architecture.md](../architecture/graph_backends_architecture.md)
- OWL/RDF + SPARQL: [docs/architecture/owl_rdf_layer.md](../architecture/owl_rdf_layer.md)
- KG-as-ETL hub (Stardog data backend, `graph_etl`, lineage): [docs/architecture/kg_etl_hub.md](../architecture/kg_etl_hub.md)
- Other recipes: [tiny](tiny.md) · [single-node-prod](single-node-prod.md) · [enterprise](enterprise.md)
- **Next:** [Delta-based ingestion via the backends](delta-ingestion.md) — turn the backend you just wired into an incremental, content-hash-deduped, background-swept ingestion store.
