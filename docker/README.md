# `docker/` — build & compose building blocks for self-deploying agent-utilities

This directory is a **toolbox of Docker images and per-tier compose files**, not a
single monolithic stack. You self-host agent-utilities by assembling **only the tiers
your deployment profile needs**. If you're unsure where to start, read the **profile
table** first, then the composition recipes.

> **TL;DR start here:** pick a profile below → follow its `docs/recipes/` guide for the
> `.env`/`config.json` → use [`recipes/README.md`](recipes/README.md) for exactly which
> compose files to bring up in what order.

## 1. Pick your profile

| Profile | What you run | Setup guide |
|---|---|---|
| **Tiny** (laptop/homelab, zero-infra) | Nothing — the KG runs **in-process**. Just `scripts/bootstrap.sh`. | [`docs/recipes/tiny.md`](../docs/recipes/tiny.md) |
| **Single-node prod** (one durable host) | `pg-age.compose.yml` + `mcp.compose.yml` + core connectors | [`docs/recipes/single-node-prod.md`](../docs/recipes/single-node-prod.md) |
| **Enterprise** (multi-host swarm, full fleet) | `pg-age` + `kafka-kraft` + `mcp` + the whole `*-mcp` fleet, via the genesis workflow | [`docs/recipes/enterprise.md`](../docs/recipes/enterprise.md) |

The **composition recipes** — which compose files to combine per profile, with copy-paste
commands — live in [`recipes/README.md`](recipes/README.md). The **narrative setup** (env
vars, `config.json`, secrets, database choice) lives in [`docs/recipes/`](../docs/recipes/).

## 2. The images

| File | Builds | Role |
|---|---|---|
| `Dockerfile` | the **agent-utilities** image (`graph-os` MCP server + KG engine + built-in MCP fleet gateway) | the one image every deployment runs |

## 3. The tiers (compose files)

**Durable KG tier — pick ONE as the authority (start with `pg-age`):**

| File | Provides | When |
|---|---|---|
| `pg-age.compose.yml` | **PostgreSQL + Apache AGE (openCypher) + pgvector** | **the default durable tier — start here** |
| `pg-age-full.compose.yml` | AGE + pgvector + **ParadeDB `pg_search` (BM25)** in one Postgres (builds `pg-age-full/`) | when you also want native full-text search ([databases.md](../docs/recipes/databases.md)) |
| `paradedb.compose.yml` | ParadeDB (pgvector + pg_search) variant | search-forward Postgres |

**The serving plane:**

| `mcp.compose.yml` | `graph-os` as a **thin FastMCP gateway** (streamable-http :8004) | the MCP tool surface every client/agent talks to |

**Optional / scale-out / contrib backends (bring up only if you need them):**

| File | Provides |
|---|---|
| `engine-shards.compose.yml` | tenant-partitioned **epistemic-graph engine shards** behind HRW routing (scale-out, see [engine_sharding.md](../docs/architecture/engine_sharding.md)) |
| `neo4j.compose.yml`, `falkordb.compose.yml` | contrib graph backends — integration tests / optional mirror tiers |
| `jena_fuseki.compose.yml` | Apache Jena Fuseki **SPARQL 1.1 / RDF** tier |
| `kafka-kraft.compose.yml`, `docker-compose.kafka.yml` | **Kafka** event backbone for the ingest queue |
| `egeria.compose.yml` | Apache **Egeria** metadata / governance / lineage system-of-record (federated into the KG) |

**Build contexts / init scripts:** `pg-age/`, `pg-age-full/`, `pg-age-init/`,
`paradedb-init/` — the Dockerfiles + init SQL the Postgres images build from.

## 4. "I just want to self-deploy" — the short path

```bash
# 1) Try it — zero infra, KG in-process
scripts/bootstrap.sh                                   # docs/recipes/tiny.md

# 2) Durable single host — bring up the KG tier, then the gateway
docker compose -f docker/pg-age.compose.yml up -d
docker compose -f docker/mcp.compose.yml up -d         # docs/recipes/single-node-prod.md

# 3) Full platform — swarm + all tiers + the *-mcp fleet
#    run the `agent-os-genesis` (alias `day0`) skill — it resolves an adaptive run plan
#    (deploy / baremetal / use-existing / skip per component) and stands the whole thing up.
#                                                       # docs/recipes/enterprise.md
```

## Related

- **Deployment recipes (compose composition):** [`recipes/README.md`](recipes/README.md)
- **Narrative guides (env/config/secrets):** [`docs/recipes/`](../docs/recipes/) — `tiny`,
  `single-node-prod`, `enterprise`, `databases`, `delta-ingestion`, `unified-feeds`,
  `unified-scheduling`
- **Per-service `*-mcp` stacks (the deployed fleet):** [`../../../services/`](../../../services/)
- **Genesis / day-0 bring-up + connector provisioning:**
  `agent_utilities/skills/workflows/agent-os-genesis/` (incl.
  `references/plane-provisioning-and-connector-auth.md` for connector auth + SSO)
