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
| **Tiny** (laptop/homelab, zero-infra) | Nothing to compose — GraphOS supervises the packaged Rust engine as an out-of-process child over a private local transport. Just `scripts/bootstrap.sh`. | [`docs/recipes/tiny.md`](../docs/recipes/tiny.md) |
| **Single-node prod** (one durable host) | `pg-age.compose.yml` + `mcp.compose.yml` + core connectors | [`docs/recipes/single-node-prod.md`](../docs/recipes/single-node-prod.md) |
| **Enterprise** (multi-host swarm, full fleet) | `pg-age` + `kafka-kraft` + `mcp` + the whole `*-mcp` fleet, via the genesis workflow | [`docs/recipes/enterprise.md`](../docs/recipes/enterprise.md) |

The **composition recipes** — which compose files to combine per profile, with copy-paste
commands — live in [`recipes/README.md`](recipes/README.md). The **narrative setup** (env
vars, `config.json`, secrets, database choice) lives in [`docs/recipes/`](../docs/recipes/).

## 2. The images

| File | Builds | Role |
|---|---|---|
| `Dockerfile` | the **agent-utilities** image (`graph-os` MCP server + KG engine + built-in MCP fleet gateway) | the one image every deployment runs |
| `graphos-unified.Dockerfile` | the **`knucklessg1/graph-os-unified`** image — the ONE self-contained image (editable-source au + engine wheel + langfuse-agent + messaging backends) that 3 of the 4 containers in the live `platform/graph-os` Kubernetes Deployment actually run | see **§5 below** for how it's built — do NOT hand-apply a Kaniko Job for this anymore |

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
# 1) Try it — zero infra, packaged engine supervised out of process
scripts/bootstrap.sh                                   # docs/recipes/tiny.md

# 2) Durable single host — bring up the KG tier, then the gateway
docker compose -f docker/pg-age.compose.yml up -d
docker compose -f docker/mcp.compose.yml up -d         # docs/recipes/single-node-prod.md

# 3) Full platform — swarm + all tiers + the *-mcp fleet
#    run the `agent-utilities-deployment` skill — it resolves an adaptive run plan
#    (deploy / baremetal / use-existing / skip per component) and stands the whole thing up.
#                                                       # docs/recipes/enterprise.md
```

## 5. Building `knucklessg1/graph-os-unified` — use the real CI pipeline, not a hand-applied Job

**Retired 2026-08-02 (closes D-IMG-2 / D-CDX-18 / D-EIMG-5).** Until now, every rebuild of
`knucklessg1/graph-os-unified` — the image the live `platform/graph-os` Deployment actually
runs in 3 of its 4 containers — meant hand-writing yet another near-duplicate Kaniko `Job`
manifest in this directory and `kubectl apply`-ing it: `graphos-unified-kaniko-job.yaml`,
`-langfuse-`, `-skill-runtime-`, `-fastmcp4-`, and (preserved-but-rejected, never landed
here) an unsafe `-fix-0801-` variant. Four to five near-identical, hand-maintained,
node-pinned, `envsubst`-templated manifests, each hard-coding a different `hostPath` build
context and destination tag, with no CI gate proving any of them still pointed at real
source (one, `graphos-unified-kaniko-job.yaml`, had rotted to point at a pruned, dangling,
non-git worktree directory — applying it today would have silently rebuilt months-old
source). That pattern is **retired**. Those files (`graphos-unified-kaniko-job.yaml`,
`graphos-unified-langfuse-kaniko-job.yaml`, `graphos-unified-skill-runtime-kaniko-job.yaml`,
`graphos-unified-fastmcp4-kaniko-job.yaml`, `kaniko-build.env.example`) have been removed
from this directory — their institutional knowledge (context-composition tricks, the
`.dockerignore` `docker/*` gotcha, why caching must stay on, why the eg wheel/langfuse-agent
mounts exist) is preserved in `git log`/`git show` on this path and, more importantly, is
now encoded as **comments in the pipeline that replaced them**.

**The real, parameterized, digest-pinned, checksum-verified GitLab CI pipeline now lives
at `homelab/containers/images/graph-os-unified`** on the internal GitLab instance
(its own small repo — agent-utilities has no GitLab remote of its own, only a GitHub
mirror, so a pipeline defined *inside* this repo could never be triggered on the internal
GitLab instance; see that repo's `.gitlab-ci.yml` header for the full rationale). It takes
build context (an agent-utilities commit) and destination tag as **pipeline
inputs/variables** — never a hard-coded host path — fetches the eg-wheel/langfuse-agent
build artifacts with explicit SHA256 / pinned-commit verification, runs kaniko pinned by
digest with a bounded `timeout:`, and always tags the result by the commit it was built
from (`knucklessg1/graph-os-unified:pipeline-validation-<au-shortsha>-<pipeline-id>` for a
validation run — never `:latest` or any tag the live Deployment references; that Deployment
is only ever moved forward by an explicit, reviewed `kubectl diff` + `apply` against a
digest, same as always).

If you need to rebuild this image: trigger that pipeline (push or **Run pipeline** in the
GitLab UI/API), not a new hand-written Job manifest.

## Related

- **Deployment recipes (compose composition):** [`recipes/README.md`](recipes/README.md)
- **Narrative guides (env/config/secrets):** [`docs/recipes/`](../docs/recipes/) — `tiny`,
  `single-node-prod`, `enterprise`, `databases`, `delta-ingestion`, `unified-feeds`,
  `unified-scheduling`
- **Per-service `*-mcp` stacks (the deployed fleet):** [`../../../services/`](../../../services/)
- **Genesis / day-0 bring-up + connector provisioning:** the
  `agent-utilities-deployment` pre-bundled workflow skill and `genesis.yaml`.
