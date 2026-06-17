# Recipe — Delta-based KG ingestion via the backends

> Goal: stand up **incremental ("delta") ingestion** for the Knowledge Graph in a
> new environment from scratch, so every connector re-ingests only what changed and
> a background daemon keeps the graph fresh. This recipe ties together the
> **backend wiring**, the **content-hash write-delta**, the **fleet sweep
> schedule**, and the **host daemon**. (CONCEPT:KG-2.9)

This page is **Claude-runnable**: hand it to Claude in a fresh environment and it
can configure delta ingestion end-to-end. It extends — not duplicates —
[Stardog + pg-age database environments](databases.md) (backend/DSN wiring),
[Graph DB Deployment & Multi-Backend](../guides/graph-db-deployment.md) (backend
selection matrix), and [Self-Setup](../guides/self-setup.md) (the config-complete
runbook). For one source's delta end-to-end see
[LeanIX integration](../guides/leanix-integration.md).

---

## What "delta ingestion" means here

Two independent layers — the second is what makes **every** connector incremental:

| Layer | Where | Effect |
|---|---|---|
| **Fetch-layer watermark** | `source_sync._DELTA_HANDLERS` (leanix, gitlab, archivebox) | Source API is queried with "changed since the last watermark" — less data pulled. |
| **Write-layer content-hash delta** | `core/materialization.write_entities` (the one writer) | Every entity is hashed; unchanged rows are **skipped before MERGE/re-reason**, even on a *full* fetch. Universal — works for all connectors and all backends. |

The write-delta needs the backend's `execute` for a one-round-trip prefetch
(`MATCH (n) WHERE n.id IN $ids RETURN n.content_hash`). `execute`/`execute_batch`
are `@abstractmethod` on `GraphBackend`, so **every backend supports the
write-delta** out of the box. It's **on by default** (`KG_WRITE_DELTA=1`).

---

## 0. Prerequisites

A backend that persists `content_hash` and answers the prefetch — i.e. any real
backend. The default `tiered` works everywhere; `neo4j` / `falkordb` / `age`
(pggraph) / `fanout` all work. (Pure SPARQL mirrors like Stardog are *publish*
targets, not the primary delta store.)

---

## 1. From scratch — the fast path (any profile)

```bash
# (a) install the framework + the graph-os MCP (one-link, or scripts/install.sh)
curl -fsSL https://knuckles-team.github.io/agent-utilities/install.sh | sh -s -- --profile tiny

# (b) generate a complete, profile-seeded config.json at ~/.config/agent-utilities/config.json
setup-config generate --profile tiny          # tiny | single-node-prod | enterprise

# (c) (prod only) provision the durable Postgres+AGE tier the delta store lives on
setup-databases --profile prod --postgres-mode managed_image --dsn "$GRAPH_DB_URI"

# (d) verify
agent-utilities-doctor --preflight --profile tiny
```

Delta ingestion is now **already active**: `KG_WRITE_DELTA` defaults on, and the
`all-sources-delta-sweep` schedule is enabled. The only remaining choice is the
backend (§2) and whether a host daemon runs the sweep (§4).

---

## 2. Choose the backend (the delta store)

Set `GRAPH_BACKEND` (env or `config.json`). All support the write-delta.

| `GRAPH_BACKEND` | Config keys | Notes |
|---|---|---|
| `tiered` *(default)* | `GRAPH_BACKEND_L1` (=`epistemic_graph`), `GRAPH_BACKEND_L2` (auto: `postgresql` if `GRAPH_DB_URI`/`PGGRAPH_DSN` set, else `ladybug`) | Zero-infra locally (L1 + embedded Ladybug); durable when an L2 DSN is set. |
| `age` / `postgresql` | `GRAPH_DB_URI`, `GRAPH_PG_AGE=1`, `GRAPH_PGGRAPH_SCHEMA` | Postgres + Apache AGE; the recommended durable prod store. |
| `neo4j` | `GRAPH_DB_URI` (`bolt://…`), `GRAPH_DB_USER`, `GRAPH_DB_PASSWORD` | |
| `falkordb` | `GRAPH_DB_HOST`, `GRAPH_DB_PORT` (6379), `GRAPH_DB_NAME` | |
| `ladybug` | `GRAPH_DB_PATH` (else XDG) | Embedded Kuzu; single-writer (host-role only). |
| `fanout` | `GRAPH_AUTHORITY` + `GRAPH_MIRROR_TARGETS` (resolved against `KG_CONNECTIONS`) | N-way mirrored writes; durable replay outbox. Delta applies on the authority. |

One-command provisioning (managed Postgres image carrying AGE + pgvector +
ParadeDB), via CLI or MCP:

```bash
setup-databases --profile prod --postgres-mode managed_image --dsn postgresql://agent@pggraph/agent_kg --verify
```
```
graph_configure(action="setup_databases", config_key="prod",
                config_value='{"postgres_mode":"managed_image","dsn":"postgresql://agent@pggraph/agent_kg"}')
graph_configure(action="verify_databases")          # probes age + vector + pg_search
```

Add extra backends (read / mirror) without re-provisioning:

```
graph_configure(action="add_connection",
  config_value='{"backend":"neo4j","uri":"bolt://neo4j:7687","user":"neo4j","password":"env://NEO4J_PASSWORD","role":"mirror"}')
```

---

## 3. The delta knobs (reference)

All are `config.json` keys / env vars (config.json is loaded into the environment
at startup, so either works). Read via the `setting()` accessor.

| Key | Default | Purpose |
|---|---|---|
| `KG_WRITE_DELTA` | `1` | Content-hash write-delta. `0` disables (full re-write every ingest). |
| `GRAPH_BACKEND` | `tiered` | The delta store (see §2). **Restart-required.** |
| `GRAPH_DB_URI` | – | Durable tier DSN (Postgres/AGE, Neo4j). **Restart-required.** |
| `KG_DAEMON_ROLE` | `auto` | `host` runs the scheduler/sweep; `client` doesn't; `auto` = host if the flock is free. **Restart-required.** |
| `KG_LOOP` | `false` | Enables the research/evolution Loop (separate from the delta sweep). |
| `KG_LOOP_INTERVAL` | `3600` | Loop cadence (seconds). |

Inspect/echo every option: `setup-config reference` or
`graph_configure(action="config_reference")`. Validate a config:
`setup-config doctor --profile <p>` or `graph_configure(action="config_doctor")`.

---

## 4. Run delta ingestion in the background (the sweep)

A single **host-role** daemon ticks the scheduler every 60s, reading
`deploy/schedules.yml`. The fleet sweep is one declarative entry (already enabled):

```yaml
- name: all-sources-delta-sweep
  cron: "*/20 * * * *"   # every 20 min — content-hash deduped
  kind: skill
  ref: all               # → sync_source(engine, "all", mode="delta") → sweep_all_sources
  action: delta
  enabled: true
```

`ref: all, action: delta` routes through the generic dispatch to
`sweep_all_sources(mode="delta")`, which fans out over the delta handlers +
configured capability sources + materialize extractors, isolating per-connector
failures (unconfigured → *skipped*, not *errored*).

Make the process the scheduler host and start the daemon:

```bash
# in config.json / .env
KG_DAEMON_ROLE=host
# then
graph-os-daemon            # the gateway-hosted daemon (flock-elected host runs the scheduler)
```

> Exactly one host runs the sweep (flock leadership). In a swarm, pin
> `KG_DAEMON_ROLE=host` to the KG node; all others run `auto`/`client` and skip it.

### Trigger a delta sync on demand

- **MCP:** `source_sync(source="all", mode="delta")` — the canonical tool; one source: `source_sync(source="leanix", mode="delta")`. (`graph_hydrate` is a back-compat alias; `graph_ingest` is for path/URL/document content.)
- **REST:** `POST /api/dashboard/hydrate/{source}` · `POST /api/dashboard/hydrate` (all) · `POST /api/dashboard/daemon/start`.

### Add a new delta-capable source

Native watermark delta requires a handler in `source_sync._DELTA_HANDLERS`
(leanix/gitlab/archivebox today). **Any other registered source still gets the
write-layer delta for free** through `write_entities` — it just fetches in full.
Give a hot source its own cadence by adding a `schedules.yml` entry
(`ref: <source>, action: delta|full|reconcile`).

---

## 5. Configure it from scratch with Claude (genesis)

Hand Claude this recipe in a new environment. The guided path:

- **tiny / single-node** → the **`agent-utilities-deployment`** skill (alias
  `self-setup`): composes `setup-config` + `setup-databases` + the
  `database-environment-setup` skill, then verifies with `agent-utilities-doctor`.
- **enterprise / multi-node** → the **`agent-os-genesis`** skill (aliases `day0`,
  `day0_bootstrap_orchestrator`), driven by the root **`genesis.yaml`** manifest.
  Its backend/config steps:
  - **A1 `agent-utilities-install`** — install; tiny writes `GRAPH_BACKEND=tiered`.
  - **A2 `graph-os-and-multiplexer`** — deploys `graph-os` pinned to the KG host
    with `KG_DAEMON_ROLE=host` and the shared `~/.config/agent-utilities/config.json`
    volume; points `GRAPH_DB_URI` at the pggraph tier for durable profiles.
  - **A4 `integrations-wiring`** — wires `pggraph` (`GRAPH_DB_URI`), Kafka,
    OpenBao, Keycloak.

Minimal genesis-aligned sequence:

```bash
scripts/install.sh --profile single-node-prod
setup-config generate --profile single-node-prod
setup-databases --profile prod --postgres-mode managed_image --dsn "$GRAPH_DB_URI" --verify
# set KG_DAEMON_ROLE=host in config.json, then:
graph-os-daemon
agent-utilities-doctor --preflight --profile single-node-prod --live
```

---

## 6. Verify the delta is working

```
# First sweep — sources sync; changed entities written.
source_sync(source="all", mode="delta")
# Re-run immediately — unchanged entities are skipped:
source_sync(source="all", mode="delta")   # each result carries "skipped_unchanged" > 0
```

- `agent-utilities-doctor` → `_check_graph_backend` health-checks the active backend.
- `graph_configure(action="system_doctor")` → holistic sweep.
- Backend reachability / the daemon role: `graph_configure(action="list_connections")`, `mirror_status`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `skipped_unchanged` always 0 on re-run | `KG_WRITE_DELTA=0`, or backend can't answer the prefetch | Set `KG_WRITE_DELTA=1`; confirm the backend persists `content_hash` (any real backend does). |
| Sweep never runs | No host daemon | Set `KG_DAEMON_ROLE=host` and run `graph-os-daemon`; confirm the flock isn't held elsewhere. |
| A source is `skipped` in the sweep | Unconfigured (no client/creds) | Add the connector's credentials; unconfigured sources are skipped, not errored. |
| Durable writes lost after restart | No L2 DSN (tiered fell back to embedded Ladybug) | Set `GRAPH_DB_URI` + `GRAPH_PG_AGE=1` (restart-required) and re-run `setup-databases`. |
