---
name: agent-utilities-deployment
skill_type: graph
description: >-
  Interactive, use-case-driven deployment of agent-utilities. Interviews the
  operator (just testing, dev, small production, or production at scale), then
  recommends and generates a complete deployment — knowledge-graph backend
  (memory / epistemic_graph / fanout + PostgreSQL mirror), run target (uvx, Docker,
  Kubernetes), the XDG config.json, and a backend .env. Run the wizard
  (scripts/deploy_wizard.py) or conduct the interview yourself using the matrix
  below. Triggers on "deploy agent-utilities", "how do I run agent-utilities",
  "set up agent-utilities for production".
tags: [deployment, interactive, wizard, mcp, ladybugdb, postgres, docker, kubernetes, uvx, gateway, production]
concept: OS-5.x
---

# Deploying agent-utilities (interactive)

This skill walks an operator through deploying agent-utilities **based on their
use case**. It is interactive: ask the questions, recommend per the matrix, and
generate the artifacts. Canonical reference: `docs/guides/deployment.md`.

## ▶️ Fast path — run the wizard

The bundled wizard does the whole interview and writes the artifacts. It is
standard-library only (no install needed) and **dry-runs by default**:

```bash
# From this skill directory:
python3 scripts/deploy_wizard.py                 # fully interactive
python3 scripts/deploy_wizard.py --use-case dev  # preset, still confirms
python3 scripts/deploy_wizard.py --use-case prod-scale --apply --output-dir ./deploy

# CI / scripted (accept all recommendations):
python3 scripts/deploy_wizard.py --use-case test --non-interactive --emit uvx
```

It asks: **use case → backend → deploy target → server/access → secrets,
messaging & observability → capacity → models**, then emits the XDG
`~/.config/agent-utilities/config.json`, a `deploy.env`, and the run artifacts
(`uvx` commands, a `docker-compose.override.yml`, or `k8s/agent-utilities.yaml`).
It warns when an `APP_PROFILE=production` choice would be rejected by the profile
guard.

## 🗣 Or conduct the interview yourself

When acting as an agent, ask the user these questions in order and apply the
recommendations. Always confirm before writing files.

### Step 1 — Use case (drives every default)

| Tier | When | Backend | Deploy | Notes |
|------|------|---------|--------|-------|
| **test** | CI, throwaway, smoke | `memory` (ephemeral, no disk) | `uvx` | no UI/auth/infra |
| **dev** | local dev, one user, persistent | `epistemic_graph` (the engine alone — durable, redb-authoritative, no server) | `uvx` or `docker` | UI optional, sqlite secrets |
| **prod-small** | a team / single node | `fanout` + a **PostgreSQL** mirror | `docker` | auth on, Vault, NATS, OTel |
| **prod-scale** | thousands of users, multi-node, HA | `fanout` + pooled **PostgreSQL/pg-age** mirror | `kubernetes` | OIDC, Vault, Kafka, `APP_PROFILE=production` |

### Step 2 — Backend (`GRAPH_PERSISTENCE_TYPE` + `GRAPH_MIRROR_TARGETS`)

The **epistemic-graph engine is the ONE authority** — it serves reads, acks
writes, AND persists durably (redb-authoritative by default, CONCEPT:AU-KG.backend.backend-modes):
it is a durable source of truth out of the box, not a rebuildable cache. There is
**no L1/L2 tier vocabulary** and no manual backend-mode switch — the engine is
always constructed as the authority; naming a mirror is what turns fan-out on:

- `GRAPH_PERSISTENCE_TYPE` — the engine's own durable-store format.
  `file` (default, redb-authoritative) or `sqlite` are single-host and
  non-production; `postgresql` is the production-grade choice. On first boot
  with a persist dir it runs a one-time `.mp`→redb migration (see the engine
  binary-promotion runbook).
- `GRAPH_MIRROR_TARGETS` — optional, comma-separated mirror names
  (interop/BI/DR). Naming one or more **automatically** enables lossless
  fan-out; leaving it unset means the engine alone serves reads/writes:
  - `age`/`postgresql` — durable, queryable Postgres/pg-age mirror; ask for
    `GRAPH_DB_URI`.
  - `ladybug`/`neo4j`/`falkordb` — other mirror targets.

### Step 3 — Deploy target

- **uvx / uv** — fastest, ephemeral; great for test/dev.
  `uvx --from 'agent-utilities[mcp]' graph-os --transport stdio`
  `uv run --with 'agent-utilities[all]' python -m agent_utilities` (full server)
- **Docker Compose** — single node, durable. Compose files in `docker/`
  (`mcp.compose.yml`, `pggraph.compose.yml`, `neo4j`/`falkordb`, `kafka-kraft`).
- **Kubernetes** — HA/multi-node. The wizard generates a Namespace + Deployment
  (with `/health` readiness) + Service; scale replicas and add an Ingress/HPA.

### Step 4 — config.json items (XDG `~/.config/agent-utilities/config.json`)

Ask and recommend per tier: `host`/`port`, `enable_web_ui`, `AUTH_TYPE`
(`none`/`static`/`jwt`/`oauth-proxy`/`oidc-proxy`/`remote-oauth`), `secrets_backend`
(`engine`/`vault` + `vault_url`), `a2a_broker`/`a2a_storage` (+ `kafka_bootstrap_servers`),
`enable_otel` (+ OTLP endpoint), `max_concurrent_agents`, and the model gateway
(`llm_base_url`, `model_id`). Backend selection is **also** written to `deploy.env`
because env vars are authoritative for backend resolution.

### Step 5 — Production safety

If `APP_PROFILE=production`, the profile guard (`core/profile_guard`) **rejects**
ephemeral single-host defaults. Require: `GRAPH_PERSISTENCE_TYPE=postgresql`
(`file`/`sqlite` are single-host and rejected) or a `GRAPH_MIRROR_TARGETS`
Postgres mirror (`GRAPH_DB_URI`), `a2a_broker='epistemic_graph'`, and
`a2a_storage='epistemic_graph'` (the native durable broker/CAS-fenced
records — `kafka_bootstrap_servers` configures the underlying transport). The
wizard prints exactly which choices would be rejected before you apply.

## ✅ Verify after deploy

```bash
python -c "from agent_utilities.knowledge_graph.backends import create_backend as c; \
b=c(); print(type(b).__name__)"   # EpistemicGraphBackend (default) or FanOutBackend
graph-os --help            # standard --transport/--host/--port
curl -s localhost:8004/health
curl -s -XPOST localhost:9000/api/graph/query -d '{"cypher":"MATCH (n) RETURN count(n)"}'
```

See also: [Deployment Guide](../../../../../docs/guides/deployment.md)
· [Configuration](../../../../../docs/guides/configuration.md).
