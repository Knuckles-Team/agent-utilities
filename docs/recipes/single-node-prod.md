# Recipe — Single-node prod

> Ladder position: this recipe combines **rung (b) — Secured single node** and
> **rung (c) — Durable single node** of the
> [supported deployment configurations](../guides/deployment-configurations.md#rung-b-secured-single-node)
> guide. The ladder has the complete `.env` for each rung (JWT identity,
> brain enforcement, `STATE_DB_URI` state externalization) — this page is the
> docker-compose walkthrough.

One host, durable, no swarm. Good for a small team or a staging box: a durable
Postgres/pg-age KG, the REST gateway, a core slice of the `*-mcp` fleet, and
optional Langfuse/OpenBao — all via `docker compose` on a single machine.

## What runs

| Component | How |
|---|---|
| REST gateway (`python -m agent_utilities`, `:9000` via `HOST`/`PORT`) | container or host process; hosts the KG daemon (flock-elected) and serves `/api/graph/*`, `/api/fleet/*`, `/metrics` |
| KG host daemon | inside the gateway process; headless alternative: `graph-os-daemon` (no HTTP) |
| Knowledge graph | `tiered` (or `fanout`) with a durable **Postgres tier** (`GRAPH_DB_URI`) — the Postgres image **must** carry **Apache AGE + pgvector + ParadeDB** (see note below) |
| Durable platform state | sessions/goals/checkpoints/task queue on the same Postgres (`STATE_DB_URI`) |
| Core `*-mcp` connectors | the `single-node-prod` profile from `mcp-fleet.registry.yml` (openbao, technitium, container-manager, vector, caddy, …) |
| Caddy | HTTPS reverse proxy in front of the gateway + connectors |
| OpenBao | optional secrets store |
| Langfuse | optional observability |
| Kafka / Keycloak / swarm | **not** in this tier (see [Enterprise](enterprise.md)) |

> **Postgres extension requirement.** The durable tier must be a Postgres that
> carries **Apache AGE** (`age`, native openCypher — the `backend: "age"` path),
> **pgvector** (`vector`), and **ParadeDB** (`pg_search`), with `age` and
> `pg_search` in `shared_preload_libraries`. The curated `registry.arpa/pg-age`
> image (`services/pg-age/`, built `FROM paradedb/paradedb` PG18 + AGE 1.7.0)
> bundles all three. The **stock `paradedb/paradedb` image has pgvector +
> pg_search but NOT AGE** — using it leaves Postgres on the bounded regex
> transpiler (`cypher_support="subset"`). See
> [Graph Backend Architecture → Extension Dependencies](../architecture/graph_backends_architecture.md#extension-dependencies).

## Steps

```bash
# 1. Bring up Postgres/pg-age (publishes host port 5433, db agent_kg,
#    user/password agent/agent)
docker compose -f docker/pg-age.compose.yml up -d
docker exec agent-pg-age psql -U agent -d agent_kg -c 'CREATE DATABASE agent_state'

# 2. Start the REST gateway pointed at it (also hosts the KG daemon)
export GRAPH_BACKEND=tiered
export GRAPH_DB_URI=postgresql://agent:agent@localhost:5433/agent_kg
export STATE_DB_URI=postgresql://agent:agent@localhost:5433/agent_state
python -m agent_utilities       # REST API on :9000 (HOST/PORT)
# Headless alternative (no REST surface): uv run graph-os-daemon

# 3. Deploy the core connector slice (single-node-prod profile)
#    Build/run each from its docker/compose.yml, or use portainer-sync-agent.
```

## `.env` (generalized)

```dotenv
GRAPH_BACKEND=tiered
GRAPH_DB_URI=postgresql://agent:REDACTED@localhost:5433/agent_kg

# Durable platform state: sessions/goals, durable-exec checkpoints, and the
# KG task queue all move onto Postgres; the task queue auto-resolves to
# `postgres` when this is set (rung c of the ladder)
STATE_DB_URI=postgresql://agent:REDACTED@localhost:5433/agent_state

KG_DAEMON_ROLE=host                 # this process owns the single KG daemon

# Identity & enforcement (rung b of the ladder — see the guide for details)
KG_AUTH_REQUIRED=1
AUTH_JWT_JWKS_URI=https://idp.example.internal/realms/agents/protocol/openid-connect/certs
AUTH_JWT_ISSUER=https://idp.example.internal/realms/agents
KG_BRAIN_ENFORCE=1

# Optional — secrets
SECRETS_VAULT_URL=http://localhost:8200
VAULT_AUTH_METHOD=token

# Optional — observability
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=lf_pk_REDACTED
LANGFUSE_SECRET_KEY=lf_sk_REDACTED

# Model provider
OPENAI_API_KEY=sk-REDACTED
```

## Connectors

Deploy the connectors tagged `single-node-prod` in
`deploy/mcp-fleet.registry.yml`. Each maps its container port `8000` to a unique
host port (`8200+`) so they coexist on one machine. Front them with Caddy
(`docker/caddy` / the `caddy-mcp` connector) for TLS + routing.

## Verify

```bash
curl -s -X POST localhost:9000/api/graph/query \
  -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"cypher":"MATCH (n) RETURN count(n) AS n"}'
# Restart the gateway — KG state, sessions, and goals persist (Postgres now).
# Without a Bearer token the same call returns 401 (KG_AUTH_REQUIRED=1).
```

## Graduate to enterprise

Add a swarm, Keycloak SSO, the Kafka event backbone, LGTM observability, and the
full connector fleet — see [Enterprise](enterprise.md), driven by the
`day0_bootstrap_orchestrator` skill-workflow. The flag-level path is rungs
(d) and (e) of the
[deployment configurations ladder](../guides/deployment-configurations.md#rung-d-scaled-multi-host).
