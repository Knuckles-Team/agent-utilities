# Recipe — Single-node prod

> Ladder position: this recipe combines **rung (b) — Secured single node** and
> **rung (c) — Durable single node** of the
> [supported deployment configurations](../guides/deployment-configurations.md#rung-b-secured-single-node)
> guide. The ladder has the complete AgentConfig projection for each rung (JWT identity,
> brain enforcement, `STATE_DB_URI` state externalization) — this page is the
> docker-compose walkthrough.

One host, durable, no swarm. Good for a small team or a staging box: the
epistemic-graph engine running as **its own container** (still the one
authority — compute, cache, ontology, and durable persistence in a single
engine), the REST gateway, a core slice of the `*-mcp` fleet, optional mirror
databases, and optional Langfuse/OpenBao — all via `docker compose` on a single
machine. The difference from [tiny](tiny.md) is purely lifecycle: instead of an
embedded child that dies with one agent process, the engine runs as a
long-lived container that the gateway and connectors share.

## What runs

| Component | How |
|---|---|
| **epistemic-graph engine** | **its own container** — the one durable authority; the gateway/connectors connect to it (local socket or `GRAPH_SERVICE_ENDPOINTS`) |
| REST gateway (`python -m agent_utilities`, `:9000` via `HOST`/`PORT`) | container or host process; hosts the KG daemon (flock-elected) and serves `/api/graph/*`, `/api/fleet/*`, `/metrics` |
| KG host daemon | inside the gateway process; headless alternative: `graph-os-daemon` (no HTTP) |
| Knowledge graph | the engine authority is durable on its own. **Optional**: fan out write-only to a Postgres/pg-age **mirror** (`GRAPH_DB_CONNECTION_PROFILE_REF`) for SQL-side querying/BI — the Postgres image **must** carry **Apache AGE + pgvector + ParadeDB** (see note below) |
| Shared support state | sessions/turns/fleet metadata and queue delivery on optional Postgres (`STATE_DB_URI`); execution checkpoints remain on native WorkItems |
| Core `*-mcp` connectors | the `single-node-prod` profile from `mcp-fleet.registry.yml` (openbao, technitium, container-manager, vector, caddy, …) |
| Caddy | HTTPS reverse proxy in front of the gateway + connectors |
| OpenBao | optional secrets store |
| Langfuse | optional observability |
| Kafka / Keycloak / an orchestrator (Kubernetes or Swarm) | **not** in this tier (see [Enterprise](enterprise.md)) |

> **Postgres mirror extension requirement.** Postgres here is an **optional
> write-only mirror** of the engine authority (not the system of record). If you
> enable it, the Postgres must carry **Apache AGE** (`age`, native openCypher —
> the `backend: "age"` path), **pgvector** (`vector`), and **ParadeDB**
> (`pg_search`), with `age` and `pg_search` in `shared_preload_libraries`. The
> image selected through `${PG_AGE_IMAGE}` (`services/pg-age/`, built `FROM
> paradedb/paradedb` PG18 + AGE 1.7.0) bundles all three. The **stock
> `paradedb/paradedb` image has pgvector + pg_search but NOT AGE** — using it
> leaves the mirror on the bounded regex transpiler
> (`cypher_support="subset"`). See
> [Graph Backend Architecture → Extension Dependencies](../architecture/graph_backends_architecture.md#extension-dependencies).

> **The engine is redb-authoritative by default (CONCEPT:AU-KG.backend.backend-modes).** Built with
> `--features full`, the engine persists durably to its `--persist-dir` volume and
> is a real source of truth — an acked write survives a crash, it is NOT a
> rebuildable cache. On its first authoritative boot it runs a one-time
> `.mp`→redb migration (which can take minutes on a large KG); the optional pg-age
> mirror below is for interop/BI/DR, not for durability.

## Steps

```bash
# 1. Bring up the epistemic-graph engine as its own durable container.
#    This is the one authority — redb-authoritative, persists to its
#    --persist-dir volume (an acked write survives a crash).
docker compose -f docker/epistemic-graph.compose.yml up -d
export GRAPH_SERVICE_ENDPOINTS=unix:///run/epistemic-graph/engine.sock
# For a remote engine, use tls:// plus ENGINE_TLS_PROFILE_REF instead.

# 2. (OPTIONAL) Bring up a Postgres/pg-age MIRROR for SQL-side querying
#    (publishes host port 5433, db agent_kg, user/password agent/agent).
docker compose -f docker/pg-age.compose.yml up -d
docker exec agent-pg-age psql -U agent -d agent_kg -c 'CREATE DATABASE agent_state'

# 3. Start the REST gateway pointed at the engine (also hosts the KG daemon)
export GRAPH_MIRROR_TARGETS=age                                         # fan out to the pg-age mirror
: "${GRAPH_DB_CONNECTION_PROFILE_REF:?inject graph mirror connection-profile reference}" # optional mirror
export STATE_DB_URI="${STATE_DB_URI:?inject durable state DSN reference}"   # optional
python -m agent_utilities       # REST API on :9000 (HOST/PORT)
# Headless alternative (no REST surface): graph-os-daemon

# 4. Deploy the core connector slice (single-node-prod profile)
#    Build/run each from its docker/compose.yml, or use portainer-sync-agent.
```

## AgentConfig projection (generalized)

The `KEY=value` notation is documentation shorthand. Store non-secret settings
under their aliases in XDG `config.json`; keep concrete values in runtime
injection or the fixed, private, referenced XDG secret source.

```text
GRAPH_MIRROR_TARGETS=age   # fan out to the pg-age mirror; drop both for engine-only

# The engine authority — its own container, shared by the gateway + connectors.
GRAPH_SERVICE_ENDPOINTS=unix:///run/epistemic-graph/engine.sock

# OPTIONAL — write-only Postgres/pg-age mirror of the engine (SQL-side querying);
# omit it (and GRAPH_MIRROR_TARGETS) for an engine-only single node.
GRAPH_DB_CONNECTION_PROFILE_REF=vault://platform/graph#profile

# Durable platform state: sessions/goals, durable-exec checkpoints, and the
# KG task queue can move onto Postgres; the task queue auto-resolves to
# `postgres` when this is set (rung c of the ladder)
STATE_DB_URI=vault://platform/state#profile

KG_DAEMON_ROLE=host                 # this process owns the single KG daemon

# Identity & enforcement (rung b of the ladder — see the guide for details)
AUTH_JWT_JWKS_URI=https://identity.example.test/realms/agents/protocol/openid-connect/certs
AUTH_JWT_ISSUER=https://identity.example.test/realms/agents
AUTH_JWT_AUDIENCE=agent-services
KG_POLICY_VERSION=policy-v1
# Tenant, session, ACL, and single-client enforcement are baked in.

# Optional — secrets
SECRETS_VAULT_URL=http://localhost:8200
VAULT_AUTH_METHOD=token

# Optional — observability
LANGFUSE_HOST=https://telemetry.example.test
LANGFUSE_PUBLIC_KEY_REF=vault://observability/langfuse-public-key
LANGFUSE_SECRET_KEY_REF=vault://observability/langfuse-secret-key
LANGFUSE_TLS_PROFILE_REF=vault://observability/langfuse-tls-profile

# Model provider
CHAT_MODELS=[{"id":"chat-model","provider":"openai","api_key_ref":"vault://platform/llm#api_key"}]
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
# Restart the gateway — KG state persists in the engine's own durable volume
# (and any optional Postgres mirror); sessions/goals persist if STATE_DB_URI set.
# Without a Bearer token the same call returns 401.
```

## Graduate to enterprise

Add an orchestrator (Kubernetes/RKE2 by default; Swarm is also selectable),
Keycloak SSO, the Kafka event backbone, LGTM observability, and the
full connector fleet — see [Enterprise](enterprise.md), driven by the
`agent-utilities-deployment` skill-workflow. The flag-level path is rungs
(d) and (e) of the
[deployment configurations ladder](../guides/deployment-configurations.md#rung-d-scaled-multi-host).
