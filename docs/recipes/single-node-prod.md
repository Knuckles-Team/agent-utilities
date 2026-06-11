# Recipe — Single-node prod

One host, durable, no swarm. Good for a small team or a staging box: a durable
Postgres/pggraph KG, the REST gateway, a core slice of the `*-mcp` fleet, and
optional Langfuse/OpenBao — all via `docker compose` on a single machine.

## What runs

| Component | How |
|---|---|
| agent-utilities + REST gateway (`:8100`) | container or host process (`graph-os-daemon`) |
| Knowledge graph | `tiered` with **Postgres/pggraph L2** (`GRAPH_DB_URI`) |
| Core `*-mcp` connectors | the `single-node-prod` profile from `mcp-fleet.registry.yml` (openbao, technitium, container-manager, vector, caddy, …) |
| Caddy | HTTPS reverse proxy in front of the gateway + connectors |
| OpenBao | optional secrets store |
| Langfuse | optional observability |
| Kafka / Keycloak / swarm | **not** in this tier (see [Enterprise](enterprise.md)) |

## Steps

```bash
# 1. Bring up Postgres/pggraph (durable KG L2)
docker compose -f docker/pggraph.compose.yml up -d

# 2. Start the REST gateway pointed at it
export GRAPH_BACKEND=tiered
export GRAPH_DB_URI=postgresql://agent:REDACTED@localhost:5432/agent_kg
uv run graph-os-daemon          # REST API on :8100, hosts the KG daemon

# 3. Deploy the core connector slice (single-node-prod profile)
#    Build/run each from its docker/compose.yml, or use portainer-sync-agent.
```

## `.env` (generalized)

```dotenv
GRAPH_BACKEND=tiered
GRAPH_DB_URI=postgresql://agent:REDACTED@localhost:5432/agent_kg
KG_DAEMON_ROLE=host                 # this process owns the single KG daemon

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
curl -s localhost:8100/api/graph/query -d '{"cypher":"MATCH (n) RETURN count(n)"}'
# Restart the gateway — the KG state persists (it's in Postgres now).
```

## Graduate to enterprise

Add a swarm, Keycloak SSO, the Kafka event backbone, LGTM observability, and the
full connector fleet — see [Enterprise](enterprise.md), driven by the
`day0_bootstrap_orchestrator` skill-workflow.
