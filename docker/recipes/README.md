# Deployment Recipes — compose composition per profile

These recipes **compose the existing `docker/*.compose.yml` files** plus the
generated `deploy/mcp-fleet.registry.yml`; they don't duplicate them. See the
narrative guides in [`docs/recipes/`](../../docs/recipes/) for `.env`/`config.json`.

| Profile | What to bring up | Files |
|---|---|---|
| **Tiny** | Nothing — the KG is in-process. Just `scripts/bootstrap.sh`. | — |
| **Single-node prod** | Durable KG + gateway + core connectors on one host | `pg-age.compose.yml` + the MCP gateway (`mcp.compose.yml`) + the `single-node-prod` services from the registry |
| **Enterprise** | Full swarm + integrations + entire fleet | `pg-age.compose.yml` + `kafka-kraft.compose.yml` + `mcp.compose.yml` + the `enterprise` services from the registry, deployed via Portainer GitOps (`portainer-sync-agent`) |
| **Engine shards (Stage 2)** | N tenant-partitioned epistemic-graph engine shards behind client-side HRW routing | `engine-shards.compose.yml` (worked 3-shard example) + `GRAPH_SERVICE_ENDPOINTS` on every client — see [docs/architecture/engine_sharding.md](../../docs/architecture/engine_sharding.md) |

## Single-node prod (one-host quickstart)

```bash
# Durable KG tier
docker compose -f docker/pg-age.compose.yml up -d
# KG MCP gateway over streamable-http
GRAPH_DB_URI=postgresql://agent:REDACTED@localhost:5432/agent_kg \
  docker compose -f docker/mcp.compose.yml up -d
# Core connectors (single-node-prod profile) — per-service stacks from the registry
#   python scripts/gen_mcp_fleet_registry.py --agents-dir <…>/agents --out deploy/mcp-fleet.registry.yml
#   then deploy each service tagged `single-node-prod` via its own docker/compose.yml
```

## Enterprise

Use the `day0_bootstrap_orchestrator` skill-workflow (enterprise profile). It
provisions the swarm, core services, and binds every connector stack from the
registry to Git for Portainer GitOps auto-sync. Backend composes
(`pg-age`, `kafka-kraft`) are deployed as swarm stacks; integrations
(OpenBao/Keycloak/Langfuse) are wired via config. See
[docs/recipes/enterprise.md](../../docs/recipes/enterprise.md).
