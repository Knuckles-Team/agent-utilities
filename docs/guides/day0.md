# Day-0 Deployment

How to stand up agent-utilities from nothing — its dependencies, the `graph-os`
MCP server + multiplexer, the `*-mcp` connector fleet, and the external
integrations — at the scale that fits you.

> Pick a **profile** and follow its recipe. Everything here is generalized and
> non-PII; substitute your own hosts/secrets. Profiles:
> [Tiny (all-local)](../recipes/tiny.md) · [Single-node prod](../recipes/single-node-prod.md) ·
> [Enterprise (swarm)](../recipes/enterprise.md).

## TL;DR — the fastest path (Tiny)

```bash
git clone https://github.com/Knuckles-Team/agent-utilities && cd agent-utilities
./scripts/bootstrap.sh          # venv + install + .env(GRAPH_BACKEND=tiered) + smoke test
```

That installs deps, writes a zero-infra `.env`, starts `graph-os`, and runs a
create-agent → graph_write → graph_query smoke test. No databases, no servers.

## The four steps (any profile)

### 1. Install dependencies

```bash
uv sync                          # or: pip install -e ".[all]"
```

The optional-dependency matrix (`pyproject.toml`) lets you install only what a
profile needs (`agent`, `mcp`, `backends`, `providers`, `embeddings`, `all`).
See [Installation](installation.md).

### 2. Run graph-os (MCP) + the multiplexer

```bash
# graph-os — the KG MCP server (stdio for IDEs, streamable-http for containers)
uv run graph-os                                            # stdio
uv run graph-os --transport streamable-http --host 0.0.0.0 --port 8004

# Or the REST gateway (one shared KG host, :8100) for UIs/scripts/fleet supervisor
uv run graph-os-daemon

# The multiplexer federates graph-os + the whole *-mcp fleet into one endpoint
uv run mcp-multiplexer --config ./mcp_config.json --transport stdio
```

See [Consumption Models](consumption-models.md) for which to choose.

### 3. Deploy the `*-mcp` connector fleet (Portainer)

The connector fleet (~50 services — ServiceNow, ERPNext, GitLab, OpenBao,
Keycloak, Technitium, Kafka, …) is described by the generated
**`deploy/mcp-fleet.registry.yml`** (regenerate with
`python scripts/gen_mcp_fleet_registry.py --agents-dir <…>/agents --out deploy/mcp-fleet.registry.yml`).

Each connector ships `docker/Dockerfile` + `docker/compose.yml` running its MCP
server over **streamable-http** (container port `8000`). Deploy them as
per-service Portainer stacks via the **`portainer-sync-agent`** skill, which
binds each stack to its Git repo for GitOps auto-sync. Which services run depends
on the profile (the `profiles:` field in the registry).

**Service + host-port table** is generated into the registry; a sample:

| Service | Console script | Container port | Host port | Profiles |
|---|---|---|---|---|
| `openbao-mcp` | `openbao-mcp` | 8000 | 8200+ | single-node-prod, enterprise |
| `technitium-dns-mcp` | `technitium-dns-mcp` | 8000 | 8200+ | single-node-prod, enterprise |
| `servicenow-mcp` | `servicenow-mcp` | 8000 | 8200+ | enterprise |
| … (52 total) | | | | |

> `genius-agent` is intentionally **not** in the fleet — it's a standalone agent
> app, not an MCP connector.

### 4. Wire integrations (à-la-carte)

Each integration is a single config switch — set only the ones your profile uses:

| Integration | Switch | Profile |
|---|---|---|
| Durable KG (Postgres/pggraph) | `GRAPH_DB_URI=postgresql://…` | single-node-prod, enterprise |
| Event backbone (Kafka) | `QUEUE_BACKEND=kafka` + `KAFKA_BOOTSTRAP_SERVERS=…` | enterprise |
| Secrets (OpenBao/Vault) | `SECRETS_VAULT_URL=…` + `VAULT_AUTH_METHOD=…` | single-node-prod, enterprise |
| SSO (Keycloak/OIDC) | `AUTH_JWT_JWKS_URI=…` / `OIDC_CONFIG_URL=…` | enterprise |
| Observability (Langfuse) | `LANGFUSE_HOST` + `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` | any (optional) |
| Tracing (OTel) | `ENABLE_OTEL=true` + `OTEL_EXPORTER_OTLP_ENDPOINT=…` | any (optional) |

Backend resolution (`create_backend()`): the default `GRAPH_BACKEND=tiered` uses
the in-process `epistemic_graph` (L1) + embedded LadybugDB (L2); set `GRAPH_DB_URI`
and the L2 auto-switches to Postgres — nothing else changes.

## Automated day-0 (skill-workflow)

For a one-command bootstrap across a fleet, the universal-skills
**`day0_bootstrap_orchestrator`** workflow is profile-driven: it asks for a
profile + integration toggles, then runs ssh/swarm/vault/dns/caddy/keycloak,
deploys graph-os + the `*-mcp` fleet from `mcp-fleet.registry.yml`, and wires the
selected integrations. The **Tiny** profile collapses to `scripts/bootstrap.sh`.

## Next

- [Recipe: Tiny](../recipes/tiny.md) — laptop / edge, zero external services.
- [Recipe: Single-node prod](../recipes/single-node-prod.md) — one host, durable.
- [Recipe: Enterprise](../recipes/enterprise.md) — swarm + full integrations.
