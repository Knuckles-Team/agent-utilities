# Self-Setup: deploy agent-utilities with every option considered

The one runbook to take agent-utilities from nothing to running — the path **Claude
follows to set itself up**, and the path an operator follows to stand up a host. It is
**config-complete**: a single command generates a `config.json` covering *all* ~261
options, and a `doctor` validates the deployment.

It composes existing pieces rather than duplicating them — the `setup-config` and
`setup-databases` commands, the `database-environment-setup` skill, and the
`agent-os-genesis` (alias **`day0`**) swarm bootstrap for the multi-node tier. The
guided version is the **`agent-utilities-deployment`** skill (alias `self-setup`).

## Pick a profile

| Profile | For | Externals |
|---|---|---|
| **tiny** | laptop / Claude self-setup / edge | none — the embedded epistemic-graph engine is the whole database |
| **single-node-prod** | one durable host | optional Postgres/pg-age mirror, optional OpenBao/Langfuse |
| **enterprise** | multi-node fleet | swarm, Postgres/Neo4j mirrors, Kafka, Keycloak, observability |

These are the rungs of [deployment-configurations.md](deployment-configurations.md);
the per-flag detail lives in [configuration.md](../architecture/configuration.md).

## 1. Install

```bash
pip install agent-utilities[all]      # or scripts/bootstrap.sh for the tiny profile
```

## 2. Generate the complete config (all options)

Don't hand-write `config.json`. Generate a full, profile-seeded one — every option at
a sensible default, with the handful of deployment-varying keys pre-filled:

```bash
setup-config generate --profile single-node-prod      # → ~/.config/agent-utilities/config.json
setup-config reference                                 # every option grouped by subsystem
```

Secret-like keys (API keys, passwords, tokens) are blanked — fill them via env or
`vault://` refs, never in the committed file. Equivalent MCP/REST surface:
`graph_configure(action="generate_config", config_key="single-node-prod")`.

## 3. Secrets (OpenBao/Vault, .env fallback)

```bash
SECRETS_BACKEND=vault
SECRETS_VAULT_URL=https://vault.example:8200
VAULT_AUTH_METHOD=approle
# then reference secrets in config: GRAPH_DB_URI=vault://agents/db/pg_age#dsn
```

Use the `secret-vault-manager` skill to unseal/seed. On a laptop, a plain `.env` works.

## 4. Databases (single-node-prod / enterprise)

Run the [databases recipe](../recipes/databases.md) / `database-environment-setup`
skill: Stardog (prod) or local `/api/sparql` (dev) + a Postgres with AGE + pgvector +
pg_search, mirror fan-out wiring (`GRAPH_BACKEND=fanout` + `GRAPH_MIRROR_TARGETS`),
and graph backfill into the AGE mirror. The **tiny** profile skips this entirely —
the embedded epistemic-graph engine is the whole database, no mirror needed.

## 5. Launch

```bash
graph-os            # KG MCP server
graph-os-daemon     # REST gateway (mounts /api/sparql, /graph/*, /metrics)
mcp-multiplexer     # unified tool gateway over the *-mcp fleet
```

Containerized: `docker compose -f docker/mcp.compose.yml up -d` (plus
`docker/pg-age-full.compose.yml` for an optional pg-age mirror).

## 6. Auth & observability (enterprise)

`KG_AUTH_REQUIRED=1` + `AUTH_JWT_JWKS_URI` (Keycloak), policy via `eunomia-policy-manager`,
and `OTEL_EXPORTER_OTLP_ENDPOINT` for metrics/traces — all pre-seeded in the generated
enterprise config.

## 7. Multi-node → agent-os-genesis (day0)

For a full swarm (SSH mesh, hardware placement, overlay networks, ingress, GitOps,
fleet deploy), hand off to the **`agent-os-genesis`** skill (alias `day0`). This guide
generates and validates the config *around* that bootstrap; it doesn't reimplement it.

## 8. Verify

Run the holistic doctor — one sweep across config, engine, backend, secrets, auth,
the MCP fleet, hooks, and observability, each line carrying a remediation + the skill
that fixes it (brew/flutter-doctor style):

```bash
agent-utilities-doctor                 # human-readable; --json for machines, --fix for safe auto-remediation
agent-utilities-doctor --live          # also probe MCP endpoints
```

It composes the focused checks too, which you can still run directly:

```bash
setup-config doctor --profile single-node-prod        # config: required keys, durability, secret refs
python scripts/validate_mcp_config.py --live          # MCP reachability (catch 502s)
```

The config check reuses the production-safety rules (`collect_production_violations`)
so a config that pins you to a single host or in-memory broker is flagged before you
ship. Also reachable as `graph_configure(action="system_doctor")` (MCP/REST).
A green doctor + a `graph_write`/`graph_query` round-trip = you're up.

## See also
- [Day-0 overview](day0.md) · [Deployment configurations](deployment-configurations.md) ·
  [Configuration reference](../architecture/configuration.md)
- Recipes: [tiny](../recipes/tiny.md) · [single-node-prod](../recipes/single-node-prod.md) ·
  [enterprise](../recipes/enterprise.md) · [databases](../recipes/databases.md)
