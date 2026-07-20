# Quick Start

From nothing to a running, **verified** agent-utilities in a few minutes. This is the
fast path; for the full config-complete walkthrough (secrets, profiles, multi-node)
see the [Self-Setup guide](self-setup.md), and for the database environment see the
[Stardog + pg-age recipe](../recipes/databases.md).

## TL;DR — self-contained local GraphOS

```bash
pip install "agent-utilities[serving]"
setup-config generate --profile tiny
agent-utilities-doctor --only graph_identity auth
graph-os --transport stdio
```

That starts a durable knowledge graph and MCP server with **no separately managed
database or service**. GraphOS supervises the packaged Rust epistemic-graph engine
as an out-of-process child over a private local transport. Scale up later by
generating the `single-node-prod` profile. The exact tiny, packaged-local stdio
boundary uses a neutral in-memory bootstrap session, so it needs no IdP credential.

---

## 1. Install

```bash
pip install "agent-utilities[serving]"             # supported GraphOS runtime
# Optional external integrations compose by name, for example:
pip install "agent-utilities[serving,owl,postgresql,stardog]"
```

## 2. Generate your config (all options)

Don't hand-write `config.json` — generate a complete, profile-seeded one that covers
**every** option at a sensible default:

```bash
setup-config generate --profile tiny       # writes the XDG AgentConfig
setup-config reference                     # browse every option by subsystem
```

Profiles: `tiny` (local/edge), `single-node-prod` (one durable host), and
`enterprise` (multi-node). Generated secret-bearing fields are blank. Persist
runtime secret references only; do not place resolved credentials, tokens,
certificate paths, or machine locations in the file.

For this tiny local stdio path, leave `GRAPH_SERVICE_ENDPOINTS`,
`KG_AUTH_TOKEN_REF`, and `KG_IDENTITY_OAUTH2` unset. GraphOS signs and validates a
short-lived JWT with an in-memory key as a one-time proof, destroys the key and
token, and returns a process-lifetime session; no user, host, endpoint,
filesystem, credential, or proof data is persisted. Validate the boundary:

```bash
agent-utilities-doctor --only graph_identity auth
```

Every network transport, non-tiny profile, explicit engine endpoint, and other
entry point requires exactly one external process identity source and its JWT
validation policy. A configured-but-invalid source fails closed without local
fallback. External stdio sessions remain bounded by a renewable shared
expiry-only lease; identity drift is rejected and failed renewal cannot extend
authority beyond the validated expiry.

## 3. (Optional) Databases — single-node-prod / enterprise

The `tiny` profile needs nothing here. For a durable Postgres tier (Apache AGE +
pgvector + ParadeDB) and/or Stardog, run:

```bash
docker compose -f docker/pg-age-full.compose.yml up -d --build   # AGE + pgvector + pg_search
setup-databases --profile dev --connection-profile-ref "secret://graph/mirror-profile"
```

Full detail (prod Stardog, dev local SPARQL, backfill into AGE, OpenBao):
[databases recipe](../recipes/databases.md) / the `database-environment-setup` skill.

## 4. Launch

```bash
graph-os                       # MCP server; choose this for MCP clients
graph-os-daemon                # optional headless work host; no HTTP API
python -m agent_utilities      # agent + REST/API gateway; also hosts background work
```

## 5. Verify

Run the doctor — one sweep across config, engine, backend, secrets, auth, MCP fleet,
hooks, and observability, each line carrying a fix + the skill that resolves it:

```bash
agent-utilities-doctor          # human-readable; --json for machines, --fix for safe auto-remediation, --live to probe endpoints
```

A `HEALTHY` (or `WARNINGS`) verdict + a `graph_write`/`graph_query` round-trip means
you're up.

---

## Use it

```python
from agent_utilities import create_agent, create_agent_server

# Quick agent (skill_types selects which skill bundles to load)
agent = create_agent(name="assistant", skill_types=["universal", "graphs"])

# Full server uses the provider/model registry from AgentConfig.
create_agent_server()
```

See [creating-an-agent.md](creating-an-agent.md) for the complete agent walkthrough.

## Console scripts (CLI reference)

Installed by the package:

| Command | What it does |
|---|---|
| `setup-config {generate,doctor,reference}` | Generate the complete config.json, validate it, or list every option by subsystem |
| `setup-databases` | Provision Stardog + pg-age and backfill the graph into Apache AGE |
| `agent-utilities-doctor` | Holistic deployment health sweep (`--fix`, `--live`, `--json`) |
| `graph-os` | The Knowledge-Graph MCP server (graph-os) |
| `graph-os-daemon` | Headless queue, maintenance, and background-work host (`--status`); no HTTP API |
| `agent-utilities-memory` | Memory store CLI |
| `python -m agent_utilities` | Launch the interactive agent (flags: `--provider`, `--model-id`, `--mcp-config`, `--web`, `--port`) |

Each command is also reachable over MCP/REST via the `graph_configure` tool
(`generate_config`, `config_doctor`, `system_doctor`, `setup_databases`, …).

## Where to go next

- [Self-Setup (config-complete, the path Claude follows)](self-setup.md)
- [Deployment configurations — the ladder](deployment-configurations.md) ·
  [Configuration reference](../architecture/configuration.md)
- [Universal external graph connectors](../architecture/universal-external-graph-connectors.md)
- Recipes: [tiny](../recipes/tiny.md) · [single-node-prod](../recipes/single-node-prod.md) ·
  [enterprise](../recipes/enterprise.md) · [databases](../recipes/databases.md)
- [Day-0 multi-node bootstrap (`agent-utilities-deployment`)](day0.md)
