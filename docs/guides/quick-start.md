# Quick Start

From nothing to a running, **verified** agent-utilities in a few minutes. This is the
fast path; for the full config-complete walkthrough (secrets, profiles, multi-node)
see the [Self-Setup guide](self-setup.md), and for the database environment see the
[Stardog + pg-age recipe](../recipes/databases.md).

## TL;DR — zero-infra, 4 commands

```bash
pip install agent-utilities[all]
setup-config generate --profile tiny        # complete config.json (every option)
graph-os &                                   # KG MCP server (zero external services)
agent-utilities-doctor                       # sweep & verify the install
```

That's a working, durable-on-disk knowledge graph + MCP server with **no database or
external service** (the `tiny` profile uses an in-process engine + embedded LadybugDB).
Scale up later by re-running `setup-config generate --profile single-node-prod`.

---

## 1. Install

```bash
pip install agent-utilities[all]
# or a narrower set, e.g.:  pip install agent-utilities[owl,postgres,stardog]
```

## 2. Generate your config (all options)

Don't hand-write `config.json` — generate a complete, profile-seeded one that covers
**every** option at a sensible default:

```bash
setup-config generate --profile tiny                 # → ~/.config/agent-utilities/config.json
setup-config reference                               # browse every option, grouped by subsystem
```

Profiles: `tiny` (laptop/edge), `single-node-prod` (one durable host),
`enterprise` (multi-node). Secret-like keys are blanked — fill them via env or
`vault://` refs, never in the committed file.

## 3. (Optional) Databases — single-node-prod / enterprise

The `tiny` profile needs nothing here. For a durable Postgres tier (Apache AGE +
pgvector + ParadeDB) and/or Stardog, run:

```bash
docker compose -f docker/pg-age-full.compose.yml up -d --build   # AGE + pgvector + pg_search
setup-databases --profile dev --dsn postgresql://agent:agent@localhost:5432/agent_kg
```

Full detail (prod Stardog, dev local SPARQL, backfill into AGE, OpenBao):
[databases recipe](../recipes/databases.md) / the `database-environment-setup` skill.

## 4. Launch

```bash
graph-os                       # KG MCP server (stdio / streamable-http)
graph-os-daemon                # REST gateway (mounts /api/graph/*, /api/sparql, /metrics)
mcp-multiplexer                # one endpoint over the whole *-mcp fleet
# …or the interactive agent:
python -m agent_utilities --provider openai --model-id gpt-4o
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
agent = create_agent(name="MyAgent", skill_types=["universal", "graphs"])

# Full server with protocols (ACP, A2A, MCP, AG-UI)
create_agent_server(provider="openai", model_id="gpt-4o", port=8000)
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
| `graph-os-daemon` | The REST gateway / KG daemon (`--status`) |
| `mcp-multiplexer` | Unified MCP tool gateway over the connector fleet |
| `agent-utilities-memory` | Memory store CLI |
| `python -m agent_utilities` | Launch the interactive agent (flags: `--provider`, `--model-id`, `--mcp-config`, `--web`, `--port`) |

Each command is also reachable over MCP/REST via the `graph_configure` tool
(`generate_config`, `config_doctor`, `system_doctor`, `setup_databases`, …).

## Where to go next

- [Self-Setup (config-complete, the path Claude follows)](self-setup.md)
- [Deployment configurations — the ladder](deployment-configurations.md) ·
  [Configuration reference](../architecture/configuration.md)
- Recipes: [tiny](../recipes/tiny.md) · [single-node-prod](../recipes/single-node-prod.md) ·
  [enterprise](../recipes/enterprise.md) · [databases](../recipes/databases.md)
- [Day-0 multi-node bootstrap (agent-os-genesis / day0)](day0.md)
