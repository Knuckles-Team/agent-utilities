# Recipe — Tiny (all-local, zero-infra)

> Ladder position: this recipe is **rung (a) — Zero-infra dev** of the
> [supported deployment configurations](../guides/deployment-configurations.md#rung-a-zero-infra-dev)
> guide, which lists every default this tier relies on.

For a laptop, a dev box, or an edge node. **No databases, no external services,
no container stack.** The knowledge graph runs entirely on this machine: the
epistemic-graph engine *is* the one database — compute, in-memory cache,
semantic/ontology reasoning, **and** durable persistence in a single engine. It
auto-spins-up as a **self-contained, lifecycle-coupled embedded child** of
whatever needs it (graph-os, or a connector's `agent_server.py`) and dies with
that process. There are **no mirror databases** (Postgres/pg-age, Neo4j,
FalkorDB, Ladybug are optional write-only fan-out targets you do not configure
here). The only thing you need is a model provider (a hosted API key, or a local
vLLM/Ollama endpoint).

## What runs

| Component | How |
|---|---|
| agent-utilities | pip/uv install, in-process |
| Knowledge graph | the **epistemic-graph engine authority** — one embedded engine, durable on disk (`--persist-dir`), no mirrors |
| Engine lifecycle | auto-spun-up as a **shared local daemon, reference-counted** (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision): the ONE resolver autostarts a detached engine on first use and it self-stops ~60s after the last client disconnects. Set `engine_lifecycle=persistent` for a **long-living** engine that never auto-stops (warm, like a local service). A remote engine (enterprise) is inherently persistent. |
| **OWL/RDF + reasoning** | **on by default** — local OWL-RL inference (epistemic-graph) over the LPG, no external triplestore |
| **SPARQL** | **local endpoint** at `GET/POST {gateway}/api/sparql` (rdflib materialization + engine `GetTriples` fast path) — zero external deps |
| graph-os MCP | optional, `uv run graph-os` (stdio) |
| External services | **none** |

There is **no 5-container requirement** for tiny — the engine is the only moving
part, and it is embedded. OWL/RDF is a **core, always-on** layer here, not an
enterprise add-on: the tiny profile consumes the bundled ontologies, infers new
relationships, and serves SPARQL **locally** with no Fuseki/Stardog.
(Fuseki/Stardog are an *optional* enterprise scale-out, configured only in the
[enterprise recipe](enterprise.md).)

## Steps

```bash
git clone https://github.com/Knuckles-Team/agent-utilities && cd agent-utilities
./scripts/bootstrap.sh
```

`bootstrap.sh` does it all: creates a venv, installs `.[all]`, writes a `.env`,
and runs a smoke test. Or do it manually:

```bash
uv sync                         # or: pip install -e ".[agent]"
cp .env.example .env            # then edit the model provider line
```

## `.env` (generalized)

```dotenv
# --- Knowledge graph: the engine is the one authority; no mirrors configured ---
GRAPH_BACKEND=epistemic_graph

# --- Engine: the ONE resolver auto-provisions a SHARED local engine (OS-5.63) ---
# Default for tiny: engine_mode=auto autostarts a detached, durable engine on
# first use (it persists to disk via --persist-dir). Because it is detached, every
# entrypoint on this host (graph-os, a connector's agent_server, the gateway)
# SHARES the ONE engine. It is reference-counted: it self-stops
# ENGINE_IDLE_SHUTDOWN_SECS (default 60) after its LAST client disconnects.
ENGINE_MODE=auto
ENGINE_LIFECYCLE=refcounted          # auto-stops when idle (the tiny default)
ENGINE_IDLE_SHUTDOWN_SECS=60
#
# Prefer a warm engine that never auto-stops, even when idle (runs like a local
# service)? Make it LONG-LIVING instead:
#   ENGINE_LIFECYCLE=persistent       # (or ENGINE_IDLE_SHUTDOWN_SECS=0)
#
# A configured remote engine (ENGINE_ENDPOINT / GRAPH_SERVICE_ENDPOINTS) switches
# to engine_mode=remote and is never autostarted — that's the enterprise path.
# Set EPISTEMIC_GRAPH_AUTOSTART=0 for a connect-only process (no autostart).

# --- Model provider: pick ONE ---
OPENAI_API_KEY=sk-REDACTED
# ...or a local OpenAI-compatible endpoint (vLLM/Ollama):
# OPENAI_BASE_URL=http://localhost:8000/v1

# Optional identity label for MCP/harness client processes
AGENT_ID=local-developer
```

## Verify

```python
from agent_utilities import create_agent
agent, _ = create_agent(name="assistant", skill_types=["universal", "graphs"])
print(agent.run_sync("Add a node 'hello' of type Greeting, then count nodes.").output)
```

Or via MCP — register `graph-os` in your IDE's `mcp_config.json`:

```json
{ "mcpServers": { "graph-os": { "command": "uv", "args": ["run", "graph-os"] } } }
```

## When to graduate

The embedded engine is already durable across restarts of *its own process*. The
moment you want the engine to run independently of any one agent process, or to
share it across containers/hosts, move to
[Single-node prod](single-node-prod.md) — there the same engine runs as its own
container; [enterprise](enterprise.md) points everything at a shared/remote
engine via `GRAPH_SERVICE_ENDPOINTS` and adds optional mirrors. The full
progression (auth, multi-host scale-out, autonomy) is the
[deployment configurations ladder](../guides/deployment-configurations.md).
