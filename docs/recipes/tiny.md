# Recipe — Tiny (all-local, zero-infra)

> Ladder position: this recipe is **rung (a) — Zero-infra dev** of the
> [supported deployment configurations](../guides/deployment-configurations.md#rung-a-zero-infra-dev)
> guide, which lists every default this tier relies on.

For a laptop, a dev box, or an edge node. **No databases, no external services.**
The knowledge graph runs entirely on this machine (the Rust engine as a local
daemon plus embedded LadybugDB); the only thing you need is a model provider
(a hosted API key, or a local vLLM/Ollama endpoint).

## What runs

| Component | How |
|---|---|
| agent-utilities | pip/uv install, in-process |
| Knowledge graph | `GRAPH_BACKEND=tiered` → `epistemic_graph` (L1) + embedded LadybugDB (L2) |
| **OWL/RDF + reasoning** | **on by default** — local OWL-RL inference (epistemic-graph) over the LPG, no external triplestore |
| **SPARQL** | **local endpoint** at `GET/POST {gateway}/api/sparql` (rdflib materialization + engine `GetTriples` fast path) — zero external deps |
| graph-os MCP | optional, `uv run graph-os` (stdio) |
| External services | **none** |

OWL/RDF is a **core, always-on** layer here — not an enterprise add-on. The tiny
profile consumes the bundled ontologies, infers new relationships, and serves
SPARQL **locally** with no Fuseki/Stardog. (Fuseki/Stardog are an *optional*
enterprise scale-out, configured only in the [enterprise recipe](enterprise.md).)

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
# --- Knowledge graph: zero-infra default (this is already the default) ---
GRAPH_BACKEND=tiered

# --- Engine: auto-spawn the local Rust engine on first connect ---
# Off by default; without it the `epistemic-graph-server` daemon (ships with
# the epistemic-graph wheel) must already be running. Autostart only ever
# applies to the local endpoint.
EPISTEMIC_GRAPH_AUTOSTART=1

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

The moment you want the KG to survive restarts on a server, or to share it across
processes/containers, move to [Single-node prod](single-node-prod.md) — it's a
one-line `GRAPH_DB_URI` change. The full progression (auth, durable state,
multi-host scale-out, autonomy) is the
[deployment configurations ladder](../guides/deployment-configurations.md).
