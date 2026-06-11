# Recipe — Tiny (all-local, zero-infra)

For a laptop, a dev box, or an edge node. **No databases, no external services.**
The knowledge graph runs in-process; the only thing you need is a model provider
(a hosted API key, or a local vLLM/Ollama endpoint).

## What runs

| Component | How |
|---|---|
| agent-utilities | pip/uv install, in-process |
| Knowledge graph | `GRAPH_BACKEND=tiered` → `epistemic_graph` (L1) + embedded LadybugDB (L2) |
| graph-os MCP | optional, `uv run graph-os` (stdio) |
| External services | **none** |

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

# --- Model provider: pick ONE ---
OPENAI_API_KEY=sk-REDACTED
# ...or a local OpenAI-compatible endpoint (vLLM/Ollama):
# OPENAI_BASE_URL=http://localhost:8000/v1

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
one-line `GRAPH_DB_URI` change.
