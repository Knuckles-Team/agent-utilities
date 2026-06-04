# Quick Start Guide


## Quick Start

```bash
# Start the agent server (CLI entrypoint is `python -m agent_utilities`)
python -m agent_utilities --provider openai --model-id gpt-4o

# Start with a custom MCP configuration and the web UI on a chosen port
python -m agent_utilities --mcp-config mcp_config.json --web --port 8000

# Run in validation mode (no API keys required)
VALIDATION_MODE=true python -m agent_utilities --debug
```

> Console scripts installed by the package are `graph-os` (the KG MCP server),
> `agent-utilities-memory`, and `mcp-multiplexer`. The interactive agent itself
> is launched via `python -m agent_utilities`.

```python
from agent_utilities import create_agent, create_agent_server

# Quick agent creation (skill_types selects which skill bundles to load)
agent = create_agent(name="MyAgent", skill_types=["universal", "graphs"])

# Full server with protocols (ACP, A2A, MCP, AG-UI)
create_agent_server(provider="openai", model_id="gpt-4o", port=8000)
```

> See [creating-an-agent.md](creating-an-agent.md) for the complete walkthrough.
