# Quick Start Guide


## Quick Start

```bash
# Start a Graph Agent server with Universal Skills
agent-utilities --provider openai --model-id gpt-4o --skill-types universal,graphs

# Start with a custom MCP configuration
agent-utilities --mcp-config mcp_config.json --web --port 8000

# Run in validation mode (no API keys required)
VALIDATION_MODE=true agent-utilities --debug
```

```python
from agent_utilities import create_agent, create_graph_agent_server

# Quick agent creation
agent = create_agent(name="MyAgent", skill_types=["universal", "graphs"])

# Full server with protocols (ACP, A2A, MCP, AG-UI)
create_graph_agent_server(provider="openai", model_id="gpt-4o", port=8000)
```

> See [docs/creating-an-agent.md](docs/pillars/4_ecosystem_and_tooling/creating-an-agent.md) for the complete walkthrough.
