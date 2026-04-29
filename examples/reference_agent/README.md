# Reference Agent Example

This is a comprehensive end-to-end example demonstrating how to use agent-utilities to build a production-ready AI agent.

## Overview

The reference agent demonstrates:
- **Agent Creation**: Using the `create_agent` factory with various configurations
- **Graph Orchestration**: Router → Planner → Dispatcher pipeline
- **MCP Integration**: Loading tools from MCP servers
- **Knowledge Graph**: Using the unified intelligence graph for context
- **Protocol Support**: ACP, A2A, and AG-UI protocol adapters
- **Memory Management**: Knowledge graph-based memory primitives
- **Human-in-the-Loop**: Tool approval and elicitation

## Quick Start

### Simple Agent

```python
from agent_utilities import create_agent

# Create a simple agent with workspace tools
agent = create_agent(name="SimpleAgent")
```

### Graph Agent with Universal Skills

```python
from agent_utilities import create_agent

# Create a powerful Graph Agent with Universal Skills
agent = create_agent(
    name="ProAgent",
    skill_types=["universal", "graphs"]
)
```

### Agent with MCP Tools

```python
from agent_utilities import create_agent

# Create an agent that uses MCP tools
agent = create_agent(
    name="MCPAgent",
    skill_types=["universal"],
    mcp_config_path="mcp_config.json"
)
```

## Running the Examples

### Basic Example

```bash
cd examples/reference_agent
python basic_agent.py
```

### Graph Orchestration Example

```bash
python graph_agent.py
```

### MCP Integration Example

```bash
python mcp_agent.py
```

### Knowledge Graph Example

```bash
python knowledge_graph_agent.py
```

## Architecture

The reference agent follows the agent-utilities architecture:

```
User Query
    ↓
ACP / AG-UI / SSE (Unified Protocol Layer)
    ↓
Router (Topology Selection)
    ↓
Dispatcher (Dynamic Routing)
    ↓
Discovery Phase (Researcher, Architect, MCP Discovery)
    ↓
Execution Phase (Programmers, Infrastructure, Specialists)
    ↓
Verifier (Quality Gate)
    ↓
Synthesizer (Response Composition)
```

## Key Files

- `basic_agent.py` - Simple agent creation and execution
- `graph_agent.py` - Graph orchestration with Router/Planner/Dispatcher
- `mcp_agent.py` - MCP tool integration
- `knowledge_graph_agent.py` - Knowledge graph usage
- `protocol_agent.py` - Protocol adapters (ACP, A2A, AG-UI)
- `memory_agent.py` - Memory primitives and knowledge base

## Testing

Run tests for the reference agent:

```bash
cd ../..
pytest tests/test_reference_agent.py -v
```

## Next Steps

- Explore the agent-utilities [README.md](../../README.md) for full documentation
- Review [AGENTS.md](../../AGENTS.md) for core principles and the [docs/](../../docs/) directory for detailed architecture
- Check the [examples](../) directory for more specialized examples
