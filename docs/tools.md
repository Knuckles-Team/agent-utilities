# Tools Registry

> CONCEPT:AU-010 — Agent Tool System

## Overview

The `tools/` module provides 18 tool modules that are exposed to agents during graph execution. Each tool module registers functions that the LLM can call via the PydanticAI tool system.

## Tool Categories

### Agent & Team Tools

| Module | Key Functions | Description |
|---|---|---|
| `agent_tools` | `agent_share_your_reasoning`, `agent_run_shell_command` | Core agent capabilities |
| `team_tools` | `delegate_to_team`, `request_review` | Multi-agent coordination |
| `a2a_tools` | `discover_agents`, `send_a2a_message` | Agent-to-Agent protocol tools |

### Developer Tools

| Module | Key Functions | Description |
|---|---|---|
| `developer_tools` | `create_file`, `replace_in_file`, `read_file` | File manipulation |
| `git_tools` | `git_status`, `git_diff`, `git_commit` | Version control |
| `workspace_tools` | `list_workspace`, `search_files`, `grep` | Workspace navigation |

### Knowledge & Memory Tools

| Module | Key Functions | Description |
|---|---|---|
| `knowledge_tools` | `query_knowledge_graph`, `store_knowledge` | KG read/write |
| `memory_tools` | `recall_memory`, `store_memory` | Conversational memory |
| `kg_evolution_tools` | `evolve_schema`, `migrate_graph` | KG schema management |
| `kg_share_tools` | `export_subgraph`, `import_subgraph` | KG data sharing |
| `pattern_tools` | `detect_pattern`, `apply_pattern` | Pattern recognition |

### Infrastructure Tools

| Module | Key Functions | Description |
|---|---|---|
| `scheduler_tools` | `schedule_task`, `list_scheduled` | Task scheduling |
| `mcp_sync_tool` | `sync_mcp_agents` | MCP server discovery |
| `sdd_tools` | `run_sdd_pipeline`, `validate_specs` | Spec-driven development |
| `self_improvement_tools` | `analyze_codebase`, `suggest_improvements` | Self-improvement |
| `style_tools` | `apply_coding_style`, `check_conventions` | Code style enforcement |
| `onboarding_tools` | `analyze_project`, `generate_onboarding` | Project onboarding |

## Tool Registration

Tools are registered with the agent during creation:

```python
from agent_utilities.agent.factory import create_agent

# Tools are automatically loaded based on skill_types
agent = create_agent(
    name="MyAgent",
    skill_types=["universal", "graphs"],
)
```

## Tool Guard (`tool_guard.py`)

Safety layer that validates tool calls before execution:
- **Allowlist/blocklist**: Restrict which tools are available
- **Rate limiting**: Prevent excessive tool calls
- **Approval gates**: Require human approval for destructive operations

## Tool Filtering (`tool_filtering.py`)

Dynamic tool selection based on context:
- Reduces token usage by only exposing relevant tools
- Uses semantic similarity to match tools to the current task
- Supports explicit tool pinning via configuration

## Tool Registry (`tool_registry.py`)

Central registry for all available tools:

```python
from agent_utilities.tool_registry import ToolRegistry

registry = ToolRegistry()
registry.register("my_tool", my_function, description="Does something")

# List all tools
for tool in registry.list_tools():
    print(f"{tool.name}: {tool.description}")
```
