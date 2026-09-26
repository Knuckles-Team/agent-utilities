# Tools Registry

> CONCEPT:AU-ECO.messaging.native-backend-abstraction — Agent Tool System

## Overview

The `tools/` package provides the PydanticAI tool surface used by direct agent
execution. `register_agent_tools` is the authoritative registry; optional groups
are enabled through `AgentConfig`, and graph orchestrators expose only
`execute_graph`, the protocol-agnostic graph execution authority, so routing
remains isolated.

## Tool Categories

### Agent & Team Tools

| Module | Key Functions | Description |
|---|---|---|
| `agent_tools` | `share_reasoning`, `invoke_specialized_agent` | Core agent capabilities |
| `team_tools` | `spawn_team`, `assign_team_task`, `message_teammate` | Multi-agent coordination |
| `a2a_tools` | `list_a2a_peers`, `register_a2a_peer`, `delete_a2a_peer` | Agent-to-Agent peer registry |

### Developer Tools

| Module | Key Functions | Description |
|---|---|---|
| `developer_tools` | `project_search` plus shared KG tools | Read-only code and knowledge discovery |
| `git_tools` | `get_git_status`, `list_worktrees` | Privacy-safe repository inspection |
| `workspace_tools` | `read_workspace_file`, `get_skill_content`, `list_files` | Bounded, read-only workspace navigation |

### Knowledge & Memory Tools

| Module | Key Functions | Description |
|---|---|---|
| `kg_tools` | `kg_search`, `kg_recall`, `kg_query` | Shared graph discovery and queries |
| `knowledge_tools` | memory CRUD, SDD synchronization, and governed knowledge-base operations | Knowledge and memory lifecycle |
| `memory_tools` | `read_agents_md` | Bounded, sanitized project instructions |
| `kg_evolution_tools` | `extract_and_ingest_triples` | Governed triple extraction and ingestion |
| `kg_share_tools` | `export_subgraph`, `import_agent_card` | KG data sharing |
| `pattern_tools` | `run_manual_test` | Pattern test execution |

### Social & Search Tools

| Module | Key Functions | Description |
|---|---|---|
| `x_search_tool` | `x_search`, `browse_x_post` | Native X search and post browsing using xAI authentication |

### Infrastructure Tools

| Module | Key Functions | Description |
|---|---|---|
| `scheduler_tools` | `schedule_task`, `list_tasks`, `delete_task`, `view_cron_log` | Task scheduling |
| `mcp_sync_tool` | `trigger_mcp_sync` | MCP server discovery |
| `sdd_tools` | project context, specification, task, import/export, and TDD operations | Spec-driven development |
| `self_improvement_tools` | improvement cycles, skill proposals, and experiment queries | Self-improvement |
| `style_tools` | `set_output_style`, `list_output_styles` | Output-style selection |
| `onboarding_tools` | `bootstrap_project` | Project onboarding |

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

## Tool Guard (`security/tool_guard.py`)

Safety layer that validates tool calls before execution:
- **Allowlist/blocklist**: Restrict which tools are available
- **Rate limiting**: Prevent excessive tool calls
- **Approval gates**: Require human approval for destructive operations

## Tool Filtering (`tool_filtering.py`)

Dynamic tool selection based on context:
- Reduces token usage by only exposing relevant tools
- Uses semantic similarity to match tools to the current task
- Supports explicit tool pinning via configuration

## Tool Registry (`tools/tool_registry.py`)

Central entry point that wires all available tool modules onto a PydanticAI
agent at creation time:

```python
from agent_utilities.tools.tool_registry import register_agent_tools

# Registers the relevant tool modules onto the agent, guarded against duplicate
# registration and the configured optional-tool policy.
register_agent_tools(agent, graph_bundle=my_graph_bundle)
```

## X & xAI Integration

The `x_search_tool` provides native capabilities to search X posts and browse individual posts directly from an agent. It leverages the secure `XaiAuthManager` to obtain OAuth 2.0 access tokens.

### Functions

- **`x_search(query: str, allowed_x_handles=None, excluded_x_handles=None, from_date="", to_date="")`**: Searches X posts for a query and returns matching posts, authors, and text. Exactly one of `allowed_x_handles`/`excluded_x_handles` may be set.
- **`browse_x_post(url: str, auto_ingest: bool = False)`**: Fetches the text, author, and engagement metrics for a specific X post by its URL; optionally classifies and ingests it into the KG.

### Example Code

```python
# The agent will automatically call these tools when given an X URL or asked to search X:
result = browse_x_post("https://x.com/gkisokay/status/2056726149074657704")
print(result)
```
