---
name: kg-delegate
skill_type: skill
tier: meta
description: >
  KG-driven agent skill delegation router. Queries the Knowledge Graph for available
  MCP servers and tools, auto-hydrates the KG from mcp_config*.json files if cold,
  then delegates execution via graph_orchestrate (execute_agent or execute_workflow).
  Bypasses IDE tool limits by offloading orchestration to the epistemic-graph backend.
tags: [orchestration, delegation, kg, mcp, router]
---

# KG-Driven Delegation Router

Routes user tasks to the appropriate MCP server agent by querying the Knowledge Graph
for registered `Server` and `CallableResource` nodes. If the KG is empty (cold start),
it auto-ingests all discovered `mcp_config*.json` files and skill directories before
delegating.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       KG Delegation Router                          │
│                                                                      │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────┐ │
│  │ Phase 1      │    │ Phase 2          │    │ Phase 3              │ │
│  │ KG Discovery │───▶│ Auto-Hydration   │───▶│ Delegation           │ │
│  │              │    │ (if cold)        │    │                      │ │
│  │ graph_query  │    │ graph_ingest     │    │ graph_orchestrate    │ │
│  │              │    │ action=          │    │ action=execute_agent │ │
│  │ Check for    │    │ agent_toolkit    │    │ OR execute_workflow  │ │
│  │ Server nodes │    │                  │    │ OR compile_workflow  │ │
│  └─────────────┘    └─────────────────┘    └──────────────────────┘ │
│         │                    │                        │              │
│         ▼                    ▼                        ▼              │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────┐ │
│  │ Epistemic    │    │ Scan known IDE   │    │ KG resolves Server   │ │
│  │ Graph        │    │ paths for        │    │ → binds MCP tools    │ │
│  │ (Rust)       │    │ mcp_config*.json │    │ → materializes       │ │
│  │              │    │ + skill dirs     │    │   pydantic-ai graph  │ │
│  │ Sub-ms query │    │ Deduplicate      │    │ → executes via LLM   │ │
│  │ performance  │    │ by name+cmd      │    │ → returns result     │ │
│  └─────────────┘    └─────────────────┘    └──────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### Performance Note

Hydration and query operations run at **sub-millisecond latency** because all graph
operations are backed by the native Rust **epistemic-graph** engine. The Rust backend
handles node creation, edge linking, Cypher-like queries, and deduplication natively
without Python GIL contention. Even ingesting 40+ MCP server configs with hundreds
of tools completes in under a second.

### Delegation Modes

The router supports three delegation modes via `graph_orchestrate`:

| Mode | Action | When to Use |
|------|--------|-------------|
| **Prompt-based** | `execute_agent` | Single-server tasks (e.g. "get GitHub pipeline status") |
| **Workflow** | `execute_workflow` | Pre-compiled multi-step workflows stored in KG |
| **Compile + Execute** | `compile_workflow` → `execute_workflow` | Ad-hoc multi-step tasks from natural language |

### In-band alternative: the dynamic multiplexer gateway (CONCEPT:ECO-4.36)

This router delegates **out-of-band** — the KG runs the chosen agent server-side
and returns a result. When the caller is itself an MCP client (e.g. Claude Code)
connected through the **mcp-multiplexer in `dynamic` mode**, there is a
complementary **in-band** path that makes the discovered tools directly callable
in the client instead of running them remotely:

1. `find_tools(query)` — same KG-backed discovery (`graph_search` over
   `CallableResource` nodes), returning ranked prefixed tool names.
2. `load_tools(tools=[...])` — mounts the owning child server lazily and exposes
   those tools live (the client is sent `tools/list_changed`).
3. call the tools natively; `unload_tools(...)` to release them.

Use this router's `graph_orchestrate` path for autonomous/headless delegation;
use the multiplexer meta-tools when an interactive client should gain the tools
itself. Both share the same KG `Server → PROVIDES → CallableResource` substrate.

## Workflow

### Step 1: KG Discovery [depends_on: none]

Check if the Knowledge Graph has any MCP servers registered.

1. Call `graph_query` with this Cypher query to check KG state:
   ```
   MATCH (s:Server)-[:PROVIDES]->(r:CallableResource)
   RETURN s.name AS server, count(r) AS tool_count
   ORDER BY tool_count DESC LIMIT 50
   ```
2. If results are **empty** (KG is cold / no Server nodes exist), proceed to Step 2.
3. If results exist, search for the server that best matches the user's task:
   ```
   MATCH (s:Server)-[:PROVIDES]->(r:CallableResource)
   WHERE toLower(r.description) CONTAINS toLower('<user_task_keywords>')
      OR toLower(s.name) CONTAINS toLower('<user_task_keywords>')
      OR toLower(r.name) CONTAINS toLower('<user_task_keywords>')
   RETURN s.name AS server, collect(r.name) AS tools
   LIMIT 3
   ```
4. If a matching server is found, proceed to Step 3 with the resolved `server` name.
5. If no match is found but servers exist, report available servers and ask the user to clarify.

### Step 2: Auto-Hydration [depends_on: step 1]

Ingest MCP server configs and skill directories into the KG.

1. **Scan known IDE paths** for `mcp_config*.json` files using glob pattern matching:
   - `~/.gemini/antigravity/mcp_config*.json` — Antigravity
   - `~/.config/agent-utilities/mcp_config*.json` — XDG agent-utilities config
   - `~/.codeium/windsurf/mcp_config*.json` — Windsurf
   - `~/.claude/mcp_config*.json` — Claude Code
   - `~/.codex/mcp_config*.json` — Codex
   - `~/.config/devin/mcp_config*.json` — Devin
   - Any user-specified path (pass as argument to the skill)

2. **Collect all discovered config paths** into a JSON array. Example:
   ```json
   [
     "/home/genius/.gemini/antigravity/mcp_config.json",
     "/home/genius/.gemini/antigravity/mcp_config_source.json"
   ]
   ```

3. **Deduplicate** across all discovered configs before ingestion:
   - If two servers share the same `command` + `args` combination, keep only the first occurrence.
   - If two `CallableResource` nodes share the same `name` AND `description`, skip the duplicate.
   - The epistemic-graph backend handles MERGE semantics natively, so duplicate Server IDs (`srv:<name>`) are automatically deduplicated at the storage layer.

4. **Ingest MCP configs** via `graph_ingest`:
   ```
   graph_ingest(
       action="agent_toolkit",
       target_path='["<path1>", "<path2>", ...]'
   )
   ```

5. **Ingest skill directories** as additional sources (pass directory paths):
   ```
   graph_ingest(
       action="agent_toolkit",
       target_path='["/home/apps/workspace/agent-packages/skills/universal-skills/universal_skills/agent-tools/"]'
   )
   ```

6. **Verify ingestion** completed successfully:
   ```
   graph_query(query="MATCH (s:Server) RETURN s.name, s.tool_count ORDER BY s.name")
   ```

7. Retry Step 1 to find the matching server.

### Step 3: Delegation [depends_on: step 2]

Execute the user's task through the KG orchestration layer.

Choose the appropriate delegation mode based on the task:

**Mode A — Prompt-based agent execution** (default for single-server tasks):
```
graph_orchestrate(
    action="execute_agent",
    agent_name="<resolved_server>",
    task="<user_task>",
    max_steps=30
)
```

This triggers the full KG-backed execution pipeline:
1. `agent_runner.run_agent()` queries the KG for the server's metadata
2. Resolves `command`, `args`, `env` from the Server node
3. Creates an `MCPServerStdio` toolset with the resolved config
4. Materializes a pydantic-ai graph with those tools bound
5. Executes the graph against the configured LLM
6. Records execution provenance as a `RunTrace` node in the KG
7. Returns the result

**Mode B — Execute a stored workflow** (for multi-step tasks with a matching workflow):
```
# Check for existing workflows:
graph_orchestrate(action="list_workflows")

# Execute if found:
graph_orchestrate(
    action="execute_workflow",
    agent_name="<workflow_name>",
    task="<user_input>"
)
```

This triggers `WorkflowRunner.execute_by_name()` which:
1. Loads the `GraphPlan` from the KG
2. Builds execution waves (parallel groups based on dependencies)
3. Executes each step via `run_agent()` with the step's agent
4. Returns aggregate `WorkflowResult` with per-step outputs

**Mode C — Compile NL workflow on-the-fly** (for ad-hoc multi-step tasks):
```
# Compile natural language into a stored workflow:
graph_orchestrate(
    action="compile_workflow",
    agent_name="<workflow_name>",
    task="<NL description of steps>"
)

# Then execute it:
graph_orchestrate(
    action="execute_workflow",
    agent_name="<workflow_name>",
    task="<user_input>"
)
```

This uses `WorkflowCompiler` to:
1. Parse NL description into discrete step intents
2. Match each step to a KG-registered agent/server
3. Build the dependency DAG
4. Persist as a reusable `GraphPlan` in the KG
5. Execute via `WorkflowRunner`

## Example: GitHub Actions Pipeline Check

To get the last pipeline run for `Knuckles-Team/agent-utilities`:

```
# Step 1: Check KG
graph_query(query="MATCH (s:Server {name: 'github-mcp'}) RETURN s.name, s.tool_count")

# Step 2: If not found, hydrate
graph_ingest(
    action="agent_toolkit",
    target_path="/home/genius/.gemini/antigravity/mcp_config_source.json"
)

# Step 3: Delegate
graph_orchestrate(
    action="execute_agent",
    agent_name="github-mcp",
    task="List the most recent workflow run for Knuckles-Team/agent-utilities. Return the run name, status, conclusion, and URL."
)
```
