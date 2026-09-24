# ORCH-1.21: Agent Runner — KG-to-LLM Execution Bridge

## Concept Summary

| Field | Value |
|-------|-------|
| **Concept ID** | `ORCH-1.21` |
| **Pillar** | 1 — Graph Orchestration Engine |
| **Status** | Implemented |
| **Source Modules** | `orchestration/agent_runner.py` |
| **Test Modules** | `test_orchestrate_mcp.py` |
| **C4 Component** | Agent Runner |

## Overview

The **Agent Runner** bridges the `graph_orchestrate` MCP tool
to the pydantic-graph execution infrastructure. It provides deep KG integration
rather than a simple passthrough — resolving agents from the Knowledge Graph,
dynamically binding MCP toolsets, and recording execution provenance.

## Architecture

<div class="admonition architecture" markdown>
<p class="admonition-title">Agent name to traced run, via KG resolution and config build</p>

`graph_orchestrate execute_agent` resolves `agent_name` against the
Knowledge Graph three ways: matching Server nodes (tools, URL, command),
matching CallableResource nodes (type, capabilities), and hybrid semantic
search (best match). All three feed a config builder, which produces
`tag_prompts` + `mcp_toolsets` for `create_graph_agent()`. The
materialized graph runs via `run_graph()` (LM Studio), producing a
`GraphResponse` that feeds `RunTrace` provenance in the KG.
</div>

## Execution Lifecycle

### 1. Agent Resolution
Queries the KG across three node types:

| Search | Cypher Pattern | What it finds |
|--------|---------------|---------------|
| Server nodes | `MATCH (s:Server) WHERE s.name = $name` | MCP servers with tools |
| CallableResource | `MATCH (r:CallableResource) WHERE r.name = $name` | Skills, A2A agents |
| Semantic search | `engine.hybrid_search(name, top_k=3)` | Best-effort fuzzy match |

### 2. Config Builder
Constructs a `create_graph_agent()` config from resolved metadata:
- **tag_prompts**: Agent name + tool descriptions + capabilities
- **mcp_toolsets**: Dynamically created `MCPToolset` (stdio / SSE / streamable-HTTP transports) from URLs, built via `agent_utilities.mcp.toolset_factory`
- **LLM settings**: Unified model routing via XDG `config.json`

### 3. Graph Execution
Calls `create_graph_agent()` → `run_graph()` with the materialized graph.
Uses the same pipeline as the A2A agent and main server.

### 4. Provenance Tracking
Creates `RunTrace` nodes in the KG:
```cypher
CREATE (t:RunTrace {
  id: 'trace:run:abc12345',
  agent_name: 'portainer-agent',
  task: 'List all containers',
  status: 'completed',
  duration_ms: 1500.0,
  timestamp: '2026-05-18T01:00:00Z'
})
MERGE (t)-[:EXECUTED_ON]->(s:Server {id: 'srv:portainer-agent'})
```

## Error Handling

- If KG resolution fails → falls back to workspace-based discovery
- If LM Studio is unreachable → logs error, records `status: 'failed'` trace
- If `create_graph_agent()` fails → caught and reported with full traceback

## MCP Interface

```
graph_orchestrate(
    agent_name='portainer-agent',
    task='List all running containers',
    max_steps=30
)
```

## Related Concepts

- **AU-ORCH.execution.service-registry-initialization**: KG Graph Factory — materializes pydantic-graph from KG templates
- **AU-ECO.mcp.toolkit-live-discovery**: Agent Toolkit Ingestor — ingests Server/CallableResource nodes consumed by runner
- **AU-ECO.mcp.toolkit-live-discovery**: MCP Live Discovery — provides cached tool metadata for tool binding
- **ORCH-1.0**: Intelligence Graph Core — base graph infrastructure
