# Graph-OS MCP Server Examples

The `graph-os` MCP server is the unified entrypoint for interacting with the Intelligence Graph Engine. It exposes 10 core tools that allow LLMs and automated pipelines to manipulate knowledge, orchestrate agents, and manage the workspace: the 7 documented below plus `graph_sessions` (durable session management), `graph_goals` (autonomous background loops), and `graph_hydrate` (instant KG hydration from external connectors).

Below are exhaustive examples of every possible tool configuration you can execute.

## 1. `mcp_graph-os_graph_ingest`

Handles smart ingestion for codebases, documents, directories, conversations, agent skills, and mcp servers.

**Example 1: Ingesting an Agent Toolkit**
Ingest `mcp_config.json` files and skill directories.
```json
{
  "action": "agent_toolkit",
  "target_path": "[\"/path/to/universal-skills\", \"/path/to/mcp_config.json\"]"
}
```

**Example 2: Ingesting a Codebase**
Trigger full AST parsing and semantic chunking of a repository.
```json
{
  "action": "ingest",
  "target_path": "/home/apps/workspace/my-repo",
  "max_depth": 3
}
```

**Example 3: Checking Job Status**
```json
{
  "action": "job_status",
  "job_id": "job-3d73bbc4"
}
```

## 2. `mcp_graph-os_graph_search`

Search the Knowledge Graph using multiple strategies.

**Example 1: Hybrid Search (Semantic + Keyword)**
```json
{
  "mode": "hybrid",
  "query": "authentication flow architecture",
  "top_k": 5
}
```

**Example 2: Concept ID Lookup**
Retrieve the exact topological subgraph for a known concept.
```json
{
  "mode": "concept",
  "query": "CONCEPT:AUTH-1.0",
  "top_k": 1
}
```

**Example 3: Tiered Memory Search**
Search episodic or procedural memory for past agent actions.
```json
{
  "mode": "memory",
  "query": "How did we fix the database locking issue?",
  "top_k": 3
}
```

## 3. `mcp_graph-os_graph_query`

Execute read-only Cypher queries directly against the graph backend.

**Example 1: Fetch all capabilities**
```json
{
  "cypher": "MATCH (r:CallableResource) RETURN r.name AS tool, r.resource_type AS type LIMIT 10",
  "scope": "local"
}
```

**Example 2: Federated Query (External Graph)**
```json
{
  "cypher": "PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT ?name WHERE { ?person foaf:name ?name } LIMIT 10",
  "scope": "federated",
  "reference_id": "EXTERNAL-WIKIDATA-01"
}
```

## 4. `mcp_graph-os_graph_analyze`

Execute complex synthesis or comparative analysis across the Knowledge Graph.

**Example 1: Blast Radius Calculation**
Determine the impact of changing a specific component.
```json
{
  "action": "blast_radius",
  "node_id": "Code:auth.py",
  "depth": 2
}
```

**Example 2: Security Scan**
Scan a subgraph for environment variables or exposed keys.
```json
{
  "action": "security_scan",
  "target": "Codebase:frontend-app"
}
```

**Example 3: Deep Extract**
Synthesize an executive summary of a concept's implementation.
```json
{
  "action": "deep_extract",
  "target": "CONCEPT:ROUTING-3.4",
  "query": "Explain how the fallback logic works."
}
```

## 5. `mcp_graph-os_graph_write`

Mutate the Knowledge Graph by adding nodes, edges, or logging chat memory.

**Example 1: Log Chat Memory**
```json
{
  "action": "log_chat",
  "properties": "{\"user_prompt\": \"Add a login button\", \"resolution\": \"Button added in header.tsx\"}"
}
```

**Example 2: Bulk Ingest**
Push a batch of telemetry or traces directly into the graph.
```json
{
  "action": "bulk_ingest",
  "nodes": "[{\"id\": \"Trace-1\", \"label\": \"ExecutionTrace\"}]"
}
```

## 6. `mcp_graph-os_graph_orchestrate`

Dispatch subagents, handle consensus, and manage loop controls.

**Example 1: Dispatch Subagent**
```json
{
  "action": "dispatch",
  "agent_name": "repository-manager",
  "task": "Fix the linting errors in main.py",
  "max_steps": 5
}
```

**Example 2: Start Debate**
Run the TradingAgents swarm debate to vet financial hypotheses.
```json
{
  "action": "start_debate",
  "task": "Should we long AAPL given the latest earnings?"
}
```

## 7. `mcp_graph-os_graph_configure`

Manage backend configurations, system credentials, and tool registration.

**Example 1: Set Secret**
```json
{
  "action": "set_secret",
  "config_key": "API_KEY_OPENAI",
  "config_value": "{\"token\": \"sk-...\"}"
}
```

**Example 2: Register MCP**
Hot-load a new MCP server without restarting the engine.
```json
{
  "action": "register_mcp",
  "config_key": "custom-server",
  "config_value": "{\"command\": \"node\", \"args\": [\"server.js\"]}"
}
```
