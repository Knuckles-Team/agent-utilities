# Graph-OS MCP Server Examples

The `graph-os` MCP server is the unified entrypoint for interacting with the Intelligence Graph Engine. It exposes the core tools that allow LLMs and automated pipelines to manipulate knowledge, orchestrate agents, and manage the workspace: the 9 documented below plus `graph_sessions` (durable session management), `graph_goals` (autonomous background loops), and `graph_hydrate` (instant KG hydration from external connectors).

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
  "query": "CONCEPT:AU-OS.identity.auth-flow",
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
  "target": "CONCEPT:AU-ORCH.routing.fallback-logic",
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

**Example 3: Spawn an agent with a curated handoff (CONCEPT:AU-ORCH.session.invoker-agent-handoff/1.39)**
Hand the spawned agent budgeted context, a least-privilege tool allow-list, a credential
*reference* (resolved to a token at spawn, never logged), and open a message channel. The response
is `{"output", "mermaid", "channel_id"}`.
```json
{
  "action": "execute_agent",
  "agent_name": "github-mcp",
  "task": "List the latest workflow run for Knuckles-Team/agent-utilities",
  "context_ref": "ctx:sess-42:brief",
  "allowed_tools": "list_workflow_runs,get_workflow_run",
  "cred_ref": "cred:sess-42",
  "open_channel": true
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

## 8. `mcp_graph-os_graph_context`

**CONCEPT:AU-ORCH.session.invoker-agent-handoff** — store/fetch curated context for an invoker→spawned-agent handoff,
persisted in the epistemic-graph so a *separately*-spawned agent can read it by id. Session-anchored
(AU-ORCH.session.session-anchored-collections-native): `list` is a reliable single-hop traversal from the `Session` node, isolated per session.

**Example 1: Put a context blob**
```json
{
  "action": "put",
  "session_id": "sess-42",
  "key": "brief",
  "content": "Deployment target is R820. Use the 9B model. Read-only task."
}
```
Returns `{"context_id": "ctx:sess-42:brief", "session_id": "sess-42"}`. Pass that `context_id` to
`graph_orchestrate(action="execute_agent", context_ref="ctx:sess-42:brief")`.

**Example 2: List all context for a session**
```json
{ "action": "list", "session_id": "sess-42" }
```

**Example 3: Fetch one blob's full content by id**
```json
{ "action": "get", "context_id": "ctx:sess-42:brief" }
```

**Example 4: Put an ephemeral blob with a TTL**
```json
{ "action": "put", "session_id": "sess-42", "key": "scratch", "content": "transient note", "ttl_s": 600 }
```

## 9. `mcp_graph-os_graph_message`

**CONCEPT:AU-ORCH.session.session-anchored-collections-native** — a bidirectional, cross-process, ordered message channel between an invoking
agent and a spawned agent, over the engine's native Communication Channels (KG-2.0). The channel id
is deterministic: `orch:{session_id}:{run_id}`.

**Example 1: Open a channel**
```json
{ "action": "open", "session_id": "sess-42", "run_id": "run-abc1" }
```
Returns `{"channel_id": "orch:sess-42:run-abc1"}`. (Or pass `open_channel=true` to
`graph_orchestrate(action="execute_agent")` and read `channel_id` from its response.)

**Example 2: Send a message (the sender auto-joins)**
```json
{ "action": "send", "channel_id": "orch:sess-42:run-abc1", "sender": "invoker", "payload": "proceed with the task" }
```

**Example 3: Send a durable message (survives engine restart)**
```json
{ "action": "send", "channel_id": "orch:sess-42:run-abc1", "sender": "invoker", "payload": "final instruction", "durable": true }
```

**Example 4: Receive new messages with a cursor**
`since` is the count already consumed; the response returns `{"messages", "cursor"}` — pass the
returned `cursor` as `since` on the next poll.
```json
{ "action": "receive", "channel_id": "orch:sess-42:run-abc1", "since": 0 }
```

**Example 5: Replay the durable history (id-anchored, restart-safe)**
```json
{ "action": "history", "channel_id": "orch:sess-42:run-abc1" }
```

**Example 6: Close the channel**
```json
{ "action": "close", "channel_id": "orch:sess-42:run-abc1" }
```
