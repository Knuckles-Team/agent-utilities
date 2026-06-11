# Capabilities — What an Agent Can Do

A concrete catalog of what agent-utilities grants an agent, each with a
copy-paste snippet in **library** form and the equivalent **MCP tool**. All tools
are exposed by the `graph-os` MCP server and mirrored 1:1 by the REST gateway
(`/api/...`) — see [Consumption Models](guides/consumption-models.md).

> Source of truth for the tool surface: `agent_utilities/mcp/kg_server.py`
> (the `graph_*`, `ontology_*`, `object_*` tools and `ACTION_TOOL_ROUTES`).

## Knowledge graph: store & recall

Add knowledge and query it back — no external DB required.

```python
from agent_utilities.mcp import kg_server

# Write a node
await kg_server._execute_tool("graph_write", action="add_node",
    node_id="svc:payments", node_type="Service",
    properties='{"team":"fintech","tier":"critical"}')

# Query it
res = await kg_server._execute_tool("graph_query",
    cypher="MATCH (n:Service) WHERE n.tier='critical' RETURN n")
```

| Capability | MCP tool | Action(s) |
|---|---|---|
| Write nodes/edges/memories | `graph_write` | `add_node`, `add_edge`, `store_memory`, `bulk_ingest` |
| Cypher / federated query | `graph_query` | `cypher`, `scope=federated` |
| Hybrid / semantic / analogy search | `graph_search` | `hybrid`, `concept`, `analogy`, `memory`, `discover` |
| Bitemporal "as of" recall | `graph_query` | `as_of=<ISO-8601>` |

## Ingestion: turn corpora into knowledge

```python
# Ingest a directory, repo, or document set into the KG
await kg_server._execute_tool("graph_ingest", action="ingest",
    path="./docs", content_type="auto")
```

`graph_ingest` actions: `ingest`, `corpus`, `jobs`, `job_status`, `distill`,
`agent_toolkit`, `ingest_knowledge_pack`. Documents become first-class
`Document`+`Chunk` ontology objects with OWL semantics.

## Orchestration: spawn teams & swarms

```python
# Decompose a goal into a coordinated team at runtime
await kg_server._execute_tool("graph_orchestrate", action="execute_agent",
    agent_name="researcher", task="Summarize Q3 incident trends")
```

`graph_orchestrate` actions: `dispatch`, `execute_agent`, `swarm`, `consensus`,
`start_debate`, `compile_workflow`, `execute_workflow`, `request_approval`,
`grant_approval`, `submit_risk_veto`. Recursive nesting, circuit breakers,
cognitive-scheduler quotas, and blast-radius scoping are built in (pillar 1).

## Memory & autonomous goals

```python
# Launch a background goal loop (durable, resumable)
await kg_server._execute_tool("graph_goals", action="create",
    goal="Keep the incident KB current", session_id="ops-1")
```

`graph_goals`: `create`, `list`, `iterations`, `cancel`. `graph_sessions`:
`list`, `get`, `reply`, `cancel`. Sessions/turns are durable (SQLite) and
resumable across restarts.

## Ontology (Palantir-Foundry parity)

```python
await kg_server._execute_tool("ontology_interface", action="list")
await kg_server._execute_tool("object_set", action="create",
    name="critical-services", filter='{"tier":"critical"}')
```

Tools: `ontology_interface`, `ontology_value_types`, `ontology_property_types`,
`ontology_derive`, `ontology_function`, `ontology_link_materialize`,
`object_edits`, `object_index`, `object_permissioning`, `object_set`,
`document_process`. Full object/link/function/action layer over the same engine
(pillar 2).

## Analysis & reasoning

`graph_analyze` actions: `synthesize`, `deep_extract`, `blast_radius`, `inspect`,
`causal`, `invariant`, `forecast`, `security_scan`, `evaluate`. Plus OWL/RDFS
forward-chaining reasoning in the Rust engine (`reason()` over the
`epistemic-graph` client).

## Reactive supervision (Agent OS)

The REST gateway exposes a native **fleet supervisor** (no separate service):
`/api/fleet/health`, `/api/fleet/topology`, `/api/fleet/pause`,
`/api/fleet/kill`, `/api/fleet/approvals` — per-domain error rates, live
topology, blast-radius containment, and a mutation/risk approval queue. See
pillar 5 and the [agent-webui](ecosystem.md) Fleet Supervisor view.

## Expose your own tools as MCP

Any `agent-packages/agents/*` connector follows the same template
(`create_mcp_server()` in `agent_utilities/mcp/server_factory.py`) and can run
as a streamable-http container — see [Day-0](guides/day0.md) and the
[ecosystem map](ecosystem.md).

---

**Full runnable examples:** `examples/reference_agent/`
(`basic_agent.py`, `graph_agent.py`, `knowledge_graph_agent.py`, `mcp_agent.py`,
`memory_agent.py`, `protocol_agent.py`).
