# Capabilities — What an Agent Can Do

A concrete catalog of what agent-utilities grants an agent, each with a
copy-paste snippet in **library** form and the equivalent **MCP tool**. All tools
are exposed by the `graph-os` MCP server and mirrored 1:1 by the REST gateway
(`/api/...`) — see [Consumption Models](guides/consumption-models.md).

> Source of truth for the tool surface: `agent_utilities/mcp/kg_server.py`
> (the `graph_*`, `ontology_*`, `object_*` tools and `ACTION_TOOL_ROUTES`).
> The parity contract test (`tests/unit/test_gateway_mcp_parity.py`) keeps the
> MCP and REST surfaces in lockstep.

## The tool surface at a glance

**25 MCP tools**, every one with an action-routed REST twin:

| Group | Tools |
|---|---|
| Graph core (14) | `graph_query`, `graph_search`, `graph_write`, `graph_ingest`, `graph_analyze`, `graph_orchestrate`, `graph_configure`, `graph_context`, `graph_feedback`, `graph_goals`, `graph_hydrate`, `graph_message`, `graph_sessions`, `document_process` |
| Ontology (6) | `ontology_property_types`, `ontology_value_types`, `ontology_interface`, `ontology_function`, `ontology_derive`, `ontology_link_materialize` |
| Objects (4) | `object_edits`, `object_index`, `object_permissioning`, `object_set` |
| Connectors (1) | `source_connector` |

The REST gateway mounts the same surface under `/api`: the 25 action-routed
twins plus granular sub-routes (`/api/graph/write/node`, `/api/graph/ingest/jobs`,
`/api/sessions`, `/api/goals`, …), alongside the fleet supervisory plane
(`/api/fleet/*` — health/topology/pause/kill/approvals plus the correlation
queries `/api/fleet/trace` and `/api/fleet/touched`), a granular **typed**
OpenAPI surface for the ontology/object layer (`/api/ontology/value-types/{name}`,
`/api/ontology/interfaces/{name}`, `/api/objects/{id}`, `/api/objects/{id}/history`,
… — see `gateway/ontology_api.py`), the service dashboard (`/api/dashboard/*`,
including daemon status/shards), and Prometheus `GET /metrics`.

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

**Document → atomic-triple extraction (KG-2.64/2.65/2.66).** Turn a document,
URL, or pasted text into `(subject) -[predicate]-> (object)` fact edges carrying
evidence span, confidence, and tags — streamed live and deduped semantically.
Use `graph_ingest action=fact_extract` (inline) or the GPU-slot-scheduled job
actions `extract_submit` / `extract_jobs` / `extract_status` / `extract_pause` /
`extract_resume` / `extract_jsonl`. The same surface is exposed at
`/api/enhanced/extract/*` (SSE stream + JSONL) and rendered interactively in all
three frontends. See
[Document → KG Fact Extraction](architecture/document_fact_extraction.md).

Ingestion scales horizontally: the durable task queue is selectable via
`TASK_QUEUE_BACKEND` (`sqlite` | `postgres` | `kafka`, fail-loud when an
explicit backend is unreachable), Kafka `kg_tasks` messages are keyed for
per-tenant/per-repo ordering, and the `kg-ingest-worker` console script joins
the `kg-ingest` consumer group from any host. See
[Event Backbone — Ingest Task Queue Scale-Out](architecture/event_backbone_architecture.md).

## Orchestration: spawn teams & swarms

```python
# Decompose a goal into a coordinated team at runtime
await kg_server._execute_tool("graph_orchestrate", action="execute_agent",
    agent_name="researcher", task="Summarize Q3 incident trends")
```

`graph_orchestrate` actions: `dispatch`, `execute_agent`, `swarm`, `consensus`,
`start_debate`, `compile_workflow`, `compile_process`, `execute_workflow`,
`request_approval`, `grant_approval`, `submit_risk_veto`, `publish_proposal`.
Recursive nesting, circuit breakers, cognitive-scheduler quotas, and
blast-radius scoping are built in (pillar 1).

With `AGENT_DISPATCH_BACKEND=queue`, agent turns dispatch through a
session-partitioned durable queue (`AgentTurnEnvelope`) consumed by a stateless
`agent-dispatch-worker` fleet on any host — per-session serial execution,
crash-safe at-least-once claims, and worker placement visible at
`/api/fleet/topology`. See [Queue-Driven Agent Dispatch](architecture/agent_dispatch.md)
and the [queue-dispatch walkthrough](examples/queue-dispatch-walkthrough.md).

Descriptive BPMN process knowledge is executable too: `compile_process` lifts a
process definition into an executable plan via the `ProcessPlanCompiler`
(ORCH-1.41), gated by ontology validation on the execution path (AU-ORCH.execution.ontology-validation-execution-path),
with run lineage written back to close the descriptive↔executable loop
(ORCH-1.43). See the [ontology-to-workflow example](examples/ontology-to-workflow.md).

## Memory & autonomous goals

```python
# Launch a background goal loop (durable, resumable)
await kg_server._execute_tool("graph_goals", action="create",
    goal="Keep the incident KB current", session_id="ops-1")
```

`graph_goals`: `create`, `list`, `iterations`, `cancel`. `graph_sessions`:
`list`, `get`, `reply`, `cancel`. Sessions, turns, and goals are durable and
resumable across restarts — per-host SQLite by default, or one shared Postgres
state store for the whole fleet via `STATE_DB_URI` (AU-OS.state.unified-durable-state-externalization, see
[State Externalization](architecture/state_externalization.md)).

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

**Enterprise operations causal graph** (Codex X-2) is a separate, more specific
tool: `graph_ops_causal` (`join`/`root_cause`/`blast_radius`/`change_risk`/
`control_evidence` actions, `mcp/tools/ops_causal_tools.py`) joins the connector
fleet's own entities (Langfuse trace/generation → agent/tool/model → service →
deployment → commit/merge-request → incident/change → LeanIX capability →
policy/control/evidence) into one causal chain and reasons over it with the
existing causal-reasoning engine — distinct from `graph_analyze`'s
general-purpose `causal`/`blast_radius` actions above, which operate on
arbitrary graph structure rather than this specific ops entity chain. See the
`kg-ops-causal` skill.

## Coding & task management (harness)

Beyond the graph-os surface, the agent harness gives coding agents the machinery to
edit code reliably and drive long-horizon work:

- **Apply code edits** — `apply_edits(edits, root, fmt)` (`tools/developer_tools.py`)
  parses SEARCH/REPLACE blocks or unified diffs and applies them with fuzzy matching
  (so edits land despite whitespace drift) plus a reflection loop on failure. See
  [Edit-Application Engine](architecture/edit_application_engine.md) (CONCEPT:AU-ORCH.execution.robust-multi-format-edit).
- **PRD → task list** — the harness MCP server (`mcp/harness_server.py`) exposes
  `task_parse_prd`, `task_analyze_complexity`, `task_next`, `task_set_status`, and
  `task_scope`: decompose a PRD into a durable, dependency-aware task list, score
  complexity, and pull the next actionable task (cycle-validated). Backed by
  `SDDManager` (CONCEPT:AU-ORCH.planning.sdd-task-ergonomics).
- **Repo-map skeleton** — `POST /api/codemap` with `{"skeleton": true, "max_tokens": N}`
  returns a token-budgeted, importance-ranked code skeleton for context injection
  (CONCEPT:AU-ORCH.planning.repo-map-skeleton).

## Identity & multi-tenancy

Every gateway request passes through server-minted identity (OS-5.14): JWT
bearer tokens are validated and scoped into an `ActorContext`
(`agent_utilities/security/request_identity.py`), permission checks fail
closed, and engine connections authenticate with an HMAC shared secret. With
`KG_AUTH_REQUIRED` set, unauthenticated requests are rejected with 401. In
sharded deployments the ambient tenant also drives graph placement (see below).
Walkthrough: [identity & JWT example](examples/identity-jwt.md).

## Scale out: shards, workers, and one shared state store

When one host is not enough, every plane scales independently — all opt-in,
all byte-for-byte unchanged at defaults:

| Plane | Mechanism | Flag / entry point |
|---|---|---|
| KG engine | Tenant-sharded engines behind client-side HRW routing, per-shard reachability at `/api/dashboard/daemon/shards` | `GRAPH_SERVICE_ENDPOINTS` (2+ endpoints), `docker/engine-shards.compose.yml` |
| Ingestion | Kafka `kg_tasks` keyed partitions + `kg-ingest` consumer group | `TASK_QUEUE_BACKEND=kafka`, `kg-ingest-worker` |
| Agent execution | Session-keyed `agent_turns` queue + dispatch-worker fleet | `AGENT_DISPATCH_BACKEND=queue`, `agent-dispatch-worker` |
| Gateway | Pre-forked workers, per-tenant token-bucket rate limiting, engine circuit breaker | `GATEWAY_WORKERS`, `GATEWAY_RATE_LIMIT` |
| Durable state | One shared Postgres store with SKIP LOCKED queue claims + advisory-lock daemon leadership | `STATE_DB_URI` |

Deep dives: [Engine Sharding](architecture/engine_sharding.md) ·
[Gateway Scaling](architecture/gateway_scaling.md) ·
[State Externalization](architecture/state_externalization.md) ·
[sharding walkthrough](examples/sharding-walkthrough.md).

## Reactive supervision & fleet autonomy (Agent OS)

The REST gateway exposes a native **fleet supervisor** (no separate service):
`/api/fleet/health`, `/api/fleet/topology`, `/api/fleet/events`,
`/api/fleet/pause`, `/api/fleet/kill`, `/api/fleet/approvals` (+ `/grant`) —
per-domain error rates, live topology (including dispatch workers), monitoring
event ingress, blast-radius containment, and a mutation/risk approval queue.
See pillar 5 and the [agent-webui](ecosystem.md) Fleet Supervisor view.

On top of the supervisor sits an opt-in autonomy control plane: every
autonomous mutating action is gated by the **ActionPolicy** decision point
(`orchestration/action_policy.py`, fail-closed, audit-logged; policies in
`deploy/action-policy.default.yml`), driving the desired-state fleet
reconciler, remediation playbooks, health-gated deploy watch with rollback,
and the reactive autoscaler. See [Fleet Autonomy](architecture/fleet_autonomy.md),
[action-policy postures](examples/action-policy-postures.md), and
[autoscaling signals](examples/autoscaling-signals.md).

## Observability

With the optional `metrics` extra, the gateway serves Prometheus metrics at
`GET /metrics` (`agent_utilities_*` series: requests, latency, rate limiting,
engine calls/breaker state, engine shard reachability, ingest queue depth and
consumer lag — also the autoscaler's default signals — dispatch queue/turns/
workers, and MCP child health). Each Rust engine shard
exposes its own `epistemic_graph_*` series on its `--metrics-addr` listener.
Catalog: [metrics reference](reference/metrics.md) ·
[observability example](examples/observability.md).

## Expose your own tools as MCP

Any `agent-packages/agents/*` connector follows the same template
(`create_mcp_server()` in `agent_utilities/mcp/server_factory.py`) and can run
as a streamable-http container — see [Day-0](guides/day0.md) and the
[ecosystem map](ecosystem.md). The `mcp-multiplexer` that aggregates the fleet
is hardened per child: concurrency limits, session pools, restart-on-crash,
circuit breakers, and a `multiplexer_status` health tool (AU-ECO.mcp.profile-differences-from-client,
`agent_utilities/mcp/child_resilience.py`).

---

**Full runnable examples:** `examples/reference_agent/`
(`basic_agent.py`, `graph_agent.py`, `knowledge_graph_agent.py`, `mcp_agent.py`,
`memory_agent.py`, `protocol_agent.py`) and the operational walkthroughs under
[docs/examples/](examples/).
