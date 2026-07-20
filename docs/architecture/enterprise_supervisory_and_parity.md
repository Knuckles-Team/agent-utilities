# Enterprise Parity, Supervisory Plane & Durable Execution

Architecture decisions for running the ecosystem as an AI-first enterprise at
scale: a single source of truth across the MCP and REST surfaces, a native
supervisory plane, crash-safe durable execution, and cross-agent trace
correlation. These records exist so the rationale survives as the surface grows.

```mermaid
flowchart TB
    subgraph clients["Consumers"]
        agent["Agents (MCP tools)"]
        http["HTTP / automation clients"]
        ui["agent-webui Fleet view"]
    end

    subgraph gw["Gateway (single process)"]
        mcp["graph-os MCP<br/>collapsed action tools"]
        rest["REST: collapsed twins<br/>(ACTION_TOOL_ROUTES)"]
        gran["REST: granular typed<br/>ontology/object GETs<br/>(ontology_api.py → OpenAPI)"]
        fleet["/api/fleet/* supervisory<br/>health · topology · pause/kill<br/>approvals · trace · touched"]
        dispatch["goal loop + dispatch worker"]
    end

    exec["_execute_tool()<br/><b>single source of truth</b>"]
    engine[("epistemic-graph engine<br/>the one authority")]
    corr["correlation_id stamped on<br/>FleetEvent + WorkItem records"]

    agent --> mcp
    http --> rest
    http --> gran
    ui --> fleet
    mcp --> exec
    rest --> exec
    gran --> exec
    fleet --> exec
    exec --> engine
    dispatch -->|"native WorkItem<br/>checkpoint · idempotency · fence"| engine
    dispatch --> exec
    fleet -.->|"trace?correlation_id<br/>touched?resource"| engine
    engine --- corr
```

## 1. Gateway ⇄ MCP parity over a shared dispatch (CONCEPT:AU-ECO.messaging.native-backend-abstraction)

**Context.** The GraphOS MCP tools and the gateway's REST routes both dispatch
through the same in-process `_execute_tool()` → `IntelligenceGraphEngine`
singleton (`agent_utilities/mcp/kg_server.py`). `_execute_tool` is therefore the
*de-facto service layer* — neither "the MCP" nor "the REST app" owns the logic.
Despite a docstring claiming the two never drift, ~17 MCP tools (the `ontology_*`,
`object_*`, `graph_context/feedback/hydrate/sessions/goals`, `document_process`,
`source_connector` surface) had **no REST route**.

**Decision.** Keep `_execute_tool` as the single source of truth. Maintain a
canonical `ACTION_TOOL_ROUTES` map (tool → collapsed REST path) and serve every
tool's REST twin from one factory (`_make_tool_endpoint`). MCP stays collapsed to
action-routed tools (context-window friendly); REST exposes the full surface.

**Granular typed surface (shipped).** Per-entity REST verbs are no longer a
follow-on — `gateway/ontology_api.py` mounts a typed FastAPI `APIRouter` of
resource-style reads over the ontology/object layer
(`GET /api/ontology/value-types/{name}`, `/ontology/interfaces/{name}`,
`/ontology/functions/{name}`, `/objects/{id}`, `/objects/{id}/history`,
`/objects/{id}/as-of`). Because it's an `APIRouter`, every route appears in
`/openapi.json` with documented params and a response envelope — the schema
enforcement the raw-Starlette collapsed routes lacked. Each handler is pure sugar:
it builds the `action` + params and dispatches through the **same**
`_execute_tool`, so there is no parallel implementation. Collapsed routes remain
for agents.

**Enforcement.** `tests/unit/test_gateway_mcp_parity.py` asserts bidirectional
parity (every MCP tool has a mounted REST twin; no phantom routes), so the
collapsed surfaces can never silently drift; the granular router is additive and
leaves that contract untouched.

## 2. Native swarm supervisory plane (CONCEPT:AU-OS.safety.ontological-guardrail)

**Context.** Supervisory data already exists — MASS swarm-health P1–P4
(`graph/social_system.py`), per-agent circuit breakers, the durable session
registry, and request/grant approvals — but was scattered and unsurfaced. A
separate supervisor *service* would add operational complexity.

**Decision.** No new service. Expose a `/api/fleet/*` plane from the existing
gateway (`gateway/fleet.py`): per-domain health/error-rates, live topology,
whole-domain pause/kill (blast-radius containment reusing `core.sessions` cancel
mechanics), the mutation/risk approval queue (read/grant via the parity-covered
`graph_query`/`graph_orchestrate` tools), and the correlation query endpoints
(`GET /api/fleet/trace?correlation_id=…`, `GET /api/fleet/touched?resource=…`,
see §4). The `agent-webui` Fleet Supervisor view is the single pane of glass over
it. Multi-agent containment + recovery are covered end-to-end by
`tests/integration/test_fleet_chaos.py` (domain pause contains only its domain;
concurrent goal loops honor pause with zero side effects).

## 3. Durable execution on native WorkItems (CONCEPT:AU-ORCH.execution.retry-predicate-raised-treating)

**Context.** A separate checkpoint database creates a second lifecycle next to
the engine's work authority. Its progress can diverge from the active lease,
allow a stale worker to advance, or report completion before the authoritative
terminal commit.

**Decision.** Keep lease, retry, idempotency, progress, and terminal outcome on
the same engine-native `WorkItem`. A live claimant persists an opaque
`checkpoint_id` only after renewing its engine-issued lease; the update is
atomically fenced on tenant, owner, lease epoch, fencing token, and current
status. Repeated matching terminal commits are idempotent. There is no SQLite or
Postgres checkpoint sidecar, and `STATE_DB_URI` does not select execution
checkpoint storage.

**Wired into the live path (shipped).** `LoopController.run_loop` resumes from
the WorkItem's last fenced `checkpoint:iteration:<n>` reference after an expired
lease is reclaimed. Agent dispatch similarly uses the turn's deterministic
WorkItem and acknowledges queue delivery only after its fenced native commit.
A crash before acknowledgement is therefore a safe redelivery; a stale worker
cannot overwrite progress or outcome from the newer lease.

Covered by `tests/unit/orchestration/test_work_item.py`,
`tests/unit/knowledge_graph/test_loop_work_item_checkpoint.py`, and
`tests/unit/test_agent_dispatch.py`.

## 4. Cross-agent trace correlation (CONCEPT:AU-OS.observability.run-wide-correlation-id)

**Context.** `@trace` nests spans within one process via contextvars, but a
multi-agent run had no shared key and side-effects carried no correlation — "which
agents touched record X?" was unanswerable across boundaries.

**Decision.** `observability/correlation.py` adds a run-wide correlation id,
W3C `traceparent` (de)serialization (`current_carrier`/`bind_carrier`) for
cross-process agent spawns, and `inject`/`extract` for outbound side-effect
headers (Kafka records, ServiceNow/connector calls). `engine.run_graph`
establishes the id at every entry point; the Langfuse exporter stamps it on every
trace so a run is one joinable story.

**Queryable from the graph (shipped).** Emission alone left "who touched X?"
answerable only from Langfuse / ad-hoc Cypher. The correlation id (plus
actor/tenant) is now **persisted** onto the durable effect nodes — `FleetEvent`
(`gateway/fleet_events.py persist_event`) and executed `Task` nodes (the dispatch
worker) — and read back through two supervisory routes: `GET /api/fleet/trace?
correlation_id=…` returns every node stamped with a run's id, and
`GET /api/fleet/touched?resource=…` returns the events + originating actors that
touched a resource (blast-radius). Covered by `tests/unit/test_fleet_correlation.py`.

## Scale note (100k–100M agents)

These decisions make the control surface complete and correct, but extreme scale
needs a **distributed execution substrate**: multiple gateway workers, a durable
queue (Kafka is deployed), shared durable graph state (pg-age), and
lease/heartbeat work distribution. Durable execution (§3) is now wired into the
live goal-loop and dispatch paths and, with `state_db_uri`, shares Postgres
checkpoints across hosts — the concrete first step; broader horizontal scale-out
is tracked separately.

## Wire contract integrity (CONCEPT:EG-KG.query.wire-protocol)

The gateway/engine boundary is out-of-process MessagePack over UDS/TCP — there is
**no PyO3 / FFI** — so the Python client mirrors the Rust `Method` enum by
sending variant names as strings. Nothing generated or type-checked that, so a
renamed/removed engine op or a client typo could drift silently. The
epistemic-graph repo now ships a CI gate (`tests/test_protocol_parity.py`, run in
`rust-ci.yml`): it parses the `Method` enum (165 variants) and the client's
`_send(...)` calls, asserts no client method lacks a Rust variant, and ratchets
the set of unbound variants against a committed baseline so a new engine op forces
a conscious client binding.
