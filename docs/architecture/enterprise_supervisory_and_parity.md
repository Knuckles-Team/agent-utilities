# Enterprise Parity, Supervisory Plane & Durable Execution

Architecture decisions for running the ecosystem as an AI-first enterprise at
scale: a single source of truth across the MCP and REST surfaces, a native
supervisory plane, crash-safe durable execution, and cross-agent trace
correlation. These records exist so the rationale survives as the surface grows.

## 1. Gateway ⇄ MCP parity over a shared dispatch (CONCEPT:ECO-4.0)

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
action-routed tools (context-window friendly); REST exposes the full surface,
with granular CRUD sub-routes layered on top for fine-grained HTTP clients. CRUD
is reachable today via each tool's `action` parameter; per-entity REST verbs
(`GET/POST/PUT/DELETE /ontology/interfaces/{id}`) are a mechanical follow-on.

**Enforcement.** `tests/unit/test_gateway_mcp_parity.py` asserts bidirectional
parity (every MCP tool has a mounted REST twin; no phantom routes), so the two
surfaces can never silently drift again.

## 2. Native swarm supervisory plane (CONCEPT:OS-5.10)

**Context.** Supervisory data already exists — MASS swarm-health P1–P4
(`graph/social_system.py`), per-agent circuit breakers, the durable session
registry, and request/grant approvals — but was scattered and unsurfaced. A
separate supervisor *service* would add operational complexity.

**Decision.** No new service. Expose a `/api/fleet/*` plane from the existing
gateway (`gateway/fleet.py`): per-domain health/error-rates, live topology,
whole-domain pause/kill (blast-radius containment reusing `core.sessions` cancel
mechanics), and the mutation/risk approval queue (read/grant via the
parity-covered `graph_query`/`graph_orchestrate` tools). The `agent-webui` Fleet
Supervisor view is the single pane of glass over it.

## 3. Durable execution on embedded SQLite (CONCEPT:ORCH-1.36)

**Context.** `DurableExecutionManager` persisted to an in-memory mock — it
survived nothing. The L1 epistemic_graph tier is a *cache* (rebuilt on restart),
so it is the wrong substrate for crash-survivable execution.

**Decision.** Persist checkpoints to an embedded, crash-safe SQLite store (the
same substrate `core.sessions` uses for recovery — no external infra).
`run_durable_action` provides **at-least-once** retries (via `ResiliencePolicy`)
and **exactly-once effects** via idempotency keys: a completed key short-circuits
re-execution and returns the recorded result, so a retry or crash-resume never
double-applies. Production may additionally mirror checkpoints into the KG for
lineage.

## 4. Cross-agent trace correlation (CONCEPT:OS-5.11)

**Context.** `@trace` nests spans within one process via contextvars, but a
multi-agent run had no shared key and side-effects carried no correlation — "which
agents touched record X?" was unanswerable across boundaries.

**Decision.** `observability/correlation.py` adds a run-wide correlation id,
W3C `traceparent` (de)serialization (`current_carrier`/`bind_carrier`) for
cross-process agent spawns, and `inject`/`extract` for outbound side-effect
headers (Kafka records, ServiceNow/connector calls). `engine.run_graph`
establishes the id at every entry point; the Langfuse exporter stamps it on every
trace so a run is one joinable story.

## Scale note (100k–100M agents)

These decisions make the control surface complete and correct, but extreme scale
needs a **distributed execution substrate**: multiple gateway workers, a durable
queue (Kafka is deployed), shared durable graph state (pggraph), and
lease/heartbeat work distribution. Durable execution (§3) is the first concrete
step; horizontal scale-out is tracked separately.
