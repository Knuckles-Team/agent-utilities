# Architecture

## Core Architecture Diagram

<div class="admonition architecture" markdown>
<p class="admonition-title">Clients in, through the unified execution layer, to specialists — with a human-in-the-loop escape</p>

**Entry.** A user's request (plus images) reaches agent-webui,
agent-terminal-ui, an external AG-UI client, or an ACP-compatible editor.
The three AG-UI-speaking clients call the agent-utilities server's
`/ag-ui` route directly; the ACP editor talks stdio JSON-RPC to
`agent-utilities-acp`, which reaches the Unified Execution Layer
(`graph/protocol_agnostic_execution.py`) directly. The server also exposes
AG-UI, A2A, ACP, and SSE protocol adapters, all of which converge on the
same Unified Execution Layer — no protocol gets its own execution path.

**Core.** The Unified Execution Layer drives a Pydantic Graph Agent, which
reads the Intelligence Graph Engine. The engine backs three things: MAGMA
(orthogonal semantic/temporal/causal/entity views), Autonomous
Self-Improvement (outcome rewards, textual-gradient critiques, prompt/skill
evolution), and the GraphBackend itself (epistemic-graph authority plus an
optional pg-age mirror).

**Discovery and dispatch.** The Unified Discovery Layer
(`core/config.py`) reads the Knowledge Graph's specialist registry via
`get_discovery_registry()`, producing an `MCPAgentRegistryModel` roster
that feeds the graph agent, which dispatches to Specialist Superstates,
which reach MCP servers and Universal Skills/Skill Graphs.

**Human-in-the-loop.** When an MCP tool needs approval, `tool_guard`
returns `DeferredToolRequests` to `ApprovalManager`, which awaits an
`asyncio.Future` via an event queue; an SSE sideband event reaches the
server, which emits an `approval_required` event to whichever UI (webui or
terminal-ui) is attached. The UI's `POST /api/approve` reaches the server,
which resolves the waiting future back in `ApprovalManager` — resuming
execution. A tool can also elicit directly via `ctx.elicit`, reaching
`global_elicitation_callback`, which queues the same future-based wait.
</div>

## Protocol Layer Architecture

The framework provides three canonical protocol adapters:

1. **ACP (Agent Client Protocol)**: Editor-launched stdio JSON-RPC for coding-agent sessions
2. **A2A (Agent-to-Agent)**: Peer-to-peer agent communication and coordination
3. **AG-UI**: Current streaming interface for native Pydantic AI clients

All protocol adapters are centralized in `agent_utilities/protocols/`:

- `acp_adapter.py`: Harness ACP stdio boundary, durable sessions, and Graph-OS session dependencies
- `a2a.py`: A2A peer discovery, JSON-RPC client, registry management
- `a2a_epistemic.py`: durable task/context state, idempotent dispatch recovery,
  delivery leases, and execution fencing
- `agui_emitter.py`: AG-UI wire format translator for direct graph execution events
- Server endpoints: `/a2a` (MOUNT), `/ag-ui` (POST); ACP is the separate `agent-utilities-acp` process

### Direct Graph Execution (Fast Path)

When a `graph_bundle` is present, the **AG-UI endpoint** invokes the
protocol-agnostic execution authority directly:

```
User Query → /ag-ui → graph.iter() → [step events] → AGUIGraphEmitter → wire format
```

This eliminates one full LLM inference round-trip per request. The fast path uses `graph.iter()` (pydantic-graph beta API) for step-by-step execution, yielding per-node events that are translated to AG-UI wire format by `AGUIGraphEmitter`.

The graph bundle must contain a real Graph object with `.iter()` support. There
is no alternate graph execution flag or model-mediated graph path.

The **ACP adapter** uses Harness `AcpSessionConfig` to bind immutable,
per-session graph context to one shared wrapper agent. Editors launch it over
stdio; it is never mounted as an HTTP application.

The **A2A path** is graph-native
(CONCEPT:AU-ECO.messaging.native-backend-abstraction): when a `graph_bundle` is
present, `PlannerGraphSkill` is registered as an A2A skill and calls the same
execution authority without an outer LLM orchestration hop.

The `/a2a` server uses one native epistemic-graph persistence contract. A task is
stored before its operation is idempotently published; a bounded background scan
repairs the persist/publish crash window. Workers heartbeat the engine-owned
visibility lease and acknowledge by the current delivery tag plus consumer.
Task and context updates carry revision and tenant-keyed digest fences, while
cancellation and completion race through durable compare-and-set transitions.
Only one terminal outcome can commit, and terminal context plus task output land
atomically. Payloads are bounded before projection and permit opaque governed
content references instead of inline file data or deployment-specific locations.

### 3-Stage Hybrid Routing (CONCEPT:AU-ORCH.adapter.hot-cache-invalidation, CONCEPT:AU-AHE.evaluation.interpretability-tests, CONCEPT:AU-KG.memory.tiered-memory-caching)

The router implements a cascading 3-stage routing strategy that avoids unnecessary LLM inference:

- **Stage 1: TeamConfig Match** (CONCEPT:AU-AHE.evaluation.interpretability-tests)
    - Check KG for a proven specialist coalition matching the query pattern
    - If found → skip LLM, dispatch the team directly
- **Stage 2: Self-Model Bias** (CONCEPT:AU-KG.memory.tiered-memory-caching)
    - Inject domain proficiency scores into the specialist prompt
    - High-proficiency domains are weighted higher in LLM selection
- **Stage 3: LLM Planning** (filtered via CONCEPT:AU-ORCH.adapter.hot-cache-invalidation)
    - Registry Hot Cache provides only top-7 relevant specialists
    - LLM sees a focused prompt instead of 50+ specialist descriptions

This strategy means the system progressively learns: the more queries it handles, the more TeamConfigs accumulate, and the fewer LLM planning round-trips are needed.

### Authentication Passthrough (`custom_headers`)

`create_agent_server()` and `create_graph_agent_server()` accept a generic `custom_headers: dict[str, Any] | None = None` kwarg that is propagated verbatim to the LLM HTTP client as request headers. agent-utilities itself is **auth-agnostic** -- it does not ship provider-specific auth code (OIDC, client-credentials flows, bearer-token fetchers, etc.) and has no opinion about where those headers come from. Downstream packages are free to populate the dict from any source: environment variables, a token-fetching library, static config, a secret manager, or a callable that refreshes on every run. TLS trust is selected independently through mandatory-verification runtime profiles.

## Graph Orchestration Architecture

<div class="admonition architecture" markdown>
<p class="admonition-title">Query to response: guard, route, discover, dispatch, verify, synthesize</p>

A user query (+ images) enters the unified ACP/AG-UI/SSE protocol layer,
then a usage guard (rate limiting): blocked requests end immediately;
allowed requests reach the router, which picks a topology — a trivial
query ends directly, a full-pipeline query reaches the dispatcher. On its
first entry the dispatcher retrieves memory context, then loops back to
itself.

**Discovery phase.** The dispatcher can send a "research first" query to
three parallel participants — a Researcher (web-search/crawler/fetch,
project search, workspace file read), an Architect (C4 architecture, spec
generation, product strategy, user research, brainstorming), and the
Unified Registry (sourced from the Knowledge Graph) — which barrier-sync
at a Research Joiner and return coalesced context to the dispatcher.

**Execution phase.** The dispatcher can also parallel-dispatch to three
specialist groups: **Programmers** (Python, TypeScript, Go, Rust, C, C++,
JavaScript — each with its own language-specific skills/docs/tools),
**Infrastructure** (DevOps, Cloud, Database), and **Specialized & Quality**
(Security, QA, UI/UX, Debugger). All three groups barrier-sync at an
Execution Joiner, which returns implementation results to the dispatcher.
The dispatcher can also route to a Council for multi-perspective
deliberation, which also feeds the Execution Joiner.

**Verification and response.** Once the dispatcher considers the plan
complete, a Verifier scores it: ≥0.7 passes to the Synthesizer for the
final response; below 0.7 fails back to the dispatcher, which can re-plan
(via the Planner) or, on terminal failure, end the run without a response.

**Spec-Driven Development, a parallel lifecycle:** Constitution
(governance) → Specification → Technical Plan
(`ImplementationPlan`) → Tasks → Execution (parallel dispatch) →
Verification (spec audit).
</div>

> **Note:** MCP ecosystem agents (AdGuard, Jellyfin, Ansible Tower, etc.) are dynamically spawned as `CallableResource` nodes in the Knowledge Graph. They are discovered at runtime from `mcp_config.json` and do not appear in this static diagram.
>
> ### Unified Toolkit Ingestion (CONCEPT:AU-ECO.messaging.native-backend-abstraction)
>
> <div class="admonition architecture" markdown>
> <p class="admonition-title">Three source shapes, auto-detected, converge on one KG insert</p>
>
> Sources (`mcp_config.json`, `SKILL.md` directories, A2A URLs) are
> auto-detected by shape: an `mcp_config` extracts servers & flags, then
> runs live tool discovery (`list_tools`); a `skill_directory` parses
> frontmatter directly; an `a2a_url` fetches `/.well-known/agent.json`
> directly. All three paths converge on inserting a `CallableResource` into
> the Knowledge Graph. This pipeline allows single-shot ingestion of all
> agent capabilities, bridging the gap between isolated codebases and the
> unified Knowledge Graph.
> </div>

### Council Deliberation Node

The **Council** is a specialized graph node that implements Karpathy's LLM Council pattern for high-stakes decision-making. It provides a 4-stage deliberative pipeline:

<div class="admonition architecture" markdown>
<p class="admonition-title">Five perspectives, anonymized, reviewed, chaired to one verdict</p>

A query fans out to five distinct perspective agents — Contrarian, First
Principles, Expansionist, Outsider, and Executor — whose outputs are all
anonymized before three independent reviewers see them (so no reviewer
can bias on which perspective said what). The three reviews converge on a
Chairman, who produces the final `CouncilVerdict`.
</div>

| Stage | Purpose | Implementation |
|-------|---------|---------------|
| **1. Advisors** | 5 parallel agents with distinct thinking styles | `run_orthogonal_regions` / sequential dispatch |
| **2. Anonymize** | Shuffle identities behind labels (A-E) | Pure Python, zero LLM cost |
| **3. Peer Review** | 3 reviewers rank, critique, find collective gaps | Independent reviewer agents |
| **4. Chairman** | Synthesize into structured `CouncilVerdict` | `output_type=CouncilVerdict` |

**Key features:**
- **Hybrid model routing**: Uses `ModelRegistry` to assign different real LLM models to different advisor roles
- **Generalized transcripts**: `AgentTranscript` and `render_agent_transcript_markdown()` work for any agent output, not just council
- **KG persistence**: Verdicts are stored as `DecisionNode` entries for future reference
- **Trigger modes**: Auto-routed by the Router, keyword-triggered ("council this"), or invocable as a tool

## Package Structure

With the recent modularization, `agent-utilities` has been restructured to cleanly separate routing, protocols, execution, and discovery mechanisms into isolated domains.

| Directory | Purpose | Key Modules |
|---|---|---|
| `core/` | Foundational primitives, exceptions, and decorators. | `workspace.py`, `exceptions.py`, `decorators.py` |
| `agent/` | Bootstrapping and configuring agent ecosystems from `workspace.yml` and CLI. | `factory.py`, `registry_builder.py` |
| `protocols/` | Interface adapters connecting outer HTTP/RPC boundaries to inner graphs. | `acp_adapter.py`, `a2a.py`, `agui_emitter.py` |
| `graph/` | The core Pydantic-Graph routing and orchestration machinery. | `protocol_agnostic_execution.py`, `steps.py`, `executor.py`, `routing/`, `planning/` |
| `mcp/` | Specific wrappers for `fastmcp` to normalize tool discovery and error handling. | `server_factory.py`, `context_helpers.py`, `agent_manager.py` |
| `security/` | Centralized identity verification, JWT validation, and API authentication. | `auth.py`, `browser_auth.py` |
| `prompts/` | Version-controlled JSON schema blueprints that replace unstructured text prompts. | `*.json` |
| `knowledge_graph/`| The unified semantic and structural memory backbone over the layered `GraphBackend` interface. | `facade.py`, `core/engine.py`, `core/maintainer.py`, `retrieval/hybrid_retriever.py` |
| `harness/` | Agentic Harness Engineering (AHE) tools for execution observability and prompt evaluation. | `verifier.py`, `trace_backend.py`, `evolve_agent.py` |
| `rlm/` | Recursive Language Model handlers for autonomous sub-shells and self-prompting loops. | `repl.py` |
| `sdd/` | Spec-Driven Development pipelines decomposing `.specify` files into actionable graphs. | `orchestrator.py` |
| `server/` | FastAPI applications hosting all HTTP, ACP, and SSE routes. | `app.py`, `routers/` |
| `gateway/` | Homepage-style service dashboard data layer (CONCEPT:AU-OS.config.gateway-service-dashboard). 50 widget types, aggregator, REST+WS API. Synthesized from former `service-dashboard-core`. | `models.py`, `registry.py`, `config.py`, `aggregator.py`, `api.py`, `ws.py`, `widgets/` |

## Hierarchical State Machine (HSM) Architecture

The graph orchestration system is a **Hierarchical State Machine**. It follows the same formal model used in robotics, game engines, UML statecharts, and SCXML workflow engines.

### HSM Level Mapping
- **Level 0: Root Graph** (N orchestration nodes)
    - `usage_guard` → `router` → `dispatcher` → `memory_selection` → `dispatcher`
    - `researcher`, `architect`, `verifier` (discovery/validation)
    - `parallel_batch_processor` → `expert_executor` (fan-out)
    - `research_joiner`, `execution_joiner` (fan-in)
    - `verifier` → `synthesizer` → END (quality gate + response composition)
    - `planner` (re-planning on verification failure)
- **Level 1: Superstates — Specialist Agents**
    - Specialist Roster (dynamically discovered from the **Knowledge Graph**) — each loads a
      name-matched prompt + discovered capabilities + mapped MCP toolsets; supports `prompt`
      (local), `mcp` (stdio), and `a2a` (remote) agent types
    - Unified Execution: dynamic routing based on registry-provided metadata
- **Level 2: Substates — Agent Internal Loop**
    - `Pydantic AI Agent.run()` = `UserPromptNode` → `ModelRequestNode` → `CallToolsNode` →
      ... (multi-turn tool iteration, max 3 iterations per specialist)
- **Level 3: Leaf States — MCP Tool Execution**
    - Each tool call invokes an MCP server subprocess via stdio/HTTP — atomic operations like
      `get_project()`, `list_branches()`, `run_cypher_query()`, etc.

### Concept Mapping
| agent-utilities Concept        | HSM Concept            | Details                                           |
|--------------------------------|------------------------|---------------------------------------------------|
| Root graph                     | Root state machine     | N Orchestration nodes                             |
| Router -> Dispatcher            | Top-level transitions  | Router generates plan, dispatcher executes        |
| Planner (re-plan only)         | Re-entry transition    | Invoked by verifier on score < 0.4                |
| Synthesizer                    | Terminal action        | Composes final response from the results          |
| `NODE_SKILL_MAP` agents        | Superstates (L1)       | N hardcoded domains                               |
| Dynamic agents (unified)       | Superstates (L1)       | N from `discover_all_specialists()` (MCP + A2A)   |
| `_execute_specialized_step()`  | Enter superstate       | Loads prompt + skills + deduplicated MCP toolsets |
| `Agent.run()` internal loop    | Substates (L2)         | Model request/tool cycles                         |
| MCP tool call (stdio)          | Leaf states (L3)       | Atomic operations                                 |
| Verifier feedback loop         | Re-entry transition    | Parent re-dispatches to child                     |
| Circuit breaker (open)         | Guard condition        | Blocks entry to failed state                      |
| `node_transitions` guard       | Watchdog timer         | Force-terminates after 50 transitions             |
| Memory-first dispatch          | Entry action           | Enriches context before first step                |
| Research-before-execution      | Phase ordering         | Discovery completes before execution              |
| Process-Guided Planning        | Knowledge Influx       | KG-native SOPs injected into Planner context      |
| Policy Guardrails              | Transition Guard       | Policies enforce constraints at state boundaries  |

### HSM Design Principles
1. **Treat subgraphs as macro-states.** A specialist should behave as a single opaque state to the dispatcher. Define clear input/output contracts.
2. **Scale horizontally, not vertically.** Add new subgraphs (new MCP servers, new agent packages) instead of adding nodes to existing graphs.
3. **Plan enhancements by level.** Routing concern -> L0. Domain behavior -> L1 specialist. Tool-level fix -> L3 MCP.
4. **Use types as boundaries.** `ExecutionStep`, `GraphPlan`, `GraphResponse`, and `MCPAgent` are the boundary contracts between levels.
5. **Defer flattening.** Never visualize the full system as one graph. Visualize one level at a time.
6. **The growth test:** If tempted to add more nodes to a graph, ask whether you should add a new state machine instead.

### Behavior Tree (BT) Concepts
The graph incorporates key Behavior Tree patterns **inside** the HSM structure.

| agent-utilities Concept | BT Concept | Details |
|---|---|---|
| `_attempt_specialist_fallback`, `static_route_query` | Selector (priority/fallback) | Specialist fallback chain, static route before LLM |
| `dispatcher_step`, `assert_state_valid` | Sequence (fail-fast) | Plan step execution with cursor |
| `_execute_dynamic_mcp_agent`, `expert_executor_step` | Retry decorator | Tool-level retries with exponential backoff |
| `asyncio.wait_for()` in specialist execution | Timeout decorator | Per-node timeout via `ExecutionStep.timeout` |
| `check_specialist_preconditions` | Precondition guard | Check server health before entering specialist |
| `assert_state_valid()` | Boundary re-evaluation | State invariants at dispatcher and verifier boundaries |

**Design rule:** If logic chooses between options -> BT concept. If logic defines long-lived phases -> HSM concept.

## Server Endpoint Reference

| Endpoint | Method | Tag | Description |
|---|---|---|---|
| `/health` | GET | Core | Status-only, non-fingerprinting liveness probe |
| `/ag-ui` | POST | Agent UI | AG-UI streaming endpoint with sideband graph events |
| `/stream` | POST | Agent UI | Generic SSE stream endpoint for graph agent execution |
| `/a2a` | MOUNT | A2A | Agent-to-Agent (fastA2A) JSON-RPC endpoint |
| `/api/approve` | POST | Human-in-the-Loop | Resolves pending tool approvals and MCP elicitation requests |
| `/chats` | GET | Core | List all stored chat sessions |
| `/chats/{chat_id}` | GET | Core | Get full message history for a specific chat |
| `/chats/{chat_id}` | DELETE | Core | Delete a specific chat session |
| `/mcp/config` | GET | Interoperability | Return the current MCP server configuration |
| `/mcp/tools` | GET | Interoperability | List all tools from connected MCP servers |
| `/mcp/reload` | POST | Interoperability | Hot-reload MCP servers and rebuild graph |
| `/api/dashboard/layout` | GET/PUT | Dashboard (AU-OS.config.gateway-service-dashboard) | Get or save dashboard service layout |
| `/api/dashboard/data` | GET | Dashboard (AU-OS.config.gateway-service-dashboard) | Fetch all widget data from 50 services |
| `/api/dashboard/data/{id}` | GET | Dashboard (AU-OS.config.gateway-service-dashboard) | Fetch single service widget data |
| `/api/dashboard/full` | GET | Dashboard (AU-OS.config.gateway-service-dashboard) | Layout + data in single request (initial load) |
| `/api/dashboard/widgets` | GET | Dashboard (AU-OS.config.gateway-service-dashboard) | List available widget types |
| `/api/dashboard/health` | GET | Dashboard (AU-OS.config.gateway-service-dashboard) | Health check across all services |
| `/api/dashboard/discover` | GET | Dashboard (AU-OS.config.gateway-service-dashboard) | Auto-discover services from mcp_config |
| `/ws/dashboard` | WS | Dashboard (AU-OS.config.gateway-service-dashboard) | Real-time streaming updates |

## The Complete Execution Journey

### Phase 1: Ingress & Protocol Handling
1. **Entry**: A user query arrives via AG-UI (`/ag-ui`), SSE (`/stream`),
   REST (`/api/chat`), A2A (`/a2a`), or the separate ACP stdio process.
2. **Direct Dispatch Check**: If a `graph_bundle` is present, AG-UI routes directly to `execute_graph_iter()`.
3. **Unified Execution**: All protocols funnel through the same graph engine via `graph/protocol_agnostic_execution.py`. The `execute_graph_iter()` entry point uses `graph.iter()` for step-by-step control.
4. **State Initialization**: A fresh `GraphState` is initialized with the synthesized `query_parts`.

### Phase 2: Safety & Policy Enforcement
4. **Usage Guard**: The `usage_guard_step` checks session's token usage and estimated cost against safety limits.
5. **Policy Check**: If enabled, a lightweight LLM check validates the query against security policies.

### Phase 3: Routing & Planning
6. **Fast-Path Check**: Trivial or conversational queries are answered directly, bypassing the full graph pipeline.
7. **Routing**: The `router_step` analyzes the multi-modal intent and generates a `GraphPlan`.
8. **Infinite-Loop Guard**: A `node_transitions` counter (max 50) prevents runaway graph execution.

### Phase 4: Context Enrichment & Dispatch
9. **Memory Selection**: On first entry, the `dispatcher` routes to `memory_selection_step` for RAG-style context injection.
10. **Research-Before-Execution**: The dispatcher reorders the plan to guarantee research steps execute before specialist steps.
11. **Dispatch**: The `dispatcher` spawns selected specialist nodes with concurrent execution via `parallel_batch_processor`.

### Phase 5: Parallel Execution
12. **Specialist Loop**: Each specialist enters a high-fidelity `Agent.run()` loop with dedicated system prompts, domain-specific toolsets, and original multi-modal query parts.
13. **Convergence**: Results are coalesced at the `execution_joiner` and written to the `results_registry`.

### Phase 6: Verification & Synthesis
14. **Verification**: The `verifier_step` compares results against user intent using a `ValidationResult` score (0.0-1.0).
15. **Feedback Loop**: Score 0.4-0.7 -> re-dispatch same plan with feedback. Score < 0.4 -> full re-plan via `planner_step`.
16. **Synthesis**: Once validated (score >= 0.7), the `synthesizer_step` composes the final markdown response.
17. **Memory Persistence**: Execution metadata is persisted to the Knowledge Graph as a `historical_execution` memory.
