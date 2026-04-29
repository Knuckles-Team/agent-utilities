# Architecture

## Core Architecture Diagram

```mermaid
graph TD
    User([User Request + Images]) --> WebUI[agent-webui]
    User --> TUI[agent-terminal-ui]
    WebUI -- ACP Protocol /acp --> Backend[agent-utilities Server]
    TUI -- AG-UI /ag-ui --> Backend
    TUI -- ACP Protocol /acp --> Backend
    External[External AG-UI Client] -- Legacy Protocol /ag-ui --> Backend

    subgraph AgentUtilities [agent-utilities]
        Backend --> UnifiedExec["Unified Execution Layer<br/>(graph/unified.py)"]
        UnifiedExec --> Graph[Pydantic Graph Agent]
        Graph --> KG[Intelligence Graph Engine]

        subgraph MemoryArchitecture [Autonomous Memory Architecture]
            KG --> MAGMA[MAGMA: Orthogonal Views]
            KG --> Lightning[Agent Lightning: Self-Improvement]
            KG --> UnifiedDB[(Unified Graph DB: Ladybug/Neo4j)]

            MAGMA --> Semantic[Semantic View]
            MAGMA --> Temporal[Temporal View]
            MAGMA --> Causal[Causal View]
            MAGMA --> Entity[Entity View]

            Lightning --> Rewards[Outcome Rewards]
            Lightning --> Critiques[Textual Gradients]
            Lightning --> Evolution[Prompt/Skill Evolution]
        end

        subgraph ProtocolAdapters [Protocol Adapters]
            AGUI_Adapter[AG-UI Adapter]
            ACP_Adapter["ACP Adapter<br/>(graph-backed)"]
            SSE_Adapter[SSE Stream]
            A2A_Adapter[A2A Adapter]
        end

        Backend --> AGUI_Adapter
        Backend --> A2A_Adapter
        Backend --> ACP_Adapter
        Backend --> SSE_Adapter
        AGUI_Adapter --> UnifiedExec
        A2A_Adapter --> UnifiedExec
        ACP_Adapter --> UnifiedExec
        SSE_Adapter --> UnifiedExec

        subgraph UnifiedDiscovery ["Unified Discovery Layer (config_helpers.py)"]
            DAL["get_discovery_registry()"]
            KG_Registry["<b>Knowledge Graph</b><br/><i>(Unified Specialist Registry)</i>"]
            KG_Registry --> DAL
            DAL --> DSRoster["MCPAgentRegistryModel"]
        end

        DSRoster --> Graph
        Graph --> Specialists[Specialist Superstates]
        Specialists --> MCP[MCP Servers]
        Specialists --> Skills[Universal Skills, Skill Graphs]

        subgraph ElicitationFlow [Human-in-the-Loop Flow]
            MCP -- 1. Tool needs approval --> TG[tool_guard: requires_approval]
            TG -- 2. DeferredToolRequests --> AM[ApprovalManager]
            AM -- 3. asyncio.Future await --> EQ[Event Queue]
            EQ -- 4. SSE sideband event --> Backend
            MCP -- 1b. ctx.elicit --> GEC[global_elicitation_callback]
            GEC -- 2b. Queue + Future --> AM
        end
    end

    Backend -- 4. approval_required event --> WebUI
    Backend -- 4. approval_required event --> TUI
    WebUI -- 5. POST /api/approve --> Backend
    TUI -- 5. POST /api/approve --> Backend
    Backend -- 6. Future.resolve --> AM
```

## Protocol Layer Architecture

The framework provides three canonical protocol adapters:

1. **ACP (Agent Communication Protocol)**: Primary protocol for standardized sessions, planning, and streaming
2. **A2A (Agent-to-Agent)**: Peer-to-peer agent communication and coordination
3. **AG-UI**: Legacy streaming interface for backward compatibility with native Pydantic AI clients

All protocol adapters are centralized in `agent_utilities/`:
- `acp_adapter.py`: ACP envelope formatting, session management, per-session `agent_factory`
- `a2a.py`: A2A peer discovery, JSON-RPC client, registry management
- `agui_emitter.py`: AG-UI wire format translator for direct graph execution events
- Server endpoints: `/acp` (MOUNT), `/a2a` (MOUNT), `/ag-ui` (POST)

### Direct Graph Execution (Fast Path)

When a `graph_bundle` is present and `GRAPH_DIRECT_EXECUTION=true` (default), the **AG-UI endpoint** bypasses the outer LLM agent entirely:

```
# Legacy (Agent-mediated):
User Query → /ag-ui → Agent.run() → LLM → "call run_graph_flow" → graph.run()

# Direct (Fast Path):
User Query → /ag-ui → graph.iter() → [step events] → AGUIGraphEmitter → wire format
```

This eliminates one full LLM inference round-trip per request. The fast path uses `graph.iter()` (pydantic-graph beta API) for step-by-step execution, yielding per-node events that are translated to AG-UI wire format by `AGUIGraphEmitter`.

The fast path is gated on:
1. `graph_bundle` containing a real Graph object with `.iter()` support
2. `GRAPH_DIRECT_EXECUTION` env var set to `true` (default)

The **ACP adapter** uses pydantic-acp's `agent_factory` callback for per-session agent creation, binding graph context directly to each session's closure.

The **A2A path** retains the LLM-mediated `run_graph_flow` tool call to support multi-agent negotiation.

### Authentication Passthrough (`custom_headers`)

`create_agent_server()` and `create_graph_agent_server()` accept a generic `custom_headers: dict[str, Any] | None = None` kwarg that is propagated verbatim to the LLM HTTP client as request headers. agent-utilities itself is **auth-agnostic** -- it does not ship provider-specific auth code (OIDC, client-credentials flows, bearer-token fetchers, etc.) and has no opinion about where those headers come from. Downstream packages are free to populate the dict from any source: environment variables, a token-fetching library, static config, a secret manager, or a callable that refreshes on every run. The same kwarg is reused by `ssl_verify` for self-signed gateways. See `agents/repository-manager/repository_manager/agent_server.py` for a reference implementation that builds the dict from `LLM_CUSTOM_HEADERS` / `LLM_HEADER_*` environment variables without pulling any provider-specific dependency into this core package.

## Graph Orchestration Architecture

```mermaid
graph TB
    Start([User Query + Images]) --> ACPLayer["<b>ACP / AG-UI / SSE</b><br/><i>(Unified Protocol Layer)</i>"]
    ACPLayer --> UsageGuard[Usage Guard: Rate Limiting]
    UsageGuard -- "Allow" --> router_step[Router: Topology Selection]
    UsageGuard -- "Block" --> End([End Result])

    router_step -- "Trivial Query" --> End
    router_step -- "Full Pipeline" --> dispatcher[Dispatcher: Dynamic Routing]
    dispatcher -- "First Entry" --> mem_step[Memory: Context Retrieval]
    mem_step --> dispatcher

    subgraph DiscoveryPhase ["Discovery Phase"]
        direction TB
        Researcher["<b>Researcher</b><br/>---<br/><i>u-skill:</i> web-search, web-crawler, web-fetch<br/><i>t-tool:</i> project_search, read_workspace_file"]
        Architect["<b>Architect</b><br/>---<br/><i>u-skill:</i> c4-architecture, spec-generator, product-strategy, user-research, brainstorming<br/><i>t-tool:</i> developer_tools"]
        MCPDiscovery["<b>Unified Registry</b><br/>---<br/><i>source:</i> Knowledge Graph"]
        res_joiner[Research Joiner: Barrier Sync]
    end

    dispatcher -- "Research First" --> Researcher
    dispatcher -- "Research First" --> Architect
    dispatcher -- "Research First" --> MCPDiscovery
    Researcher --> res_joiner
    Architect --> res_joiner
    MCPDiscovery --> res_joiner
    res_joiner -- "Coalesced Context" --> dispatcher

    subgraph ExecutionPhase ["Execution Phase"]
        direction TB

        subgraph Programmers ["Programmers"]
            direction LR
            PyP["<b>Python</b><br/>---<br/><i>u-skill:</i> agent-builder, tdd-methodology, mcp-builder, jupyter-notebook<br/><i>g-skill:</i> python-docs, fastapi-docs, pydantic-ai-docs<br/><i>t-tool:</i> developer_tools"]
            TSP["<b>TypeScript</b><br/>---<br/><i>u-skill:</i> react-development, web-artifacts, tdd-methodology, canvas-design<br/><i>g-skill:</i> nodejs-docs, react-docs, nextjs-docs, shadcn-docs<br/><i>t-tool:</i> developer_tools"]
            GoP["<b>Go</b><br/>---<br/><i>u-skill:</i> tdd-methodology<br/><i>g-skill:</i> go-docs<br/><i>t-tool:</i> developer_tools"]
            RustP["<b>Rust</b><br/>---<br/><i>u-skill:</i> tdd-methodology<br/><i>g-skill:</i> rust-docs<br/><i>t-tool:</i> developer_tools"]
            CSP["<b>C Programmer</b><br/>---<br/><i>u-skill:</i> developer-utilities<br/><i>g-skill:</i> c-docs<br/><i>t-tool:</i> developer_tools"]
            CPP["<b>C++ Programmer</b><br/>---<br/><i>u-skill:</i> developer-utilities<br/><i>t-tool:</i> developer_tools"]
            JSP["<b>JavaScript</b><br/>---<br/><i>u-skill:</i> web-artifacts, canvas-design, developer-utilities<br/><i>g-skill:</i> nodejs-docs, react-docs<br/><i>t-tool:</i> developer_tools"]
        end

        subgraph InfraGroup ["Infrastructure"]
            direction LR
            DevOps["<b>DevOps</b><br/>---<br/><i>u-skill:</i> cloudflare-deploy<br/><i>g-skill:</i> docker-docs, terraform-docs<br/><i>t-tool:</i> developer_tools"]
            Cloud["<b>Cloud</b><br/>---<br/><i>u-skill:</i> c4-architecture<br/><i>g-skill:</i> aws-docs, azure-docs, gcp-docs<br/><i>t-tool:</i> developer_tools"]
            DBA["<b>Database</b><br/>---<br/><i>u-skill:</i> database-tools<br/><i>g-skill:</i> postgres-docs, mongodb-docs, redis-docs<br/><i>t-tool:</i> developer_tools"]
        end

        subgraph Specialized ["Specialized & Quality"]
            direction LR
            Sec["<b>Security</b><br/>---<br/><i>u-skill:</i> security-tools<br/><i>g-skill:</i> linux-docs<br/><i>t-tool:</i> developer_tools"]
            QA["<b>QA</b><br/>---<br/><i>u-skill:</i> spec-verifier, tdd-methodology<br/><i>g-skill:</i> testing-library-docs<br/><i>t-tool:</i> developer_tools"]
            UIUX["<b>UI/UX</b><br/>---<br/><i>u-skill:</i> theme-factory, brand-guidelines, algorithmic-art<br/><i>g-skill:</i> shadcn-docs, framer-docs<br/><i>t-tool:</i> developer_tools"]
            Debug["<b>Debugger</b><br/>---<br/><i>u-skill:</i> developer-utilities, agent-builder<br/><i>t-tool:</i> developer_tools"]
        end
    end

    dispatcher -- "Parallel Dispatch" --> Programmers
    dispatcher -- "Parallel Dispatch" --> InfraGroup
    dispatcher -- "Parallel Dispatch" --> Specialized

    Programmers --> exe_joiner[Execution Joiner: Barrier Sync]
    InfraGroup --> exe_joiner
    Specialized --> exe_joiner

    exe_joiner -- "Implementation Results" --> dispatcher

    dispatcher -- "Plan Complete" --> verifier[Verifier: Quality Gate]
    dispatcher -- "Council" --> council[Council: Multi-Perspective Deliberation]
    council --> exe_joiner
    verifier -- "Pass: Score >= 0.7" --> synthesizer[Synthesizer: Response Composition]
    verifier -- "Fail: Score < 0.7" --> dispatcher
    dispatcher -- "Terminal Failure" --> End
    planner_step[Planner: Re-plan] --> dispatcher
    synthesizer -- "Final Response" --> End

    subgraph SDD_Lifecycle ["Spec-Driven Development"]
        direction TB
        Const["<b>Constitution</b><br/>(Governance)"] --> Spec["<b>Specification</b><br/>(Spec)"]
        Spec --> SDDPlan["<b>Technical Plan</b><br/>(ImplementationPlan)"]
        SDDPlan --> SDDTasks["<b>Tasks</b><br/>(Tasks)"]
        SDDTasks --> SDDExec["<b>Execution</b><br/>(Parallel Dispatch)"]
        SDDExec --> SDDVerify["<b>Verification</b><br/>(Spec Audit)"]
    end

    style Researcher fill:#e1d5e7,stroke:#9673a6,stroke-width:2px
    style Architect fill:#e1d5e7,stroke:#9673a6,stroke-width:2px
    style MCPDiscovery fill:#e1d5e7,stroke:#9673a6,stroke-width:2px
    style Programmers fill:#dae8fe,stroke:#6c8ebf,stroke-width:2px
    style InfraGroup fill:#fad9b8,stroke:#d6b656,stroke-width:2px
    style Specialized fill:#e0d3f5,stroke:#82b366,stroke-width:2px
    style verifier fill:#fff2cc,stroke:#d6b656,stroke-width:2px
    style council fill:#fce4ec,stroke:#c62828,stroke-width:2px
    style synthesizer fill:#d5e8d4,stroke:#82b366,stroke-width:2px
    style planner_step fill:#dae8fe,stroke:#6c8ebf,stroke-width:2px
    style End fill:#f8cecc,stroke:#b85450,stroke-width:2px
    style res_joiner fill:#f5f5f5,stroke:#666,stroke-dasharray: 5 5
    style exe_joiner fill:#f5f5f5,stroke:#666,stroke-dasharray: 5 5
    style dispatcher fill:#d5e8d4,stroke:#666,stroke-width:2px
    style Start fill:#38B6FF
    style ACPLayer fill:#38B6FF,stroke-width:2px
```

> **Note:** MCP ecosystem agents (AdGuard, Jellyfin, Ansible Tower, etc.) are dynamically spawned as `CallableResource` nodes in the Knowledge Graph. They are discovered at runtime from `mcp_config.json` and do not appear in this static diagram.

### Council Deliberation Node

The **Council** is a specialized graph node that implements Karpathy's LLM Council pattern for high-stakes decision-making. It provides a 4-stage deliberative pipeline:

```mermaid
graph LR
    Q[Query] --> A1[Contrarian]
    Q --> A2[First Principles]
    Q --> A3[Expansionist]
    Q --> A4[Outsider]
    Q --> A5[Executor]
    A1 --> Anon[Anonymize]
    A2 --> Anon
    A3 --> Anon
    A4 --> Anon
    A5 --> Anon
    Anon --> R1[Reviewer 1]
    Anon --> R2[Reviewer 2]
    Anon --> R3[Reviewer 3]
    R1 --> Chair[Chairman]
    R2 --> Chair
    R3 --> Chair
    Chair --> V[CouncilVerdict]

    style Anon fill:#fff2cc,stroke:#d6b656
    style Chair fill:#d5e8d4,stroke:#82b366
    style V fill:#c8e6c9,stroke:#2e7d32
```

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

## Hierarchical State Machine (HSM) Architecture

The graph orchestration system is a **Hierarchical State Machine**. It follows the same formal model used in robotics, game engines, UML statecharts, and SCXML workflow engines.

### HSM Level Mapping
```
Level 0: Root Graph (N Orchestration Nodes)
├── usage_guard → router → dispatcher → memory_selection → dispatcher
├── researcher, architect, verifier (discovery/validation)
├── parallel_batch_processor → expert_executor (fan-out)
├── research_joiner, execution_joiner (fan-in)
├── verifier → synthesizer → END (quality gate + response composition)
└── planner (re-planning on verification failure)

Level 1: Superstates - Specialist Agents
├── Specialist Roster (Dynamically discovered from the **Knowledge Graph**)
│   Each loads: name-matched prompt + discovered capabilities + mapped MCP toolsets
│   Supports: 'prompt' (local), 'mcp' (stdio), and 'a2a' (remote) agent types
└── Unified Execution: Dynamic routing based on registry-provided metadata

Level 2: Substates - Agent Internal Loop
└── Pydantic AI Agent.run() = UserPromptNode → ModelRequestNode → CallToolsNode → ...
    Multi-turn tool iteration (max 3 iterations per specialist)

Level 3: Leaf States - MCP Tool Execution
└── Each tool call invokes an MCP server subprocess via stdio/HTTP
    Atomic operations: get_project(), list_branches(), run_cypher_query(), etc.
```

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
| `/health` | GET | Core | Health check and server metadata |
| `/ag-ui` | POST | Agent UI | AG-UI streaming endpoint with sideband graph events |
| `/stream` | POST | Agent UI | Generic SSE stream endpoint for graph agent execution |
| `/acp` | MOUNT | ACP | Agent Communication Protocol (pydantic-acp) |
| `/a2a` | MOUNT | A2A | Agent-to-Agent (fastA2A) JSON-RPC endpoint |
| `/api/approve` | POST | Human-in-the-Loop | Resolves pending tool approvals and MCP elicitation requests |
| `/chats` | GET | Core | List all stored chat sessions |
| `/chats/{chat_id}` | GET | Core | Get full message history for a specific chat |
| `/chats/{chat_id}` | DELETE | Core | Delete a specific chat session |
| `/mcp/config` | GET | Interoperability | Return the current MCP server configuration |
| `/mcp/tools` | GET | Interoperability | List all tools from connected MCP servers |
| `/mcp/reload` | POST | Interoperability | Hot-reload MCP servers and rebuild graph |

## The Complete Execution Journey

### Phase 1: Ingress & Protocol Handling
1. **Entry**: A user query (text + optional images) arrives via any supported protocol: AG-UI (`/ag-ui`), ACP (`/acp`), SSE (`/stream`), or REST (`/api/chat`).
2. **Direct Dispatch Check**: If a `graph_bundle` is present and `GRAPH_DIRECT_EXECUTION=true`, AG-UI routes directly to `execute_graph_iter()` — bypassing the outer LLM agent.
3. **Unified Execution**: All protocols funnel through the same graph engine via `graph/unified.py`. The `execute_graph_iter()` entry point uses `graph.iter()` for step-by-step control.
4. **State Initialization**: A fresh `GraphState` is initialized with the consolidated `query_parts`.

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
