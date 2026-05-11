# agent-utilities C4 Architecture

This document provides formal C4 architecture diagrams showing how the 5 pillars
of `agent-utilities` interconnect with each other and with external IDE consumers.

## Level 1: System Context

Shows `agent-utilities` in the broader ecosystem — all IDE and agent consumers.

```mermaid
C4Context
    title agent-utilities — System Context

    Person(dev, "Developer", "Uses IDE or CLI to build and run agents")
    Person(agent, "Autonomous Agent", "Runs tasks without human intervention")

    System(au, "agent-utilities", "Core agent OS kernel with KG-native intelligence, 5-pillar architecture")

    System_Ext(antigravity, "Antigravity IDE", "Primary development environment")
    System_Ext(claude, "Claude Code", "Anthropic coding agent")
    System_Ext(opencode, "OpenCode", "Open-source coding agent")
    System_Ext(devin, "Devin", "Cognition coding agent")
    System_Ext(terminal, "agent-terminal-ui", "Textual TUI client")
    System_Ext(webui, "agent-webui", "React web client")
    System_Ext(skills, "universal-skills", "Skill graph and SDD tooling")

    Rel(dev, antigravity, "Develops in")
    Rel(dev, terminal, "CLI interaction")
    Rel(antigravity, au, "MCP: KG queries, tool execution")
    Rel(claude, au, "MCP: shared KG read/write")
    Rel(opencode, au, "MCP: shared KG read")
    Rel(devin, au, "MCP: shared KG read")
    Rel(terminal, au, "Direct Python API (zero-copy)")
    Rel(webui, au, "ACP/AG-UI protocol")
    Rel(skills, au, "DSTDD pipeline, skill ingestion")
    Rel(agent, au, "Orchestrated execution")
```

## Level 2: Container Diagram

Shows the 5 pillars as containers with data flows between them.

```mermaid
C4Container
    title agent-utilities — Container Diagram (5 Pillars)

    Person(user, "Developer / Agent")

    System_Boundary(au, "agent-utilities") {
        Container(orch, "ORCH: Orchestration Engine", "Python", "Router, Planner, Dispatcher, Capability Wiring")
        Container(kg, "KG: Knowledge Graph", "Python + LadybugDB", "15-phase pipeline, OWL ontology, hybrid retrieval")
        Container(ahe, "AHE: Agentic Harness", "Python", "Self-model, TeamConfig, evolution, evaluation")
        Container(eco, "ECO: Ecosystem Peripherals", "Python + FastMCP", "MCP server factory, A2A, skill management")
        Container(os_k, "OS: Agent OS Kernel", "Python + FastAPI", "Auth, guardrails, lifecycle, telemetry")
        ContainerDb(kgdb, "Knowledge Graph DB", "LadybugDB/SQLite", "~/.local/share/agent-utilities/kg/")
    }

    Rel(user, os_k, "Authenticated request")
    Rel(os_k, orch, "Validated task dispatch")
    Rel(orch, kg, "Queries for routing & specialist selection")
    Rel(kg, kgdb, "Cypher queries / persistence")
    Rel(kg, ahe, "Feeds Self-Model & TeamConfig")
    Rel(ahe, eco, "Promotes proven coalitions to MCP/A2A")
    Rel(eco, os_k, "Tool execution through kernel guardrails")
    Rel(os_k, kg, "Persists execution traces & telemetry")
```

## Level 3: Component Diagram — Per Pillar

### Pillar 1: Orchestration Engine (ORCH)

```mermaid
C4Component
    title ORCH — Orchestration Engine Components

    Container_Boundary(orch, "Orchestration Engine") {
        Component(router, "KG Router", "Python", "Ontological routing via KG topology")
        Component(planner, "Agentic Planner", "Python", "HTN recursive goal decomposition")
        Component(dispatcher, "Graph Dispatcher", "Python", "Parallel batch execution")
        Component(wiring, "Capability Wiring Engine", "Python", "Dynamic capability discovery")
        Component(orchestrator, "Agent Orchestrator", "Python", "Unified harness for multi-agent execution")
    }

    Rel(router, planner, "Routes task to planning")
    Rel(planner, dispatcher, "Decomposes into parallel batches")
    Rel(dispatcher, wiring, "Discovers required capabilities")
    Rel(orchestrator, router, "Entry point for all orchestration")
```

### Pillar 2: Knowledge Graph (KG)

```mermaid
C4Component
    title KG — Knowledge Graph Components

    Container_Boundary(kg, "Knowledge Graph") {
        Component(engine, "IntelligenceGraphEngine", "Python", "Core engine with 10 mixins")
        Component(backend, "Graph Backends", "Python", "LadybugDB, FalkorDB, Neo4j, PostgreSQL")
        Component(pipeline, "15-Phase Pipeline", "Python", "Ingest, enrich, index, materialize")
        Component(retrieval, "Hybrid Retriever", "Python", "Semantic 72% + keyword 28% search")
        Component(ontology, "OWL Bridge", "Python + RDFLib", "Formal ontology with FIBO/BFO")
        Component(memory, "Memory Tiers", "Python", "Episodic, semantic, procedural memory")
    }

    Rel(engine, backend, "Tier 1: Cypher persistence")
    Rel(engine, pipeline, "Orchestrates 15-phase ingestion")
    Rel(retrieval, engine, "Queries via hybrid scoring")
    Rel(ontology, engine, "Schema enforcement via OWL")
    Rel(memory, engine, "CRUD for tiered memories")
```

### Pillar 3: Agentic Harness (AHE)

```mermaid
C4Component
    title AHE — Agentic Harness Components

    Container_Boundary(ahe, "Agentic Harness") {
        Component(eval, "Continuous Evaluation Engine", "Python", "Multi-strategy EvalRunner")
        Component(evolve, "Evolution Engine", "Python", "Skill neologism, genetic crossover")
        Component(selfmodel, "Self-Model", "Python", "Dynamic capability self-assessment")
        Component(team, "TeamConfig Composer", "Python", "Coalition formation & promotion")
        Component(sdd, "DSTDD Manager", "Python", "Design-Spec-Test pipeline")
    }

    Rel(eval, selfmodel, "Updates self-assessment scores")
    Rel(evolve, eval, "Triggers evolution on failure patterns")
    Rel(team, selfmodel, "Uses capability scores for composition")
    Rel(sdd, eval, "Validates features against KG integrity")
```

### Pillar 4: Ecosystem Peripherals (ECO)

```mermaid
C4Component
    title ECO — Ecosystem Components

    Container_Boundary(eco, "Ecosystem Peripherals") {
        Component(mcp_factory, "MCP Server Factory", "Python + FastMCP", "create_mcp_server with auth stack")
        Component(kg_mcp, "KG MCP Server", "Python", "Thin wrapper exposing KG as MCP tools")
        Component(a2a, "A2A Network", "Python", "Agent-to-agent discovery and delegation")
        Component(skill_mgr, "Skill Manager", "Python", "Dynamic tool loading and skill evolution")
        Component(bridge, "Ecosystem Bridge", "Python", "Cross-package integration")
    }

    Rel(mcp_factory, kg_mcp, "Creates KG MCP instance")
    Rel(kg_mcp, a2a, "Shares KG data across agent network")
    Rel(skill_mgr, bridge, "Loads skills from universal-skills")
```

### Pillar 5: Agent OS Kernel (OS)

```mermaid
C4Component
    title OS — Agent OS Kernel Components

    Container_Boundary(os_k, "Agent OS Kernel") {
        Component(auth, "Security Policy Middleware", "Python", "JWT, API key, MCP auth")
        Component(threat, "Threat Defense Engine", "Python", "Prompt injection, jailbreak detection")
        Component(guardrails, "Guardrail Engine", "Python", "Tool guard, rate limit, content filter")
        Component(telemetry, "Telemetry Pipeline", "Python", "OTEL, token tracking, audit logging")
        Component(paths, "XDG Paths Module", "Python + platformdirs", "Centralized path resolution")
    }

    Rel(auth, threat, "Validates before routing")
    Rel(threat, guardrails, "Applies runtime constraints")
    Rel(guardrails, telemetry, "Records enforcement decisions")
    Rel(paths, auth, "Provides config/data locations")
```

## Cross-Pillar Data Flows

```mermaid
flowchart LR
    subgraph "Cross-Pillar Data Flows"
        direction TB

        subgraph INGEST ["Ingestion Flow"]
            direction LR
            ECO_MCP["ECO: MCP Tool Call"] --> ORCH_ROUTE["ORCH: Router"]
            ORCH_ROUTE --> KG_INGEST["KG: Ingest Engine"]
            KG_INGEST --> KG_OWL["KG: OWL Bridge"]
        end

        subgraph EXECUTE ["Execution Flow"]
            direction LR
            ORCH_PLAN["ORCH: Planner"] --> ORCH_DISPATCH["ORCH: Dispatcher"]
            ORCH_DISPATCH --> ECO_TOOL["ECO: Tool Executor"]
            ECO_TOOL --> OS_GUARD["OS: Guardrails"]
            OS_GUARD --> AHE_EVAL["AHE: Evaluator"]
        end

        subgraph LEARN ["Learning Flow"]
            direction LR
            AHE_EVAL2["AHE: EvalRunner"] --> KG_MEMORY["KG: Memory Tier"]
            KG_MEMORY --> AHE_EVOLVE["AHE: Evolution"]
            AHE_EVOLVE --> ECO_SKILL["ECO: Skill Evolver"]
        end

        subgraph SECURE ["Security Flow"]
            direction LR
            OS_SCAN["OS: Threat Scanner"] --> KG_RISK["KG: Risk Ontology"]
            KG_RISK --> AHE_IMMUNE["AHE: Immunity"]
            AHE_IMMUNE --> OS_POLICY["OS: Policy Engine"]
        end
    end
```

## Pillar Interconnection Matrix

```mermaid
graph TD
    subgraph "Pillar Interconnection Matrix"
        P1["ORCH: Orchestration"]
        P2["KG: Knowledge Graph"]
        P3["AHE: Agentic Harness"]
        P4["ECO: Ecosystem"]
        P5["OS: Agent OS"]
    end

    P1 <-->|"Router queries KG for specialist selection<br/>KG provides ontological routing tables"| P2
    P1 -->|"Planner delegates to MCP tools<br/>Capability Wiring discovers tool registry"| P4
    P1 <-->|"Orchestrator feeds results to evaluator<br/>Evaluator adjusts routing weights"| P3

    P2 -->|"Memory tiers feed Self-Model<br/>TeamConfig promotes proven coalitions"| P3
    P2 -->|"Ecosystem Topology Map materializes<br/>40-repo graph as KG nodes"| P4
    P2 <-->|"Execution traces persist to KG<br/>Telemetry feeds observability"| P5

    P3 -->|"Evolved skills promoted to MCP/A2A<br/>Skill neologisms create new tools"| P4
    P3 -->|"Adaptive Immunity Pipeline<br/>updates security patterns"| P5

    P4 -->|"MCP middleware stack enforces<br/>auth, rate limits, guardrails"| P5

    P5 -->|"Policy engine governs all<br/>execution paths and prompt safety"| P1

    style P1 fill:#dae8fe,stroke:#6c8ebf,stroke-width:3px
    style P2 fill:#d5e8d4,stroke:#82b366,stroke-width:3px
    style P3 fill:#fff2cc,stroke:#d6b656,stroke-width:3px
    style P4 fill:#e6ccff,stroke:#9673a6,stroke-width:3px
    style P5 fill:#cce5ff,stroke:#004085,stroke-width:3px
```

> **Key Insight**: Every pillar has at least one bidirectional dependency with another pillar.
> The system is a closed feedback loop, not a layered stack. This is why isolated concept
> additions are dangerous — they must wire into the loop.
