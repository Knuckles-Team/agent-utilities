# agent-utilities C4 Architecture

This document provides formal C4 architecture diagrams showing how the 6 pillars
of `agent-utilities` interconnect with each other and with external IDE consumers.

> [!NOTE]
> Components marked with 🔬 are research-backed additions from the
> comparative analysis pipeline (papers 2605.05701v1, 2605.03310v1,
> 2604.20874v1, 2605.05242v1).

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
    System_Ext(geniusbot, "geniusbot", "Premium multi-platform PySide6 Systems & Finance Cockpit")
    System_Ext(skills, "universal-skills", "Skill graph and SDD tooling")

    System_Ext(enterprise, "Enterprise Systems", "ITSM/ERP/BPM/EA tools — ServiceNow OR ERPNext, Camunda, Archi, LeanIX, GitLab — interchangeable via the vendor-neutral crosswalk")

    Rel(dev, antigravity, "Develops in")
    Rel(dev, terminal, "CLI interaction")
    Rel(dev, geniusbot, "Visual interaction")
    Rel(antigravity, au, "MCP: KG queries, tool execution")
    Rel(claude, au, "MCP: shared KG read/write")
    Rel(opencode, au, "MCP: shared KG read")
    Rel(devin, au, "MCP: shared KG read")
    Rel(terminal, au, "Direct Python API (zero-copy)")
    Rel(webui, au, "ACP/AG-UI protocol")
    Rel(geniusbot, au, "Direct Python API & AgentBridge (asynchronous)")
    Rel(skills, au, "DSTDD pipeline, skill ingestion")
    Rel(agent, au, "Orchestrated execution")
    Rel(au, enterprise, "Vendor adapters lift REST APIs → canonical ArchiMate nodes; virtual REST federation queries live data (KG-2.9 / KG-2.1)")
```

## Level 2: Container Diagram

Shows the 5 pillars as containers with data flows between them.

```mermaid
C4Container
    title agent-utilities — Container Diagram (5 Pillars)

    Person(user, "Developer / Agent")

    System_Boundary(au, "agent-utilities") {
        Container(orch, "ORCH: Orchestration Engine", "Python", "Router, Planner, Dispatcher, Capability Wiring")
        Container(kg, "KG: Knowledge Graph", "Python + epistemic-graph (Unix Sockets)", "Native graph-os ingestion, OWL ontology via Rust-compiled Datalog, hybrid retrieval")
        Container(ahe, "AHE: Agentic Harness", "Python", "Self-model, TeamConfig, evolution, evaluation")
        Container(eco, "ECO: Ecosystem Peripherals", "Python + FastMCP", "MCP server factory, A2A, skill management")
        Container(os_k, "OS: Agent OS Kernel", "Python + FastAPI", "Auth, guardrails, lifecycle, telemetry")
        ContainerDb(kgdb, "Knowledge Graph DB", "epistemic-graph (L0/L1) + PostgreSQL (durable); SQLite/file fallback", "~/.local/share/agent-utilities/kg/")
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
        Component(coord, "🔬 Coordination Layer", "Python", "ORCH-1.0: Pluggable coordination protocols. Research: 2605.03310v1")
        Component(kgfactory, "KG Graph Factory", "Python", "ORCH-1.20: Materializes pydantic-graph topologies from KG AgentTemplates")
        Component(agentrunner, "Agent Runner", "Python", "ORCH-1.21: KG-to-LLM execution bridge — resolves agents, binds tools, tracks provenance")
        Component(workflowstore, "Workflow Store", "Python", "ORCH-1.22: Persists GraphPlan workflows as KG subgraphs with versioning")
        Component(workflowcompiler, "Workflow Compiler", "Python", "ORCH-1.23: NL → GraphPlan DAG compiler with KG agent matching")
        Component(workflowcatalog, "Workflow Catalog", "Python + YAML", "ORCH-1.24: Externally-consumable workflow definitions with KG persistence")
        Component(workflowrunner, "Workflow Runner", "Python", "ORCH-1.24: Executes stored workflows via wave-based parallel dispatch")
        Component(pll, "🔬 Prediction Linkage Layer", "Python", "ORCH-1.6: Fuses confidence matrices for ensemble modeling")
        Component(mas, "🔬 RecursiveMAS Latent Orchestrator", "Python", "ORCH-1.7: Continuous latent loop or simulated semantic collaboration")
    }

    Rel(router, planner, "Routes task to planning")
    Rel(planner, dispatcher, "Decomposes into parallel batches")
    Rel(dispatcher, wiring, "Discovers required capabilities")
    Rel(orchestrator, router, "Entry point for all orchestration")
    Rel(orchestrator, coord, "Selects protocol before execution")
    Rel(orchestrator, pll, "Aggregates quant predictions")
    Rel(orchestrator, mas, "Delegates latent multi-agent loops")
    Rel(mas, agentrunner, "Registers latent/simulated execution traces")
    Rel(coord, dispatcher, "Applies consensus/voting/delegation")
    Rel(router, kgfactory, "Materializes graph from KG templates")
    Rel(kgfactory, dispatcher, "Provides topology + specialist configs")
    Rel(agentrunner, kgfactory, "Materializes agent-specific graph")
    Rel(agentrunner, router, "Resolves agent from KG, dispatches task")
    Rel(workflowcatalog, workflowstore, "Registers scenarios in KG")
    Rel(workflowcompiler, workflowstore, "Persists compiled workflows")
    Rel(workflowrunner, agentrunner, "Executes steps via run_agent()")
    Rel(workflowrunner, workflowstore, "Loads workflows by name")
    Rel(pll, orchestrator, "Returns fused predictions")
    Rel(mas, planner, "Bypasses standard planning via latent loops")
```

### Pillar 2: Knowledge Graph (KG)

```mermaid
C4Component
    title KG — Knowledge Graph Components

    Container_Boundary(kg, "Knowledge Graph") {
        Component(engine, "IntelligenceGraphEngine", "Python", "Core engine composed of 8 focused mixins (Query, Memory, Ingestion, MCPDiscovery, Registry, TaskManager, Federation, AHE)")
        Component(backend, "Graph Backends", "Python", "Primary: epistemic-graph + PostgreSQL (TieredGraphBackend); contrib: Ladybug, FalkorDB, Neo4j")
        Component(pipeline, "Graph-OS Ingestion", "Python", "Ingest, enrich, index, materialize, evolve via MCP")
        Component(retrieval, "Hybrid Retriever", "Python", "Semantic 72% + keyword 28% search")
        Component(dci, "🔬 DCI Retriever", "Python", "KG-2.3: Multi-hop graph traversal retrieval. Research: 2605.05242v1")
        Component(ontology, "OWL Bridge + SPARQL", "Python + epistemic-graph", "Formal ontology, SPARQL endpoint, Rust Datalog reasoning via out-of-process epistemic-graph client (no PyO3)")
        Component(epistemic_compute, "🔬 EpistemicGraph Compute Engine", "Rust (Unix Sockets)", "KG-2.7: Compiled sub-millisecond topological processing and Datalog reasoning")
        Component(quant_compute, "🔬 Quant Compute Engine", "Rust (Unix Sockets)", "KG-2.7: C-speed rolling variance, moving averages, and order matching simulation")
        Component(sdd_ont, "SDD Ontology", "OWL/Turtle", "KG-2.6: Spec, Feature, Requirement, TestCase classes")
        Component(shacl, "SHACL Validator", "Python + pyshacl", "KG-2.6: Enterprise governance shape validation")
        Component(publisher, "Ontology Publisher", "Python", "KG-2.6: Push to Stardog/Fuseki")
        Component(loader, "Ontology Loader", "Python", "KG-2.6: owl:imports resolver with caching")
        Component(memory, "Memory Tiers", "Python", "Temporally-Aware Epistemic Memory (Episodic, Semantic, Procedural)")
        Component(evolving_memory, "🔬 Evolving Memory API", "Python", "KG-2.4: Ebbinghaus fact decay & GraphRAG traversal")
        Component(ctxbudget, "🔬 Context Budget Optimizer", "Python", "KG-2.1: Root Theorem compaction. Research: 2604.20874v1")
        Component(argraph, "🔬 AR-Graph", "Python", "KG-2.3: Dynamic Agent Relationship Graph")
        Component(tsgraph, "🔬 Time-Series Graph", "Python", "KG-2.6: Temporal weighted decay graphs")
        Component(stream_ingest, "Stream Hydration / R2RML", "Python", "Dynamic dynamic-free parallel streaming from external APIs (ServiceNow, GitLab, Jira, Slack) and Event Substrates (Kafka)")
        Component(db_schema, "Database Schema Hydrator", "Python", "KG-2.7: Extracts SQL schema relations and auto-aligns with infrastructure ontology")
        Component(process_mod, "Process Modeling Engine", "Python", "KG-2.7: Maps individual workflow steps directly to process_step nodes via :precedes edges")
        Component(cache_fabric, "Shared Ephemeral Cache Fabric", "Valkey / Redis / Filesystem", "Memory sharing between agents with TTL-based decay")
        Component(dspy_bridge, "DSPy KG Bridge", "Python", "KG-2.2: Instantly persists evolved prompts and optimization traces")
        Component(align_bridge, "Ontology Alignment Bridge", "Python", "KG-2.7: Unifies disparate silos (Enterprise Architecture Repositories [EARs], ServiceNow) via cosine_similarity & owl:sameAs")
        Component(entail_scope, "Entailment-Aware Permission Scoper", "Python", "KG-2.7: Intersects security classifications for Rust Datalog inferred edges")
        Component(crosswalk, "Vendor-Neutral Crosswalk", "OWL/Turtle", "KG-2.9: ontology_archimate.ttl — binds each vendor class (ServiceNow :Incident, ERPNext :ErpNextIssue, Camunda :BusinessTask) to one canonical ArchiMate concept via subClassOf/equivalentClass")
        Component(vendor_ext, "Vendor Source Extractors", "Python", "KG-2.9: Self-registering adapters (servicenow/erpnext/camunda/leanix) lift REST APIs into canonical GraphNodes")
        Component(realizes, "Code→Capability Bridge", "Python", "KG-2.8: realizes.py — REALIZES edges from code features to BusinessCapability (match / mint / curated)")
        Component(cap_wb, "Capability Write-Back", "Python", "KG-2.8: Pushes provisional/derived capabilities back to Archi (add_element) & LeanIX (postbusinesscapability)")
        Component(rest_fed, "Virtual REST Federation", "Python", "KG-2.1: register_rest_source — query live REST systems on-demand via extractors, TTL-cached")
        Component(brain_guard, "Brain-Guarded Backend", "Python", "KG-2.6: write-path provenance + source-authority arbitration (trust decay). Installed when KG_BRAIN_ENFORCE=1")
        Component(secured, "Secured Reads", "Python", "KG-2.6: read-path ACL filter + tenant scope + read audit; entailment-aware ACL inheritance")
        Component(feedback, "Feedback Service", "Python", "KG-2.8: human correction → reward / durable governance rule / eval case (graph_feedback tool)")
        Component(govrules, "Governance Rules", "Python", "KG-2.8: rules consulted at retrieval time to filter/re-rank designations")
        Component(budget, "Retrieval Budget", "Python", "KG-2.1: token-budgeted, task-scoped retrieval (no context bloat)")
        Component(streams, "Stream Adapters", "Python", "KG-2.6: real Kafka/NATS ingestion (optional deps)")
        Component(intel, "Intelligence Extractors", "Python", "KG-2.8: distil calls/docs → Insight/Fact/Framework/Playbook")
    }

    Rel(engine, backend, "Tier 1: Cypher persistence")
    Rel(engine, pipeline, "Orchestrates graph-os native ingestion")
    Rel(retrieval, engine, "Queries via hybrid scoring")
    Rel(dci, retrieval, "Seeds from hybrid, then graph traversal")
    Rel(ontology, engine, "Schema enforcement via OWL")
    Rel(ontology, epistemic_compute, "Delegates Datalog reasoning")
    Rel(engine, quant_compute, "Executes vectorized calculations")
    Rel(ontology, sdd_ont, "owl:imports SDD classes")
    Rel(shacl, ontology, "Validates materialized RDF")
    Rel(publisher, ontology, "Exports/pushes ontology")
    Rel(loader, ontology, "Resolves owl:imports")
    Rel(memory, engine, "CRUD for tiered memories")
    Rel(ctxbudget, memory, "Compacts recall results within budget")
    Rel(engine, argraph, "Tracks inter-agent communication topologies")
    Rel(engine, tsgraph, "Applies temporal decay to HNSW edges")
    Rel(engine, stream_ingest, "Hydrates graph from high-throughput API streams")
    Rel(stream_ingest, ontology, "Enforces schema correctness during ingestion via dynamic OWL classification")
    Rel(stream_ingest, align_bridge, "Resolves topological alignments for disparate systems")
    Rel(ontology, entail_scope, "Delegates security classification filtering for inferred graphs")
    Rel(engine, cache_fabric, "Stores and invalidates ephemeral agent contexts with dynamic TTL tracking")
    Rel(dspy_bridge, engine, "Fast-path Cypher MERGE")
    Rel(vendor_ext, engine, "Writes canonical GraphNodes via single backend interface")
    Rel(vendor_ext, crosswalk, "Emits canonical types bound by the crosswalk")
    Rel(crosswalk, ontology, "Loaded as a sibling ontology; HermiT propagates rdf:type to canonical concepts")
    Rel(realizes, engine, "Writes REALIZES edges + provisional capabilities")
    Rel(realizes, cap_wb, "Hands minted capabilities for write-back")
    Rel(rest_fed, vendor_ext, "Invokes extractors at query-time (TTL-cached, no materialization)")
    Rel(brain_guard, backend, "Wraps the store: provenance + authority-arbitrated writes (KG_BRAIN_ENFORCE)")
    Rel(secured, engine, "Filters/scopes/audits reads on the facade path")
    Rel(feedback, govrules, "Persists rules consumed by")
    Rel(govrules, retrieval, "Re-ranks/filters designations at retrieval time")
    Rel(feedback, engine, "Writes Correction/rule/eval nodes")
    Rel(budget, retrieval, "Caps retrieved context to a token budget")
    Rel(streams, pipeline, "Feeds live events into ingestion")
    Rel(intel, pipeline, "Distils documents/calls into operating-intelligence nodes")
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
        Component(dasm, "🔬 Distributed Agent State Manager", "Python", "AHE-3.7: Optimistic locking with optional Redis support")
        Component(distill, "Workflow Distillation Hook", "Python", "ORCH-1.8: Auto-promotes successful patterns to Workflow Skills")
        Component(dspy, "DSPy Compiler", "Python", "AHE-3.1: Mathematical prompt optimization")
        Component(physdistill, "🔬 Physical Knowledge Distiller", "Python", "AHE-3.9: Distills evolved prompts/tools to physical git-tracked files")
        Component(dynoptimizer, "🔬 Dynamic Optimizer Selector", "Python", "AHE-3.10: Dynamically selects optimal optimizer (MIPROv2, FewShot, etc.) based on cluster scale")
        Component(gitops_bound, "🔬 GitOps Evolution Boundary", "Python", "AHE-3.11: Enforces git boundaries and registers evolutionary changes in KG")
    }

    Rel(eval, selfmodel, "Updates self-assessment scores")
    Rel(evolve, eval, "Triggers evolution on failure patterns")
    Rel(team, selfmodel, "Uses capability scores for composition")
    Rel(sdd, eval, "Validates features against KG integrity")
    Rel(team, dasm, "Syncs concurrent agent state")
    Rel(distill, team, "Promotes proven team compositions")
    Rel(distill, evolve, "Feeds back distilled patterns")
    Rel(evolve, dspy, "Offloads trace-based tuning to DSPy")
    Rel(evolve, physdistill, "Offloads evolved structures for physical write")
    Rel(physdistill, gitops_bound, "Triggers git changes and commits via boundaries")
    Rel(evolve, dynoptimizer, "Selects dynamic optimizer strategy based on failure characteristics")
```

### Pillar 4: Ecosystem Peripherals (ECO)

```mermaid
C4Component
    title ECO — Ecosystem Components

    Container_Boundary(eco, "Ecosystem Peripherals") {
        Component(mcp_factory, "MCP Server Factory", "Python + FastMCP", "create_mcp_server with auth stack")
        Component(kg_mcp, "KG MCP Server", "Python", "Thin wrapper exposing KG as MCP tools")
        Component(a2a, "A2A Network", "Python", "Agent-to-agent discovery and delegation")
        Component(coord_a2a, "🔬 Coordinated A2A Skill", "Python", "ECO-4.1: A2A with coordination negotiation. Research: 2605.03310v1")
        Component(skill_mgr, "Skill Manager", "Python", "Dynamic tool loading and skill evolution")
        Component(bridge, "Ecosystem Bridge", "Python", "Cross-package integration")
        Component(quantapi, "🔬 Quant Agent API (SAAPI)", "Python", "ECO-4.7: Base QuantAgentTemplate")
        Component(dataflows, "🔬 Market Dataflows", "Python", "ECO-4.8: Temporal ticker stream connector")
        Component(quant_mcp, "Unified Quant MCP Tool", "Python", "ECO-4.9: Single 'quant' tool routing to orchestrate, data, execute, portfolio")
        Component(toolkit_ingest, "Agent Toolkit Ingestor", "Python", "ECO-4.6: Unified MCP/Skill/A2A ingestion with auto-detection")
        Component(mcp_discover, "MCP Live Discovery", "Python", "ECO-4.6: Live list_tools() + KG cache + freshness verification")
        Component(quant_micro, "🔬 Microstructure Engine", "Python", "ECO-4.6: High-Frequency OBI & Micro-Price")
        Component(quant_arb, "🔬 Stat Arb Engine", "Python", "ECO-4.7: Cross-Market Cointegration & OU Modeling")
    }

    Rel(mcp_factory, kg_mcp, "Creates KG MCP instance")
    Rel(kg_mcp, a2a, "Shares KG data across agent network")
    Rel(coord_a2a, a2a, "Extends with coordination protocol negotiation")
    Rel(skill_mgr, bridge, "Loads skills from universal-skills")
    Rel(quantapi, a2a, "Provides trading execution interfaces")
    Rel(dataflows, quantapi, "Streams continuous temporal events")
    Rel(quant_mcp, dataflows, "Telemetry and Orders via SAAPI")
    Rel(toolkit_ingest, mcp_discover, "Delegates live tool discovery")
    Rel(toolkit_ingest, skill_mgr, "Ingests skill directories")
    Rel(toolkit_ingest, a2a, "Fetches A2A agent cards")
    Rel(mcp_discover, mcp_factory, "Connects to MCP servers via stdio")
    Rel(dataflows, quant_micro, "Feeds tick-level order book")
    Rel(quant_micro, quant_arb, "Provides micro-price edges")
    Rel(quant_arb, quant_mcp, "Generates stat-arb signals")
```

### Pillar 5: Agent OS Kernel (OS)

```mermaid
C4Component
    title OS — Agent OS Kernel Components

    Container_Boundary(os_k, "Agent OS Kernel") {
        Component(auth, "Security Policy Middleware", "Python", "JWT, API key, MCP auth")
        Component(threat, "Threat Defense Engine", "Python", "Prompt injection, jailbreak detection")
        Component(guardrails, "Guardrail Engine", "Python", "Tool guard, rate limit, content filter")
        Component(scheduler, "Cognitive Scheduler", "Python", "Priority queue, preemption, context paging")
        Component(budget, "🔬 Inference Budget Controller", "Python", "OS-5.2: Cost-aware tier fallback. Research: 2605.05701v1")
        Component(telemetry, "Telemetry Pipeline", "Python", "OTEL, token tracking, audit logging")
        Component(paths, "XDG Paths Module", "Python + platformdirs", "Centralized path resolution")
        Component(gateway, "Gateway Service Dashboard", "Python + FastAPI", "OS-5.9: 50-widget registry, aggregator, REST+WS API, MCP auto-discovery")
    }

    Rel(auth, threat, "Validates before routing")
    Rel(threat, guardrails, "Applies runtime constraints")
    Rel(guardrails, telemetry, "Records enforcement decisions")
    Rel(scheduler, budget, "Tracks cost + auto-downgrades model tier")
    Rel(paths, auth, "Provides config/data locations")
    Rel(paths, gateway, "XDG config + data paths")
    Rel(gateway, telemetry, "Reports widget fetch metrics")
```

### Pillar 6: GeniusBot Cockpit (GUI)

```mermaid
C4Component
    title GUI — GeniusBot Cockpit Components

    Container_Boundary(gui, "GeniusBot Cockpit") {
        Component(bridge, "AgentBridge", "Python", "Async Python-to-Qt bridge for agent I/O")
        Component(dashboard, "Systems Dashboard", "PySide6", "Real-time infrastructure health, container status, DNS")
        Component(finance, "Finance Cockpit", "PySide6 + QtCharts", "Portfolio analytics, P&L, risk dashboards")
        Component(chat, "Agent Chat", "PySide6 + QWebEngineView", "Conversational UI with streaming markdown")
        Component(kg_viz, "KG Visualizer", "PySide6 + D3.js", "Interactive graph exploration and traversal")
        Component(settings, "Settings Manager", "PySide6", "MCP server configuration, model selection, theme")
    }

    Rel(bridge, dashboard, "Pushes system metrics")
    Rel(bridge, finance, "Streams portfolio data")
    Rel(bridge, chat, "SSE streaming agent responses")
    Rel(bridge, kg_viz, "Graph query results")
    Rel(settings, bridge, "Configures agent connections")
```

## Cross-Pillar Data Flows

```mermaid
flowchart LR
    subgraph "Cross-Pillar Data Flows"
        direction TB

        subgraph INGEST ["Ingestion Flow"]
            direction LR
            ECO_MCP["ECO-4.0: MCP Tool Call"] --> ORCH_ROUTE["ORCH-1.2: Router"]
            ORCH_ROUTE --> KG_INGEST["KG-2.0: KG: Ingest Engine"]
            KG_INGEST --> KG_OWL["KG-2.2: KG: OWL Bridge"]
        end

        subgraph EXECUTE ["Execution Flow"]
            direction LR
            ORCH_PLAN["ORCH-1.1: Planner"] --> ORCH_DISPATCH["ORCH-1.0: Dispatcher"]
            ORCH_DISPATCH --> ECO_TOOL["ECO-4.0: ECO: Tool Executor"]
            ECO_TOOL --> OS_GUARD["OS-5.2: OS: Guardrails"]
            OS_GUARD --> AHE_EVAL["AHE-3.1: AHE: Evaluator"]
        end

        subgraph LEARN ["Learning Flow"]
            direction LR
            AHE_EVAL2["AHE-3.1: EvalRunner"] --> KG_MEMORY["KG-2.1: Memory Tier"]
            KG_MEMORY --> AHE_EVOLVE["AHE-3.2: AHE: Evolution"]
            AHE_EVOLVE --> ECO_SKILL["AHE-3.2: ECO: Skill Evolver"]
        end

        subgraph SECURE ["Security Flow"]
            direction LR
            OS_SCAN["OS-5.1: Threat Scanner"] --> KG_RISK["KG-2.2: Risk Ontology"]
            KG_RISK --> AHE_IMMUNE["AHE-3.3: AHE: Immunity"]
            AHE_IMMUNE --> OS_POLICY["OS-5.1: OS: Policy Engine"]
        end

        subgraph CONTINUOUS ["Continuous Ingestion Flow"]
            direction LR
            GIT_HOOK["Git: post-commit hook"] --> DIFF_SUBMIT["scripts/submit_diff.py"]
            DIFF_SUBMIT --> KG_TASKS["KG-2.0: KG: TaskManager"]
            KG_TASKS --> KG_DIFF["KG-2.0: KG: DiffEntry Node"]
        end

        subgraph LIFECYCLE ["Entity Lifecycle Flow"]
            direction LR
            KG_ACTIVE["KG-2.0: Active Node"] -->|"soft-delete"| KG_ARCHIVED["KG-2.0: status=ARCHIVED"]
            KG_ARCHIVED -->|"restore"| KG_ACTIVE2["KG-2.0: KG: status=ACTIVE"]
            KG_ARCHIVED -->|"hard-delete (age)"| KG_REMOVED["KG-2.0: KG: Permanently Removed"]
        end

        subgraph RESEARCH ["🔬 Research Integration Flow"]
            direction LR
            SCHOLAR["ECO-4.0: ScholarX Paper Search"] -->|"download"| KG_INGEST2["KG-2.6: Ingest Paper"]
            KG_INGEST2 -->|"discover mode"| KG_DISCOVER["ORCH-1.2: KG: Innovation Discovery"]
            KG_DISCOVER -->|"cross-ref"| CONCEPT_MAP["KG-2.2: KG: Concept Map"]
            CONCEPT_MAP -->|"assimilate"| KG_ASSIMILATE["KG-2.0: KG: ASSIMILATED_INTO edges"]
        end

        subgraph ENTERPRISE ["Enterprise Federation Flow"]
            direction LR
            KG_MATERIALIZE["KG-2.2: OWL Materialize"] -->|"rdflib"| SPARQL_EP["KG-2.6: SPARQL HTTP Endpoint"]
            SPARQL_EP -->|"query"| EXT_CONSUMER["ECO-4.0: External Consumer"]
            KG_MATERIALIZE -->|"validate"| SHACL_V["KG-2.2: SHACL Validator"]
            KG_MATERIALIZE -->|"export"| ONT_PUB["KG-2.2: Ontology Publisher"]
            ONT_PUB -->|"push"| STARDOG["KG-2.6: Stardog / Fuseki"]
            STARDOG -->|"owl:imports"| ONT_LOAD["KG-2.2: Ontology Loader"]
            ONT_LOAD -->|"merge"| KG_MATERIALIZE
        end

        subgraph VENDORNEUTRAL ["Vendor-Neutral Crosswalk Flow (KG-2.9)"]
            direction LR
            VN_SN["ServiceNow :Incident"] -->|"extractor"| VN_NODES["KG-2.9: Canonical GraphNodes"]
            VN_ERP["ERPNext :ErpNextIssue"] -->|"extractor"| VN_NODES
            VN_CAM["Camunda :BusinessTask"] -->|"extractor"| VN_NODES
            VN_NODES -->|"promote"| VN_REASON["KG-2.2: owl_bridge + HermiT"]
            VN_REASON -->|"subClassOf / equivalentClass"| VN_CANON["KG-2.9: :ApplicationEvent / :BusinessProcess"]
            VN_CANON -->|"one query, all vendors"| VN_QUERY["KG-2.6: SELECT ?e a :ApplicationEvent"]
            VN_LIVE["KG-2.1: register_rest_source"] -.->|"query-time, TTL-cached"| VN_QUERY
        end

        subgraph REALIZES_FLOW ["Code → Capability Flow (KG-2.8)"]
            direction LR
            RZ_CODE["KG-2.7: Rust AST → features"] -->|"resolve_realizes"| RZ_MATCH["KG-2.8: match / mint / registry"]
            RZ_LEANIX["LeanIX/Archi BusinessCapability"] --> RZ_MATCH
            RZ_MATCH -->|"REALIZES edge"| RZ_KG["KG-2.0: KG Persistence"]
            RZ_MATCH -->|"provisional capability"| RZ_WB["KG-2.8: capability_writeback"]
            RZ_WB -->|"add_element / postbusinesscapability"| RZ_EA["Archi / LeanIX"]
        end

        subgraph MATERIALIZE ["KG Graph Materialization Flow (ORCH-1.20)"]
            direction LR
            QUERY_IN["ORCH-1.0: User Query"] -->|"router_step"| KG_SEARCH["KG-2.3: Hybrid Search"]
            KG_SEARCH -->|"AgentTemplate nodes"| TOPO_SORT["ORCH-1.20: Factory: Topological Sort"]
            TOPO_SORT -->|"DEPENDS_ON edges"| PROMPT_RESOLVE["ORCH-1.20: Factory: Prompt Resolution"]
            PROMPT_RESOLVE -->|"USES_PROMPT edges"| TOOL_BIND["ORCH-1.20: Factory: Tool Binding"]
            TOOL_BIND -->|"REQUIRES_TOOLSET edges"| GRAPH_BUILD["ORCH-1.20: Factory: Graph Build"]
            GRAPH_BUILD -->|"KGGraphResult"| DISPATCH_OUT["ORCH-1.0: Dispatcher"]
        end

        subgraph TOOLKIT ["Agent Toolkit Ingestion Flow (ECO-4.6 / ECO-4.6)"]
            direction LR
            TK_SRC["ECO-4.1: Sources: mcp_config.json / skill dirs / A2A URLs"] -->|"auto-detect"| TK_DETECT["ECO-4.6: Type Detector"]
            TK_DETECT -->|"JSON + mcpServers"| TK_MCP["ECO-4.6: MCP Config Parser"]
            TK_DETECT -->|"directory + SKILL.md"| TK_SKILL["ECO-4.6: Skill Parser"]
            TK_DETECT -->|"http:// URL"| TK_A2A["ECO-4.1: A2A Card Fetcher"]
            TK_MCP -->|"live connect"| TK_LIVE["ECO-4.6: Live list_tools()"]
            TK_LIVE -->|"tool metadata"| TK_KG["ECO-4.6: KG: Server + CallableResource nodes"]
            TK_MCP -->|"fallback"| TK_FLAGS["ECO-4.6: Tool Flag Parser"]
            TK_FLAGS --> TK_KG
            TK_SKILL --> TK_KG
            TK_A2A -->|"/.well-known/agent.json"| TK_KG
            TK_KG -->|"config hash"| TK_FRESH["ECO-4.6: Freshness Check"]
        end

        subgraph AGENT_EXEC ["Agent Execution Flow (ORCH-1.21)"]
            direction LR
            AE_CMD["ORCH-1.21: graph_orchestrate: execute_agent"] -->|"agent_name"| AE_RESOLVE["ORCH-1.20: Agent Resolution"]
            AE_RESOLVE -->|"Server/Skill/A2A nodes"| AE_CONFIG["ORCH-1.21: Config Builder"]
            AE_CONFIG -->|"tag_prompts + mcp_toolsets"| AE_GRAPH["ORCH-1.20: create_graph_agent()"]
            AE_GRAPH -->|"materialized graph"| AE_RUN["ORCH-1.21: run_graph() → LM Studio"]
            AE_RUN -->|"GraphResponse"| AE_TRACE["OS-5.4: KG: RunTrace provenance"]
        end

        subgraph WORKFLOW ["Workflow Lifecycle Flow (ORCH-1.22 / 1.23 / 1.24)"]
            direction LR
            WF_YAML["ORCH-1.24: catalog.yaml"] -->|"load()"| WF_CATALOG["ORCH-1.24: WorkflowCatalog"]
            WF_NL["User: Natural Language"] -->|"compile()"| WF_COMPILER["ORCH-1.23: WorkflowCompiler"]
            WF_CATALOG -->|"to_graph_plans()"| WF_PLANS["GraphPlan[]"]
            WF_COMPILER -->|"NL → DAG"| WF_PLANS
            WF_CATALOG -->|"register_in_kg()"| WF_STORE["ORCH-1.22: WorkflowStore"]
            WF_COMPILER -->|"compile_and_store()"| WF_STORE
            WF_STORE -->|"KG: WorkflowDefinition"| WF_KG["KG-2.0: KG Persistence"]
            WF_KG -->|"load_workflow()"| WF_PLANS
            WF_PLANS -->|"execute()"| WF_RUNNER["ORCH-1.24: WorkflowRunner"]
            WF_RUNNER -->|"wave-based dispatch"| AE_RESOLVE2["ORCH-1.21: run_agent()"]
            WF_RUNNER -->|"session traces"| WF_LANGFUSE["OS-5.1: Langfuse"]
        end

        subgraph DISTILL ["Workflow Distillation Flow (ORCH-1.8)"]
            direction LR
            WD_SYNTH["ORCH-1.0: Synthesizer"] -->|"success"| WD_HOOK["ORCH-1.8: Distillation Hook"]
            WD_HOOK -->|"threshold met"| WD_STORE["ORCH-1.22: WorkflowStore"]
            WD_HOOK -->|"promote"| WD_TEAM["AHE-3.3: TeamConfig Composer"]
            WD_STORE -->|"versioned"| WD_KG["KG-2.0: KG Persistence"]
            WD_TEAM -->|"proven team"| WD_KG
            WD_KG -->|"bundle export"| WD_BUNDLE["ORCH-1.8: Bundle Exporter"]
            WD_BUNDLE -->|"YAML / JSON"| WD_PRESET["ORCH-1.8: Domain Presets"]
            WD_PRESET -->|"seed_into_kg()"| WD_KG
        end

        subgraph GATEWAY ["Gateway Service Dashboard Flow (OS-5.9)"]
            direction LR
            GW_MCP["OS-5.9: mcp_config.json"] -->|"auto-discover"| GW_CONFIG["OS-5.9: ConfigManager"]
            GW_CONFIG -->|"ServiceConfig[]"| GW_REG["OS-5.9: Widget Registry"]
            GW_REG -->|"lazy-import"| GW_WIDGET["OS-5.9: 50 Widget Modules"]
            GW_WIDGET -->|"fetch_data()"| GW_AGG["OS-5.9: Aggregator"]
            GW_AGG -->|"WidgetData{}"| GW_API["OS-5.9: REST /api/dashboard"]
            GW_AGG -->|"stream"| GW_WS["OS-5.9: WebSocket /ws/dashboard"]
            GW_API -->|"JSON"| GW_WEBUI["agent-webui"]
            GW_WS -->|"real-time"| GW_WEBUI
            GW_AGG -->|"direct Python"| GW_TUI["agent-terminal-ui"]
            GW_AGG -->|"QThread"| GW_GUI["geniusbot"]
        end
    end
```

## Pillar Interconnection Matrix

```mermaid
graph TD
    subgraph "Pillar Interconnection Matrix"
            P1["<b>ORCH-1.0: Orchestration</b><br/>Orchestrates multi-agent workflows"]
            P2["<b>KG-2.0: Knowledge Graph</b><br/>epistemic-graph + PostgreSQL semantic engine"]
            P3["<b>AHE-3.0: Agentic Harness</b><br/>Continuous evaluation and evolution"]
            P4["<b>ECO-4.0: Ecosystem</b><br/>MCP server connections and APIs"]
            P5["<b>OS-5.0: Agent OS</b><br/>Runtime environment and security"]
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
    P5 -->|"🔬 InferenceBudget tracks cost<br/>auto-downgrades model tier"| P1
    P1 -->|"🔬 CoordinationLayer selects<br/>protocol per team composition"| P4

    style P1 fill:#dae8fe,stroke:#6c8ebf,stroke-width:3px
    style P2 fill:#d5e8d4,stroke:#82b366,stroke-width:3px
    style P3 fill:#fff2cc,stroke:#d6b656,stroke-width:3px
    style P4 fill:#e6ccff,stroke:#9673a6,stroke-width:3px
    style P5 fill:#cce5ff,stroke:#004085,stroke-width:3px
```

> **Key Insight**: Every pillar has at least one bidirectional dependency with another pillar.
> The system is a closed feedback loop, not a layered stack. This is why isolated concept
> additions are dangerous — they must wire into the loop.


### Ecosystem Dependency Graph

```mermaid
graph TD
    subgraph Packages ["Core Ecosystem Packages"]
        direction TB
        Utility["<b>agent-utilities</b><br/>(Python)"]
        Terminal["<b>agent-terminal-ui</b><br/>(Python/Textual)"]
        Web["<b>agent-webui</b><br/>(React/Next.js)"]
        Genius["<b>geniusbot</b><br/>(Python/PySide6)"]
    end

    subgraph Internal_Deps ["Internal Interface Layer"]
        direction LR
        Terminal -- depends on --> Utility
        Web -- interfaces with --> Utility
        Genius -- interfaces with --> Utility
        Terminal -. "gateway.Aggregator" .-> Utility
        Web -. "gateway.api + ws" .-> Utility
        Genius -. "gateway.Aggregator" .-> Utility
    end

    subgraph External_Utility ["agent-utilities Dependencies"]
        direction TB
        PAI[pydantic-ai]
        PGraph[pydantic-graph]
        PACP[pydantic-acp]
        PAISkills[pydantic-ai-skills]
        FastMCP[ECO-4.0: fastmcp]
        FastAPI[fastapi]
        Logfire[OS-5.4: logfire]
    end

    subgraph External_Terminal ["agent-terminal-ui Dependencies"]
        direction TB
        Textual[textual]
        Rich[rich]
        HTTPX_T[httpx]
    end

    subgraph External_Web ["agent-webui Dependencies"]
        direction TB
        ASDK["@ai-sdk/react (Vercel)"]
        AI["ORCH-1.0: ai (Vercel SDK)"]
        React[react]
        Tailwind[ECO-4.0: tailwindcss]
        Vite[vite]
    end

    subgraph External_Genius ["geniusbot Dependencies"]
        direction TB
        PySide[PySide6]
        QtCharts[QtCharts]
        WebEngine[QWebEngineView]
    end

    Utility --> PAI
    Utility --> PGraph
    Utility --> PACP
    Utility --> PAISkills
    Utility --> FastMCP
    Utility --> FastAPI
    Utility --> Logfire

    Terminal --> Textual
    Terminal --> Rich
    Terminal --> HTTPX_T

    Web --> ASDK
    Web --> AI
    Web --> React
    Web --> Tailwind
    Web --> Vite

    Genius --> PySide
    Genius --> QtCharts
    Genius --> WebEngine
```

### C4 Container Diagram
```mermaid
C4Container
    title Container diagram for Agent Orchestration System

    Person(user, "User", "Interacts via Web UI")

    Container_Boundary(c1, "Agent Ecosystem") {
        Container(webui, "Agent WebUI", "React, Tailwind", "Renders streaming responses and graph activity visualization")
        Container(tui, "Agent Terminal UI", "Python, Textual", "Provides a high-performance terminal interface for direct CLI interaction")
        Container(gateway, "Agent Gateway (FastAPI)", "Python, Pydantic-AI", "Handles ACP sessions and SSE streams, merges graph events into chat annotations")
        Container(orchestrator, "Graph Orchestrator", "Pydantic-Graph", "Routes queries, executes parallel domains, validates results")
        Container(subagent, "Domain Sub-Agents", "Pydantic-AI", "Specialized agents for Git, Web, Cloud, etc.")
    }

    System_Ext(mcp, "MCP Servers", "Contextual tools (GitHub, Slack, etc.)")
    System_Ext(otel, "OpenTelemetry Collector", "Tracing and monitoring")

    Rel(user, webui, "Uses", "HTTPS/WSS")
    Rel(user, tui, "Uses", "Terminal/CLI")
    Rel(webui, gateway, "Queries", "ACP /acp (SSE/RPC)")
    Rel(tui, gateway, "Queries", "ACP /acp (SSE/RPC)")
    Rel(gateway, orchestrator, "Dispatches", "Async Python")
    Rel(orchestrator, subagent, "Delegates", "Parallel Execution")
    Rel(subagent, mcp, "Invokes Tools", "JSON-RPC (stdio/SSE)")
    Rel(orchestrator, otel, "Exports Spans", "OTLP")
```
