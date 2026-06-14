# Agent Utilities — Concept Overview

> See [docs/concept_map.md](concept_map.md) for the canonical concept registry.
> See [docs/pillars/architecture_c4.md](pillars/architecture_c4.md) for C4 architecture diagrams.

## Pillar Summaries

1. [Pillar 1: Graph Orchestration Engine](pillars/1_graph_orchestration.md)
2. [Pillar 2: Epistemic Knowledge Graph](pillars/2_epistemic_knowledge_graph.md)
3. [Pillar 3: Agentic Harness Engineering](pillars/3_agentic_harness_engineering.md)
4. [Pillar 4: Ecosystem & Peripherals](pillars/4_ecosystem_peripherals.md)
5. [Pillar 5: Agent OS Infrastructure](pillars/5_agent_os_infrastructure.md)

## Engine Facades

| Engine | Concept | Path | Description |
|--------|---------|------|-------------|
| **IntelligenceGraphEngine** | KG-2.0 | `knowledge_graph/core/engine.py` | Core graph engine with 15-phase pipeline |
| **TopologicalAnalysisEngine** | KG-2.5 | `knowledge_graph/core/topological_analysis_engine.py` | Analogy, spectral, blast radius |
| **EvaluationEngine** | AHE-3.1 | `harness/evaluation_engine.py` | Decomposed reward signals + trace distillation |
| **ParallelEngine** | ORCH-1.8 | `graph/parallel_engine.py` | Unified 1→300+ agent parallel execution engine |
| **Gateway Aggregator** | OS-5.9 | `gateway/aggregator.py` | 50-widget parallel service dashboard data layer |

## The 5 Core Pillars Architecture

```mermaid
graph TD
    %% Pillar 1: Graph Orchestration Engine
    subgraph P1 ["Pillar 1: Graph Orchestration Engine"]
        ORCH10["<b>ORCH-1.0: Core Orchestration Engine</b>"]
        ORCH11["<b>ORCH-1.1: HTN Planning Pipeline</b>"]
        ORCH12["<b>ORCH-1.2: Specialist Routing</b>"]
        ORCH13["<b>ORCH-1.3: Execution Safety</b>"]
        ORCH14["<b>ORCH-1.4: Capability Wiring</b>"]
        ORCH16["<b>ORCH-1.5: DSTDD Pipeline</b>"]
        ORCH17["<b>ORCH-1.6: Prediction Linkage Layer</b>"]
        ORCH18["<b>ORCH-1.7: RecursiveMAS Latent Orchestrator</b>"]
        ORCH125["<b>ORCH-1.8: Parallel Execution & Synthesis Engine</b>"]
        ORCH127["<b>ORCH-1.9: Dept Orchestration</b>"]
        ORCH128["<b>ORCH-1.10: Reactive Dispatch</b>"]
        ORCH129["<b>ORCH-1.11: WASM Sandbox</b>"]
        ORCH141["<b>ORCH-1.41-1.43: Ontology-to-Workflow Execution</b>"]
        ORCH145["<b>ORCH-1.45: Queue-Driven Agent Dispatch</b>"]
    end

    %% Pillar 2: Epistemic Knowledge Graph
    subgraph P2 ["Pillar 2: Epistemic Knowledge Graph"]
        KG20["<b>KG-2.0: Active Knowledge Graph</b>"]
        KG21["<b>KG-2.1: Tiered Memory</b>"]
        KG22["<b>KG-2.2: Ontology & Epistemics</b>"]
        KG23["<b>KG-2.3: Unified Retrieval & Graph Integrity</b>"]
        KG24["<b>KG-2.4: Inductive Hypergraphs</b>"]
        KG25["<b>KG-2.5: Topological Analysis</b>"]
        KG26["<b>KG-2.6: Domain Ontologies & Vertical Subgraphs</b>"]
        KG28["<b>KG-2.6: Memory Stability</b>"]
        KG29["<b>KG-2.7: Quant Orchestration</b>"]
        KG215["<b>KG-2.7: Transaction Proxy</b>"]
        KG216["<b>KG-2.7: Rust-Native High-Performance Compute (FFI)</b>"]
        KG219["<b>KG-2.7: Event Backbone</b>"]
        KG220["<b>KG-2.7: Query Router</b>"]
        KG221["<b>KG-2.21: Working Set Manager</b>"]
        KG260["<b>KG-2.7: Single Company Brain</b>"]
        KG255["<b>KG-2.55-2.57: Kafka Ingest Scale-Out</b>"]
        KG258["<b>KG-2.58: Tenant-Sharded Engines (HRW)</b>"]
    end

    %% Pillar 3: Agentic Harness Engineering
    subgraph P3 ["Pillar 3: Agentic Harness Engineering"]
        AHE30["<b>AHE-3.0: Harness Core</b>"]
        AHE31["<b>AHE-3.1: Evaluation Engine</b>"]
        AHE32["<b>AHE-3.2: Evolution Engine</b>"]
        AHE33["<b>AHE-3.3: Team Optimization</b>"]
        AHE34["<b>AHE-3.4: Distributed Evolution</b>"]
        AHE35["<b>AHE-3.5: Heavy Thinking</b>"]
        AHE36["<b>AHE-3.6: Backtest & Curriculum</b>"]
        AHE315["<b>AHE-3.8: Interpretability & Model Evolution</b>"]
        AHE321["<b>AHE-3.18-3.21: Failure-Driven Evolution & Branch Publication</b>"]
    end

    %% Pillar 4: Ecosystem & Peripherals
    subgraph P4 ["Pillar 4: Ecosystem & Peripherals"]
        ECO40["<b>ECO-4.0: Tool Interface & MCP</b>"]
        ECO41["<b>ECO-4.1: A2A Network</b>"]
        ECO42["<b>ECO-4.2: Telemetry & Ecosystem</b>"]
        ECO43["<b>ECO-4.3: Market Data</b>"]
        ECO44["<b>ECO-4.4: KG MCP Server</b>"]
        ECO410["<b>ECO-4.6: Dynamic Capability Ingestion & Discovery</b>"]
        ECO414["<b>ECO-4.14: Infrastructure Blueprint Library</b>"]
        ECO415["<b>ECO-4.9: Queue Backend</b>"]
        ECO416["<b>ECO-4.10: Automated Documentation & AGENTS.md Governance</b>"]
        ECO418["<b>ECO-4.11: Lint Enforcement</b>"]
        ECO419["<b>ECO-4.12: Plugin Bundles</b>"]
        ECO420["<b>ECO-4.13: Ecosystem Governance & Policy Engine</b>"]
        ECO434["<b>ECO-4.34: Multiplexer Child Resilience</b>"]
    end

    %% Pillar 5: Agent OS Infrastructure
    subgraph P5 ["Pillar 5: Agent OS Infrastructure"]
        OS50["<b>OS-5.0: Agent OS Kernel</b>"]
        OS51["<b>OS-5.1: Security & Auth</b>"]
        OS52["<b>OS-5.2: Resource Scheduling</b>"]
        OS53["<b>OS-5.3: OS Guardrails & Safety Boundaries</b>"]
        OS54["<b>OS-5.4: Telemetry</b>"]
        OS56["<b>OS-5.5: Massive Scale</b>"]
        OS514["<b>OS-5.14: Server-Minted JWT Identity</b>"]
        OS516["<b>OS-5.16-5.18: Externalized Durable State</b>"]
        OS523["<b>OS-5.23: Gateway Hardening & /metrics</b>"]
        OS524["<b>OS-5.24-5.29: Fleet Autonomy Control Plane</b>"]
    end

    %% Cross-pillar relationships
    ORCH10 <--> KG20
    ORCH11 --> ORCH12
    ORCH12 --> KG22
    ORCH14 --> ECO40
    KG21 --> KG20
    KG25 --> KG20
    KG26 --> KG20
    KG29 --> KG20
    KG215 --> KG20
    KG216 --> KG20
    KG219 --> KG20
    KG220 --> KG20
    KG221 --> KG216
    KG260 --> KG22
    AHE31 --> KG20
    AHE33 --> ORCH12
    AHE34 --> ECO42
    ECO44 --> KG20
    ECO410 --> ECO40
    ECO415 --> ECO44
    OS51 --> ORCH13
    OS53 --> OS51
    OS54 --> KG20
    OS56 --> OS50
    ECO416 --> KG20
    ECO420 --> OS51
    ORCH145 --> OS516
    KG255 --> KG20
    KG258 --> KG20
    AHE321 --> OS524
    ECO434 --> OS523
    OS514 --> OS51
    OS524 --> OS514

    style P1 fill:#dae8fe,stroke:#6c8ebf,stroke-width:2px
    style P2 fill:#d5e8d4,stroke:#82b366,stroke-width:2px
    style P3 fill:#fff2cc,stroke:#d6b656,stroke-width:2px
    style P4 fill:#e6ccff,stroke:#9673a6,stroke-width:2px
    style P5 fill:#cce5ff,stroke:#004085,stroke-width:2px
```

## Concept Index

> **Canonical Registry**: See [concept_map.md](concept_map.md) for the full canonical concept registry with module paths.

### Pillar 1: Graph Orchestration Engine (ORCH-1.0 – 1.45)

| ID | Concept | Description |
|---|---|---|
| ORCH-1.0 | Core Orchestration Engine | Pydantic Graph-based DAG execution with state management and multi-agent execution |
| ORCH-1.1 | HTN Planning Pipeline | Recursive hierarchical task network decomposition |
| ORCH-1.2 | Specialist Routing & Discovery | Ontological routing, specialist tag loading, and fallback chains |
| ORCH-1.3 | Execution Safety & State | Checkpointing, retry, and state persistence |
| ORCH-1.4 | Capability Wiring Engine | Dynamic capability discovery and capability auto-activation |
| ORCH-1.5 | DSTDD Pipeline | Design-Spec-Test Driven Development lifecycle |
| ORCH-1.6 | Prediction Linkage Layer 🔬 | Prediction linking across execution iterations |
| ORCH-1.7 | RecursiveMAS Latent Orchestrator 🔬 | Continuous latent space multi-agent recursion and projection |
| ORCH-1.8 | Parallel Execution & Synthesis Engine | Unified 1→300+ agent execution engine with concurrency, DAG scheduling, and output synthesis |
| ORCH-1.9 | Autonomous Department Orchestration | OWL-materialized company departments with `reportsTo` hierarchy |
| ORCH-1.10 | Reactive Event Sourcing | Reactive event-driven state and graph staging dispatcher |
| ORCH-1.11 | WASM Micro-Agent Sandbox | Isolated micro-agent WebAssembly sandbox runner with gas/memory limits and Python emulation fallback |
| ORCH-1.27 | Role-Specialized Model Routing | Binds planner/generator/learner/judge + RLM (executor/proposer/sub-LM) roles to model tiers+tags over the registry |
| ORCH-1.28 | Composable Skills + Generic Adapter | Structured Skill units (instructions+packages+modules+tools) + merge; minimal generic env adapter preserving the host evaluator |
| ORCH-1.29 | RLM Resilience + Telemetry | Structured RunTrace + FailureClass taxonomy; recoverable tool timeout vs fatal sandbox error |
| ORCH-1.30 | Generalizing GEPA | Held-out feedback/Pareto split + AgentSpec anti-overfit grounding + held-out candidate selection (transferable skills) |
| ORCH-1.31 | Graph-Native Optimization State | Persist the GEPA Pareto frontier + ancestry to the epistemic-graph; resumable, cross-session optimization |
| ORCH-1.41 | Process Plan Compiler | `compile_process` MCP/REST: lifts a descriptive BPMN process into an executable plan (`ProcessPlanCompiler`) |
| ORCH-1.42 | Execution Ontology Gate | Ontology validation on the execution path (`workflow_gate.py`) before a compiled process runs |
| ORCH-1.43 | Workflow Lineage Close-Out | Run lineage written back to the KG, closing the descriptive↔executable provenance loop |
| ORCH-1.44 | Durable Goal Registry | Goals persist across restarts; stranded runs rehydrate as orphaned instead of silently vanishing |
| ORCH-1.45 | Queue-Driven Agent Dispatch | Session-keyed `agent_turns` queue (`AgentTurnEnvelope`) + stateless `agent-dispatch-worker` fleet with fleet-visible placement |

### Pillar 2: Epistemic Knowledge Graph (KG-2.0 – 2.58)

| ID | Concept | Description |
|---|---|---|
| KG-2.0 | Active Knowledge Graph | Core 15-phase pipeline, OGM, IntelligenceGraphEngine |
| KG-2.1 | Tiered Memory & Context 🔬 | Episodic/semantic/procedural memory, context compaction |
| KG-2.2 | Ontology & Epistemics | OWL ontology bridge, FIBO/BFO, semantic subsumption |
| KG-2.3 | Unified Retrieval & Graph Integrity 🔬 | Fingerprinting, vectorized semantic indexing, hybrid retriever, consistency validation |
| KG-2.4 | Inductive Knowledge | Knowledge synthesis and cross-pillar synergy engine |
| KG-2.5 | Topological Analysis | Analogy engine, spectral clusters, blast radius |
| KG-2.6 | Domain Ontologies & Vertical Subgraphs | Aggregated vertical domains including Finance, Enterprise, Company Operations, and Research |
| KG-2.6 | Memory Stability | Self-reflecting memory observer and stability checks |
| KG-2.7 | Multi-Domain Architecture | Decoupled graph frameworks and multi-domain graph orchestration |
| KG-2.7 | Transaction Proxy | Centralized gateway and transactional persistence layer |
| KG-2.7 | Rust-Native High-Performance Compute | High-performance quantitative execution, graph traversal, and epistemic reasoning via the out-of-process epistemic-graph engine (MessagePack/UDS client; no PyO3) plus Rustworkx |
| KG-2.7 | Event Backbone | Protocol-based pub/sub with MemoryEventBackend (default) and RedpandaEventBackend (distributed) |
| KG-2.7 | Query Router | Cost-based query planner routing to L1 Rust / L2 Cache / L3 Persistent / L4 Vector tiers |
| KG-2.11 | Bi-Temporal Memory Layers | Event-time vs storage-time + valid_from/valid_to on the graph; as-of queries and event-time contradiction precedence; procedural memory layer |
| KG-2.12 | Memory-First Retrieval (HyDE) | HyDE query expansion + dual thresholds + self-correcting two-pass + quantitative-fidelity ledger over the hybrid retriever |
| KG-2.13 | Background Learning Engine | Async, semaphore-bounded learner emitting typed, outcome-grounded ADD/UPDATE/DELETE bi-temporal memory edits |
| KG-2.14 | Ground-Truth Context Authority | Authority-ranked startup context + a Ground-Truth preamble so injected memory is treated as authoritative (no re-fetching) |
| KG-2.15 | Resilient Retrieval | 4-level retrieval fallback cascade + social-closer triviality gate |
| KG-2.17 | Memory Hygiene | Scheduled decay scanner (archive via valid_to, never delete) + semantic-merge dedup |
| KG-2.18 | Evidence-Weighted Memory | Bayesian trust feedback loop + recall/usage telemetry + generation lineage extending the quality gate |
| KG-2.19 | Self-Curating Wiki | Delta-skip (SHA-256) continuous ingest of a markdown knowledge vault, reusing the ingestion engine + synthesis |
| KG-2.20 | Rust-Native Finance Compute Suite | epistemic-graph quant kernels (KG-2.20f/g/h/i): market-making (Avellaneda-Stoikov/GLT/logit), microstructure (OFI/VPIN/microprice/Hawkes), sizing (Kelly/Bayesian/empirical), validation (purged-CPCV/DSR/PBO/Brier), forensic scores, state-space/stat-arb (Kalman/OU/ADF), signal combination (alpha-engine/IR=IC√N) |
| KG-2.21 | Working Set Manager | LRU-evicting subgraph cache for L1 Rust engine with 50K node cap |
| KG-2.22 | Data Science Primitives | Rust-backed OLS / K-means / PCA / estimators (ridge/lasso/RF/GB/SVR) replacing scikit-learn on the hot path, parity-validated |
| KG-2.7 | Single Company Brain | Extensible operational state layer encompassing Ontology Bridges, Enterprise Architecture Repositories, and Entailment-Aware Permissions |
| KG-2.49 | Remote VCS Enumeration | Enterprise-scale ingestion: enumerate every repository across a GitHub org/user or GitLab instance/groups (keyset / affiliation pagination) into a manifest for bulk workspace onboarding (repository-manager `vcs_enumerator`) |
| KG-2.52 | Ontology Publisher Tick | Background publish of the authoritative TBox to Fuseki (`core/ontology_publisher.py`) |
| KG-2.53 | BPMN Process Lift | Step-level shape for the descriptive process world (Camunda extractor + `owl_bridge`) |
| KG-2.54 | Cross-Host Task Queue | Atomic SKIP LOCKED claims + visibility-timeout recovery on the shared Postgres state store |
| KG-2.55 | Fail-Loud Queue Backend Selection | `TASK_QUEUE_BACKEND=sqlite\|postgres\|kafka`; explicit backends fail loud at startup instead of silently degrading |
| KG-2.56 | Keyed Ingest Partitions | `kg_tasks` partition keys (tenant → repo/corpus → task type) for per-tenant/per-repo ordering |
| KG-2.57 | Decoupled kg-ingest Consumer Group | `kg-ingest-worker` runs ingest as engine clients on any host; at-least-once, idempotent claims, lag metrics |
| KG-2.58 | Tenant-Partitioned Engine Sharding | HRW graph→shard routing over `GRAPH_SERVICE_ENDPOINTS` with tenant→named-graph placement |
| KG-2.70 | Evidence-Subgraph Task Synthesis | Build a bounded evidence-graph workspace around an answer entity for shortcut-resistant deep-search task synthesis (`knowledge_graph/search_synthesis/evidence_subgraph.py`; distills FORT-Searcher arXiv:2606.12087) |
| KG-2.71 | Shortcut-Risk Detectors | Four graph-query detectors — single-clue selectivity, evidence co-coverage, exposed constants, prior-knowledge binding — over the evidence graph (`knowledge_graph/search_synthesis/shortcut_risks.py`) |
| KG-2.72 | Question Formulation & Adversarial Refinement | Render a clue subgraph as a verifiable question (name withholding) and adversarially refine until no shortcut trips (`knowledge_graph/search_synthesis/question_formulation.py`) |
| KG-2.73 | Learned World-Model Backend + SAI Track | Parametric latent-dynamics backend for the world model that generalizes to unseen `(state, action)` (ridge map over embeddings), plus a `WorldModelVerifier` making prediction accuracy a SAI specialization domain (`knowledge_graph/core/world_model.py`, `harness/world_model_task.py`) |

### Pillar 3: Agentic Harness Engineering (AHE-3.0 – 3.21)

| ID | Concept | Description |
|---|---|---|
| AHE-3.0 | Agentic Harness Core | Harness lifecycle, initialization, SDD integration |
| AHE-3.1 | Continuous Evaluation Engine | Multi-strategy EvalRunner, decomposed rewards |
| AHE-3.2 | Agentic Evolution Engine | Skill neologism, config versioning, variant pool |
| AHE-3.3 | Team & Synergy Optimization | TeamConfig, coalition composition, synergy scoring |
| AHE-3.4 | Distributed Agentic Evolution | Self-model, stability, ecosystem PR generation |
| AHE-3.5 | Heavy Thinking & Background Intelligence | Heavy thinking, background intelligence |
| AHE-3.6 | Backtest & Curriculum | Backtest harness, horizon-aware curriculum |
| AHE-3.8 | Interpretability & Model Evolution | Agent-Interpretable Model Evolver workflows and LLM-Graded Interpretability Tests |
| AHE-3.12 | LongMemEval-S Validation Harness | FastAPI /benchmark surface (Quarq-runner compatible) + frozen corpus + CI floor gate proving the memory-first stack vs 98.2% |
| AHE-3.18 | Failure-Driven Evolution | Langfuse failure ingest → `failure_gap` Concept topics → golden-loop remediation → regression-gated merge |
| AHE-3.19 | Performance Anomaly Consumer | Turns persisted `PerformanceAnomaly` nodes into evolution topics (`adaptation/anomaly_consumer.py`) |
| AHE-3.20 | Promotion Governance Validator | Governed validation gate every promoted proposal must pass (`research/promotion_governance.py`) |
| AHE-3.21 | Evolution-to-Branch Bridge | Change synthesis + RLM-sandbox validation + ActionPolicy-gated `ChangePublisher` publishing promoted proposals as reviewable local git branches |
| AHE-3.27 | Adaptation-Speed Metric | SAI primary measure — per-task time-to-target + sample-complexity + learning-AUC over a verified-reward curve (`harness/adaptation_speed.py`) |
| AHE-3.28 | Specialization Task + Verifier Contract | The `(task, verifier, target, human-baseline)` contract + machine-verifiable `Verifier` protocol every specialization track shares (`harness/sai_task.py`) |
| AHE-3.29 | SAI Factory Controller | Closed scaffolding+weights specialization loop steered by adaptation speed, ratchet-gated promotion (`research/sai_factory.py`) |
| AHE-3.30 | Realized Search-Difficulty Signatures | Trajectory signatures — solving cost, answer hit time, prior-shortcut rate — that diagnose whether a deep-search task forced real search and gate task acceptance (`graph/training_signals.py`; distills FORT-Searcher arXiv:2606.12087) |

### Pillar 4: Ecosystem & Peripherals (ECO-4.0 – 4.34)

| ID | Concept | Description |
|---|---|---|
| ECO-4.0 | Tool Interface & MCP Factory | MCP server factory, skill loading, tool assignment |
| ECO-4.1 | A2A Network & Consensus 🔬 | Agent-to-agent discovery, delegation, consensus |
| ECO-4.2 | Community Telemetry & Ecosystem Map | Ecosystem topology, 40-repo graph, telemetry |
| ECO-4.3 | Market Data KG Node Models | Connector/fetch-record ontology nodes for data-source provenance |
| ECO-4.4 | KG MCP Server & Execution | KG MCP exposure, durable execution, sandbox |
| ECO-4.6 | Dynamic Capability Ingestion & Discovery | Ingests external agent toolkits, discovers MCP endpoints in real-time, and builds self-documenting skill-graphs |
| ECO-4.7 | Domain Workflow Bindings | Parallel execution workflows and capability bindings for specialized domain processes |
| ECO-4.9 | Queue Backend | Abstract QueueBackend with Memory, Nats, and Kafka implementations for multi-scale event distribution |
| ECO-4.10 | Automated Documentation & AGENTS.md Governance | Deterministic hierarchical AGENTS.md management, self-improving reflectors, and codebase map generation |
| ECO-4.11 | Deterministic Lint Enforcement Hook | PRE_TOOL_USE subprocess hook for ruff/mypy/eslint enforcement |
| ECO-4.12 | Plugin Bundle Distribution System | Manifest-based skill/hook/config packaging with KG registry |
| ECO-4.13 | Ecosystem Governance & Policy Engine | Unified engine managing permission policies, configuration staleness auditing, and governance workflows |
| ECO-4.14 | Infrastructure Blueprint Library | Library of modular, declarative system infrastructure configurations |
| ECO-4.34 | Fleet-Scale MCP Multiplexer Hardening | Per-child concurrency limits, session pools, restart-on-crash, circuit breakers, `multiplexer_status` tool (`mcp/child_resilience.py`) |
| ECO-4.36 | Dynamic MCP Tool Gateway | Boots with meta-tools (`find_tools`/`load_tools`/`unload_tools`), KG-discovers and lazily mounts child tools at runtime via FastMCP `tools/list_changed` with catalog-aware, collision-free prefix assignment — solves whole-fleet tool overload (`mcp/multiplexer.py`) |
| ECO-4.37 | Surface Gateway Client SDK | The single shared gateway client every frontend (webui, terminal-ui, geniusbot) uses, so transport/auth/retry live in one place (`gateway_client/`) |
| ECO-4.38 | Usage Session Ingestion | Auto-detecting parser registry that turns agent session logs into normalized usage records (agentsview assimilation) |
| ECO-4.39 | Usage Analytics Store | Backend-abstracted pydantic usage/cost models — the single source of token cost for the whole stack |
| ECO-4.40 | LiteLLM Pricing Source | Per-model token pricing feeding cost across the stack |
| ECO-4.41 | Cross-UI Usage & Cost Surface | The usage/cost/observability surface (`/api/observability`, `usage_query`) rendered in all three frontends |
| ECO-4.42 | Remote Usage Ingest | Clients parse local logs and POST normalized usage to the server sink |
| ECO-4.43 | Document → KG Fact Extraction UI | Interactive document/URL → atomic-triple extraction with live force-graph, edge-fact cards, and JSONL across all three frontends over `/api/enhanced/extract/*` |

### Pillar 5: Agent OS Infrastructure (OS-5.0 – 5.29)

| ID | Concept | Description |
|---|---|---|
| OS-5.0 | Agent OS Kernel & XDG Paths | Kernel lifecycle, XDG path resolution |
| OS-5.1 | Security & Auth | JWT/API auth, session concurrency, injection scanner |
| OS-5.2 | Resource Scheduling 🔬 | Cognitive scheduler, token quotas, preemption |
| OS-5.3 | OS Guardrails & Safety Boundaries | Holistic boundary definition integrating tool guards, reactive budgets/homeostasis, and ontological guardrails |
| OS-5.4 | Telemetry & Observability | OTEL, token tracking, audit logging |
| OS-5.5 | Massive Scale | Pluggable distributed queues, epistemic-graph Rust UDS RPC, and wasmtime sandbox integration |
| OS-5.14 | Server-Minted JWT Identity | `ActorContext` minted server-side from validated JWTs; fail-closed permissioning; HMAC engine auth (`security/request_identity.py`, `security/auth.py`) |
| OS-5.15 | Fleet Event Ingress | `POST /api/fleet/events` webhook persisting monitoring alerts as `FleetEvent` KG nodes + triage seam |
| OS-5.16 | Unified Durable-State Externalization | One `STATE_DB_URI` flag moves checkpoints, sessions/goals, and the task queue onto shared Postgres |
| OS-5.17 | Cross-Host Daemon Leadership | Postgres advisory-lock election so singleton background ticks run on exactly one host fleet-wide |
| OS-5.18 | Fleet Supervisory Plane at Scale | SQL aggregation, paginated/filtered session queries, desired-state pause/kill reconciliation across hosts |
| OS-5.23 | Gateway Middle-Tier Hardening | Prometheus `/metrics`, per-tenant token-bucket rate limiting, engine circuit breaker, `GATEWAY_WORKERS` pre-fork |
| OS-5.24 | ActionPolicy Decision Point | Per-action autonomy tiers, durable rate limits, maintenance windows, blast-radius caps; fail-closed; audit-logged |
| OS-5.25 | Desired-State Fleet Reconciler | Leader-only tick diffing the fleet registry against a `FleetObserver`, converging through ActionPolicy (dry-run default) |
| OS-5.26 | Remediation Playbooks | `service_down`/`service_flapping`/`resource_pressure` playbooks with stepwise verification on the fleet-event triage seam |
| OS-5.27 | Health-Gated Deploy Watch | Durable post-deploy health watch; failure invokes policy-gated rollback + escalation |
| OS-5.28 | Shard Topology Visibility | Per-shard reachability/breaker status surfaces + per-endpoint engine gauges and counters |
| OS-5.29 | Reactive Replica Autoscaling | Registry-declared scaling bounds + pluggable signal providers + leader-only target-tracking autoscaler behind the policy gate |
| OS-5.32 | Multiplexer Outbound Service Token | The MCP multiplexer mints + refreshes a Keycloak client-credentials token (audience `agent-services`) and attaches it to every remote child so jwt-enforced connectors are reachable through the aggregator (`mcp/client_credentials.py`, opt-in `MCP_CLIENT_AUTH=oidc-client-credentials`) |

### Gateway Service Dashboard (OS-5.9)

| ID | Concept | Description |
|---|---|---|
| OS-5.9 | Gateway Service Dashboard | Unified 50-widget dashboard data layer with registry, aggregator, REST+WS API, and MCP auto-discovery. Synthesized from former `service-dashboard-core` into `agent_utilities/gateway/`. |

## Agent OS Architecture

The Agent OS is a multi-subsystem architecture where the **Active Knowledge Graph (KG-2.0)** drives all tool discovery and routing:

| Subsystem | Package | Role |
|:---|:---|:---|
| 🧠 **Kernel** | `agent-utilities` | Models, logic, graph orchestration, KG, default catalog |
| 🖥️ **Desktop Cockpit** | `geniusbot` | Premium multi-platform PySide6 Systems & Finance Cockpit GUI (CONCEPT:GBOT-6.0) |
| ⚙️ **OS Layer** | `systems-manager` | Host OS operations + Agent OS MCP wrappers (23+ tools) |
| 📦 **Container Runtime** | `container-manager-mcp` | Docker/Podman lifecycle (60+ tools) |
| 🌐 **Network Stack** | `tunnel-manager` | SSH tunnels, remote exec, file transfer (43 tools) |
| 📂 **Workspace** | `repository-manager` | Git workspace mgmt, dependency graphs (24 tools) |

## Query Lifecycle Walkthrough

1. **Protocol Ingress (ECO-4.0)**: Query arrives via `/acp`, `/ag-ui`, or `/a2a`.
2. **Usage Guard (OS-5.1)**: Validates rate limits, execution budgets (ORCH-1.3).
3. **TeamConfig Check (AHE-3.3)**: Router checks KG for proven specialist coalition.
4. **Planner (ORCH-1.1)**: HTN goal decomposition and LATS fallback logic.
5. **Memory Injection (KG-2.1)**: Fetches Virtual Context Blocks and rationales.
6. **Dispatcher (ORCH-1.0)**: Spawns Specialist Superstates in parallel.
7. **Execution (ECO-4.0)**: Specialists interact with MCP servers or Universal Skills.
8. **Verification (AHE-3.1)**: Quality scoring with feedback loop on `< 0.7`.
9. **Persistence (KG-2.0)**: Traces/evaluations stored into the Knowledge Graph.

## Evolution Pipeline — Super-Assimilation Architecture

The evolution pipeline (`agent-utilities-evolution`) provides autonomous, KG-driven
assimilation of external codebases and research papers into the `agent-utilities` core.

### Assimilation Heuristic

All assimilation follows the **Wire or Discard** principle:

1. **Wire-First**: Every feature MUST connect to an existing hot path (≤3 hops from entry point)
2. **Extend, Don't Duplicate**: Overlap ≥ 0.7 similarity → extend existing CONCEPT:ID
3. **No Dead Code**: No live call path → rejected
4. **Constitution Preservation**: External codebases' governance rules are ingested as PolicyNodes

### 4-Phase Pipeline

```mermaid
flowchart TD
    A[Phase 1: Ecosystem Ingestion] --> B[Phase 2: Assimilation Codification]
    B --> C[Phase 3: Parallel Comparative Analysis]
    C --> D[Phase 4: SDD Plan Generation]

    subgraph P1 [Phase 1: Ingestion]
        A1[ECO-4.6: agent-packages] --> A3[ORCH-1.0: IntelligencePipeline]
        A2[ECO-4.6: open-source-libraries] --> A3
        A3 --> A4[KG-2.2: PolicyIngestor: Constitution Rules]
    end

    subgraph P3 [Phase 3: Analysis]
        C1[ORCH Background Research] --> C6[Synthesis]
        C2[KG Background Research] --> C6
        C3[AHE Background Research] --> C6
        C4[ECO Background Research] --> C6
        C5[OS Background Research] --> C6
        C6 --> C7[KG-2.2: Concept Cross-Reference Matrix]
    end

    subgraph P4 [Phase 4: SDD]
        D1[Feature Recommendations] --> D2[Wiring Audit]
        D2 --> D3[KG-2.2: Constitution Compliance]
        D3 --> D4[ORCH-1.5: SDD Implementation Plan]
    end
```

### Integration Points

| Component | Role in Evolution |
|-----------|------------------|
| `PolicyIngestor` (KG-2.2) | Ingests external constitutions as PolicyNodes |
| `IntelligencePipeline` (KG-2.0) | Bulk codebase ingestion via graph-os MCP native ingestion |
| `graph_analyze` (KG-2.0) | Parallelized L1→L2→L3→OWL analysis per pillar |
| `concept_map.md` | Source of truth for 70 canonical concepts to cross-reference |
| `constitution.md` | Assimilation Governance rules enforced during SDD |

### KG Node Types

| Node Type | Purpose |
|-----------|---------|
| `EvolutionCycle` | Tracks each evolution pipeline run with metrics |
| `SDDPlan` | Generated implementation plan from analysis |
| `ResearchTopic` | Topics detected for research scanning |
| `PolicyNode` | Constitution rules from ingested codebases |
