# agent-utilities C4 Architecture

This document provides formal C4 architecture diagrams showing how the 5 pillars
of `agent-utilities` interconnect with each other and with external IDE consumers.

## Current authority flow (normative)

The detailed C4 views below inventory capability ownership. They do not create
alternate graph, identity, work-state, model-context, or connector authorities.
Every supported entry path converges on this flow:

<div class="admonition architecture" markdown>
<p class="admonition-title">Current authority flow: every entry path converges on one engine</p>

Every client (library, MCP, REST, or delegated skill) crosses a verified
identity boundary into a `GraphSession` (actor, tenant, graph, scopes,
policy). The session reaches the GraphOS action core directly and also
through the mandatory `ContextCompiler` (required before model
execution). The action core reaches the epistemic-graph engine — the
sole graph and durable-work authority — via one process-wide
`GraphComputeEngine` client, and separately drives engine-native
WorkItems (claim, lease, fence, result) into the same engine.

Runtime secret and TLS profile refs feed external schema discovery
(mapping proposal, approval, drift check), which produces a
`ChangeEnvelope`/`MutationBatch` that also reaches the engine through the
same `GraphComputeEngine` client.

The action core emits metadata-only traces to Langfuse through a
resolved TLS profile. The engine asynchronously and governedly projects
into optional mirrors, which are never an authority.
</div>

See [Graph Authority Convergence](../architecture/graph-authority-convergence.md),
[Mandatory ContextCompiler](../architecture/mandatory-context-compiler.md), and
[Universal External Graph Connectors](https://knuckles-team.github.io/agent-connector-sdk/architecture/universal-graph-connectors/)
for the executable contracts behind the diagram.

> [!NOTE]
> Components marked with 🔬 are research-backed additions from the
> comparative analysis pipeline (papers 2605.05701v1, 2605.03310v1,
> 2604.20874v1, 2605.05242v1).

## Level 1: System Context

Shows `agent-utilities` in the broader ecosystem — all IDE and agent consumers.

<div class="admonition architecture" markdown>
<p class="admonition-title">Level 1: system context — agent-utilities among IDE and agent consumers</p>

A developer develops in Antigravity IDE, interacts via
`agent-terminal-ui` (CLI), and interacts visually via geniusbot.
Autonomous agents run orchestrated execution against `agent-utilities`
(the core agent OS kernel with KG-native intelligence, 5-pillar
architecture) directly.

Every external consumer reaches `agent-utilities` its own way: Antigravity
IDE and Claude Code via MCP (KG queries/tool execution, shared KG
read/write); OpenCode and Devin via MCP (shared KG read); `agent-terminal-ui`
via the library API through the shared GraphOS runtime; `agent-webui` via
ACP/AG-UI; geniusbot via the library API + `AgentBridge` through the
shared runtime; `universal-skills` via the DSTDD pipeline and skill
ingestion.

`agent-utilities` reaches Enterprise Systems (ITSM/ERP/BPM/EA tools —
ServiceNow OR ERPNext, Camunda, Archi, LeanIX, GitLab, interchangeable
via the vendor-neutral crosswalk): vendor adapters lift REST APIs into
canonical ArchiMate nodes, and virtual REST federation queries live data
(KG-2.9 / KG-2.1).
</div>

## Level 2: Container Diagram

Shows the 5 pillars as containers with data flows between them.

<div class="admonition architecture" markdown>
<p class="admonition-title">Level 2: container diagram — 5 pillars + scale-out planes</p>

A developer/agent sends an authenticated request (JWT -> `ActorContext`)
to the **OS Agent OS Kernel** (FastAPI: identity minting, guardrails,
lifecycle, telemetry, Prometheus, rate limiting), which dispatches
validated tasks to the **ORCH Orchestration Engine** (router, planner,
dispatcher, capability wiring; queue-driven turn dispatch).

ORCH queries the **KG Knowledge Graph** (native ingestion, OWL/Datalog,
hybrid retrieval) for routing and specialist selection, and enqueues
`AgentTurnEnvelope`s onto Work Queues (Kafka/Postgres/SQLite, keyed by
tenant/repo or session) in queue mode; KG separately enqueues ingest
tasks onto the same queues. Worker Fleets (`kg-ingest-worker`,
`agent-dispatch-worker`, stateless, any host) claim from the queues
at-least-once/idempotently, execute as engine clients (HMAC auth) against
KG, and write back durably + heartbeat to the Shared State Store
(Postgres or per-host SQLite).

KG persists to/queries the Knowledge Graph DB (the epistemic-graph
engine — the authority — with optional mirrors), routed by catalog to a
Raft group. The OS Kernel also persists sessions/goals/leadership/
approvals to the Shared State Store, and persists execution traces +
telemetry back to KG.

The **OS Fleet Autonomy Plane** (ActionPolicy gate, fleet reconciler,
remediation playbooks, deploy watch, autoscaler) exchanges fleet events
and approvals with the OS Kernel (`/api/fleet/*`).

KG feeds the **AHE Agentic Harness** (self-model, TeamConfig, evolution,
evaluation), which promotes proven coalitions to the **ECO Ecosystem
Peripherals** (MCP server factory, A2A, skill management) and publishes
proposals through the Fleet Autonomy Plane's `ActionPolicy` gate. ECO's
tool execution runs through the OS Kernel's guardrails.
</div>

## Level 3: Component Diagram — Per Pillar

### Pillar 1: Orchestration Engine (ORCH)

<div class="admonition architecture" markdown>
<p class="admonition-title">Pillar 1 components: Orchestration Engine</p>

The **Agent Orchestrator** is the entry point for all orchestration: it
routes to the **KG Router** (ontological routing via KG topology), which
routes tasks to the **Agentic Planner** (HTN recursive goal
decomposition), which decomposes into parallel batches for the **Graph
Dispatcher**, which discovers required capabilities via the
**Capability Wiring Engine**.

The Orchestrator also selects a protocol via the research-backed
**Coordination Layer** before execution (applied by the Dispatcher as
consensus/voting/delegation), aggregates quant predictions from the
**Prediction Linkage Layer**, and delegates latent multi-agent loops to
the **RecursiveMAS Latent Orchestrator** (which can bypass standard
planning and registers its traces with the **Agent Runner**).

The KG Router materializes graphs from KG templates via the **KG Graph
Factory**, which provides topology + specialist configs to the
Dispatcher. The **Agent Runner** (KG-to-LLM execution bridge) also
materializes agent-specific graphs via the KG Graph Factory and resolves
agents from the KG Router to dispatch tasks.

Workflow components: **Workflow Catalog** registers scenarios into
**Workflow Store**; **Workflow Compiler** (NL -> GraphPlan) persists
compiled workflows into Workflow Store; **Workflow Runner** loads
workflows by name from Workflow Store and executes their steps via
`run_agent()` on the Agent Runner.

After each wave, the Dispatcher feeds **Global Workspace Attention**
(select + broadcast winners, feeding back `get_attention_score` to the
KG Router) and the **Multi-Agent Social System** (swarm-health snapshot
-> telemetry). The Dispatcher also routes oversized output / long-horizon
tasks to the **Recursive Language Model** (persistent REPL), which
registers its recursive trajectories as provenance with the Agent
Runner.
</div>

> **GWT loop & MASS:** see [Global Workspace Attention](../architecture/global_workspace_attention.md)
> and [Multi-Agent Social System](../architecture/multi_agent_social_system.md). Both are driven by
> `ParallelEngine.execute` after a multi-agent wave and surface in `ExecutionResult.telemetry`.

### Pillar 2: Knowledge Graph (KG)

<div class="admonition architecture" markdown>
<p class="admonition-title">Pillar 2 components: Knowledge Graph</p>

**Core engine and backends.** `IntelligenceGraphEngine` (composed of 8
focused mixins: Query, Memory, Ingestion, MCPDiscovery, Registry,
TaskManager, Federation, AHE) reads/writes Graph Backends (authority:
epistemic-graph engine; fan-out mirrors: PostgreSQL/pg-age, Neo4j,
FalkorDB, Ladybug) via Cypher, and orchestrates Graph-OS Ingestion
(ingest, enrich, index, materialize, evolve via MCP).

**Retrieval.** Hybrid Retriever (semantic 72% + keyword 28%) queries the
engine; DCI Retriever seeds from hybrid results then does multi-hop
graph traversal. Retrieval Budget caps retrieved context to a token
budget; Governance Rules re-rank/filter designations at retrieval time.

**Ontology and compute.** OWL Bridge + SPARQL enforces schema via the
engine and delegates Datalog reasoning to the Rust EpistemicGraph Compute
Engine; the engine also executes vectorized calculations via the Rust
Quant Compute Engine. SDD Ontology is imported by OWL Bridge; SHACL
Validator validates materialized RDF; Ontology Publisher exports to
Stardog/Fuseki; Ontology Loader resolves `owl:imports`.

**Memory.** Memory Tiers provide CRUD for tiered (episodic/semantic/
procedural) memory on the engine; Context Budget Optimizer compacts
recall results within budget; Evolving Memory API adds Ebbinghaus decay +
GraphRAG traversal.

**Cross-system alignment.** Vendor Source Extractors write canonical
GraphNodes to the engine and emit types bound by the Vendor-Neutral
Crosswalk, which loads as a sibling ontology (HermiT propagates
`rdf:type` to canonical concepts). Ontology Alignment Bridge resolves
topological alignments for disparate systems during Stream
Hydration/R2RML ingestion, which also enforces schema via OWL Bridge and
feeds the engine. Database Schema Hydrator and Process Modeling Engine
extract SQL/workflow structure into the ontology.

**Code-to-capability.** Code->Capability Bridge writes REALIZES edges +
provisional capabilities to the engine and hands minted capabilities to
Capability Write-Back (pushes to Archi/LeanIX). Virtual REST Federation
invokes Vendor Source Extractors at query-time (TTL-cached, no
materialization).

**Governance and security.** Brain-Guarded Backend wraps the backend
store with mandatory provenance + authority-arbitrated writes; Secured
Reads filters/scopes/audits reads on the facade path; Entailment-Aware
Permission Scoper intersects security classifications for inferred
Datalog edges. Feedback Service writes Correction/rule/eval nodes to the
engine and persists rules consumed by Governance Rules.

**Reasoning.** Reasoner Router routes reasoning paradigms via
`CapabilityIndex` (reward-EMA) against Hybrid Retriever, dispatching to
either the World Model (action-conditioned rollout) or Program Synthesis
(inductive DSL search, MDL/Occam prior).

**Operational hardening.** Bounded Reads does type-scoped reads via
`get_nodes_by_label` on the backend (never a full graph scan); Engine
Breaker + Adaptive Retry guards every op against the Rust compute engine
and self-heals transient drops; Engine Response Guard (Rust) caps
oversized dumps with lock-gap histograms; Ingest Profiler times
read/extract/embed/write + token usage per ingest against the pipeline.
Shared Ephemeral Cache Fabric and Native Program Jobs (Rust) round out
the engine's supporting infrastructure; Stream Adapters and Intelligence
Extractors feed the ingestion pipeline with live events and distilled
operating-intelligence nodes respectively.
</div>

### Self-Improving Reasoning Substrate (cross-pillar)

The reasoning router (AU-KG.compute.first-class-reasoner-paradigm), world model (AU-KG.compute.first-class-action-conditioned) and program synthesizer (AU-KG.coordination.inductive-program-synthesis-search)
above are the REASON stage of a single closed loop that spans EG-KG.compute.backend, AHE-3, SAFE-1 and OS-5:
**route → reason → measure → learn**, cost-bounded and corrigible, with winning traces
distilled back into training data at scale. The router *learns which paradigm works for
which task class* by reusing the reward-aware `CapabilityIndex` — paradigm selection
self-improves. See **[Self-Improving Reasoning Substrate](../architecture/self_improving_reasoning_substrate.md)**
for the full component + dynamic diagrams and the concept→role map.

<div class="admonition architecture" markdown>
<p class="admonition-title">The closed reasoning loop: route, reason, measure, learn</p>

A task feeds the router (ROUTE), which selects among reasoning paradigms
(inductive, model-based/action-conditioned, deductive, generative)
(REASON). The result is scored by SAFE-1.1 + capability-benchmark
regression ratchet (MEASURE), which feeds `record_outcome` -> reward EMA
(LEARN). LEARN feeds a routing reward back to ROUTE, closing the loop,
and also feeds an RSI ledger. MEASURE's winning traces feed a
collapse-guarded distillation pipeline; REASON, at scale, feeds the
ORCH-1.46/47/48 collective.
</div>

### Pillar 3: Agentic Harness (AHE)

<div class="admonition architecture" markdown>
<p class="admonition-title">Pillar 3 components: Agentic Harness</p>

**Core loop.** Continuous Evaluation Engine (multi-strategy `EvalRunner`)
updates Self-Model's scores; Evolution Engine (skill neologism, genetic
crossover) triggers on failure patterns from evaluation; TeamConfig
Composer uses Self-Model's scores for coalition composition and syncs
concurrent agent state via the Distributed Agent State Manager
(optimistic locking, optional Redis); DSTDD Manager validates features
against KG integrity via evaluation. Workflow Distillation Hook promotes
proven team compositions and feeds distilled patterns back into
Evolution Engine.

**Evolution outputs.** Evolution Engine submits trace-derived governed
program jobs to the Native Program Optimizer (Rust); offloads evolved
structures to the Physical Knowledge Distiller, which triggers git
changes via the GitOps Evolution Boundary; selects a strategy via the
Dynamic Optimizer Selector based on failure characteristics; and pushes
decisive cycles to the Prioritized Replay Buffer (inverse-frequency
resurfacing of rare states).

**Training substrate.** Training Reward Spine (advantage / failure-point
/ composite-reward / difficulty-floor) feeds the In-House Training
Substrate (SFT/DPO/GRPO trainers + Rust kernels), whose `eval_hooks`
bridge checkpoints back into the evaluation reliability suite.
Agent-Step PO contributes per-step advantage into the reward spine;
Test-Time Diversity (VPO) raises test-time pass@k via evaluation;
Preference-Corpus Reliability feeds DPO-ready preference pairs to the
training substrate; MemoryData Bake-off feeds retrieval-config scores
into evaluation evidence.
</div>

> **Training substrate:** the reward spine + replay buffer feed the cross-repo
> [In-House Training Substrate](../architecture/in_house_training_substrate.md)
> (data-science-mcp gradient trainers + epistemic-graph Rust kernels); trained
> checkpoints go live via the model-registry role seam with no hot-path edit.

### Pillar 4: Ecosystem Peripherals (ECO)

<div class="admonition architecture" markdown>
<p class="admonition-title">Pillar 4 components: Ecosystem Peripherals</p>

MCP Server Factory creates the KG MCP Server instance, which shares KG
data across the A2A Network (agent-to-agent discovery/delegation);
Coordinated A2A Skill extends A2A with coordination protocol
negotiation. Skill Manager loads skills from `universal-skills` via the
Ecosystem Bridge.

Agent Toolkit Ingestor (unified MCP/Skill/A2A ingestion with
auto-detection) delegates live tool discovery to MCP Live Discovery,
ingests skill directories via Skill Manager, and fetches A2A agent cards
via the A2A Network; MCP Live Discovery uses MCP Server Factory's
canonical bounded stdio/HTTP/SSE child probe.

The Unified Quant MCP Tool (routes to orchestrate/data/execute/
portfolio) exchanges telemetry and orders with the Microstructure Engine
(high-frequency OBI & micro-price), which provides micro-price edges to
the Stat Arb Engine (cross-market cointegration & OU modeling), which
generates stat-arb signals back to the Quant MCP Tool.
</div>

### Pillar 5: Agent OS Kernel (OS)

<div class="admonition architecture" markdown>
<p class="admonition-title">Pillar 5 components: Agent OS Kernel</p>

**Request path.** Actor Identity Middleware (server-minted JWT
`ActorContext`, fail-closed) scopes each request, feeding Security
Policy Middleware (JWT/API key/MCP auth), which validates before routing
to the Threat Defense Engine (prompt injection, jailbreak detection),
which applies runtime constraints via the Guardrail Engine (tool guard,
rate limit, content filter), which records enforcement decisions to the
Telemetry Pipeline (OTEL, token tracking, audit logging). Gateway
Metrics + Rate Limit exposes Prometheus series to Telemetry. Cognitive
Scheduler tracks cost and auto-downgrades model tier via the Inference
Budget Controller.

**Paths and dashboard.** XDG Paths Module provides config/data locations
to Security Policy Middleware and to the Gateway Service Dashboard
(50-widget registry, aggregator, REST+WS API), which reports widget
fetch metrics to Telemetry.

**Fleet supervision.** Fleet Supervisory Plane (`/api/fleet/*`) runs
paginated session/goal queries and approvals against the State Store
Seam (`STATE_DB_URI`), and feeds FleetEvents + desired-state input to the
Fleet Reconciler + Autoscaler. Every mutating action from the reconciler
consults the ActionPolicy Decision Point (per-action autonomy tiers,
durable rate limits, blast-radius caps, fail-closed); allowed
deploys/restarts get a health watch from Deploy Watch, whose own
rollback decisions are themselves policy-gated back through
ActionPolicy.
</div>

### Pillar 6: GeniusBot Cockpit (GUI)

<div class="admonition architecture" markdown>
<p class="admonition-title">Pillar 6 components: GeniusBot Cockpit</p>

`AgentBridge` (async Python-to-Qt bridge) pushes system metrics to the
Systems Dashboard, streams portfolio data to the Finance Cockpit,
streams agent responses (SSE) to Agent Chat, and delivers graph query
results to the KG Visualizer. Settings Manager configures
`AgentBridge`'s agent connections.
</div>

## Cross-Pillar Data Flows

<div class="admonition architecture" markdown>
<p class="admonition-title">Cross-pillar data flows, 21 named flows</p>

**Ingestion Flow.** An MCP tool call reaches the ORCH Router, which
reaches the KG Ingest Engine, which reaches the KG OWL Bridge.

**Execution Flow.** The Planner feeds the Dispatcher, which calls the
ECO Tool Executor, which passes through OS Guardrails to the AHE
Evaluator.

**Learning Flow.** `EvalRunner` writes to the KG Memory Tier, which
feeds AHE Evolution, which feeds the ECO Skill Evolver.

**Security Flow.** The OS Threat Scanner feeds the KG Risk Ontology,
which feeds AHE Immunity, which feeds the OS Policy Engine.

**Continuous Ingestion Flow.** A git post-commit hook runs
`scripts/submit_diff.py`, which reaches the KG TaskManager, which
creates a KG `DiffEntry` node.

**Entity Lifecycle Flow.** An active KG node soft-deletes to
`status=ARCHIVED`, which can either restore back to `status=ACTIVE` or
hard-delete (by age) to permanently removed.

**Research Integration Flow.** ScholarX paper search downloads into KG
paper ingestion, which discover-mode feeds KG Innovation Discovery,
which cross-refs the KG Concept Map, which assimilates into
`ASSIMILATED_INTO` edges.

**Enterprise Federation Flow.** OWL Materialize feeds a SPARQL HTTP
endpoint (via rdflib) that external consumers query; it also feeds the
SHACL Validator (validate) and the Ontology Publisher (export), which
pushes to Stardog/Fuseki; that in turn feeds the Ontology Loader
(`owl:imports`), which merges back into OWL Materialize — closing the
loop.

**Vendor-Neutral Crosswalk Flow (KG-2.9).** ServiceNow `:Incident`,
ERPNext `:ErpNextIssue`, and Camunda `:BusinessTask` each extract into
canonical GraphNodes, which promote through `owl_bridge` + HermiT into
canonical concepts (`:ApplicationEvent`/`:BusinessProcess` via
`subClassOf`/`equivalentClass`), answering one query across all vendors;
`register_rest_source` also feeds that same query, query-time and
TTL-cached.

**Code -> Capability Flow (KG-2.8).** Rust AST-derived features and
LeanIX/Archi `BusinessCapability` both feed `resolve_realizes`
(match/mint/registry), which writes a `REALIZES` edge to KG persistence
and, for a provisional capability, feeds `capability_writeback`, which
pushes to Archi/LeanIX (`add_element`/`postbusinesscapability`).

**KG Graph Materialization Flow.** A user query's `router_step` reaches
KG hybrid search (returning `AgentTemplate` nodes), which feeds
topological sort (`DEPENDS_ON`), then prompt resolution (`USES_PROMPT`),
then tool binding (`REQUIRES_TOOLSET`), then graph build, producing a
`KGGraphResult` for the Dispatcher.

**Agent Toolkit Ingestion Flow.** Sources (`mcp_config.json`, skill
dirs, A2A URLs) auto-detect into the Type Detector, which routes JSON
with `mcpServers` to the MCP Config Parser, a directory with `SKILL.md`
to the Skill Parser, and an `http://` URL to the A2A Card Fetcher. The
MCP Config Parser live-connects to `list_tools()`, writing tool metadata
to KG `Server`/`CallableResource` nodes (or falls back to the Tool Flag
Parser, which writes the same nodes); the Skill Parser and A2A Card
Fetcher (via `/.well-known/agent.json`) write there too. Every write
feeds a config-hash freshness check.

**Agent Execution Flow.** `graph_orchestrate execute_agent` resolves
`agent_name` into Server/Skill/A2A nodes, which feed the Config Builder
(`tag_prompts` + `mcp_toolsets`), which feeds `create_graph_agent()`,
which produces a materialized graph for `run_graph()` (LM Studio),
producing a `GraphResponse` that feeds `RunTrace` provenance.

**Workflow Lifecycle Flow.** `catalog.yaml` loads into `WorkflowCatalog`
and natural language compiles via `WorkflowCompiler`; both produce
`GraphPlan[]`. `WorkflowCatalog` also registers into `WorkflowStore` and
`WorkflowCompiler` compiles-and-stores there too; `WorkflowStore`
persists a KG `WorkflowDefinition`, which loads back into `GraphPlan[]`.
Plans execute via `WorkflowRunner`, which wave-dispatches to
`run_agent()` and emits session traces to Langfuse.

**Workflow Distillation Flow.** A successful Synthesizer run feeds the
Distillation Hook, which — once threshold is met — promotes into both
`WorkflowStore` (versioned) and the TeamConfig Composer (proven team);
both write to KG persistence, which bundle-exports to YAML/JSON domain
presets, which seed back into the KG.

**Queue-Driven Dispatch Flow.** `graph_orchestrate`'s dispatch/goal loop
becomes an `AgentTurnEnvelope` (queue-only), keyed by session id onto the
`agent_turns` queue (Kafka/Postgres/SQLite). A worker claims it under
session lock, rehydrates and executes the existing orchestration body,
durably writes back and acks to the shared state store, and heartbeats
to the fleet topology endpoint.

**Ingest Scale-Out Flow.** `graph_ingest submit` reaches the `kg_tasks`
topic (keyed tenant -> repo -> type, fail-loud backend selection),
consumed by the `kg-ingest` consumer group across both
`kg-ingest-worker` processes and the host engine worker pool. Workers
claim idempotently by `job_id` against the epistemic-graph engine, and
record a per-ingest `IngestProfile` (stages_ms + tokens/cost) into
`profile_report`. The topic also exposes lag/depth gauges to
`/metrics`; the engine exposes a `RESULT_TOO_LARGE` guard + write-lock
histograms, and coalesces per-graph write contention (N writes -> 1
txn).

**Engine Sharding Flow.** A graph operation resolves, for its verified
graph/tenant, a `PlacementRoute` (group + epoch + fence) to the
authoritative group in the MultiRaft engine cluster, whose reachability
and breaker state feed the daemon/shards topology dashboard.

**Fleet Autonomy Flow.** Alertmanager/Uptime Kuma `POST`s
`/api/fleet/events`, creating `FleetEvent` nodes that triage into
remediation playbooks. Separately, the fleet registry drives the fleet
reconciler and (with scaling bounds) the autoscaler. Playbooks,
reconciler, and autoscaler all consult the `ActionPolicy` gate: an allow
reaches the `FleetActuator` (dry-run by default); anything else queues
for approval. Actuated deploys/restarts get a deploy watch, whose
sustained-failure rollback is itself routed back through `ActionPolicy`.

**Evolution Publication Flow.** Langfuse failures / performance
anomalies feed `failure_gap` topics into the golden loop, which feeds a
promoted proposal into the promotion governance validator, which
regression-gates change synthesis + an RLM sandbox, which publishes its
proposal through `ActionPolicy` into a reviewable local git branch
(never pushed).

**Gateway Service Dashboard Flow.** `mcp_config.json` auto-discovers
into `ConfigManager`, producing `ServiceConfig[]` for the Widget
Registry, which lazy-imports the 50 widget modules, which fetch data
through the Aggregator. The Aggregator serves `WidgetData{}` via REST
(`/api/dashboard`) and streams via WebSocket (`/ws/dashboard`) to
`agent-webui`, and serves `agent-terminal-ui` (direct Python) and
geniusbot (QThread) directly.
</div>

## Pillar Interconnection Matrix

<div class="admonition architecture" markdown>
<p class="admonition-title">Pillar interconnection matrix: a closed feedback loop, not a stack</p>

The five pillars — ORCH-1.0 Orchestration, KG-2.0 Knowledge Graph
(epistemic-graph engine authority + optional mirrors), AHE-3.0 Agentic
Harness, ECO-4.0 Ecosystem, and OS-5.0 Agent OS — interconnect
bidirectionally in several places:

- ORCH <-> KG: the router queries KG for specialist selection; KG
  provides ontological routing tables.
- ORCH -> ECO: the planner delegates to MCP tools; Capability Wiring
  discovers the tool registry.
- ORCH <-> AHE: the orchestrator feeds results to the evaluator; the
  evaluator adjusts routing weights.
- KG -> AHE: memory tiers feed Self-Model; TeamConfig promotes proven
  coalitions.
- KG -> ECO: the Ecosystem Topology Map materializes the 40-repo graph
  as KG nodes.
- KG <-> OS: execution traces persist to KG; telemetry feeds
  observability.
- AHE -> ECO: evolved skills promote to MCP/A2A; skill neologisms create
  new tools.
- AHE -> OS: the Adaptive Immunity Pipeline updates security patterns.
- ECO -> OS: the MCP middleware stack enforces auth, rate limits,
  guardrails.
- OS -> ORCH: the policy engine governs all execution paths and prompt
  safety; the Inference Budget tracker auto-downgrades model tier.
- ORCH -> ECO: the Coordination Layer selects a protocol per team
  composition.
</div>

> **Key Insight**: Every pillar has at least one bidirectional dependency with another pillar.
> The system is a closed feedback loop, not a layered stack. This is why isolated concept
> additions are dangerous — they must wire into the loop.


### Ecosystem Dependency Graph

<div class="admonition architecture" markdown>
<p class="admonition-title">Ecosystem dependency graph: three clients around one core</p>

Three client packages depend on `agent-utilities` (Python):
`agent-terminal-ui` (Python/Textual) depends on it directly and via
`gateway.Aggregator`; `agent-webui` (React/Next.js) interfaces with it
directly and via `gateway.api` + WS; `geniusbot` (Python/PySide6)
interfaces with it directly and via `gateway.Aggregator`.

`agent-utilities` itself depends on `pydantic-ai`, `pydantic-graph`,
`pydantic-ai-harness` ACP, `pydantic-ai-skills`, `fastmcp`, `fastapi`,
and `logfire`. `agent-terminal-ui` depends on `textual`, `rich`, and
`httpx`. `agent-webui` depends on `@ai-sdk/react` (Vercel), `ai` (Vercel
SDK), `react`, `tailwindcss`, and `vite`. `geniusbot` depends on
`PySide6`, `QtCharts`, and `QWebEngineView`.
</div>

### C4 Container Diagram
<div class="admonition architecture" markdown>
<p class="admonition-title">C4 container diagram: the agent orchestration system end to end</p>

A user uses Agent WebUI (React/Tailwind, HTTPS/WSS) or Agent Terminal UI
(Python/Textual, CLI); both query the Agent Gateway (FastAPI +
Pydantic-AI: ACP sessions, SSE streams, JWT-minted identity, per-tenant
rate limits, `/metrics`) via AG-UI/SSE.

The Gateway dispatches to the Graph Orchestrator (Pydantic-Graph:
routes queries, executes parallel domains, validates results) and
enqueues turns in queue mode (`AgentTurnEnvelope`) onto Kafka topics
(`kg_tasks` + `agent_turns`, keyed partitions, Postgres/SQLite
fallbacks). Those topics feed session-keyed claims to the
`agent-dispatch-worker` fleet (which rehydrates and durably writes back
to the shared support-state store) and tenant/repo-keyed claims to the
`kg-ingest-worker` fleet (which ingests as engine clients, MessagePack +
HMAC, against the epistemic-graph cell).

The Orchestrator itself reaches the epistemic-graph cell directly
(graph ops, placement-epoch routed, MessagePack/UDS or TCP), reads/writes
sessions/goals/approvals/leadership in the shared support-state store via
the Gateway, delegates to Domain Sub-Agents (Pydantic-AI: Git, Web,
Cloud, etc.) for parallel execution, and exports spans to the
OpenTelemetry Collector. Sub-agents invoke MCP Servers (contextual
tools, behind the hardened multiplexer) via JSON-RPC. Prometheus scrapes
the Gateway's `/metrics`.
</div>
