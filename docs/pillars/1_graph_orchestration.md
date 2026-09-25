# Pillar 1: Graph Orchestration Engine

## Overview

The **Graph Orchestration Engine** represents the foundational execution layer of the `agent-utilities` ecosystem. Moving away from rigid LLM chains and monolithic prompt contexts, this pillar implements a Hierarchical Task Network (HTN) backed by Pydantic Graph, transitioning linear execution into dynamic, topological routing.

## Why We Built This (Rationale)

As our agent ecosystem scaled to include dozens of domain specialists (Python, TS, CI/CD, DB) and hundreds of MCP tools, we encountered three critical failure modes:
1. **Prompt Bloat & Context Pollution**: Injecting all available tools into a single prompt exceeded context limits and degraded LLM reasoning accuracy.
2. **Sequential Bottlenecks**: Large features were executed linearly, squandering the opportunity for parallel discovery and implementation.
3. **Catastrophic Forgetting & Loop Cycles**: Agents would forget successful tool combinations or fall into infinite retry loops without an enforced architectural guardrail.

## How It Works (Implementation)

The architecture solves these bottlenecks through several interdependent primitives:

### Registry Hot Cache & Unified Specialists (ORCH-1.2)
We collapsed the artificial boundary between `prompt` and `mcp` agents into a singular `specialist` type. The **Registry Hot Cache** maintains an O(1) session-scoped index of these specialists. Instead of passing 50+ specialists to the orchestrator, it filters down to the Top-7 relevant specialists per query, reducing prompt token bloat by ~7x.

### Spec-Driven Development Pipeline (AU-ORCH.planning.spec-driven-pipeline)
The orchestrator implements a multi-stage SDD pipeline:
- **Discovery & Requirements**: Generates structured `Spec` models with measurable success criteria.
- **Task Decomposition**: Emits a `Tasks` dependency graph, identifying which subtasks can be executed in parallel (e.g., frontend and backend).
- **Parallel Dispatch**: Fuses tasks out to specific `specialist` workers, leveraging the `Execution Visibility Graph` to constrain context so a backend specialist only sees backend-related prior steps.

### Learned Agent Routing & Execution Budgets (AU-ORCH.planning.journey-milestone & ORCH-1.3)
Routing isn't static. `TraceLearnedPolicy` uses softmax scoring over historical `ExecutionTrace` records with an exponential moving average (EMA) to actively down-weight specialists with low success rates. `ExecutionBudget` acts as an absolute cost governor, preempting infinite loops by enforcing USD/token constraints at the dispatcher step.

## Benefits Introduced

- **Cost Efficiency**: By utilizing `Confidence-Gated Model Routing`, trivial queries fallback to smaller models (`gpt-4o-mini`), saving reasoning tokens for complex HTN planning.
- **Architectural Safety**: `Subagent Lifecycle Patterns` and recursive execution constraints ensure the system fails gracefully and retries contextually rather than spinning in infinite loops.
- **Test-Time Scaling**: The system achieves zero-shot generalization by spawning parallel agent rollouts and selecting the optimal path via dynamic subgraph convergence and evolutionary aggregation.

## Key Concepts Leveraged
- **ORCH-1.0**: Orchestration Engine
- **ORCH-1.1**: Agentic Planning Engine (Planning)
- **ORCH-1.2**: Agentic Planning Engine (Routing)
- **ORCH-1.3**: Execution Budgets & State Safety
- **AU-ORCH.planning.spec-driven-pipeline**: Spec-Driven Development
- **AU-ORCH.planning.journey-milestone**: Learned Agent Routing
- **AU-ORCH.execution.service-registry-initialization**: KG-Driven Graph Factory — materializes pydantic-graph topologies from AgentTemplate nodes
- **ORCH-1.21**: Agent Runner — KG-to-LLM execution bridge with dynamic tool binding and provenance tracking
- **ORCH-1.22**: RecursiveMAS Latent Orchestrator 🔬 — continuous latent loop or simulated semantic collaboration
- **ORCH-1.8**: [**Parallel Engine**](1_graph_orchestration/ORCH-1.8-Parallel_Engine.md) — unified 1→300+ agent execution engine with semaphore-governed concurrency, DAG scheduling, and tiered synthesis
- **ORCH-1.8**: RLM-Native Hierarchical Synthesis — flat/hierarchical/progressive/rlm output merging strategies
- **AU-ORCH.execution.autonomous-department-orchestration**: Autonomous Department Orchestration — OWL-materialized company departments with `reportsTo` hierarchy
- **ORCH-1.10**: Reactive Event Sourcing — reactive event-driven state and graph staging dispatcher
- **AU-ORCH.sandbox.compiled-orchestration-kernel**: WASM Micro-Agent Execution — isolated WebAssembly sandbox runner with gas/memory limits and Python emulation fallback (the RLM execution tier is now realized by **ORCH-1.38**)
- **ORCH-1.12**: [**Structured Predict-RLM Runtime**](1_graph_orchestration/ORCH-1.12-Structured_RLM_Outputs.md) — standard Pydantic signatures and dynamic skill injection wrapper for sandboxed REPL, plus **schema-constrained subagent contracts** so RLM fan-out returns typed values (bool/model/list) instead of free-form prose
- **ORCH-1.38**: [**Tiered RLM Code Sandbox + Capability Router**](1_graph_orchestration/ORCH-1.38-Tiered_RLM_Sandbox.md) — deterministic capability routing across isolated backends; unavailable isolation fails closed, while legacy local `exec()` requires an explicit dangerous opt-in
- **AU-ORCH.optimization.optimize-skill-prompt-gepa**: [**GEPA Reflective Prompt Optimizer**](1_graph_orchestration/ORCH-1.13-GEPA_Optimization.md) — Genetic-Pareto optimization loop with reflective mutation and structural crossover for prompt evolution
- **ORCH-1.37**: Orchestration execution-flow mermaid-diagram surfacing in `graph_orchestrate` responses (additive, backward-compatible)
- **ORCH-1.39**: Invoker→spawned-agent handoff of curated context, token budget, tool scope, and credential *reference* (raw secret never persisted/logged) — see [KG-Native Orchestration § Invoker to Spawned Handoff](../guides/kg_native_orchestration.md#invoker-to-spawned-agent-handoff-and-native-channels)
- **AU-ORCH.session.session-anchored-collections-native**: Session-anchored collections (`Session` node + `HAS_CONTEXT`/`HAS_MESSAGE`/`HAS_RUN` edges) and native cross-process invoker↔spawned message channels with a durable backstop and elicitation bridge (`graph_context`, `graph_message` MCP tools)
- **ORCH-1.41**: Process Plan Compiler — `graph_workflows(action="compile_process")` lifts a descriptive BPMN process into an executable plan (see [Ontology-to-Workflow Execution](#ontology-workflow-execution))
- **AU-ORCH.execution.ontology-validation-execution-path**: Execution Ontology Gate — ontology validation on the execution path before a compiled process runs (`knowledge_graph/core/workflow_gate.py`)
- **ORCH-1.43**: Workflow Lineage Close-Out — run lineage written back to the KG, closing the descriptive↔executable provenance loop (`workflows/runner.py`)
- **AU-ORCH.session.durable-goal-registry-goals**: Durable Goal Registry — goals persist across restarts; stranded runs rehydrate as orphaned instead of silently vanishing (see [State Externalization](../architecture/state_externalization.md))
- **ORCH-1.45**: Queue-Driven Agent Dispatch — session-keyed `agent_turns` queue (`AgentTurnEnvelope`) consumed by a stateless `agent-dispatch-worker` fleet with fleet-visible placement (see [Agent Dispatch](../architecture/agent_dispatch.md))

## 🧬 First Principles Architecture

The **First Principles Architecture** (CONCEPT:AU-ORCH.adapter.hot-cache-invalidation through CONCEPT:AU-ECO.messaging.native-backend-abstraction) rewires the routing, dispatch, and feedback layers from basic primitives. These four concepts solve the key scalability and intelligence bottlenecks that emerge when managing dozens of specialists and hundreds of tools.

| Concept | Problem Solved | Solution |
|:--------|:--------------|:---------|
| **CONCEPT:AU-ORCH.adapter.hot-cache-invalidation: Registry Hot Cache** | O(N) specialist lookups on every routing call | Session-scoped cache with O(1) lookups, event-driven invalidation |
| **CONCEPT:AU-AHE.evaluation.interpretability-tests: TeamConfig Promotion** | LLM re-discovers same specialist teams for recurring patterns | Persist proven coalitions as reusable templates in the KG |
| **CONCEPT:AU-ORCH.adapter.hot-cache-invalidation: AgentCapability System** | Static tool bindings; no dynamic capability activation | First-class KG capability nodes with trigger conditions |
| **CONCEPT:AU-ECO.messaging.native-backend-abstraction: PlannerGraphSkill** | A2A requests require full LLM round-trip | Direct graph-backed A2A routing, bypassing LLM overhead |
| **CONCEPT:AU-ECO.messaging.native-backend-abstraction: A2A Config File** | No mechanism to discover/register external A2A agents | File-based auto-discovery with `secret://` auth & periodic refresh |
| **CONCEPT:AU-ORCH.adapter.hot-cache-invalidation: Unified Specialist** | Artificial `prompt`/`mcp` type split complicates dispatch | Single `specialist` type hosting any tools/skills combination |

<div class="admonition architecture" markdown>
<p class="admonition-title">3-stage hybrid routing, closed by a reward loop</p>

A user query first checks `TeamConfig` for a match: a hit dispatches
directly; a miss goes through `Self-Model` bias into the LLM planner
(top-7 filtered), which then dispatches. Either way, dispatch reaches
specialist execution, which is verified, and the verification feeds a
`Self-Model` update + `TeamConfig` reward — closing the loop back into
the `TeamConfig` match step for the next query.
</div>

→ **Deep-dive**: [first-principles.md](../guides/first-principles.md) · [registry-cache.md](../guides/registry-cache.md) · [process-lifecycle.md](../guides/process-lifecycle.md)

## Architecture & Orchestration Overview

| `adguard-home-agent` | Graph |
| `agent-utilities` | Library | Production-grade Orchestration. Supports Parallel execution, Real-time sub-agent streaming, High-fidelity observability, and Session Resumability |
| `agent-webui` | Library | Cinematic Graph Activity Visualization. |
| `agent-terminal-ui` | Library | High-performance Terminal User Interface (TUI) achieving feature parity with **Claude Code** (Slash commands, Keyboard shortcuts, File mentions). |

`agent-utilities` implements a multi-stage execution pipeline using `pydantic-graph` for maximum precision and resilience. Protocol adapters (AG-UI, ACP) leverage `graph.iter()` for direct, step-by-step graph execution — bypassing the outer LLM agent entirely when a graph is present.

### Spec-Driven Development (SDD) Lifecycle

`agent-utilities` implements a rigorous SDD workflow to ensure that complex feature requests are handled with absolute technical fidelity and measurable success criteria.

1.  **Project Constitution** (`constitution-generator`): Establishes the governing principles, tech stack standards, and quality gates for the entire agent workshop.
2.  **Requirement Specification** (`spec-generator`): Decomposes user intent into a formal `Spec` including user scenarios, functional requirements, and measurable success metrics.
3.  **Technical Implementation Plan** (`task-planner`): Generates a step-by-step architectural approach and a `Tasks` model with explicit dependencies and file-path affinity for collision-free parallel execution.
4.  **Baseline & Manual Testing**: Integrates `first_run_tests` and `run_manual_test` into the implementation phase to ensure baseline stability and exploratory verification.
5.  **Parallel Execution** (`SDDManager`): The `dispatcher` leverages the SDD analysis engine to identify safe parallel execution batches, fanning out implementation tasks to domain specialists (Python, TS, etc.).
6.  **Quality Verification & Documentation**: Audits results via `spec-verifier`, then generates `code-walkthrough` and `interactive-explain` artifacts to document the final implementation.

### Execution Flow: Dynamic Multi-Layer Parallelism
`agent-utilities` implements a multi-stage execution pipeline with **autonomous gap analysis** and **resilient feedback loops**. The system can "fan out" research tasks in parallel before coalescing results. If implementation fails, it can automatically retry locally or loop back to research.

<div class="admonition architecture" markdown>
<p class="admonition-title">Execution flow: dynamic multi-layer parallelism</p>

A user query + images enters through the unified protocol layer
(ACP/AG-UI/SSE), passes the Usage Guard (rate limiting) — a block ends
the run immediately — and reaches the router, which selects a topology:
a trivial query ends immediately; a full pipeline reaches the
Dispatcher.

On first entry, the Dispatcher retrieves context via the Memory step
before continuing. For a "research first" plan, the Dispatcher fans out
in parallel to the **Discovery Phase**: Researcher (web-search,
web-crawler, web-fetch; project_search, read_workspace_file), Architect
(c4-architecture, spec-generator, product-strategy, user-research,
brainstorming; developer_tools), and Unified Discovery (Knowledge
Graph). All three join at a barrier sync (Research Joiner), which
returns coalesced context to the Dispatcher.

For implementation, the Dispatcher fans out to the **Execution Phase**,
grouped into three pools: Programmers (Python, TypeScript, Go, Rust, C,
C++, JavaScript — each with its own skill/tool set), Infrastructure
(DevOps, Cloud, Database), and Specialized & Quality (Security, QA,
UI/UX, Debugger). All three pools join at a second barrier sync
(Execution Joiner), which returns implementation results to the
Dispatcher.

Once the plan is complete, the Verifier scores the result: >= 0.7
proceeds to the Synthesizer, which composes the final response and
ends; 0.4-0.7 loops back to the Dispatcher; < 0.4 goes to the Planner
to re-plan with feedback, which also loops back to the Dispatcher. A
terminal failure at the Dispatcher also ends the run.
</div>

---

## 🔬 RecursiveMAS Latent Orchestration (research / not yet wired)

> **Status:** research design only — the native open-weights pipeline described
> below (`RecursiveLink`, a dedicated `rlm/mas_local.py` module) is **not yet
> implemented** in the codebase. The shipped RLM substrate lives in
> `agent_utilities/rlm/` (`RLMEnvironment` in `rlm/repl.py`, `rlm/specialist.py`,
> `rlm/predict_rlm.py`). Note also that the concept ID `ORCH-1.22` is now
> assigned to *Workflow Persistence & Replay Relationships* in
> `docs/concepts.yaml`; this section is retained as forward-looking design.

**RecursiveMAS** (Recursive Multi-Agent System) is a research-backed design that redefines multi-agent collaboration by passing **continuous representations in latent space** rather than raw text sequences.

### Why RecursiveMAS?
In traditional multi-agent architectures, agents communicate via textual generation. This forces the system to undergo costly autoregressive decoding/generation cycles at every step, creating:
1. **Severe Latency Bottlenecks**: Each agent in a chain must wait for the preceding agent to fully compile its reasoning text.
2. **Context Blowup & Token Bloat**: Spells out intermediate thoughts, rapidly consuming the prompt context window and driving up costs.
3. **Discrete Handoff Loss**: Text sequences lose the continuous, high-dimensional semantic richness of the model's internal last-layer hidden states.

By routing communication through embedding projections (latent space), RecursiveMAS achieves **1.2x to 2.4x inference speedup** and **up to 75.6% token usage reduction** by the third recursion round, while improving reasoning accuracy by **8.3%** on complex mathematical and coding benchmarks.

---

### Latent Collaboration Architecture

<div class="admonition architecture" markdown>
<p class="admonition-title">Latent collaboration: two frozen agents linked by RecursiveLink projections</p>

A user query reaches Agent A (e.g. Planner, model frozen). Agent A's
hidden states loop internally through Inner RecursiveLink A (a
self-thoughts loop), self-feeding embeddings back into Agent A. Agent
A's raw output activations also pass through Outer RecursiveLink AB
(projection & dimension matching) as projected embeddings into Agent B
(e.g. Specialist, model frozen).

Agent B likewise loops internally through Inner RecursiveLink B. Agent
B's raw output activations pass through Outer RecursiveLink BA (a
recurrent loop link) back to Agent A as the round t+1 latent feed —
closing the collaboration loop. On the final round, Agent B's output
instead reaches the final text decoder/response.
</div>

#### Key Components:
* **RecursiveLink**: A lightweight, multi-layer projection module that acts as the connective tissue between models, leaving original LLM weights completely frozen:
  * **Inner RecursiveLink**: Maps an agent's newly generated hidden states directly back into its own input embedding space, enabling continuous internal reasoning loops without token decoding.
  * **Outer RecursiveLink**: Maps embedding dimensions between disparate model shapes (e.g. Llama-3's 4096-d space to Gemma-3's 3072-d space) to bridge latent states between heterogeneous agents.
* **Shared Backbone Brain**: Multiple agent roles (e.g., Planner, Coder, Critic) can reside on the exact same foundation model instance in VRAM, sharing base weights while utilizing lightweight individual `RecursiveLink` modules for role specialization.

---

### Dual-Architecture Implementation Strategy

To implement RecursiveMAS without introducing performance overhead or heavy dependencies into the core framework, `agent-utilities` employs a decoupled **Dual-Architecture Strategy**:

<div class="admonition architecture" markdown>
<p class="admonition-title">Dual-architecture strategy: local weights vs. API simulation</p>

An incoming task checks whether its model is local & open-weights. If
yes (local GPU), it runs the native open-weights pipeline: dynamic
PyTorch/vLLM import, last-hidden-state weight access, direct projection
tensors. If no (off-the-shelf API), it runs the universal API semantic
simulator: zero extra VRAM/PyTorch imports, local REPL variable state
passing, semantic thought vectors via embeddings. Both paths converge on
Orchestration Engine execution.
</div>

#### 1. Native Open-Weights Pipeline (Optional GPU Mode)
For specialized local runs executing open-source weights (via PyTorch, Hugging Face `transformers`, or custom vLLM adapters):
* The system accesses the model's `last_hidden_state` activations during generation, runs them through the lightweight PyTorch `RecursiveLink` projection layers, and injects them directly into the input attention space of the next agent.
* **Decoupled Security**: All neural modeling code would be isolated in a modular wrapper (planned `agent_utilities/rlm/mas_local.py`, not yet present). This ensures **zero dependencies** (like PyTorch) are imported during standard framework operations, maintaining a strict zero-overhead baseline.

#### 2. Universal API Semantic Simulator (Off-the-Shelf Fallback)
For standard deployments using cloud-hosted models (Gemini, OpenAI, Claude) where hidden layer access is technically restricted, the orchestrator extrapolates the core benefits of RecursiveMAS via **API-level symbolic emulation**:

* **State Containment via persistent REPL**: Rather than passing raw chat logs back and forth via the API, the orchestrator keeps all intermediate calculations, databases, and heavy text dumps stored locally in variables inside the persistent RLM Python REPL (`RLMEnvironment`).
* **Metadata-Only Prompting**: The API prompt is fed only constant-size **metadata** (e.g. variable names, types, and lengths) rather than raw variables. Agents interact by writing python code to mutate REPL states, achieving the whitepaper's **75% token reduction** and preventing context pollution entirely.
* **Semantic Embedding Vectors**: Agents share intermediate thought states by passing lightweight high-dimensional embedding vectors (retrieved via cheap off-the-shelf embedding endpoints) representing the semantic "Thought Mementos". These vectors are used to programmatically query the local Knowledge Graph or rank context slices without generating raw text, mimicking the continuous latent state hand-off of the neural pipeline.

### ORCH-1.27 — Role-Specialized Model Routing

Assimilated from Quarq Agent's three-specialized-model pattern (planner / generator /
learner; `agent-oss/agent.py:58-92`), generalized so functional **roles** bind to model
*tiers + capability tags* rather than hardcoded model ids. `ModelRegistry.pick_for_role(role)`
resolves `planner|generator|learner|judge` through the existing `pick_for_task` tier-fallback,
so the same configuration runs on any provider pool (local LM Studio, cloud frontier, mixed) and
degrades gracefully. Overridable per-call, via `ModelRegistry.role_routing`, via
`AgentConfig.role_routing`, or live through `graph_configure(action="set_role_routing")`. This is
the routing substrate for the memory-first synergy pipeline: the HyDE planner (KG-2.12), the
background learner (KG-2.13), and the LongMemEval judge all request their role here. Extends ORCH-1.2.

### ORCH-1.2 — Global Workspace Attention loop (revived + instrumented)

After each multi-agent wave, `ParallelEngine` drives a Global Workspace Theory loop:
`WorkspaceAttention` scores specialist outputs (relevance·track-record·confidence),
selects the top-K, and **broadcasts** the winners to the KG as `ProposalNode`s.
`get_attention_score(specialist)` reads those back as each specialist's runtime
standing, feeding routing/confidence in `pick_specialist_model`. The loop is
instrumented with write/read counters and a `suspected_engine_mismatch` guard
(surfaced in `ExecutionResult.telemetry["workspace_attention"]`; strict mode via
`AGENT_UTILITIES_GWT_STRICT`). Winners are also recorded into the
`EvolvingMemoryStore` INSIGHT bank. Full design:
[Global Workspace Attention](../architecture/global_workspace_attention.md).

### ORCH-1.32 — Multi-Agent Social System (MASS)

The swarm is modeled as a social system `S=(f,g,G)`: archetype-tagged agents over
an explicit interaction graph (built from manifest `depends_on` edges), with
local-neighborhood observability, a co-evolution edge-update loop, and a P1–P4
**swarm-health** snapshot (degree-partition heterogeneity, topology variance,
neighbor co-evolution slope, Wasserstein-1 drift). `ParallelEngine` attaches the
snapshot to `ExecutionResult.telemetry["social_system"]`. Full design:
[Multi-Agent Social System](../architecture/multi_agent_social_system.md).

### AU-ECO.bus.agentbus-federated-agent-agent/4.88 — Agent-to-agent coordination over the AgentBus

Swarm and sub-agent coordination is not limited to manifest fan-in/synthesis: every agent
inherits the **AgentBus** as a native capability (CONCEPT:AU-ECO.bus.agent-bus-awareness) and can message peers —
across sessions, providers, and hosts — through the universal `bus_join`/`bus_peers`/`bus_send`/
`bus_check` tools (or the `graph_bus` MCP tool). The orchestrator (the "graph shaper") knows it
can stand up agent-to-agent communication, and `action=swarm` gives each wave a shared bus topic
so peers announce work and share findings instead of duplicating. Heavy work is handed to the
fleet with `graph_bus(action='dispatch')`. Full design:
[Agent Communication Bus](https://knuckles-team.github.io/graph-os/architecture/agent-bus/).

### ORCH-1.41 / 1.42 / 1.43 — Ontology-to-Workflow Execution Path { #ontology-workflow-execution }

Descriptive process knowledge in the KG is now executable, with the ontology in
the loop at every step:

- **ORCH-1.41 — Process Plan Compiler** (`knowledge_graph/process_plan_compiler.py`):
  `graph_workflows(action="compile_process")` (REST twin
  `/api/graph/workflows`) lifts a descriptive BPMN process —
  ingested via the Camunda extractor and given step-level ontology shape by
  **AU-KG.ontology.descriptive-process-world-gains** — into an executable plan.
- **AU-ORCH.execution.ontology-validation-execution-path — Execution Ontology Gate** (`knowledge_graph/core/workflow_gate.py`):
  ontology validation sits on the execution path, so a compiled process is
  checked against the published ontology before it runs.
- **ORCH-1.43 — Lineage Close-Out** (`workflows/runner.py` + `core/owl_bridge.py`):
  workflow runs write lineage back to the KG, closing the
  descriptive↔executable provenance loop — the process model, the compiled
  plan, and the run that executed it stay connected.

The workflow gate reads the ontology from the epistemic-graph authority. The
former AU Fuseki publishing tick is retired; it was never wired into the
engine task scheduler.
Walkthrough: [ontology-to-workflow example](../examples/ontology-to-workflow.md).

### AU-ORCH.session.durable-goal-registry-goals — Durable Goal Registry

Goals are durable records in the externalized state store (AU-OS.state.unified-durable-state-externalization), not
in-process objects: they persist across gateway restarts, and a run stranded by
a crashed host rehydrates as `orphaned` instead of silently vanishing
(`core/sessions.py`, `models/goal.py`). See
[State Externalization](../architecture/state_externalization.md).

### ORCH-1.45 — Queue-Driven Agent Dispatch

Agent turns (goal-loop iterations and orchestrator jobs) can dispatch through a
session-partitioned durable queue instead of the in-process scheduler:
`graph_jobs(action="dispatch")` and the goal machinery enqueue a
typed `AgentTurnEnvelope`
(`orchestration/agent_dispatch.py` — job id as idempotency key; payload stays a
*reference* into the state store) onto the `agent_turns` queue (Kafka, Postgres
SKIP LOCKED, or per-host SQLite — composing the KG-2.55 transport stack with a
`session:<id>` partition key above the KG-2.56 tenant key). A stateless
**`agent-dispatch-worker`** fleet claims turns under per-session mutual
exclusion, rehydrates from the shared state store, executes the existing
goal/orchestration bodies, and writes back durably before acking —
at-least-once with idempotent re-claims, so a crashed worker is crash recovery,
not data loss. Workers heartbeat into the `dispatch_workers` registry,
`/api/fleet/topology` lists them, and `graph_orchestrate job/{id}` reports the
executing worker. The `inline` default is byte-for-byte the previous behavior.
Full design: [Queue-Driven Agent Dispatch](../architecture/agent_dispatch.md);
walkthrough: [queue-dispatch example](../examples/queue-dispatch-walkthrough.md).

### ORCH-1.50 — Task-Management Ergonomics on SDD

The Spec-Driven Development pipeline (AU-ORCH.planning.spec-driven-pipeline) already persists a durable,
dependency-aware task list — `Spec` / `Task` / `Tasks` / `ImplementationPlan`
(`models/sdd.py`) round-tripped to `.specify/` by `SDDManager` (`sdd/__init__.py`).
ORCH-1.50 adds the *loop-driving* ergonomics on top, so a long-horizon goal can be
decomposed, scored, and worked one actionable task at a time:

- **`parse_prd(prd_text, feature_id)`** — decompose a PRD into sequential, dependency-linked
  tasks (zero-infra structural parser by default; an LLM decomposer is injectable).
- **`analyze_complexity(feature_id)`** — score each task 0–10 and recommend a subtask
  count (a deployable structural heuristic by default; an LLM scorer is injectable),
  persisting a report under `.specify/reports/`.
- **`Tasks.next_task()`** — pick the next actionable task, preferring subtasks of an
  in-progress parent, then top-level tasks whose dependencies are satisfied, breaking
  ties by priority → fewer deps → id. `detect_cycles()` / `validate_dependencies()`
  reject an unschedulable graph before work starts.
- **`scope_task(... "up"|"down")`** — renegotiate scope, preserving done/in-progress
  subtasks. **Tagged contexts** (via `feature_id`, with `branch_tasks` / `list_task_contexts`)
  give parallel task streams.

New fields on `Task` (`priority`, `complexity_score`, `recommended_subtasks`,
`test_strategy`, `expansion_prompt`) round-trip through a full-fidelity `tasks.json`
sidecar (the markdown mirror stays human-readable). Surfaced over the harness MCP
server as `task_parse_prd`, `task_analyze_complexity`, `task_next`, `task_set_status`,
and `task_scope` (`mcp/harness_server.py`).

<div class="admonition architecture" markdown>
<p class="admonition-title">Task lifecycle: parse, score, scope, validate, work</p>

`parse_prd` turns a PRD/goal into Tasks (`.specify` + `tasks.json`).
`analyze_complexity` scores those tasks and recommends subtasks, which
`scope_task` (up/down) writes back into Tasks. `validate_dependencies`
checks for a cycle: a cycle rejects; otherwise `next_task` selects the
next task, and working it calls `set_task_status`, writing back into
Tasks.
</div>
