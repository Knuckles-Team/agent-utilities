# Agentic Design Patterns — Architecture Alignment

> Full coverage matrix showing alignment between *Agentic Design Patterns*
> (Antonio Gulli, 2025) and the `agent-utilities` concept architecture.

## Coverage Matrix

| # | Design Pattern | agent-utilities Concept(s) | Status |
|---|---|---|---|
| 1 | **Prompt Chaining** | CONCEPT:AU-ORCH.planning.recursion-nesting-depth (Prompt Chain Executor) | ✅ |
| 2 | **Routing** | CONCEPT:AU-ORCH.execution.inject-signal-board-observations (Graph Orchestration / Router) | ✅ |
| 3 | **Parallelization** | CONCEPT:AU-ORCH.execution.inject-signal-board-observations (parallel_batch_processor, fan-out/fan-in) | ✅ |
| 4 | **Reflection** | CONCEPT:AU-ORCH.planning.recursion-nesting-depth (AHE critique), CONCEPT:AU-KG.memory.tiered-memory-caching (Self-Model) | ✅ |
| 5 | **Tool Use** | CONCEPT:AU-ECO.messaging.native-backend-abstraction (Tool/MCP Registry) | ✅ |
| 6 | **Planning** | CONCEPT:AU-AHE.harness.harness-evolution (SDD pipeline), Planner node | ✅ |
| 7 | **Multi-Agent** | CONCEPT:AU-KG.query.object-graph-mapper (Swarm), Council, Teams | ✅ |
| 8 | **Memory Management** | KG engine, MAGMA views, episodic memory | ✅ |
| 9 | **Learning & Adaptation** | CONCEPT:AU-ORCH.execution.inject-signal-board-observations (Evolutionary Variants), CONCEPT:AU-ORCH.planning.recursion-nesting-depth (AHE) | ✅ |
| 10 | **MCP** | CONCEPT:AU-ECO.messaging.native-backend-abstraction, `mcp/` module | ✅ |
| 11 | **Goal Setting** | CONCEPT:AU-AHE.harness.harness-evolution (SDD spec/verify), Verifier node | ✅ |
| 12 | **Exception Handling** | CONCEPT:AU-OS.state.cognitive-scheduler-preemption (stuck-loop, circuit-breaker) | ✅ |
| 13 | **Human-in-the-Loop** | ApprovalManager, tool_guard, elicitation | ✅ |
| 14 | **Knowledge Retrieval** | HybridRetriever, KB layer, MAGMA | ✅ |
| 15 | **Inter-Agent Comm** | CONCEPT:AU-OS.safety.doom-loop-detection (A2A adapter) | ✅ |
| 16 | **Resource Optimization** | CONCEPT:AU-OS.state.cognitive-scheduler-preemption (ResourceOptimizer) | ✅ |
| 17 | **Reasoning** | CONCEPT:AU-ORCH.planning.recursion-nesting-depth (RLM), Council deliberation | ✅ |
| 18 | **Guardrails/Safety** | `security/guardrails.py`, tool_guard | ✅ |
| 19 | **Evaluation & Monitoring** | CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort (EvaluationMonitor) | ✅ |
| 20 | **Prioritization** | CONCEPT:AU-ORCH.planning.recursion-nesting-depth (PrioritizationEngine) | ✅ |
| 21 | **Exploration & Discovery** | CONCEPT:AU-AHE.harness.evolutionary-aggregation (ExplorationEngine) | ✅ |

## New Concepts (CONCEPT:AU-ORCH.planning.recursion-nesting-depth through CONCEPT:AU-AHE.harness.evolutionary-aggregation)

### CONCEPT:AU-ORCH.planning.recursion-nesting-depth: Prompt Chaining Pattern

**Module**: `agent_utilities/patterns/prompt_chain.py`

Declarative multi-step prompt pipelines with intermediate validation,
conditional branching, and KG persistence. Each chain step transforms
its predecessor's output and optionally validates against a Pydantic model.

**Key Classes**:
- `PromptChainStep` — Single step definition with template and branch logic
- `PromptChain` — Ordered collection of steps with retry configuration
- `PromptChainExecutor` — Async executor managing state flow between steps

**OWL**: `:PromptChain rdfs:subClassOf :Procedure`

---

### CONCEPT:AU-OS.state.cognitive-scheduler-preemption: Resource-Aware Optimization

**Module**: `agent_utilities/core/resource_optimizer.py`

Cost-aware model selection, per-specialist budget allocation, and latency
routing with configurable industry-standard defaults.

**Key Classes**:
- `ResourceBudget` — Per-session budget with token/cost/latency tracking
- `ResourceOptimizer` — Budget allocation, model selection, usage recording

**Configuration** (env vars):
- `SESSION_TOKEN_BUDGET` — Default: 500,000 tokens
- `SESSION_COST_BUDGET_USD` — Default: $5.00
- `SESSION_LATENCY_BUDGET_MS` — Default: 30,000ms

**OWL**: `:ResourceUsage rdfs:subClassOf bfo:Process`

---

### CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort: Evaluation & Monitoring

**Module**: `agent_utilities/harness/continuous_evaluation_engine.py`

Multi-dimensional evaluation with LLM-as-Judge rubrics, trend monitoring,
and quality alerting. Backward-compatible with existing verifier gate
through composite score.

**Key Classes**:
- `EvaluationDimension` — Single dimension score with rubric and evidence
- `MultiDimensionalEvaluation` — Aggregated dimensions with composite
- `EvaluationMonitor` — Trend tracking, alerting, KG persistence

**Dimensions**: correctness (35%), completeness (25%), relevance (25%), safety (15%)

**OWL**: `:EvaluationRecord rdfs:subClassOf :Observation`

---

### CONCEPT:AU-ORCH.planning.recursion-nesting-depth: Task Prioritization

**Module**: `agent_utilities/patterns/prioritization.py`

Multi-factor priority scoring with dependency tracking, priority
inheritance, and capability-based specialist assignment.

**Key Classes**:
- `PriorityScore` — Multi-factor score (urgency × impact × effort × risk)
- `PrioritizationEngine` — Task management with reprioritization and ordering

**Weight Formula** (global defaults):
- Urgency: 35%, Impact: 30%, Effort: 20% (inverse), Risk: 15%

**OWL**: `:PrioritizedTask rdfs:subClassOf :Action`, `:blocks owl:TransitiveProperty`

---

### CONCEPT:AU-AHE.harness.evolutionary-aggregation: Exploration & Discovery

**Module**: `agent_utilities/patterns/exploration.py`

Autonomous exploration loop for arbitrary domain discovery (code, medical,
finance, etc.) with hypothesis generation, experiment design, and
multi-reviewer evaluation.

**Key Classes**:
- `KnowledgeGap` — Identified knowledge deficit with severity scoring
- `Hypothesis` — Testable prediction linked to a gap
- `Experiment` — Structured test with variables and success criteria
- `ExplorationEngine` — Full lifecycle management from gap → discovery

**Lineage**: `KnowledgeGap → Hypothesis → Experiment → Discovery → KBFact`

**OWL**: `:Experiment rdfs:subClassOf :Procedure`, `:testsHypothesis`, `:exploredGap`

## KG Node & Edge Types Added

### Node Types
| Type | Concept | Description |
|---|---|---|
| `PROMPT_CHAIN` | CONCEPT:AU-ORCH.planning.recursion-nesting-depth | Prompt pipeline definition with steps |
| `RESOURCE_USAGE` | CONCEPT:AU-OS.state.cognitive-scheduler-preemption | Per-session resource consumption record |
| `EVALUATION_RECORD` | CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort | Multi-dimensional evaluation result |
| `PRIORITIZED_TASK` | CONCEPT:AU-ORCH.planning.recursion-nesting-depth | Task with multi-factor priority scoring |
| `KNOWLEDGE_GAP` | CONCEPT:AU-AHE.harness.evolutionary-aggregation | Identified knowledge deficit |
| `EXPLORATION_EXPERIMENT` | CONCEPT:AU-AHE.harness.evolutionary-aggregation | Experiment testing a hypothesis |

### Edge Types
| Type | Concept | Description |
|---|---|---|
| `CHAIN_STEP` | CONCEPT:AU-ORCH.planning.recursion-nesting-depth | Ordered step within a prompt chain |
| `BRANCHES_TO` | CONCEPT:AU-ORCH.planning.recursion-nesting-depth | Conditional branch between steps |
| `CONSUMED_RESOURCE` | CONCEPT:AU-OS.state.cognitive-scheduler-preemption | Episode → resource usage link |
| `EVALUATED_WITH` | CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort | Episode → evaluation record link |
| `CALIBRATED_AGAINST` | CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort | Machine vs human evaluation link |
| `BLOCKS` | CONCEPT:AU-ORCH.planning.recursion-nesting-depth | Task blocking relationship |
| `ASSIGNED_TO_SPECIALIST` | CONCEPT:AU-ORCH.planning.recursion-nesting-depth | Task → specialist assignment |
| `TESTS_HYPOTHESIS` | CONCEPT:AU-AHE.harness.evolutionary-aggregation | Experiment → hypothesis link |
| `EXPLORED_GAP` | CONCEPT:AU-AHE.harness.evolutionary-aggregation | Hypothesis → knowledge gap link |
| `RESULTED_IN_DISCOVERY` | CONCEPT:AU-AHE.harness.evolutionary-aggregation | Experiment → fact/discovery link |
