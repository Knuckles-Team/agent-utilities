# Agentic Design Patterns — Architecture Alignment

> Full coverage matrix showing alignment between *Agentic Design Patterns*
> (Antonio Gulli, 2025) and the `agent-utilities` concept architecture.

## Coverage Matrix

| # | Design Pattern | agent-utilities Concept(s) | Status |
|---|---|---|---|
| 1 | **Prompt Chaining** | CONCEPT:ORCH-1.1 (Prompt Chain Executor) | ✅ |
| 2 | **Routing** | CONCEPT:ORCH-1.0 (Graph Orchestration / Router) | ✅ |
| 3 | **Parallelization** | CONCEPT:ORCH-1.0 (parallel_batch_processor, fan-out/fan-in) | ✅ |
| 4 | **Reflection** | CONCEPT:ORCH-1.1 (AHE critique), CONCEPT:KG-2.1 (Self-Model) | ✅ |
| 5 | **Tool Use** | CONCEPT:ECO-4.1 (Tool/MCP Registry) | ✅ |
| 6 | **Planning** | CONCEPT:AHE-3.0 (SDD pipeline), Planner node | ✅ |
| 7 | **Multi-Agent** | CONCEPT:KG-2.0 (Swarm), Council, Teams | ✅ |
| 8 | **Memory Management** | KG engine, MAGMA views, episodic memory | ✅ |
| 9 | **Learning & Adaptation** | CONCEPT:ORCH-1.0 (Evolutionary Variants), CONCEPT:ORCH-1.1 (AHE) | ✅ |
| 10 | **MCP** | CONCEPT:ECO-4.1, `mcp/` module | ✅ |
| 11 | **Goal Setting** | CONCEPT:AHE-3.0 (SDD spec/verify), Verifier node | ✅ |
| 12 | **Exception Handling** | CONCEPT:OS-5.2 (stuck-loop, circuit-breaker) | ✅ |
| 13 | **Human-in-the-Loop** | ApprovalManager, tool_guard, elicitation | ✅ |
| 14 | **Knowledge Retrieval** | HybridRetriever, KB layer, MAGMA | ✅ |
| 15 | **Inter-Agent Comm** | CONCEPT:OS-5.0 (A2A adapter) | ✅ |
| 16 | **Resource Optimization** | CONCEPT:OS-5.2 (ResourceOptimizer) | ✅ |
| 17 | **Reasoning** | CONCEPT:ORCH-1.1 (RLM), Council deliberation | ✅ |
| 18 | **Guardrails/Safety** | `security/guardrails.py`, tool_guard | ✅ |
| 19 | **Evaluation & Monitoring** | CONCEPT:AHE-3.1 (EvaluationMonitor) | ✅ |
| 20 | **Prioritization** | CONCEPT:ORCH-1.1 (PrioritizationEngine) | ✅ |
| 21 | **Exploration & Discovery** | CONCEPT:AHE-3.2 (ExplorationEngine) | ✅ |

## New Concepts (CONCEPT:ORCH-1.1 through CONCEPT:AHE-3.2)

### CONCEPT:ORCH-1.1: Prompt Chaining Pattern

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

### CONCEPT:OS-5.2: Resource-Aware Optimization

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

### CONCEPT:AHE-3.1: Evaluation & Monitoring

**Module**: `agent_utilities/observability/evaluation.py`

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

### CONCEPT:ORCH-1.1: Task Prioritization

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

### CONCEPT:AHE-3.2: Exploration & Discovery

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
| `PROMPT_CHAIN` | CONCEPT:ORCH-1.1 | Prompt pipeline definition with steps |
| `RESOURCE_USAGE` | CONCEPT:OS-5.2 | Per-session resource consumption record |
| `EVALUATION_RECORD` | CONCEPT:AHE-3.1 | Multi-dimensional evaluation result |
| `PRIORITIZED_TASK` | CONCEPT:ORCH-1.1 | Task with multi-factor priority scoring |
| `KNOWLEDGE_GAP` | CONCEPT:AHE-3.2 | Identified knowledge deficit |
| `EXPLORATION_EXPERIMENT` | CONCEPT:AHE-3.2 | Experiment testing a hypothesis |

### Edge Types
| Type | Concept | Description |
|---|---|---|
| `CHAIN_STEP` | CONCEPT:ORCH-1.1 | Ordered step within a prompt chain |
| `BRANCHES_TO` | CONCEPT:ORCH-1.1 | Conditional branch between steps |
| `CONSUMED_RESOURCE` | CONCEPT:OS-5.2 | Episode → resource usage link |
| `EVALUATED_WITH` | CONCEPT:AHE-3.1 | Episode → evaluation record link |
| `CALIBRATED_AGAINST` | CONCEPT:AHE-3.1 | Machine vs human evaluation link |
| `BLOCKS` | CONCEPT:ORCH-1.1 | Task blocking relationship |
| `ASSIGNED_TO_SPECIALIST` | CONCEPT:ORCH-1.1 | Task → specialist assignment |
| `TESTS_HYPOTHESIS` | CONCEPT:AHE-3.2 | Experiment → hypothesis link |
| `EXPLORED_GAP` | CONCEPT:AHE-3.2 | Hypothesis → knowledge gap link |
| `RESULTED_IN_DISCOVERY` | CONCEPT:AHE-3.2 | Experiment → fact/discovery link |
