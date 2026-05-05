# Agentic Design Patterns — Architecture Alignment

> Full coverage matrix showing alignment between *Agentic Design Patterns*
> (Antonio Gulli, 2025) and the `agent-utilities` concept architecture.

## Coverage Matrix

| # | Design Pattern | agent-utilities Concept(s) | Status |
|---|---|---|---|
| 1 | **Prompt Chaining** | AU-018 (Prompt Chain Executor) | ✅ |
| 2 | **Routing** | AU-002 (Graph Orchestration / Router) | ✅ |
| 3 | **Parallelization** | AU-002 (parallel_batch_processor, fan-out/fan-in) | ✅ |
| 4 | **Reflection** | AU-012 (AHE critique), AU-016 (Self-Model) | ✅ |
| 5 | **Tool Use** | AU-010 (Tool/MCP Registry) | ✅ |
| 6 | **Planning** | AU-009 (SDD pipeline), Planner node | ✅ |
| 7 | **Multi-Agent** | AU-014 (Swarm), Council, Teams | ✅ |
| 8 | **Memory Management** | KG engine, MAGMA views, episodic memory | ✅ |
| 9 | **Learning & Adaptation** | AU-015 (Evolutionary Variants), AU-012 (AHE) | ✅ |
| 10 | **MCP** | AU-010, `mcp/` module | ✅ |
| 11 | **Goal Setting** | AU-009 (SDD spec/verify), Verifier node | ✅ |
| 12 | **Exception Handling** | AU-008 (stuck-loop, circuit-breaker) | ✅ |
| 13 | **Human-in-the-Loop** | ApprovalManager, tool_guard, elicitation | ✅ |
| 14 | **Knowledge Retrieval** | HybridRetriever, KB layer, MAGMA | ✅ |
| 15 | **Inter-Agent Comm** | AU-004 (A2A adapter) | ✅ |
| 16 | **Resource Optimization** | AU-019 (ResourceOptimizer) | ✅ |
| 17 | **Reasoning** | AU-007 (RLM), Council deliberation | ✅ |
| 18 | **Guardrails/Safety** | `security/guardrails.py`, tool_guard | ✅ |
| 19 | **Evaluation & Monitoring** | AU-020 (EvaluationMonitor) | ✅ |
| 20 | **Prioritization** | AU-021 (PrioritizationEngine) | ✅ |
| 21 | **Exploration & Discovery** | AU-022 (ExplorationEngine) | ✅ |

## New Concepts (AU-018 through AU-022)

### AU-018: Prompt Chaining Pattern

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

### AU-019: Resource-Aware Optimization

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

### AU-020: Evaluation & Monitoring

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

### AU-021: Task Prioritization

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

### AU-022: Exploration & Discovery

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
| `PROMPT_CHAIN` | AU-018 | Prompt pipeline definition with steps |
| `RESOURCE_USAGE` | AU-019 | Per-session resource consumption record |
| `EVALUATION_RECORD` | AU-020 | Multi-dimensional evaluation result |
| `PRIORITIZED_TASK` | AU-021 | Task with multi-factor priority scoring |
| `KNOWLEDGE_GAP` | AU-022 | Identified knowledge deficit |
| `EXPLORATION_EXPERIMENT` | AU-022 | Experiment testing a hypothesis |

### Edge Types
| Type | Concept | Description |
|---|---|---|
| `CHAIN_STEP` | AU-018 | Ordered step within a prompt chain |
| `BRANCHES_TO` | AU-018 | Conditional branch between steps |
| `CONSUMED_RESOURCE` | AU-019 | Episode → resource usage link |
| `EVALUATED_WITH` | AU-020 | Episode → evaluation record link |
| `CALIBRATED_AGAINST` | AU-020 | Machine vs human evaluation link |
| `BLOCKS` | AU-021 | Task blocking relationship |
| `ASSIGNED_TO_SPECIALIST` | AU-021 | Task → specialist assignment |
| `TESTS_HYPOTHESIS` | AU-022 | Experiment → hypothesis link |
| `EXPLORED_GAP` | AU-022 | Hypothesis → knowledge gap link |
| `RESULTED_IN_DISCOVERY` | AU-022 | Experiment → fact/discovery link |
