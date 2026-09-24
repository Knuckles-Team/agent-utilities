# Conductor Orchestration (CONCEPT:AU-ORCH.planning.recursion-nesting-depth to CONCEPT:AU-ORCH.planning.recursion-nesting-depth)

> **Source**: *"Learning to Orchestrate Agents in Natural Language with the Conductor"* — Nielsen et al., ICLR 2026, Sakana AI

This document describes four architectural concepts inspired by the RL Conductor paper that enhance the agent-utilities orchestration pipeline with refined subtask decomposition, context isolation, model synergy tracking, and recursive self-referential graph execution.

## CONCEPT:AU-ORCH.planning.recursion-nesting-depth: Conductor Workflow Specification

**Module**: `agent_utilities/models/sdd.py` — `ExecutionStep.refined_subtask`

### Motivation

In the original orchestration pipeline, each specialist receives the raw user query verbatim. The Conductor paper demonstrates that a trained policy generates focused, specialist-specific sub-instructions that significantly outperform raw query forwarding.

### Design

The `refined_subtask` field on `ExecutionStep` carries a natural-language instruction crafted by the router/planner for each specific specialist:

```python
ExecutionStep(
    node_id="python_programmer",
    refined_subtask="Implement a FastAPI REST API with JWT authentication middleware",
    access_list=["researcher"],  # CONCEPT:AU-ORCH.planning.recursion-nesting-depth: only sees researcher output
)
```

**Executor preference chain**: `refined_subtask` → `input_data` → `state.query`

### Router Integration

The router's system prompt now includes CONCEPT:AU-ORCH.planning.recursion-nesting-depth instructions:

> For EACH step in your plan, include a `refined_subtask` — a focused,
> specific instruction tailored for that specialist.

---

## CONCEPT:AU-ORCH.planning.recursion-nesting-depth: Execution Visibility Graph

**Module**: `agent_utilities/graph/executor.py` — `_resolve_access_context()`

### Motivation

The Conductor paper defines an `access_list` per workflow step that controls which prior step outputs are visible. This prevents context pollution where a specialist receives irrelevant information from unrelated prior steps.

### Design

The `access_list` field on `ExecutionStep` supports three modes:

| Access List | Behavior |
|:---|:---|
| `[]` (empty) | No prior results shared |
| `["all"]` | Full `results_registry` injected |
| `["researcher", "architect"]` | Only matching results injected |

### Data Flow

<div class="admonition architecture" markdown>
<p class="admonition-title">The Synthesizer only sees what its access_list names</p>

Researcher, Architect, and Programmer each write their results into
`results_registry` under their own key. A filter reads that registry
against the Synthesizer's `access_list` (here: researcher, architect) —
those two results pass through; the Programmer's result, not on the list,
is blocked and hidden from the Synthesizer entirely.
</div>

### Helper Function

```python
def _resolve_access_context(
    step: ExecutionStep,
    results_registry: dict[str, Any],
) -> str:
    """Filter results_registry based on step's access_list."""
```

---

## CONCEPT:AU-AHE.evaluation.interpretability-tests: Model Synergy Tracker

**Module**: `agent_utilities/knowledge_graph/retrieval/memory_retriever.py` (`MemoryRetriever`)

### Motivation

When multiple models collaborate in a session, certain combinations consistently outperform others. The Conductor paper's adaptive worker pool selection inspires tracking which model combinations work best together.

### Design

The `model_synergies` field on `MemoryRetrieverNode` (in `models/knowledge_graph.py`) stores sorted, pipe-delimited model combination keys with EMA success rates:

```python
model_synergies = {
    "model-a|model-b": 0.85,    # Strong combination
    "model-c|model-d": 0.45,    # Weak combination
}
```

### EMA Update Rule

```
new_rate = α × session_success + (1 - α) × old_rate
```

Where `α = 0.3` and `session_success ∈ {0.0, 1.0}`.

### Query Interface

```python
synergies = self_model.get_best_synergies(
    available_models=["model-a", "model-b", "model-c"],
    top_k=3,
)
# Returns: [("model-a|model-b", 0.85), ...]
```

---

## CONCEPT:AU-ORCH.planning.recursion-nesting-depth: Recursive Graph Orchestration

**Module**: `agent_utilities/graph/hierarchical_planner.py` (`RecursiveContext`, `MAX_RECURSION_DEPTH`, `RecursionDepthExceeded`). The `recursive_orchestrator` specialist itself is registered in `agent_utilities/graph/executor.py`.

### Motivation

The Conductor paper demonstrates that allowing the orchestrator to specify *itself* as a worker enables adaptive test-time scaling. When a plan fails, a recursive call can devise and execute a fundamentally different strategy using the parent's error context.

### Design

<div class="admonition architecture" markdown>
<p class="admonition-title">A failed plan recurses one level, bounded by MAX_RECURSION_DEPTH</p>

A user query drives parent graph execution. When its plan fails,
`recursive_orchestrator` builds a `RecursiveContext` and spawns a recursive
graph execution, whose result feeds back into the parent. If nesting depth
would exceed the configured maximum, the recursive execution raises
`RecursionDepthExceeded` instead of recursing further.
</div>

### Depth Control

| Configuration | Default | Description |
|:---|:---|:---|
| `MAX_RECURSION_DEPTH` env var | `2` | Hard ceiling on nesting depth |
| `GraphState.recursion_depth` | `0` | Current nesting level |

### RecursiveContext

```python
@dataclass
class RecursiveContext:
    parent_query: str
    parent_plan_summary: str
    parent_error: str
    parent_results: dict[str, Any]
    recursion_depth: int = 1
```

### Composition with CONCEPT:AU-ORCH.planning.recursion-nesting-depth (RLM)

Both CONCEPT:AU-ORCH.planning.recursion-nesting-depth (RLM) and CONCEPT:AU-ORCH.planning.recursion-nesting-depth provide recursive execution, but at different levels:

- **RLM (CONCEPT:AU-ORCH.planning.recursion-nesting-depth)**: Sub-shell-level recursion within a single specialist
- **Recursive Orchestration (CONCEPT:AU-ORCH.planning.recursion-nesting-depth)**: Graph-level recursion that re-plans the entire specialist topology

These compose naturally — an inner recursive graph can still use RLM within its specialists.

---

## Configuration Reference

| Variable | Default | Description |
|:---|:---|:---|
| `MAX_RECURSION_DEPTH` | `2` | Maximum nesting depth for recursive orchestration (CONCEPT:AU-ORCH.planning.recursion-nesting-depth) |

## Related Concepts

| Concept | Relationship |
|:---|:---|
| CONCEPT:AU-ORCH.execution.inject-signal-board-observations (Graph Orchestration) | CONCEPT:AU-ORCH.planning.recursion-nesting-depth extend the graph plan model |
| CONCEPT:AU-ORCH.planning.recursion-nesting-depth (RLM) | CONCEPT:AU-ORCH.planning.recursion-nesting-depth composes with RLM at different recursion levels |
| CONCEPT:AU-KG.memory.tiered-memory-caching (Self-Model) | CONCEPT:AU-AHE.evaluation.interpretability-tests extends SelfModel with synergy tracking |
| CONCEPT:AU-ORCH.adapter.hot-cache-invalidation (Confidence-Gated Router) | CONCEPT:AU-AHE.evaluation.interpretability-tests feeds synergy data into routing decisions |
| CONCEPT:AU-KG.ingest.engineering-rules (KG Eval Capture) | Future: reward tuples from CONCEPT:AU-KG.ingest.engineering-rules will train the Conductor policy |
