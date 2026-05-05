# Conductor Orchestration (AU-044 to AU-047)

> **Source**: *"Learning to Orchestrate Agents in Natural Language with the Conductor"* — Nielsen et al., ICLR 2026, Sakana AI

This document describes four architectural concepts inspired by the RL Conductor paper that enhance the agent-utilities orchestration pipeline with refined subtask decomposition, context isolation, model synergy tracking, and recursive self-referential graph execution.

## AU-044: Conductor Workflow Specification

**Module**: `agent_utilities/models/graph.py` — `ExecutionStep.refined_subtask`

### Motivation

In the original orchestration pipeline, each specialist receives the raw user query verbatim. The Conductor paper demonstrates that a trained policy generates focused, specialist-specific sub-instructions that significantly outperform raw query forwarding.

### Design

The `refined_subtask` field on `ExecutionStep` carries a natural-language instruction crafted by the router/planner for each specific specialist:

```python
ExecutionStep(
    node_id="python_programmer",
    refined_subtask="Implement a FastAPI REST API with JWT authentication middleware",
    access_list=["researcher"],  # AU-045: only sees researcher output
)
```

**Executor preference chain**: `refined_subtask` → `input_data` → `state.query`

### Router Integration

The router's system prompt now includes AU-044 instructions:

> For EACH step in your plan, include a `refined_subtask` — a focused,
> specific instruction tailored for that specialist.

---

## AU-045: Execution Visibility Graph

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

```mermaid
graph LR
    R[Researcher] -->|"results_registry['researcher']"| F[Filter]
    A[Architect] -->|"results_registry['architect']"| F
    P[Programmer] -->|"results_registry['programmer']"| F
    F -->|access_list: researcher,architect| S[Synthesizer]
    F -.->|blocked| X[Programmer output hidden]
```

### Helper Function

```python
def _resolve_access_context(
    step: ExecutionStep,
    results_registry: dict[str, Any],
) -> str:
    """Filter results_registry based on step's access_list."""
```

---

## AU-046: Model Synergy Tracker

**Module**: `agent_utilities/knowledge_graph/self_model.py`

### Motivation

When multiple models collaborate in a session, certain combinations consistently outperform others. The Conductor paper's adaptive worker pool selection inspires tracking which model combinations work best together.

### Design

The `model_synergies` field on `SelfModelNode` stores sorted, pipe-delimited model combination keys with EMA success rates:

```python
model_synergies = {
    "gpt-4o|claude-sonnet": 0.85,    # Strong combination
    "gemini-2.5|llama-3": 0.45,       # Weak combination
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
    available_models=["gpt-4o", "claude-sonnet", "gemini-2.5"],
    top_k=3,
)
# Returns: [("gpt-4o|claude-sonnet", 0.85), ...]
```

---

## AU-047: Recursive Graph Orchestration

**Module**: `agent_utilities/graph/recursive_executor.py`

### Motivation

The Conductor paper demonstrates that allowing the orchestrator to specify *itself* as a worker enables adaptive test-time scaling. When a plan fails, a recursive call can devise and execute a fundamentally different strategy using the parent's error context.

### Design

```mermaid
graph TD
    Q[User Query] --> G1[Graph Execution L0]
    G1 -->|Plan fails| RC[recursive_orchestrator]
    RC -->|RecursiveContext| G2[Graph Execution L1]
    G2 -->|Result| G1
    G2 -.->|depth > MAX| ERR[RecursionDepthExceeded]
```

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

### Composition with AU-007 (RLM)

Both AU-007 (RLM) and AU-047 provide recursive execution, but at different levels:

- **RLM (AU-007)**: Sub-shell-level recursion within a single specialist
- **Recursive Orchestration (AU-047)**: Graph-level recursion that re-plans the entire specialist topology

These compose naturally — an inner recursive graph can still use RLM within its specialists.

---

## Configuration Reference

| Variable | Default | Description |
|:---|:---|:---|
| `MAX_RECURSION_DEPTH` | `2` | Maximum nesting depth for recursive orchestration (AU-047) |

## Related Concepts

| Concept | Relationship |
|:---|:---|
| AU-002 (Graph Orchestration) | AU-044/AU-045 extend the graph plan model |
| AU-007 (RLM) | AU-047 composes with RLM at different recursion levels |
| AU-016 (Self-Model) | AU-046 extends SelfModel with synergy tracking |
| AU-039 (Confidence-Gated Router) | AU-046 feeds synergy data into routing decisions |
| AU-043 (KG Eval Capture) | Future: reward tuples from AU-043 will train the Conductor policy |
