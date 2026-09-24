# Epistemic Resource Scheduler (CONCEPT:AU-OS.scaling.epistemic-dynamic-priority-quota)

## Overview
Traditional schedulers allocate compute based on basic heuristics like NICE levels or first-come, first-served queues. The **Epistemic Resource Scheduler** scales scheduling priority, CPU affinity, and execution token budgets dynamically based on the topological importance of each agent inside the active Knowledge Graph.

By computing the degree centrality of active specialists, the scheduler prioritizes nodes that act as reasoning bottlenecks (e.g., shared planners or legal validators).

## Priority Formula

The scheduler applies a two-stage priority adjustment when an agent process is submitted:

### Stage 1: Centrality Computation

```python
centrality = degree(agent_id) / (num_nodes - 1)
```

Where `degree(agent_id)` is the number of edges (both in and out) connected to the agent's node in the active Knowledge Graph. This approximates eigenvector centrality for sparse graphs.

| Centrality Range | Interpretation |
|---|---|
| 0.0 – 0.3 | Peripheral agent (leaf node, few connections) |
| 0.3 – 0.6 | Mid-tier agent (moderate connectivity) |
| 0.6 – 1.0 | Hub agent (routing bottleneck, many dependents) |

### Stage 2: Dynamic Scaling

Two adjustments are made based on centrality:

#### Token Quota Scaling
```
if centrality > 0.5:
    final_quota = base_quota × (1.0 + centrality)
```

| Base Quota | Centrality | Final Quota |
|---|---|---|
| 100,000 | 0.3 | 100,000 (unchanged) |
| 100,000 | 0.6 | 160,000 (+60%) |
| 100,000 | 0.8 | 180,000 (+80%) |
| 100,000 | 1.0 | 200,000 (+100%) |

#### Priority Boost
```
if centrality > 0.6 and priority > CRITICAL:
    adjusted_priority = max(CRITICAL, priority - 1)
```

| Original Priority | Centrality | Adjusted Priority |
|---|---|---|
| LOW (3) | 0.7 | NORMAL (2) |
| NORMAL (2) | 0.8 | HIGH (1) |
| HIGH (1) | 0.9 | CRITICAL (0) |
| CRITICAL (0) | 1.0 | CRITICAL (0) — unchanged |

## Preemption Protocol

The Cognitive Scheduler implements a multi-stage preemption cascade:

1. **Budget warning** (85% threshold) — log a `NEAR_QUOTA` warning; no
   action taken.
2. **Cost-aware auto-downgrade** (70% cost threshold) — switch to a
   cheaper model tier (super -> standard -> lite); continue execution
   with degraded quality.
3. **Token quota exceeded** (100%) — checkpoint context to the KG; move
   the process to `PAUSED`; schedule the next waiting process.
4. **Cost budget exceeded** — try one more auto-downgrade; if no cheaper
   tier is available, checkpoint and preempt.

### Context Paging

When a process is preempted, its context is serialized to a KG checkpoint:

```python
checkpoint_id = f"ckpt:{uuid.uuid4().hex[:8]}"
proc.checkpoint_id = checkpoint_id
proc.state = ProcessState.PAUSED
proc.preempted_at = time.time()
```

Resumption restores the checkpoint:
```python
scheduler.resume(process_id)
# → RUNNING if capacity available, WAITING if full
```

## Inference Budget Control

Each `AgentProcess` carries an `InferenceBudget` with cost-aware tier management (Research: 2605.05701v1):

```python
budget = InferenceBudget(
    cost_budget_usd=1.0,        # Max $1 spend
    current_tier="standard",     # Start with GPT-4o class
    auto_downgrade=True,         # Degrade before preempting
    fallback_chain=["super", "standard", "lite"],
    downgrade_threshold=0.70,    # Downgrade at 70% budget usage
)
```

### Tier Cost Model

| Tier | Cost per 1K Tokens | Example Models |
|---|---|---|
| `lite` | $0.00015 | Gemini Flash, GPT-4o-mini |
| `standard` | $0.002 | Gemini Pro, GPT-4o |
| `super` | $0.015 | Gemini Ultra, o3 |

### API

```python
# Record an inference call with cost tracking
result = scheduler.record_inference(proc_id, tokens=5000, model_tier="standard")
# → {"within_budget": True, "cost_incurred": 0.01, "recommended_tier": "standard", "downgraded": False}

# Get budget statistics
stats = scheduler.get_budget_stats(proc_id)
# → {"budget_usage_pct": 45.2, "cost_remaining_usd": 0.548, ...}

# Get recommended tier
tier = scheduler.get_recommended_tier(proc_id)
# → "lite" (if budget pressure is high)
```

## Process Lifecycle States

`WAITING` moves to `RUNNING` when capacity becomes available; `RUNNING`
moves to `COMPLETED` on completion or `FAILED` on failure, or to
`PAUSED` on preemption; `PAUSED` moves back to `WAITING` on resume when
no capacity is available.

## Integration Points

- **Cognitive Scheduler (OS-5.2)**: This concept is implemented directly within `CognitiveScheduler`
- **Knowledge Graph (KG-2.0)**: Centrality computed from `engine.graph` (NetworkX)
- **Convergence Monitor (AHE-3.2)**: Optional multi-loop convergence tracking
- **AgentProcessNode (KG model)**: Processes persisted as KG nodes for observability

## Implementation Details
- **Source Code**: [`cognitive_scheduler.py`](https://github.com/Knuckles-Team/agent-utilities/blob/main/agent_utilities/core/cognitive_scheduler.py) (817 lines)
- **Classes**: `CognitiveScheduler`, `AgentProcess`, `InferenceBudget`, `SchedulerPriority`, `ProcessState`
- **Tests**: [`test_cognitive_scheduler.py`](https://github.com/Knuckles-Team/agent-utilities/blob/main/tests/test_cognitive_scheduler.py)
- **Pillar**: OS
- **Package Export**: `agent_utilities.core.CognitiveScheduler`
