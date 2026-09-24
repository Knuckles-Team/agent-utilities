# Squeeze Evolve: Confidence-Gated Routing & Evolutionary Aggregation

> **Concepts**: CONCEPT:AU-ORCH.adapter.hot-cache-invalidation (Confidence-Gated Router), CONCEPT:AU-ORCH.adapter.hot-cache-invalidation (Evolutionary Aggregation Engine)
>
> **Based on**: *SQUEEZE EVOLVE: Adaptive Multi-Model Orchestration for Mathematical Reasoning*
> (Maheswaran et al., 2026) — [arXiv:2604.07725v2](https://arxiv.org/abs/2604.07725v2)

## Overview

Agent-utilities integrates two key innovations from the Squeeze Evolve paper
to optimise model dispatch efficiency and reasoning quality:

1. **CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Confidence-Gated Router**: Adaptively selects cheaper or more
   expensive models based on runtime confidence signals from specialist consensus
   and historical performance.

2. **CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Evolutionary Aggregation Engine**: Groups specialist outputs,
   computes group-level fitness (confidence + diversity), and routes aggregation
   to the most cost-effective strategy.

### Core Principle

> *Allocate model capability where it has the highest marginal utility.*

Strong models should initialise solutions (where reasoning depth matters most),
while cheaper models can handle aggregation when specialists already agree.
This achieves significant cost reduction with equivalent or improved accuracy.

---

## CONCEPT:AU-ORCH.adapter.hot-cache-invalidation: Confidence-Gated Model Router

### How It Works

<div class="admonition architecture" markdown>
<p class="admonition-title">Two signals blend into one confidence-gated tier decision</p>

`WorkspaceAttention`'s runtime score and `SelfModel`'s historical
proficiency blend (70% runtime + 30% historical) into a confidence value,
checked against a threshold: high confidence tiers down to a cheaper
model, neutral keeps the current tier, low confidence tiers up to a
stronger model. Every outcome reaches `ModelRegistry.pick_for_task`.
</div>

### Composition with CONCEPT:AU-OS.state.cognitive-scheduler-preemption (Homeostatic Downgrade)

The confidence-gated router composes with the existing budget-pressure
downgrade (CONCEPT:AU-OS.state.cognitive-scheduler-preemption):

1. **CONCEPT:AU-OS.state.cognitive-scheduler-preemption runs first**: If the ResourceOptimizer detects budget pressure, it
   reduces the tier (e.g., `heavy → medium`).
2. **CONCEPT:AU-ORCH.adapter.hot-cache-invalidation runs second**: The confidence signal further adjusts within the
   budget-allowed range (e.g., `medium → light` if confidence is high).

This means budget pressure always takes precedence, but confidence can
further optimise within the budget constraint.

### Self-Model Integration (Soft Dependency)

CONCEPT:AU-ORCH.adapter.hot-cache-invalidation optionally uses historical proficiency from CONCEPT:AU-KG.memory.tiered-memory-caching (Persistent Self-Model):

```python
# When SelfModel is available:
historical_confidence = self_model.tool_proficiency.get(specialist_id, 0.5)
effective = 0.7 * runtime_confidence + 0.3 * historical_confidence

# When no SelfModel (graceful degradation):
historical_confidence = 0.5  # Neutral — no bias
effective = 0.7 * runtime_confidence + 0.15  # Pure runtime signal
```

### Configuration

| Parameter | Location | Default | Description |
|:---|:---|:---|:---|
| `routing_percentile` | `GraphDeps` | `50.0` | Threshold (0–100). Lower = more aggressive downgrade. |
| `ROUTING_PERCENTILE` | Env var | `50.0` | Default for `routing_percentile`. |

### API Reference

```python
from agent_utilities.models.model_registry import ModelRegistry

registry = ModelRegistry(models=[...])
chosen = registry.pick_for_task_adaptive(
    complexity="medium",
    confidence_signal=0.85,     # High confidence → downgrade
    routing_percentile=30.0,    # Aggressive: route 70% to cheap models
    required_tags=["code"],
)
```

---

## CONCEPT:AU-ORCH.adapter.hot-cache-invalidation: Evolutionary Aggregation Engine

### Group Fitness Signals

The engine computes two signals for each group of specialist proposals:

| Signal | Formula | Description |
|:---|:---|:---|
| **Group Confidence (GC)** | `GC(g) = (1/K) × Σ C(τ)` | Mean confidence across proposals |
| **Group Diversity (D)** | `D(g) = \|{answer(τ) : τ ∈ g}\|` | Unique answer cluster count |

### Three-Tier Aggregation

<div class="admonition architecture" markdown>
<p class="admonition-title">Grouped proposals pick an aggregation strategy by consensus vs. diversity</p>

Four proposals group in pairs (Group 1: proposals 1-2, Group 2: proposals
3-4). Each group is scored for group confidence (GC) and diversity (D).
Group 1's example score (GC=0.9, D=1 — high confidence, low diversity)
routes to `MAJORITY_VOTE` (free, no LLM call). Group 2's example score
(GC=0.3, D=3 — low confidence, high diversity) routes to `HEAVY_MODEL`
(expensive, reasoning-tier aggregation).
</div>

| Strategy | When | Cost |
|:---|:---|:---|
| `MAJORITY_VOTE` | High GC + Low D (specialists agree) | **Free** — no LLM call |
| `LIGHT_MODEL` | High GC + Some D (partial agreement) | **Cheap** — fast model synthesis |
| `HEAVY_MODEL` | Low GC + High D (specialists disagree) | **Expensive** — reasoning model |

### Convergence Monitor

The `ConvergenceMonitor` tracks diversity across evolutionary iterations
and signals early stopping when diversity collapses:

```python
from agent_utilities.graph.hierarchical_planner import ConvergenceMonitor

monitor = ConvergenceMonitor(convergence_threshold=0.1, patience=3)

for iteration in evolutionary_loop:
    diversity = compute_diversity(proposals)
    if monitor.update(diversity):
        break  # Diversity collapsed — stop early
```

### Configuration

| Parameter | Default | Description |
|:---|:---|:---|
| `population_size` | `4` | Max proposals to consider (N) |
| `group_size` | `2` | Proposals per fitness group (K) |
| `confidence_threshold` | `0.7` | GC above this → lite aggregation |
| `diversity_threshold` | `2` | D above this + low GC → heavy aggregation |
| `convergence_threshold` | `0.1` | Diversity below this = converged |
| `patience` | `3` | Consecutive low-diversity readings before stopping |

---

## Architecture Integration

<div class="admonition architecture" markdown>
<p class="admonition-title">Confidence Router and Evolutionary Aggregation, fed by the same signals</p>

The Confidence Router reads historical proficiency from the Self-Model,
confidence scores from Workspace Attention, and budget tier from
Homeostatic Downgrade, and picks an adaptive model via the Model Registry.
Workspace Attention's proposals also feed Evolutionary Aggregation
directly, alongside the Cognitive Scheduler's Convergence Monitor (which
the scheduler hosts).
</div>

## Related Concepts

| Concept | Relationship |
|:---|:---|
| [CONCEPT:AU-KG.memory.tiered-memory-caching Self-Model](../overview.md) | Provides historical confidence priors |
| [CONCEPT:AU-ORCH.adapter.hot-cache-invalidation Workspace Attention](../overview.md) | Provides runtime scoring and proposals |
| [CONCEPT:AU-OS.state.cognitive-scheduler-preemption Resource Optimization](../overview.md) | Budget-aware model selection |
| [CONCEPT:AU-OS.state.cognitive-scheduler-preemption Cognitive Scheduler](../overview.md) | Hosts convergence monitor |
| [CONCEPT:AU-OS.state.cognitive-scheduler-preemption Homeostatic Downgrade](../overview.md) | Composes with confidence routing |
