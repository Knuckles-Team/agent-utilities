# AGI→ASI Implementation Guide

> What was built from the "From AGI to ASI" gap analysis (arXiv:2606.12683), where it
> lives, and how to use it. The pieces compose into one loop — see
> **[Self-Improving Reasoning Substrate](../architecture/self_improving_reasoning_substrate.md)**
> for the architecture. The gap analysis itself is in
> `reports/agi-to-asi-gap-analysis-2026-06-13.md`.

## What it is

The paper frames the AGI→ASI transition as four pathways — scaling, paradigm shifts,
recursive self-improvement, multi-agent collectives — gated by frictions, with the
recurring rule: *for every friction, build a countermeasure or a way to measure whether
it binds.* These concepts make AU **measure its own dynamics, reason in plural paradigms,
bound cost, and stay corrigible** — the substrate for moving along that continuum.

## Concept reference

| Concept | Module | What it does | How to use |
|---|---|---|---|
| **KG-2.68** | `knowledge_graph/core/reasoner.py` | Outcome-learning paradigm router (keystone) | `kg.reason(ReasoningTask(goal, tags, payload))` |
| **KG-2.69** | `harness/program_synthesis.py` | Inductive program synthesis + MDL/Occam prior | `synthesize(primitives, examples)`; `select_top_k(..., method="mdl")` |
| **KG-2.67** | `knowledge_graph/core/world_model.py` | Action-conditioned world model | `WorldModel().observe(...); wm.rollout(start, policy, horizon)` |
| **AHE-3.22** | `knowledge_graph/research/code_synthesis.py` | Autonomous single-file code generation in the evolution loop | `governed_publish(...)` (default-on, sandbox + ActionPolicy gated) |
| **AHE-3.23 / 3.24** | `knowledge_graph/research/capability_ratchet.py` | Verified apply→verify→rollback + monotone capability ratchet | runs post-publish in `governed_publish`; consulted by promotion governance |
| **AHE-3.26 / SAFE-1.3** | `knowledge_graph/research/improvement_ledger.py` | RSI velocity ledger (improving/stalling) | `ImprovementLedger(engine).summarize()`; auto-recorded each golden-loop cycle |
| **SAFE-1.1** | `harness/frontier_scorers.py` | Non-saturating progress: compression, Elo, saturation detector | `CompressionScorer` (in reliability suite); `saturation_detector(pass_rates)` |
| **SAFE-1.5** | `core/corrigibility.py` | Corrigibility + irreversibility aversion + knowledge-seeking reward | wired into the goal loop; `ACTION_IRREVERSIBILITY_AVERSION=1` for the policy gate |
| **OS-5.35** | `orchestration/cost_governor.py` | Throughput-per-dollar scale-up cap | `FLEET_SCALE_BUDGET_USD_PER_HOUR` (opt-in; unset = unchanged) |

## Using the reasoning router

```python
from agent_utilities.knowledge_graph.facade import KnowledgeGraph
from agent_utilities.knowledge_graph.core.reasoner import ReasoningTask

kg = KnowledgeGraph()

# Inductive: learn the shortest program fitting examples (routes to KG-2.69)
kg.reason(ReasoningTask(
    goal="learn the mapping", tags=("induction",),
    payload={"primitives": {"double": lambda x: x*2}, "examples": [(1, 2), (3, 6)]},
))

# Symbolic: forward-chain to a goal fact (routes to the deductive paradigm)
kg.reason(ReasoningTask(
    goal="derive C", tags=("deduction",),
    payload={"facts": ["A"], "rules": [(("A",), "B"), (("B",), "C")], "goal_fact": "C"},
))
```

Each call routes to the paradigm whose learned reward EMA + capability tags best fit, runs
it, and feeds the result's score back — so the router self-improves. Register a new
paradigm with `get_reasoner_router().register(my_reasoner)`; it needs only a `name`,
`capability_tags`, and `reason(task) -> ReasoningResult(score)`.

## Safety & cost envelope

- **Corrigibility (SAFE-1.5):** autonomous goal loops checkpoint and yield to a supervisor
  shutdown signal without resisting; set `ACTION_IRREVERSIBILITY_AVERSION=1` to route
  irreversible actions (delete/destroy/merge/deploy) to a human even under an auto tier.
- **Cost (OS-5.35):** set `FLEET_SCALE_BUDGET_USD_PER_HOUR` to cap autoscaler scale-ups at a
  throughput-per-dollar budget; every scaling action carries an `est_cost_usd_per_hour`.
- **Capability ratchet (AHE-3.24):** a published evolution branch that measurably regresses
  capability is abandoned; the verdict feeds promotion governance.

## Measuring the loop

`ImprovementLedger(engine).summarize()` reads the loop's own audit streams (EvolutionCycle,
ProposalPublication, CapabilityRatchetResult) into a velocity reading — cycle cadence, the
genotypic-vs-prose mechanism split, capability pass-rate, and an improving/stalling verdict
whose signals flag the paper's "research-gets-harder" mode. It is recorded automatically each
golden-loop cycle as an `ImprovementVelocity` node.

## Status & roadmap

Implemented + merged (local `main`): KG-2.67/2.68/2.69, AHE-3.22/3.23/3.24/3.26, SAFE-1.1/1.3/1.5,
OS-5.35. Extending the same loop next: **OS-5.34/AHE-3.25** (distil winning reasoning traces into
training data, model-collapse-guarded by **SAFE-1.4**) and **ORCH-1.46/47/48** (the router at
population scale — market allocation, emergent specialists, hierarchical coordination). Tracked
in `reports/agi-to-asi-gap-analysis-2026-06-13.md`.
