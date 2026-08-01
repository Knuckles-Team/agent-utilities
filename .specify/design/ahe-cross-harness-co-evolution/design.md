# Design Document: Cross-harness GRPO reuses existing training primitives; promotion gates on a held-out split HarnessX lacks

CONCEPT:AU-AHE.harness.co-evolution-loop ·
CONCEPT:AU-AHE.harness.kg-held-out-certification

> `agent_utilities/harness/co_evolution.py`.

## Decision — realise cross-harness GRPO by composing existing primitives, not a new training pipeline

`CONCEPT:AU-AHE.harness.co-evolution-loop`

The module docstring (`co_evolution.py:1-9`) states the decision directly:
HarnessX co-evolves harness and model over one shared replay buffer via
cross-harness GRPO — grouping trajectories by *task identity across harness
versions* so the model internalises strategies that succeeded under
successive scaffolds. Rather than build a bespoke training loop for this, the
implementation composes four primitives that already existed for other
purposes: `PrioritizedReplayBuffer` (keyed by task, `co_evolution.py:56-59`,
so successive rounds accumulate instead of overwrite), the GRPO
`batch_normalized_advantage(group_ids=...)` (the cross-harness grouping
criterion is literally `group_ids = task`, `co_evolution.py:65-67`),
`SubstrateTrainer`'s `GrpoSample` corpus shape (deferred GPU job — "replay
reuse at no added rollout cost," `co_evolution.py:70-76`), and
`SuperhumanCertifier` for promotion gating.

**The rejected alternative is building HarnessX's training pipeline
fresh** — a new advantage-grouping mechanism and a new corpus format
specific to cross-harness co-evolution. That would duplicate machinery this
codebase already has for ordinary GRPO training, and would forfeit "replay
reuse at no added rollout cost," since a separate pipeline would need its own
rollout collection rather than reusing the shared prioritized buffer.

### Pointer — `CONCEPT:AU-AHE.harness.kg-held-out-certification`

`co_evolution.py:78-81`. `certify_promotion` gates variant promotion on a
**held-out split**: `held_out_rewards` are scored via `SuperhumanCertifier`,
certified only if the bootstrap confidence-interval lower bound clears the
baseline. The docstring names the gap this closes explicitly: this is "the
held-out evaluation HarnessX lacks." **The rejected alternative is HarnessX's
own approach** — evaluating promotion against the same distribution the
model trained on, with no held-out check — which risks certifying a variant
that overfit the training replay buffer rather than one that generalises.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/co_evolution.py`,
  `agent_utilities/graph/training_signals.py` (`batch_normalized_advantage`),
  `agent_utilities/harness/replay_buffer.py`,
  `agent_utilities/harness/substrate_trainer.py`,
  `agent_utilities/harness/superhuman_gate.py`.
- **Backward Compatible**: Yes — `CrossHarnessCoEvolution` is a new
  composition over existing primitives; none of them changed their own
  public contract to support it.
- **Known weak point**: the cross-harness grouping criterion is exactly
  `group_ids = task` — trajectories for a task ID that means something
  different across harness versions (a renamed or redefined benchmark task)
  would be silently grouped together as if they were the same underlying
  challenge.
