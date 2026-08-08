# Design Document: A specialization task is defined by a real, comparable reward — never a pass/fail bit

CONCEPT:AU-AHE.harness.sai-task

> `agent_utilities/harness/sai_task.py`.

## Decision — `Verifier` is the one Protocol every specialization track shares, and it must return a real reward, not a boolean

The module docstring (`sai_task.py:4-19`) states the contract directly: a
`SpecializationTask` bundles a corpus of inputs, a `Verifier` that scores any
candidate with a real reward, a target `tau` defining "good enough," and an
optional `human_baseline` the certifier compares against for a superhuman
verdict. The `Verifier` Protocol is deliberately the single seam every
specialization track shares — a GPU-kernel verifier (compile + correctness +
measured speedup), a classification-accuracy verifier, or a
world-model next-state-prediction verifier all satisfy the same contract, so
the controller and the adaptation-speed metric stay task-agnostic.

**The rejected alternative is a pass/fail bit** — the obvious, simpler
verifier contract, and the one many benchmark harnesses actually use. It's
explicitly rejected here because a boolean can't serve double duty as this
system needs it to: "the reward *is* the training signal the weight arm
distills... and the curve the controller optimizes." A pass/fail signal
gives the weight arm nothing to distill gradations from, and gives the
adaptation-speed curve nothing to plot except a step function — it couldn't
answer "how close did this get" or "is this run improving," only "did it
pass this time," which is exactly the terminal-quality-only view
`per-task-adaptation-speed` was built to move past.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/sai_task.py`,
  `agent_utilities/knowledge_graph/research/sai_factory.py` (the controller
  that consumes this contract), `agent_utilities/harness/adaptation_speed.py`.
- **Backward Compatible**: Yes — a Protocol definition; any verifier
  implementation satisfying the interface can plug in.
- **Known weak point**: nothing in the Protocol itself constrains a
  `Verifier` implementation's reward to be well-calibrated or stable across
  runs — a verifier that returns noisy or drifting rewards for the same
  candidate would corrupt both the training signal and the adaptation-speed
  curve without violating the contract.
