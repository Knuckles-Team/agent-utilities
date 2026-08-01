# Design Document: Measure how fast a specialization run learns, not only how good the end result is

CONCEPT:AU-AHE.harness.per-task-adaptation-speed

> `agent_utilities/harness/adaptation_speed.py`.

## Decision — instrument time-to-target, sample-complexity, and learning-AUC over best-so-far reward, not just terminal quality

The module docstring (`adaptation_speed.py:4-19`) names the gap directly:
the SAI thesis (arXiv:2602.23643) defines an adaptable agent's primary
metric as "the speed and efficiency with which new skills are acquired under
realistic resource constraints" — explicitly **not** a fixed-competency
checklist. This codebase already measured terminal quality everywhere
(`reliability_scorers`) and cross-cycle repo cadence (`ImprovementVelocity`),
but nothing measured the per-task *learning curve* of a single
specialization run. `AdaptationCurve` is that missing measurement:
`time_to_target` (wall-seconds to first reach reward `tau`),
`sample_complexity` (examples/rollouts consumed to first reach `tau`), and
`learning_auc` (normalized area under best-so-far reward vs. samples).

**The rejected alternative is exactly what existed before: score a
specialization run only on its final quality.** Two runs that reach the same
terminal reward can differ enormously in how fast they got there — one
converging in 10 samples, another in 10,000 — and a terminal-quality-only
metric can't distinguish them. A second, smaller decision inside the same
module: rewards are transformed to **best-so-far** before any threshold test.
**The rejected alternative there is testing the raw, un-smoothed reward
directly against the target** — which would let a single noisy verifier
regression on one candidate make an already-met target register as
"un-met," corrupting `time_to_target` with verifier noise rather than
genuine regression.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/adaptation_speed.py`,
  the SAI factory controller (AHE-3.29) that optimizes against it,
  `agent_utilities/harness/hote_tri_evolution.py` (reuses
  `AdaptationCurve`/`marginal_speed_gain` for co-evolution measurement).
- **Backward Compatible**: Yes — additive instrumentation alongside existing
  terminal-quality scorers, not a replacement for them.
- **Known weak point**: `learning_auc` normalizes area under best-so-far
  reward vs. samples, which rewards early gains disproportionately relative
  to gains near the target — a run that improves slowly-but-steadily and one
  that jumps early then plateaus below target can produce similar AUC values
  despite very different practical usefulness.
