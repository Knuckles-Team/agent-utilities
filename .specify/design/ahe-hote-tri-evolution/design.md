# Design Document: Co-evolve proposer/solver/judge together with interdependent rewards, not one at a time

CONCEPT:AU-AHE.harness.co-evolve-research

> `agent_utilities/harness/hote_tri_evolution.py`, invoked opt-in from
> `agent_utilities/knowledge_graph/research/loop_controller.py:578-594`.

## Decision — HOTE's joint reward coupling replaces sequential one-at-a-time evolution for the three deep-research modules

The module docstring (`hote_tri_evolution.py:4-19`) states the prior state and
the change directly. The ecosystem already owned the three deep-research
modules as separate pieces before this: a proposer
(`OntologyReasoningDriver.extrapolate`), a solver (`ResearchPipelineRunner` +
ARA artifacts), and a judge (`ConceptMatcher` LLM-judge) — and it evolved them
**one at a time**, via `SaiFactoryController`/`EvolveAgent`. Distilling *Hybrid
Open-Ended Tri-Evolution Makes Better Deep Researcher* (HOTE, arXiv:2606.13710),
this module's contribution is co-evolving all three **together** with
interdependent rewards:

- **Solver** improves only from *frontier* tasks (max learning signal at an
  intermediate success rate), and only as fast as the *judge* is calibrated —
  its reward is the judge's score.
- **Proposer** is rewarded for keeping the solver near a productive-struggle
  band, so it must track the solver's *rising* skill — a frozen proposer
  makes tasks trivial as the solver improves, collapsing the learning signal.
- **Judge** is rewarded for calibration against a verifier; a miscalibrated
  judge feeds the solver a biased reward and slows it.

**The rejected alternative is exactly what preceded this module: evolve each
of the three independently.** The docstring states the paper's claim this
distills — that joint co-evolution is *indispensable* — and makes it
falsifiable rather than assumed: `run_ablation` runs an analytic, fully
deterministic version of the coupling in a CPU unit test, comparing joint
evolution against freezing any one module solo. Freezing one module and
letting the others run demonstrates the stall the paper predicts. Every
module is also injectable so the real `OntologyReasoningDriver`/ARA/
`ConceptMatcher` can drive the same controller in production instead of the
deterministic ablation stand-ins.

Reuses `AdaptationCurve` (`CONCEPT:AU-AHE.harness.per-task-adaptation-speed`)
and `marginal_speed_gain` so this co-evolution is measured with the same
adaptation-speed instrument SAI specialization already uses — a second
instance of composing an existing measurement tool rather than inventing a
tri-evolution-specific one.

Wired into the research loop as **opt-in** (`tri_evolution` flag,
`loop_controller.py:578-583`): off by default so the ordinary zero-infra cycle
stays cheap; the CPU ablation harness runs without any LLM calls, while the
LLM-backed integration of the real proposer/solver/judge is the production
path once enabled.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/hote_tri_evolution.py`,
  `agent_utilities/knowledge_graph/research/loop_controller.py`.
- **Backward Compatible**: Yes — opt-in via the `tri_evolution` loop-cycle
  flag; the CPU ablation path has no external dependencies.
- **Known weak point**: the indispensability verdict is only as trustworthy
  as the deterministic ablation's fidelity to the real proposer/solver/judge
  dynamics — a production run using the real LLM-backed modules could in
  principle behave differently from what the CPU-only ablation predicts,
  since the ablation's coupling is analytic, not learned.
