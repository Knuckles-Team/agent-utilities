# Design Document: The SLOW-loop trainer builds the corpus and emits a job spec; it never runs gradient descent itself

CONCEPT:AU-ORCH.execution.substrate-training-job-emission

> `agent_utilities/harness/substrate_trainer.py` (the decision, `SubstrateTrainer`)
> and `agent_utilities/harness/agentic_evolution_engine.py` (the wiring —
> `FastSlowController(..., trainer_fn=self._substrate_trainer.as_trainer_fn())`).
> Factoring documented in
> `docs/architecture/in_house_training_substrate.md` and
> `docs/architecture/multi_source_assimilation.md`. Tested in
> `tests/unit/harness/test_substrate_trainer.py`.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-ORCH.execution.feed-cycle-outcome-fast` | `FastSlowController` — the FAST loop this trainer serves as the SLOW-loop `trainer_fn` for; the caller, not this decision | 0.55 | ORCH |
| `AU-AHE.harness.when-task-is-scope` | `SelfGuidedSelfPlay`, an unrelated sibling subsystem in the same evolution-cycle wiring block | 0.20 | AHE |

### Extension Analysis

- **Primary Extension Point**: `SubstrateTrainer.train` /
  `SubstrateTrainer.as_trainer_fn` (`substrate_trainer.py`), and the injected
  `dispatch_fn` it calls.
- **Extension Strategy**: augment — a real GPU substrate is wired in by
  passing a real `dispatch_fn`; the default stays record-only.
- **New Concept Required?**: No.

## Decision — agent-utilities owns the reward spine and corpus building; gradient descent is dispatched to an external, GPU-gated substrate, never run in-process

`agent_utilities/harness/substrate_trainer.py:1-30`

The module docstring states the factoring directly: "agent-utilities owns the
*reward spine*, *corpus building*, and *training-job dispatch*; the actual
gradient trainers (torch / PEFT, GRPO / DPO / SFT) live in data-science-mcp
(DSM) and run on GPU. So this class does NOT run gradient descent — that
belongs to the substrate and is GPU-gated."

Concretely, `SubstrateTrainer`:

1. Turns a recurring trace group into a GRPO corpus of
   group-normalized-advantage samples, reusing the canonical
   `batch_normalized_advantage` (the reward-spine responsibility, kept in
   agent-utilities because it is cheap, CPU-only, and needs no GPU).
2. Assembles a `TrainingJobSpec` and emits it to the gradient substrate via an
   **injected** `dispatch_fn: Callable[[TrainingJobSpec], bool]`
   (`substrate_trainer.py:109`) — dependency-injected specifically so "the
   whole flow is testable with no DSM and no GPU."

**The rejected alternative is named explicitly in the docstring: a no-op
default sitting where the real trainer belongs.** Before this class, the
`FastSlowController`'s slow loop had "a no-op default would otherwise sit" —
i.e., the choice was between (a) leaving the slow loop's `trainer_fn` an
inert stub until a real GPU-coupled trainer is built directly inside the
harness process, or (b) factoring the CPU-cheap, always-available half
(corpus building from traces) into agent-utilities now and dispatching the
GPU-bound half out to DSM through an injected boundary. Running gradient
descent in-process was rejected because it would couple agent-utilities'
harness process to GPU availability and torch/PEFT dependencies it does not
otherwise need, and would make the slow loop untestable without a live GPU.

**Degradation is explicit and lossless, not silent.** `dispatch_fn` defaults
to record-only: no substrate is called, and the job is queued locally with
`status="recorded"` so it is never lost when DSM/GPU is unreachable (e.g. a
hardware fault). A real `dispatch_fn` returning `True` marks the job
`"dispatched"`; one that raises (substrate unreachable) is caught and the job
is marked `"skipped_no_substrate"` rather than crashing the slow loop
(`substrate_trainer.py:23-28`, `108-111`).

### Pointer — wiring site: `agent_utilities/harness/agentic_evolution_engine.py:146-166`

The evolution engine's `try`/`except` wiring block constructs
`SubstrateTrainer()` and passes `self._substrate_trainer.as_trainer_fn()` as
the `FastSlowController`'s `trainer_fn`, alongside the comment naming the
paper this implements (Fast-Slow Training, arXiv:2605.12484) and restating
the same split: "It builds a GRPO corpus from the recurring group and emits a
training-job spec to the gradient substrate (DSM/GPU); the gradient step runs
in data-science-mcp and is GPU-gated; jobs are recorded (queued) when no
substrate is reachable, never lost." Both `self._fast_slow` and
`self._substrate_trainer` are reset to `None` together on any construction
failure — the same lazy-init-once degradation pattern used elsewhere in the
same constructor, so no half-constructed pair is left dangling. This is the
same one decision viewed from its call site, not a second decision.

## Data Flow

1. **ORCH**: `FastSlowController.slow_step()` finds recurring `task_key`s and
   calls the injected `trainer_fn` — which is `SubstrateTrainer.as_trainer_fn()`.
2. **KG**: none directly.
3. **AHE**: this *is* the AHE golden-loop's gradient-update boundary — the
   point where accumulated trace evidence becomes a weight-update job.
4. **ECO**: the actual gradient step is dispatched to data-science-mcp (DSM),
   an external fleet server, over the injected `dispatch_fn`.
5. **OS**: none directly — GPU-gating is DSM's concern, not agent-utilities'.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/substrate_trainer.py`,
  `agent_utilities/harness/agentic_evolution_engine.py`,
  `agent_utilities/harness/fast_slow_controller.py`.
- **Backward Compatible**: Yes — the record-only default means the trainer
  is safe to wire in with no DSM/GPU present at all; nothing regresses when
  the substrate is absent, jobs simply accumulate as `"recorded"`.
- **Breaking Changes**: None.
- **Known weak point**: a `"recorded"` job with no substrate reachable
  accumulates locally with no automatic re-dispatch path described at the
  read sites — a human/agent presumably has to notice the backlog and wire a
  real `dispatch_fn` (or re-run against one) to drain it.
