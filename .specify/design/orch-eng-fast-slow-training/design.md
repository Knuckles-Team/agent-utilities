# Design Document: Two coupled learning loops, not one — harness updates now, weights only for what recurs

CONCEPT:AU-ORCH.execution.feed-cycle-outcome-fast

> `agent_utilities/harness/fast_slow_controller.py` (`FastSlowController`,
> primary), wired in `agent_utilities/harness/agentic_evolution_engine.py`.
> Distilled from "Owning Your Token Capital / Enterprise AI Learning Loop"
> (Fast-Slow Training, arXiv:2605.12484).

## The real decision

`FastSlowController` (`harness/fast_slow_controller.py:129-`) runs **two
coupled learning loops over the same stream of production traces**, not one:

> *"The **FAST loop** updates the *harness* — the prompts, scaffolding and tool
> wiring — for what the task in front of the agent needs *right now*. It is
> cheap, immediate, and (critically) **model-swap-safe**: because it lives in
> the harness rather than in any single model's weights, the learning survives
> swapping the frontier model the controller still calls."*
>
> *"The **SLOW loop** absorbs what *recurs* across the organization's work into
> an *owned* model's weights. Only task kinds seen often enough to be worth the
> capital of a weight update are promoted; the owned model then compounds
> alongside the frontier models it keeps calling."*
> (`fast_slow_controller.py:9-20`)

The thesis, named directly in the docstring, is **"own your token capital"**:
every production trace is a learning opportunity, but not every trace is worth
the same kind of investment. The wiring in
`harness/agentic_evolution_engine.py:435-449` (`_fast_slow_stage`) shows the
mechanics: each cycle observes a `Trace` keyed by `base_id` (so recurrence
across bases is detectable), runs `fast_step()` unconditionally, then
`slow_step()` — which only fires for task keys that have actually recurred
often enough (`recurrence_threshold`).

## The rejected alternative — a single loop, in either direction

Two single-loop designs are implicitly rejected by running both, and the
docstring's own framing makes each one's failure mode explicit:

**All-fast, no weight absorption.** Every trace only ever updates prompts and
scaffolding. This is cheap and safe across model swaps, but nothing ever
*compounds* — the same class of task that recurs a thousand times gets a
thousand independent prompt-level nudges instead of ever being folded into
owned weights that would make the base model itself better at it. The
"capital" in "own your token capital" is never invested; it is spent every
time.

**All-slow, weight-train on every trace.** The opposite failure: training on
every production trace regardless of recurrence is expensive, and — the
docstring's specific concern — **not model-swap-safe**. Learning baked
directly into one model's weights does not survive swapping the frontier model
the controller calls; a one-off trace that will never recur gets the same
capital-intensive treatment as a task pattern seen a thousand times.

The chosen design accepts the complexity of running two loops specifically to
avoid both failure modes: cheap/safe/immediate for the general case (fast),
capital investment gated on demonstrated recurrence (slow) — and
`swap_model()` (`fast_slow_controller.py`) records a frontier-model swap
**without discarding any accumulated learning**, which only works because the
fast-loop's learning was never tied to the swapped model's weights in the
first place.

## What is deliberately deferred, and why that is not a gap in this decision

The actual weight trainer is **deferred by design**: the controller accepts an
injected `trainer_fn` that defaults to a no-op
(`fast_slow_controller.py:26-34`), so *"the full control flow and the GRPO
advantage computation are exercised and testable today, while no real
training is implemented here."* This follows the same AHE-3.x convention
named in the module docstring: *"controller + its data spine together, trainer
micro-mechanics specified-not-implemented."* The slow loop is a live consumer
of `agent_utilities.graph.training_signals.batch_normalized_advantage` — the
GRPO advantage computation genuinely runs — but the actual gradient step is
injected, not stubbed out of the decision; `SubstrateTrainer`
(`harness/substrate_trainer.py`, `CONCEPT:AU-ORCH.execution.substrate-training-job-emission`,
not in this batch) is the real trainer adapter that gets injected in
production wiring.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/fast_slow_controller.py`,
  `agent_utilities/harness/agentic_evolution_engine.py`.
- **Backward Compatible**: Yes — `trainer_fn` defaults to a no-op, so the
  control flow is exercised without requiring a real substrate connection.
- **Known weak point**: with the trainer deferred to an injected no-op by
  default, `slow_step()`'s effect on production behaviour is entirely a
  function of what the caller wires in — a misconfigured or absent
  `trainer_fn` silently produces a fast-only system with none of the
  compounding the SLOW loop exists to provide, and nothing in this module
  surfaces that degradation.
