# Design Document: The durable asset AHE optimizes against is the eval set, not the prompt — so every production failure becomes a new eval case

CONCEPT:AU-ORCH.execution.eval-set-optimization-compounding

> `agent_utilities/rlm/eval_set_optimizer.py` (the eval-set data model + optimizer) and
> `agent_utilities/harness/continuous_evaluation_engine.py` (where each distillation round
> harvests its failures into that set).

## Decision — harvest every production failure into a growing `EvalSet`; optimize the evals, not the prompts

`agent_utilities/rlm/eval_set_optimizer.py:4-17` names its source directly: "GEPA enterprise
learning loop — 'Owning Your Token Capital / Enterprise AI Learning Loop' (GEPA,
arXiv:2507.19457). The article's compounding-IP thesis is that the durable asset is not the
prompt but the EVAL SET: start from a small seed of expert annotations, then make every
production failure a new eval case and re-optimize the harness against the ever-growing eval
set." The module docstring is explicit about scope: "This module optimizes the EVAL SET (the
source of truth), not prompts or programs. Program optimization is owned exclusively by the
native epistemic-graph optimizer; the trace -> eval loop here only sharpens the evaluation
bar" (`eval_set_optimizer.py:15-17`).

The mechanism lives in `TraceDistiller` (`agent_utilities/harness/continuous_evaluation_engine.py`):
an `EvalSet` is constructed once per distiller instance (`continuous_evaluation_engine.py:104-109`,
`self.eval_set = EvalSet()`, commented "the compounding eval set: each round's failures are
harvested as new eval cases, so the org's eval suite (its real IP) grows with every production
failure (GEPA enterprise learning loop)"). Then, at the end of every distillation round's
triage step, every failing entry is turned into an `EvalCase` and added to that set
(`continuous_evaluation_engine.py:235-249`):

```python
# ORCH-1.55 — harvest each failure as a new eval case so the eval set (the
# compounding IP) grows every round; the next harness optimization is scored
# against this ever-growing suite.
for entry in corpus.entries:
    if not entry.pass_fail:
        self.eval_set.add(EvalCase(
            case_id=f"{corpus.round_id}:{entry.task_id}",
            input=entry.content or entry.task_id,
            expected=entry.root_cause or "(must pass)",
            source="production_failure",
        ))
```

Each `EvalCase.source` is tagged `"seed"` (expert annotation), `"production_failure"`
(harvested from a live miss), or `"refined"` (a sharpened case) — provenance the eval set
keeps rather than discards.

**The rejected alternative**, named by the module's own scoping statement, is optimizing the
*prompt* (or program) directly against a fixed, hand-authored eval set — the more common
pattern where the eval set is a static artifact and iteration happens on the model-facing
side. That loses on the article's own thesis this module cites: a static eval set caps how
much the harness can ever be shown to have improved, because it never grows to cover the
failure modes production actually surfaces. The design instead keeps program/prompt
optimization as a *separate, excluded* concern (explicitly deferred to "the native
epistemic-graph optimizer") and makes this module's only job widening the evaluation bar
itself — `case_id` is a stable identity `f"{round_id}:{task_id}"` specifically so the set can
be deduplicated across rounds rather than re-harvesting the same failure repeatedly
(`eval_set_optimizer.py:46-49`, `EvalCase.case_id` docstring: "Stable identity used for dedup
across rounds").

This decision is documented separately from — not folded under — `rlm-experience-observability`
(the RLM-vs-keyword failure-clustering choice in the same file): that concept is about *how*
AHE clusters failures when there are many of them, using RLM specifically; this one is about
*what happens to every failure regardless of cluster*, and never invokes RLM at all — `EvalSet.add`
is a plain dataclass append, dependency-injected and LLM-free by explicit design constraint
(`eval_set_optimizer.py:19-24`: "No `os.environ`, no stubs... Fully dependency-INJECTED... so
the whole module is unit-testable with NO LLM").

## Risk Assessment

- **Blast Radius**: `agent_utilities/rlm/eval_set_optimizer.py`,
  `agent_utilities/harness/continuous_evaluation_engine.py`, and
  `docs/architecture/multi_source_assimilation.md:270` (which documents this loop for the
  wider assimilation-pipeline audience).
- **Backward Compatible**: Yes — additive; distillation rounds worked before this without an
  eval set, they just didn't compound.
- **Known weak point**: the harvested `expected` value falls back to the literal string
  `"(must pass)"` when `entry.root_cause` is absent (`continuous_evaluation_engine.py:246`) —
  a case with no root cause captured becomes a weak, non-actionable eval anchor rather than
  being excluded from the set.
