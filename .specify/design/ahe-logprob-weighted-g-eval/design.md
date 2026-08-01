# Design Document: Logprob-weighted continuous judge score, not a discrete 1-5 pick

CONCEPT:AU-AHE.harness.ahe-2 ·
CONCEPT:AU-AHE.harness.ahe-3

> `agent_utilities/harness/g_eval.py` (primary), rendered into the bake-off
> report by `agent_utilities/harness/memorydata/scoreboard.py` (pointer).

## Decision — G-Eval's judge score is a probability-weighted average over candidate score digits, not the single emitted token

`CONCEPT:AU-AHE.harness.ahe-2`

`g_eval.py:4-19` states both mechanics G-Eval (Liu et al., 2023) contributes,
absorbed from Opik and improved here:

1. **Chain-of-thought rubric, generated once and cached.** The judge writes
   explicit evaluation steps from a task description + criteria; that rubric
   is LRU-cached per `(task, criteria, model)` so the CoT generation cost is
   paid once, not on every scoring call.
2. **Logprob-weighted continuous score.** Instead of trusting the single score
   digit the model happens to emit, `GEval` requests top-logprobs on the score
   token and computes a probability-weighted average over the candidate
   digits — turning a discrete 1-5 judgement into a smooth 0..1 value that is
   materially more stable run-to-run.

**The rejected alternative, named directly by contrast, is the naive
approach both G-Eval predecessors and a first-pass implementation would take:
take the model's single emitted score token at face value.** A single-token
score is maximally sensitive to sampling noise right at the decision boundary
(a judge that's 51% "4" and 49% "3" reports a hard 4 either way); the
logprob-weighted average captures that near-boundary uncertainty as a
continuous value instead of discarding it. The implementation degrades
gracefully to the plain point score when the provider returns no logprobs
(`g_eval.py`, rubric-generation failure path), rather than failing the
evaluation outright — a swallowed rubric-generation exception falls back to
criteria-only scoring, not a crash or a silently stale cached rubric.

`rubric_version` is a separate, smaller decision inside the same class
(`CONCEPT:AU-AHE.evaluation.judge-calibration`, already tracked elsewhere): an
explicit version wins when set; otherwise it auto-derives from a content hash
of `(task_introduction, evaluation_criteria)` so an unversioned rubric edit
still surfaces as a different fingerprint rather than drifting silently under
the same identity.

### Pointer — `CONCEPT:AU-AHE.harness.ahe-3`

`scoreboard.py:4-45`. `render_scoreboard` turns a list of `BakeoffResult`
(which includes each cell's G-Eval `judge_score`) into a three-section
markdown report: measured results per `family/task/config`, a per-family
"best config" attribution table, and — **only when a router result is present
in the input set** — a router-vs-best-single comparison section. The
conditional inclusion is the concrete decision: a router-vs-single comparison
section that's rendered unconditionally would show a misleading empty/N/A
row whenever a bake-off run doesn't include the router config, so the
renderer only emits that section when there is actually a router result to
compare against. `MEMORYDATA_BASELINES` is deliberately left as an all-`None`
placeholder table (reserved for a future delta column against the 22 published
MemoryData preset baselines) rather than hand-filled with paper numbers that
would need a second maintenance path to stay accurate.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/g_eval.py`,
  `agent_utilities/harness/memorydata/scoreboard.py`,
  `agent_utilities/harness/memorydata/bakeoff.py`.
- **Backward Compatible**: Yes — both are additive evaluation/reporting
  utilities, not gates on any existing pipeline.
- **Known weak point**: the logprob-weighted average is only as good as the
  provider's returned top-logprobs; a provider that never exposes them (or
  exposes them for the wrong token position) silently degrades every score to
  the plain point value with no visible signal in the report that this
  happened for a given run.
