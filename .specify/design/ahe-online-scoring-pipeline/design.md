# Design Document: One judge path for production monitoring AND regression assertions, off the hot path

CONCEPT:AU-AHE.harness.receives-trace-id-must ·
CONCEPT:AU-AHE.harness.instead-context-stuffing-small ·
CONCEPT:AU-AHE.harness.onlinescorenode

> `agent_utilities/harness/online_scoring.py` (primary),
> `agent_utilities/harness/tool_judge.py` (pointer).

## Decision — production automation-rule scoring and offline regression assertions run through the SAME judge, deferred off the traced call's hot path

`CONCEPT:AU-AHE.harness.receives-trace-id-must`

The module docstring (`online_scoring.py:4-19`) states the architecture:
when a root trace completes, `KGTraceBackend` fires a fast hook that defers
scoring off the hot path; a sampler then judges the trace against registered
automation rules (`EvalRunner._assertion_judge` — "the SAME code" used for
offline eval) and against matching regression assertions, links every
verdict `SCORED_BY` the trace, and on a FAILED assertion feeds it back into
the eval corpus so the same break is caught from now on.

**The rejected alternative is two separate judge implementations** — one for
production monitoring, one for offline regression testing — which is the
obvious shape given they run at different times against different trigger
sources. Sharing one judge path instead means a change to judging logic
(prompt, rubric, model) can't silently diverge between what production
sees and what CI sees. The judge running on a small thread pool off the hot
path is a second, independent decision within the same concept: **the
rejected alternative there is judging inline before the traced call
returns**, which would make every traced agent run pay scoring latency
regardless of whether anyone is watching the score in real time.

### Pointer — `CONCEPT:AU-AHE.harness.instead-context-stuffing-small`

`tool_judge.py:4-13`, `online_scoring.py:222-227`. A multi-MB agent trace
blows the judge's context window if stuffed in whole. Absorbed from Opik's
agentic judge, the tool-judge instead gives the judge TOOLS to navigate the
span subgraph on demand — list spans, drill into one, read the final I/O —
rather than inlining the full trace. **The rejected alternative, named in the
concept id itself, is context-stuffing**: serializing the entire trace into
the judge's prompt. `should_use` selects automatically per trace (size
threshold `TOOL_JUDGE_THRESHOLD=8000` chars or `TOOL_JUDGE_MAX_SPANS=12`), so
small traces still use the cheap inline judge and only pay the tool-loop cost
when the trace is actually too large to inline.

### Pointer — `CONCEPT:AU-AHE.harness.onlinescorenode`

`online_scoring.py:60-73,249-259,75-121`. Beyond LLM-judged automation
rules, a `Metric` lets an operator register **user-defined Python** run on
every sampled trace. **The rejected alternative is executing that
user-supplied Python inline in the scoring process** — instead
`_run_metric_isolated` routes it through `SandboxRouter`/`default_sandboxes`,
approved isolated RLM backends only, with a 2 MiB trace-view size cap, a
single allowed output channel (`FINAL_VAR('__metric_score__', ...)`), and a
hard `SandboxFatalError` if no approved backend accepts the code — it fails
closed rather than falling back to unsandboxed execution.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/online_scoring.py`,
  `agent_utilities/harness/tool_judge.py`,
  `agent_utilities/harness/continuous_evaluation_engine.py` (`EvalRunner`),
  `agent_utilities/rlm/sandboxes/*`.
- **Backward Compatible**: Yes — additive scoring pipeline; no existing trace
  path is required to opt in.
- **Known weak point**: `should_use`'s size/span thresholds are static
  constants, not adaptive to the judge model's actual context window — a
  future smaller-context judge model swapped in without revisiting
  `TOOL_JUDGE_THRESHOLD` could still get context-stuffed on a trace just
  under the current threshold.
