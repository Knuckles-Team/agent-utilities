# Design Document: A narrow AST heuristic on a shrinking ratchet baseline, not a hard gate or dataflow analysis

CONCEPT:AU-ORCH.execution.event-loop-blocking-sweep

> `scripts/check_event_loop_blocking.py` (the gate, primary), with two
> production wiring sites: `agent_utilities/capabilities/checkpointing.py:194`,
> `agent_utilities/mcp/tools/write_ingest_tools.py:1268`.

## The real decision

The gate exists because of a real production incident, stated directly in its
own module docstring: *"graph-os has been liveness-killed in production (exit
137, `/health/ready` 'context deadline exceeded' while the engine logs a
sustained `engine_breaker: slow engine call ... duration=3-8s` stream) — the
signature of synchronous KG/file/network calls executed directly on the
request-serving event loop, starving the health probe."* Three independent
manual sweeps (agent-runner/router, `governed_dynamic_workflow`, and this
script covering the rest of `mcp/tools/` + background loop controllers) each
found blocking sites the others missed — *"the pattern is systemic rather than
a handful of one-off bugs"* — so this is the automated guard that keeps it
from silently regressing (`scripts/check_event_loop_blocking.py:2-11`).

Two design choices, both deliberate trade-offs of thoroughness for
tractability:

**1. A static AST heuristic, not dataflow/type analysis.** The scanner flags a
*known* blocking call shape — a synchronous KG/engine write or read, a
blocking file read/write, `subprocess`, a synchronous HTTP client call, or
`time.sleep` — written as a literal call expression directly inside an
`async def` body, with no thread hop in between. It explicitly does not
attempt dataflow or type analysis, so *"it cannot see a blocking call hidden
behind an arbitrary helper function of an unrecognized name three modules
away"* (`scripts/check_event_loop_blocking.py:19-20`). `mcp/tools/write_ingest_tools.py:1268`
documents the real consequence: `persist_facts` loops issuing synchronous KG
round trips is invisible to the scanner because it is called through a plain
helper, not an `engine.*`-shaped attribute call — tracked as a known blind
spot (`D-W15-6`), not silently accepted as complete coverage.

The scanner exempts calls already isolated via the sanctioned hop —
`run_blocking_ordered`, `asyncio.to_thread`, `run_blocking`/
`invoke_client_method`, `loop.run_in_executor` — including the specific
"nested sync closure" shape used throughout the codebase: define a `def
_do_thing(): ...` inside the `async def`, then `await
run_blocking_ordered(_do_thing)` (`checkpointing.py:191-206` is a live
instance of exactly this shape).

**2. A ratchet baseline, not a hard gate against the whole population.** The
gate is *"RATCHET, not report-only"* — it fails on **new** candidate sites,
not the entire pre-existing population. The reasoning is explicit: *"this
codebase still has a large, untriaged population of matching call shapes...
that nobody has individually verified is either genuinely non-blocking...or
already safe for another reason... Turning this into a hard gate against the
WHOLE population would fail on a mountain of pre-existing, untriaged code
(noisy-always-failing is worse than no gate)"* (`scripts/check_event_loop_blocking.py:28-35`).
The pre-existing population is frozen in `scripts/event_loop_blocking_baseline.txt`;
fixing a baselined site is always allowed and shrinks the baseline on the next
`--update-baseline`.

## The rejected alternative

Both halves of the decision reject the more thorough option in favour of the
tractable one, and the docstring names the cost of each rejected alternative
directly:

- Dataflow/type analysis would catch the `write_ingest_tools.py`-class blind
  spot this gate misses, but at implementation and maintenance cost the
  scanner's author judged not worth paying for a static AST heuristic that
  "mirrors every other `scripts/check_*.py` gate in this repo."
- A hard gate against the whole population would be strictly more protective,
  but would fail from day one on a mountain of pre-existing, individually
  unverified call sites — a gate that is always red trains engineers to
  ignore it, which is worse than a gate that only ever complains about *new*
  regressions. The ratchet, mirroring `check_no_env_sprawl.py`'s pattern,
  makes the gate meaningful (green means "no new regressions") while the
  baseline is worked down over time.

## Risk Assessment

- **Blast Radius**: `scripts/check_event_loop_blocking.py`,
  `scripts/event_loop_blocking_baseline.txt`, and every `async def` in
  `agent_utilities/` the ratchet has not yet baselined.
- **Backward Compatible**: Yes — CI-time gate only, no runtime behaviour
  change.
- **Known weak point**: two compounding blind spots — the AST heuristic
  cannot see through a helper function (`D-W15-6`, demonstrated live in
  `write_ingest_tools.py:1268`), and the ratchet only catches *new* sites, so
  a blocking call hidden behind a helper in *already-baselined* code is
  invisible on both axes simultaneously.
