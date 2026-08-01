# Design Document: The execution shape is planned per-job by an escalating cascade, not fixed per-entrypoint

CONCEPT:AU-ORCH.execution.dynamic-execution-profile ·
CONCEPT:AU-ORCH.execution.residual-ambiguous ·
CONCEPT:AU-ORCH.execution.execution-profile ·
CONCEPT:AU-ORCH.execution.focused-tools-altitude ·
CONCEPT:AU-ORCH.execution.direct-completion-shape ·
CONCEPT:AU-ORCH.execution.per-job-shape-construction ·
CONCEPT:AU-ORCH.execution.shape-policy-learning ·
CONCEPT:AU-ORCH.execution.planner-failure-feedback

> `agent_utilities/orchestration/execution_profile.py` (the whole module —
> `ExecutionProfile`, `plan_execution_shape`, `_plan_base_shape`,
> `_apply_shape_policy`), `agent_utilities/graph/routing/strategies/fast_path.py`
> (`orchestration_signal_strength`, `needs_full_orchestration`), and the one
> call site that constructs it, `agent_utilities/orchestration/agent_runner.py:836`.

## Decision — `plan_execution_shape` is an escalating cascade (cache → free structural signal → free KG lexical gate → paid KG search → LLM planning) that builds ONE dynamic `ExecutionProfile` per job, replacing a fixed per-entrypoint preset

`CONCEPT:AU-ORCH.execution.dynamic-execution-profile`

`execution_profile.py:64-67` states the change directly: "the profile is no
longer a fixed per-entrypoint *preset*; it is the **dynamically-constructed
execution shape** for ONE job." Before this, an `ExecutionProfile` was
selected once per entrypoint class (`"chat"` vs `"task"`) and used unchanged
for every job that entrypoint handled — a trivial "hi" and a genuine
multi-step task from the same messaging channel got the identical shape.
`_plan_base_shape` (`execution_profile.py:405-490`) replaces that with a
**graded escalating planner** — "a classifier for the classifier" — where
each stage costs strictly more than the last and only runs when the cheaper
stage wasn't confident:

- **Stage 0 — recipe cache**: an identical job (normalized word-set +
  entrypoint altitude, hashed) reuses its previously-planned shape, skipping
  all resolution.
- **Stage 1 — free structural signals** (`orchestration_signal_strength`):
  slash-command, over-length, or multi-clause — purely mechanical, no KG
  round-trip.
- **Stage 1.5 — free ontology lexical gate**: does the turn literally name a
  registered fleet capability (aho-corasick match against KG capability
  nodes, ~µs)?
- **Stage 2 — paid, Rust-routed semantic search** for the residual ambiguous
  middle: a substantial turn that named no capability lexically (a
  paraphrase like "get my containers running again").
- **Stage 3 — LLM planning** (documented as planned) for genuinely
  complex/uncertain jobs.

**The rejected alternative is the fixed-preset status quo it replaces**: pick
`"chat"` or `"task"` once per entrypoint and run every job through the same
node timeouts, the same discovery/verifier/agent-resolution steps, the same
reasoning toggle. That couldn't distinguish a one-word greeting from a
multi-step delegation arriving through the same channel — both paid the same
apparatus. The escalating design's own justification is cost-shaped: a
trivial turn pays only the free stages (cache check + a microsecond
structural check), while a genuinely ambiguous job earns the KG search or LLM
planning it actually needs — cost is proportional to how hard the job is to
classify, not a flat per-entrypoint tax.

### Pointer — `CONCEPT:AU-ORCH.execution.residual-ambiguous`

`execution_profile.py:325-338` (`_resolve_job_capabilities`, `_refine_with_kg`)
and `fast_path.py:93-110` (`orchestration_signal_strength`). This is
specifically Stages 2-3 of the cascade above — the graded structural signal
(`0` / `2+`) that decides whether a turn is confidently trivial, confidently
full, or lands in the "residual ambiguous middle" that must pay for the
Rust-routed `search_hybrid` disambiguation (~4.5s including query embedding,
vs. >70s for a cold per-process Python HNSW build — the comment names the
~15× speedup as the reason this stage is affordable at all). A search that
finds real capability hits stays full-graph; one that succeeds but finds
nothing relevant downgrades the turn to lean — "the borderline turn is
conversational after all."

### Pointer — `CONCEPT:AU-ORCH.execution.execution-profile`

`fast_path.py:1-45` and `execution_profile.py:340-357` (`_names_capability`).
The specific, narrower decision to DELETE the module's old hardcoded
`_ESCALATION_KEYWORDS` word list (deploy/restart/list/…) and replace it with
a live-KG lexical gate. The rejected alternative — the list itself — is named
explicitly as broken on both sides: "it both missed real capabilities (no
read verbs) and could not name the fleet" (a new fleet server like
`portainer-mcp` had no way to get added to a frozen word list without a code
change). The replacement queries `engine.match_ontology_terms` against live
KG capability nodes, so domain vocabulary grows with the fleet automatically.
`fast_path.py` itself is left "PURELY STRUCTURAL" post-deletion — it now only
answers "does this LOOK complex," never "does this name a capability."

### Pointer — `CONCEPT:AU-ORCH.execution.focused-tools-altitude`

`execution_profile.py:114-125, 456-473`. A THIRD altitude between lean and
full: when the lexical gate names concrete fleet server(s), the job binds
`tool_servers` and gets ONE direct agent loop calling exactly those servers'
tools in parallel — no planner, no discovery, no agent resolution, no
verifier, no expert fan-out. This takes precedence over the structural signal
check: a multi-clause, over-length turn like "fetch my github issues AND
list my portainer stacks" would score `strength>=2` (full-graph-worthy by
structure alone) but is precisely the parallel-tool case the comment says
"over-decomposes" if sent to the full planning graph instead. `run_agent`
falls through to the full graph if the direct loop fails, so a genuine
multi-step workflow that happened to name a tool still degrades safely
rather than getting stuck in a too-narrow toolset.

### Pointer — `CONCEPT:AU-ORCH.execution.direct-completion-shape`

`execution_profile.py:70-114` (the `ExecutionProfile` dataclass fields
themselves: `direct_complete`, `skip_usage_guard`, `run_discovery`,
`run_verifier`, `resolve_agent`, `enable_reasoning`). These are the actual
LEVERS the planner sets — each graph node reads its own relevant field and
either does its work or passes through for this specific job. All default to
the prior full-graph behavior, so an unplanned/legacy construction of
`ExecutionProfile` is unchanged; the planner is what opts a job INTO the lean
shape (`_LEAN_FIELDS`, `execution_profile.py:230-238`), never the other way.
This is the mechanism the escalating cascade above ultimately produces output
through — the cascade decides "how much graph," this dataclass is where that
decision is encoded per node.

### Pointer — `CONCEPT:AU-ORCH.execution.per-job-shape-construction`

`agent_utilities/orchestration/agent_runner.py:816-833`. The shape is
constructed exactly ONCE per job, up front, before any node runs — "the
escalating planner decides how much graph the job needs from cheap signals;
a trivial turn gets a lean shape... so the heavy apparatus never runs for a
simple chat reply." The rejected alternative is each downstream node
(router, verifier, tool-selector) independently re-deciding its own altitude
from scratch — which would duplicate the planning cost per node instead of
once per job, and risks different nodes disagreeing about how "big" the job
is.

### Pointer — `CONCEPT:AU-ORCH.execution.shape-policy-learning`

`execution_profile.py:475-540`, `agent_utilities/orchestration/outcome_router.py`.
The heuristic cascade above is explicitly a PRIOR, not the final word: `_apply_shape_policy`
overlays a **learned per-task-class policy** on top of the heuristic base,
computed fresh on every call (never cached, so it reflects the latest
learning) from the shared `OutcomeRouter`/`CapabilityIndex` reward-EMA — the
same spine the KG-2.68 reasoner router and AHE-3.38 profile evolution already
use, explicitly "not a new bandit." The policy flips the archetype
(`lean`/`full`) only when the learned reward for the alternative exceeds the
prior's for this task-class; otherwise the heuristic's choice stands
unchanged. Rejected alternative: trusting the heuristic cascade forever with
no feedback loop — a heuristic that's systematically wrong for one task class
(e.g. under-escalating a class of jobs that always need the full graph) would
never self-correct.

### Pointer — `CONCEPT:AU-ORCH.execution.planner-failure-feedback`

`execution_profile.py:220-320` (`_RECIPE_CACHE`, `record_shape_outcome`).
Two layers close the loop on a planned shape's actual result: the in-process
recipe cache (Stage 0 above) is EVICTED on a failed run, so the next
identical job re-plans instead of blindly repeating a shape that just failed
— a success leaves it cached. Independently, `record_shape_outcome` feeds
`success × latency` into the shape-policy reward-EMA (the pointer above) so
the learned policy improves from real outcomes, not just the cache. Both
paths are explicitly best-effort and exception-isolated ("never raises into
the caller's result path") — a broken feedback write degrades to "planning
doesn't improve this one time," never to a failed run.

## Risk Assessment

- **Blast Radius**: `agent_utilities/orchestration/execution_profile.py`,
  `agent_utilities/graph/routing/strategies/fast_path.py`,
  `agent_utilities/orchestration/agent_runner.py`,
  `agent_utilities/orchestration/outcome_router.py`,
  `agent_utilities/graph/verification.py` (consumes `shape.run_verifier`).
- **Backward Compatible**: Yes — every shape field defaults to the pre-planner
  full-graph behavior; a caller that never invokes `plan_execution_shape`
  (or an engine-less context) degrades to the safe full/heuristic path at
  every stage.
- **Known weak point**: the in-process `_RECIPE_CACHE`/`_SHAPE_ROUTER` state
  is per-process — a multi-process deployment (multiple graph-os workers)
  learns and caches independently per process, so the SAME job can be shaped
  differently depending on which worker handles it, and a failure learned by
  one worker's recipe-cache eviction is invisible to the others until they
  independently hit the same failure.
