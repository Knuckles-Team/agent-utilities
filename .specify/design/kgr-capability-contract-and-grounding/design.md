# Design Document: One capability contract for Tool/Skill kinds, and one fail-closed grounding policy at the model-transport boundary

CONCEPT:AU-KG.retrieval.unified-capability-contract ·
CONCEPT:AU-KG.retrieval.fail-closed-grounding-contract

> `agent_utilities/core/capability_contract.py` (Decision 1),
> `agent_utilities/core/contextual_model.py:85-116` (Decision 2), both
> surfaced in `agent_utilities/orchestration/manager.py` and
> `agent_utilities/mcp/multiplexer.py`.

## Decision 1 — `Capability` is the one contract a `Tool` node and a `Skill`/`CallableResource` node both satisfy

`CONCEPT:AU-KG.retrieval.unified-capability-contract`

`capability_contract.py:1-26` names the prior state directly: "Before this
module, `find_tools` ranked only fleet `Tool` entries and
`Orchestrator.resolve_capability` special-cased `Skill`/`WorkflowDefinition`
node types with no notion of a `Tool` at all — **two disjoint,
kind-branching paths a caller had to know about in advance**." A `skill://`
resource and an MCP `Tool` are different wire shapes but the same thing to a
caller ranking or binding a capability: a named, describable, rankable,
bindable unit of work. `Capability` gives both a `kind`, a stable `id`, a
display `name`, a ranking `score`, an optional owning `server`, and
`to_binding()` — the exact keyword arguments `graph_orchestrate`/
`Orchestrator.execute_agent`/`execute_capability` already accept — so a
caller resolving from ranked intent search never branches on `kind`: it calls
`.to_binding()` and spreads the result into the same delegation call,
whichever kind won the ranking. `manager.py:640-651`'s `_search_hit_kind`
delegates to the SAME classifier `find`/`find_tools` use, so a `Tool` node
resolves consistently everywhere rather than being silently dropped as
unclassified in a code path the classifier forgot to reach.

**The rejected alternative** is the two-disjoint-paths status quo itself:
every NEW caller ranking or binding a capability would have had to
re-implement the kind-branching logic, and any caller that forgot to handle
one kind (as `search_hybrid`'s hit classification originally did for `Tool`)
silently produced worse results rather than an error. The module is
deliberately dependency-free (no KG/engine imports) specifically so both the
low-level fleet multiplexer and the orchestration layer can share this one
classification+binding contract without a new cross-layer dependency being
introduced to get there.

## Decision 2 — the mandatory evidence-compilation policy is FAIL CLOSED by default; degraded operation is an explicit, marked, per-run opt-in

`CONCEPT:AU-KG.retrieval.fail-closed-grounding-contract`

`contextual_model.py:85-95` states the policy directly: `"required"` (the
default) means a compile timeout, a compile error, or a retrieval-quality-
gate failure **refuses the model call outright**
(`GroundingUnavailableError`) "rather than silently sending an ungrounded
request." `"best_effort"`/`"none"` are explicit per-run opt-ins that let the
request proceed degraded — but always with an explicit marker in the
messages themselves and on the current OTel span, and the run-level outcome
is tracked via `grounding_snapshot` so `agent_runner.run_agent` "can refuse
to record a degraded run as a plain success for reward/learning purposes"
(`manager.py:506-511` threads `grounding=` through every delegation
entrypoint on this same contract).

**The rejected alternative, and the concrete bug that motivated the fix**, is
documented in the code itself (`contextual_model.py:100-116`): the aggregate
degradation outcome is stored in a **mutable dict inside a `ContextVar`**,
deliberately, not a pair of plain bool/str `ContextVar`s. A `ContextVar`
*write* inside a child `asyncio.Task` (or `to_thread`) is invisible to the
parent — each runs in a context COPY — and the model call that discovers a
degradation frequently runs one task-boundary below the scope that reads it
back. "Measured: with plain-value ContextVars the degraded outcome was
silently lost across a single `create_task` hop, so a degraded run was
recorded as a plain success — the exact defect this contract exists to
prevent." Using a mutable dict REFERENCE means the child's in-place mutation
is visible to the parent scope that installed it, closing the gap a
plain-value ContextVar could not.

A second rejected alternative, on the same class boundary: running the
mandatory compile inline on the asyncio event loop. It is a blocking,
engine-contended retrieval; running it inline once SIGKILLed graph-os because
a contended retrieval blocked the loop long enough that even status-only
`/health` timed out. It now runs off-loop (`asyncio.to_thread`), bounded by
`_CONTEXT_COMPILE_TIMEOUT_S` (~10s, a liveness guard, deliberately a named
module constant rather than an operator env knob), with a per-process
circuit breaker (`_CTX_COMPILE_BREAKER_THRESHOLD=3` consecutive
degradations opens it for `_CTX_COMPILE_BREAKER_COOLDOWN_S=30s`) so a
consistently slow/broken retrieval leg stops paying the timeout on every
call once it has already proven unhealthy.

## Risk Assessment

- **Blast Radius**: `capability_contract.py`, `manager.py`,
  `mcp/multiplexer.py`, `contextual_model.py`, `agent_runner.py`,
  `workflows/runner.py`.
- **Backward Compatible**: Yes — `Capability`/`to_binding()` is an additive
  classification layer; `grounding="required"` is the pre-existing implicit
  behavior made explicit and enforced, so unmodified callers see no change.
- **Known weak point**: the fail-closed default trades availability for
  correctness — a genuinely healthy model call that happens to land during a
  transient compile-timeout window is refused outright rather than served
  ungrounded-but-probably-fine; the circuit breaker's cooldown window is a
  fixed constant, not adaptively tuned to observed retrieval-leg recovery
  time.
