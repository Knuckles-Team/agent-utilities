# Design Document: The unified durable-execution tool surface (DE2)

CONCEPT:AU-ORCH.execution.durable-routing-trait

> `agent_utilities/orchestration/durable_tool_surface.py`

## Decision — one small `ctx.*`-shaped mental model, routed by `eg-durable`'s contract, backed honestly where a backend isn't Python-reachable yet

Program: `docs/architecture/durable-execution.md`,
`plans/au-eg-program/program/durable-execution-native.md` Part 3. The routing
decision itself is `eg-durable`'s `CallShape`/`WorkShape`/`DurableBackendKind`
contract (landed at DE0, `crates/eg-durable/src/route.rs`);
`durable_tool_surface.py` is its one AU-side caller, giving every agent/tool a
single, documented mental model — the literal analog of a restate SDK author's
`ctx.*` — regardless of which of the four backends actually serves a call:

- `durable_run` — `CallShape::Run` → `WorkShape::AgentLoopContinuation` →
  `DurableBackendKind::PythonDurableRun` → `DurableRun`. Real, fully wired.
- `durable_sleep` — `CallShape::Sleep` → `WorkShape::AsyncCheckpointedWork` →
  routed at DE0 to `DurableBackendKind::Jobs` (`eg-jobs`'s cron/interval
  triggers), which has no Python-callable submit/query surface today. This
  module backs `durable_sleep` with `DurableRun.sleep_until` instead — a
  documented, honest substitution, not DE0's routing table — delivering the
  SAME externally-observed contract (a durable deadline bound to one
  in-flight execution) on the one backend Python can actually reach,
  poll-based rather than `eg-jobs`'s tick-driven triggers.
- `durable_call` (`CallShape::Call` → `WorkShape::CrossStoreAtomicStep` →
  `DurableBackendKind::MutationStoreSaga`) and `durable_state_get`/
  `durable_state_set` (→ `DurableBackendKind::Statechart`) are named by the
  contract but have no Python-callable backend yet — tracked as a real gap,
  not silently narrowed away (`docs/architecture/durable-execution.md`).

**The rejected alternative** was exposing each of the four `eg-durable`
backend kinds as its own bespoke Python API. That would leak the
implementation detail of WHICH backend serves a given call into every caller,
defeating the entire point of the `restate`-equivalent unification — the
program's headline finding is that these four systems already exist
independently; the value is in the ONE mental model over them, not in a
faithful 1:1 Python binding per backend.

## Risk Assessment

- **Blast Radius**: `agent_utilities/orchestration/durable_tool_surface.py`
  and every caller of `durable_run`/`durable_sleep`.
- **Backward Compatible**: Yes — additive tool surface; existing
  `DurableExecutionManager`/`DurableRun` callers are untouched.
- **Known weak point**: `durable_sleep`'s poll-based substitution for
  `eg-jobs`'s tick-driven triggers is an honest but real semantic gap — a
  caller relying specifically on `eg-jobs`'s own cron/interval firing
  semantics must bypass this surface and use that backend directly.
  `durable_call`/`durable_state_get`/`durable_state_set` are unimplemented
  stubs pending a Python-callable wire surface for their backends.
