# Design Document: Synchronous KG round-trips on the reply path move off the event loop via `to_thread`, collapsed into one call — as the Python-side mitigation until the engine offers a single-round-trip `discover()`

CONCEPT:AU-ORCH.routing.offload-sync-roundtrip

> Realised at `agent_utilities/graph/_router_impl.py:120-127` and `:141-147`
> (pre-LLM discovery bundle), `agent_utilities/orchestration/agent_runner.py:828-833`
> (`_resolve_agent_from_kg`), and `agent_utilities/graph/executor.py:429-435`.
> The deferred replacement is recorded in
> `docs/architecture/non-blocking-execution.md:299-306`. Introduced by commit
> `b794e6af` ("perf(orchestration): chat execution profile + non-blocking reply
> path").

## Decision — wrap the synchronous calls rather than make them async, and collapse the bundle into ONE offload rather than one per call

Agent resolution, the pre-LLM discovery bundle, checkpoint saves and registry
hydration are all synchronous backend round-trips, and they all sit on the
async reply path. The comment at `_router_impl.py:141-143` states the
consequence plainly: *"Running them directly on the event loop stalled the
async reply path."* A stalled loop does not just slow the turn that caused it —
it delays every other concurrent turn in the process.

Two choices were made together. First, the calls are offloaded with
`asyncio.to_thread` rather than rewritten as native async. Second — and this is
the part that matters for latency — the discovery bundle is collapsed into a
**single** `to_thread` call rather than one offload per lookup, and the
`find_agent_for_tool` N+1 is deduped into a single pass. Offloading N sync
calls individually removes the loop stall but keeps the N sequential
round-trips; collapsing them removes both.

**The rejected alternative is the one this codebase actually wants, and it is
explicitly deferred rather than dismissed.** The real fix is a single
round-trip `discover()` on the Rust engine, which would replace the whole
bundle with one native call. `docs/architecture/non-blocking-execution.md:299-306`
records it as P2, contract-only, not yet implemented, and describes the
`to_thread` + dedupe approach as *"the Python-side mitigation"* until it lands.
This is a deliberate choice to fix the symptom now at the orchestration layer
rather than block a latency regression on an engine change: the thread pool
costs a context switch per bundle, which is cheap next to the round-trips
themselves, and the mitigation can be deleted wholesale when `discover()`
exists.

Rewriting the callees as natively async was the third option and was not taken.
The synchronous surface is the engine client's; making it async would push the
change down into a dependency shared by many callers, for a benefit the P2
`discover()` work supersedes anyway.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/_router_impl.py`,
  `agent_utilities/orchestration/agent_runner.py`,
  `agent_utilities/graph/executor.py`, plus `hierarchical_planner.py`,
  `verification.py` and `engine.py` — roughly 20 call sites applying one
  pattern.
- **Backward Compatible**: Yes — same calls, same results, different thread.
- **Known weak point**: `to_thread` moves work to the default executor, whose
  pool is shared process-wide. Under enough concurrent turns the offloaded
  round-trips contend for pool slots, which converts a loop stall into pool
  starvation rather than eliminating the wait. This is acceptable only because
  the bundle is collapsed to one offload per turn; re-introducing per-call
  offloads would reintroduce the pressure.
