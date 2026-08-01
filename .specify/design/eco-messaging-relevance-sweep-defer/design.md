# Design Document: Defer the first relevance sweep so a restart doesn't starve the messaging reply loop

CONCEPT:AU-ECO.messaging.debounce-relevance-sweep

> `agent_utilities/knowledge_graph/core/engine_tasks.py:1067`

## Decision — the hourly relevance sweep's FIRST run is deferred by one full interval, not fired immediately on start

`_tick_kg_analysis` schedules a relevance sweep hourly
(`RELEVANCE_SWEEP_INTERVAL = 3600.0`) and selects the highest-degree stale
`Concept` for background deep analysis, run by the consolidated maintenance
scheduler. The prior behavior — a `0.0` default for "time since last sweep"
— meant a freshly-started process fired the heavy sweep immediately.

**The rejected alternative is exactly that prior default.** The code
comment names the concrete failure mode it caused: this scheduler is
co-located with the messaging router (both run in the same maintenance
process/host), so a startup sweep saturates the process and starves the
inbound reply loop right when the system has just come back up — the worst
possible moment for chat replies to stall. The fix defers the first sweep:
`_last_relevance_sweep` is initialized to `now` on first tick and the
function returns without sweeping; the sweep only fires once a FULL interval
has actually elapsed since then (`engine_tasks.py:1067-1075`).

This is a narrow, single-purpose fix, not a general scheduler redesign — it
only changes the FIRST sweep's timing; every subsequent hourly sweep behaves
as before.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/core/engine_tasks.py`.
- **Backward Compatible**: Yes — makes a restart cheaper, doesn't change
  steady-state behavior.
- **Known weak point**: a process that restarts more often than once per
  hour (e.g. crash-looping) never actually runs a relevance sweep, since
  every restart resets the deferred timer — this trades startup latency for
  a starvation risk under a different failure mode (frequent restarts).
