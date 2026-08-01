# Design Document: A high-volume scheduled backfill runs in its OWN task lane, not capped at the best-effort maintenance floor

CONCEPT:AU-KG.ontology.capability-card-backfill-lane

> `agent_utilities/core/schedule_engine.py:243-266`
> (`ScheduleEntry.task_type`), `agent_utilities/knowledge_graph/core/engine_tasks.py`.

## Decision — `task_type` on a schedule entry selects the FUNCTIONAL LANE the scheduled tick runs in, defaulting to the best-effort `maint` lane but overridable per schedule

`schedule_engine.py:243-251` states the mechanism: `task_type` "selects the
FUNCTIONAL LANE the tick runs in ... Defaults to `scheduled_job` (the `maint`
lane). A high-volume schedule whose work is a throughput backfill (e.g. OWL
card enrichment) overrides this so it runs in its OWN lane instead of being
capped at the best-effort maint floor. The worker routes any of these types
through the SAME `run_scheduled_job` dispatcher, so only the lane (and thus
the worker share + model role) differs."

**The rejected alternative is running every scheduled job through one shared
`maint` lane** — the simpler default, and the one every OTHER schedule entry
still uses. A capability-card backfill (bulk enrichment of OWL capability
cards) is throughput-shaped work: if it ran in the shared best-effort `maint`
lane, it would compete for worker share with every other maintenance tick and
be capped at that lane's floor, meaning a large backfill could starve or be
starved by unrelated maintenance work. Giving it its own `task_type` (and
therefore its own lane) means its worker-share and model-role allocation are
tuned for a bulk-throughput job specifically, without a new dispatcher — the
SAME `run_scheduled_job` function handles it, only the lane routing differs.

## Risk Assessment

- **Blast Radius**: `core/schedule_engine.py` (`ScheduleEntry.task_type`),
  `knowledge_graph/core/task_lanes.py`, `knowledge_graph/core/engine_tasks.py`
  (`run_scheduled_job` dispatch).
- **Backward Compatible**: Yes — `task_type` defaults to `scheduled_job`
  (unchanged prior behavior); only a schedule that explicitly opts into a
  dedicated lane changes behavior.
- **Known weak point**: lane selection is a per-schedule-entry string with no
  validation shown at this call site that the named lane actually exists in
  `task_lanes` — a typo'd `task_type` would presumably fail at dispatch time
  rather than at schedule-authoring time.
