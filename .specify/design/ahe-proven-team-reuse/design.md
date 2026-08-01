# Design Document: Search for a proven team composition before assembling a new one from scratch

CONCEPT:AU-AHE.harness.proven-team-reuse

> `agent_utilities/core/registry/kg_adapter.py:819-895`.

## Decision — a successful ad hoc coalition is promoted into a reusable, success-rate-ranked `TeamConfig`, and future tasks search it before cold-starting a new team

`find_matching_team_config` (`kg_adapter.py:823-843`) searches `TeamConfig`
nodes whose `task_pattern` matches a query (keyword overlap today, with a
noted future upgrade to cosine similarity), returning matches sorted by
`success_rate` descending. `promote_coalition_to_template`
(`kg_adapter.py:882-895`) is the write side: when a `SwarmCoalition` proves
successful, it is promoted into a `TeamConfigNode`, linked via
`REUSED_TEAM` back to the originating coalition, and the router's hot cache
is invalidated so the new template is immediately visible.

**The rejected alternative is composing a fresh team for every task,
regardless of whether a proven-successful composition for a similar task
already exists.** That throws away accumulated evidence about what worked —
every task pays the full cost of team assembly (agent selection,
capability matching) from a cold start, and a composition that reliably
succeeds on a task pattern gets no advantage over one that's never been
tried. Ranking matches by `success_rate` means reuse itself is
evidence-weighted: a task facing several matching prior compositions
prefers the one with the best track record, not just the most recent or
first-found match.

## Risk Assessment

- **Blast Radius**: `agent_utilities/core/registry/kg_adapter.py`, the team
  router that reads `TeamConfig` matches, the hot cache the promotion step
  invalidates.
- **Backward Compatible**: Yes — falls back to cold-start team assembly when
  no `TeamConfig` match is found; nothing requires a match to exist.
- **Known weak point**: matching is plain keyword overlap between the query
  and `task_pattern` ("cosine similarity can be added later" is noted
  in-code as a future upgrade) — a semantically similar task phrased with
  different vocabulary won't match an otherwise-proven `TeamConfig`, so
  reuse silently degrades to cold-start assembly rather than surfacing a
  near-miss.
