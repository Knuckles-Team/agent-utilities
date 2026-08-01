# Design Document: Check for a proven team BEFORE paying for LLM planning, with the reuse decision owned by one strategy module

CONCEPT:AU-AHE.harness.team-config-precheck

> `agent_utilities/graph/_router_impl.py:235-255`.

## Decision — the router precheck's ONLY job is to fetch candidates and hand the actual reuse decision to `team_reuse.select_reusable_team`

The router (`_router_impl.py:235-252`) queries `find_matching_team_config`
(proven-team-reuse's search) BEFORE invoking LLM-based planning, off the
event loop (`asyncio.to_thread`) since it's a synchronous KG round-trip. The
comment at the decision point is explicit about scope: "R2 — reuse decision
owned by the team_reuse strategy (single source of truth)" — the router
calls `select_reusable_team(matching_teams)` rather than deciding itself
whether a match is good enough to reuse.

**The rejected alternative is two things, both explicitly avoided.** First,
running LLM-based planning unconditionally and only consulting proven teams
as a post-hoc suggestion — that would mean paying full planning cost (an LLM
call) even on a query that already has an excellent matching `TeamConfig`,
the exact cost `proven-team-reuse` exists to avoid. Second, inlining the
match/accept-or-reject logic directly in the router — that would duplicate
decision logic that already lives in the `team_reuse` strategy module,
risking the two copies drifting (the router's inline copy staying stale
while `team_reuse`'s real logic evolves). Keeping the router as a thin
precheck that only fetches candidates and defers the accept/reject call to
one owning module is the concrete design choice this concept id names.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/_router_impl.py`,
  `agent_utilities/graph/routing/strategies/team_reuse.py`
  (`select_reusable_team`), `agent_utilities/core/registry/kg_adapter.py`
  (`find_matching_team_config`).
- **Backward Compatible**: Yes — a precheck gate; when no match qualifies,
  routing falls through to LLM-based planning exactly as before this
  precheck existed.
- **Known weak point**: `find_matching_team_config` is called with `top_k=1`
  at this call site — the precheck only ever considers the single best
  keyword-matched candidate, so a near-tie where the second-best match would
  actually have been selected by `select_reusable_team`'s own criteria is
  never presented to it.
