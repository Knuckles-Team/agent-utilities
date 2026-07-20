---
name: kg-runvcs
skill_type: skill
description: >-
  Agent-native run version-control — fork, revert, and replay a LIVE agent run
  as content-addressed commits binding its conversation + filesystem +
  process/event frontier into one exact world; PLUS the Agent Digital Twin
  (X-8) — capture, deterministically replay, counterfactually re-drive under a
  swapped policy/model, and step through a PAST run for incident
  investigation. Use for "checkpoint this run", "revert the run to before that
  tool call", "branch a new run from here", "replay this run
  deterministically", "what commits does this run have", "capture a digital
  twin of this run", "would this have been approved under the new policy",
  "walk me through what happened in this incident".
license: MIT
tags: [graph-os, runvcs, fork, revert, replay, twin, digital-twin, incident]
tier: core
metadata:
  author: Genius
  version: '0.2.0'
---

# KG RunVCS — fork / revert / replay a live agent run, and the Agent Digital Twin

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `graph_runvcs` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_runvcs"])` once per session (as below), then proceed exactly as documented; or (2) call the `act` intent verb with the same natural-language request — the resolver routes to `graph_runvcs` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


`graph_runvcs` (CONCEPT:AU-ORCH.runvcs.run-commit) is agent-native version control
for a *live* run: it snapshots a run's conversation, filesystem, and
process/event frontier together into one content-addressed `RunCommit`, so you
can rewind, branch, or deterministically replay a run exactly like `git` does
for a repo — except the "repo" is the whole live execution state.

Live-run actions:
- **`list`** — live run sessions.
- **`status`** — a run's event/commit/message counts + log digest.
- **`commit`** (+`label`) — snapshot messages + fs + events into one `RunCommit`.
- **`revert`** (+`commit_id`) — restore a run's files + process + messages to a
  prior commit.
- **`fork`** (+`commit_id`) — branch a NEW run from a commit into a fresh
  workspace; the parent run is untouched.
- **`discard`** — drop the uncommitted event delta.
- **`replay`** — deterministically replay the run's event log (a recorded
  exchange stands in for the model) and verify reproduction.

Retained-output accept/discard of a finished run is governed by the
`run.select` ActionPolicy gate — the live-run actions above only cover a
*live* session; the twin actions below cover a *past* run independently of
any live session.

## Agent Digital Twin (X-8, CONCEPT:AU-ORCH.twin.agent-digital-twin)

A durable, replayable projection of one PAST agent run — the exact
model/prompt/tool/skill/policy VERSIONS it executed under, its run graph
(`WorkItem` ids), every tool call + evidence, the policy decisions it made,
and its outcome — for regression testing, incident investigation,
counterfactual policy/model evaluation, and safe-evolution proposals. Piggybacks
onto `graph_runvcs` exactly like the live-run actions above (same tool, same
REST twin, same `_execute_tool` core — Two-surfaces by construction).

- **`twin_capture`** (+`run_id`, `agent_name`, `task`, `versions` JSON,
  `outcome`, `persist`) — best-effort hydrate a twin for `run_id` from the
  KG's already-recorded `:ToolCall`/`:WorkItem` rows, optionally persist it as
  a durable `:AgentDigitalTwin` node (`persist=true`, the default), and return
  the full serialized twin JSON. Pass that JSON back in as `twin` to every
  action below.
- **`twin_replay`** (+`twin`) — regression-replay: re-drive the twin's
  recorded event log verbatim (never re-executes a real tool/model call) and
  report whether the reconstruction is bit-for-bit `deterministic`.
- **`twin_counterfactual`** (+`twin`, `policy_overrides` JSON and/or
  `model_responses` JSON, optional `versions` JSON) — re-drive the SAME
  recorded run under a swapped policy ruleset (genuinely re-invokes
  `ActionPolicy.decide()`) and/or a substituted model/prompt response,
  surfacing a `decision_delta`/stream divergence — "would this have been
  approved under the new policy", "what would a different model have done
  here".
- **`twin_incident`** (+`twin`) — read-only, ordered step-through of the
  recorded run (what was proposed, what was recorded as its outcome) for
  human-inspectable incident investigation. Never re-runs anything.

## Invoke
- **MCP:** `load_tools(tools=["graph_runvcs"])`, then
  `graph_runvcs(action="commit", run_id="<id>", label="before risky edit")`.
- **REST twin:** `POST /graph/runvcs` with
  `{"action": "fork", "run_id": "<id>", "commit_id": "<commit>"}`.

## Example

```jsonc
// checkpoint the current run
graph_runvcs(action="commit", run_id="run-42", label="pre-refactor")

// branch a fresh run from that checkpoint, parent untouched
graph_runvcs(action="fork", run_id="run-42", commit_id="<returned commit_id>")

// something went wrong — revert to the checkpoint
graph_runvcs(action="revert", run_id="run-42", commit_id="<commit_id>")

// --- Agent Digital Twin: investigate a PAST incident run ---
// 1. capture a twin from the KG's own recorded provenance
graph_runvcs(action="twin_capture", run_id="run:incident-77", persist=true)

// 2. step through exactly what happened, in order
graph_runvcs(action="twin_incident", twin="<twin JSON from step 1>")

// 3. ask "would today's stricter policy have blocked this?"
graph_runvcs(
  action="twin_counterfactual",
  twin="<twin JSON from step 1>",
  policy_overrides="{\"version\": 2, \"defaults\": {\"tier\": \"approval_required\"}, \"rules\": []}"
)
```
