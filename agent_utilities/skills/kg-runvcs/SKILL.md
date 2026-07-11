---
name: kg-runvcs
skill_type: skill
description: >-
  Agent-native run version-control — fork, revert, and replay a LIVE agent run
  as content-addressed commits binding its conversation + filesystem +
  process/event frontier into one exact world. Use for "checkpoint this run",
  "revert the run to before that tool call", "branch a new run from here",
  "replay this run deterministically", "what commits does this run have".
license: MIT
tags: [graph-os, runvcs, fork, revert, replay]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# KG RunVCS — fork / revert / replay a live agent run

> **Condensed intent-surface note (Seam 8).** Under the small/cheap-LLM profile (`MCP_TOOL_MODE=intent`), `graph_runvcs` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_runvcs"])` once per session (as below), then proceed exactly as documented; or (2) call the `act` intent verb with the same natural-language request — the resolver routes to `graph_runvcs` for you and returns the result plus a routing justification. The default `MCP_TOOL_MODE=condensed` is completely unaffected.


`graph_runvcs` (CONCEPT:AU-ORCH.runvcs.run-commit) is agent-native version control
for a *live* run: it snapshots a run's conversation, filesystem, and
process/event frontier together into one content-addressed `RunCommit`, so you
can rewind, branch, or deterministically replay a run exactly like `git` does
for a repo — except the "repo" is the whole live execution state.

Actions:
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
`run.select` ActionPolicy gate — this skill only covers a *live* session.

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
```
