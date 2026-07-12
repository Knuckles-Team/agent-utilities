---
name: kg-feedback
skill_type: skill
description: >-
  Records a human/agent correction so the system learns durably — reward adjustments,
  governance rules, eval cases, reads-avoided and action-outcome loops, and pinned
  gotchas. Use to teach or correct the KG — "record this correction", "pin a gotcha", "log
  the outcome of this action".
license: MIT
tags: [graph-os, feedback, learning]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-feedback

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `graph_feedback` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_feedback"])` once per session (as below), then proceed exactly as documented; or (2) call the `why` intent verb with the same natural-language request — the resolver routes to `graph_feedback` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


`graph_feedback` persists corrections keyed by `correction_type`: `outcome` (adjust an entity's reward), `rule` (durable governance/voice/source rule consulted at retrieval), `eval` (add a regression case), `reads_avoided` (close the code_context reads-avoided loop, AU-AHE.evaluation.reads-avoided-feedback), `action_outcome` (close the loop on any autonomous action so routing prefers what works, AU-AHE.evaluation.action-outcome-feedback), `gotcha` (pin a hard-won trap to a file/module so `kg-code` surfaces it).

## Invoke
- **MCP:** `load_tools(tools=["graph_feedback"])`, then `graph_feedback(correction_type="gotcha", target_id="path/to/file.py", corrected_value="...")`.
- **REST twin:** `POST /graph/feedback` with `{"correction_type": "action_outcome", "target_id": "...", "corrected_value": "{...}"}`.

## Example
```
graph_feedback(correction_type="action_outcome", target_id="model_route:chat",
               corrected_value='{"success":true,"reward":1.0}')
```
