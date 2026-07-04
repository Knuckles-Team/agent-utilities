---
name: kg-writeback
description: >-
  Backfeeds KG-derived knowledge into an external system-of-record (LeanIX, ServiceNow,
  ERPNext, process/capability) — fail-closed, dry-run-first. Use to push
  inferences/enrichments/creations/retirements upstream — "write these inferences back to
  LeanIX", "sync enrichments to ServiceNow".
license: MIT
tags: [graph-os, writeback]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-writeback

`graph_writeback` (CONCEPT:EG-KG.storage.nonblocking-checkpoint/2.9) pushes KG intelligence back to a `target` system-of-record (`leanix|servicenow|erpnext|process|capability|…`). Ops: `inferences_json` `[{source,rel_type,target}]`, `enrichments_json` `[{node,patches,tag}]`, `creations_json` `[{type,name}]`, `retirements_json` `[{node}]`. **Fail-closed:** `dry_run=true` (default) previews the exact proposed writes; live writes need the target's enable flag (e.g. `LEANIX_ENABLE_WRITE`). `action`: `write` | `proposals` (list queued high-stakes) | `approve` (apply a `proposal_id`).

## Invoke
- **MCP:** `load_tools(tools=["graph_writeback"])`, then `graph_writeback(target="leanix", dry_run=true, inferences_json="[...]")`.
- **REST twin:** `POST /graph/writeback` with `{"target": "servicenow", "action": "write", "enrichments_json": "[...]"}`.

## Example
```
graph_writeback(target="leanix", inferences_json='[{"source":"appA","rel_type":"DEPENDS_ON","target":"dbB"}]', dry_run=true)
```
