---
name: kg-ops-causal
skill_type: skill
description: >-
  Enterprise operations causal graph (Codex X-2): joins Langfuse traces to the
  agent/tool/model that ran, the service, its deployment/container, the
  commit/merge-request that changed it, the incident/change ticket, the
  capability/owner, and the governing policy/control/evidence into one causal
  chain, then runs root-cause ranking, blast-radius, change-risk, and
  control-evidence analyses on it. Use for "what caused this failure", "what's
  downstream of this change", "how risky is this change", "gather the
  evidence chain for this control".
license: MIT
tags: [graph-os, ops, causal, root-cause, blast-radius, change-risk]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-ops-causal

`graph_ops_causal` joins entities ALREADY ingested by the connector fleet (langfuse-agent,
container-manager-mcp, gitlab-api/repository-manager, servicenow-api/atlassian-agent,
leanix-agent) into one operations causal chain, and runs the causal-reasoning engine
already shipped (`StructuralCausalModel` + `CausalVerifier` + `SpuriousnessDetector`)
over it — no new traversal algorithm.

Actions:
- `root_cause` — rank probable root-cause changes/services for a failure `node_id`
  (upstream, favoring true topological-source causes over closer intermediate symptoms).
- `blast_radius` — downstream impact of a change `node_id`.
- `change_risk` — predict the risk of a proposed change `node_id` from its blast
  radius plus `incident_history_json`.
- `control_evidence` — gather + verify the evidence chain for a governance control `node_id`.
- `join` — materialize `links_json` as real graph edges via the shared enrichment
  writer (creates zero new nodes — only edges between ids that already exist).

Supply the causal edges explicitly via `links_json` for an offline/test-friendly
model, or omit it with an active engine + `node_id` to load the neighborhood live
from the KG.

## Invoke
- **MCP:** `load_tools(tools=["graph_ops_causal"])`, then
  `graph_ops_causal(action="root_cause", node_id="trace:1", links_json="[...]")`.
- **REST twin:** `POST /ops/causal` with `{"action": "blast_radius", "node_id": "commit:abc123"}`.

## Example
```
graph_ops_causal(action="change_risk", node_id="commit:abc123",
                  incident_history_json='[{"node_id":"incident:INC001","severity":0.8}]')
```

## Delegation
If graph-os is reachable, offload composite multi-step work via `graph_orchestrate`
(`execute_agent` / `execute_workflow`) instead of hand-running the steps — let the
local LLM + Loop engine do it, and resolve only the exceptions.
