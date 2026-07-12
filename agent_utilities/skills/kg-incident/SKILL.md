---
name: kg-incident
skill_type: skill
description: >-
  Browse the cross-layer Incident Brain: correlate recent hardware/os/orchestration/
  service/network :HealthAnomaly rows into deduplicated :Incident nodes, list open
  incidents, and pull one incident's full detail. Read-only — proposes and executes
  no remediation. Use for "what's broken right now", "show me open incidents", "run
  a correlation pass", "what's the root-cause layer for this incident".
license: MIT
tags: [graph-os, incident, observability, aiops, correlation]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-incident

`graph_incident` is a thin, read-only wrapper over the correlation engine already
implemented in `observability/incidents.py`: recent `:HealthAnomaly` rows from every
producer (fan-manager=hardware, systems-manager=os, container-manager-mcp=
orchestration, lgtm-mcp/uptime-kuma-agent=service, tunnel-manager=network) that land
on the same host/node within a short time window are grouped into ONE `:Incident`,
with the deepest contributing layer (hardware < os < orchestration < service <
network) estimated as the likely root cause.

Three actions, all read-only:

- `correlate` — run one correlation pass now (`window_s`/`days` control the
  clustering window / lookback). Idempotent: an already-open incident with the same
  signature is deduped (`"deduped": true`), not re-written. This is the SAME pass a
  CronJob runs periodically — use this to get a fresh answer on demand instead of
  waiting for the next tick.
- `list` — recent `:Incident` nodes, optionally `status`-filtered (e.g. `status="open"`),
  newest (`opened_at`) first.
- `get` — one `:Incident` node's full detail by `incident_id`.

**No remediation is proposed or executed by this tool.** `propose_remediation`
(writes a report-only `:RemediationProposal`) and ticket routing stay CronJob-driven
and report-only by design — see the `incidents.py` module docstring. An operator
wires a proposal onto the `ActionPolicy`-gated `FleetActuator`/`graph_loops` dispatch
path separately when ready for autonomous actuation on a given action kind.

## Invoke

- **MCP:** `load_tools(tools=["graph_incident"])`, then
  `graph_incident(action="list", status="open")`.
- **REST twin:** `POST /incident` with `{"action": "correlate", "window_s": 300, "days": 1}`.

## Example

```
graph_incident(action="correlate", window_s=300, days=1)
// -> {"surface": "incident", "action": "correlate", "count": 2, "incidents": [...]}

graph_incident(action="list", status="open", limit=10)
// -> {"surface": "incident", "count": 2, "incidents": [{"id": "health:incident:r510:...", ...}]}

graph_incident(action="get", incident_id="health:incident:r510:abc123")
// -> {"surface": "incident", "incident": {"id": ..., "root_cause_layer": "hardware", ...}}
```

## Delegation

If graph-os is reachable, prefer `graph_orchestrate(action="execute_workflow")` for a
composite triage (correlate -> inspect open incidents -> hand off to the relevant
package's own troubleshooting skill, e.g. `host-doctor`/`fan-manager-control`) rather
than hand-chaining calls yourself.
