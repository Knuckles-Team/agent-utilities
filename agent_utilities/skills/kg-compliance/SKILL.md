---
name: kg-compliance
skill_type: skill
description: >-
  Compliance posture rollup + redacted bulk export — joins the tamper-evident
  hash-chained audit ledger with governance node counts (Control/Policy/Risk/
  ComplianceRequirement/ComplianceGate/Regulation/Assessment/Incident/Finding/...
  already ingested from CISO Assistant + the TRM portfolio-intelligence engine)
  into one auditor-facing view, and bulk-exports a redacted subgraph via the
  engine's own disclosure_level (Full/Skeleton/ExistenceOnly) redaction. Use for
  "what's our compliance posture", "is the audit chain intact", "export these
  controls redacted for an auditor", "how many open compliance gaps do we have".
license: MIT
tags: [graph-os, compliance, audit, governance, redaction]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-compliance

`graph_compliance` is an aggregation layer over primitives that already exist — it
adds no new compliance or redaction logic:

- **`posture`** joins `graph_audit`'s `verify()` report (the engine's SHA-256
  hash-chained mutation-log check) with a node-count + status-breakdown rollup of
  the governance labels the CISO Assistant extractor (`Control`/`Policy`/`Risk`/
  `RiskAssessment`/`ComplianceAssessment`/`Framework`/`Asset`/`Incident`/
  `SecurityException`/`Finding`/`Entity`) and the TRM portfolio-intelligence engine
  (`ComplianceRequirement`/`ComplianceGate`/`Regulation`/`Assessment`) already
  ingest into the KG.
- **`export`** bulk-exports a policy-redacted subgraph: given `node_ids` (a JSON
  array) or a read-only `cypher` query selecting an `id` column, it calls the
  engine's OWN `explain_belief(node_id, disclosure_level)` per id — the SAME
  per-node redaction primitive `kg-epistemic-answer`'s `why` action / `graph_epistemic`
  use — and collects the results, bounded by `limit`. `disclosure_level` is
  `Full` (default) / `Skeleton` / `ExistenceOnly`.

## Invoke

- **MCP:** `load_tools(tools=["graph_compliance"])`, then `graph_compliance(action="posture")`.
- **REST twin:** `POST /compliance` with `{"action": "posture"}`.

## Example

```
graph_compliance(action="posture")
// -> {"surface": "compliance", "audit_ledger": {"ok": true, "entries": 412, ...},
//     "node_counts": {"Control": 58, "ComplianceGate": 6, "Incident": 3, ...},
//     "status_breakdown": {"Control": {"satisfied": 51, "gap": 7}, ...}}

graph_compliance(action="export",
                  cypher="MATCH (n:Control) WHERE n.status = 'gap' RETURN n.id AS id LIMIT 50",
                  disclosure_level="Skeleton")
// -> {"surface": "compliance", "requested": 7, "exported": 7,
//     "entries": [{"node_id": "ciso_assistant_control:...", "belief": {"root": {...}}}, ...]}
```

## Honest limitations

- `posture`'s node counts are a live snapshot of whatever the CISO Assistant /
  Egeria / TRM connectors have already ingested — this tool does not itself sync
  those sources (see the `agent-utilities-source-integration` skill for that).
- `export`'s redaction is exactly as strong as `ExplainBelief`'s `disclosure_level`
  handling on the connected engine build — a build without the `epistemic-redaction`
  feature degrades `disclosure_level` to a no-op (full disclosure), same as
  `kg-epistemic-answer`'s own documented limitation.

## Delegation

If graph-os is reachable, prefer `graph_orchestrate(action="execute_workflow")` for a
recurring posture report or a large bulk export rather than hand-driving the calls
yourself.
