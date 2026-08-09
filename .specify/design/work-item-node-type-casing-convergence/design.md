# Design Document: WorkItem `node_type` Casing Convergence

CONCEPT:AU-KG.ontology.node-type-casing-convergence

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-OS.governance.fail-closed-degraded-read` | same family of "a component tolerates a bad input instead of failing loud", but about read-time degradation, not write-time identity | 0.35 | OS |
| `AU-KG.ingest.mcp-tool-connector` | unrelated ingestion-shape concept; only adjacent because it also touches `add_node` call sites | 0.20 | KG |

### Extension Analysis

- **Primary Extension Point**: `agent_utilities/knowledge_graph/core/engine_tasks.py`
  `_ControlPlaneWorkItemEngine.add_node` (the control-plane graph adapter used
  for ingestion-scheduled WorkItems), alongside the pre-existing,
  already-correct `IntelligenceGraphEngine.add_node` (main-graph adapter).
- **Extension Strategy**: augment — no new module, no new adapter. The fix
  reorders three lines inside the existing adapter method so the
  caller-supplied `node_type` parameter always wins over whatever the
  `properties` dict happens to carry.
- **New Concept Required?**: Yes — this is a real, measured architectural
  defect with its own root cause and fix, not a restatement of an existing
  decision.

### New Concept Proposal

- **Proposed ID**: `CONCEPT:AU-KG.ontology.node-type-casing-convergence`
- **Augments Pillar**: KG (domain `ontology`)
- **15-Phase Pipeline Integration**: Phase 3 (Ingest/Write) — every WorkItem
  write on the control-plane graph now stamps the same `node_type` casing the
  main-graph adapter always stamped.
- **Justification**: `work_item.py` has exactly one authority-resolving call
  site, `_authority(engine).add_node(item_id, "WorkItem",
  properties=node.to_graph_properties())`, but it resolves to one of **two**
  different adapter implementations depending on which graph a given
  WorkItem is scheduled on:
  - `IntelligenceGraphEngine.add_node` (main graph) always overwrites
    `props["node_type"] = node_type` — the label parameter it was called
    with (`"WorkItem"`, PascalCase, matching the schema label every
    `work_item.py` Cypher query filters on).
  - `_ControlPlaneWorkItemEngine.add_node` (control-plane graph, used for
    ingestion-scheduled WorkItems) used to spread `properties` unreconciled
    into the backend call. `RegistryNode.to_graph_properties()` already
    writes its **own** `node_type` key — the lowercase, snake_case
    `RegistryNodeType` enum value (e.g. `"work_item"`) — and because nothing
    reconciled the two, that lowercase value silently overrode the adapter's
    own label on this path only.

  Live measurement (2026-08-06) found the split in the graph itself:
  **4,590** nodes stamped `node_type="WorkItem"` against **3,760** stamped
  `node_type="work_item"` — the same logical class, split across two
  incompatible property values purely by which scheduling path a WorkItem
  happened to take. Every query, index, or gate written against one casing
  silently missed 45% of the population.

  The fix is applied at the chokepoint both adapters converge on before
  reaching their respective backends (`agent_utilities/AGENTS.md` — *Enforce
  at the chokepoint, not one entrypoint*): `_ControlPlaneWorkItemEngine
  .add_node` now builds `props = dict(properties or {})` and then
  unconditionally overwrites `props["node_type"] = node_type` — the label
  parameter — immediately before calling the backend's `add_node(node_id,
  label=node_type, **props)`. Neither adapter can diverge from the
  caller-supplied class identity again, regardless of what a future
  `RegistryNode` subclass happens to fold into its own properties dict.

## C4 Context Diagram

```mermaid
C4Context
    title WorkItem node_type Casing Convergence — Integration Context

    System_Boundary(b1, "agent-utilities Core") {
        System(wi, "work_item.py", "single WorkItem write call site: _authority(engine).add_node(item_id, \"WorkItem\", properties=...)")
        System(main, "IntelligenceGraphEngine.add_node", "main-graph adapter — already stamped node_type from the label param")
        System(cp, "_ControlPlaneWorkItemEngine.add_node", "control-plane adapter — FIXED to converge on the label param too")
        System(rn, "RegistryNode.to_graph_properties()", "supplies properties dict, incl. its OWN lowercase node_type key")
    }

    Rel(wi, main, "add_node(..., \"WorkItem\", properties)")
    Rel(wi, cp, "add_node(..., \"WorkItem\", properties) — ingestion-scheduled WorkItems")
    Rel(rn, wi, "properties dict (node_type=\"work_item\", lowercase)")
    Rel(cp, cp, "props[\"node_type\"] = node_type overwrites properties' own key")
```

## Data Flow

1. **ORCH**: `work_item.py`'s claim/lease/status Cypher queries filter on
   `node_type = "WorkItem"` (PascalCase, the schema label) — before the fix,
   45% of WorkItems (the control-plane-scheduled slice) were invisible to
   any such filter.
2. **KG**: both graph adapters (`IntelligenceGraphEngine`,
   `_ControlPlaneWorkItemEngine`) now write the identical `node_type` value
   for the same logical class, converging the schema on one authoritative
   casing.
3. **AHE**: none directly — this is a write-path identity-integrity fix, not
   an evolution/learning surface.
4. **ECO**: none — internal to the KG write path, no MCP/REST surface change.
5. **OS**: no configuration surface; the fix is unconditional (see *Native by
   default* — this is not a flag, it is the adapter behaving as it was
   already documented to behave).

## Risk Assessment

- **Blast Radius**: every WorkItem scheduled through the control-plane graph
  (ingestion-scheduled work) — measured 3,760 nodes at the casing split, plus
  every future write on that path.
- **Backward Compatible**: additive/corrective for new writes. Existing
  `work_item` (lowercase)-stamped nodes are not migrated by this change; a
  query or index relying on the lowercase value for historical rows would
  need a one-time data migration (per `AGENTS.md` "No Legacy" — the
  persisted-state exception) to re-stamp them as `"WorkItem"`, which is
  intentionally out of scope for this write-path fix.
- **Breaking Changes**: none to any public API — `add_node`'s signature and
  contract are unchanged; only which value wins when both the label
  parameter and the properties dict independently claim `node_type` is
  affected, and only on the previously-inconsistent path.
- **What would make this wrong later**: if a future adapter or backend
  legitimately needs to preserve a caller-supplied `node_type` distinct from
  the label (no known case today — every `work_item.py` call site passes the
  same string for both), this convergence would need to become conditional
  rather than unconditional. No such caller exists at time of writing.
