# Design Document: ARIS EPC step structure is lifted to the SAME shape as Camunda BPMN, not left at coarse model-level granularity

CONCEPT:AU-KG.ontology.descriptive-process-world-gains

> `agent_utilities/knowledge_graph/enrichment/extractors/aris.py`.

## Decision — when the ARIS client exposes the EPC detail surface, lift function/rule/event structure to the same `BusinessTask`/`FLOWS_TO` shape Camunda BPMN produces

`aris.py:13-37` states the design directly: "**Step-level structure lift.**
When the injected client also exposes the EPC detail surface
(`list_model_objects`/`list_model_connections`) ... each *process* model's
Event-driven-Process-Chain is lifted to the SAME shape the Camunda BPMN
extractor produces, so an ARIS EPC and its Camunda implementation are
queryable/reasoned over identically": a function (`OT_FUNC`) becomes a
`BusinessTask`; a rule/operator (`OT_RULE`) becomes a `BusinessTask` flagged
`is_gateway=True` with a `gateway_kind` (AND/OR/XOR) — "the EPC analogue of a
BPMN gateway, so the `ProcessPlanCompiler` collapses branching identically";
an event (`OT_EVT`) is deliberately NOT lifted — "collapsed through, like BPMN
start/end events" — so `FLOWS_TO` ordering between functions and rules
survives without an extra hop. Cross-vendor identity reconciliation is a
third decision in the same file: a model record carrying a Camunda key or
Egeria GUID gets an `ALIGNED_WITH` equivalence edge to its Camunda/Egeria
twin, so the SAME real-world process modeled in two tools "collapse to one
identity under reasoning."

**The rejected alternative is model-level-only ARIS ingestion** — treat each
ARIS process model as one opaque `BusinessProcess` node (which the base
extractor already does when the client lacks the detail surface) and stop
there. That would leave ARIS strictly coarser than Camunda: `governance_import.py`'s
process→`WorkflowDefinition` translation (see
`.specify/design/kgo-e1-governance-process-normalization/design.md`) needs
step-level structure to build an executable DAG, and cross-vendor reasoning
(does this ARIS process match that Camunda one) needs comparable granularity
on both sides. Lifting the EPC detail — but ONLY when the client actually
exposes it — means ARIS ingestion degrades gracefully to model-level when the
richer surface isn't available, rather than requiring it unconditionally.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/enrichment/extractors/aris.py`,
  `governance_import.py`'s ARIS importer (shares the same lifted shape),
  `ProcessPlanCompiler` (gateway-collapsing logic).
- **Backward Compatible**: Yes — the step lift only activates when the
  injected ARIS client exposes the detail-surface methods; a client without
  them yields the pre-existing model-level-only extraction.
- **Known weak point**: event collapsing ("NOT lifted, collapsed through")
  means an EPC's own event semantics (which are meaningfully different from a
  BPMN start/end event in some ARIS models) are not preserved as distinct
  graph structure — only the ordering effect they have on `FLOWS_TO` survives.
