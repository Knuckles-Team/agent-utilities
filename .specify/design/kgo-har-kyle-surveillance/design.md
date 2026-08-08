# Design Document: Microstructure surveillance signals are ontology Interfaces, not raw untyped nodes

CONCEPT:AU-KG.ontology.kyle-insider-stealth-surveillance

> `agent_utilities/knowledge_graph/ontology/finance_objects.py`.

## Decision — the Kyle insider/stealth-trading surveillance signal is a first-class typed Interface, linked to its research grounding and the capability registry, not a bespoke untyped node

`finance_objects.py:1-22` names the provenance and the DEFENSIVE framing
directly: this makes "the Kyle insider/stealth-trading surveillance work
(engine kernel KG-2.20k, detector EE-042, gate EE-043 — distilling
arXiv:2605.27684) *ontologically driven*" — informed-flow surveillance and
maker adverse-selection **protection**, explicitly "not trade concealment."
`register_finance_ontology` (`finance_objects.py:43-49`) registers
`MicrostructureSignal` and `SurveillanceSignal` as ontology **Interfaces**
(not concrete node types) with typed **Links** the OWL bridge reasons over:
`GROUNDED_IN` (the signal is grounded in the originating research paper,
inverse `SUPPORTS`) and `RELATES_TO` (the signal relates to an ecosystem
`Concept`). Registration is idempotent — re-import safe, skipping an
interface/link already registered.

**The rejected alternative is the status quo the module explicitly preserves
rather than replaces**: "the emerald-exchange MCP keeps writing raw
`microstructure_signal` nodes for robustness; this layer gives them a governed
ontology **schema** (conformance, link cardinality, OWL/SHACL emission,
cross-domain reasoning)" (`finance_objects.py:18-20`). The decision was not
"replace the raw ingestion path" (which would be a riskier, coupled change to
a live trading-adjacent system) but "layer a governed shape over it" — an
Interface conformance check can validate a raw node without that node's
writer ever needing to change. Modeling the signal as an Interface rather than
a concrete node type is itself the second half of the decision: it lets
`GROUNDED_IN`/`RELATES_TO` reasoning generalize across every concrete type
that implements `MicrostructureSignal`, rather than hard-wiring the link
types to one specific node label.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/finance_objects.py`,
  `emerald-exchange` MCP's raw `microstructure_signal`/`surveillance_signal`
  node writers (unaffected, still the ingestion path), `interfaces.py` /
  `links.py` default registries.
- **Backward Compatible**: Yes — additive governance layer; raw node writes
  are untouched.
- **Known weak point**: because the raw ingestion path is deliberately left
  unchanged ("kept for robustness"), a raw node that violates the
  `MicrostructureSignal` interface's shape is not rejected at write time —
  only detectable later by a caller that runs `InterfaceRegistry.conforms`
  against it; conformance is advisory, not enforced at ingestion.
