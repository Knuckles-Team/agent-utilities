# Design Document: The harness-evolution gate reasons over a formal OWL ontology, not per-task scores

CONCEPT:AU-KG.ontology.harness-gate ·
CONCEPT:AU-KG.ontology.harness-ontology ·
CONCEPT:AU-KG.ontology.owl-bridge

> `agent_utilities/harness/harness_gate.py`,
> `agent_utilities/knowledge_graph/core/owl_bridge.py` (promotable
> node/edge-type tables + `HARNESS_INVERSE_EDGES`),
> `agent_utilities/knowledge_graph/shapes/harness.shapes.ttl`.

## Decision — model harness-evolution facts as RDF and validate with SHACL, so sub-threshold coupling is caught before the tipping point

`harness_gate.py:1-9` states the motivating gap directly: the formal seesaw
HarnessX (arXiv:2606.14249)'s per-edit pass@2 gate "cannot see *sub-threshold
coupling*" — its own τ³-Bench Telecom run shipped five same-dimension edits
whose ACCUMULATED coupling caused a tipping-point −14% regression that the
paper's gate never detected. `build_evolution_graph` (`harness_gate.py:49-100`)
answers this by modeling edits/dimensions/hooks/processors/pathologies as RDF
individuals (`HarnessEdit`, `HarnessDimension`, `HarnessHook`, `Processor`,
…) and validating the resulting graph against concentration / no-regression /
pathology SHACL shapes — "the gate detects and blocks concentration **before**
the tipping point, reasoned over the harness ontology rather than read off
per-task scores."

**The rejected alternative is HarnessX's own per-edit gate** — evaluate each
edit in isolation against pass@2 and ship if it passes. That gate is
structurally blind to accumulation: five edits that each individually clear
the bar can jointly tip the system into regression, and nothing in a per-edit
check can see that, because no single edit crossed a threshold. Reasoning over
a graph of ALL edits (not one) is what makes concentration visible.

`CONCEPT:AU-KG.ontology.harness-ontology` names the "harness foundry: evolvable
harness as reasoned-over ontology" (`owl_bridge.py:80`) — the specific
node-type additions (`harness`, `processor`, `harness_dimension`,
`harness_edit`, `harness_variant`, `harness_pathology`) and edge-type additions
(`targets_dimension`, `has_variant`/`variant_of`, `applies_edit`,
`exhibits_pathology`, `mitigates_pathology`/`mitigated_by`,
`causes_regression`, `confirms_fix`) that make the harness a first-class
promotable ontology domain rather than opaque application data. `harness-gate`
narrows to the SHACL-gate side of the same architecture: the two read-only
lifecycle hooks (`step_end`, `task_end`, `harness_gate.py:46`) and the
substitution-algebra facts (`editOperation`, `atHook`, `modifiesField`) stamped
into the same data graph so the hook-contract shape is self-contained
(`harness_gate.py:43-45, 75`).

`CONCEPT:AU-KG.ontology.owl-bridge` marks the reasoning payoff that makes this
whole approach work: `HARNESS_INVERSE_EDGES` (`owl_bridge.py:664-667`) always
materializes the `has_variant ↔ variant_of` and `mitigates_pathology ↔
mitigated_by` inverses. The comment states it plainly: "This is what makes
HarnessX's 'operational mirror' a formal operational **ontology**
(CONCEPT:AU-KG.ontology.owl-bridge): reasoning materialises the … inverses so
the SHACL concentration/no-regression/pathology gate (AHE-3.53) queries a
COMPLETE, INFERRED graph rather than relying on the paper's RL↔symbolic
analogy" (`owl_bridge.py:661-663`) — i.e. the rejected alternative here is
requiring every caller to write both directions of a relationship by hand;
instead the OWL reasoner's inverse-property inference guarantees completeness.

## Risk Assessment

- **Blast Radius**: `harness/harness_gate.py`, `knowledge_graph/core/owl_bridge.py`
  (promotable-type tables, `HARNESS_INVERSE_EDGES`),
  `knowledge_graph/shapes/harness.shapes.ttl`.
- **Backward Compatible**: Yes — an additive reasoning/gating layer over
  harness-evolution facts that must already be supplied as structured dicts.
- **Known weak point**: the gate is only as complete as the concentration/
  no-regression/pathology SHACL shapes actually encode — a genuinely new
  pathology shape not yet modeled in `harness.shapes.ttl` would not be caught,
  even though the underlying RDF graph contains the raw facts needed to detect
  it.
