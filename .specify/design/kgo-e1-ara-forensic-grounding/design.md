# Design Document: Agent-Native Research Artifact claims are reasoned OWL interfaces, not opaque text — grounding/implementation chains are always-on

CONCEPT:AU-KG.ontology.verified-by-implemented-by

> `agent_utilities/knowledge_graph/ontology/interfaces.py:841-863`
> (`VerifiableClaim`), `agent_utilities/knowledge_graph/core/owl_bridge.py:
> 642-657` (`ARA_TRANSITIVE_EDGES`/`ARA_INVERSE_EDGES`),
> `agent_utilities/gateway/research_api.py`.

## Decision — every ARA claim is a `VerifiableClaim` ontology Interface with mandatory grounding, and the grounding chain is always-on OWL reasoning, independent of any schema pack

`interfaces.py:841-846` states why an Agent-Native Research Artifact's claims
are modeled as ontology interfaces rather than left as prose: "An ARA is a
4-layer artifact whose /logic claims are verifiable and grounded; making these
ontology interfaces (`owl:Class` + SHACL `NodeShape`) lets the reasoner
extrapolate over claim↔evidence↔code↔ecosystem **rather than treat the
artifact as opaque text**." The `VerifiableClaim` interface
(`interfaces.py:847-863`) mirrors "the ARA Seal's 'every claim is backed'
invariant as a shape contract" — a claim must be `grounded_in` evidence and,
where executable, `implemented_by` a code spec.

`owl_bridge.py:651-657` names the reasoning payoff directly: "ARA
forensic-edge OWL object-property characteristics, **always on**" —
independent of any active schema pack, so every reasoning cycle chains a
claim's grounding (`claim -grounded_in-> evidence -grounded_in-> source ⟹
claim -grounded_in-> source`, `ARA_TRANSITIVE_EDGES = {"grounded_in"}`) and
materializes the evidence→claim `supports` inverse
(`ARA_INVERSE_EDGES = {"grounded_in": "supports"}`) — "letting reasoning
relate a /logic claim to the ecosystem code/services that substantiate it."

**The rejected alternative — and the reading that led the machine triage tool
to flag this id as a "retire" candidate (a bare slugified prose fragment) — is
treating a claim's grounding as two separate, non-reasoned foreign-key-style
fields** (`verified_by`, `implemented_by`) with no transitive/inverse
semantics: a caller would have to manually walk multi-hop grounding chains
and manually maintain the evidence-side inverse, and nothing would relate a
claim to code it wasn't DIRECTLY linked to, even when a chain of grounding
evidence connects them. Making the edges OWL object properties with
always-on transitive/inverse characteristics means that reasoning, not
application code, walks the chain — and doing it unconditionally (not gated
behind a schema pack) means claim-grounding works for every deployment, even
one with no domain-specific pack loaded at all. `research_api.py:4-14`
confirms this is a production surface, not a prototype: a "granular, typed
REST surface" that dispatches through the SAME in-process `research_artifact`
MCP tool the gateway and MCP paths share, exposing `reason`/`compile`/
`review`/`capture` over the whole ecosystem.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/interfaces.py`
  (`VerifiableClaim` and sibling ARA interfaces), `knowledge_graph/core/owl_bridge.py`
  (`ARA_TRANSITIVE_EDGES`/`ARA_INVERSE_EDGES`), `gateway/research_api.py`,
  every schema pack's own closure sets (layered on top of these, not replacing
  them — see `.specify/design/kgo-e1-schema-packs/design.md`).
- **Backward Compatible**: Yes — always-on reasoning characteristics, additive
  to any pack-declared ones.
- **Known weak point**: transitive grounding means a claim can inherit
  "groundedness" through a long, indirect evidence chain — the reasoner
  cannot by itself distinguish a short, strong grounding chain from a long,
  weak one; both materialize the same `grounded_in` inference.
