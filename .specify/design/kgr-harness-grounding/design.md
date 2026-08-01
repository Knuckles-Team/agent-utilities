# Design Document: The ONE ontology-driven KG is the cross-team substrate HarnessX's own paper says co-evolution needs

CONCEPT:AU-KG.retrieval.harness-grounding

> `agent_utilities/harness/harness_grounding.py`, second site
> `agent_utilities/protocols/source_connectors/connectors/mcp_tool.py:112-115`.

## Decision — ground a harness variant in evidence from the WHOLE connector fleet via the transitive `grounded_in` chain, not a single benchmark verifier

`harness_grounding.py:1-11` states the rejected alternative in the paper's
own terms: "HarnessX evolves over benchmark verifiers only, and its
co-evolution needs one team owning harness+model ('impractical without
cross-team coordination')." **Our ONE ontology-driven KG is offered as
exactly that cross-team substrate**: a harness variant is grounded in
evidence from the whole connector fleet via the `grounded_in` chain, which is
transitive (KG-2.80, already reasoned by `owl_bridge`) — so `ground_variant`
only has to assert direct `grounded_in` edges from a variant to its immediate
evidence ids (traces/test-results/metric-reports); the engine's own OWL
reasoning materializes the transitive variant → source chain, rather than the
module having to walk and assert every transitive hop itself. Sealing
(`seal_variant`/`seal_level_for`) maps a held-out certification result to an
L1/L2/L3 seal level by comparing the certified confidence-interval lower
bound against the human baseline with a margin — L3 for a clear margin, L2
for certified without one, L1 otherwise — and the seal node is itself grounded
back to the variant it certifies, so a seal is traceable evidence, not a bare
label.

**The rejected alternative, spelled out directly**: a siloed per-team harness
whose co-evolution with its model requires that one team to own both — which
"is impractical without cross-team coordination" per the paper this module
responds to. Grounding in the shared KG instead means "reasoning chains
harness-edit → dimension → service → node, something a siloed per-agent
harness cannot" — because the harness's behavioral dimensions link to the
live `ecosystem_topology` services they touch, a connection that only exists
because the harness and the service topology share one graph rather than
living in separate, team-owned systems.

The second site (`mcp_tool.py:112-115`) is the connector-side half of the
same decision: harness-run traces are ingestible from ANY fleet
evolution/governance server via a generic `"harness-runs"` connector
declaration (`server`/`tool`/`action`/`records_path`), extensible by adding a
new server entry rather than writing a new bespoke ingestion path per
governance server — the same fleet-as-evidence-substrate principle applied to
where the evidence itself comes from.

## Risk Assessment

- **Blast Radius**: `harness_grounding.py`, `superhuman_gate.py`
  (`CertificationResult`), `mcp_tool.py`'s connector registry.
- **Backward Compatible**: Yes — grounding/sealing are additive edges over
  existing harness-variant and certification data.
- **Known weak point**: `seal_level_for`'s L1/L2/L3 thresholds are fixed
  margin constants (`>= 0.1` for L3), not derived from a calibration study of
  what margin actually predicts durable superiority — a variant just above or
  below the 0.1 line receives a materially different seal level for a
  difference in measured margin that may not itself be statistically
  meaningful.
