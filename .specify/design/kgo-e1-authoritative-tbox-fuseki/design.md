# Design Document: The Fuseki publish tick is opt-in and separately-gated from the endpoint configuration itself

CONCEPT:AU-KG.ontology.authoritative-tbox

> `agent_utilities/knowledge_graph/core/ontology_publisher.py:1-21`
> (`publish_ontology_to_fuseki`), `agent_utilities/core/config.py:3684-3705`
> (`kg_fuseki_publish`, `kg_fuseki_endpoint`).

## Decision — TWO independent gates: whether an endpoint is configured, and whether the daemon actually publishes to it

`ontology_publisher.py:16-21` states the mechanism: `publish_ontology_to_fuseki`
"collects every bundled `ontology*.ttl` module into one rdflib graph and
pushes it through `OntologyPublisher.push_to_jena_fuseki`, so the engine's
maintenance scheduler (`fuseki_publish` tick, gated by `KG_FUSEKI_PUBLISH`)
keeps an optional enterprise Fuseki deployment in sync with the evolving
authoritative ontology." `config.py:3684-3692` names the deliberate split
directly: "the *publish tick* stays off by default (`KG_FUSEKI_PUBLISH`)
because writing to Fuseki is an opt-in action even when an endpoint is
reachable; the *endpoint itself* has no environment-specific default." A
deployment with no Fuseki never flips `kg_fuseki_publish` on, and the
`jena_fuseki` backend is never selected unless requested.

**The rejected alternative is one combined flag** — publish automatically
whenever an endpoint happens to be configured (e.g. inherited from a shared
config template). That would make a Fuseki write a side effect of
configuration presence rather than a deliberate choice, which is unsafe for
any deployment where "an endpoint is reachable" and "we want THIS deployment
pushing the authoritative TBox to it" are genuinely different facts (e.g. a
shared/staging Fuseki instance another team owns). Splitting the two gates
means an operator can point at a real endpoint for read/inspection purposes
without that alone causing writes.

`kg_fuseki_endpoint` (`config.py:3694-3705`) is documented as "THE canonical
Fuseki endpoint — the single field every Fuseki reader resolves through": the
publish tick, `publish_ontology_to_fuseki`'s fallback, the `fuseki`-kind
SPARQL smoke query, and the `jena_fuseki` query backend all resolve the SAME
field, with an explicit per-call `endpoint=`/`jena_fuseki_url=` override
available — one canonical source, not four independently-configured ones that
could disagree.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/core/ontology_publisher.py`,
  `core/config.py` (`kg_fuseki_*` fields), `engine_tasks._tick_fuseki_publish`,
  `backends/sparql/jena_fuseki_backend.py`.
- **Backward Compatible**: Yes — both flags default to off/empty; a
  deployment that never sets either is fully unaffected.
- **Known weak point**: because the endpoint has "no environment-specific
  default," an operator who sets `KG_FUSEKI_PUBLISH=1` without also setting
  `KG_FUSEKI_ENDPOINT` gets a publish tick that runs against an empty
  endpoint — the split-gate design assumes both are set together, and nothing
  enforces that pairing at config-validation time.
