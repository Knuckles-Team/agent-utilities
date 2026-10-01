# AU semantic client and generated contract

**ID:** AU-SEMANTIC-001 · **Owner:** agent-utilities · **Delivery:** SPECIFIED, acceptance NOT AUDITED.
**Items:** AU-SEMANTIC-R001, AU-SEMANTIC-R002 (AU follow-up), AU-SEMANTIC-R003, AU-SEMANTIC-R004–AU-SEMANTIC-R008, AU-SEMANTIC-R009–AU-SEMANTIC-R026. See [requirements.md](requirements.md) for the definition of every requirement ID, including AU-SEMANTIC-R027 and AU-SEMANTIC-R028, and [status.json](status.json) for delivery state and evidence. Graph-owned portions of these items are dependencies; this spec states the AU cut and verification in full.

## Outcome

AU runs model-backed agents and compiles context by calling one generated, version-pinned epistemic-graph client. AU has no RDF/OWL/SHACL runtime, ontology lifecycle, graph schema writer, query engine, retrieval engine, deterministic ingester, durable graph job, or durable memory authority.

## Requirements

1. Replace local ontology and shape files with EG-owned source identities. AU supplies an IRI and typed request; it does not parse, mutate, validate or hash RDF locally. Reject unknown source/digest and conflicting declarations before any graph mutation.
2. `knowledge_graph/ontology/lifecycle.py`, `ontology_integrity.py`, pack/schema loaders and extraction-schema readers become thin generated-client calls or disappear. AU tests use served EG validation, never an optional `pyshacl` skip.
3. Remove runtime and test imports of `rdflib`, `pyshacl`, `owlrl`, and `owlready2`; remove their dependency declarations when no AU behavior needs them. A static check covers all first-party runtime and tests.
4. AU's graph/session, tenancy, work, ontology, ingest, schema drift, retrieval and memory operations import generated EG request/result types from one pinned release digest. No second DTO, digest implementation, graph parser or local durable fallback.
5. Preserve agent-specific context compilation and candidate-claim generation. Claims include tenant, graph, source identity, RunSpec digest and provenance; EG alone validates and commits them.
6. Before deleting any legacy entry point, enumerate its callers and prove the public path reaches the generated method with equivalent authorization, idempotency, errors and receipt semantics. Unknown legacy records are quarantined, not silently promoted.
7. Distinguish read projections from authority: an AU cache may hold scoped, expiring context for one run, but EG controls invalidation and durable truth.

## Acceptance

No forbidden semantic imports or owned files remain in AU; generated-client conformance, positive and negative served tests, old-caller closure and normal quality gates pass at the same merged head. Built source on a lane is not acceptance.
