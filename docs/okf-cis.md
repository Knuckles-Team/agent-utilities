# OKF-CIS concept IDs

OKF-CIS is the only accepted concept identifier standard across the ecosystem.
Every marker has the form:

```text
CONCEPT:<SLUG>-<PILLAR>.<domain>.<concept>[.<facet>...]
```

Use semantic lowercase kebab-case segments and the closed pillar/domain
vocabularies. For example:

```text
CONCEPT:AU-OS.governance.concept-hierarchy-standardization
CONCEPT:EG-KG.storage.redb
```

The grammar is implemented once in
`agent_utilities/governance/concept_hierarchy.py`. The concept registry is
generated from source markers, not edited by hand.

## Workflow

1. Choose a registered repository slug, pillar, and domain.
2. Reserve the complete ID with `agent-utilities concept reserve --id <ID>`.
3. Add the exact marker to source and its design evidence.
4. Run `python scripts/build_concepts_yaml.py`.
5. Run `python scripts/check_concepts.py` and
   `python scripts/check_domain_vocab.py`.
6. Rebuild RDF with `python scripts/build_concept_rdf.py --registry
   docs/concepts.yaml --out
   agent_utilities/knowledge_graph/ontology_concepts.ttl`.

`agent-utilities concept resolve --id <ID>` returns the parsed slug, pillar,
domain, semantic segments, OKF path, and IRI. It rejects every noncanonical
form.

## Governance files

| File | Purpose |
|---|---|
| `agent_utilities/governance/concept_hierarchy.py` | Grammar and projections |
| `agent_utilities/governance/domain_vocab.yaml` | Closed domain vocabulary |
| `agent_utilities/governance/slug_registry.yaml` | Unique repository slugs |
| `docs/concepts.yaml` | Generated exact-ID registry |
| `docs/concept_reservations.yaml` | Atomic exact-ID claims with opaque references |
| `agent_utilities/knowledge_graph/ontology_concepts.ttl` | Generated concept RDF |
