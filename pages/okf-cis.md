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

The grammar is implemented once in repository-manager's
`repository_manager/governance/concept_hierarchy.py` (moved there by OQ-3; this
repository's gates resolve it through `scripts/governance_tool.py`). The concept registry is
generated from source markers, not edited by hand.

## Workflow

1. Choose a registered repository slug, pillar, and domain.
2. For linked worktrees on one host, reserve the complete ID with
   `repository-manager-governance concept reserve --id <ID>`. For separate hosts, use the
   graph-os native reservation service; the CLI/file ledger is not globally
   atomic and must not be used as a fallback. Native callers must inject the
   authority-owned, versioned namespace/range policy.
3. Add the exact marker to source and its design evidence.
4. Run `python scripts/build_concepts_yaml.py`.
5. Run `python scripts/check_concepts.py` and
   `python scripts/check_domain_vocab.py`.
6. Rebuild RDF with `python scripts/build_concept_rdf.py --registry
   registry/concepts.yaml --out
   agent_utilities/knowledge_graph/ontology_concepts.ttl`.

`repository-manager-governance concept resolve --id <ID>` returns the parsed slug, pillar,
domain, semantic segments, OKF path, and IRI. It rejects every noncanonical
form.

## Governance files

| File | Purpose |
|---|---|
| `repository_manager/governance/concept_hierarchy.py` | Grammar and projections |
| `repository_manager/governance/domain_vocab.yaml` | Closed domain vocabulary |
| `repository_manager/governance/slug_registry.yaml` | Unique repository slugs |
| `registry/concepts.yaml` | Generated exact-ID registry |
| `registry/concept_reservations.yaml` | Generated compatibility projection of exact-ID claims |
| `docs/concept_lineage.yaml` | Hand-authored lineage record (parents, retirements, renames) |
| `repository_manager/governance/concept_reservation.py` | Cross-host authority port, native adapter, lifecycle, and read-only reconciliation |
| `agent_utilities/knowledge_graph/ontology_concepts.ttl` | Generated concept RDF |
