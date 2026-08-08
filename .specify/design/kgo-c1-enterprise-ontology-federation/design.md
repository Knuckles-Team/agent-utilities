# Design Document: `owl:imports` resolves through a layered fallback chain, so a moved/federated module never dangles

CONCEPT:AU-KG.ontology.enterprise-ontology-distribution ·
CONCEPT:AU-KG.ontology.federation-resolution

> `agent_utilities/knowledge_graph/core/ontology_loader.py`.

## Decision — "inherit from central, extend locally": resolve `owl:imports` from HTTP/SPARQL/file, cached, recursive

`CONCEPT:AU-KG.ontology.enterprise-ontology-distribution`

`ontology_loader.py:2-15` names the pattern this module enables directly:
"Modular Ontology Federation ... Resolves `owl:imports` declarations in TTL
files, fetching remote ontologies from HTTP URLs or SPARQL endpoints and
merging them into the local rdflib graph. Enables the 'inherit from central,
extend locally' enterprise pattern." It supports file-based (`file://`),
HTTP/HTTPS remote imports, TTL-based caching with a configurable TTL, and
recursive import resolution.

**The rejected alternative is bundling every enterprise ontology module
in-tree** — the simpler design, but one that forces every consumer to vendor
and manually sync a copy of any central ontology it depends on. Resolving
`owl:imports` at load time instead lets a module declare a dependency on the
canonical `knuckles.team/kg` base (or another module) and have it actually
fetched/merged, with a short bounded remote-absence cache (60s,
`_REMOTE_ABSENCE_CACHE_TTL_SECONDS`, `ontology_loader.py:42`) so a real 404/410
doesn't trigger a fresh network round-trip on every import resolution within
that window — while auth/policy/TLS/transport/config failures are deliberately
never cached, so fixing those takes effect on the very next call.

### Pointer — `CONCEPT:AU-KG.ontology.federation-resolution`

`ontology_loader.py:259-330`. The general HTTP-import resolver has a
specifically-decided fallback chain for the `knuckles.team/kg/<X>` namespace:
(1) look for `ontology_<X>.ttl` as a sibling of the importing file; if
missing, (2) check the bundled `knowledge_graph/` directory directly — "so a
federated module's import of the base `.../kg` ontology resolves locally
instead of hitting the network"; if still missing, (3) fall back to
`_federated_path_for`, which searches the discovered ontology-provider
directories (`resolve_provider_ontologies()`), matching first by the
candidate file's own declared `owl:Ontology` IRI (authoritative) and only then
by a `<suffix>.ttl`/`ontology_<suffix>.ttl` filename match. The comment states
why step (3) exists: "once modules move into fleet-package wheels" a sibling
file no longer exists next to the importing ontology, so "the canonical
bundle's import of a moved module (e.g. servicenow) still resolves and the
gate's 'no dangling import' check stays green." Failure-isolated throughout:
an absent federation registry or an unmatched suffix returns `None` rather
than raising, so one unresolvable import never blocks the rest of a load.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/core/ontology_loader.py`, every TTL file
  with an `owl:imports` declaration, `ontology_federation.py`
  (`resolve_provider_ontologies`).
- **Backward Compatible**: Yes — file-sibling resolution (the pre-federation
  behavior) is tried first; federation is an additive fallback.
- **Known weak point**: `_federated_path_for`'s filename-match fallback
  (`ttl.stem in (suffix, f"ontology_{suffix}")`) is a heuristic used only when
  the IRI-match fails — two providers shipping same-named-but-different
  ontology files could resolve to the wrong one if the IRI-declared match
  isn't found first.
