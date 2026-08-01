# Design Document: Federation sync reuses the ONE ontology load path for every package-contributed .ttl, never a second load routine

CONCEPT:AU-KG.ontology.federation-runtime

> `agent_utilities/mcp/tools/ontology_tools.py` (`_sync_package_ontologies`),
> `agent_utilities/mcp/kg_server.py:1287-1299`.

## Decision — `sync_packages` iterates every discovered provider ontology and calls the EXISTING `OntologyLifecycle.load` per file, per-file failure-isolated

`_sync_package_ontologies` (`ontology_tools.py:52-68`) is explicit: it
iterates `resolve_provider_ontologies()` (installed distributions via
`importlib.metadata.entry_points()`) **plus**
`resolve_workspace_provider_ontologies()` (the separately-covered
`CONCEPT:AU-KG.ontology.workspace-provider-discovery` mechanism — see
`.specify/design/ontology-governed-evolution/design.md` for why workspace-mounted,
non-installed sibling repos also had to be discovered) and calls "the EXISTING
`OntologyLifecycle.load` per contributed `.ttl` (parse + SHACL-validate +
register + activate for reasoning). No new load logic — reuse the one path
`graph_ontology action='load'` and boot hydration already use." Shape files
(`shapes/*.ttl`) are explicitly skipped — they are validation constraints, not
loadable ontologies, so blindly loading every `.ttl` a package ships would
misclassify a SHACL shapes file as an ontology. Each file's load is
failure-isolated so one bad package contribution never blocks the rest of the
federation from loading. The action is reachable both at graph-os boot and
on-demand via `graph_ontology action='sync_packages'` / the REST twin
`POST /graph/ontology/sync-packages` (`kg_server.py:1287-1299`).

**The rejected alternative is a separate federation-specific load routine** —
plausible, since federation has different discovery inputs (multiple packages,
not one caller-supplied source) than a single `load()` call. The decision was
to keep discovery and loading strictly separated: `resolve_provider_ontologies`
+ `resolve_workspace_provider_ontologies` are pure discovery (find `.ttl`
files), and every one of them is fed through the identical single-file
`OntologyLifecycle.load` used everywhere else — so a federation-loaded
ontology gets exactly the same SHACL validation, idempotency-on-`(iri,version)`,
and fail-closed activation semantics as any manually loaded one, with zero
federation-specific parsing/validation code to keep in sync with the primary
path.

## Risk Assessment

- **Blast Radius**: every fleet package that ships an `ontology_providers`
  entry point or a workspace-mounted sibling repo's `.ttl` files; graph-os
  boot sequence.
- **Backward Compatible**: Yes — a package that contributes no ontology is
  unaffected; sync is additive.
- **Known weak point**: per-file failure isolation means a malformed `.ttl`
  from one package is silently skipped rather than failing the whole sync —
  correct for availability, but a broken package's ontology can go missing
  from the federation with only a log line marking it, not a visible error at
  the call site unless the caller inspects the per-file results.
