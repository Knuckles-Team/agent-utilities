# Design Document: One CI gate enforces VALID + CONNECTED + DOCUMENTED over the whole ontology library, catching exactly the two failure modes that already happened once

CONCEPT:AU-KG.ontology.anti-sprawl-gate ·
CONCEPT:AU-KG.ontology.package-owned-ontology

> `scripts/check_ontology.py`.

## Decision — a single sweep gate checks 7 invariants across bundled AND package-federated ontology modules, exit 1 on ANY violation

`check_ontology.py:1-35` states the gate exists to prevent a recurrence:
"keep the ontology library from rotting into the state we just fixed (a
divergent duplicate `core/ontology.ttl` the reasoner silently loaded instead
of the real one, and ~17 domain modules that no canonical file referenced)."
The seven checks split into three groups: **VALID** (parses as Turtle; no two
files declare the same `owl:Ontology` IRI; the merged ontology survives
OWL-RL closure; every SHACL shapes file loads and runs), **CONNECTED** (every
domain module is `owl:imports`-referenced by the canonical `ontology.ttl` —
"an unreferenced module is a build failure, not a warning"; every in-namespace
`owl:imports` target resolves to a present file, external standard
vocabularies excepted), and **DOCUMENTED** (every `.ttl` on disk is listed in
`docs/architecture/ontology_library.md`).

**The rejected alternative is catching these failure modes individually, at
their own point of origin** — a duplicate-IRI check here, a dangling-import
check there. The gate is deliberately a single sweep because the two real
incidents it names (a silently-loaded duplicate ontology, and 17 orphaned
modules) were each invisible to any check narrower than "walk the whole
library and verify every file is both valid and reachable" — a per-file check
run only on the file being edited would never have caught either.

### Pointer — `CONCEPT:AU-KG.ontology.package-owned-ontology`

`check_ontology.py:250-308`. The gate's domain-module sweep is not limited to
files bundled in this wheel: `_provider_ttls()` reuses the SAME federation
read-path resolver the runtime uses (XDG-first) to sweep "package-contributed
ontologies from the same place the runtime does," and `_federated_iris()`
tracks "known package-owned ontology IRIs" so the canonical bundle's
`owl:imports` edge to a package-owned module is recognized as "a superset
no-op, not a dangling reference" even when that package isn't installed in the
current environment — the alternative (treating an uninstalled provider's
import as dangling) would make `check_ontology` fail in every environment that
doesn't have every fleet package installed, which is not a realistic bar for
a base install. Both federation lookups are explicitly failure-isolated:
"federation is additive; base gate must not break" — an exception resolving
providers degrades to an empty superset rather than failing the whole gate.

## Risk Assessment

- **Blast Radius**: every `*.ttl` file bundled in `agent_utilities/knowledge_graph/`
  or contributed by an installed/federated ontology-provider package; CI.
- **Backward Compatible**: Yes — a validation gate, not a runtime behavior
  change.
- **Known weak point**: `_federated_iris()`'s failure-isolation means a
  genuinely broken federation registry (not just "package not installed")
  silently reduces the CONNECTED check's federated coverage to zero rather
  than failing loudly — a real dangling import introduced alongside a broken
  registry could pass the gate undetected.
