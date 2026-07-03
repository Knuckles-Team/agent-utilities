# Concept Hierarchy — the 3-level `NS-<pillar>.<concept>.<segment>` grammar

> Status: **reviewable proposal** (W4 / B5). Design + migration tooling + gate/CLI
> teach are landed on `feat/w4-concept-hierarchy`; the fleet-wide cutover
> (`--apply`) is **NOT** run. Flat ids remain valid forever.
> Concept: `CONCEPT:OS-5.76`.

## Why

Concept ids are the ecosystem's traceability spine: every `CONCEPT:<id>` marker
in code ties an implementation to a documented capability, and the parity gate
keeps docs honest with code. But the id scheme drifted into **two incompatible
shapes**:

| Shape | Example | Namespaces using it |
|-------|---------|---------------------|
| flat `NS-<n>` | `EG-321`, `EG-010`, `EE-033`, `ML-011` | epistemic-graph (EG) + a few AU pillars |
| dotted `NS-<pillar>.<concept>` | `KG-2.312`, `ORCH-1.105`, `AHE-3.49`, `OS-5.72` | KG, ORCH, ECO, OS, AHE |

The flat shape has no pillar, so the registry generator treats *each* flat id as
its own "pillar" (`EG-009`, `EG-010`, `EG-321` each became a bogus pillar). That
breaks any grouping/roll-up query and blocks **cross-project interweaving** (an
EG concept that extends a KG concept can't be modeled).

B5 standardizes on ONE grammar, models concepts as first-class KG nodes, and
does it **non-breakingly** — every string ever written stays a valid alias.

## The grammar

```
NS-<pillar>.<concept>[.<segment>]
```

* **`NS`** — a namespace. Two families:
  * **Project/pillar namespaces** (the cross-project framework/engine pillars:
    `EG ↔ KG ↔ ECO ↔ OS ↔ AHE ↔ ORCH`, plus `EE ML CE SAFE LGC CTX UTIL`) adopt
    the dotted grammar. Curated in
    `agent_utilities/governance/concept_hierarchy.py::PROJECT_NAMESPACES`, and a
    namespace is *also* promoted to project-scope at runtime if it is observed
    carrying any 2+-segment id.
  * **Package namespaces** (letters-only local registries: `KEY OKTA JELLYFIN
    HUB SX GBOT …`) keep their existing `PKG-NNN` form untouched — they are a
    separate, legitimate scheme and are recognized + passed through.
* **`<pillar>`** — the coarse grouping index inside the namespace (`2` of `KG-2`).
* **`<concept>`** — the concept index inside the pillar.
* **`<segment>`** — an OPTIONAL third level for finer subdivisions minted *going
  forward* (e.g. `EG-3.31.20`). Absent ⇒ implicit `.0`. **Never** auto-assigned
  to a legacy id.

Example (from the spec): `EG-3.31.20` = namespace `EG`, pillar `3`, concept `31`,
segment `20`.

## Flat → dotted mapping rules (deterministic + reversible)

The single implementation is `concept_hierarchy.parse_concept_id`. Given a raw id:

1. Split into `NS` (leading `[A-Z]+`) and the dot-separated remainder.
2. Classify `NS` (project vs package) via `classify_namespace`.
3. **Package NS** → passthrough. `canonical == raw`. Flagged `package-scoped`.
4. **Project NS**, by segment count:
   * **1 segment** `NS-<n>` (legacy flat, e.g. `EG-321`): it carries no pillar.
     Look up `PILLAR_MAP[(NS, n)]`; if present use it, else assign the reserved
     **legacy pillar `0`** and flag `legacy-pillar-0`. → `EG-321` ⇒ `EG-0.321`.
   * **2 segments** `NS-<p>.<c>` (e.g. `KG-2.312`): pillar `p`, concept `c`,
     no segment. Already grammar-compliant → `canonical == raw` (no rewrite).
   * **3 segments** `NS-<p>.<c>.<s>` (e.g. `EG-3.31.20`): fully canonical.
   * **>3 segments**: taken as `p.c.s` + flagged `over-segmented` for review.

### How pillars are assigned

* Dotted ids already state their pillar — kept verbatim.
* Legacy flat ids default to pillar `0` ("unclassified/legacy"), which is
  deterministic and reversible (`EG-0.321` → drop pillar `0` → `EG-321`). A
  reviewer curates real pillars by populating `PILLAR_MAP` *before* `--apply`;
  the dry-run report lists every id that needs curation, grouped by namespace.

### How segments are assigned

Never automatically. The third level is reserved for *new* work that subdivides
an existing concept; it is minted on demand via the allocator (below). No legacy
id gains a segment during migration.

## Alias strategy — flat stays valid forever

Canonicalization is **additive, never destructive**:

* `ConceptId.aliases` always includes both the raw string and the canonical
  dotted string, so `EG-321` and `EG-0.321` resolve to the **same** concept.
* `build_alias_index(ids)` builds `{alias → canonical}` for resolution.
* The **parity gate** (`scripts/check_concepts.py`) accepts a code marker if its
  raw id *or* its canonical form is registered — so a repo can migrate markers
  at its own pace, mixing flat and dotted, and stay green.
* The **registry** (`docs/concepts.yaml`) now carries additive `dotted`,
  `aliases`, `canonical_pillar`, and `needs_curation` keys per concept; existing
  consumers that read only `id` are unaffected.
* An `--apply` rewrite records the flat id inline (`CONCEPT:EG-0.321  # alias:EG-321`)
  and is idempotent (re-running is a no-op) and reversible (the alias resolves both).

## Cross-project interweaving — the edges

Derived deterministically by `derive_part_of_edges` and modeled in the ontology:

* **`partOf`** (mereology, transitive) — every concept is `partOf` its pillar;
  every pillar is `partOf` its namespace; a package concept is `partOf` its
  namespace directly. This gives free roll-ups (`concept → pillar → namespace`).
* **`extends`** — a concept refines/builds on another, *possibly across
  projects* (an EG concept extending a KG concept). Modeled as a subproperty of
  `dependsOn` (an extension necessarily depends on what it extends). Declared,
  not machine-guessed; the dry-run reports candidate cross-references for a human
  to confirm.
* **`dependsOn`** — generic concept-to-concept dependency (reuses the existing
  canonical property).
* **`aliasOf`** — links a legacy/alternate id node to its canonical concept.

## Ontology model (extends the canonical `ontology.ttl` — no new file)

Added to `agent_utilities/knowledge_graph/ontology.ttl` (validated by
`scripts/check_ontology.py`):

| Term | Kind | Notes |
|------|------|-------|
| `:Pillar` | `owl:Class` ⊑ `:Concept` | a coarse grouping of governed concepts |
| `:GovernedConcept` | `owl:Class` ⊑ `:Concept` | a first-class concept-id node |
| `:extends` | `owl:ObjectProperty` ⊑ `:dependsOn` | cross-project refinement |
| `:aliasOf` | `owl:ObjectProperty` | legacy id ⇄ canonical concept |
| `:flatId` | `owl:DatatypeProperty` | the permanent flat alias literal |
| `:dottedId` | `owl:DatatypeProperty` | the canonical dotted id literal |

`:partOf` and `:dependsOn` already exist canonically and are reused. Governed
concepts are typed as `:Concept` (via subclass), satisfying "first-class
`:Concept` nodes" while `:Pillar` groups them — this deliberately avoids
overloading the plain `:Concept` (ArchiMate/SKOS knowledge unit) it specializes.

## Tooling

| Tool | Role |
|------|------|
| `agent_utilities/governance/concept_hierarchy.py` | the ONE parse/classify/canonicalize/alias/edge implementation |
| `scripts/migrate_concepts_hierarchy.py` | `--dry-run` (default) fleet mapping report; `--apply` (gated) in-place rewrite |
| `scripts/build_concepts_yaml.py` | emits additive `dotted`/`aliases`/`canonical_pillar`/`needs_curation` |
| `scripts/check_concepts.py` | parity gate — accepts flat OR dotted via alias resolution |
| `agent_utilities/governance/concept_allocator.py` | `MARKER_RE` accepts 3-level; mints 3rd-level segments (`KG-2.312.<seg>`) |
| `agent-utilities concept resolve --id <flat\|dotted>` | canonicalize any id → dotted + aliases + flags |

## To apply (NOT done in this proposal)

1. Review `reports/w4-concept-hierarchy-dryrun.md`.
2. Populate `PILLAR_MAP` for the `needs-curation` (legacy pillar 0) ids to give
   EG (and friends) real pillars.
3. Per repo: `python scripts/migrate_concepts_hierarchy.py --apply --root <repo>`.
4. Re-gate each repo: `check_concepts`, `check_ontology`, `ruff`.
5. Land per-repo (each repo owns its own markers + ledger).
