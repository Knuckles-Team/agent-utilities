# Design Document: A Knowledge Pack ships CONTENT (data instances); a Schema Pack ships STRUCTURE — kept as two separate models, not one

CONCEPT:AU-KG.ontology.knowledge-packs

> `agent_utilities/models/knowledge_pack.py`.

## Decision — `KnowledgePack` is a versioned, idempotently-seedable bundle of actual Node/Edge data, deliberately separate from `SchemaPack`'s structural type-subset concern

`knowledge_pack.py:4-14` draws the line explicitly: "A `KnowledgePack` defines
a set of actual data instances (Nodes and Edges) that can be seeded into the
Knowledge Graph as a cohesive bundle. While a `SchemaPack` defines the
*structure* (which ontology types are allowed), a `KnowledgePack` provides the
*content* (e.g., specific papers, repositories, entities)." Knowledge Packs
use deterministic ID generation "to ensure idempotent imports, allowing them
to be easily shared, versioned, and injected into different environments" —
`KnowledgePackImporter.load()` reads a YAML bundle and `seed_into_kg()` seeds
it into a live engine.

**The rejected alternative is one combined pack type covering both structure
and content** — plausible since both are "domain-specific presets," but
collapsing them would force every content bundle to also carry (or
re-declare) type-scoping decisions, and every schema decision to potentially
carry data. Keeping them separate means a `SchemaPack` can be applied WITHOUT
any specific data (defining what's allowed before anything exists), and a
`KnowledgePack` can be seeded into a graph whose schema was scoped by a
completely independent `SchemaPack` decision — the two compose rather than
being coupled. Deterministic IDs are the specific mechanism that makes a
pack re-injectable across environments without duplicate-node drift: seeding
the same pack twice (dev, then staging, then prod) produces the SAME node
identities each time rather than three divergent copies.

## Risk Assessment

- **Blast Radius**: `models/knowledge_pack.py`, `presets/*.yaml` bundles,
  any engine a pack is seeded into.
- **Backward Compatible**: Yes — an additive seeding mechanism; a deployment
  that never loads a knowledge pack is unaffected.
- **Known weak point**: idempotency depends on deterministic ID generation
  staying stable across pack VERSIONS — a pack revision that changes how an
  entity's id is derived (rather than just its content) would silently create
  a duplicate node instead of updating the existing one, since nothing here
  detects an id-derivation change as distinct from a content change.
