# Design Document: Schema Packs — a domain-configurable slice of the ontology, additive by default, with observe-only drift audit and pack-scoped OWL closure

CONCEPT:AU-KG.ontology.schema-packs ·
CONCEPT:AU-KG.ontology.pack-identity-rules ·
CONCEPT:AU-KG.ontology.pack-owl-closure ·
CONCEPT:AU-KG.ontology.schema-pack-lifecycle-audit

> `agent_utilities/models/schema_pack.py`, `models/schema_packs/__init__.py`,
> `agent_utilities/knowledge_graph/assimilation/identity_candidates.py`,
> `agent_utilities/models/schema_pack_audit.py`,
> `agent_utilities/knowledge_graph/core/owl_bridge.py` (pack-seeded reasoning).

## Decision — a SchemaPack is a domain-scoped subset of the 90+ type catalog, ADDITIVE by default so an agent focuses without losing the core

`schema_pack.py:4-31` states the provenance and the shape: inspired by the
gbrain schema-pack proposal, a `SchemaPack` "defines a domain-specific subset
of the Knowledge Graph ontology, allowing agents to focus on the node/edge
types relevant to their domain (research, biomedical, finance, etc.) without
loading the full 90+ type catalog." Two modes: `ADDITIVE` (default — pack
types layered ON TOP of the always-present core set) and `EXCLUSIVE` (only the
pack's types plus a protected core are active).

**The rejected alternative is EXCLUSIVE-only** — a pack that fully replaces
the active type set is simpler to reason about but forces every domain to
re-declare the core types it still needs (provenance, audit, capability
plumbing). Making ADDITIVE the default means a domain pack only has to name
what's NEW to it; `EXCLUSIVE` remains available for the narrower case where a
deployment genuinely wants to restrict the active surface (e.g. a constrained
agent that must never see finance types).

### Pointer — `CONCEPT:AU-KG.ontology.pack-identity-rules`

`identity_candidates.py:52-62`. Entity-resolution identity fields are
corpus-specific — "a CMDB id means something different in a ServiceNow pack
than in a research pack" — so `IdentityRule` is declared ON the active
`SchemaPack` (`identity_rules`/`identity_rules_for`) and threaded through
entity resolution as plain data, rather than the resolver hardcoding domain
knowledge. **The rejected alternative is one global identity-rule set** — the
module states it hardcodes exactly ONE generic fallback (`cmdb_id`/
`external_id`/`id`) for when no pack rule applies, "never a per-corpus rule" —
i.e. the fallback is deliberately minimal precisely because per-corpus rules
belong in the pack, not in this module.

### Pointer — `CONCEPT:AU-KG.ontology.pack-owl-closure`

`owl_bridge.py:718-739, 807-829`. A pack may declare its edge types as
transitive/symmetric/inverse OWL object-properties (`get_owl_closure_sets()`);
those are unioned into the reasoning engine's seed axioms so "the existing
promote→reason→downfeed cycle materialises multi-hop and inverse edges for
free — e.g. a research pack's `supports_belief` transitive chains and
`cites_source`/`cited_by` inverses." The always-on ARA forensic edges
(`CONCEPT:AU-KG.ontology.verified-by-implemented-by`, see
`.specify/design/kgo-e1-ara-forensic-grounding/design.md`) and harness
inverses are unioned in FIRST, then pack-declared ones layer on top — so
claim-grounding and harness reasoning work even with no schema pack active.
**The rejected alternative is per-pack reasoning code** — a hand-written
closure routine per domain. Instead, packs only DECLARE characteristics
(transitive/symmetric/inverse sets); the one engine-native-first,
Python-fallback reasoning path (`_lightweight_reasoning`) consumes whichever
sets are active, so "both paths [stay] in agreement" regardless of which pack
is loaded.

### Pointer — `CONCEPT:AU-KG.ontology.schema-pack-lifecycle-audit`

`schema_pack_audit.py:1-38`. When an EXCLUSIVE pack is active and a write
introduces a node/edge type outside the active set, that type is recorded as
a *candidate* rather than rejected — "the auditor is **observe-only**: it must
never reject or block a write" (mirroring gbrain's `candidate-audit.ts`).
**The rejected alternative is enforcement** — reject a write that introduces
an unmodeled type. That would make an EXCLUSIVE pack a hard schema wall,
breaking any write the pack's author didn't anticipate; observe-only instead
turns pack gaps into a discoverable `graph_configure(action="schema_candidates")`
review queue. The audit is also privacy-first by default: raw type names are
stored as a salted-free SHA-256 prefix + 4-char slug (`_redact`,
`schema_pack_audit.py:34-38`) "so a leaked audit log does not reveal a
deployment's private domain vocabulary," with `GRAPH_SCHEMA_AUDIT_VERBOSE=1`
as the explicit opt-in to store raw names.

## Risk Assessment

- **Blast Radius**: `models/schema_pack.py`, `models/schema_packs/`,
  `knowledge_graph/assimilation/identity_candidates.py`,
  `models/schema_pack_audit.py`, `knowledge_graph/core/owl_bridge.py`
  (reasoning-cycle seeding).
- **Backward Compatible**: Yes — no active pack behaves as the pre-pack system
  (full core catalog, generic identity fallback, ARA/harness-only closure).
- **Known weak point**: the audit's default redaction (hash + 4-char prefix)
  is a privacy/debuggability trade-off — a redacted candidate-type log is
  hard for an operator to act on without also having `GRAPH_SCHEMA_AUDIT_VERBOSE=1`
  set, which is an explicit but easy-to-forget opt-in when actually
  investigating a schema gap.
