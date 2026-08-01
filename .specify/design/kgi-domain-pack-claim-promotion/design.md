# Design Document: Domain packs bring NEW vocabulary in without code; governed promotion is the ONE gate every candidate claim must survive to become fact

> `agent_utilities/knowledge_graph/domain_packs/domain_pack.py` (the pack
> manifest schema + DSL), `agent_utilities/knowledge_graph/ingestion/promotion.py`
> (governed claim promotion).

CONCEPT:AU-KG.ingest.domain-pack-framework ·
CONCEPT:AU-KG.ingest.mapping-dsl ·
CONCEPT:AU-KG.ingest.governed-claim-promotion

## Decision 1 — a domain pack is a versioned, installable artifact, not a code change

`domain_pack.py:1-35`.

**The rejected alternative, named directly and by name**: `SchemaPack`
(`models/schema_pack.py`) — the codebase's EXISTING nearby concept, deliberately
NOT reused for this. The docstring explicitly surveys it first: `SchemaPack`
activates a SUBSET of an already-fixed, hardcoded `RegistryNodeType`/
`RegistryEdgeType` `StrEnum` for retrieval tuning — "a lens over what the
engine already knows," resolved to exactly ONE process-wide active pack at a
time (a singleton switch, not coexistence). It has no version field, no
on-disk signed artifact, no install/remove lifecycle, and cannot declare a
genuinely new domain class — "you cannot add 'Runbook' to it without editing
the Python enum in code."

**The design chosen**: a domain pack is the OPPOSITE axis — how a genuinely
NEW, corpus-specific class vocabulary gets INTO the graph in the first
place, as a real on-disk, hash-verified, semver-versioned artifact (one
`domain_pack.yml` bundling an ontology extension, SHACL shapes, an
extraction/fragment schema, declarative mapping rules, and evaluation
cases). Several packs mount SIMULTANEOUSLY (`pack_loader.DomainPackRegistry`
— mirroring the fleet's existing "many domain `ontology_*.ttl` modules, one
engine authority" pattern), which `SchemaPack`'s singleton model structurally
cannot provide. The two compose rather than compete: once a pack's facts are
ingested, an operator wanting retrieval tuning over the new vocabulary still
reaches for `SchemaPack` on top.

### Pointer — `CONCEPT:AU-KG.ingest.mapping-dsl`

`domain_packs/dsl.py:1`, `domain_pack.py:269`. The declarative mapping
rules a domain pack bundles are expressed in a purpose-built DSL — turning
"a predictable-structure corpus into graph facts WITHOUT any custom code" (the
domain-pack framework's own stated goal). The rejected alternative is a
Python callback/plugin per domain pack — code the pack author would have to
write and the platform would have to trust/sandbox; the DSL keeps every
domain pack a pure DATA artifact (mapping rules, not executable code),
consistent with the framework's "no custom code" invariant.

## Decision 2 — a candidate claim survives ONE assembled, governed path to become a fact, never a bespoke shortcut

`CONCEPT:AU-KG.ingest.governed-claim-promotion` — `promotion.py:1-20`.

**The rejected alternative, named directly**: the advisory `SHACLValidator`/
`shacl_gate` phase used elsewhere in the codebase — explicitly called out as
NOT what this module uses, because it "fails OPEN on a missing shapes file or
an uninstalled `pyshacl`" and is "documented as advisory, not a second
security authority." Using that phase for governed promotion would mean a
missing dependency silently downgrades governance to a no-op.

**The design chosen**: one assembled path (Track A of the "universal-ingestion
program") a candidate claim must survive to become a materialized fact — SHACL
validation via the connector ingestion boundary's FAIL-CLOSED gate
(`envelope_ingest.validate_rows_against_shacl`, the public alias of
`_shacl_validate_rows` — the SAME fail-closed gate connectors already use,
not a second implementation), policy/classification checks
(`envelope_ingest.validate_envelope`, likewise fail-closed), PII handling,
deduplication, contradiction detection, per-pack confidence thresholds, and a
REAL steward-review hold. The module does not reimplement any of these
checks — it ASSEMBLES existing pieces into one governed sequence, so a claim
can't take a shortcut around any individual check by entering through a
different code path.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/domain_packs/*`,
  `agent_utilities/knowledge_graph/ingestion/promotion.py`,
  `agent_utilities/knowledge_graph/ingestion/envelope_ingest.py`,
  `agent_utilities/mcp/tools/claim_tools.py`.
- **Backward Compatible**: Yes — domain packs are opt-in installs; governed
  promotion is an assembly of already-existing fail-closed gates, not a
  new independent check with its own failure mode.
- **Breaking Changes**: None.
- **Known weak point**: domain packs mounting SIMULTANEOUSLY (the explicit
  design goal over `SchemaPack`'s singleton) means two packs COULD declare
  conflicting mappings for overlapping source shapes — nothing in the
  framework as documented here resolves a cross-pack mapping conflict at
  install time; it would surface only as inconsistent ingestion behavior.
