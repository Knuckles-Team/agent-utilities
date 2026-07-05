# OKF round-trip + openwiki, standardized on skill-graphs

Track 2 of the OKF / knowledge-catalog assimilation program. Extends our mature
OKF stack (`distillation/okf_bundle.py`, `governance/concept_hierarchy.py`, the
`protocols/source_connectors/` framework) rather than inventing a parallel format.
The user-mandated invariant: **OKF bundles ARE the serialization of skill-graphs
distilled from the KG** — every path routes through
`distillation/skill_graph_pipeline.py` → `okf_bundle.write_okf_conformance()`.

## Extend-Before-Invent mapping

Each new capability is an extension of an existing pillar object, not a new service.

### CONCEPT:AU-KG.research.okf-overlay-mode

Thin-overlay concept mode in `okf_bundle.py` (the openalgo pattern). An overlay
concept node = `type` + `resource` (the source URI) + cross-links, with a body that
defers to the source ("edit the source, not this file"). Lets the 80+ fleet repos'
existing markdown be catalogued as OKF concepts **without content duplication**.
Also carries the **permissive-consumer** half of the OKF standard (SPEC §9):
`read_frontmatter` tolerates unknown types/keys and never rejects a document;
`REQUIRED_KEYS` is aligned with SPEC §4.1 (only `type` strictly required).

### CONCEPT:AU-ECO.connector.okf-roundtrip-sync

Bidirectional round-trip sync — the one real capability we lacked (we were one-way,
source→KG). `OkfRoundTripSync` pushes a KG-distilled skill-graph → an OKF bundle on
disk following the mdcode Catalog-Snapshot contract: a `.catalog.state` checksum file
separates tool state from user content, a push **fails fast** on an interim conflict
(a target modified out-of-band → require a pull to resolve), and files in state but
absent from the new snapshot are **intent-to-delete**. Wired onto the EXISTING
`graph_writeback` surface via an `OkfSink` (domain `okf`, `enable_flag`
`OKF_ENABLE_WRITE`) — no new verb, so surface-parity is preserved; `dry_run` previews
the exact create/update/conflict/delete plan.

### CONCEPT:AU-ECO.connector.openwiki-preset

openwiki multi-instance ingest as a **preset** (CLAUDE.md rule: preset, not a new
connector module). A `FILESYSTEM_PRESETS["openwiki"]` entry points the existing
`FilesystemConnector` at each repo's `openwiki/` directory; the delta watermark is the
SHA-256 snapshot of `openwiki/.last-update.json` (tighter than mtime and stable across
git checkouts). OKF frontmatter is stamped on ingest (openwiki md carries none) and
per-repo provenance is the SLUG. Federation/identity is entirely our side.

### CONCEPT:AU-KG.ingest.okf-type-mapping

External free-`type` → governed `domain_vocab` mapping. OKF `type` is an open
vocabulary (SPEC §4.1); to normalize external bundles into our closed OKF-CIS `domain`
axis, `map_external_type` resolves a seed map then signal-matches the closed vocab. An
unmapped type is **not dropped** — it is parked on a review queue for a curation
decision and the concept still ingests under a permissive default domain (SPEC §9).

### CONCEPT:AU-KG.ingest.broken-link-tolerance

Broken-link tolerance in the markdown-link extractor path
(`ontology/document_processing.py`). Inline `[label](target)` links become `LINKS_TO`
edges; a target with no already-materialized node becomes a lightweight `dangling`
placeholder node so the edge is **never dropped** (OKF SPEC §5: consumers MUST tolerate
broken links — they may be not-yet-written knowledge). Opt-in (`extract_links=False`
default) so the existing KG-2.48 slice stays byte-identical.

## Verification

- Distill a skill-graph → OKF bundle (skill-graph standard); round-trip push
  KG→bundle→re-ingest with `.catalog.state` conflict detection + intent-to-delete.
- Ingest 2+ repos' `openwiki/` dirs as the preset; confirm OKF-CIS-stamped nodes and
  the `.last-update.json` snapshot delta skip on re-run.
- `graph_writeback(target="okf", dry_run=true)` previews the plan; live push gated on
  `OKF_ENABLE_WRITE`.
- All static gates (`check_domain_vocab`, `check_no_legacy_markers`, surface-parity,
  mypy) + pytest green.
