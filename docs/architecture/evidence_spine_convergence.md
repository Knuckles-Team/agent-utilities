# Evidence-Spine Convergence (Seam 2)

**Concepts:** AU-KG.identity.evidence-spine-convergence (this doc) ·
AU-KG.identity.asset-occurrence (AU-P1-4, `media_store.py`'s existing
`Blob`/`Rendition`/`AssetOccurrence` identity chain) · EG-X1 (epistemic-graph's
multimodal evidence-graph spine + citation resolver,
`crates/eg-epistemic/src/evidence.rs`, feature `evidence-graph`).

## The gap this closes

Before this change there were **two parallel evidence chains** for the same
`EvidenceSpan` shape:

* AU stored *that* some bytes occurred — a `:SourceObject -> :AssetOccurrence ->
  :Blob` identity chain (AU-P1-4) — but had no way to say *where inside those
  bytes* a claim's evidence sat.
* epistemic-graph's own evidence-graph (EG-X1) already resolved a located
  `EvidenceSpan` locus (`PageBox`/`ImageRegion`/`AudioSegment`/…) off an
  `:Evidence` node's `evidence_span`/`occurrence_id`/`blob_ref` properties via
  `Method::ExplainEvidence` / `eg_epistemic::evidence_citations` — but nothing
  ever wrote an AU-produced occurrence into that shape.

A citation resolved through AU and a citation resolved through EG were
answering two different questions from two different graphs of record. Seam 2
converges them: **one write path, one resolver.**

## What changed

`MediaStore.store_document_page_evidence` (`agent_utilities/knowledge_graph/
memory/media_store.py`) is a new, **opt-in** method — nothing about
`store_media`/`store_rendition` changed, and a caller that never calls it writes
nothing extra. When a caller HAS a document page + bounding box for the bytes
it's storing, this method:

1. Stores the bytes via the existing `store_media` (AU-P1-4's `:AssetOccurrence
   -> :Blob` chain, unchanged).
2. Writes/reuses a `:SourceObject` node for the owning document
   (`sourceobject:<document_id>`, upserted once) plus a structural
   `hasOccurrence` edge to the new occurrence.
3. Writes an `:Evidence` node carrying the located `PageBox` `EvidenceSpan`
   locus (the externally-tagged `{"PageBox": {document_id, page, x, y, width,
   height}}` shape `eg_epistemic::BeliefGraph::from_graph_view` decodes) plus
   `occurrence_id`/`blob_ref` — the SAME identity-chain convention
   `eg_epistemic::evidence` documents — and a structural `extractedFrom` edge
   back to the occurrence.
4. When a `claim_id` is given, links the evidence to it with a
   `relationship_type: "SUPPORTS"` edge — the SAME convention
   `eg_epistemic`'s own claim materialization
   (`src/server/handlers/mining.rs::materialize_claim`) writes, so
   `evidence_citations`'s support/contradiction/attack walk recognizes it with
   **no engine-side change**.

**No second resolver, no new engine write endpoint.** The generic
`AddNode`/`AddEdge` RPCs (`client.nodes.add`/`client.edges.add`) the rest of
`MediaStore` already uses are sufficient to produce the exact property/edge
shape the engine's real decoder expects — reading citations back always goes
through epistemic-graph's own `Method::ExplainEvidence`, never a second,
AU-side implementation of the same resolution logic.

```mermaid
flowchart LR
    subgraph AU["agent-utilities (Python)"]
        SM["MediaStore.store_media()\n(unchanged, AU-P1-4)"]
        SD["MediaStore.store_document_page_evidence()\n(NEW, opt-in)"]
        SD --> SM
    end

    SM -->|AddNode/AddEdge| SO[":SourceObject"]
    SM -->|AddNode/AddEdge, hasBlob| OC[":AssetOccurrence"]
    OC --> BL[":Blob"]
    SD -->|hasOccurrence| SO
    SO --> OC
    SD -->|AddNode: evidence_span/occurrence_id/blob_ref| EV[":Evidence"]
    SD -->|extractedFrom| OC
    SD -->|"SUPPORTS (relationship_type)"| CL[":Claim"]
    EV --> CL

    subgraph EG["epistemic-graph engine (Rust)"]
        BGV["BeliefGraph::from_graph_view()"]
        EC["evidence_citations() /\nMethod::ExplainEvidence"]
        BGV --> EC
    end

    EV -.decoded by.-> BGV
    CL -.decoded by.-> BGV
```

## Proof (the vertical slice)

One modality, end-to-end: **document page-box** (`EvidenceSpan::PageBox`).

* AU half (no live engine needed): `tests/unit/knowledge_graph/
  test_media_store_evidence_spine.py` proves `store_document_page_evidence`
  writes the exact node/edge shape — `evidence_span`/`occurrence_id`/`blob_ref`
  on the `:Evidence` node, structural `hasOccurrence`/`extractedFrom`/`hasBlob`
  edges, and the `relationship_type: "SUPPORTS"` edge when a `claim_id` is
  given.
* EG half (epistemic-graph repo): `crates/eg-epistemic/tests/
  x1_au_occurrence_chain.rs` mirrors those EXACT literal values into a real
  `GraphView`, decodes them through the REAL `BeliefGraph::from_graph_view`,
  and asserts `evidence_citations`/`resolve_locus` return the exact `PageBox`
  locus + occurrence/blob identity — the same acceptance shape EG-X1's own
  `x1_evidence_chain.rs` established for a hand-built fixture, now keyed off
  AU's actual write shape.

Together the two prove the round trip without requiring a live server built
with the (opt-in, non-default) `evidence-graph` Cargo feature in this repo's
test harness — see "What remains" below.

## All eleven loci now wired (AU half) — Seam 2 completion

`eg_modality::EvidenceSpan` defines eleven located-locus variants. The
page-box slice above proved the pattern for `PageBox`; the SAME pattern now
extends to the remaining ten via a shared private skeleton,
`MediaStore._store_located_evidence` (`agent_utilities/knowledge_graph/
memory/media_store.py`) — the generalized form of
`store_document_page_evidence`'s steps 2-4 (upsert `:SourceObject`, write the
`:Evidence` node with `evidence_span`/`occurrence_id`/`blob_ref`, the
`extractedFrom` edge, and the `SUPPORTS` edge when `claim_id` is given),
parameterized over `about_id` (whichever locus field identifies the owning
artifact) and the caller-built externally-tagged `evidence_span` dict. Each
public wrapper below returns the (identically-shaped) `EvidenceLocus`
dataclass and is opt-in exactly like `store_document_page_evidence` — nothing
about `store_media`/`store_rendition` changes.

| Locus (`eg_modality::EvidenceSpan`) | `MediaStore` method | Producer wiring |
|---|---|---|
| `PageBox` | `store_document_page_evidence` | **No live external caller found** (2026-07-22 re-check) — only `tests/unit/knowledge_graph/test_media_store_evidence_spine.py` calls it; the "Shipped Seam 2 slice" label below predates that re-check and overstated it, exactly the kind of drift the seam-closure audit (`reports/seam-closure-audit-2026-07-22.md`) was run to catch. Left as-is (out of this pass's scope — no candidate producer was surveyed for it) rather than silently fixed. |
| `DocumentSpan` | `store_document_span_evidence` | **Wired 2026-07-22** — `IngestionEngine._extract_facts_into_graph` (`agent_utilities/knowledge_graph/ingestion/engine.py`), one locus per persisted fact whose `evidence_span` is a real substring of the window it was extracted from. |
| `TableCellRange` | `store_table_cell_evidence` | **Wired 2026-07-22** — `readers_office.read_xlsx` (`agent_utilities/knowledge_graph/extraction/readers_office.py`), one locus per worksheet covering its full used range. |
| `ImageRegion` | `store_image_region_evidence` | **Wired 2026-07-22, RapidOCR branch only** — `readers_media._ocr_with_rapidocr` (`agent_utilities/knowledge_graph/extraction/readers_media.py`). The `pytesseract` branch (`_ocr_with_pytesseract`) never computes a box at all (`image_to_string` has no box output), so there is nothing to wire there without adding new computation — left unwired by design. |
| `AudioSegment` | `store_audio_segment_evidence` | Wired (pre-existing, confirmed 2026-07-21) — `messaging/router.py`. |
| `VideoShot` | `store_video_shot_evidence` | No natural producer (see below) |
| `VideoFrameRange` | `store_video_frame_range_evidence` | No natural producer (see below) |
| `MetricWindow` | `store_metric_window_evidence` | Not wired — see below |
| `RowVersion` | `store_row_version_evidence` | Not wired — see below |
| `CodeSymbol` | `store_code_symbol_evidence` | Not wired — see below |
| `TraceSpan` | `store_trace_span_evidence` | Not wired — see below |

**Proof:** `tests/unit/knowledge_graph/test_media_store_evidence_spine.py`'s
`test_store_locus_evidence_writes_the_full_identity_chain` /
`test_store_locus_evidence_links_supports_edge_when_claim_given` are
parametrized over all ten new loci (`LOCUS_CASES`), each asserting the exact
`evidence_span` literal plus `occurrence_id`/`blob_ref` and the structural
`hasOccurrence`/`extractedFrom`/`hasBlob` edges, mirroring the page-box test's
approach node-for-node.

### Producer wiring — 2026-07-22 pass (three more closed)

The seam-closure audit (`reports/seam-closure-audit-2026-07-22.md`) re-counted
this survey against live callers and found only 2 of 11 loci (`PageBox`,
`AudioSegment`) had one — the other nine had a tested `MediaStore` method with
the underlying data computed and discarded right next to it. This pass wired
three of the nine (`DocumentSpan`, `TableCellRange`, `ImageRegion`), following
`AudioSegment`'s own producer (`messaging/router.py`) as the template — a
best-effort side effect at the exact point the data was already being thrown
away, reached via `knowledge_graph.memory.native_ingest.media_store()` (the
same ambient-engine accessor `native_ingest.py`'s other typed writers already
use, so no new context plumbing was needed to reach a bound `MediaStore`):

* **`DocumentSpan`** — `IngestionEngine._extract_facts_into_graph`
  (`knowledge_graph/ingestion/engine.py`) now locates each persisted fact's
  `evidence_span` back into the SAME window `extract_facts` mined it from via
  `str.find` (a real offset lookup, not a new computation) and writes one
  locus per fact. `persist_facts()` itself is untouched — the evidence write
  sits alongside it, exactly like `_persist_audio_segment_evidence` sits
  alongside `store_media` in the router.
* **`TableCellRange`** — `readers_office.read_xlsx` now writes ONE locus per
  worksheet spanning its real used range (`row_end`/`col_end` from the same
  row/column extent `_format_rows` was already iterating over) rather than one
  per cell/row — a large sheet would otherwise mean thousands of writes for a
  single ingest pass, an unreasonable multiple of the sheet's own read. The
  `.csv`/`.tsv` reader (owned by `extraction/readers.py`, not
  `readers_office.py`) was left unwired this pass — same shape, not yet done.
* **`ImageRegion`** — `readers_media._ocr_with_rapidocr` now writes one locus
  per recognised text line, the quadrilateral box RapidOCR already returns
  collapsed to the axis-aligned rectangle the locus expects (min/max over the
  same points, no new detection). Scoped to the RapidOCR branch only —
  `_ocr_with_pytesseract`'s `image_to_string` call never computes a box at
  all, so there is nothing to wire there without adding new computation
  (fabricating one would violate this seam's own "no invented data" charter).

Still not wired, re-surveyed this pass:

* **`CodeSymbol`** / **`TraceSpan`** — both candidate producers
  (`enrichment/extractors/code_test.py`'s AST mapper;
  `harness/trace_backend.py::KGTraceBackend.record_event`) sit on genuinely
  HOT, high-volume paths (every symbol across the whole continuously-reingested
  fleet; every span/generation in the harness). Wiring either unconditionally
  would mean an evidence write (blob + occurrence + SourceObject + Evidence
  node) per symbol/span with no `claim_id` ever attached — orphaned evidence at
  a volume disproportionate to the audio/image/table producers above, not a
  same-shape "wire the discarded data" fix. Left unwired rather than forced;
  the right trigger is "a claim actually cites this symbol/span", which
  doesn't exist as a call site yet.
* **`RowVersion`** — `DatabaseConnector.poll()`
  (`protocols/source_connectors/connectors/database.py`) has `row_id` + a
  string `updated_at` watermark, but no `table` field (the connector wraps an
  arbitrary `SELECT`, not one named table) and no integer `version`. Neither
  field is fabricable from what the connector tracks without guessing —
  genuinely no computed match for the locus's required fields.
* **`MetricWindow`** — re-surveyed beyond the original candidate:
  `observability/health.py`'s `detect_anomaly`/`HealthTrendBuffer` (the
  fan-manager-derived trend/baseline/anomaly kernel, the closest real match —
  it reasons over exactly a windowed set of readings) and
  `observability/health_ingest.py`'s `ingest_health_anomaly` have **no live AU
  caller at all** (`grep` for both together returns nothing outside their own
  module and tests) — this pair is itself an unwired kernel in this repo (a
  separate, pre-existing gap, not something to paper over by inventing a
  caller here). `EngineTimeSeriesBackend.query()` remains a read path, not an
  ingestion path. No computed-and-discarded window exists in AU today.
* **`VideoShot`/`VideoFrameRange`** — unchanged: no shot-detection or
  frame-accurate video-processing path exists in AU; `tools/media_tools.py`'s
  `generate_video()` only generates video, it does not analyze one.

## What remains for full convergence

* **Six loci still have a tested `MediaStore` method but no live AU producer
  call** (`VideoShot`, `VideoFrameRange`, `MetricWindow`, `RowVersion`,
  `CodeSymbol`, `TraceSpan` — see the survey above); `PageBox` also has no live
  external caller found this pass despite its earlier "shipped" label.
* **No live-engine round-trip test in AU's own suite.** `evidence-graph` is an
  opt-in, non-default Cargo feature (not folded into any tier, including
  `full`/`default`) — AU's shared ephemeral-engine test fixture
  (`tests/_test_engine.py`, `tiny_engine`/`engine_graph`) does not build or
  probe for it, so there is no `pytest.mark.engine` test in AU asserting
  `client.query.explain_evidence(...)` against a REAL running server for this
  seam yet. The EG-side Rust test is the closest thing to that proof today
  (it runs the actual engine decode/resolve code, just not over the wire).
  Standing up a dedicated `evidence-graph`-featured test engine (or folding
  `evidence-graph` into a test-only tier) is the natural follow-up.
* **`tests/_test_engine.py`'s `BUILD_TIER = "pi-max"` fallback build path is
  stale** relative to epistemic-graph's own tier retirement (EG-371: `pi`/
  `pi-max`/`node` no longer exist as Cargo features — `full` **is** `default`
  now) — a pre-existing drift unrelated to this seam, noted here because it's
  exactly what would need fixing (or superseding) to add the live-engine test
  above.
