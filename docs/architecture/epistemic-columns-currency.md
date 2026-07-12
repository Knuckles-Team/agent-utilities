# Epistemic-columns currency (Seam 1) — consuming epistemic-graph's `KnowledgeBatch`

> **The keystone cross-repo seam.** agent-utilities' primary query path used to flatten
> every engine result to a plain `dict`, dropping the epistemic columns
> (`eg_plan::KnowledgeSet`/`KnowledgeBatch`, CONCEPT:EG-P1-2) the Rust engine had
> already computed for those exact rows — confidence, a bitemporal valid/tx window,
> evidence provenance, policy labels. This page documents the wire surface added to
> close that gap and the ONE primary AU path (`KnowledgeGraph.query`) that adopts it.
> Concepts: **CONCEPT:EG-KB-CURRENCY** (engine side), **CONCEPT:AU-KB-CURRENCY** (facade side).

## The gap

`epistemic-graph` resolves a rich per-row envelope internally
(`crates/eg-plan/src/knowledge.rs`'s `KnowledgeRow` / `knowledge_batch.rs`'s
`KnowledgeBatch`) — score, belief confidence, `(valid_from, valid_until)`,
`(tx_from, tx_to)`, evidence spans, source refs, policy labels, and more. But the
query surfaces a Python caller actually reaches were narrower:

* `Method::UnifiedQuery`/`UnifiedQueryText` return bare `[id, score|nil]` rows — no
  epistemic columns at all.
* `Method::ExplainProvenance` came closer (it already builds a `KnowledgeSet` per
  request) but its wire row (`ExplainProvenanceRowWire`) only carried
  `id`/`kind`/`source_refs`/`evidence_spans` — confidence, the bitemporal window, and
  policy labels were computed server-side and then dropped before the response left
  the engine.
* On the AU side, `agent_utilities/knowledge_graph/facade.py`'s `KnowledgeGraph.query`
  (the guarded, tenant-scoped, audited Cypher read every execution-plane caller uses)
  returned `list[dict]` built by `EpistemicGraphBackend.execute` — plain node property
  projections, with no path to the engine's epistemic envelope at all.

## The wire surface (EG side)

Two additive, non-breaking changes in `epistemic-graph`:

1. **`ExplainProvenanceRowWire`** (`crates/eg-types/src/protocol.rs`) widened with
   `score`, `confidence`, `valid_time`, `tx_time`, `policy_labels` — straight field
   copies off the `KnowledgeRow` the handler already builds
   (`src/server/handlers/query.rs`'s `explain_provenance_result`). `score`/
   `confidence`/`valid_time`/`tx_time` are populated regardless of the `epistemic`
   cargo feature; `source_refs`/`policy_labels`/`evidence_spans` stay empty (never
   fabricated) when `epistemic` is off, exactly as `resolved: false` already
   documented.
2. **`Method::ExplainProvenanceByIds { ids: Vec<String> }`** — a new, ID-seeded
   sibling of `Method::ExplainProvenance`. A caller that already has a set of node
   ids from ANY other read path (a Cypher `MATCH`, a SQL `SELECT`, a prior
   `UnifiedQuery`) does not need to hand-build an `Op` plan just to fetch the
   epistemic envelope for those exact ids — it builds the `KnowledgeSet` straight
   from `RowSet::from_ids(ids)` and resolves the IDENTICAL row shape
   `ExplainProvenance` does. This is the primitive AU's facade calls.

Both changes are purely additive: no existing `Method` variant's shape changed, no
existing wire field was removed or renamed, and the capability-policy ledger
(`eg-capabilities`) was updated in lockstep (`ExplainProvenanceByIds` classified
identically to `ExplainProvenance` — read-only, snapshot-isolated, `explain:read`).

Python surface: `epistemic_graph/client.py`'s `QueryClient.explain_provenance_by_ids`
mirrors the existing `explain_provenance` convenience method.

## The AU primary path

`agent_utilities/knowledge_graph/facade.py`'s `KnowledgeGraph.query` gained one new,
default-off parameter:

```python
kg.query(cypher, include_epistemic=True)
```

* **Default (`include_epistemic=False`, unchanged)** — byte-for-byte the same
  `list[dict]` every existing caller already gets.
* **Opt-in (`include_epistemic=True`)** — the SAME rows, widened. The method:
  1. runs the Cypher exactly as before, through the existing guarded path (tenant
     scope, ACL row-filter, fine-grained object permissioning, read audit) — nothing
     about the read's authorization changes;
  2. extracts the distinct node ids the plain rows project
     (`core/epistemic_row.py`'s `row_ids_from_plain_rows` — recognizes both a bare
     `RETURN n` node-dict projection and a `RETURN n.id AS id` scalar projection);
  3. resolves their epistemic envelope in ONE round-trip via
     `GraphComputeEngine.explain_provenance_by_ids` (`Method::ExplainProvenanceByIds`);
  4. returns `list[EpistemicRow]` — each one the engine's envelope zipped back with
     the plain row's own properties, so opting in never loses information a plain
     `dict` row would have carried.

`EpistemicRow` (`agent_utilities/knowledge_graph/core/epistemic_row.py`) is the typed
carrier: `id`, `kind`, `score`, `confidence`, `evidence_refs`, `source_refs`,
`valid_time`, `tx_time`, `policy_labels`, `properties` — plus a `calibration` property
that aliases `confidence` (the engine substrate's confidence value IS the calibration
signal it produces today; there is no second, independently-computed calibration
column on the wire, so this is documented as an alias rather than a fabricated second
number).

## Proof

`tests/integration/knowledge_graph/test_kb_currency_epistemic_facade.py` writes a
Claim + Evidence node pair (with a real confidence, a bitemporal window, and a
`SUPPORTS` edge) directly into a real, ephemeral `epistemic-graph-server`, then
asserts `KnowledgeGraph.query(..., include_epistemic=True)` returns an `EpistemicRow`
whose `confidence`/`valid_time`/`tx_time` match what was written AND whose
`source_refs`/`policy_labels` are the belief-substrate's OWN derived values from the
`SUPPORTS` edge (not stored properties, not client-side echoes) — proving the
envelope originated in the engine's `KnowledgeSet`, not fabricated AU-side.

## Adoption path for other AU read surfaces (shipped)

The keystone slice above shipped on ONE primary path (`KnowledgeGraph.query`). The
follow-up workstream threaded the SAME pattern — a shared
`agent_utilities.knowledge_graph.core.epistemic_row.attach_epistemic_rows(rows, fetch)`
helper (extract ids in the rows' own order via `row_ids_from_plain_rows`, resolve
them in one round-trip via an id-seeded `explain_provenance_by_ids`-shaped `fetch`,
zip each engine row back with the plain row's own properties) — through every other
read surface that returned bare rows:

* **`GraphComputeEngine.query_unified(..., include_epistemic=True)`**
  (`agent_utilities/knowledge_graph/core/graph_compute.py`) and
  **`IntelligenceGraphEngine.uql(..., include_epistemic=True)`**
  (`agent_utilities/knowledge_graph/orchestration/engine_query.py`) — the cross-modal
  `UnifiedQuery`/`UnifiedQueryText` surfaces. Both default off (byte-for-byte the
  existing `[{"id", "score"}]` rows) and, opted in, currency-upgrade the SAME ids in
  the SAME post-`Rank` order into `list[EpistemicRow]`. Proof:
  `tests/integration/knowledge_graph/test_kb_currency_epistemic_query_paths.py::test_query_unified_include_epistemic_carries_engine_envelope`
  and `::test_uql_include_epistemic_carries_engine_envelope` — both seed a real
  Claim+Evidence(+`SUPPORTS`) pair into a real ephemeral engine and assert the
  returned confidence/bitemporal window/`source_refs`/`policy_labels` are the
  engine's own resolution.
* **`GraphBackend.execute(..., include_epistemic=True)`** (the ABC in
  `agent_utilities/knowledge_graph/backends/base.py`) — the `store.execute`
  counterpart of `KnowledgeGraph.query`'s opt-in, for internal/unscoped callers that
  bypass the facade. `EpistemicGraphBackend.execute` (the one backend whose
  `GraphComputeEngine` exposes `explain_provenance_by_ids`) implements it for real
  (own `execute` body renamed to `_execute_rows`, with a thin wrapper attaching the
  envelope). `FanOutBackend.execute` forwards the flag to its authority backend on
  the read path (honored when the authority itself supports it). Every OTHER
  concrete backend (`AGEBackend`/`PostgreSQLBackend`/`Neo4jBackend`/
  `FalkorDBBackend`/`LadybugBackend`/`JenaFusekiBackend`/`StardogSparqlBackend`) has
  no id-seeded epistemic-envelope primitive, so a `True` request degrades to `[]`
  (never raises, never silently returns plain `dict` rows under a `True` request) —
  the documented ABC contract. Proof:
  `test_kb_currency_epistemic_query_paths.py::test_store_execute_include_epistemic_carries_engine_envelope`
  (real engine) and `::test_store_execute_include_epistemic_degrades_on_unsupported_backend`
  (the degrade contract, no engine needed).
* **Typed evidence-span view** — `EpistemicRow.evidence_refs` still carries the
  wire's raw `EvidenceSpanWire` dicts verbatim (never removed, so nothing regresses
  for an existing caller), but a new `EpistemicRow.typed_evidence_refs` property and
  `EvidenceSpan` dataclass (`core/epistemic_row.py`) parse each externally-tagged
  `{"<Variant>": {...}}` wire entry into a typed, attribute-accessible view (one
  dataclass covering all 11 `eg_modality::EvidenceSpan` variants' fields, plus a
  `raw` escape hatch for a field this dataclass hasn't been widened for). An entry
  that doesn't parse as a recognized single-key-map shape is skipped, never
  fabricated. Proof: `tests/unit/knowledge_graph/core/test_epistemic_row.py`.

## What remains for full adoption (honest scope)

One documented gap remains — the Arrow/columnar surface:

* **`KnowledgeBatch`'s Arrow/columnar surface is NOT wired to any AU caller, and
  genuinely can't be with additive Python-only changes** — this was investigated,
  not just deferred. Concretely, as of this workstream:
  - `KnowledgeBatch`/`to_record_batch`/`ChunkedKnowledgeCursor::to_arrow_ipc_stream`
    (`epistemic-graph/crates/eg-plan/src/knowledge_batch.rs`) exist and are unit-
    tested, but live behind the `knowledge-batch` cargo feature
    (`crates/eg-plan/Cargo.toml`), which is **not** part of `full`/`default` — a
    stock server build (including the one this workstream tested against) links no
    Arrow at all for this path.
  - There is **no `Method` variant, no `eg-types` wire DTO, and no dispatch/handler**
    (`src/server/dispatch.rs` / `src/protocol.rs` / `src/server/handlers/query.rs`)
    that builds a `KnowledgeSet`/`KnowledgeBatch` and serializes it back over the
    existing MessagePack RPC transport — `KnowledgeBatch` is reachable only from
    Rust code already inside the `eg-plan` crate. The ONE existing server-side Arrow
    export (`src/server/dataset_handle.rs`) is a completely separate path built from
    `eg_query::exec_sql_arrow`, not from a `KnowledgeSet`, and isn't reused here.
  - The Python client (`epistemic_graph/client.py`) has no Arrow/PyArrow dependency
    or IPC-decoding method at all.
  - **What shipping this for real needs** (EG-side, not a Python-only change): (1) a
    new `Method::ExplainProvenanceBatch{ByIds}`-shaped variant + `eg-types` wire DTO
    carrying raw Arrow IPC-stream bytes (`Vec<u8>`), gated behind `knowledge-batch`
    (which the facade's `Cargo.toml` would need to additionally forward into `full`
    or a new opt-in server feature); (2) a dispatch/handler arm that resolves a
    `KnowledgeSet` (the SAME `RowSet::from_ids` primitive `ExplainProvenanceByIds`
    already uses) and encodes it via `ChunkedKnowledgeCursor::to_arrow_ipc_stream`
    into the response bytes; (3) an `epistemic_graph` Python client method that
    depends on `pyarrow` to decode the IPC stream into a `RecordBatch`/`Table`; (4)
    an AU consumer (e.g. `GraphComputeEngine.explain_provenance_batch_by_ids`)
    exposing that `Table` to a bulk/vectorized caller. This is a genuinely new wire
    surface spanning both repos, not an extension of the additive
    `ExplainProvenanceByIds` pattern this seam otherwise reused everywhere else —
    tracked as a separate, larger follow-up.
