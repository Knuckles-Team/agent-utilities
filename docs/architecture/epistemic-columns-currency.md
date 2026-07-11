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

## What remains for full adoption (honest scope)

This establishes the CURRENCY and the pattern on ONE primary path — it does not
retrofit every AU read surface:

* `GraphComputeEngine.unified`/`uql` (the cross-modal `UnifiedQuery`/`UnifiedQueryText`
  surfaces) still return bare `[id, score]` rows; adopting the same pattern there
  means the SAME `explain_provenance_by_ids` currency-upgrade, applied to a ranked
  plan result instead of a Cypher result.
* `store.execute` (the ungaurded, unaudited direct backend path some internal/
  unscoped callers use instead of `KnowledgeGraph.query`) has no `include_epistemic`
  parameter — only the facade's guarded `query` method does.
* `EpistemicRow.evidence_refs` carries the wire's raw `EvidenceSpanWire` dicts
  verbatim rather than a second AU-side typed dataclass per evidence-span variant —
  a reasonable follow-up once a concrete consumer needs typed access to a specific
  span kind (`CodeSymbol`/`TraceSpan`/…).
* The KnowledgeBatch's Arrow/columnar surface (`to_record_batch`/`to_arrow_ipc_stream`)
  is not threaded through at all yet — this seam adopts the per-row `KnowledgeSet`
  shape (already exposed over the existing RPC transport), not the Arrow IPC stream;
  a bulk/vectorized AU consumer wanting `RecordBatch`es directly is a separate,
  larger wiring step (a new streamed wire method + a PyArrow-consuming AU client),
  out of this workstream's scope.
