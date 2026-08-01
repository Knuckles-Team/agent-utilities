# Design Document: One typed ChangeEnvelope, one atomic native transaction per record

> `agent_utilities/knowledge_graph/ingestion/change_envelope.py` (the type, AU-P1-6)
> and `agent_utilities/knowledge_graph/core/source_sync.py` (the consolidation
> that routes delta handlers through it, AU-P1-5).

CONCEPT:AU-KG.ingest.change-envelope ·
CONCEPT:AU-KG.ingest.envelope-atomic ·
CONCEPT:AU-KG.ingest.envelope-atomic-transaction

## Decision 1 — a connector emits one typed, self-describing `ChangeEnvelope`, not an ad hoc dict

`CONCEPT:AU-KG.ingest.change-envelope` — `change_envelope.py:1-60`.

**The problem, named directly in the module docstring**: every connector handed
`sync_source`/`ingest_external_batch` an ad hoc `{"id": ..., "type": ...,
**props}` dict. Each connector independently decided what `"id"`/`"updated"`/
`"acl"` meant; provenance was stamped separately
(`enrichment.provenance.stamp_source`), watermarks tracked separately
(`_read_watermark`/`_write_watermark`), and ACL was a bespoke
`ExternalAccess` bolted on per-connector. Nothing was one typed unit a
consumer (write layer, a future CDC/webhook receiver, an audit trail, a
replay tool) could reason about uniformly.

**The rejected alternative** is exactly that status quo: per-connector
translation into a loosely-typed dict, re-derived independently by every
connector author. It is what the code explicitly frames as "the gap this
closes" — accepted for years because it worked, rejected now because it gives
no single place to validate identity, bitemporal timestamps, governance
(ACL/classification/retention/legal-hold), or a deterministic idempotency key.

**The design chosen**: `ChangeEnvelope` (`change_envelope.py:112`), a frozen
dataclass carrying identity (`envelope_id`, `idempotency_key` — auto-derived
in `__post_init__` from connector+tenant+instance+object+version+operation, so
redelivery of the same logical change is provably a no-op), provenance/lineage,
bitemporal timestamps (`event_time`/`valid_time`/`observed_time`), payload
(`operation` ∈ `upsert`/`delete`/`snapshot_complete`), governance fields, and
an operational `checkpoint` (the typed twin of the old bare-string watermark).
Two adapters bridge the boundary: `from_connector_record` (bridge IN from
today's dict shape) and `to_entity_dict` (bridge OUT, so an envelope can still
feed the legacy `engine.ingest_external_batch`/`write_entities` unchanged) —
deliberately scoped (AU-P1-6) to the type + one adapter each way, not a
rewrite of every connector at once.

## Decision 2 — every delta handler commits through ONE atomic native transaction, never the old ad hoc write sequence

`CONCEPT:AU-KG.ingest.envelope-atomic` / `CONCEPT:AU-KG.ingest.envelope-atomic-transaction` —
`source_sync.py:15-30` and 23 call sites through the file (e.g. `source_sync.py:834`
`_sync_leanix`, the flagship migration).

**The rejected alternative, named explicitly in the code**: the historical
ad hoc write sequence — `engine.ingest_external_batch(domain, entities,
relationships)` followed by a separate post-commit watermark write. Under
that model a crash mid-batch could silently skip the watermark advance for
objects that DID get written, so a resumed sync could re-process (or worse,
never re-check) records whose write already landed. The batch was not truly
atomic at record granularity.

**The design chosen**: every durable delta handler, materialize extractor,
and generic capability hydrate routes normalized graph material through
`ChangeEnvelope` / `ingestion.envelope_ingest.ingest_graph_slice` — one
native `ApplyChangeEnvelope` redb/Raft commit per record, covering graph
material, policy, lineage, typed content version, source cursor, and
CDC/projection outbox in the SAME transaction. This makes the unit
crash-resume safe at PER-RECORD granularity: a batch call is all-or-visible,
and a native capability/session/persistence failure is fail-closed — it
never silently downgrades to the historical Python write sequence. Every
handler in `ENVELOPE_NATIVE_SOURCES` commits through this boundary (LeanIX's
`_sync_leanix` was the flagship migration, referenced by name in
`change_envelope`'s own module docstring; the ARD registry handler
`_sync_ard` is a later adopter — see `.specify/design/kgi-source-sync-misc-decisions/design.md`).
Sources not yet in `ENVELOPE_NATIVE_SOURCES` still sync through the one
`source_sync` entrypoint — they fall back to a full hydrate via the
capability registry rather than being forced onto the envelope path before
they are ready, keeping the external surface uniform while being honest
about which sources are incremental/atomic today.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/ingestion/change_envelope.py`,
  `agent_utilities/knowledge_graph/ingestion/envelope_ingest.py`,
  `agent_utilities/knowledge_graph/core/source_sync.py`,
  `agent_utilities/knowledge_graph/ontology/connector_manifest_gate.py`.
- **Backward Compatible**: Yes. `to_entity_dict` lets an envelope still feed
  the legacy batch-write API; non-migrated sources are unaffected until moved
  into `ENVELOPE_NATIVE_SOURCES`.
- **Breaking Changes**: None for callers; each migrated source's crash-resume
  behavior changes (strictly improves — from batch-level to record-level
  atomicity).
- **Known weak point**: migration is per-source and manual — a source left
  outside `ENVELOPE_NATIVE_SOURCES` still has the old batch-level
  crash-resume gap. There is no mechanical audit that flags a
  should-have-migrated source; it relies on the source's own delta-handler
  author opting in.
