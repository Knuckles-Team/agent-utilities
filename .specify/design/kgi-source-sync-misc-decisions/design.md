# Design Document: Two small, independent `source_sync` hardening decisions

> Both live in `agent_utilities/knowledge_graph/core/source_sync.py`, but are
> unrelated decisions that happen to share a file — grouped here only for
> documentation locality, not because they are the same choice.

CONCEPT:AU-KG.ingest.fleet-sync-rejected-row-cache ·
CONCEPT:AU-KG.ingest.source-sync-canonical

## Decision 1 — a best-effort local cache of known-bad rows, never authoritative

`CONCEPT:AU-KG.ingest.fleet-sync-rejected-row-cache` — `source_sync.py:530-555`
(`_load_rejected_row_cache`), introduced by commit `c9fdf6f9` "fix(kg): D-SH-2
— persist rejected fleet-catalog rows, skip re-bisecting known offenders".

**The rejected alternative, implicit in the fallback behavior itself**: the
prior state, where every fleet-catalog sync re-attempted (re-bisected) every
row from scratch, including rows already known to fail from a previous run —
wasted work repeated every sync cycle. The code's own comment makes the
fallback-safety property explicit: "a missing/corrupt/unreadable file
degrades to 'nothing known bad yet', never an error" — i.e., "same as before
this cache existed" is the deliberately preserved fallback, not a regression
risk. The cache maps `{row_id: content_hash}` for rows known-bad from a prior
sync, read defensively (any exception → empty dict, logged at `debug`, sync
proceeds as if the cache never existed). It is a pure optimization: a cache
miss or read failure costs re-work, never correctness.

## Decision 2 — external ARD registries route through the SAME envelope-atomic path as every other typed source

`CONCEPT:AU-KG.ingest.source-sync-canonical` — `source_sync.py:4110-4130`
(`_resolve_ard_registries`) and `4131-4155` (`_sync_ard`).

**What this is**: ingestion of external Attested Resource Discovery (ARD)
registries (e.g. Hugging Face-style catalogs, configured via `ARD_REGISTRIES`
as `[{"name": "hf", "preset": "huggingface"}]`) as typed, discoverable
resources — each registry resource mapped to a typed `:MCPServer`/`:Skill`
node linked to its `:ResourceRegistry` and capabilities.

**The rejected alternative**: a bespoke ingestion/write path specific to ARD
registries, separate from how every other typed `source_sync` handler
commits. Instead `_sync_ard` explicitly notes it is "AU-P1-5 envelope-native"
— each mapped resource emits one `ChangeEnvelope` through the SAME atomic
native-transaction path documented in
`.specify/design/kgi-change-envelope-atomic/design.md` (Decision 2), not a
special-cased write. `mode='reconcile'` additionally tombstones resources no
longer present in a registry sweep, and `client` may inject a fetch function
for offline tests — the connector itself is signature-verified before
draining, so an untrusted or tampered registry response cannot silently
inject nodes.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/core/source_sync.py`
  only (both decisions are self-contained functions within it).
- **Backward Compatible**: Yes for both — the rejected-row cache is
  transparently absent-safe; ARD registries are opt-in via
  `ARD_REGISTRIES` and ingest zero resources when unset.
- **Breaking Changes**: None.
- **Known weak point**: the rejected-row cache has no TTL/eviction — a row
  that was transiently bad (e.g. a temporary upstream 500) but is now fixable
  stays in the "known bad" cache until its content hash changes upstream or
  the cache file is manually cleared, so a real fix can go undetected for an
  arbitrary number of sync cycles.
