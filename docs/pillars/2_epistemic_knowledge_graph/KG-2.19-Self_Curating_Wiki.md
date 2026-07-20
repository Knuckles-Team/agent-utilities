# Self-Curating Wiki (CONCEPT:EG-KG.query.wire-protocol)

## Overview

The Self-Curating Wiki continuously ingests a markdown knowledge vault into the graph but **only
when pages change** — each file's SHA-256 is tracked so unchanged pages are skipped, and state is
written crash-safely. Assimilated from memory-os (`scripts/wiki_continuous_ingest.py`). Extends
**KG-2.7** (Research Intelligence / Ingestion).

## How it works

- **Delta-skip ingest.** `WikiCurator.changed_files` returns only new/modified pages (by content
  hash) vs the stored state; `curate` ingests just those via an injected `ingest_fn` and commits the
  new hash state. Re-running an unchanged vault ingests nothing.
- **Reuses existing ingestion.** `curate_wiki` routes changed pages through `IngestionEngine.ingest`
  (which already does concept/entity extraction, KG-2.8 delta hashing, and linking); the existing
  `SynthesisEngine` auto-curates promotions (episode→preference, decision→principle).
- **Crash-safe state.** State is written via tempfile → `fsync` → atomic `os.replace`, so a crash
  mid-write never corrupts the ingest ledger.

## Key files / API

| Piece | Location |
|---|---|
| Curator | `knowledge_graph/ingestion/wiki_curator.py` (`WikiCurator`, `file_hash`, `curate_wiki`) |
| Entry point | MCP `graph_ingest(action="curate_wiki", target_path=<wiki dir>)` |

## Wiring (≤3 hops)

`graph_ingest(action="curate_wiki")` → `curate_wiki` → `WikiCurator.curate` → `IngestionEngine.ingest`.

## Research provenance

memory-os wiki continuous-ingest (SHA-256 diff + atomic state) — `scripts/wiki_continuous_ingest.py:47-109` (verified).
