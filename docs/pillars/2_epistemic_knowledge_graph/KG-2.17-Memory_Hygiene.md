# Memory Hygiene (CONCEPT:KG-2.17)

## Overview

Memory Hygiene is a maintenance pass that **bounds memory growth without destroying information**:
a decay scanner archives stale AI-generated memory by closing its bi-temporal interval, and a
semantic-merge pass collapses near-duplicates. Assimilated from memory-os
(`scripts/decay_scanner.py`, `scripts/semantic_dedup.py`). Extends **KG-2.1 / KG-2.3**.

## How it works

- **Decay scan.** Exponential decay with an importance-tiered half-life (90d if importance ≥ 0.3,
  else 30d). Low-decay AI content is **archived** by setting KG-2.11 `valid_to` + `status=ARCHIVED`
  — never hard-deleted, so as-of queries before the archival instant still see it. High-confidence
  stale items are **alerted** for review instead of archived; human/procedural memory is exempt.
- **Semantic merge.** Near-duplicate memories (cosine ≥ 0.92) are merged with a cheap length-ratio
  pre-filter (skip cosine when sizes differ > 2×), unioning tags and keeping max importance.
- **Pure decisions.** `classify_node`, `plan_decay`, `semantic_merge_groups`, `decay_score` are
  LLM/engine-free and unit-tested; `MemoryHygiene.run` applies the plan to the durable backend.

## Key files / API

| Piece | Location |
|---|---|
| Hygiene module | `knowledge_graph/memory/hygiene.py` (`classify_node`, `plan_decay`, `semantic_merge_groups`, `decay_score`, `MemoryHygiene`, `run_hygiene`) |
| Entry point | `knowledge_graph/memory/cli.py` (`agent-utilities-memory hygiene`) |

## Wiring (≤3 hops)

`agent-utilities-memory hygiene` → `run_hygiene` → `MemoryHygiene.run` → backend (≤3 hops). Also
runs **automatically** as the `hygiene` job in the consolidated `engine_tasks` maintenance scheduler
(`_tick_hygiene`), gated by `KG_HYGIENE_DAEMON` (default on) on a `KG_HYGIENE_INTERVAL` cadence
(default daily), behind the shared foreground-throttle gate.

## Research provenance

memory-os decay scanner + semantic dedup — `scripts/decay_scanner.py:43-64`, `scripts/semantic_dedup.py:1-189` (verified).
