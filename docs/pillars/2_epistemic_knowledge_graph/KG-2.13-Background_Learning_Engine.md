# Background Learning Engine (CONCEPT:AU-KG.memory.background-learning-engine)

## Overview

The Background Learning Engine runs an asynchronous, concurrency-bounded learner that turns
conversation transcripts into **targeted ADD / UPDATE / DELETE memory edits** — not raw dumps —
and writes them as **bi-temporal graph mutations** (KG-2.11). Assimilated from Quarq Agent's async
learner (`agent-oss/agent.py`), with a memory-os-inspired **typed, outcome-grounded** extraction
enhancement. Extends **KG-2.1** (+AHE-3 self-improvement).

## How it works

- **Targeted edits.** `extract_edits` (ORCH-1.27 `learner` role) emits a `MemoryEdit` list:
  `ADD` new facts, `UPDATE` outdated ones, `DELETE` contradictions.
- **Typed + grounded (enhancement).** Each edit carries `entry_type` (decision/resolution/note/fact),
  a `training_value`, an `outcome_gate`, and `evidence_ids`. A decision/resolution that claims
  grounding but cites no evidence is **dropped** (not stored as an unverified fact); persisted
  decisions get `GROUNDED_BY` edges to their evidence and `type:`/`train:` tags.
- **Bi-temporal mutations.** `apply_edits` ADDs with full temporal stamps; UPDATE re-stamps and
  supersedes; **DELETE is soft** (`status=REMOVED` + `valid_to`), preserving history.
- **Async controls.** `Semaphore(4)`, bounded exponential backoff (`with_backoff`, 2→60s, capped so
  CI never hangs), `schedule` + `await_pending` sync barrier. `resolve_relative_dates` converts
  "yesterday"/"N weeks ago" to absolute dates at learn time.

## Key files / API

| Piece | Location |
|---|---|
| Engine + edits | `knowledge_graph/memory/learning_engine.py` (`MemoryEdit`, `BackgroundLearner`, `extract_edits`, `run_learner`, `resolve_relative_dates`, `parse_memory_edits`, `with_backoff`) |
| Entry point | `knowledge_graph/memory/cli.py` (`agent-utilities-memory learn`) |

## Wiring (≤3 hops)

`agent-utilities-memory learn` → `run_learner` → `engine.add/update/delete_memory_node` (3 hops, verified).

## Research provenance

Quarq async learner — `agent-oss/agent.py:99-160, 2951-3007, 3303/3646`; typed-extraction enhancement
— memory-os `icarus/hooks.py` (`ClaudioDrews/memory-os@a4ca094`).
