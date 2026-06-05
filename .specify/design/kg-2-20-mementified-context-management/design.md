# Design — KG-2.20 Mementified Context Management

**Concept:** KG-2.20 (extends KG-2.1). **Source:** Memento (Kontonis et al., MSR AI Frontiers 2026).
**Status:** Implemented (MEM-0…MEM-4).

## Problem

Long multi-turn agent runs accumulate context until the window drowns; today agent-utilities only
*warns* (`ContextLimitWarner`) or evicts oversized *tool outputs* (`ToolOutputEviction`). It never
compresses-and-evicts its own running reasoning blocks mid-run. A near-complete Memento compressor
already existed in `agent_context.py` but was **dead code** (broken `.memento_compressor` import) and
unwired.

## Approach (orchestration-layer Memento)

Segment the running message history into semantic **blocks** (action↔observation cycles), compress
each completed block into a dense **memento** via an LLM with a judge-refine loop, and **evict** the
raw blocks from the list sent to the model — keeping `mementos + current block`. Eviction is lossless
(evicted block persisted + `SUMMARIZES` pointer). Default ON.

## C4 (component)

- **MementoCompaction capability** (`capabilities/memento.py`) — `before_model_request` hook;
  transforms `ModelRequestContext.messages`. Registered in `agent/factory.py`.
- **memento_compressor** (`knowledge_graph/memory/memento_compressor.py`) — compress + judge-refine +
  segmentation + lossless persist/recover.
- **ContextCompactor.memento_blocks** (`agent_context.py`) — LLM-free block-aware compaction strategy.

## Data flow

`agent.run` → before_model_request → `mementoize_messages` → `plan_block_eviction`
(`segment_into_blocks` + token budget) → per evicted block `compress_to_memento`
(`compressor → judge → recompress`) → `_persist_memento` (Memento + EvictedBlock + `SUMMARIZES`) →
return `ModelRequestContext` with mementos replacing evicted blocks.

## Honest limitation

No KV-cache control ⇒ this is the paper's "restart mode" (loses the −15pp implicit dual channel).
Mitigated, not equalled, by lossless recovery. No model training (OpenMementos data-gen out of scope).

## Wiring & success metrics

See [`docs/pillars/2_epistemic_knowledge_graph/KG-2.20-Mementified_Context_Management.md`]. check_wiring
passes (0 violations); live-path test shows −77% tokens on a synthetic trajectory.
