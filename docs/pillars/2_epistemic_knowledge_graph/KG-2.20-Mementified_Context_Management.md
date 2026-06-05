# KG-2.20 — Mementified Context Management

> Assimilated from *Memento: Teaching LLMs to Manage Their Own Context* (Kontonis et al., Microsoft
> Research AI Frontiers, 2026). Extends **KG-2.1** (Tiered Memory & Context).

## What it is

A **memento** is not a human summary — it is a *lemma*: a terse, information-dense compression of a
completed reasoning/conversation **block** that preserves exact formulas, key intermediate values,
commands and their outcomes, and the current execution state, so the model can reason *forward* from
the memento alone and the raw block can be **evicted** from the live context. Running this on a
multi-turn agent produces the paper's **sawtooth** context profile: tokens climb while a block is in
progress, then drop sharply when the block is compressed to a memento and evicted.

The paper teaches this skill *into a model* via SFT + a vLLM KV-cache fork (≈2–2.5× peak KV
reduction). agent-utilities runs hosted/API models, so we adopt the **pattern at the orchestration
layer** — which the paper itself flags as the prime next application: *"Terminal and CLI agents are
naturally multi-turn, where each action-observation cycle is laid out as a natural block."*

## The five pieces (MEM-0…MEM-4)

| | Piece | Where |
|---|---|---|
| MEM-0 | Canonical `memento_compressor.py` (strangled out of the 1.8k-line `agent_context.py`); fixes the previously **silently-broken** `from .memento_compressor import …` in `observer.py`/`memory_engine.py` that left the memento write path dead | `knowledge_graph/memory/memento_compressor.py` |
| MEM-1 | **Live `MementoCompaction` capability** — on `before_model_request`, when the running history exceeds budget, segment → compress completed blocks → **evict** raw blocks, keeping `mementos + current block`. Default **ON** in `agent/factory.py` (also covers the RLM multi-turn repl loop via its factory agent) | `capabilities/memento.py` |
| MEM-2 | **Judge-refine loop** — compressor→judge→recompress on a six-dimension rubric (formulas-verbatim, values, methods, validation, no-hallucination, result-first), `τ=8/10`, `≤2` iters. The paper measured single-shot mementos at **28%** rubric pass vs **92%** after two judge passes | `memento_compressor.compress_to_memento` |
| MEM-3 | **Semantic-boundary segmentation** — `boundary_score` (never cut mid-derivation; cut at turn / action↔observation boundaries) + `segment_into_blocks` (min-block floor, no tiny danglers). New `memento_blocks` `ContextCompactor` strategy is the LLM-free path | `memento_compressor.segment_into_blocks`, `agent_context.ContextCompactor` |
| MEM-4 | **Lossless recoverability** — each evicted block is persisted as an `EvictedBlock` node linked `Memento -[:SUMMARIZES]-> EvictedBlock`; `recover_evicted_block()` re-fetches it on demand | `memento_compressor._persist_memento` / `recover_evicted_block` |

## Honest limitation (why this is not the paper, end-to-end)

The paper's headline result is a **dual information stream**: because masking happens *in-place inside
one forward pass*, the memento's KV-cache entries retain implicit information from the block they
replaced — removing that channel costs **−15pp** (their "restart mode"). We do not control the
inference engine's KV cache, so an orchestration-level memento **is** restart mode and cannot
reproduce that implicit channel. MEM-4's lossless `expand`/`recover` is the substitute (the evicted
block is re-fetchable), not an equivalent. We also do not train models (the SFT curriculum and the
OpenMementos data-gen pipeline are out of scope; noted for any future RLM-role fine-tuning).

## Wiring (Wire-First)

Entry point → `agent/factory.py` registers `MementoCompaction` in `agent_capabilities`
(`memento_compaction=True` by default) → pydantic-ai `before_model_request` hook receives
`ModelRequestContext.messages` (the list actually sent to the model) → eviction transform. The memento
**write** path is also reachable from `mcp/kg_server.py` via `observer.observe_transcript`. Verified
by `check_wiring.py` (passed, 0 violations) and a `*_live_path` test that exercises the capability on
real `ModelMessage` objects.

## Success metrics

- Peak context tokens/session **−≥40%** vs no-compaction on a multi-turn run at **≥95%** task-success
  parity (the live-path test shows −77% on a synthetic 8-cycle trajectory).
- Memento acceptance (rubric ≥8/10) **≥90%** within ≤2 judge iterations (paper: 92%).
- **100%** of evicted blocks recoverable (lossless pointer present).

## Tests

`tests/knowledge_graph/memory/test_kg_2_20_memento.py` (judge-refine, segmentation, lossless recall,
capability live-path, factory-default-ON) + `test_memento_compressor.py`.
