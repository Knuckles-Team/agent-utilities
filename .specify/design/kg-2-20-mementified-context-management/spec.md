# Spec — KG-2.20 Mementified Context Management

## Requirements

1. **MEM-0** The canonical memento compressor lives in `memento_compressor.py`; `observer.py` and
   `memory_engine.py` import it successfully (no silent ImportError). The `MemoryEngine.compactor` /
   `startup_builder` facade properties resolve.
2. **MEM-1** A `MementoCompaction` capability evicts old completed blocks from the live message list
   when estimated tokens exceed `auto_compaction_ratio × budget`, replacing each with a memento
   `SystemPromptPart`. Registered in `agent/factory.py` with `memento_compaction=True` (default ON).
   The head (system prompt) and the most recent block are never evicted.
3. **MEM-2** `compress_to_memento(refine=True)` runs compressor→judge→recompress; accepts at rubric
   score `≥8/10`; `≤2` refine iterations; degrades to single-shot when no judge LLM is available.
4. **MEM-3** `segment_into_blocks` never cuts mid-derivation (boundary `0` when prior unit ends with
   `:`/`=` or next starts with a continuation word), enforces a 200-token min-block floor, and emits
   no tiny dangling block. Exposed as the `memento_blocks` `ContextCompactor` strategy.
5. **MEM-4** Eviction is lossless: `Memento -[:SUMMARIZES]-> EvictedBlock`; `recover_evicted_block`
   returns the raw block.

## Acceptance (success metrics)

- Peak tokens/session −≥40% at ≥95% task parity (live-path test: −77% on synthetic trajectory).
- Memento acceptance ≥90% within ≤2 iters.
- 100% evicted blocks recoverable.
- `check_wiring.py` passes; no new stub/sprawl/concept-gate violations.

## Non-goals

Model fine-tuning / SFT; vLLM KV-cache masking; reproducing the implicit dual channel; OpenMementos
data-generation pipeline (future, only if RLM roles are fine-tuned).
