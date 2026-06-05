# Tasks — KG-2.20 Mementified Context Management

- [x] MEM-0 Strangle memento block into `memento_compressor.py`; repoint `__init__`, `observer`,
  `memory_engine`; fix `elastic_context_manager`/`startup_context` broken facade imports.
- [x] MEM-2 Judge-refine loop (`judge_memento`, rubric, τ=8, ≤2 iters) in `compress_to_memento`.
- [x] MEM-3 `boundary_score` + `segment_into_blocks` + `plan_block_eviction`; `memento_blocks`
  `ContextCompactor` strategy.
- [x] MEM-1 `MementoCompaction` capability (`before_model_request` evict); wire into `factory.py`
  (`memento_compaction=True`); register in `capabilities/__init__.py`.
- [x] MEM-4 Lossless persist (`EvictedBlock` + `SUMMARIZES`) + `recover_evicted_block`.
- [x] Tests `test_kg_2_20_memento.py` (judge-refine, segmentation, lossless, live-path, factory-ON);
  update `test_memento_compressor.py` for lossless-by-default.
- [x] Docs: deep-dive, `concept_map.md` row, `concepts.yaml` regen (85), CHANGELOG, AGENTS regen.
- [x] `check_wiring.py` (0 violations).
- [ ] Live-LLM validation of judge-refine acceptance-rate + token-reduction on a real multi-turn run
  (deferred to the live-testing pass).
