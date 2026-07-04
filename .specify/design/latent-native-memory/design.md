# Design Document: Latent-Native Memory (KG-2.73b, AU-KG.ontology.optional-populated-from, AU-AHE.harness.empirical-evidence-that-latent)

> Distilled from arXiv:2606.09828 (Mirage). Cross-domain transfer of latent-native
> persistent memory into the world-model rollout, retrieval ranking, and a benchmark.
> Status: **implemented + verified** (worktree `feat/latent-memory-mirage`).

## KG Analysis (Extend-Before-Invent)

### Nearest existing concepts
| Concept ID | Name | Relation | Pillar |
|---|---|---|---|
| AU-KG.compute.kg-3 | LatentDynamicsModel (learned world-model backend) | **extends** (predict_latent + rollout memory) | KG |
| AU-KG.compute.first-class-action-conditioned | Action-conditioned WorldModel | extends (step/rollout cache) | KG |
| AU-KG.ontology.batch-incremental-sync-live | Object-index funnel / HNSW CapabilityIndex | **extends** (ontology-type prior) | KG |
| AU-AHE.assimilation.empirical-parity-evidence-assimilation | Assimilation empirical-parity benchmark | template/sibling | AHE |
| KG-2.3 | Embedding factory | reuse | KG |

### Extension analysis
- **KG-2.73b** — Extension Point: AU-KG.compute.kg-3 `LatentDynamicsModel`. Strategy: *augment*
  (`predict_latent`; `predict` delegates — No-Legacy). Carry/EMA-blend the latent in
  `WorldModel.step/rollout`. New concept required: Yes (new behavior, suffix ID).
- **AU-KG.ontology.optional-populated-from** — Extension Point: AU-KG.ontology.batch-incremental-sync-live `CapabilityIndex.designate`. Strategy:
  *augment* (additive ontology-type prior term, generalizing the reward blend). Types
  threaded via `add(node_type=…)` / `build_from_edges` / funnel `upsert`. New concept: Yes.
- **AU-AHE.harness.empirical-evidence-that-latent** — Strategy: *compose* (new harness module reusing AU-AHE.assimilation.empirical-parity-evidence-assimilation shapes).

### ID rationale
KG pillar minor space is exhausted at `.99`; the major digit encodes the pillar.
Letter-suffix IDs anchored to the parent (`KG-2.73b`, `AU-KG.ontology.optional-populated-from`) follow the repo
precedent (`EG-KG.domains.forensic-accounting-kernels`, `AU-ORCH.execution.subagent-pattern-variant`) and the `check_concepts` regex `[0-9A-Za-z]+`.

## C4 / data flow
See [`docs/architecture/latent_native_memory.md`](../../../docs/architecture/latent_native_memory.md).

## Wire-First (two surfaces, default-on)
- `graph_analyze action="world_model_rollout"` (MCP) + `/graph/analyze` (REST) →
  shared `_execute_tool` core → `WorldModel.from_engine(latent=True).rollout()`.
- `graph_analyze action="latent_efficiency_benchmark"` → `latent_efficiency_benchmark.run_all`.
- AU-KG.ontology.optional-populated-from prior is on by default inside `CapabilityIndex.designate` (the central
  `facade.designate` live path), neutral when no types/`prior_weight=0` (parity).

## Success metrics (met)
- Rollout drift reduced (5.10 → 3.37) with hit-rate preserved; memoryless parity asserted.
- Top-k type coherence improved (0.67 → 1.0); pure-cosine parity asserted.
- Benchmark reachable on both surfaces; 2/2 claims reproduced.
- `check_concepts` / `check_surface_parity` / `check_no_env_sprawl` / `ruff` green.

## Tasks (done)
- T1 implement KG-2.73b (world_model.py) + T2 wire MCP action + T3 tests + T4 docs.
- T1 implement AU-KG.ontology.optional-populated-from (capability_index.py, funnel.py) + T2 funnel/live wiring + T3 tests.
- T1 implement AU-AHE.harness.empirical-evidence-that-latent (latent_efficiency_benchmark.py) + T2 MCP action + export + T3 live-path test.
- Architecture doc + pillar docs + mkdocs nav + concepts.yaml regen.
