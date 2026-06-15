# Comparative Analysis — Mirage (Latent Spatial Memory for Video World Models) vs agent-utilities

**Date:** 2026-06-15
**Source paper:** arXiv:2606.09828 — *"Latent Spatial Memory for Video World Models"*
(Mirage), Wang et al., cs.CV, 2026-06-08.
**Target:** `agent-utilities` @ `main` db826ca (worktree `feat/latent-memory-mirage`).
**Mode:** cross-domain innovation extraction (the paper is computer-vision / video
diffusion; the transfer is architectural, not literal).
**PDF:** `…/agent-utilities/research/papers/4fe7da000b1c_latent_spatial_memory_for_video_world_models.pdf`.

## 1. The paper in one paragraph

Video world models that keep 3D consistency across generated frames usually maintain
an **explicit point-cloud memory in RGB/pixel space**, which is expensive (repeated
render + VAE encode) and lossy (the pixel round-trip discards learned-latent features).
Mirage instead keeps a **persistent 3D cache directly in the diffusion latent space**:
it (1) lifts latent tokens into 3D by depth-guided back-projection (a *geometric prior*
structures the cache), (2) queries it by **warping the stored latents** to synthesize
novel views — no re-encode/re-render — and (3) exploits the generative model's built-in
geometric prior. Result: **10.57× faster end-to-end generation, 55× smaller memory**
vs explicit-3D baselines, SOTA on WorldScore.

## 2. Transferable principles → agent-utilities mapping

| Mirage principle | agent-utilities locus | Status before |
|---|---|---|
| Persistent cache **in latent space**, not a lossy reconstructed form | KG embeddings stored as native arrays (`EpistemicGraphBackend._embeddings`), HNSW `CapabilityIndex` | **Already strong** (at rest) |
| Carry latent **forward across steps** to stay coherent | `WorldModel.rollout()` / `LatentDynamicsModel` (KG-2.67/2.73) | **Gap** — rollout discarded the latent |
| **Geometric prior** structures the cache (depth back-projection) | `CapabilityIndex.designate` ranking; OWL/RDF ontology | **Gap** — flat cosine, ontology unused |
| **Measured** efficiency win (10.57×/55×) | harness benchmark suite (AHE-3.x) | **Gap** — no benchmark for the above |
| Query-by-warping (no recompute on read) | KG read paths | **Already satisfied** — stored latents are not re-embedded on read |

**Honest finding:** agent-utilities already stores knowledge latent-natively at rest
and does *not* round-trip stored vectors through a reconstructed surface form on read
(verified in `EpistemicGraphBackend.semantic_search` and `CapabilityIndex._rank`). So
this is not a catch-up. The genuine, non-redundant gaps were two *flows* that still
threw away latent structure, plus the missing measurement. A standalone
"eliminate re-embed-on-read" concept was **evaluated and rejected** as redundant — its
real residue lives in the rollout loop, captured by KG-2.73b.

## 3. Innovation ledger (verified, implemented)

| # | Concept | Mechanism transferred | Verification |
|---|---|---|---|
| 1 | **KG-2.73b** Persistent latent rollout memory | Carry + EMA-blend the predicted latent across rollout steps (principles 1+3) | drift 5.10 → 3.37 under fixed seed; parity proven when off |
| 2 | **KG-2.44b** Ontology-prior retrieval ranking | Re-project flat cosine through ontology type structure (principles 2+4) | top-3 type-coherence 0.67 → 1.0; pure-cosine parity when off |
| 3 | **AHE-3.48** Latent-native efficiency benchmark | Measured baseline-vs-ours lift (principle 5 methodology) | 2/2 claims reproduced; reachable via `graph_analyze` |

Concept IDs use letter suffixes anchored to the parent concept (`KG-2.73b` extends
KG-2.73, `KG-2.44b` extends KG-2.44) because the KG pillar's numeric minor space is
exhausted at `.99` and the major digit encodes the pillar — the repo's established
escape valve (`KG-2.20g`, `ORCH-1.3b`).

## 4. What was built (all default-on, two-surface, parity-safe)

- **KG-2.73b** — `LatentDynamicsModel.predict_latent()` returns the latent it already
  computes; `WorldModel.step/rollout` own a per-rollout latent cache and EMA-blend it
  (`memory_weight=0.25`, opt-out parity); `Transition` carries `latent_norm`/`drift`,
  persisted in the `WorldModelRollout` node. Surface:
  `graph_analyze action="world_model_rollout"`.
- **KG-2.44b** — `CapabilityIndex.designate()` blends an ontology-type-coherence prior
  (dominant type among the strongest cosine hits) additively with the existing reward
  EMA; node types flow in via `add(node_type=…)` / `build_from_edges` / funnel `upsert`;
  injectable `ontology_prior` seam for subsumption-aware callers; `prior_weight=0`
  restores pure cosine.
- **AHE-3.48** — `harness/latent_efficiency_benchmark.py` (reuses the AHE-3.47
  `BenchmarkResult`/`to_markdown` shapes). Surface:
  `graph_analyze action="latent_efficiency_benchmark"`.

## 5. Measured results (fixed seed, CPU, reproducible)

| Mechanism | Metric | Baseline | Ours | Lift |
|---|---|---|---|---|
| Latent rollout memory KG-2.73b | trajectory-drift (lower better) | 5.096 | 3.367 | **+1.730** |
| Ontology-prior retrieval KG-2.44b | top-3 type-coherence (higher better) | 0.667 | 1.000 | **+0.333** |

## 6. Gates & tests

- New tests: `tests/test_kg_2_73b_latent_rollout_memory.py` (8),
  `tests/retrieval/test_kg_2_44b_ontology_prior.py` (5),
  `tests/test_ahe_3_48_latent_efficiency_live_path.py` (2) — all green; 91-test broader
  slice (world-model, retrieval, assimilation, MCP server, gateway parity) green.
- Gates: `check_concepts` (290 concepts, 3 new registered), `check_no_env_sprawl` (0),
  `check_surface_parity` (0 drift / 0 new unexposed), `check_wiring`, `ruff`,
  mermaid-lint all green.

## 7. Architecture

See [`docs/architecture/latent_native_memory.md`](../docs/architecture/latent_native_memory.md)
(Mermaid flow) and the pillar entries in
`docs/pillars/2_epistemic_knowledge_graph.md` (KG-2.73b, KG-2.44b) and
`docs/pillars/3_agentic_harness_engineering.md` (AHE-3.48).
