# Design Document: While failed-over, the embed capacity guard keys off the FALLBACK endpoint's own config, not the primary's

CONCEPT:AU-KG.ingest.keys-off

> `agent_utilities/core/config.py:2588-2600` (the `embedding:fallback`
> model-key resolution), `agent_utilities/knowledge_graph/enrichment/semantic.py:66-90`
> (`_joint_budget_cap` — the consumer of that resolution), `docs/architecture/intelligent-ingestion.md:148`.

## Decision — resolving `model="embedding:fallback"` returns the fallback endpoint's own `ChatModelConfig`/`EmbeddingModelConfig` (its own `gpu_group`/`max_concurrent_requests`), not the primary's

`config.py:2587-2596` resolves `"embedding:fallback"` as a first-class model
key: `primary = self.default_embedding_model; cfg = primary.fallback`. The
comment states why this matters: **"the WHOLE capacity guard —
server_ceiling, adaptive capacity, gpu_group budget — keys off the FALLBACK
endpoint's config (its own gpu_group / max_concurrent_requests) while
failed-over, so fallback embeds inherit the shared GPU's joint budget and
can't OOM it."**

**The rejected alternative is keeping the capacity guard keyed off the
PRIMARY's config even while traffic has failed over to the fallback
endpoint.** It is the simpler implementation — one config object governs
capacity regardless of which endpoint is actually serving — and it loses
because the primary's capacity numbers (its own dedicated GPU, its own
`max_concurrent_requests`) describe a DIFFERENT physical resource than the
fallback endpoint the traffic is actually hitting. If the fallback endpoint
shares a GPU with another service (the generator model, per
`semantic.py:75-89`), governing its concurrency by the primary's numbers
means the fan-out could exceed what the shared accelerator can actually
absorb — an OOM risk on hardware the guard was never actually measuring.

`semantic.py:66-90`'s `_joint_budget_cap` is the consumer that makes this
concrete: the embed fan-out normally passes an explicit `capacity` (a
CPU/load-derived anchor) to `map_concurrent_sync`, which **bypasses**
`resolve_capacity()` and therefore the per-GPU joint budget — "exactly right
for the PRIMARY embedder on its own dedicated endpoint (no contention)." But
while failed over to a GPU shared with the generator, the joint budget MUST
govern, so `resolve_capacity(model_key)` is consulted specifically to seed
the group's priority peers and apply the budget. When no GPU budget is
configured (the primary's own endpoint), `group_allowed` is `None` and the
clamp is a no-op — "zero regression" on the unfailed-over path
(`semantic.py:83-89`).

## Risk Assessment

- **Blast Radius**: `core/config.py` (`resolve_model`/model-key lookup),
  `enrichment/semantic.py` (`_joint_budget_cap`, the embed fan-out path).
- **Backward Compatible**: Yes — the primary (non-failed-over) path is
  explicitly a no-op regression-wise.
- **Breaking Changes**: None.
- **Known weak point**: correctness depends on `default_embedding_model.fallback`
  always carrying its OWN accurate `gpu_group`/`max_concurrent_requests`
  rather than inheriting/copying the primary's at config-authoring time — a
  misconfigured fallback endpoint config would silently under- or
  over-constrain concurrency with no runtime signal distinguishing it from a
  correctly configured one.
