# Single-GPU LLM serving — tuning for extraction throughput

This guide consolidates what actually moves throughput when serving a single
mid-size GPU (e.g. an L4-class card, or our GB10 host) for the fact-extraction
workload (KG-2.64) and other JSON-structured generation. The findings are
distilled from an empirical benchmark sweep on an L4 serving a 35B-A3B MoE model;
they generalize to any bandwidth-bound single-stream decode.

> **Why this matters here.** Fact extraction is a long structured-JSON generation
> on one contended GPU slot (KG-2.65 schedules access to it). The knobs below are
> the difference between ~56 and ~76 tok/s with *no quality loss* — a 34% wall-clock
> win on every extraction job.

## The one-line summary

**Low-bit k-quant + speculative decoding are synergistic, not independent.** On a
bandwidth-bound single stream, the win comes from doing *fewer, cheaper* memory
passes per accepted token — quantization shrinks the pass, speculation amortizes
it. Neither alone reaches the ceiling.

## What works (apply these)

| Lever | Setting | Effect | Why |
|---|---|---|---|
| **Quantization** | Q3_K_XL (~3.5 bpw) over Q4_K_XL | **+34%** decode tok/s, zero quality loss | Decode is memory-bandwidth bound; smaller weights = faster passes. |
| **Speculative decoding** | MTP / `draft-mtp`, `n-max 3`, `p-min 0.1` | +39% at Q3 (only +13% at Q4) | Drafts cheap tokens, verifies in one pass; the win scales with how cheap the pass is (so it compounds with low-bit). |
| **Flash attention** | on, **uniform KV precision** | baseline-critical | Mixing KV precisions *disables* flash-attn → **−57%**. Keep KV uniform. |
| **Parallelism** | `--parallel 1` | best single-job throughput | BS=1 MoE decode already saturates bandwidth; extra streams *split* it (−10%). |
| **KV-cache reuse** | reuse across rounds on the same doc prefix | faster rounds 2+ | The prompt+document prefix is identical across our multi-round recall (KG-2.64); cache it. |

## What does NOT help (don't bother)

- **KV-cache quantization** — KV is tiny next to weights; no measurable effect, and
  mixed precision kills flash-attn.
- **Smaller context window** — same reason; the weights, not the KV, fill VRAM.
- **More draft tokens** (`n-max 4/5`) — acceptance falls, verify cost rises (−2% / −7%).
- **Aggressive draft gating** (`p-min 0.5`) — drops baseline facts for a −2% net.

## The quality floor (do not cross)

- **~3.5 bpw (k-quant) is the floor.** Below it, completeness and groundedness
  collapse: IQ3_XXS (3.1 bpw) was faster but lost ~32% of extracted facts;
  Q2_K_XL fabricated (groundedness 0.67 → 0.26). **i-quants fail this workload.**
- Validate any new quant against extraction *coverage* (facts per doc) and
  *groundedness* (evidence_span actually substring-matches), not just tok/s.

## Applying it to our stack

- We serve via **vLLM** (`vllm.arpa`), not llama.cpp — the *knowledge* transfers,
  the flags differ. The equivalents: pick a ~Q4/AWQ-or-better quant that holds
  groundedness, enable speculative decoding where the engine supports it, keep one
  high-throughput stream per extraction job, and let KG-2.65 serialize slot access
  rather than running concurrent decode streams.
- The fact extractor already sets the matching sampling profile (temperature 0.7,
  top_p 0.8, top_k 20, presence_penalty 1.5, `enable_thinking=False`) and a strict
  JSON-schema response format — see `knowledge_graph/extraction/fact_extractor.py`.
- The single-GPU slot is a *scheduled* resource: submit extraction as a job
  (`graph_ingest action=extract_submit`) so preemption + backfill keep the GPU
  busy without oversubscription.
