# Comparative Analysis — Memento vs agent-utilities

**Mode:** Lightweight (research-paper → codebase innovation extraction).
**Source (pinned):** *Memento: Teaching LLMs to Manage Their Own Context*, Kontonis et al.,
Microsoft Research AI Frontiers (2026). `prompts/conversations/old/memento.pdf` + the OpenMementos
launch article. Claims verified against the PDF (pp. 1–6) and the published blog.
**Target:** `agent-utilities` (primary). Concept registry `docs/concept_map.md`; C4 at
`docs/pillars/architecture_c4.md`.
**Date:** 2026-06-05.

---

## 1. What Memento actually is (verified against the paper)

Memento teaches a **model** (via SFT) to segment its own chain-of-thought into semantically
coherent **blocks**, compress each completed block into a dense **memento** (a *lemma* — exact
formulas, key values, decisions, current state; NOT a human summary), then **mask the raw block**
from attention and reason forward from `mementos + current block`. Verified numbers:

| Claim | Paper value | Status |
|---|---|---|
| Peak KV cache reduction | ~2–2.5× | verified (Fig 1, p.1; abstract) |
| Throughput | up to ~1.75–2× (4,290 vs 2,447 tok/s, B200) | verified (blog; abstract "up to 2×") |
| Trace-level compression | ~6× (~10,900 → ~1,850 tok/trace; per-block ~5–20×) | verified (p.6 §Dataset stats) |
| **Dual information stream** | removing the implicit KV channel = **−15pp** AIME'24 (66.1→50.8) | verified (p.2 §Intro; §6.2.1) |
| Judge-refine memento quality | single-shot **28%** → 2 iters **92%** pass (rubric ≥8/10) | verified (p.5 §Stage 4) |
| Accuracy gap (32B) | 2.6pp on AIME'26 at ~2× KV cut; shrinks with scale, closes with RL | verified (p.2) |

**The data-gen pipeline (OpenMementos, 228K traces):** (1) sentence/atomic splitting →
(2) **LLM boundary scoring** 0–3 (local question, LLMs do well) → (3) **DP segmentation**
maximizing boundary quality − λ·(σ/μ size penalty), min 200 tok/block → (4) **compressor LLM**
("STATE-COMPRESSOR: minimize tokens subject to fully capturing all logically relevant information")
→ (5) **judge LLM** scoring 6 dims (formulas-verbatim, values, methods, validation, no-hallucination,
result-first), τ=8, ≤2 refine iterations.

**Training:** two-stage curriculum — Stage 1 full causal attention (learn format), Stage 2 block
masking (compression pressure). ~30K samples suffice.

## 2. The honest adoptability boundary (TRIZ "separation in space")

agent-utilities is a **Python agent-orchestration framework** over pydantic-ai talking to
**hosted/API models**. It does **not** train models and does **not** control the inference engine's
KV cache. That cleanly partitions Memento:

| Memento element | Adoptable here? | Why / how |
|---|---|---|
| SFT curriculum teaching the *model* to compress | **No** | We don't fine-tune. (Future: could feed RLM-role fine-tuning — note only.) |
| vLLM in-place block masking / physical KV eviction | **No** | API models; no KV-cache control. |
| **Dual information stream (implicit KV channel)** | **No (inherent limit)** | Computed *inside a single forward pass*. Any orchestration-level memento is exactly the paper's **"restart mode"**, which the paper measures at **−15pp** vs in-engine masking. We must state this honestly and mitigate it (see lossless recoverability below). |
| **Block→compress→evict "sawtooth" at the orchestration layer** | **YES** | This is the paper's own flagged next application: *"Terminal and CLI agents are naturally multi-turn, where each action-observation cycle is laid out as a natural block."* |
| **DP segmentation (boundary scoring + CV size penalty)** | **YES** | A real upgrade over crude drop-middle/progressive truncation. Agent block boundaries = action-observation cycles. |
| **Judge-refine memento loop (28%→92%)** | **YES (highest leverage)** | Directly upgrades the existing single-shot compressor. |
| **Lossless recoverability** (orchestration analogue of the implicit channel) | **YES** | Keep `SUMMARIZES` pointers so evicted blocks are RAG-recoverable on demand — the external substitute for the lost KV side-channel. |

## 3. Extend-Before-Invent — agent-utilities already half-built Memento (then left it dead)

This is the dominant finding. **The compression primitives already exist** under `CONCEPT:KG-2.1`
(Tiered Memory & Context), and one of them is a *near-verbatim Memento*:

- `knowledge_graph/memory/agent_context.py`
  - `compress_to_memento()` (L1682) with `MEMENTO_SYSTEM_PROMPT` (L1669): *"state-compression Memento
    generator… NOT summarizing for a human… extract exact formulas, key intermediate values…
    reason forward from"* — this **is** a Memento compressor. Plus `_persist_memento` →
    `Memento` KG nodes, `get_recent_mementos()`.
  - `ContextCompactor` — 3 strategies (`summarize_tools`, `drop_middle`, `progressive`),
    `should_compact()` (0.8 auto-threshold), `persist_compaction`/`escalate`/`expand_summary`
    (LCM lossless Summary DAG via `SUMMARIZES` edges).
  - `AgentContextManager` — 5 elastic operators (SKIP/COMPRESS/ROLLBACK/SNIPPET/DELETE) + checkpoints.
- `knowledge_graph/memory/memory_engine.py` — `MemoryEngine.compact_if_needed()` / `force_compact()` /
  `compress_to_memento()` facade.
- `capabilities/` — the live-loop hook surface: `ToolOutputEviction` (evicts big tool outputs to KB),
  `ContextLimitWarner` (warns at 70/90% — **warns only, never compacts**), `CheckpointMiddleware`.

### Wiring reality (Wire-First audit — reachable ≠ invoked)

| Path | Status (evidence) |
|---|---|
| `agent_runner.py:411` **reads** last-3 mementos → injects as `tag_prompts["mementos"]` | **Live, but read-only** — no write/evict. The "sawtooth" comment is aspirational. |
| `observer.py:116` **writes** a memento when `len(messages) >= 5` | **Live entry (via `mcp/kg_server.py`) but BROKEN:** imports `from .memento_compressor import compress_to_memento` — **`memento_compressor.py` does not exist** (canonical def is in `agent_context.py`). The ImportError is swallowed by the surrounding `except Exception` → **0 mementos written**. Same broken import at `memory_engine.py:212`. |
| `MemoryEngine.compact_if_needed()` | Referenced **only in a docstring** — dead. |
| `engine_tasks.py:1005` `compact_thread` | Live **offline daemon** (KG thread compaction), not in-flight. |
| **In-flight block→compress→evict during the running agent loop** | **Does not exist anywhere.** No capability mutates the live `message_history` to evict compressed blocks. |

So the model's *core contribution applied to agents* — the live sawtooth — is **absent**, the one
real write path is **silently broken**, and the engine that would do it is **already written**.
Assimilating Memento here is **~80% wiring + bug-fix, ~20% new quality logic**, not a rebuild.

---

## 4. Innovation Ledger

Pillar = KG (Epistemic Knowledge Graph). All rows **extend `KG-2.1`** (similarity ≥0.7 ⇒ extend, do
not invent). New umbrella concept **`KG-2.20` — Mementified Context Management** (next free id; 2.20
& 2.22 free). Effort/Risk 1–5; Leverage 1–5.

| id | innovation | extends | wire (≤3 hops) | entry point | success metric | L/E/R |
|---|---|---|---|---|---|---|
| **MEM-0** (P0 bug) | Fix broken `memento_compressor` import (2 sites) → route memento write through canonical `agent_context.compress_to_memento`; remove the silent `except` swallow | KG-2.1 | `observer.observe_transcript` / `memory_engine.compress_to_memento` → `agent_context.compress_to_memento` (1 hop) | MCP `graph` observe; `memory learn` CLI | observer run writes **≥1** `Memento` node (today: 0) | 5/1/1 |
| **MEM-1** (P0) | **Live block-compress-evict capability** — new `MementoCompaction` capability: on `before_model_run`, when over budget, segment running `message_history` into blocks, compress completed blocks via `compress_to_memento`, **evict raw blocks**, keep `mementos + current block`. Default **ON**. | KG-2.1 (`ContextCompactor`, `compress_to_memento`) | `factory.py` `agent_capabilities` → `Agent(capabilities=)` → `before_model_run` (1–2 hops) | `agent.run` (agent_runner/manager), MCP, A2A | peak context tokens/session **−≥40%** vs baseline at **≥95%** task-success parity (mirrors paper 96.4% solve-overlap) | 5/3/3 |
| **MEM-2** (P1) | **Judge-refine memento loop** — extend single-shot `compress_to_memento` with compressor→judge→recompress (rubric: formulas/values/methods/validation/no-halluc/result-first; τ=8; ≤2 iters) | KG-2.1 | inside `compress_to_memento` (0 hops; both MEM-0 & MEM-1 callers inherit it) | same as MEM-0/MEM-1 | memento acceptance (≥8/10) **≥90%** within ≤2 iters (paper 92%); downstream success ≥ single-shot | 5/2/2 |
| **MEM-3** (P1) | **Semantic-boundary DP segmentation** — add a `memento_blocks` strategy: boundary-score action/observation cycles, DP-partition maximizing boundary quality − λ·CV(size), min-block floor; replaces "drop the middle" | KG-2.1 (`ContextCompactor` strategy enum) | `ContextCompactor.compact(strategy="memento_blocks")` (1 hop from MEM-1) | via MEM-1 capability | fewer mid-thought cuts + better block-size balance (CV) vs `drop_middle` on a fixed transcript set | 3/3/2 |
| **MEM-4** (P1) | **Lossless recoverability** (external substitute for the implicit KV channel) — every evicted block keeps a `SUMMARIZES` pointer; expose on-demand `expand_summary` so reasoning can re-fetch an evicted block (RAG fallback) | KG-2.1 (`persist_compaction`/`expand_summary` already exist) | MEM-1 evict → `persist_compaction`; recall → `expand_summary` (1 hop) | MEM-1 capability; MCP `graph` expand | **100%** of evicted blocks recoverable (pointer present); honest −15pp caveat documented | 4/2/2 |
| **MEM-5** (future, not scheduled) | OpenMementos-style data-gen (boundary-score + DP + judge-refine) to mint fine-tuning data **if** RLM roles are ever fine-tuned | ORCH-1.27/RLM | n/a (offline data tool) | — | out of scope now; tracked | 2/5/3 |

**Build order (topo):** MEM-0 → MEM-2 (improves every caller) → MEM-1 (the sawtooth) →
MEM-3 ∥ MEM-4. MEM-5 deferred.

## 5. Honest limitations (stated up front)

1. **No dual information stream.** External mementos are the paper's "restart mode" (−15pp in their
   setting). We cannot recover the implicit KV channel without owning the inference engine. MEM-4's
   lossless `expand_summary` recall is the orchestration-layer mitigation, not an equivalent.
2. **No training.** We adopt the *pattern and the data-pipeline ideas*, not the SFT/vLLM mechanism.
3. **Token estimation is heuristic** (1.33 tok/word). Budgets are approximate; fine for triggering,
   not for exact KV accounting.

## 6. Recommendation

**Adopt — high value, low risk, mostly wiring.** Memento is strong external validation of a pattern
agent-utilities already designed (`KG-2.1` compaction + `compress_to_memento` + Summary DAG) but left
unwired and partly broken. The work is: fix the silent bug (MEM-0), add the judge-refine quality loop
the paper proves is essential (MEM-2, 28%→92%), wire the live sawtooth the paper flags as the agent
application (MEM-1), and add semantic segmentation + lossless recall (MEM-3/4). Net-new concept
**KG-2.20** umbrellas it; everything **extends KG-2.1**.
