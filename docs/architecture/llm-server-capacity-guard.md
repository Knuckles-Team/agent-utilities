# LLM / Embedding Server-Capacity Guard

> CONCEPT:ORCH-1.102 (per-endpoint server-capacity ceiling) · CONCEPT:ORCH-1.103
> (capacity-aware backpressure + circuit breaking) · CONCEPT:KG-2.298
> (`max_concurrent_requests` config).
> Composes with — does not replace — the Resource-Priority Edict
> ([`resource-priority-edict.md`](resource-priority-edict.md), ORCH-1.98/1.99),
> the adaptive concurrency controller
> ([`adaptive_model_concurrency.md`](adaptive_model_concurrency.md), KG-2.145), and
> the shared-GPU budget ([`distributed_gpu_concurrency.md`](distributed_gpu_concurrency.md),
> KG-2.146).

## The failure this prevents

Under concurrent ecosystem load — ingestion embeddings (`bge-m3`) **plus** concept/
fact enrichment (`qwen`) **plus** agent orchestration (`qwen`), all hitting the
**same** GB10 vLLM host — the GB10's **unified** memory (121 GB shared CPU+GPU) was
exhausted:

```
NVRM: Out of memory [NV_ERR_NO_MEMORY]   →  GPU/driver OOM  →  the whole HOST went unreachable
```

A 35B model on a 121 GB unified box has a **finite** concurrent-sequence budget
(KV-cache + activations). Exceed it and memory exhausts. The risk is the **remote
server's** capacity, but our concurrency limits were sized from the **local** host:

| Local-derived limit (the bug) | Where |
|---|---|
| embed fan-out ≈ `2 × compute_ingest_worker_count` (cpu/load), capped 16 | `knowledge_graph/enrichment/semantic.py::_embed_concurrency` |
| ingest write/parse pools ≈ cpu/mem anchor | `knowledge_graph/core/engine_tasks.py::compute_ingest_worker_count` |
| adaptive ramp ceiling = `MODEL_MAX_CONCURRENCY` (default **512**) | `core/model_capacity_autoscale.py` |
| `PriorityModelGate` capacity = adaptive target (could ramp to 512) | `core/resource_priority.py` |

Three demand sources, each sized locally, can sum to **hundreds** of in-flight
requests — far more than the GB10 can serve without OOM.

## The guard

### 1. A per-endpoint server-capacity ceiling (ORCH-1.102 / KG-2.298)

`server_ceiling(model)` (`core/model_concurrency.py`) is the **one** number the
remote **server** dictates — its vLLM `--max-num-seqs` / KV-cache + unified-memory
budget — resolved as:

1. the model's explicit **`max_concurrent_requests`** (config, KG-2.298) — wins
   absolutely, may be *below* the optimistic `parallel_instances × max_parallel_calls`
   product if the box genuinely can't sustain it;
2. else `max(total_capacity, MODEL_MAX_CONCURRENT_REQUESTS)` — a conservative
   default (**32**), so an under-declared model can never ramp to 512.

This ceiling clamps **both** the adaptive ramp (`resolve_capacity` →
`min(adaptive, server_ceiling)`) **and** the shared admission gate.

**All three demand sources share ONE ceiling per endpoint.** Every fan-out
(`map_concurrent` / `map_concurrent_sync`) routes through the per-model
`PriorityModelGate` sized to `server_ceiling(model)`. Enrichment and orchestration
both target `qwen` → the **same** gate key → their **sum** can never exceed the
ceiling. Embeds target `bge-m3` → a separate endpoint → its own ceiling (and the
two endpoints share the physical GB10 via the shared-GPU budget, below).

Two layers, composed:

```
demand source ─▶ circuit breaker (ORCH-1.103, back off if server is shedding)
              ─▶ PriorityModelGate(capacity = server_ceiling)   ← shared, hard aggregate cap + priority reserve
              ─▶ per-fan-out width (semaphore / thread-pool, ≤ ceiling)  ← this call's own slice
              ─▶ the call
```

### 2. Capacity-aware backpressure + circuit breaking (ORCH-1.103)

`core/model_circuit_breaker.py` — a three-state breaker per endpoint
(`ModelCircuitBreaker`), fed the same `(ok, status)` samples the fan-out already
collects:

- **CLOSED** → calls pass through.
- An **overload status** (`OVERLOAD_STATUSES = {429, 502, 503, 504, 529}`), a
  timeout, **or** an opaque-under-load failure (`ok=False` with no discernible
  status) → trips toward **OPEN** after `MODEL_BREAKER_FAIL_THRESHOLD` consecutive
  overloads (default **1** — react to the first sign). OPEN: new calls **back off**
  for an exponential-backoff cooldown (`0.5s → 1s → 2s … → 30s`) instead of hammering
  a server already over its memory budget. *Anti-retry-storm.*
- After the cooldown → **HALF_OPEN**: exactly **one** probe is admitted (reserved for
  the first caller that finds the cooldown elapsed; concurrent callers wait a short
  `0.25s` slice and re-check). Any **non-overload** outcome — a success **or a benign
  (non-capacity) error** — closes the breaker and **resets the backoff**; an overload
  re-opens it with a **longer** cooldown (`backoff_factor ^ (trips-1)`, capped at
  `max_cooldown_s`).

The breaker sits **OUTSIDE** the ceiling gate (`before_call` / `before_call_sync`
run before `priority_slot`): the ceiling decides the steady-state max; the breaker
reacts to a server *already* saturating by throttling the client to near-zero until
it recovers. One breaker per model key, so embeds + enrichment + orchestration on the
**same** endpoint trip and recover together. A saturating server slows the
**client**; it never gets crashed by it.

The read-only `is_tripped()` probe (true only while OPEN *and still within cooldown*,
and side-effect-free — it does **not** consume the HALF_OPEN probe) is what the
embedder-failover router (`embedding_failover.py`, KG-2.299) consults to route away
from a shedding primary onto its fallback, getting automatic recovery for free. See
[`distributed_gpu_concurrency.md`](distributed_gpu_concurrency.md).

### 3. The priority edict still holds (ORCH-1.98/1.99)

The ceiling gate is the **same** `PriorityModelGate`, so background ingestion still
yields its reserved headroom to interactive / orchestration / hydration. The two
**compose**: **priority decides the ORDER within the ceiling; the ceiling decides
the MAX.** Background never starves the box; interactive never waits behind it.

## Configuration

| Knob | Default | Meaning |
|---|---|---|
| `max_concurrent_requests` (per-model config) | unset | The model **server's** real safe in-flight budget (its `--max-num-seqs`). The hard ceiling. |
| `MODEL_MAX_CONCURRENT_REQUESTS` (env) | `32` | Global conservative ceiling default when a model sets no explicit value. |
| `MODEL_CIRCUIT_BREAKER` | `true` | Enable the per-endpoint breaker. |
| `MODEL_BREAKER_FAIL_THRESHOLD` | `1` | Consecutive overloads before tripping. |
| `MODEL_BREAKER_BASE_COOLDOWN_S` / `_MAX_COOLDOWN_S` / `_BACKOFF_FACTOR` | `0.5` / `30` / `2.0` | Exponential backoff envelope. |
| `gpu_group` (per-model config field) | unset → `base_url` host | Joins models on one physical GPU into one budget bucket. Set the **same** tag on endpoints served from *different* hosts that share a GPU (KG-2.146). Resolved by `Config.gpu_group`. |
| `GPU_CONCURRENCY_BUDGETS` (env, KG-2.146) | unset | Per-physical-GPU budget capping the **sum** across endpoints sharing one box. |

Resolution precedence for the per-endpoint ceiling (`server_ceiling(model)` in
`core/model_concurrency.py`): an explicit per-model `max_concurrent_requests`
(`Config.model_max_concurrent_requests`, KG-2.298) wins absolutely — it may be set
*below* the optimistic `parallel_instances × max_parallel_calls` product when the box
genuinely can't sustain it. Otherwise the ceiling is
`max(model's declared total_capacity, MODEL_MAX_CONCURRENT_REQUESTS)`, so an
under-declared model can never let its adaptive controller ramp to
`MODEL_MAX_CONCURRENCY` (512). Any config error fail-safes to the `32` default.

The ceiling is a legitimate **explicit config** (Configuration-discipline): it
reflects the *server's* capacity, which **cannot** be auto-derived from the local
host — a GB10 ≠ a Pi ≠ a cluster.

## Recommended GB10-class vLLM envelope (operator-side; we do NOT manage their compose)

Two models share the GB10's 121 GB unified memory (generator + embedder), so the
server config must leave room for **both** plus the system. Align the server's
`--max-num-seqs` with our client ceiling (**client ceiling ≤ server `--max-num-seqs`**):

```bash
# qwen generator (the 35B MoE) — the latency-sensitive model
vllm serve qwen/qwen3.6-27b \
  --gpu-memory-utilization 0.55 \   # leave headroom; the embedder + system share the 121 GB
  --max-num-seqs 32 \               # the concurrent-sequence budget; set the client ceiling ≤ this
  --max-model-len 131072            # KV-cache scales with this × max-num-seqs — keep it bounded

# bge-m3 embedder — a SEPARATE endpoint on the SAME box
vllm serve bge-m3 \
  --gpu-memory-utilization 0.25 \   # the remaining slice; 0.55 + 0.25 + system < 1.0
  --max-num-seqs 16
```

Then set, per model in our config:

```jsonc
// qwen — server says --max-num-seqs 32
{ "id": "qwen/qwen3.6-27b", "max_concurrent_requests": 32, "gpu_group": "gb10" }
// bge-m3 — server says --max-num-seqs 16
{ "id": "bge-m3",              "max_concurrent_requests": 16, "gpu_group": "gb10" }
```

`gpu_group: "gb10"` on both makes the shared-GPU budget (KG-2.146,
`GPU_CONCURRENCY_BUDGETS={"gb10": 40}`) cap their **joint** in-flight sum across the
two endpoints, so the physical box is protected even though they are different
endpoints. Rule of thumb: `Σ client ceilings ≤ Σ server --max-num-seqs`, and the
GPU-group budget ≤ what 121 GB can hold for both models' KV-cache at once.

## Validation

`tests/unit/core/test_model_server_capacity_guard.py`:

- **(a)** three concurrent demand sources hammering one endpoint never exceed its
  configured ceiling (aggregate in-flight bound);
- **(b)** a simulated `503`/`429` trips the breaker → measurable backoff + single
  half-open recovery probe (no retry storm); exponential backoff grows on repeat;
- **(c)** the priority edict holds **within** the ceiling (interactive admitted
  ahead of a saturating background fan-out, through `map_concurrent`);
- **(d)** the default ceiling is conservative (`32`) and configurable per endpoint
  (explicit `max_concurrent_requests` wins; never throttles below declared).
