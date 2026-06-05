# Comparative Analysis — Kimi Agent Swarm vs agent-utilities

**Mode:** Lightweight (article/system → codebase innovation extraction).
**Sources:** three articles on **Kimi K2.6 Agent Swarm** (Moonshot AI): the A–Z guide, the
"300 agents built a SaaS" build log, and the "How to Build AI Agent Swarms" guide. Underlying
papers referenced: Kimi K2 (MuonClip, arXiv 2507.20534), **PARL** (Parallel-Agent RL, arXiv
2602.02276), Mooncake (KVCache-disaggregated serving, arXiv 2407.00079).
**Target:** `agent-utilities` (primary), enhancing the `graph_orchestrate` (kg_orchestrate) MCP tool.
**Date:** 2026-06-05.
**Status: IMPLEMENTED** as **CONCEPT:ORCH-1.32 — KG-Governed Agent Swarm** (SWARM-1…7). See
`docs/pillars/1_graph_orchestration/ORCH-1.32-KG_Governed_Agent_Swarm.md` and
`tests/unit/graph/test_orch_1_32_kg_governed_swarm.py`. SWARM-8 (native multi-format rendering)
deferred to an ECO-4 peripheral.

---

## 1. What Kimi Agent Swarm is (and is not)

Kimi Agent Swarm is a **trained-in** behavior of the K2.6 model (1T-param MoE), not an
application-layer framework: an orchestrator decomposes one goal into a dependency graph, spawns up
to **300 sub-agents** across **4,000 coordinated steps**, runs them in **dependency-ordered parallel
waves** (each sub-agent: bounded context + structured output + scoped tools), and a coordinator
**synthesizes** one deliverable. Three load-bearing ideas:

1. **PARL** — the orchestrator is RL-trained (subagents frozen) against three rewards: *parallelism*
   (spawn concurrently), *finish* (no do-nothing agents), *performance* (output quality). The
   optimized metric is **critical-path length, not total steps** — shorten the longest dependency
   chain, which is what actually cuts wall-clock.
2. **Mooncake** — KV-cache-disaggregated serving (prefill/decode split; KV evicted to DRAM/SSD and
   recalled) is *why* 300 parallel agents don't melt the GPUs.
3. **The hybrid** the articles actually recommend: **Opus 4.8 plans + verifies, Kimi swarm
   executes** — brain decomposes into a JSON task tree, hands off, swarm builds, brain reviews each
   leaf against the spec and assembles. "The loop only closes when something with real judgment signs
   off."

**Not adoptable here:** PARL training and Mooncake serving are *model/infra* layers — agent-utilities
orchestrates hosted/API pydantic-ai agents and does not train K2.6 or run vLLM. What **is** adoptable
is the *architecture* (decompose → parallel waves → synthesize → verify) and the *six building
blocks + seven guardrails* the third article codifies.

## 2. Extend-Before-Invent — agent-utilities already has ~85% of the swarm engine

The dominant finding (verified against code, not the Explore summary alone):

| Kimi building block | Already in agent-utilities | Evidence |
|---|---|---|
| **Orchestrator / parallel waves** | **`ParallelEngine.execute`** (ORCH-1.8) — `_schedule_waves` → `rx.topological_generations(dag)` (dependency waves) → `asyncio.gather(..., return_exceptions=True)` under a `CognitiveScheduler` semaphore | `graph/parallel_engine.py:96,161,347,397,500` |
| **Goal → task DAG decomposition** | `graph/planning` `decompose(goal)→Plan`; `hierarchical_planner.planner_step` emits `parallel=True` subtask partitions (e.g. WideSearch→batches); `manifest_generators.py` builds `ExecutionManifest` | `graph/planning/__init__.py:165`, `hierarchical_planner.py:126,224`, `manifest_generators.py` |
| **Sub-agents w/ bounded context + tools** | `AgentSpec` + `executor._execute_agent` (per-agent tools, timeout); **Memento (KG-2.20, just shipped)** gives each agent bounded context via block-evict | `graph/executor.py`, `capabilities/memento.py` |
| **Coordinator / synthesis** | `ParallelEngine._synthesize` — flat (≤10), hierarchical (≤50), **RLM-programmatic (>50)**; `enrichment/synthesize.py` | `graph/parallel_engine.py:227` |
| **Memory (shared state)** | the **KG itself** — durable graph state, bi-temporal (KG-2.11), provenance; manifests/results persisted | `_persist_execution`, KG backends |
| **Handoffs / routing** | OWL-driven **`CapabilityIndex.designate()`** + `graph/routing` strategies, `coordination.py` protocol selection | `retrieval/capability_index.py`, `graph/routing/` |
| **Guardrails: cost/budget** | **`UsageGuard`** (`estimated_cost_usd > cost_limit or total_tokens > token_limit` → raise); `budget.max_cost_usd` enforced in router | `graph/lifecycle.py:86`, `graph/state.py:413`, `graph/_router_impl.py:777` |
| **Guardrails: HITL approval** | `ApprovalManager` (async pause/resume; pydantic-ai `DeferredToolRequests`); `graph_orchestrate` has `request_approval`/`grant_approval` endpoints | `observability/approval_manager.py`, `mcp/kg_server.py:1151,1167` |
| **Guardrails: depth/timeout/policy** | `RLMConfig.max_depth`, router/agent timeouts, `security/guardrails.py` (PII, token, output) | `rlm/config.py`, `security/guardrails.py` |
| **Async job dispatch + status** | `graph_orchestrate` `dispatch`/`status` endpoints (non-blocking jobs + polling) | `mcp/kg_server.py:1125,1142` |
| **PARL-adjacent** | `graph/reward_decomposition.py` (`decompose` of reward signals) | `graph/reward_decomposition.py:325` |
| **RLM parallel sub-calls** | `RLMEnvironment.run_parallel_sub_calls` (`asyncio.gather`) — model-driven decomposition | `rlm/repl.py` |

**agent-utilities' decisive advantage over Kimi:** Kimi's swarm is a **black-box trained behavior** —
opaque decomposition, no capability ontology, no durable provenance, no governance hooks.
agent-utilities orchestrates the same shape but **grounded in the KG + OWL**: capability-typed
designation, ontology-reasoned routing, bi-temporal provenance, HITL approval gates, per-session cost
guard, and now **Memento context compaction** so each long-running sub-agent stays under budget.
"Far surpasses the article" is not about *more agents* — it's about a **governed, KG-grounded,
verifiable swarm** where Kimi has raw throughput.

## 3. The genuine gaps (what to build — the deltas, not the engine)

| id | gap (vs Kimi) | extends | wire (≤3 hops) | success metric | L/E/R |
|---|---|---|---|---|---|
| **SWARM-1** (P0) | **One-shot goal→swarm action** on `graph_orchestrate`: `action="swarm"` taking a one-line goal → `decompose` → `ParallelEngine` waves → `_synthesize`, returning a single deliverable. Today decomposition + execution exist but aren't fused behind one kg_orchestrate action with the planner→execute→synthesize loop pre-wired. | ORCH-1.8/1.1, `graph_orchestrate` | `mcp/kg_server.py graph_orchestrate(action="swarm")` → `planning.decompose` → `ParallelEngine.execute` (2 hops) | one prompt → synthesized multi-part deliverable; ≥1 KG-persisted run | 5/3/3 |
| **SWARM-2** (P0) | **Closed planner→execute→verify loop** (the article's "brain signs off"). Add a verify pass: each leaf checked against its `success_criteria`; drift → re-dispatch (bounded). We have `adversarial_verifier` (AHE-3.1) + `verification.py` but no orchestrate-level critic-refiner that gates the final assembly. | AHE-3.1, ORCH-1.8 | `ParallelEngine` post-wave → verifier → conditional re-dispatch | % leaves passing `success_criteria` before assembly; bounded re-dispatch | 5/3/3 |
| **SWARM-3** (P1) | **Critical-path scheduling metric (PARL insight):** expose/optimize the longest dependency chain, not raw concurrency; report critical-path length + parallelism ratio per run. | ORCH-1.8 `_schedule_waves` | inside wave scheduler (0 hops) | wall-clock ≈ max-chain not sum; metric surfaced in result | 3/2/2 |
| **SWARM-4** (P1) | **Per-agent structured-output enforcement** (Kimi guardrail #3): each `AgentSpec` carries an output schema; sub-agent forced to pydantic `output_type`; prose from intermediates rejected. | ORCH-1.8 `AgentSpec`/executor | `executor._execute_agent` | 100% intermediate outputs schema-valid or quarantined | 4/2/2 |
| **SWARM-5** (P1) | **Retry-with-backoff + failed-agent reassignment** (Kimi 12h fault tolerance). Circuit breaker exists; add exponential backoff + reassign on stall (distinct from breaker's disable). | ORCH-1.8 scheduler | wrap `_execute_agent` | transient failures auto-recover; no run aborted by one agent | 3/2/2 |
| **SWARM-6** (P1) | **Heterogeneous-model swarm (Claw Groups):** per-`AgentSpec` model role (Opus=plan/verify, light=bulk, local=cost-sensitive) via the existing `ModelRole` routing — the article's model-diversity pattern, KG-routed. | ORCH-1.27 (model roles) + ORCH-1.8 | `AgentSpec.model` → `model_registry` role routing | mixed-model run; cost ↓ vs all-frontier at parity | 4/2/2 |
| **SWARM-7** (P2) | **Explicit scale ceilings + telemetry:** document/raise `CognitiveScheduler` max_concurrent toward the 300/4,000 envelope; per-wave cost/latency/parallelism telemetry on the result (today UsageGuard caps but doesn't surface per-wave). | ORCH-1.8 | scheduler config + result schema | 300-agent manifest runs within cost guard; telemetry emitted | 3/3/3 |
| **SWARM-8** (P2, future) | **Native multi-format deliverable rendering** (Kimi's PDF/PPT/Excel/web-in-one-run). Likely an ECO-4 ecosystem peripheral (deck/sheet writers) the synthesis step can emit. Out of scope for the orchestration core; note. | ECO-4 | synthesis emit-hooks | — | 2/4/3 |

**Build order:** SWARM-1 → SWARM-2 (the loop the article says "most setups skip") → SWARM-4 ∥ SWARM-6
∥ SWARM-3 → SWARM-5 → SWARM-7. SWARM-8 deferred to an ecosystem peripheral.

New umbrella concept proposal: **ORCH-1.32 — KG-Governed Agent Swarm** (verify next-free id at build
time), extending ORCH-1.8 (ParallelEngine) + ORCH-1.1 (planner) + ORCH-1.27 (model roles); surfaced
through the existing `graph_orchestrate` MCP tool.

## 4. Honest boundaries

1. **No PARL training / no Mooncake.** We adopt the architecture + guardrails, not the trained
   orchestrator or KV-disaggregated serving. Our orchestrator is rule/LLM-driven decomposition, not
   RL-optimized — a deliberate trade (transparent + governed vs. trained + opaque).
2. **"300 agents" is a ceiling, not a guarantee.** Like the article warns, quality scales with
   decomposition + verification, not agent count. SWARM-2's verify loop is the value, not SWARM-7's
   number.
3. **Synthesis bottleneck** (the article's noted weakness of centralized coordinators) is real; our
   RLM-programmatic synthesis (>50 agents) + hierarchical reduce is the mitigation.

## 5. Recommendation

**Adopt as an enhancement of `graph_orchestrate`, not a new engine.** agent-utilities already has the
parallel DAG executor, decomposition, synthesis, cost guard, HITL, and KG/OWL governance Kimi lacks.
The high-value deltas are the **one-shot goal→swarm action (SWARM-1)** and the **planner→execute→
verify loop (SWARM-2)** — the exact step the articles say "most throw-more-agents setups skip, and
it's why they produce impressive-looking garbage." Layering those on top of the KG+OWL orchestrator,
with per-agent schemas, heterogeneous models, and critical-path scheduling, yields a **governed,
verifiable, KG-grounded swarm** that out-classes a black-box trained one. Full SDD to follow on
approval; this pass delivers the analysis + ledger.
