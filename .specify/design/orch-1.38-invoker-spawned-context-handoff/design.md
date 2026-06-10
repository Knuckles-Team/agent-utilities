# Design: Invoker→Spawned-Agent Shared Context Handoff (ORCH-1.38)

> A capability so an invoking agent can **curate context, persist it to the epistemic-graph,
> and have the spawned agent reference it** — giving the spawned agent the right model + tools +
> prompt **and the right context**, as determined by the invoker. Extends ORCH-1.21 (KG-to-LLM
> Execution Bridge) and composes with KG-2.1/2.12 (memory + memory-first retrieval).

## Investigation findings (current state)

**Today an invoker CANNOT pass curated context to a spawned agent except by inlining it into the
`task` string.** `run_agent`/`Orchestrator.execute_agent`/`graph_orchestrate(execute_agent)` accept
only `agent_name, task, max_steps, return_mermaid` (`orchestration/agent_runner.py:37-43`,
`orchestration/manager.py:71-92`). The spawned agent's model/tools/system_prompt are reconstructed
server-side from KG metadata; `sub_agents`/`discovery_metadata` are hardcoded empty.

What already exists (and is reusable):
- **Injection sinks (intra-run):** `ExecutionManifest.context` → appended to every swarm agent's task
  (`graph/parallel_engine.py:758`); the `### CONTEXT` block reads `ctx.state.exploration_notes`
  (`graph/executor.py:1508`); `access_list` filters `results_registry` into a "PRIOR STEP RESULTS"
  block (`graph/executor.py:97-131, 716-735`); `signal_board` stigmergy. All in-memory, one run.
- **Cross-process KG/epistemic-graph store:** `Memento{source}` is a *working* write→read loop already
  injected into spawned agents (`knowledge_graph/memory/memento_compressor.py`;
  `orchestration/agent_runner.py:421-434`), keyed by `agent_name`. `RunTrace{run_id}` is created at the
  spawn boundary with a free property dict + `EXECUTED_ON` edge (`agent_runner.py:634-682`).
  `graph_write add_node` + `engine.get_node_properties` (O(1) by id) over the UDS Tokio engine
  (`backends/epistemic_graph_backend.py:926,943`) is the generic blob store; visible across processes.
- **Budgeting:** `StartupContextBuilder.build_payload(budget_chars=...)`
  (`knowledge_graph/memory/memory_engine.py:900`), `estimate_tokens` (`memory/agent_context.py:50`),
  per-model `context_window` (`models/model_registry.py:143`; 9B=64K, qwen-lite=32K).
- **Avoid:** `graph_write store_memory/recall_memory` imports non-existent
  `agent_utilities.memory.manager` (`mcp/kg_server.py:1960-1979`) — dead.

The gaps: (1) no invoker-facing `context` param; (2) no session/run-scoped, TTL'd context store
(Memento is permanent + LLM-compressed); (3) no spawn-time read+budget hook that injects the
invoker's context into the spawned agent's window respecting the target model's size.

## Proposed design — three thin layers over existing primitives

### A. Persistence: session/run-scoped `ContextBlob` node (epistemic-graph)
- A `ContextBlob{ id: "ctx:{session_id}[:{key}]", content, content_type, created_at, ttl_s,
  producer_agent, session_id }` node, written **verbatim** (no lossy LLM compression — curated blobs
  are authoritative) via the existing `graph_write add_node` / `engine.add_node`.
- Link `(:RunTrace {run_id})-[:HAS_CONTEXT]->(:ContextBlob)` at spawn for provenance + lifecycle.
- TTL/eviction is the only genuinely new mechanism (a pruner pass keyed on `created_at + ttl_s`),
  reusing the backend `prune` hook (`backends/base.py`).
- Generalizes the proven `Memento{source}` pattern: same write/read shape, but keyed by
  `session_id`/`run_id` instead of `agent_name`, verbatim, with TTL.

### B. Handoff API: an explicit invoker `context` channel
- Add optional `context: str | dict | None` (and/or `context_ref: str` pointing at a `ContextBlob` id)
  to `run_agent`, `Orchestrator.execute_agent`, the `graph_orchestrate(execute_agent)` MCP handler,
  and `ExecutionManifest.context` (swarm). Backward-compatible (defaults None).
- Thread it once into a new `GraphState.invoker_context: str` (seed from `config` at GraphState
  construction, `orchestration/engine.py:343/400`), mirroring how `exploration_notes` flows. Every
  spawn path already reads `ctx.state`, so one field feeds them all.

### C. Spawn-time read + budget + inject
- At each spawn assembler, pull the invoker context (from the passed blob or by `ctx:{session_id}` id
  from the engine — same fetch shape as `get_recent_mementos`), budget it, and inject:
  - **small** curated context → `### INVOKER CONTEXT` system-prompt section
    (`graph/executor.py:1508`, the MCP persona block ~`executor.py:691`, dynamic KG-Prompt spawn
    `_router_impl.py:1265`). Pinned, immune to eviction — right for compact, durable directives.
  - **large** curated context → seeded `message_history` synthetic prior turn
    (`executor.py:1560-1564` / `1011-1013`; the dynamic spawn currently passes none) — naturally
    compactable by `MementoCompaction`/`ToolOutputEviction` when budget is exceeded.
- **Budget vs model window:** resolve the chosen specialist model's `context_window` from the
  registry (qwen3.5-9b=64K, qwen-lite=32K), reserve a fraction (≈15%) for invoker context minus an
  `estimate_tokens` estimate of the already-assembled persona/tools/notes, convert to chars, and run
  through `StartupContextBuilder.build_payload(budget_chars=...)`; if still over, summarize via
  `recursive_reasoner_tool` rather than hard-truncate. Smaller window (qwen-lite) summarizes sooner.
  This respects BOTH the invoker's and the spawned model's constraints.

### D. Additional shareables beyond curated context (honest value assessment)
The handoff envelope the invoker passes to the spawned agent should carry more than free-form
context. Ranked by value:

- **Token/cost budget (HIGH).** Invoker passes a remaining-budget so the sub-agent self-limits
  (compose with OS-5.2 cognitive scheduler / `AGENT_TOKEN_QUOTA`, and the `UsageLimits` added in
  ORCH-1.37). Stops a sub-agent consuming the whole budget. Add `budget_tokens` to the handoff →
  thread to the spawn's `UsageLimits`/scheduler.
- **Tool-permission scope (HIGH).** Invoker declares which tools the sub-agent MAY use
  (least-privilege allow-list), enforced via the existing permissioning layer
  (`object_permissioning`, `PERMISSIONS_SIGNING_KEY`, tool-guard). Security + focus. Add
  `allowed_tools`/`tool_scope` to the handoff → intersect with the resolved toolset at spawn.
- **Goal / success-criteria (MEDIUM).** The invoker's higher-level objective so the sub-agent stays
  aligned; can ride in the context blob or a dedicated `goal` field.
- **Retrieved-context references (MEDIUM).** Pass *pointers* to KG nodes the invoker already
  retrieved (ids), so the sub-agent skips re-retrieval — saves latency + tokens. Composes directly
  with the `ContextBlob` design (a list of node ids the spawned agent hydrates on demand).
- **Provenance linkage (LOW / already done).** `run_id`/`trace_id` already links sub-agent work to
  the invoker via `RunTrace` (`agent_runner.py:634-682`); surface it, don't rebuild it.

### E. Credential / auth sharing — mostly already handled; one real gap
- **Static creds (MCP servers, connector DSNs, API keys): ALREADY shared — no re-auth, no new
  mechanism.** Spawned MCP stdio servers inherit the full secrets env (`SECRETS_BACKEND`,
  `SECRETS_VAULT_URL`, `AGENT_API_KEY`, `{ALIAS}_DSN`); connectors resolve from the same OpenBao/
  Vault backend (OS-5.1). The sub-agent pulls the same secrets the invoker uses.
- **Real gap: ephemeral/runtime-acquired tokens** (interactive OAuth, short-lived delegated JWTs)
  are NOT propagated. Handoff via the **secrets layer, NOT the context blob**: invoker writes the
  token to Vault/SecretsClient under `cred:{session_id}` with a short TTL and passes the sub-agent a
  **reference id** (gated by permissioning); the sub-agent resolves it from the secrets backend.
  **Never put raw secrets in a graph node / context blob.** Value: high for OAuth/interactive flows,
  near-zero when everything is already in Vault. Build only when an interactive-token pattern exists.

### F. Bidirectional invoker↔spawned message channel — DEFERRED (honest: usually not worth it)
For the common request/response + DAG-wave patterns, a live message bus is complexity without
payoff (LLM sub-agents don't poll/subscribe mid-run well; you'd own concurrency/ordering). Existing
primitives already cover the realistic needs: `elicitation_queue` (sub-agent → invoker/user
questions, `models/agent.py:17`), `signal_board` stigmergy (in-process peer pub/sub), the A2A
protocol (`protocols/a2a.py`), the epistemic-graph **EventBus** (cross-process pub/sub substrate),
and DAG dependency outputs. **Recommendation: do NOT build a general bus speculatively.** Only if a
concrete pattern emerges (streaming progress, mid-run steering, cross-process peer negotiation) add a
**thin session-scoped append-only `Message` relation** on the epistemic-graph (or reuse the EventBus)
keyed by `session_id` — small and targeted. Until then, lean on `elicitation_queue` + A2A.

## The handoff envelope (consolidated)
A single optional `SpawnContext` passed at the entrypoint, all fields optional/backward-compatible:
`{ context: str|dict, context_refs: [node_id], goal: str, budget_tokens: int,
  allowed_tools: [str], cred_ref: "cred:{session_id}" }`. Threaded into `GraphState` and consumed at
the spawn assemblers (budget→UsageLimits, allowed_tools→toolset intersection, context/refs→prompt or
seeded message_history, cred_ref→secrets resolution). The message channel is explicitly out of this
envelope (deferred per F).

## Phased plan
1. **MVP (in-process, smallest):** add the `SpawnContext` param (start with `context` + `goal` +
   `budget_tokens` + `allowed_tools`) → `GraphState.invoker_context` (+ budget/allowed_tools fields) →
   inject `### INVOKER CONTEXT` at `executor.py:1508` + `_router_impl.py:1265` (budgeted to the model
   window), feed `budget_tokens` to the spawn `UsageLimits`, and intersect `allowed_tools` with the
   resolved toolset. Validate end-to-end via direct `run_agent` (no new storage yet). Unit + live test.
2. **Cross-process persistence:** `ContextBlob` node + `graph_context` MCP action (put/get/by-ref),
   `RunTrace -[:HAS_CONTEXT]->` link, TTL pruner, plus `context_refs` hydration. Lets a separately-
   spawned MCP/A2A agent read curated context + retrieved-node pointers by id.
3. **Swarm + message_history:** wire `ExecutionManifest.context` from the entrypoint; large-context
   path via seeded `message_history`.
4. **Ephemeral credential reference (only when an interactive-token pattern exists):** `cred_ref`
   resolved via SecretsClient/Vault under `cred:{session_id}` (short TTL, permissioned) — never raw
   secrets in graph. Static creds already flow via the shared Vault/env backend (no work needed).
5. **Hardening:** budgeting helper extracted + reused; provenance surfaced in RunTrace; OWL layer for
   `ContextBlob`/`HAS_CONTEXT` per the constitution.

**Explicitly deferred (per §F):** the bidirectional invoker↔spawned message channel — not built
speculatively; lean on `elicitation_queue` + A2A + EventBus until a concrete streaming/steering/peer
pattern justifies a thin session-scoped `Message` relation.

## Concept & wiring
- **Proposed CONCEPT:ORCH-1.38** — sub-concept of ORCH-1.21 (execution bridge); composes with
  KG-2.1 (memory) + KG-2.12 (memory-first retrieval). Wire-First: ≤2 hops from
  `graph_orchestrate(execute_agent)` → `run_agent(context=...)` → spawn assembler.
- Schema addition (`ContextBlob`, `HAS_CONTEXT`) → consider OWL ontology entry (constitution).

## Critical files
- `orchestration/agent_runner.py` (entrypoint `context` param; RunTrace anchor; existing memento-read pattern)
- `orchestration/manager.py`, `mcp/kg_server.py` (graph_orchestrate handler; new `graph_context` action)
- `graph/state.py` (`invoker_context` field), `orchestration/engine.py` (seed from config)
- `graph/executor.py`, `graph/_router_impl.py` (spawn-time inject hooks)
- `knowledge_graph/memory/memory_engine.py` (`build_payload` budgeter), `memory/agent_context.py` (`estimate_tokens`), `models/model_registry.py` (`context_window`)
- `knowledge_graph/memory/memento_compressor.py` (pattern to generalize), `backends/epistemic_graph_backend.py` (`add_node`/`get_node_properties`/`prune`)
