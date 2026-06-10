# Deferred Follow-Ups — ORCH-1.37 mermaid surfacing + execution-loop optimization

> Captured after committing `51db51f` (feat/orch-mermaid-surfacing). The shipped changes are
> coded, lint-clean, unit-tested, and **validated in-process** via direct `run_agent`. The items
> below are known-incomplete / intentionally deferred, plus the operational steps needed to make
> the changes live on the `graph_orchestrate` MCP path.

## 1. Daemon runs the new code (BLOCKS live effect — highest priority)
`graph_orchestrate` (execute_agent/swarm/workflow) is **dispatched to the running graph-os daemon**
(observed pid 706258), which executes the *installed* `agent_utilities`, NOT the worktree. Proven:
an unconditional probe at the top of `router_step` never fired via `graph_orchestrate` but fired
immediately via direct in-process `run_agent`. So every fix + perf change here is dormant on the MCP
path until the daemon runs the new code.
- **Action:** merge to main → reinstall/redeploy agent-utilities → restart the singleton daemon
  (it owns the durable tier — restart carefully, expect a brief self-heal/role re-election per the
  KG-daemon-singleton design). Then re-run the E2E harness through `graph_orchestrate` to confirm
  the fast-path + mermaid surface on the live path. Interim validation is the direct-call harness
  (`scratch/probe_direct.py`).

## 2. Structured-output for `supports_json=false` models
Both `qwen/qwen3.5-9b` and `qwen-lite` have `supports_json=false`. The planner forces
`output_type=GraphPlan`, so even the 9B sometimes emits an empty `GraphPlan` (the residual
"empty plan" escalations). The `supports_json` config flag is currently **dead** (parsed, persisted
to the KG schema, never consumed in `create_model`/the router).
- **Action:** in `_router_impl.py` planner/router agent construction (~router LLM run) and the RLM
  path, choose the pydantic-ai output mode by the flag — native/tool JSON when `supports_json=true`,
  else `PromptedOutput` (instruction-based) which small/non-JSON models handle far better. This makes
  the flag live and removes the empty-plan churn even when direct-dispatch doesn't apply (multi-server).

## 3. Host-resource rebalance (`services/vllm` + config.json)
`qwen-lite` is over-provisioned (`parallel_instances: 8`) for what is now a cheap/occasional
sub-step model, while `qwen/qwen3.5-9b` (`:4`) absorbs routing+exec+verify+KG after the `can_route`
flip. Keep the embedding endpoint (`bge-m3`) separate.
- **Action:** shift `parallel_instances` toward the 9B (and the actual vLLM replica counts in
  `services/vllm/compose.yml`), then redeploy via Portainer. Re-measure throughput under concurrency.

## 4. Explicit context bounding between re-plan attempts
Capped re-plans (OPT-3) and `UsageLimits` (OPT-5) prevent the runaway, but the planner still
concatenates `feedback/error/results/policies/process/architectural_decisions/exploration_notes`
unbounded per attempt (`hierarchical_planner.py:162-184, 235-242`; `_router_impl.py` failure_context;
`exploration_notes` only grows).
- **Action:** add a `MAX_REPLAN_CONTEXT_CHARS` budget (truncate oldest-first), cap `previous_results`
  to top-K entries, replace full prior-results with a one-line-per-node digest, and bound
  `exploration_notes`/`validation_feedback` where they are grown. Pre-flight assert the assembled
  prompt fits a safe fraction (~60%) of the model context window.

## 5. KG dual-write of the ORCH-1.37 spec (constitution post-mod mandate)
The spec/design/tasks are on disk under `.specify/` but not yet ingested into the KG via `kg_ingest`
(Post-Modification Artifact Mandate item 5). Do this once the daemon is on new code (or via a direct
ingest call) so the concept is queryable.

## 6. Full E2E harness through `graph_orchestrate` (post-daemon)
`scratch/test_workflow_e2e.py` exercises the live MCP path (ingest → execute_agent → compile/execute
workflow → swarm → capture mermaid). Re-run it after item 1 to capture *real* execution-flow diagrams
and confirm topology counts (the FieldInfo fix makes the counts work).

## 7. Shared invoker→spawned-agent context layer (ORCH-1.38)
Design: `.specify/design/orch-1.38-invoker-spawned-context-handoff/design.md`.
- **MVP (Phase 1) DONE** (merged): `context` param on run_agent/execute_agent/graph_orchestrate →
  `GraphState.invoker_context` → budgeted `### INVOKER CONTEXT` injected into every task-executing
  spawn assembler. Validated (4 unit tests + runtime prompt-capture proof).
- **Phase 3.5 DONE** (merged): `budget_tokens` → `GraphState.invoker_budget_tokens` → enforced as
  `UsageLimits.total_tokens_limit` via `executor.spawn_usage_limits()` at every spawn run site.
- **Phase 2 DONE** (merged + live): `graph_context` MCP tool (put/get/list) persists a `ContextBlob`
  node in the epistemic-graph; `context_ref` on run_agent/execute_agent/graph_orchestrate resolves a
  blob's content into the spawned agent's context. Validated (unit + live put→get round-trip).
- **allowed_tools DONE** (merged + live): `invoker_allowed_tools` → `executor.apply_tool_scope()`
  filters function tools by name + wraps toolsets with pydantic-ai `.filtered()` at every spawn site.
- **ContextBlob TTL + provenance DONE** (merged + live): `engine.add_edge()` added;
  `RunTrace -[:HAS_CONTEXT]-> ContextBlob` linked on `context_ref` use; `graph_context get` honors
  TTL (expired→gone); new `graph_context prune` action.
- **FU-2 DONE** (merged + live): planner/LATS use `PromptedOutput` for `supports_json=false` models.
- **FU-4 DONE** (merged + live): re-plan context bounded (recent-5 results, char budgets).
- **Phase 3 DONE** (live): swarm `ExecutionManifest.context` wiring (curated context → every wave agent).
- **Phase 4 DONE** (live): `cred_ref` — invoker passes a REFERENCE to a secret; resolved to the raw
  token onto the transient `AgentDeps.auth_token` at spawn (`executor._resolve_invoker_cred`), never
  stored in GraphState/graph/logs. Threaded through run_agent/execute_agent/graph_orchestrate.
- **OWL DONE** (live): `:ContextBlob` class + `:hasContext` property in `ontology_orchestration.ttl`.
- **graph_context list HARDENED**: no more cross-contaminated rows (inline-property match +
  client-side `ctx:`-prefix filter). NOTE: full session-scan is still backend-limited — the
  epistemic-graph Cypher reliably matches by node id, not by arbitrary-property scans. The reliable
  handoff path is **get/context_ref by id**. Real fix = improve backend WHERE-shape support OR
  maintain a session→blob-id index node.
- **Message channel: DEFERRED (final)** — existing `elicitation_queue` + A2A + EventBus cover the
  realistic needs, AND a graph-backed message log would hit the same backend session-scan limitation
  as `graph_context list`. Not worth a half-working implementation; revisit if a concrete
  streaming/steering/peer-negotiation pattern emerges (and after the backend Cypher improvement).

## 9. FU-3 — RESOLVED by consolidating onto the 9B (rebalance no longer needed)
Investigation showed `qwen-lite` was the *executor* model (9B routed/planned/verified, 3B executed),
and the 3B executor caused execution flakiness. Decision: **consolidate everything onto
`qwen/qwen3.5-9b`** (simpler, better execution quality, eliminates the rebalance):
- `config.json`: removed the `qwen-lite` chat model. All roles now resolve to the 9B
  (`lite_chat_model`→`default_chat_model` fallback). **Live** (daemon restarted; verified).
- `services/vllm/compose.yml`: `vllm-lite` service removed — **edit left UNCOMMITTED** for your
  review; **you do the Portainer GitOps redeploy** to free the GPU. Restore from git if cheap
  parallel fan-out is needed later.
- Optional tuning: bump the 9B `parallel_instances` / replicas if it becomes a throughput
  bottleneck now that it carries all roles.

## 10. Phase 3 (swarm context) DONE; message_history seeding deferred
`graph_orchestrate(action='swarm')` now threads invoker context/context_ref into
`ExecutionManifest.context` (ParallelEngine injects it into every wave agent). Large-context
`message_history` seeding deferred — the budgeted system-prompt injection already handles large
context functionally; message_history is a future evictability optimization.

## 8. Daemon redeploy for the latest merges
The dedicated host daemon (tmux session `graphos-host`) was started on merge `b200bbd`; the ORCH-1.38
MVP (`0bf18cb`) is merged to canonical `main` + served by the venv but **not yet loaded by the
running daemon**. Restart the tmux host daemon (or normal supervisor) to make the MVP live on the
`graph_orchestrate` path. (The host should also move from the ad-hoc tmux session back under the
normal supervisor for durability.)
