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
- **Remaining:** `allowed_tools`→toolset intersection (least-privilege; needs real toolset filtering,
  not a prompt hint); ContextBlob **TTL pruner** + **RunTrace `HAS_CONTEXT` provenance edge** (engine
  lacks `add_edge` — add it or use backend); Phase 3 swarm `ExecutionManifest.context` +
  large-context `message_history`; Phase 4 ephemeral `cred_ref` via SecretsClient; OWL layer for
  `ContextBlob`/`HAS_CONTEXT`.
- **Message channel: deferred** (elicitation/A2A/EventBus cover realistic needs).

## 8. Daemon redeploy for the latest merges
The dedicated host daemon (tmux session `graphos-host`) was started on merge `b200bbd`; the ORCH-1.38
MVP (`0bf18cb`) is merged to canonical `main` + served by the venv but **not yet loaded by the
running daemon**. Restart the tmux host daemon (or normal supervisor) to make the MVP live on the
`graph_orchestrate` path. (The host should also move from the ad-hoc tmux session back under the
normal supervisor for durability.)
