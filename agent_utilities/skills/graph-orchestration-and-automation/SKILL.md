---
name: graph-orchestration-and-automation
skill_type: skill
description: >-
  Plan, delegate, schedule, and verify Graph-OS work across agents and workflows.
  Use for multi-step goals, agent dispatch, workflow execution, loops, schedules,
  sandboxes, run forks or replay, messaging, bus coordination, reachability,
  approvals, or parallel work with dependency-aware synthesis.
---

# Graph orchestration and automation

Convert a goal into bounded work items, execute independent work in parallel,
and verify the synthesized result.

## Direct or delegated

Use a direct tool call when the task is one bounded operation with an obvious
owner and verification. Use `graph_orchestrate` for one named agent,
`graph_agents` for a governed collective, and `graph_workflows` when dependencies
form a reusable DAG.

Do not delegate a trivial lookup. Do not keep a complex task direct merely to
avoid expressing its acceptance criteria.

## Action reference

| Tool | Actions | Notes |
|---|---|---|
| `graph_orchestrate` | `dispatch`, `swarm` (goal → decompose → parallel waves → verify → synthesize), `execute_agent`, `execute_workflow`/`compile_workflow`, `status`, `request_approval`/`grant_approval`, `consensus`/`start_debate`, `computer_use` (GUI agent on a gui-sandbox), `optimize_component` (DSPy pass), `distill_skills`, `loop_cycle`, `publish_proposal`, `failure_ingest` | the fleet execution entrypoint |
| `graph_loops` | `submit` (`objective` + `kind`=research\|develop\|skill), `list`, `run` (advance all active Loops one cycle), `drive` (run ONE Loop by id to completion, durably/resumable), `cancel`, `prioritize`, `state` (live EvolutionState — stage, saturation, backlog), `specs` (SpecProposal backlog), `review` (approve/edit/reject a distilled spec) | one entrypoint for research/develop/skill Loops, plus observing and steering the self-evolution flywheel |
| `graph_goals` / `spec_ticket` | `create` (`goal` description + `max_iterations`), `list`, `iterations` (progress for a `goal_id`), `cancel` | background/autonomous goal loops |
| `graph_schedules` | `list`, `enable`, `disable`, `prioritize` (bucket 0-3 or critical\|high\|normal\|background), `set_interval` (new `interval_s`), `run_now` | the durable scheduler — `:Schedule` nodes enqueue jobs |
| `graph_sandbox` | `status` (per-rung availability + pooled warm-parent count + per-rung reward EMA), `reap` (close idle warm parents + idle dev-workspaces), `warm` (pre-pay a `rung`'s startup so the next fan-out forks cheaply) | lifecycle + visibility over the RLM warm-fork tier (forkserver/os.fork, Wizer-warmed wasm, warm container pool, firecracker microVM); code execution itself stays inside the governed RLM loop |
| `graph_fork` | `branches_json` (JSON list of per-branch snippets) OR `code`+`n` (same snippet across n branches); `vars_json` seeds the shared namespace forked into every branch; `sandbox` optionally pins a rung | warm-fork fan-out over the ORCH-1.86..93 primitive — pay warm-up once for a parent context, fork N copy-on-write branches concurrently; degrades cleanly with no warm-fork rung on this host |
| `graph_runvcs` (live run) | `list`, `status` (event/commit/message counts + log digest), `commit` (+`label`), `revert` (+`commit_id`), `fork` (+`commit_id`, branches a NEW run into a fresh workspace, parent untouched), `discard` (drop the uncommitted event delta), `replay` (deterministically replay the event log and verify reproduction) | agent-native version control for a LIVE run — snapshots conversation + filesystem + process/event frontier together into one content-addressed `RunCommit` |
| `graph_runvcs` (Agent Digital Twin, X-8) | `twin_capture` (+`run_id`, best-effort hydrate from the KG's recorded `:ToolCall`/`:WorkItem` rows, `persist=true` default), `twin_replay` (regression-replay the recorded event log verbatim, never re-executes a real tool/model call; reports bit-for-bit `deterministic`), `twin_counterfactual` (re-drive the SAME recorded run under a swapped `policy_overrides` and/or `model_responses`, surfacing a `decision_delta`), `twin_incident` (read-only ordered step-through for human-inspectable investigation) | a durable, replayable projection of one PAST run — exact model/prompt/tool/skill/policy versions, its run graph, every tool call + evidence, its outcome — for regression testing, incident investigation, and counterfactual policy/model evaluation; same tool, same REST twin, same `_execute_tool` core as the live-run actions |
| `graph_message` | `open` (from a `session_id`/`run_id` → `channel_id`), `send` (`channel_id`+`sender`+`payload`, `durable=true` persists as a replayable `AgentMessage` node), `receive` (with a `since` cursor), `history`, `close` | ephemeral (optionally durable) message channels for a run and its spawned agents |
| `graph_bus` | `register`/`heartbeat`/`leave`/`status`, `roster` (discover peers + presence), `send` (`sender`+`payload`+`to`\|`topic`), `receive` (+`since`), `subscribe`/`unsubscribe`, `ack`, `dispatch` (hand an objective to the fleet as a Loop); mesh/federation: `register_hub`/`list_hubs`/`federate`/`federate_in` | the federated, durable, cross-host agent-to-agent bus (state lives in the KG) — distinct from `graph_message`'s per-run channels |
| `graph_broker` | `declare_exchange` (+`exchange_type`), `declare_queue`, `bind` (`queue`+`exchange`[+`routing_key`]), `publish` (`exchange`+`routing_key`+`payload`), `consume` (+`max_messages`/`ack` via `params_json`), `stats`/`list_queues`/`list_exchanges` | the engine's AMQP-style message broker (exchanges + queues + streams), distinct from the agent-to-agent bus and from the modality-tier `engine_broker` |
| `graph_reach` | `reach_user` (`text`[+`user_id`] → the user's last-active channel, else the configured default), `send` (explicit `platform`+`channel_id`+`text`), `list_channels` (`platform`), `last_channel` ([`user_id`]), `status` | outbound messages to the human user; every send is governed by the ActionPolicy gate and mirrored into conversational memory |

### Cold-start delegation (KG-driven router)

When the KG has no registered `Server`/`CallableResource` nodes yet (cold start),
auto-hydrate before delegating rather than failing:

1. **Discover**: `graph_query(cypher="MATCH (s:Server)-[:PROVIDES]->(r:CallableResource)
   RETURN s.name AS server, count(r) AS tool_count ORDER BY tool_count DESC LIMIT 50")`
   — an empty result means the KG is cold.
2. **Hydrate** (only if cold): ingest every discoverable `mcp_config*.json` (scan known IDE
   paths — Antigravity, XDG agent-utilities, Windsurf, Claude Code, Codex, Devin — dedupe
   by `command`+`args`) and the skill directories via `graph_ingest(action="agent_toolkit",
   target_path='["<path1>", ...]')`. The epistemic-graph backend MERGE-deduplicates
   `Server` ids natively, so re-hydration is idempotent.
3. **Delegate** via `graph_orchestrate` in one of three modes: **`execute_agent`**
   (single-server, prompt-based — the default), **`execute_workflow`** (a pre-compiled
   multi-step `GraphPlan` already stored in the KG), or **`compile_workflow`** then
   `execute_workflow` (compile an ad-hoc natural-language multi-step task into a reusable
   workflow first, then run it).

An **in-band alternative** exists when the caller is itself an MCP client connected
through the mcp-multiplexer in `dynamic` mode: `find_tools`/`load_tools` mounts the
discovered tools directly in the client instead of running them remotely. Use
`graph_orchestrate` for autonomous/headless delegation; use the multiplexer meta-tools
when an interactive client should gain the tools itself. Both share the same KG
`Server -[:PROVIDES]-> CallableResource` substrate.

## Workflow

### 1. Define the goal

Specify the desired state, scope, constraints, evidence required, deadline, and
what must not change. Convert vague completion language into observable checks.

### 2. Build the work graph

- Create one work item per independently verifiable outcome.
- Add dependencies only where data or authorization truly requires ordering.
- Assign domain skills rather than individual one-tool wrappers.
- Mark mutating or externally visible steps for policy review.
- Select an economical model class for deterministic extraction, classification,
  or formatting; reserve stronger reasoning for planning, critique, and synthesis.

### 3. Execute

For a plan or review-only request, return the bounded work graph, dependency
barriers, approval points, and verification criteria, then stop. Do not dispatch,
schedule, persist, load tools, or otherwise execute the plan.

| Need | Primary operation |
|---|---|
| Run one named agent | `graph_orchestrate` |
| Run a swarm or runtime org | `graph_agents` |
| Compile, run, or inspect a workflow | `graph_workflows` |
| Dispatch or inspect a durable job | `graph_jobs` |
| Manage goal state | `graph_goals`, `spec_ticket` |
| Advance a controlled loop | `graph_loops` |
| Create recurring work | `graph_schedules` |
| Isolate execution | `graph_sandbox` |
| Fork, revert, or replay a run | `graph_fork`, `graph_runvcs` |
| Exchange task messages | `graph_message`, `graph_bus`, `graph_broker` |
| Check participants or capability reach | `graph_reach` |

Launch independent work items together, then wait at explicit dependency
barriers. Carry the original goal and acceptance criteria into every delegation.

### 4. Verify and synthesize

- Require each work item to return evidence and a clear status.
- Re-run critical checks from the coordinator rather than trusting summaries.
- Resolve contradictory outputs explicitly.
- Report partial completion and blockers; never convert a timeout into success.

### 5. Close the loop

Persist the final outcome, provenance, approvals, and follow-up items. Remove
temporary schedules or loaded tools that were created only for the run.

## Guardrails

- Keep authorization scope unchanged across delegation.
- Do not send secrets or unrelated context to child agents.
- Bound fan-out, retries, depth, and wall time.
- Require human direction before irreversible or externally visible expansion of
  the requested scope.
