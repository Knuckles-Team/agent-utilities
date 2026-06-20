# Ontology-native classification — full handoff (Phase A → checkpoint → Phase B)

> **Status:** APPROVED 2026-06-20. Phase A1 done (injection point located); A2–A5 + B not
> started. **Decisions:** (1) Phase-A capability source of truth = the **served multiplexer /
> registry catalog** (not AST); (2) build **Phase A fully → checkpoint for review → then B**.
> **Goal:** classify a chat turn (lean-chat vs full multi-agent graph) from the **live ontology**
> — **zero hardcoded domain keywords** — following *maximize "free" reasoning, escalate only for
> complexity*.

---

## 0. Session context — what we did and why (the road to this plan)

This work came out of debugging "why is the Telegram bot slow / not using tools / not
formatted". Four things shipped (all **local `main`, NOT pushed**, deployed to the live
messaging daemon via `ssh R820 "docker service update --force agent-utilities-messaging…"`),
and the investigation then converged on the real root cause below.

**Shipped + deployed this session:**

1. **Dispatcher broadcast-fork fix** — `ORCH-1.68`, au `71b87e35`, merged `8c86494b`. The full
   multi-agent graph executed *no* plan: an earlier `router → end_node` edge I'd added made
   pydantic-graph **broadcast-fork** the router to `{__end__, dispatcher}`, so every full-graph
   turn died at `__end__`. Found by instrumenting `_GraphIterator._run_task`'s node sequence.
   Fix: answer direct-completion **outside** the graph (`agent_runner._run_direct_completion`,
   short-circuited in `_execute_graph`), revert the fork edge, delete the dead in-graph router
   block. Validated: full graph now runs router→dispatcher→…→expert_executor. See
   `full-graph-dispatcher-routing-bug` memory.
2. **Messaging tool-turn latency** — `ORCH-1.72`, merged `d2509544`. (a) `_PASSTHROUGH_AGENTS`
   = `{messaging-assistant}` skips a ~21 s mis-firing `_resolve_agent_from_kg`; (b)
   `ExecutionProfile.reply_budget_s` (direct ~25 s / full ~190 s) → dynamic messaging timeout;
   (c) `ExecutionProfile.is_interactive` (≤50 s) → **ack-now / deliver-later**: a heavy turn
   gets an immediate "On it…" ack threaded to the user message, runs in the background, and
   delivers the result as a follow-up.
3. **Telegram Markdown rendering** — `ECO-4.0`, committed `e91812e2` (this worktree,
   **not merged/deployed yet**). The agent replies in Markdown but the backend sent it with
   `parse_mode=HTML`, so `**bold**` / `## h` / `` `code` `` arrived as raw markers. Added
   `messaging/render.py` (Markdown → Telegram-HTML subset) applied in `telegram.send_message`,
   with a plain-text retry if Telegram rejects the HTML. 6 tests.
4. **(superseded)** A read-verb keyword expansion in `fast_path` — reverted; the user correctly
   rejected growing the hardcoded list (see §2).

**The live test that exposed the real problem (the 2 cases in §6):** with the dispatcher fixed,
the user sent "list portainer stacks" and "github open issues". Portainer **direct-completed in
5 s with no tool** ("can't access portainer"); GitHub correctly ran the full graph but **exceeded
the 190 s budget** and timed out. Diagnosis chain:
- *Why portainer answered tool-less:* it was classified trivial — the `_ESCALATION_KEYWORDS`
  list had write-verbs (deploy/fetch…) but **no read verbs** (list/get/show) and no tool
  reference, so a retrieval query scored 0 → lean → direct-completion.
- The expedient fix (add read verbs) is **bad practice** — an unbounded hardcoded word list,
  antithetical to the ontology-through-Rust design. The right gate asks the ontology.
- *Why the ontology gate is hard right now:* the **KG lacks the fleet capability vocabulary**.
  `portainer`/`github` exist only as `Code` nodes, not `Tool`/`Skill` capability nodes. So the
  hardcoded keywords were *masking* a hole in the data. → **Phase A (data) before Phase B (gate).**

---

## 1. Problem (verified against the live KG, 2026-06-20)

- KG = 73.7k nodes. Capability types exist — `NativeTool` (89), `Skill` (94),
  `BusinessCapability` (806), `Route` (179), `Resource` (42) — **but only from graph-os's own
  served catalog + the skill library** (`read_file`, `run_command`, `agent-package-builder`…).
- The **~62 fleet MCP servers** (portainer-agent, github-mcp, …) were AST-ingested as generic
  `Code` symbols; their `@mcp.tool()` functions were never elevated. `MATCH (n) WHERE n.name
  CONTAINS 'portainer'|'github'` → **0 capability nodes** (portainer = only
  `code:portainer_agent/.../mcp_docker.py::portainer_docker`).
- `physical_distiller.distill_mcp_tool` (physical_distiller.py:136) **has no live caller**; the
  `@register_source("mcp")` connector (mcp_package.py:153) didn't cover the fleet.

**Chain:** ingestion mis-classifies fleet tools → KG lacks capability vocabulary → no ontology
gate can match → can't escalate. (Same missing vocabulary = the dispatcher bug's secondary "no
fleet specialist registered" — **Phase A fixes both**.)

## 2. Target architecture — the escalation ladder

| Tier | Mechanism | Cost | Decides |
|---|---|---|---|
| 0 structural | slash / length / multi-clause / greeting (`fast_path`) | ~0 | obvious trivial vs obvious action |
| 1 **free ontology lexical** | `engine.match_ontology_terms` (epistemic-graph, no embedding) | ~ms | names a real capability? → full |
| 2 semantic | `search_hybrid` (vector) | ~4.5s | paraphrase/synonym the lexical missed |
| 3 reasoning | LLM HTN planner | LLM | genuinely complex multi-step |

Backend-agnostic by design: the lexical tier lives in **epistemic-graph** (the one universal L1
every deployment runs), **NOT** ParadeDB BM25. `postgresql_backend.lexical_search` (BM25,
`paradedb.rank_bm25`) exists and works, but it is **Postgres-only** — neo4j/falkordb/ladybug have
no equivalent — so binding the gate to it would silently fail on other backends.

---

## 3. Phase A — Ingestion classification fix (the data; do first)

- **A1 (DONE). Injection point located.** The bootstrap in `agent_utilities/mcp/kg_server.py`
  (~lines 2095–2160) already ingests: MCP **servers** from `mcp_config.json` → `MCPServer` nodes
  (~2100); graph-os's OWN native tools (`agent_utilities.tools` pkg, functions with
  `__agentic_version__`) → `NativeTool` nodes (~2115–2143 — the `add_node(f"native_tool_{name}",
  "NativeTool", {name, description, version, module, disabled})` pattern to mirror); skills dir →
  `Skill` (~2147). **GAP:** it never enumerates each fleet server's TOOLS.
- **A2. Add a served-catalog step** (Native-by-default, no flag) in that bootstrap (or a
  dedicated connector): enumerate the multiplexer catalog (server → tools, via `list_catalog` /
  MCP introspection of each server in `mcp_config`) and `engine.add_node` each tool as a `Tool`
  capability node `{name, description, server}`, linked to its `MCPServer` (`SERVES` / `servedBy`)
  and cross-linked to the existing `Code` node for provenance. Reuse
  `physical_distiller.distill_mcp_tool` / `skill_synthesizer.classify` where they fit.
  - *Catalog access:* prefer the multiplexer's aggregated catalog (it already lists every fleet
    tool with name+description+server). Fallback: iterate `mcp_config` servers and MCP-introspect
    each (`list_tools`). Handle unreachable servers gracefully (skip, log) — coverage = currently
    registered servers (the accepted trade-off of the "served catalog" decision).
- **A3. Standardize the capability node schema** (`name`, `description`, `server`, `kind`,
  `synonyms`/aliases) + the ontology mapping (`owl_bridge`) so the **gate AND the dispatcher's
  specialist routing** query it uniformly. Surface over `graph_*` MCP + REST (two-surface rule).
- **A4. Re-ingest the fleet.** Run the live E2E protocol (`ingestion-validation-protocol` memory):
  restart graph-os, `go__source_sync` (or the bootstrap), re-run to prove `skipped_unchanged`.
  **Needs a healthy engine/embeddings (GB10/vLLM).**
- **A5. Verify + profile.** `graph_query` for github/portainer returns `Tool`/`Skill` capability
  nodes; capability-node count covers the fleet. **Then profile** the two slow turns (§6) —
  routing is now correct, so the profile is meaningful (where do the 3 min go: planning loops vs
  the actual tool call?).

**Done when:** portainer/github are capability nodes, and case 1 (§6) routes to the full graph
AND the dispatcher reaches the portainer tool.

---

## 4. Phase B — Engine-native lexical matcher + cascade rewire (after A checkpoint)

- **B1. epistemic-graph (Rust) — `match_ontology_terms` (5 mapped edits):**
  - `crates/eg-core/src/graph.rs` — `GraphCore::match_ontology_terms(query)`: aho-corasick over
    capability-node labels+synonyms (filter to `Tool`/`NativeTool`/`Skill`/`Server`/
    `BusinessCapability`/`Resource`); add `aho-corasick` to `crates/eg-core/Cargo.toml`. Start
    **per-request build** (mirror `get_nodes_by_label`, graph.rs:284); cache on `GraphCore` only
    if profiling demands.
  - `crates/eg-types/src/protocol.rs` — `MatchOntologyTerms { query: String }` variant (Semantic
    Compute section, ~:394). Not feature-gated.
  - `src/server/handlers/graph_ops.rs` — handler arm after `SemanticSearch` (~:173) →
    `ResultPayload::raw(&results)` of `{term, node_type, label, score}`.
  - `src/server/dispatch.rs` — **no edit** (read routes via the catch-all `_ =>` at :401);
    confirm **not** added to `requires_write` (access.rs).
  - `epistemic_graph/client.py` `GraphOperationsClient` (~:247) + agent-utilities
    `knowledge_graph/core/graph_compute.py` (~:557) — `match_ontology_terms(query)` wrappers.
  - Build `cargo build --release --features server`; tests `cargo test --features server --lib` +
    pytest round-trip + `scripts/check_no_pyo3.sh`. The running daemon needs rebuild+restart.
- **B2. Cascade** (`orchestration/execution_profile.py`, `plan_execution_shape`): insert the free
  lexical stage — structural `strength ≥2` → full; else `engine.match_ontology_terms` →
  **capability hit → full**; weak/none → `search_hybrid` (existing stage 2); still nothing → lean.
  LLM HTN stays the stage-3 escape.
- **B3. Delete `_ESCALATION_KEYWORDS`** from `fast_path`. `orchestration_signal_strength` /
  `needs_full_orchestration` become **structural-only** (slash / length / multi-clause /
  greeting). Domain vocabulary now lives in the KG.
- **B4. Tests + live validate** against §6: portainer/github → full via the engine lexical tier;
  trivial chat → lean; the free tier <50 ms.

---

## 5. Sequencing, risks, IDs, where things live

- **Order:** A (incl. re-ingest + profiling) → checkpoint → B. A unblocks B.
- **Risks:** re-ingestion needs healthy engine/embeddings (GB10/vLLM); the A3 capability schema
  must match what the dispatcher's specialist routing expects (coordinate with that); aho-corasick
  per-request vs cached (start simple).
- **Concept IDs to reserve:** `KG-2.x` (ingestion capability elevation), `EG-x`
  (`match_ontology_terms`), `ORCH-1.x` (ontology-driven cascade, extends `ORCH-1.69`).
- **No-legacy:** B3 deletes the keyword list outright (no dual path); the gate is the engine.
- **Worktree:** `feat-classify` (branch `feat/classify-fix`) holds this doc + the Telegram fix
  (`render.py`, `telegram.py`, `tests/unit/messaging/test_render.py` incl. the two §6 case
  renders, commit `e91812e2`). `fast_path.py` keyword change was reverted. The Telegram fix is
  **committed but not merged/deployed** — ship independently if desired.
- **Memory:** `ontology-native-classification-build` (recall), `full-graph-dispatcher-routing-bug`,
  `optimization-campaign-checkpoint`, `ingestion-validation-protocol`,
  `deployment-model-docker-portainer` (redeploy = `ssh R820 docker service update --force`).

---

## 6. Validation test cases (the 2 Telegram turns)

> The **formatting** half of each case is captured as a runnable regression in
> `tests/unit/messaging/test_render.py` (`test_case1_portainer_reply_renders` /
> `test_case2_github_reply_renders`) — realistic tool-reply Markdown → asserted Telegram HTML.
> The **classification + routing + tool** half is the manual runbook below (needs the live KG /
> daemon; becomes unit-testable once A+B land).

These two real turns exercise the whole stack — classification, routing, tool execution, the
dynamic budget + ack/deliver, and Telegram formatting. Use them to validate each phase.

### Case 1 — Portainer (retrieval, no explicit tool word)
- **Send (Telegram):** `Can you list the stacks I have running on portainer?`
- **Now (broken):** classified trivial → `direct_complete` → 5 s, **no tool**, "I can't access
  portainer." (The classification gap.)
- **Expected after A+B:** routes to the **full graph** (engine lexical matches the `portainer`
  capability node) → ack "On it… ⏳" → dispatcher reaches the portainer tool → follow-up lists the
  stacks, **rendered** (bold/bullets), within the dynamic budget.
- **Validate:** daemon log shows NO `direct_complete` for this turn; shows
  `[ORCH-1.72] … acknowledging now`; shows the portainer tool call; the reply renders formatted.

### Case 2 — GitHub (explicit "github mcp", multi-clause, long)
- **Send (Telegram):** `Can you use the github mcp to fetch me the github open issues for my
  Knuckles-Team organization repositories? Is there a way to get issues wholistically open against
  all projects in the organization?`
- **Now (partial):** already routes to the full graph, but **exceeds the 190 s budget** → graceful
  "responding slowly" timeout (the tool turn is too slow / rediscovers the tool).
- **Expected after A+B:** routes to full graph; with the github tool now a registered capability
  node the expert step **binds it directly** (no rediscovery) → returns the issues inside budget,
  rendered. (If still slow, A5 profiling says where the time goes.)
- **Validate:** the follow-up contains real issues (not the timeout message); profile shows the
  time in the tool call, not planning loops.

**Cross-cutting (already fixed, re-verify):** both turns must (a) get an immediate ack then a
follow-up (ack-now/deliver-later, ORCH-1.72), and (b) render Markdown as formatted Telegram HTML
(ECO-4.0) — the latter only once the Telegram fix is merged+deployed.

**How to run:** send each via Telegram to the bot and watch
`docker logs -f $(docker ps -q -f name=agent-utilities-messaging)`. For the routing half without
Telegram, the harness `scratch/disp_e2e.py` / `scratch/disp_seq.py` drives `execute_agent` for the
messaging-assistant and prints the node sequence (use `execution_profile="task"` to force the full
graph, `"chat"` to test classification).
