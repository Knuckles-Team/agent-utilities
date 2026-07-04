# Optimization Campaign — Session Checkpoint (2026-06-19)

> Resume point for the "universal orchestrator + latency" campaign. All work is on **local
> `main`, NOT pushed, NOT redeployed** unless noted. Constraints still in force: never push;
> OpenBao is secret-source-of-truth (no plaintext secret files); work in git worktrees; never
> merge from inside a worktree; a **separate session owns the OIDC realm switch** (Keycloak
> master → homelab) — don't touch auth/keycloak.

## The goal (north star)
`agent-utilities` is **one pydantic-ai KG orchestrator**; messaging/webui/terminal/geniusbot/
`agent_server.py` are **thin entrypoints** that inherit core capabilities. Enforced by the
*Universal capability* + *Sprawl boundaries* rules in `AGENTS.md`. Plans:
`docs/architecture/entrypoint-unification.md`, `non-blocking-execution.md`, `reactions.md`.

## ✅ DONE — merged to local `main` (not pushed)
| Commit | What |
|---|---|
| `d3d1166e` | **Messaging → universal graph** (ECO-4.78). Chat now routes `_graph_agent_reply → Orchestrator.execute_agent → run_agent`, session = `messaging:{platform}:{channel_id}`, continuity via core **mementos** (`memento_source` threaded through `run_agent`). Deleted the bespoke path (`_model_routed_reply`, `_recall_history`, `_get_messaging_agent`, `_run_until_text`, `_auto_approvable`, `recall_recent_messages`, the ECO-4.76 band-aid) — **−495 lines**. Plain-chat fallback kept. |
| `dd348db7` | **Reactions system-wide, core slice** (ECO-4.79/4.80/4.81). `orchestration/reactions.py::AgentReaction` + `EmoteRegistry` (governed via `ActionPolicy` `reaction` kind) + `decide_reaction` moved into core; `MessagingService.render_reaction` paints it on Telegram. |
| `de3ba747` | **Profiling + 2 bug fixes.** FIXED: the prompt-injection gate never fired on `execute_agent`/`compile_workflow` (guarded on a non-existent `analyze` method → dead security code; now `scan_text(...).is_malicious`). FIXED: stray `print()` on spawn path → `logger.debug`. Wrote `non-blocking-execution.md` design doc. |
| `78ec47cd` | **P0/P1 latency** (ORCH-1.62 chat exec profile, AU-ORCH.routing.original-rule-was-far widened fast-path, ORCH-1.64 graph cache, ORCH-1.65 de-block reply path + router N+1 collapse, AU-KG.memory.refresh-per-session-memento memento pre-prime cache). 114 unit tests green. **NOTE: merged + unit-green but NOT effective live — see below.** |
| `d0d076db` | Earlier continuity+delegation (ECO-4.76/4.77) — **superseded** by `d3d1166e`. |
| `af071cc1`, `64ef84fa` | Docs + edicts: `entrypoint-unification.md`, *Universal capability* + *Sprawl boundaries* rules in AGENTS.md (heavy AI→data-science-mcp, finance→emerald-exchange, heavy/KG compute→epistemic-graph; ontology extends canonical .ttl; no new daemons — extend or use AU-KG.ingest.mcp-tool-connector connector presets). |

Also done earlier this session: full containerization (5/5, zero systemd), torch-free core
(re-homed to data-science-mcp), **slim serving image 2.06GB→1.11GB**, messaging reply
timeout+fallback + psycopg.

## 🔬 THE KEY LIVE FINDING (decisive, sharply scopes the next step)
Tested live after restarting messaging onto the merged code:
- **vLLM is FAST and healthy**: a direct `vllm.arpa/v1/chat/completions` = **0.87s** (200), serving `qwen/qwen3.5-9b`. GB10 (`10.0.0.18`, SSH works w/ host RSA key) **up ~4 days, 0% GPU, no power fault active** — the "vLLM degraded / GB10 power fault" assumption was STALE.
- **The universal chat reply STILL takes 45s live** (hits `MESSAGING_REPLY_TIMEOUT`, returns the graceful no-double-LLM message). `_graph_agent_reply(eng,'what is 2 plus 2?',session=...)` → **45.4s**.
- ∴ **The 45s is 100% orchestration overhead/hang, NOT vLLM.** At 0.87s/call it cannot be "a few LLM rounds" — it's a hang or serial stall in the graph. **The P0/P1 fixes did NOT take effect live**: a trivial question should have hit the single-round fast-path (~1-2s); the chat-profile 12s node caps didn't fail it fast either. The **no-double-LLM fix DID work** live.

## ⏳ PENDING — resume here (in priority order)
1. **[P0] LIVE latency debug — THE next action.** Restart messaging, then instrument the live `_graph_agent_reply → execute_agent → run_agent` path (faulthandler / asyncio task-stack dump ~15s into a turn, or stage timing logs) to find WHERE the 45s goes. Specifically verify: (a) is `is_trivial_query`/`needs_full_orchestration` actually classifying a simple question as fast-path live? (b) is the `chat` ExecutionProfile (ORCH-1.62) reaching the graph node timeouts (router/verifier ~12s) or still 300s? (c) is a **sync KG/discovery call hanging** (the engine, `_resolve_agent_from_kg`, the router discovery bundle) that the node timeout doesn't cover? Close the unit↔live gap. This is sharply scoped now: vLLM is fine; the cost is entirely in the graph.
2. **[feature] Frontend reaction renderers** — `agent-webui` (chips), `agent-terminal-ui` (glyph), `geniusbot` (desktop), `agent_server.py` (response-envelope `reaction` field). NEVER STARTED (the agent was killed by API rate-limiting at spawn, 0 work). Contract is ready in `docs/architecture/reactions.md` — thin renderers off the `AgentReaction` wire-shape, each in its own repo.
3. **[P2] Rust-offload** — engine `discover(query,k) -> {matched_agents, hybrid_hits, policies, processes}` (one UDS round-trip) to collapse the router N+1; demote `retrieval/capability_index.py` (numpy cosine) + `generative_recommender.py` (np.argsort) behind `semantic_search`. Rate-limited MID-WORK (~80 tool-uses), **NOT merged**. **Leftover `/home/apps/worktrees/epistemic-graph` worktree to inspect/clean.** Contract = `TODO(CONCEPT:AU-ORCH.execution.chat-profile-timeouts P2)` in `graph/routing/_router_impl.py::router_step`. Rust repo: `agent-packages/epistemic-graph` (NOT the broken empty worktree).

## How to resume / validate live
- **Reload merged code:** `ssh R820 "docker service update --force agent-utilities-messaging_agent-utilities-messaging --detach"` (and `graph-os_graph-os` / `graph-os-host` likewise). Containers mount `/au` = this canonical checkout, so a restart loads current `main`. Messaging is on `agent-utilities:serving` (slim), placement `node.labels.name == RW710`, manager = R820.
- **Timed live test** (in the messaging container): `_graph_agent_reply(IntelligenceGraphEngine.get_active(), 'what is 2 plus 2?', session='messaging:telegram:lat')` wrapped in `asyncio.wait_for`.
- **Full E2E:** two real Telegram turns → expect (after the latency fix) seconds-not-45s, turn-2 continuity via mementos, and a reaction rendering.

## Infra snapshot
5/5 containerized, zero systemd: engine (`ubuntu:26.04`, UDS `/run/epistemic-graph`), graph-os + graph-os-host + messaging (`agent-utilities:serving` 1.11GB torch-free), multiplexer (`python:3.11-slim`). Engine data intact (~72.7K nodes). `graph-os.arpa/health 200`, `mcp-multiplexer.arpa 401` (JWT). Rate-limiting (server-side) was actively killing spawned agents at end of session — relaunch defeated agents when it clears.
