# Unified Agent Entrypoint — one seam into the graph agents (verified)

> **Status:** **verified as-built (2026-06-28).** This is the *verification + routing map*
> companion to the north-star [`entrypoint-unification.md`](entrypoint-unification.md): it
> traces the **real call paths** of every surface that kicks off a pydantic-ai graph agent,
> confirms they funnel into one orchestration system, and documents the one divergence found
> and closed (the streaming REST surface, CONCEPT:AU-ORCH.session.session-continuity-entrypoint).

## Verdict

**Already unified at the graph + engine level.** There is exactly **one** pydantic-ai graph
(`graph/builder.py::create_graph_agent`, materialized once into the server's `app.state.graph_bundle`)
and **one** executor (`orchestration/engine.py::AgentOrchestrationEngine`). Every surface runs
*that* graph — there is no parallel orchestrator.

`orchestration/agent_runner.py::run_agent` is the **instrumented seam**: the single function that
wraps the graph with (a) the escalating execution-shape planner, (b) KG agent resolution, (c)
**recalled-memory priming** (mementos), (d) code-context priming, and (e) **`:RunTrace` + `:ToolCall`
provenance** (KG-2.296) + Session anchoring. The MCP, messaging, and workflow surfaces all reach the
graph **through** `run_agent`, so they inherit memory + provenance for free.

The **one gap** was the streaming REST gateway (`/ag-ui`, `/stream` — the path the `agent-webui` /
`agent-terminal-ui` frontends hit *directly*). It streams the **same** graph via
`AgentOrchestrationEngine.iter_graph` (token-by-token, which cannot return through `run_agent`'s
string contract), and historically skipped the seam's memory + provenance wrapper — leaving those
sessions memory-siloed. **CONCEPT:AU-ORCH.session.session-continuity-entrypoint** wires that surface into the *same* KG-backed
continuity model (recall + persist), without a parallel orchestrator. See
[Shared memory](#shared-memory-kg-backed-cross-surface).

```
  graph-os MCP            messaging (18 backends)        WorkflowRunner
  graph_orchestrate       Telegram/Mattermost/Discord/   (per step)
  execute_agent           Slack/Signal/Teams/…
       │                       │  InboundRouter._dispatch      │
       │                       │  → planner default handler    │
       ▼                       ▼                               ▼
  Orchestrator.execute_agent / execute_workflow ───────────────┤
       │                                                       │
       ▼                                                       ▼
  ┌───────────────────────────  run_agent  ──────────────────────────┐
  │  shape planner · KG resolve · MEMORY PRIME · code-context ·       │
  │  RunTrace + :ToolCall provenance · Session anchor                 │
  └───────────────────────────────┬──────────────────────────────────┘
                                   │ create_graph_agent
   agent-webui / agent-terminal-ui │            ┌──────────────────────┐
   POST /ag-ui · /stream           │            │ session_continuity   │
        │  iter_graph (streaming)  │            │ prime + persist       │
        └───────► AgentOrchestrationEngine ◄────┤ (ORCH-1.104)         │
                  (THE one graph + executor)    └──────────────────────┘
                                   │
                                   ▼
                    pydantic-ai graph agents  ──►  Knowledge Graph
```

## Per-surface routing table (verified)

| Surface | Entry | Reaches the seam via | Through `run_agent`? | Memory + provenance |
|---|---|---|---|---|
| **graph-os MCP** | `graph_orchestrate(action='execute_agent')` | `mcp/tools/analysis_tools.py:1886` → `Orchestrator.execute_agent` (`orchestration/manager.py:103`) → `run_agent` | ✅ direct | ✅ full (prime + RunTrace + ToolCall) |
| **graph-os MCP (workflow)** | `graph_orchestrate(action='execute_workflow')` | `analysis_tools.py:2034` → `Orchestrator.execute_workflow` → `workflows/runner.py::WorkflowRunner` → `run_agent` per step | ✅ per step | ✅ full per step |
| **messaging** (Telegram live; Mattermost, Discord, Slack, Signal, Teams, Matrix, IRC, … 18 backends) | `Backend.listen()` → `InboundRouter._dispatch` (`messaging/router.py:273`) → planner default handler (`daemon.py:47`) | `messaging/router.py:880` → `Orchestrator.execute_agent` → `run_agent` | ✅ direct | ✅ full + `_persist_and_enrich` writes the per-channel memento (`router.py:447/596`) |
| **agent-webui / agent-terminal-ui** (separate repos) | `POST /ag-ui`, `POST /stream` on the gateway | `server/routers/agent_ui.py` → `execute_graph_iter` → `AgentOrchestrationEngine.iter_graph` (the **same** graph) | ⚠️ **No** (streaming) — but now joins the same continuity seam via **`session_continuity`** (ORCH-1.104) | ✅ after ORCH-1.104: `prime_session_context` (recall) + `persist_session_turn` (RunTrace + memento) |
| **dedicated `agent_server.py`** | `server/__init__.py::create_agent_server` / `_run_agent_server` | This **is** the gateway that hosts `/ag-ui` + the MCP — it does **not** duplicate orchestration; it serves the routers above | — | inherits the surfaces' wiring |
| **geniusbot (desktop)** | separate repo | reaches the platform through the graph-os MCP / the gateway REST — no own agent-kickoff | via MCP/REST | inherits the surface it calls |

**Surfaces that bypass `run_agent`'s string contract by necessity** (streaming) still share the
**one** graph + executor; ORCH-1.104 gives them the same memory + provenance side-effects.

## Shared memory (KG-backed, cross-surface)

Conversation memory is **mementos** in the Knowledge Graph, written/read by the core primitives
`compress_to_memento(engine, turn, source=…)` and `get_recent_mementos(engine, source=…)` plus the
hot-path `session_memento_cache`. The **`source` key is the join**: any two surfaces that pass the
**same `source`** share recall.

| Surface | Recall on turn start | Persist on turn end | `source` key |
|---|---|---|---|
| MCP / workflow | `run_agent::_prime_recent_mementos` | `run_agent` post-run (`compress_to_memento`) | `memento_source` or agent name |
| messaging | `run_agent` priming | `messaging/router.py::_persist_and_enrich` (`compress_to_memento` + cache refresh) | the channel session id |
| **agent-webui / agent-terminal-ui** | **`session_continuity.prime_session_context`** (injected as `invoker_context`) | **`session_continuity.persist_session_turn`** (RunTrace + `compress_to_memento` + cache refresh) | the request `session_id` (== `run_id`) |

**Cross-surface recall** therefore works whenever the surfaces are keyed to the **same** stable,
user-scoped `session_id`. A webui session that sends `session_id="user:alice"` will recall the
mementos written by an MCP call or a messaging thread keyed to `"user:alice"`, and vice-versa. (The
client must send a stable id; an absent id yields a fresh per-request `run_id` and an intentional
anonymous one-shot with no cross-turn recall.)

### ORCH-1.104 — what was wired

`orchestration/session_continuity.py` factors the seam's two memory/provenance concerns into a
reusable pair (reusing the *existing* primitives — `get_recent_mementos`,
`agent_runner._record_execution_trace`, `compress_to_memento`, `refresh_session_memento_cache`; no
new orchestrator):

- `prime_session_context(engine, session_id)` → recalls the session's mementos as an
  `invoker_context` block; the `/ag-ui` fast-path injects it into the per-turn config so the graph
  sees prior memory (`server/routers/agent_ui.py`).
- `persist_session_turn(engine, session_id, query, reply)` → off the reply path, records a
  `:RunTrace` (+ `Session`→`HAS_RUN`) and compresses the turn into a per-session memento + refreshes
  the cache, so the next turn — on **any** surface keyed to the same session — recalls it.

This composes with the **delegation-first operating model** (the local LLM + graph-os do the work)
and the **`:ToolCall` provenance** spine (KG-2.296): a webui/terminal-ui turn is now as queryable
over graph-os as a delegated MCP run.

## Remaining gaps

- **Streaming surfaces still don't emit per-tool `:ToolCall` nodes.** `run_agent` persists each tool
  call (`_persist_tool_calls`, KG-2.296); the streaming path records the run-level `:RunTrace` +
  memento (ORCH-1.104) but not yet per-call nodes (the `iter_graph` event stream would need to
  surface tool-call deltas). Low priority — run-level provenance + memory parity is in place.
- **Discord is implemented as a backend** (`messaging/backends/discord.py`) and flows through the
  same `InboundRouter`; whether it is *live* is a deployment/config matter (token + `listen`), not a
  code gap. Telegram is the live, verified channel.
- **`iter_graph`'s `GraphState` omits `session_id`** (set in `execute_graph` but not `iter_graph`);
  ORCH-1.104 keys continuity at the wrapper level so this is cosmetic, but worth aligning later.

## Files

- Seam: `agent_utilities/orchestration/agent_runner.py::run_agent`
- Executor + the one graph: `agent_utilities/orchestration/engine.py`, `agent_utilities/graph/builder.py`
- MCP path: `agent_utilities/mcp/tools/analysis_tools.py`, `agent_utilities/orchestration/manager.py`
- Messaging path: `agent_utilities/messaging/router.py`, `agent_utilities/messaging/daemon.py`
- Streaming surface: `agent_utilities/server/routers/agent_ui.py`, `agent_utilities/graph/protocol_agnostic_execution.py`
- **ORCH-1.104 wiring:** `agent_utilities/orchestration/session_continuity.py`
- Test: `tests/test_orch_1_104_unified_entrypoint_continuity.py`
