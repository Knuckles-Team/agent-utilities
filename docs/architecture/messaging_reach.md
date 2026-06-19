# Messaging reach — Claude & agents message the user (ECO-4.48–4.54)

The **reach** capability lets Claude (over MCP) and the pydantic-ai graph agents
proactively message the operator on whatever channel they last used — Telegram, Slack,
Discord, and 14 other backends — and route the user's replies back into the graph. It
finishes the wiring of the pre-existing `CONCEPT:ECO-4.0` messaging framework
(`agent_utilities/messaging/`), which shipped 17 backends, a registry, an inbound router,
and KG auto-ingest but had **no live caller**.

## What was added

| Concept | What | Where |
|---|---|---|
| ECO-4.48 | `MessagingService` — one core: connected backends, governed sends, routing | `messaging/service.py` |
| ECO-4.49 | Last-active channel state (durable `UserChannelPreference` node) | `messaging/service.py`, `messaging/router.py` |
| ECO-4.50 | `graph_reach` MCP tool + `/graph/reach` REST twin | `mcp/tools/reach_tools.py` |
| ECO-4.51 | Inbound router in the host daemon + real graph-agent reply (replaced the stub) | `gateway/daemon.py`, `messaging/router.py` |
| ECO-4.52 | Elicitation bridge — a blocked loop/agent question reaches the user and resumes on reply | `observability/approval_manager.py`, `messaging/service.py` |
| ECO-4.53 | Universal `reach_user` agent tool | `tools/agent_tools.py`, `tools/tool_registry.py` |
| ECO-4.54 | `MessagingChannel` ontology interface (owl:Class) | `knowledge_graph/ontology/interfaces.py` |

## How routing works (OpenClaw-style)

`reach_user(text)` delivers to the user's **last-active channel**: every inbound message
updates a durable `UserChannelPreference` node; `reach_user` reads the most recent one and
falls back to the configured default (`MESSAGING_DEFAULT_PLATFORM` /
`MESSAGING_DEFAULT_CHANNEL`) so a fresh system still works. Every send passes the
fail-closed **ActionPolicy** gate (`message.send`, default `auto_notify`) and is mirrored
into KG conversational memory (`kg_ingest`), so history is recallable cross-platform.

## Flow

```mermaid
flowchart TD
    subgraph Outbound
        Claude([Claude / MCP]) -->|go__graph_reach| Reach[graph_reach tool]
        Agent([pydantic-ai agent]) -->|reach_user tool| SVC
        Loop([goal-loop / elicitation]) -->|reach_user_and_wait| SVC
        Reach --> SVC[MessagingService]
        SVC -->|ActionPolicy gate| Gate{message.send}
        Gate -->|allow| Backend[(Telegram backend)]
        SVC -->|mirror| KG[(KG memory)]
        Backend --> User((User on Telegram))
    end
    subgraph Inbound
        User -->|reply| Backend
        Backend -->|listen| Router[InboundRouter]
        Router -->|record_inbound| Pref[(UserChannelPreference)]
        Router -->|deliver_reply?| SVC
        Router -->|else| Planner[graph agent reply]
        Planner --> Backend
    end
```

The daemon (`gateway/daemon.py`) auto-starts the `InboundRouter` whenever a backend token
is configured (opt-out, auto-detected). When the user's reply answers a question a loop
asked, `deliver_reply` resolves the waiting future and the message is **not** re-routed to
the planner; otherwise the planner drafts a real reply via the graph agent
(`MESSAGING_AGENT`).

## Responder routing — local LLM by default, Claude on request (ECO-4.55)

Inbound messages are answered by a **dedicated messaging agent** (ECO-4.56) — built with
`create_agent`, so it inherits the **same universal tools (incl. `reach_user` + KG search),
agent skills, and MCP server fleet** as the rest of agent-utilities, with its **own system
prompt** at `agent_utilities/prompts/messaging_assistant.json`. The agent is built once and
**cached per model inside the single gateway daemon** (MCP/skills wiring paid once, never
rebuilt per message, never a second daemon). The model is routed per message,
**defaulting to the local LLM**:

- **Local LLM (default):** the configured local model (`qwen` on `vllm.arpa` in the homelab).
- **Claude (addressed):** a message starting with the trigger (`MESSAGING_CLAUDE_TRIGGER`,
  default `/claude`) uses an Anthropic model — requires `ANTHROPIC_API_KEY`; without it it
  falls back to local and says so.
- **Full named agent (override):** if `MESSAGING_AGENT` is set, that named graph agent
  handles the message instead (Orchestrator path).

Every reply is tagged with who answered (`[local]` / `[claude]`).

## Instinctive reactions (ECO-4.60)

The agent reacts to your messages with an emoji where the platform supports it — 👍 to
acknowledge a request, ❤️ for praise/thanks, etc. A cheap, **model-agnostic** decision
(`_decide_reaction`, a tool-free completion) runs per inbound message, so reactions work
even on local models that can't call tools; set `MESSAGING_REACTIONS=0` to disable.
`MessagingService.react()` dispatches to the backend's `send_reaction` (Telegram
`setMessageReaction` is implemented; other backends expose `send_reaction` as the extension
point and degrade gracefully where unsupported — the capability matrix declares support).

## Voice & image input (ECO-4.67/4.68)

- **Voice (ECO-4.68):** a voice note / audio with no text is transcribed via the
  audio-transcriber Whisper backend (`transcribe_voice`, lazy-loaded, off the event loop)
  and the transcript flows through the normal path — so you can just talk. Opt-out
  `MESSAGING_VOICE=0`; model via `MESSAGING_VOICE_MODEL` (default `base`).
- **Image (ECO-4.67):** image attachments are downloaded and passed as inline
  `BinaryContent` to the **vision-capable** model (qwen confirmed), so you can upload a
  picture and ask about it. Images ride the same burst → one multimodal agent turn.

## Burst coalescing (ECO-4.63)

When you fire several messages in quick succession, the agent collapses them into **one
holistic reply with one LLM call** instead of answering each separately. A per-conversation
debounce (`BurstCoalescer`, `messaging/coalescer.py`) accumulates messages and flushes the
batch when you pause for `MESSAGING_BURST_WINDOW_S` (default 2.5s) or `MESSAGING_BURST_MAX_S`
(default 12s) elapses. Per-message side effects that must stay immediate — last-active
channel, KG history ingest, loop-reply delivery, `/commands` — run per message; only the
agent reply (and its single reaction) coalesce. `BurstCoalescer` is a shared core primitive
agent-terminal-ui reuses, so burst behavior is identical across surfaces.

## Conversation history / continuity (ECO-4.76)

Every reply is grounded in **bounded, fast conversation history** so multi-message tasks
have continuity. Before drafting a reply the burst path (`_reply_to_burst` → `_recall_history`)
recalls the last `MESSAGING_HISTORY_TURNS` (default 8) turns — both user and assistant — for
**this** `(platform, channel_id)` and formats them into a compact `Recent conversation:` block
passed as the reply context (`_model_routed_reply`).

This is a cheap **exact-match recency query** (`kg_ingest.recall_recent_messages`) over a flat
`channel_key` scalar stamped on each message at ingest — **not** the heavy semantic
`recall_memory` (HNSW + cross-encoder), which was removed from the reply path because its
CPU-bound rerank stalled replies. The fetch is wrapped in `asyncio.wait_for` bounded by
`MESSAGING_RECALL_TIMEOUT`; on timeout/empty it degrades to no history rather than blocking
the answer. Deeper semantic KG context is still pulled **on demand** by the agent's
auto-approved `kg_search`/`kg_recall` tools when a question actually needs it.

## Universal commands (ECO-4.57)

Commands are defined once in `agent_utilities/messaging/commands.py` (`COMMANDS`) — the
single source of truth shared by every platform and importable by agent-terminal-ui
(`command_specs()`). On connect the daemon calls `backend.register_commands(...)` on every
backend; each registers the menu where its platform supports a **runtime** command API
(Telegram `setMyCommands`) and no-ops where commands are set via app-manifest/admin
(Slack/Teams/Mattermost) or a separate interaction model (Discord). Regardless of menu
support, commands also work as **typed `/cmd` text on any backend** — the inbound handler
parses a leading `/cmd` and `handle_command` answers built-ins (`/help`, `/status`,
`/tools`); `/claude` and `/skill` fall through to the model/agent. Add a command once and
it appears everywhere.

## Multiple services at once

The router runs **every configured backend concurrently** — set tokens for any of
Telegram, Slack, Teams, Mattermost, Discord, … and `start_messaging_router` connects and
listens on all of them. Last-active routing stores `platform + channel` per user, so
`reach_user` follows the user to whichever service they last used; `graph_reach
action=send` targets a specific service explicitly.

## Configuration

| Setting | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` / `SLACK_BOT_TOKEN` / `MATTERMOST_TOKEN` / `MSTEAMS_APP_ID`… | Enable each backend (auto-detected; multiple may be set together) |
| `MESSAGING_DEFAULT_PLATFORM` | Default platform when no last-active channel (default `telegram`) |
| `MESSAGING_DEFAULT_CHANNEL` | Default channel id for `reach_user` fallback |
| `MESSAGING_AGENT` | Optional: full graph agent that handles inbound (overrides model routing) |
| `MESSAGING_CLAUDE_TRIGGER` | Prefix that routes a message to Claude (default `/claude`) |
| `MESSAGING_CLAUDE_MODEL` | Anthropic model for the Claude route (default `claude-sonnet-4-6`) |
| `MESSAGING_LOCAL_MODEL` | Override the local responder model id |
| `ANTHROPIC_API_KEY` | Required for the Claude route |
| `MESSAGING_ENABLE_SKILLS` | Pre-load the full skill library (default `0` = lean; fleet MCP tools still load on demand) |
| `MESSAGING_SKILL_TYPES` | Comma-list: pre-load only these skill types |
| `MESSAGING_TOOL_TAGS` | Comma-list: scope the universal toolset to these tags |
| `MESSAGING_MCP_URL` | graph-os MCP for the agent (e.g. `http://127.0.0.1:8100/sse`) — gives `graph_orchestrate`/`graph_search` directly (ECO-4.59/4.75) |
| `MESSAGING_MCP_CONFIG` | Path to an MCP config whose `mcp-multiplexer` server fronts the whole fleet (`find_tools`/`load_tools`) — attach this **and** `MESSAGING_MCP_URL` for the two-server setup |
| `MCP_CLIENT_AUTH` / `OIDC_CLIENT_ID` / `OIDC_CLIENT_SECRET` / `OIDC_AUDIENCE` / `OIDC_TOKEN_URL` | Fleet OIDC client-credentials — loaded into the daemon env so the spawned multiplexer + nested agents authenticate. **Source from OpenBao**, never a plaintext file (ECO-4.75) |

### Fleet delegation via graph-os + multiplexer (ECO-4.59/4.75)

Rather than carrying every connector, the lean messaging agent gets the **same two MCP
servers Claude uses** and delegates through them:

1. **graph-os** (`MESSAGING_MCP_URL`, the served `kg_server --transport sse`) →
   `graph_orchestrate(action=execute_agent)` spawns a specialist (github-agent, …), runs it,
   relays the result; plus `graph_search` over the KG.
2. **mcp-multiplexer** (`MESSAGING_MCP_CONFIG`, pointed at the same fleet `mcp_config.json`) →
   dynamic `find_tools`/`load_tools` over the *whole* fleet on demand.

So the agent keeps a tiny context yet can reach any connector. Three things make it work
(all default-on once the two servers are set):

- **MCP context at run time** — the reply runs inside `async with agent` so the MCP toolsets
  actually connect and their tools load; otherwise the model only sees them as text.
- **Fleet auth** — the daemon loads OIDC client-credentials into its process env at startup
  (`env → OpenBao apps/mcp-multiplexer → local Claude MCP config`; values never logged), so
  the spawned multiplexer **and every nested `graph_orchestrate`-spawned agent** (via
  `_spawn_auth_headers`) authenticate to the jwt-protected fleet. **OpenBao is the source of
  truth** — never put these creds in a plaintext config/env file.
- **Chat tool policy** — the agent auto-runs the delegation/discovery surface
  (`graph_orchestrate`, `graph_search`, `find_tools`, `load_tools`) and read-only fleet tools
  from a chat message; mutating tools stay gated, and a spawned specialist's own fleet actions
  remain governed by the fail-closed ActionPolicy gate (OS-5.24).
- **One delegation surface (ECO-4.77)** — `graph_orchestrate(action=execute_agent)` is the
  single delegation entrypoint. The universal `invoke_specialized_agent` tool is a thin
  wrapper over the **same** orchestration core (`Orchestrator.execute_agent` → `run_agent`),
  not a parallel discovery/A2A/sub-agent-build path — so however the model phrases delegation
  it converges on one core (and one governance/identity path).

> The multiplexer is run as a stdio server today (debug); when it is deployed as a remote
> MCP server (Portainer), point `MESSAGING_MCP_CONFIG`'s entry at its URL instead — no other
> change. Genesis wires all of this in messaging **Step A4c**.

### Context burden (ECO-4.58)

The messaging agent is **lean by default**: the skill library is *not* pre-loaded, and
fleet MCP tools load **on demand** via the mcp-multiplexer's dynamic mode
(`find_tools`/`load_tools`). Context is also bounded per turn — the current message plus
top-K KG memory recall, **not** full chat history (history lives in the KG, retrieved as
needed). Opt into more via the settings above.
