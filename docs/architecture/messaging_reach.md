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

## Configuration

| Setting | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Enables the Telegram backend (auto-detected) |
| `MESSAGING_DEFAULT_PLATFORM` | Default platform when no last-active channel (default `telegram`) |
| `MESSAGING_DEFAULT_CHANNEL` | Default chat id for `reach_user` fallback |
| `MESSAGING_AGENT` | Agent name the inbound planner routes spontaneous messages to |
