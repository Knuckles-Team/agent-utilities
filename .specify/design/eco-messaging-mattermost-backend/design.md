# Design Document: Mattermost as a first-class bidirectional messaging backend

CONCEPT:AU-ECO.messaging.mattermost-backend

> `agent_utilities/messaging/backends/mattermost.py`

## Decision — Mattermost is modelled exactly on the Telegram backend, with lazy WebSocket start

The Mattermost backend implements the abstraction documented in
`.specify/design/eco-messaging-native-backend-abstraction/design.md`. The
module docstring is explicit about the reference implementation it copies:
"modelled exactly on the Telegram backend" — the universal orchestrator is
the one agent, this backend is a thin transport that only (1) receives
Mattermost posts and normalizes them into the shared `InboundEvent`, and (2)
renders the orchestrator's reply back via a bot token
(`mattermost.py:1-7`). All capability (memory, dynamic delegation,
reactions, last-active routing) lives in the core, not in this file.

**The rejected alternative** each new backend re-deriving its own shape from
scratch — instead the Mattermost implementation is deliberately a
close structural mirror of Telegram's, so the two backends can be read
side-by-side and reviewed for behavioral parity.

**The one Mattermost-specific decision that earns its own reasoning:** the
WebSocket event consumer (`mattermostdriver`'s `posted` event stream) is
started lazily by `listen()`, NOT by `connect()`
(`mattermost.py:10-13`). `connect()` opens only the REST session (bot
login, resolve the bot's own user id so the inbound stream can filter out
its own posts). This matters because a send-only consumer — e.g. the
`graph_reach` MCP tool running in a client process that only needs to POST
messages — calls `connect()` but never `listen()`, and therefore opens only
the REST session and never a second WebSocket that would duplicate the
daemon's own inbound stream. Collapsing connect+listen into one step would
mean every send-only caller also opened a redundant WebSocket connection to
Mattermost.

Install is an optional extra (`pip install
agent-utilities[messaging-mattermost]`), consistent with every other
backend — the abstraction supports 17 platforms without forcing every
deployment to install all 17 platforms' SDKs.

## Risk Assessment

- **Blast Radius**: `agent_utilities/messaging/backends/mattermost.py`.
- **Backward Compatible**: Yes.
- **Known weak point**: bot-id resolution (`get_user, "me"`) is best-effort
  and falls back to the configured `MATTERMOST_BOT_USER` on failure
  (`mattermost.py:134`) — a misconfigured fallback would cause the backend
  to fail to filter out its own posts, echoing them back as inbound events.
