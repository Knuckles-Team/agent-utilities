# Design Document: One thin `graph_reach` MCP tool (+ REST twin) wraps `MessagingService`, instead of each messaging backend exposing its own tool

CONCEPT:AU-ECO.mcp.graph-reach-mcp-tool

> `agent_utilities/mcp/tools/reach_tools.py:1-60` (`register_reach_tools`,
> `graph_reach`).

## Decision — expose outbound user messaging as ONE action-routed `graph_reach` MCP tool (`reach_user` / `send` / `list_channels` / `last_channel` / `status`) that is a thin wrapper over `MessagingService`, with the REST route (`/graph/reach`) dispatching into the SAME service so the two surfaces never drift

The platform supports multiple messaging backends (Telegram, and whatever else is
configured). `graph_reach` gives an MCP client (Claude, or any other caller) one tool
with one schema to reach the operator, regardless of how many backends are
configured: `reach_user` routes to the user's **last-active** channel (OpenClaw-style
routing — "the channel the operator was last seen on wins"), falling back to a
configured default; `send` targets an explicit platform+channel; `list_channels`
and `last_channel` are read-only introspection; `status` reports configured/connected
backends. Every send is governed by the `ActionPolicy` gate and mirrored into KG
conversational memory (`reach_tools.py:29-31`), and `register_reach_tools` is
explicitly a "thin wrapper" (`reach_tools.py:3`) — all routing/backend logic lives in
`MessagingService`, not in the tool function.

## Rejected alternative — a separate MCP tool per messaging backend (`telegram_send`, `slack_send`, …)

Because the fleet's action-routed-tool convention already exists for exactly this
shape (one condensed tool, an `action` field, dispatch to the right backend
internally — the same pattern `AU-ECO.mcp.standardized-interfaces` documents for API
connectors generally), the alternative actually available here was per-backend
tools, mirroring how per-connector fleet tools work (`atlassian_*`, `github_*`, …).
That was rejected specifically for outbound messaging because the caller's real
intent is almost never "send via Telegram" — it is "reach the operator", and which
backend/channel currently reaches them is a routing decision (last-active-channel)
that the CALLER should not have to make. A tool per backend would push that decision
onto every caller (and onto the growing tool-count budget the condensed-surface
philosophy exists to protect, `AU-ECO.mcp.intent-surface-condensed-collapse`) instead
of onto the one service that actually knows the routing state.

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/tools/reach_tools.py`,
  `agent_utilities/messaging/service.py`, `gateway/graph_api.py` (`/graph/reach`).
- **Backward Compatible**: Yes — additive tool/route; no existing surface changes.
- **Known weak point**: `last_channel` routing depends on the platform having
  correctly recorded "last active" state for the user; a backend that does not
  report presence/activity events accurately degrades `reach_user` to the
  configured static default silently rather than erroring.
