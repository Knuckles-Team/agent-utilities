# Design Document: `reach_user` is a native agent tool over the messaging reach service

CONCEPT:AU-ECO.messaging.universal-agent-reach-user

> `agent_utilities/tools/agent_tools.py:108`, registered in
> `agent_utilities/tools/tool_registry.py:211`.

## Decision — the agent proactively reaches the user through the SAME governed reach service, as a tool call

`reach_user` is registered as a first-class tool in the universal agent's
tool registry (`tool_registry.py:211`, alongside `invoke_specialized_agent`,
`share_reasoning`, and the bus tools) — so the agent can proactively tell
the user something, or ask a question and optionally wait for a reply
(`wait_for_reply=True`), as an ordinary tool call rather than a
messaging-specific side channel the agent has no visibility into.

**The rejected alternative** is a reply-only messaging model, where the
agent can only respond to an inbound message and has no way to initiate
contact — e.g. surfacing a long-running background task's result, or asking
a clarifying question mid-task without the user having sent anything first.
Exposing `reach_user` as a tool means any agent run (not just one already
inside a chat turn) can reach the user.

`reach_user` dispatches into the SAME `MessagingService` documented in
`.specify/design/eco-messaging-reach-service/design.md`: routing follows the
user's most-recently-used channel (falling back to the configured default),
and the send is governed by the same fail-closed `ActionPolicy` gate every
other outbound path goes through — the tool is a thin wrapper over the one
governed service, not a bypass of it.

## Risk Assessment

- **Blast Radius**: `agent_utilities/tools/agent_tools.py`,
  `agent_utilities/tools/tool_registry.py`.
- **Backward Compatible**: Yes.
- **Known weak point**: `wait_for_reply=True` blocks the calling agent run on
  a human reply with no messaging backend configured, this tool degrades
  silently to a no-op send (per the reach service's own no-op-when-unconfigured
  behavior) rather than the agent being told up front that no channel exists.
