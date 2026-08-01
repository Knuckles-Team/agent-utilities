# Design Document: A blocked loop/elicitation reaches the user through chat, and resumes on reply

CONCEPT:AU-ECO.messaging.last-active-channel ·
CONCEPT:AU-ECO.messaging.elicitation-loop-bridge

> `agent_utilities/observability/approval_manager.py` (the trigger side),
> `agent_utilities/messaging/service.py` (the wait/resolve side),
> `agent_utilities/messaging/router.py` (the resolve trigger).

> This document merges two triage-suggested heads (`last-active-channel` was
> proposed as its own decision; `elicitation-loop-bridge` as a pointer into a
> larger, unrelated 10-member cluster). Reading both sites shows they are the
> SAME bridge described from its two ends — the trigger
> (`approval_manager.py`) and the wait/resolve mechanism
> (`service.py`/`router.py`) — so one document, not two, is the honest unit
> here.

## Decision — a blocked elicitation/goal-loop is unblocked by a chat reply, on the SAME question

When an MCP elicitation (or a blocked goal-loop question) needs a human
answer, `_bridge_elicitation_to_messaging` (`approval_manager.py:325`) pushes
the question to the user's **last-active channel** (`CONCEPT:AU-ECO.messaging.last-active-channel`)
so the question is answerable from chat, not only from the web UI. This
spawns a background task; when the user replies (delivered by the inbound
router), the SAME elicitation is resolved — so a goal-loop/agent blocked on
input can be unblocked from chat.

**The rejected alternative** is a chat-only or UI-only elicitation path.
Instead both surfaces answer the SAME pending request: the web UI's
`/api/approve` path is unchanged, and whichever surface answers first wins
(`approval_manager.py:350-352`) — a user who is not at their desk can still
unblock an agent from their phone.

The resolve mechanism (`CONCEPT:AU-ECO.messaging.elicitation-loop-bridge`) lives in
`MessagingService`: `reach_user_and_wait` registers a `Future` in
`self._pending`, keyed by `"platform:channel_id"` (`service.py:60`); the
inbound router recognizes that an incoming message on that channel is the
answer to a pending question — rather than re-routing it to the universal
agent as a fresh turn — and resolves the future (`router.py:422`, the
handler's step 1: *"Delivers the message to a waiting goal-loop if it is the
answer to a question the loop asked... in which case it is NOT re-routed to
the agent"*). This pending-future state is explicitly **in-process (daemon)
only** (`service.py:60` comment) — it does not survive a daemon restart,
unlike the durable inbound queue documented in
`.specify/design/eco-messaging-durable-inbound/design.md`.

A synchronous wrapper, `reach_user_sync` (`service.py:439`), exists
specifically for the goal-loop, which runs OUTSIDE an event loop and
therefore cannot `await` the async `reach_user`/`reach_user_and_wait`
directly.

## Risk Assessment

- **Blast Radius**: `agent_utilities/observability/approval_manager.py`,
  `agent_utilities/messaging/service.py`,
  `agent_utilities/messaging/router.py`.
- **Backward Compatible**: Yes.
- **Known weak point**: the pending-future registry is in-process only — a
  daemon restart while a question is outstanding silently drops the pending
  future; the caller that awaited it sees it never resolve (bounded by
  whatever timeout the caller applies, not by this mechanism).
