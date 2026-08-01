# Design Document: Durable inbound inbox + retry — nothing goes unanswered

CONCEPT:AU-ECO.messaging.durable-inbound-pending ·
CONCEPT:AU-ECO.messaging.fire-and-forget-tasks ·
CONCEPT:AU-ECO.messaging.sending-reply-failed

> `agent_utilities/messaging/inbox.py`, consumed by
> `agent_utilities/messaging/router.py`

## Decision — record the inbound turn as durable BEFORE attempting a reply, close it only when actually delivered

`CONCEPT:AU-ECO.messaging.durable-inbound-pending`

Every inbound chat turn is recorded as a durable `:InboundMessage` node
BEFORE the reply is attempted, and marked `answered` only when a reply is
actually delivered (`inbox.py:1-6`). A reaper in the messaging daemon
re-attempts still-pending messages (the engine was down, a transient error)
with bounded retries (`MAX_ATTEMPTS = 4`).

**The rejected alternative** is best-effort delivery with no record: if the
KG engine is down when a message arrives, or the reply attempt throws, the
message is simply lost — the user gets no answer and no record exists that
they were ever owed one. This module turns "I saved your message" from an
aspirational log line into a real, checkable promise: a turn that fails
mid-flight is found and answered when the system recovers, not silently
dropped.

On retry, the reaper re-writes the FULL record (platform, channel_id,
message_id, text, session, received_at), not just the incremented attempts
counter (`inbox.py:151-165`) — the durable backend replaces a node's
property blob on upsert, so a bare `{attempts}` write would wipe the exact
fields (`platform`/`channel_id`/`text`) the NEXT retry needs to re-send. A
counter-only write would silently degrade every subsequent retry into a
no-op with an incrementing number and no content to act on.

### Pointer — `CONCEPT:AU-ECO.messaging.fire-and-forget-tasks`

`router.py:706`, `_BG_TASKS: set[asyncio.Task[Any]]`. The background-task
primitive `_spawn_bg` uses throughout the router — reactions, KG enrichment,
the durable-inbox close — to run work off the reply path while keeping a
STRONG reference to the task until it finishes. Python's `asyncio` does not
keep a strong reference to a bare `create_task()` result; without this set,
the task object can be garbage-collected mid-flight and silently cancelled.
This is the mechanical enabler that makes "off the reply path" actually
reliable rather than a a task that may or may not survive to completion.

### Pointer — `CONCEPT:AU-ECO.messaging.sending-reply-failed`

`router.py:527`, the `_send` helper's `except` clause: a failed
`send_message`/`reply_to` call is logged and swallowed, not re-raised. This
is safe SPECIFICALLY BECAUSE of the durable-inbox decision above — the
inbound message is still `pending` at this point (it is only marked
`answered` after `_send` succeeds), so a swallowed send failure does not
lose the turn; the reaper retries it. Removing the durable inbox without
also changing this swallow would turn "log and continue" into silent
message loss.

## Risk Assessment

- **Blast Radius**: `agent_utilities/messaging/inbox.py`,
  `agent_utilities/messaging/router.py`.
- **Backward Compatible**: Yes.
- **Known weak point**: `MAX_ATTEMPTS = 4` is a hard bound — a message whose
  failure cause outlives 4 reaper passes is left `pending` forever with no
  further automatic retry or operator alert.
