# Design Document: Burst-mode message coalescing — one holistic reply per burst

CONCEPT:AU-ECO.messaging.burst-mode-coalescing ·
CONCEPT:AU-ECO.messaging.debounce-timer-cancel

> `agent_utilities/messaging/coalescer.py`, consumed by
> `agent_utilities/messaging/router.py`

## Decision — collapse a burst of rapid messages into ONE agent turn, not N replies

`CONCEPT:AU-ECO.messaging.burst-mode-coalescing`

When a user fires several messages in quick succession, `BurstCoalescer`
accumulates them per-conversation key (e.g. `"telegram:<chat>"`) and flushes
the whole batch to one handler — one holistic reply, one LLM call — instead
of answering each message individually. A debounce window (`window_s`,
default 2.5s) restarts on every `submit`; a hard cap (`max_wait_s`, default
12.0s) guarantees a continuous typer still gets a reply rather than being
debounced forever. `window_s=0` disables coalescing (flush each item
immediately) — an explicit escape hatch, not a special-cased code path.

**The rejected alternative** is answering each inbound message as its own
turn. For a user who sends "wait", "actually", "never mind, do X instead" as
three separate messages half a second apart, three independent replies would
each answer a stale, superseded request — the coalescer exists so the agent
sees the FINAL intent as one turn.

This is a shared core primitive: `agent-terminal-ui` imports the same
`BurstCoalescer` (module docstring) so burst behavior is identical across
every chat surface, not re-implemented per surface.

### Pointer — `CONCEPT:AU-ECO.messaging.debounce-timer-cancel`

`coalescer.py:74`. The single sharpest correctness detail in this module: the
debounce timer is cancelled **only** when `_flush` is invoked from a
*different* task than the timer's own (the hard-cap path in `submit`). When
`_wait_and_flush` — the timer task itself — calls `_flush`, cancelling
`timer` would cancel the CURRENTLY RUNNING task and kill `_on_flush` (the
whole reply) at its very first `await`. The code's own comment names the
regression this fixes: *"message received, reaction maybe, but no reply"* —
a user-visible bug that traces to exactly this self-cancellation. The guard
(`timer is not asyncio.current_task()`) is what makes the two flush paths
(hard-cap vs. natural debounce-expiry) safe to share one `_flush`
implementation.

## Risk Assessment

- **Blast Radius**: `agent_utilities/messaging/coalescer.py`,
  `agent_utilities/messaging/router.py` (burst handling call sites),
  `agent_utilities/core/config.py:2407` (window/cap settings).
- **Backward Compatible**: Yes.
- **Known weak point**: coalescing is per-key (per-conversation) in-memory
  state; a process restart mid-burst drops the buffered-but-unflushed items
  for that key (they were never durable — contrast with
  `.specify/design/eco-messaging-durable-inbound/design.md`, which IS
  durable, for the reply-delivery side).
