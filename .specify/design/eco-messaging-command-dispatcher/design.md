# Design Document: One command registry, answered before the agent ever sees the turn

CONCEPT:AU-ECO.messaging.single-inbound-command-dispatcher ·
CONCEPT:AU-ECO.messaging.shared-by-every-messaging ·
CONCEPT:AU-ECO.messaging.eco-2

> `agent_utilities/messaging/commands.py`

## Decision — a single, cross-platform command spec, checked BEFORE the agent turn

`CONCEPT:AU-ECO.messaging.single-inbound-command-dispatcher`

Slash-style commands (`/help`, `/status`, `/tools`, …) are answered by
`handle_command` (`commands.py:92`) at step 3b of the router's inbound
handling — AFTER the elicitation-bridge check (step 3a: is this message the
answer to a pending question?) and BEFORE the burst coalescer hands the turn
to the universal agent (step 4) (`router.py:625-650`). A recognized built-in
command short-circuits: answered immediately, never coalesced, never sent to
the LLM. `/claude`, `/skill`, and anything unrecognized return `None` from
`handle_command` and fall through to the normal agent turn.

**The rejected alternative** is per-platform command menus: Telegram's
`setMyCommands`, a Slack/Mattermost slash-command config, and
`agent-terminal-ui`'s own command list, each independently maintained and
each potentially answering `/status` differently. Instead one command spec
is registered on each platform via its native mechanism AND imported
directly by `agent-terminal-ui`, so the user gets the SAME commands with the
SAME behavior everywhere.

### Pointer — `CONCEPT:AU-ECO.messaging.shared-by-every-messaging`

`commands.py:33,38`. The `COMMANDS` tuple itself, and each
`MessagingCommand.surfaces` field (which surfaces expose a given command;
`("messaging", "terminal")` by default) — this is the single source of truth
the dispatcher above reads, and the mechanism by which `agent-terminal-ui`
consuming the same registry keeps chat and TUI commands in lockstep. Not a
second decision from the dispatcher, but the data structure that makes "one
command spec" literally true rather than aspirational.

### Pointer — `CONCEPT:AU-ECO.messaging.eco-2`

`commands.py:124`, `_capability_summary` — the implementation behind the
`/tools` command. It answers "what can you do" by counting the ingested
`Server`/`Tool`/`Skill` catalog **from the shared KG** and naming a few
examples, rather than loading every tool/skill definition into the model's
context to answer the question. The domain-triage tool flagged this id as a
bare legacy pillar citation (a retire candidate); reading the function shows
a real, deliberate choice (answer from the KG catalog, not a context dump),
so it is documented here as a pointer rather than retired.

## Risk Assessment

- **Blast Radius**: `agent_utilities/messaging/commands.py`,
  `agent_utilities/messaging/router.py` (dispatch order),
  `agent_utilities/messaging/backends/telegram.py` (`setMyCommands`
  registration).
- **Backward Compatible**: Yes.
- **Known weak point**: dispatch order is a hand-maintained sequence in
  `router.py` (elicitation check → command check → coalesce); nothing
  mechanically enforces that a future step insertion preserves "commands
  never reach the LLM."
