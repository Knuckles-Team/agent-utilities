# Entrypoint Unification — one orchestrator, thin entrypoints

> **Status:** north-star plan (parts in flight). The principle is enforced by the
> *Universal Capability / Entrypoint Parity* rule in `AGENTS.md`.

## The vision

`agent-utilities` is **one pydantic-ai Knowledge-Graph orchestrator**. Everything that talks
to a user or a system is a **thin entrypoint (transport)** that feeds that single orchestrator
and renders its output — never a place that re-implements agent capability:

| Entrypoint | What it is |
|---|---|
| messaging stack (Telegram, Slack, Teams, … — `messaging/`) | chat transport |
| `agent-webui` | browser transport |
| `agent-terminal-ui` | TUI transport |
| `geniusbot` | desktop transport |
| `agents/*/…/agent_server.py` (e.g. servicenow-api) | A2A/HTTP transport |

A capability (memory, RLM/mementos, dynamic agent/swarm/tool/skill selection, slash commands,
reactions/emotes, multimodal input, streaming, governance) is implemented **once in the core
orchestrator** and **inherited by every entrypoint**. An entrypoint contributes only: (1) how it
receives input, (2) how it renders the orchestrator's output for its medium. Nothing else.

**Reference wins already on this path:**
- **Slash commands** — one universal command set; `agent-terminal-ui` and Telegram both feed off
  it (CONCEPT:AU-ECO.messaging.single-inbound-command-dispatcher). The model to copy.
- **Messaging → universal graph** (in flight) — the messaging reply path is being collapsed onto
  `Orchestrator.execute_agent`/`run_agent`, so chat inherits memory + RLM mementos + dynamic
  capability selection instead of a bespoke messaging-only path (replaces the ECO-4.76 band-aid).

## Emotes / reactions, system-wide ✅ (core + messaging renderer done)

**Status:** the core capability and the messaging renderer are **done** (CONCEPT:AU-ECO.reactions.emitted-alongside-reply /
AU-ECO.reactions.one-emote-registry-governance / ECO-4.81). Reactions are no longer messaging-only — they are a first-class output
of the universal orchestrator, and the Telegram/messaging layer is now a *renderer* of that
output. The per-frontend renderers (webui / terminal-ui / geniusbot / `agent_server.py`) are
specified as a thin contract for their separate repos. **Full design + renderer contract:
[`reactions.md`](reactions.md).**

What landed:

1. **Core output type** ✅ — `agent_utilities/orchestration/reactions.py::AgentReaction`
   (`{emote, target_message_id?, intensity?}`), emitted by any agent turn. The instinctive
   heuristic (`decide_reaction`) moved out of `messaging/router.py` into core, so every
   entrypoint shares one decision (CONCEPT:AU-ECO.reactions.emitted-alongside-reply).
2. **One registry + governance** ✅ — `EmoteRegistry`: the single emote menu + an `allows()`
   gate reusing the `ActionPolicy` decision point (`reaction` kind). No per-surface emote
   list (CONCEPT:AU-ECO.reactions.one-emote-registry-governance).
3. **Messaging renderer** ✅ — `MessagingService.render_reaction(...)` renders a core
   `AgentReaction` via the backend `send_reaction` / Telegram `setMessageReaction`; the
   router's background reaction step calls the core decision and paints the result
   (CONCEPT:AU-ECO.messaging.messaging-as-renderer).
4. **Renderer contract** ✅ — the thin interface each entrypoint implements is documented in
   [`reactions.md`](reactions.md): Telegram (done) plus the stubs/contract for `agent-webui`
   reaction chips, `agent-terminal-ui` emote glyph, `geniusbot`, and the `agent_server.py`
   response-envelope `reaction` field. Those frontends are separate repos — the contract is
   defined here; the per-frontend renderers are the remaining follow-up.

Outcome: "react with 👍" is decided once in the orchestrator and shows up correctly in chat
(live), and — once each frontend implements the small renderer contract — in web, terminal,
desktop, and API, with zero duplicated emote logic.

## More unification opportunities (same pattern)

Each of these is currently (or at risk of being) re-implemented per surface; each should live once
in the core and be inherited:

- **Conversation history / continuity** — core memory + mementos keyed by session, NOT a
  per-surface recall (the messaging refactor; remove `recall_recent_messages`/`MESSAGING_HISTORY_TURNS`).
- **Multimodal input** (voice transcription, image/vision) — a core "normalize input → parts"
  step; entrypoints just hand raw attachments. (Messaging added these; they belong in core.)
- **Streaming / typing / progress** — one streamed-output protocol; renderers map it to typing
  indicators (Telegram), token streaming (webui/terminal), spinners (geniusbot).
- **Slash commands** ✅ already unified — keep extending the *same* registry, never a second one.
- **Dynamic capability selection** — agents/swarms/skills/tools chosen by the orchestrator
  (`graph_orchestrate`/`execute_agent`), never a per-entrypoint tool list.
- **Identity / auth / governance** — one ActorContext + ActionPolicy + Eunomia path; entrypoints
  present a token, they don't define policy.
- **Last-active routing / addressing** — `reach_user` resolves the channel; entrypoints register,
  they don't each invent routing.
- **Rate limiting / quotas / cost** — one governance layer over the orchestrator.
- **Reactions/emotes** — this plan.

## Definition of done for an entrypoint

An entrypoint is "correct" when it is *only* input-adaptation + output-rendering, and a new core
capability appears in it **with no entrypoint code change**. If adding a feature means editing N
entrypoints, the feature is in the wrong layer.
