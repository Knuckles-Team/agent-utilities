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
  it (CONCEPT:ECO-4.57). The model to copy.
- **Messaging → universal graph** (in flight) — the messaging reply path is being collapsed onto
  `Orchestrator.execute_agent`/`run_agent`, so chat inherits memory + RLM mementos + dynamic
  capability selection instead of a bespoke messaging-only path (replaces the ECO-4.76 band-aid).

## Saved plan: emotes / reactions, system-wide

Today reactions live only in the messaging layer (`send_reaction`, Telegram `setMessageReaction`,
the instinctive-reaction logic, CONCEPT:ECO-4.60). Promote it to a **first-class orchestrator
output** so every surface inherits it natively.

1. **Core output type** — let the orchestrator emit a structured `AgentReaction`/emote alongside
   (or instead of) text: `{target, emote, intensity?}`. The agent decides to react as a normal
   part of producing a turn (the existing instinctive-reaction heuristic moves into core).
2. **Per-entrypoint renderers** (the ONLY per-surface code):
   - Telegram/messaging → `setMessageReaction` (already exists; becomes a renderer of the core output).
   - `agent-webui` → emoji reaction chips on a message.
   - `agent-terminal-ui` → inline emote glyph / reaction line in the TUI.
   - `geniusbot` → desktop reaction affordance.
   - `agent_server.py` → reaction field in the A2A/HTTP response envelope.
3. **One registry** of available emotes + governance (which emotes a principal/context may use),
   shared by all renderers — no per-surface emote list.

Outcome: "react with 👍" is decided once in the orchestrator and shows up correctly in chat, web,
terminal, desktop, and API — zero duplicated emote logic.

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
