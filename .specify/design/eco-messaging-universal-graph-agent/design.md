# Design Document: Messaging is thin transport to the ONE universal agent

CONCEPT:AU-ECO.messaging.universal-graph-agent ·
CONCEPT:AU-ECO.messaging.messaging-as-renderer ·
CONCEPT:AU-ECO.messaging.messaging-renderer-core-reaction ·
CONCEPT:AU-ECO.messaging.model-routed-inbound-responder ·
CONCEPT:AU-ECO.messaging.inbound-router ·
CONCEPT:AU-ECO.messaging.image-attachment-fallback ·
CONCEPT:AU-ECO.messaging.voice-attachment-fallback

> `agent_utilities/messaging/router.py` (primary), reaction rendering split
> across `agent_utilities/orchestration/reactions.py` (decision) and each
> backend's `send_reaction` (`agent_utilities/messaging/backends/*.py`,
> rendering).

## Decision — a chat turn IS a run of the universal orchestration pipeline, not a bespoke messaging reply path

`CONCEPT:AU-ECO.messaging.universal-graph-agent`

The router's own module docstring states the architecture directly: a chat
turn is NOT handled by a messaging-only reply path; it IS a run of
`Orchestrator.execute_agent` → `run_agent`, session-scoped per channel
(`router.py:1-12`). That single path natively provides what a messaging-only
implementation would have to hand-roll twice: conversation CONTINUITY (the
core memory primes each run with the channel's session mementos and persists
this turn back as one) and DYNAMIC CAPABILITY selection (specialists /
skills / A2A / swarms / fleet tools, ActionPolicy-governed).

**The rejected alternative** is a dedicated "messaging agent" with its own
tool/skill/delegation capability, parallel to (and inevitably drifting from)
the one every other entrypoint (web UI, CLI, MCP) uses. The router stays a
transport: receive → run the universal agent → send its text back, with a
hard reply-timeout plain-chat fallback (`_plain_chat_reply`) so a slow/hung
graph run still answers something.

### Pointer — `CONCEPT:AU-ECO.messaging.messaging-as-renderer` / `CONCEPT:AU-ECO.messaging.messaging-renderer-core-reaction`

`router.py:657-699`. The same principle applied to reactions specifically:
the instinctive, model-agnostic decision of WHICH emoji to react with lives
in the core orchestrator (`orchestration.reactions.decide_reaction`) so
EVERY entrypoint produces reactions from the same one heuristic. Messaging
is now just a RENDERER — `_react_in_background` calls `svc.react`, which
dispatches to the connected backend's own rendering primitive
(Mattermost/Slack/Telegram's `send_reaction`/`setMessageReaction` — the
`messaging-renderer-core-reaction` marker sites, one per backend). Reaction
rendering is best-effort and bounded: the reply is generated and sent on an
independent path, so a slow or failing reaction never delays or blocks the
user getting an answer (`router.py:679-699`).

### Pointer — fallback responder subsystem: `CONCEPT:AU-ECO.messaging.model-routed-inbound-responder`, `CONCEPT:AU-ECO.messaging.inbound-router`, `CONCEPT:AU-ECO.messaging.image-attachment-fallback`, `CONCEPT:AU-ECO.messaging.voice-attachment-fallback`

`_plain_chat_reply` (`router.py:1650`) is, in its own docstring's words,
"the only responder still owned by the messaging layer": a bare, tool-free
chat completion used ONLY when the universal graph agent times out or
errors, so a slow/hung graph run never leaves a message unanswered. Its full
tool/skill/MCP/delegation capability is NOT duplicated here — it lives on
the universal path. Four smaller decisions live entirely inside this
fallback:

- **`model-routed-inbound-responder`** (`router.py:1596`) — `_select_responder`
  defaults to the local LLM; an explicit `/claude` address (configurable via
  `MESSAGING_CLAUDE_TRIGGER`) routes to Claude, falling back to local with a
  visible note when no Anthropic key is configured — the fallback path still
  respects the same model-choice UX as the primary path, rather than always
  hard-coding one model.
- **`inbound-router`** (`router.py:1627`) — `_messaging_system_prompt` loads a
  dedicated messaging-assistant system prompt file
  (`prompts/messaging_assistant.json`) for this fallback specifically, with a
  hardcoded generic fallback if the file can't be read — the plain-chat path
  still gets messaging-appropriate framing, not the universal agent's system
  prompt (which assumes tool access this fallback doesn't have).
- **`image-attachment-fallback`** (`router.py:467`, `_fetch_image_parts`,
  `_sniff_image_media_type`) — image attachments are downloaded and INLINED
  as `pydantic_ai.BinaryContent`, not passed as a URL, so the vision model
  never has to fetch an external/token-bearing URL itself. The content-type
  is never trusted from the transport (Telegram serves `octet-stream`);
  magic-byte sniffing decides the real MIME type before model ingestion.
- **`voice-attachment-fallback`** (`router.py:986`, `_transcribe_attachments`)
  — voice/audio attachments are transcribed (via
  `.specify/design/eco-messaging-whisper-transcription/design.md`) and the
  text folded into the fallback prompt, so a voice message still gets a
  coherent answer even when the primary graph path is unavailable.

Both attachment-fallback paths are best-effort: an unreachable/invalid image
is skipped, not fatal to the rest of the turn.

## Risk Assessment

- **Blast Radius**: `agent_utilities/messaging/router.py`,
  `agent_utilities/orchestration/reactions.py`,
  `agent_utilities/orchestration/manager.py`,
  `agent_utilities/orchestration/agent_runner.py`.
- **Backward Compatible**: Yes.
- **Known weak point**: the fallback responder is deliberately capability-poor
  (no tools) — a user who hits the fallback during a graph-run timeout gets a
  materially different (weaker) agent than usual, with only the model-routing
  label as a visible signal of degraded mode.
