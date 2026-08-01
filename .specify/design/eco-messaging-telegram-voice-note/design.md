# Design Document: Telegram voice notes are classified as a distinct attachment type

CONCEPT:AU-ECO.messaging.telegram-voice-note

> `agent_utilities/messaging/backends/telegram.py:105`

## Decision — `msg.voice` becomes `MediaType.VOICE_NOTE`, not a generic `MediaType.AUDIO`

Telegram's Bot API distinguishes a "voice note" (`msg.voice`, the round
push-to-talk bubble) from a regular audio file attachment (`msg.audio`).
The backend preserves that distinction when normalizing into the shared
`MediaAttachment` model, tagging voice notes as `MediaType.VOICE_NOTE`
specifically (`telegram.py:105`) rather than collapsing both into
`MediaType.AUDIO`.

**The rejected alternative** is normalizing both into one generic audio
type — the platform-native distinction would be lost, and any downstream
consumer (notably transcription, see
`.specify/design/eco-messaging-whisper-transcription/design.md`) could not
tell "the user tapped-and-held to talk" from "the user forwarded an MP3"
without re-inspecting Telegram-specific metadata. Keeping the type at the
normalization boundary means every backend-agnostic downstream consumer
(coalescer, router, transcription) can branch on `MediaType.VOICE_NOTE`
without knowing which platform produced it.

## Risk Assessment

- **Blast Radius**: `agent_utilities/messaging/backends/telegram.py`,
  `agent_utilities/messaging/models.py` (`MediaType` enum).
- **Backward Compatible**: Yes.
- **Known weak point**: this distinction is currently sourced ONLY from the
  Telegram backend; other platforms with an equivalent native voice-note
  concept would need to make the same classification call independently —
  there is no shared cross-platform voice-note detector.
