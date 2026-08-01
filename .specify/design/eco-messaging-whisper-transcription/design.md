# Design Document: Voice-note transcription — a lean in-process Whisper fallback

CONCEPT:AU-ECO.messaging.whisper-transcription

> `agent_utilities/messaging/voice.py`

## Decision — transcribe voice/audio to text so "just talk to the agent" works, with a lean CPU-only backend

When a user sends a voice note or audio attachment instead of text,
`voice.py` transcribes it with a Whisper backend and feeds the transcript
into the normal message flow — the reply path never needs to know the input
started as audio. Best-effort and opt-out (`MESSAGING_VOICE=0`); the model
loads lazily once and transcription runs off the event loop
(`voice.py:1-8`).

**The rejected alternative**, named directly in `_FasterWhisper`'s own
docstring, is depending on the full `audio-transcriber` package, which
hard-depends on `pyaudio`/`portaudio` — native audio-device bindings the
messaging container has no use for and does not ship. Instead this module
carries a thin, in-process `_FasterWhisper` transcriber using only
`faster-whisper`, the lean path, so the messaging container's dependency
footprint stays small. CPU-safe defaults are explicit: `device="cpu"`,
`compute_type="int8"` (the fastest CPU compute type), because the messaging
host has no GPU allocated (`voice.py:39-45`) — this is a deployment-topology
decision encoded directly in the transcription call, not a tunable left to
chance.

Timed segments (start/end per segment) are kept, not concatenated away,
specifically so a caller holding the source audio bytes can locate a
transcript's evidence loci (`voice.py:53-59`, citing
`CONCEPT:AU-KG.identity.evidence-spine-convergence` — the transcript is
KG evidence, not a throwaway string).

## Risk Assessment

- **Blast Radius**: `agent_utilities/messaging/voice.py`.
- **Backward Compatible**: Yes.
- **Known weak point**: `int8` CPU inference trades transcription latency
  and possibly accuracy for footprint; a deployment that later gets a GPU
  allocated to the messaging host would need this hard-coded `device="cpu"`
  changed, not auto-detected.
