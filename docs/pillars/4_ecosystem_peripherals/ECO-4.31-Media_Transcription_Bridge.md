# Media Transcription Bridge (CONCEPT:ECO-4.31)

## Overview
A thin client for the `faster-whisper` service (OpenAI-compatible
`/v1/audio/transcriptions`), giving the harness speech-to-text to complement the speech
*generation* in ECO-4.30. Transcribed text flows into the KG through the document-source
connector framework like any other document.

## Implementation Details
- **Source Code**: `agent_utilities/ecosystem/media/transcription.py`
- **Pillar**: ECO
