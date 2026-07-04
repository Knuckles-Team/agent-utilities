# Checkpointed Incremental Poll (CONCEPT:AU-ECO.connector.incremental-poll-watermark)

## Overview
A serializable `ConnectorCheckpoint` (cursor / watermark / seen-ids / state) returned by a
`PollConnector` so the next poll resumes exactly where it left off. Round-trips through the
existing `DeltaManifest` (KG-2.8) under a `connector_checkpoint` category — an unchanged
source costs a delta, not a full re-scan.

## Implementation Details
- **Source Code**: `agent_utilities/protocols/source_connectors/checkpoint.py`
- **Pillar**: ECO
