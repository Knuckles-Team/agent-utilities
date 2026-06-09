# External Permission Sync (CONCEPT:ECO-4.28)

## Overview
Maps a connector-reported `ExternalAccess` (source groups/users/markings) onto the existing
KG-2.46 permissioning model: groups/users become `read_roles` on a `NodeACL`, compartments
become mandatory `Marking`s, and the document's controls propagate to its chunks along
`HAS_CHUNK` edges. No new permission store — the default-on `enforce()` read gate then filters
retrieval for actors who lack access.

## Implementation Details
- **Source Code**: `agent_utilities/protocols/source_connectors/permission_sync.py`
- **Pillar**: ECO
