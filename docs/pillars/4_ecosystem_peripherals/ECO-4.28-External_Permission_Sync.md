# External Permission Sync (CONCEPT:AU-ECO.connector.external-permission-sync)

## Overview
Maps a connector-reported `ExternalAccess` (source groups/users/markings) onto the existing
AU-KG.ontology.redact-object-materialize-restricted permissioning model: groups/users become `read_roles` on a `NodeACL`, compartments
become mandatory `Marking`s, and the document's controls propagate to its chunks along
`HAS_CHUNK` edges. No new permission store — the default-on `enforce()` read gate then filters
retrieval for actors who lack access.

**Fail-closed default (AU-P0-4).** An `ExternalAccess` with `is_public=False` and empty
`group_ids`/`user_emails` registers no discretionary ACL at all — `sync_access` only builds one
when `roles` is non-empty — and would otherwise silently fall through to the default-allow read
gate. `ExternalAccess.quarantined()` (`is_public=False` + the `connector-unconfigured-acl`
marking, no actor holds it by default) closes that gap: an unknown/unconfigured connector
document is denied until an operator reviews it and grants the marking. This is now the default
the generic `mcp_package`/`mcp_tool` connectors report when a preset/instance has no `acl_*`
fields configured, via `default_external_access()` — replacing the previous
`ExternalAccess.public()` default. `CONNECTOR_DEFAULT_PUBLIC=true` opts a dev/local deployment
back into the old public-by-default behavior (default `false`).

## Implementation Details
- **Source Code**: `agent_utilities/protocols/source_connectors/permission_sync.py`,
  `agent_utilities/protocols/source_connectors/base.py` (`ExternalAccess.quarantined()`,
  `default_external_access()`, `CONNECTOR_UNCONFIGURED_MARKING`)
- **Pillar**: ECO
