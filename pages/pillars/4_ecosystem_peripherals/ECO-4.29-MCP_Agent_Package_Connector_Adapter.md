# MCP Agent-Package Connector Adapter (CONCEPT:AU-ECO.connector.mcp-package-adapter)

## Overview
A single adapter that turns *any* `agent-packages/agents/*` MCP server into a document-source
connector by calling a declared document-yielding tool and mapping the JSON result to
`SourceDocument`s. The entire fleet becomes ingestable with **config, not code** via a preset
catalog, which also encodes an explicit **Onyx connector-parity map** (every Onyx source routes
to a native package or a generic web/rest/database/filesystem connector).

## Implementation Details
- **Source Code**: `agent_utilities/protocols/source_connectors/connectors/mcp_package.py`,
  `connectors/package_manifest.py` (`PACKAGE_PRESETS`, `ONYX_CONNECTOR_PARITY`)
- **Pillar**: ECO
