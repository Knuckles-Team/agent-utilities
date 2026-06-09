# Connector Registry + Factory (CONCEPT:ECO-4.27)

## Overview
A `@register_source` decorator + `build_connector(source_type, config)` factory + `pkgutil`
`discover()` so connectors self-register and are built by name. Discovery runs on the live
ingestion path (Wire-First; the import-graph checker cannot see decorator registration).

## Implementation Details
- **Source Code**: `agent_utilities/protocols/source_connectors/registry.py`
- **Pillar**: ECO
