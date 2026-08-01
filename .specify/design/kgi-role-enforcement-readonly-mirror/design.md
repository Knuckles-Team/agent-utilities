# Design Document: A mirror connection can never be written to directly — only through its fan-out outbox

> `agent_utilities/mcp/tools/write_ingest_tools.py:317-330`.

CONCEPT:AU-KG.ingest.role-enforcement

## Decision — 'read' and 'mirror' connections reject direct `target=` writes

`write_ingest_tools.py:317`.

**The problem**: `graph_write` accepts a `target=` connection name so a
caller can direct a write at a specific registered graph connection (not
just the default engine). Some registered connections are `'read'` (a data
source the graph consumes FROM) or `'mirror'` (a fan-out replica the graph
pushes TO through its own outbox mechanism) rather than a normal writable
target.

**The rejected alternative**: trusting every registered connection to accept
a direct `target=` write regardless of its declared role — the simpler,
un-guarded implementation, and a real hazard for a mirror specifically: a
direct write to a mirror bypasses the fan-out outbox that is supposed to be
the ONLY path material reaches it, meaning the mirror's content could
silently diverge from what the outbox believes it has replicated.

**The design chosen**: before honoring a `target=` write, `graph_write`
checks `registry.is_writable(name)` for every targeted connection. A `'read'`
or `'mirror'` connection is rejected with an explicit per-connection error
(`"connection '{name}' is read-only (role={registry.role(name)})"`) rather
than silently dropped or silently accepted — the caller gets a role-specific
reason, not a generic failure. Mirrors are written ONLY through the fan-out
outbox (see `.specify/design/kg-connector-mirroring/design.md` for the
mirroring architecture this protects), never through this direct-write path.

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/tools/write_ingest_tools.py`
  (`graph_write`'s target-resolution/role-check branch) and
  `kg_server.get_connection_registry()`.
- **Backward Compatible**: Yes — a connection never targeted with `target=`
  writes before this check is unaffected; the check only activates on an
  explicit `target=` write attempt.
- **Breaking Changes**: A caller that was previously (incorrectly) writing
  directly to a `'read'`/`'mirror'` connection now gets an explicit rejection
  instead of a silent write — a deliberate behavior change, framed as
  closing a hazard rather than a regression.
- **Known weak point**: role enforcement happens at the `graph_write` MCP
  tool boundary specifically — any OTHER write path that resolves a
  connection by name and writes to it directly (bypassing this specific
  tool) would not inherit this check unless it independently calls
  `registry.is_writable`.
