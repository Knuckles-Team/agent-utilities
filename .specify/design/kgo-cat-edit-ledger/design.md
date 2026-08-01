# Design Document: Object edits are a durable, revertible ledger — with an atomic compare-and-set for concurrent agent writes

CONCEPT:AU-KG.ontology.edit-ledger-writeback ·
CONCEPT:AU-KG.ontology.optimistic-concurrency-object-property

> `agent_utilities/mcp/tools/ontology_tools.py:1720-1900` (`object_edits` MCP
> tool), `agent_utilities/knowledge_graph/ontology/edits.py`.

## Decision — every object mutation is recorded to a durable ledger with revert + as-of-time query, not applied as a bare, unaudited write

`CONCEPT:AU-KG.ontology.edit-ledger-writeback`

The `object_edits` tool description states the contract directly
(`ontology_tools.py:1721-1729`): a "durable object-edit ledger" that can
`record` a structured edit (`property_set`/`link_add`/`link_remove`/
`object_create`/`object_delete`), `revert` an edit by id, or read per-object
`history` / an `as_of` point-in-time snapshot. Every edit is attributed to an
`actor` (default `"system"`, but explicit per-call). This is a real alternative
to two simpler designs: applying writes directly with no ledger (loses
history/revert entirely), or a bolt-on audit log that records *that* a write
happened without the structure needed to *replay/revert* it. Recording the
edit as a typed, reversible unit means "who changed what, and can we undo it"
is answerable for every object mutation that goes through this path, and
`as_of` lets a caller reconstruct an object's state at a prior point in time
from the ledger rather than needing a separate snapshot/backup mechanism.

### Pointer — `CONCEPT:AU-KG.ontology.optimistic-concurrency-object-property`

`ontology_tools.py:1759-1770, 1804-1826`. A `property_set` edit accepts an
optional `expect` map: field → the value the object must *still* hold for the
set to apply. When `expect` is non-empty, the set runs through the engine's
atomic `compare_and_set_node_fields` under the write lock, and **the ledger
edit is recorded only if the precondition still held** — an unapplied set
returns `{'applied': false}` and writes nothing to the ledger, "never a
misleading audit edit" (`ontology_tools.py:1825`). This is the specific
mechanism that makes the ledger safe for concurrent agents editing the same
object: without it, two agents racing a `property_set` on the same object
would have the second write silently clobber the first with no signal that a
conflict occurred. Empty `expect` (the default) is unconditional set,
identical to prior behavior — so the concurrency guard is strictly opt-in and
backward compatible.

## Risk Assessment

- **Blast Radius**: `mcp/tools/ontology_tools.py` (`object_edits` tool),
  `knowledge_graph/ontology/edits.py` (`Edit`, `EditType`, `revert_edit`).
- **Backward Compatible**: Yes — `expect` defaults to empty (unconditional
  set); the ledger itself is the pre-existing edit-recording path.
- **Known weak point**: `compare_and_set_node_fields`'s atomicity depends on
  the live engine's backend supporting it — the code checks
  `backend = getattr(engine, "backend", None)` and only takes the
  compare-and-set path when a backend is present; a caller relying on
  `expect` against an engine/backend combination that doesn't support the
  atomic primitive would not get the safety guarantee the tool description
  promises (falls through to a different, unguarded path).
