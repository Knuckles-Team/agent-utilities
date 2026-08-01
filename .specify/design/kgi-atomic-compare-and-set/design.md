# Design Document: `compare_and_set` — a first-class optimistic-concurrency write primitive on the MCP mutation surface

> `agent_utilities/mcp/tools/write_ingest_tools.py:120-210` (the `graph_write`
> tool's `compare_and_set` action).

CONCEPT:AU-KG.ingest.atomic-compare-and-set

## Decision — a conditional-update primitive, not "read then write" left to the caller

`write_ingest_tools.py:131`, `189`.

**The problem**: two agents (or an agent and a background task) can hold
stale reads of the same node and race to mutate it. The conventional
client-side pattern — read the node, decide an update, write it back — is
inherently racy: nothing prevents a second writer's update from landing
between the first writer's read and write, silently clobbering it (lost
update).

**The rejected alternative**: leaving concurrency safety to the caller (read
-then-write, or a caller-managed lock/mutex around the whole read-modify
-write sequence). That pattern is what MCP write tools have historically
exposed for every other action on `graph_write`; `compare_and_set` is
explicitly called out as making atomic conditional-update "a first-class
agent capability" instead.

**The design chosen**: `compare_and_set` merges `updates` into `node_id`
ONLY if every field in `conditions` still equals the node's current value
(a missing field is treated as null), evaluated and applied under the
engine's write lock. It returns `{"action": "compare_and_set", "node_id":
..., "applied": <bool>}` — `applied=False` means the precondition failed (a
different agent won the race), and this result is surfaced directly to the
caller, never swallowed or silently retried. This gives agents three
distinct primitives on one call: optimistic concurrency (retry on
`applied=False`), conditional state transitions (only flip a status field if
it's still in the expected prior state), and atomic reservations (claim a
resource only if unclaimed).

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/tools/write_ingest_tools.py`
  (`graph_write`'s `compare_and_set` branch) and the engine's write-lock path
  it evaluates under.
- **Backward Compatible**: Yes — an additive action on the existing
  `graph_write` tool; other actions (`upsert`, bulk ingest, etc.) are
  unaffected.
- **Breaking Changes**: None.
- **Known weak point**: `applied=False` is returned, not raised — a caller
  that doesn't explicitly check the `applied` field in the response can
  silently treat a lost race as a successful write, since the tool call
  itself does not error.
