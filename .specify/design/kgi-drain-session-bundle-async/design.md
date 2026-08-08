# Design Document: Session-bundle upload is a background task, not a synchronous loop that blows the MCP call window

> `agent_utilities/mcp/tools/write_ingest_tools.py:1648-1670` (the `upload`
> action).

CONCEPT:AU-KG.ingest.drain-session-bundle

## Decision — enqueue a durable `session_upload` background task, mirroring `source_sync`/`graph_ingest`

`write_ingest_tools.py:1651-1663`.

**The problem, named directly in the code comment**: each uploaded session
bundle expands into MANY usage-store rows (sessions + events + tool calls +
the FTS index). The OLD synchronous `record_bundle` loop "blew past the 60s
MCP client window under load EVEN AT batch=10" — i.e. even a small batch of
10 sessions could exceed the client's call timeout once expanded into its
full row set.

**The rejected alternative is that prior synchronous loop**: process every
bundle inline within the MCP call, returning only once all bundles are fully
recorded. It worked for small batches but failed exactly where it mattered —
under real load.

**The design chosen**: mirror the pattern already used by `source_sync` and
`graph_ingest` (see `.specify/design/kg-connector-mirroring/design.md`'s
Decision 2 for the analogous `chunked_drain` pagination pattern) — ENQUEUE
the bundles as a durable `session_upload` background task and return a
`job_id` IMMEDIATELY. The host daemon's task worker drains it (parse → usage
store) off the call path, so the MCP client is never blocked on the full
expansion cost. A deliberate exception: a genuinely TINY batch still runs
inline (auto-sized, no user-exposed knob) since the overhead of a background
task round-trip isn't worth it below some threshold — the caller doesn't
choose sync-vs-async; the tool decides based on batch size.

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/tools/write_ingest_tools.py`
  (`upload` action), `agent_utilities/usage/models.py`
  (`ParsedSessionBundle`), `agent_utilities/usage/recorder.py`
  (`get_usage_recorder`).
- **Backward Compatible**: Yes — the caller-facing contract (submit bundles,
  get a result) is preserved; only the synchronous-vs-background execution
  path changed, transparently to small batches.
- **Breaking Changes**: A caller that assumed the OLD synchronous behavior
  (result available immediately in the response) now gets a `job_id` for
  batches above the inline-auto-size threshold and must poll — a deliberate,
  load-bearing behavior change, not a regression.
- **Known weak point**: the inline-vs-background threshold is "auto-sized,
  no user knob" — there is no way for a caller to force one mode or the
  other, so a caller that specifically wants synchronous confirmation for a
  batch just above the auto threshold has no override.
