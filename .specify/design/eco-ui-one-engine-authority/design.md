# Design Document: Entrypoint engine-authority gate — ONE core, thin entrypoints

CONCEPT:AU-ECO.ui.one-engine-authority

> `scripts/check_entrypoint_engine_construction.py`

## Decision — every entrypoint reaches the graph through ONE process-wide engine authority

`AGENTS.md`'s *Universal capability — ONE core, thin entrypoints* rule states
that every user/system-facing surface (the messaging stack, the A2A protocol
layer every `agents/*/agent_server.py` shares, `agent-webui`,
`agent-terminal-ui`, `geniusbot`) is a thin transport that reaches the graph
through the ONE process-wide engine authority
(`IntelligenceGraphEngine.get_active()` / `.get_or_create()`) — never a second,
hand-rolled construction path. This gate enforces it mechanically: an
entrypoint file may call `get_active()`/`get_or_create()` freely, but may never
(1) call the `IntelligenceGraphEngine` constructor directly, or (2) call
`create_backend(...)` with an explicit `backend_type=` (reserved for
connection-registry source adapters and focused backend tests). Test files are
exempt — a test legitimately builds a throwaway engine over an in-memory
backend as a fixture; this gate is about the SERVING path only.

This is not speculative: it is the exact shape of a real, found bug (D-WD-7).
`agent_webui.api_extensions.get_engine()`'s lazy-init fallback used to call
`create_backend(backend_type='ladybug', db_path=...)` and construct
`IntelligenceGraphEngine(...)` directly. `IntelligenceGraphEngine` is a
process-wide singleton (`_ACTIVE_ENGINE`) that ANY caller can win the
construction race for — so whichever entrypoint reached that hand-rolled
branch FIRST silently became the engine authority for the entire process,
handing every other route a disconnected, empty local LadybugDB instead of the
real operational graph ("Workflows shows nothing"). `get_or_create()` cannot be
raced around this way: it returns the existing singleton if one exists, and
otherwise builds the one sanctioned operational-authority backend
(`create_backend()` called with NO `backend_type` — the epistemic-graph engine
plus configured mirrors).

**The rejected alternative** was fixing only the one known offender
(`agent_webui.api_extensions.get_engine()`) and leaving the rule as prose in
`AGENTS.md`. A prose-only rule does not prevent the NEXT entrypoint from
reintroducing the same race — this is a mechanical, tree-wide static gate
(`scripts/check_entrypoint_engine_construction.py`, wired into
`.pre-commit-config.yaml`/CI) precisely because the failure mode is silent and
non-local: the symptom shows up in a DIFFERENT entrypoint than the one that
won the race.

## Risk Assessment

- **Blast Radius**: `scripts/check_entrypoint_engine_construction.py` (the
  gate), every entrypoint tree it scans (`agent_utilities/messaging/`,
  `agent-webui`, `agent-terminal-ui`, `geniusbot`, `agents/*/agent_server.py`).
- **Backward Compatible**: Yes — a static check with no runtime behavior
  change; it only fails a commit that reintroduces the banned construction
  pattern.
- **Known weak point**: static-analysis gates can miss dynamically-constructed
  call patterns (e.g. `getattr(engine_module, ctor_name)()`); this gate scans
  literal call syntax, not arbitrary indirection.
