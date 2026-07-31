# fastmcp 4 as the default MCP stack (CONCEPT:AU-ECO.mcp.protocol-compat-bridge)

## Summary

`agent-utilities` targets `fastmcp>=4.0.0b1` as the single default in the `[mcp]` extra —
there is no `mcp-v4` opt-in fork. Every extra that pulls MCP support (`mcp`,
`agent-runtime`, `agent-headless`, `serving`, `test`, `test-backends`, `all`) resolves
to the same `fastmcp==4.0.0b1` / `fastmcp-slim==4.0.0b1` / `mcp==2.0.0` version set in
one unified `uv.lock` — no `[tool.uv.conflicts]` fork.

## Why this needed more than bumping a version floor

`pydantic-ai-slim[mcp]` (the client-side toolset au's own code calls into) still declares
`fastmcp-slim[client]<4,>=3.3.0` as of 2.21.0, the latest published release. That cap is
conservative rather than a known incompatibility — 2.18.0 declared the same extra with no
upper bound, and the `<4` was added defensively in 2.19.0 while fastmcp 4 was still
pre-release. Empirical testing (a real fastmcp-4 server driven end-to-end through
`pydantic_ai.mcp.MCPToolset`: connect, list tools, call a tool) found the cap is not
protecting against a fundamental incompatibility, but it IS protecting against two real
gaps that had to be bridged:

1. **`fastmcp.client.Client` defaults to `mode="auto"`**, which negotiates the modern
   `server/discover` connect era against a fastmcp-4 server and leaves
   `Client.initialize_result` as `None`. `MCPToolset.__aenter__` unconditionally asserts
   `client.initialize_result is not None` — a hard failure on every real connection unless
   the client is pinned to `mode="legacy"` (today's initialize handshake).
2. **MCP SDK v2 renamed several protocol fields from camelCase to snake_case.** `fastmcp`
   ships its own deprecation bridge (`fastmcp._compat`) covering most of what
   `pydantic_ai.mcp` reads, but not `PromptsCapability`/`ResourcesCapability`/
   `ToolsCapability.listChanged` (read unconditionally on every `MCPToolset.__aenter__`)
   or `ToolExecution.taskSupport` (read by `get_tools()`). `mcp.shared.exceptions.McpError`
   was also renamed to `MCPError`, breaking `pydantic_ai.mcp`'s tool-call error handling.

Both gaps live inside `pydantic_ai.mcp` / `fastmcp`'s own code, not anywhere this package
calls directly — they can't be fixed by changing how *we* invoke the API, only by bridging
the renamed/defaulted surface at the boundary where we construct MCP clients.

## The bridge: `agent_utilities/mcp/protocol_compat.py`

- `install_mcp_v2_bridge()` — patches the four missing camelCase properties + the
  `McpError`/`MCPError` alias, using the same technique `fastmcp._compat` uses for the
  rest (a plain warn-once property), guarded so it never shadows a real upstream fix.
- `force_legacy_protocol_mode(toolset)` — pins an already-constructed `MCPToolset`'s
  underlying `fastmcp.Client.mode` to `"legacy"` before first use. `Client.mode` is a
  plain instance attribute read lazily at connect time, and `MCPToolset` doesn't expose a
  `mode` passthrough for its convenience constructors, so this reaches in post-construction
  instead (unwrapping `WrapperToolset.wrapped`, e.g. the `PrefixedToolset` that
  `pydantic_ai.mcp.load_mcp_toolsets` returns).

Both are wired into every toolset-construction call site in this package:
`mcp/toolset_factory.py` (`build_http_toolset`, `build_stdio_toolset`),
`agent/factory.py` (the raw in-process `FastMCP` instance path), `core/config.py`
(`load_mcp_servers_from_config`), and `graph/executor.py` (the lazy per-agent MCP load).

```mermaid
flowchart TD
    A[toolset_factory.build_http_toolset / build_stdio_toolset] --> C[MCPToolset construction]
    B[agent/factory.py: MCPToolset(server)] --> C
    D[core/config.py: load_mcp_servers_from_config] --> C
    E[graph/executor.py: lazy MCP load] --> C
    C --> F[force_legacy_protocol_mode]
    F --> G[toolset.client.mode = 'legacy']
    H[install_mcp_v2_bridge, called once per call site] --> I[mcp.types capability/exception patches]
    G --> J[Real fastmcp-4 server: connect / list tools / call tool]
    I --> J
```

## Guarding the declared floor: `check_mcp_sdk_floor()`

The bridges above assume the *installed* `mcp`/`fastmcp` actually satisfy the `[mcp]`
extra's declared floor (`fastmcp>=4.0.0b1`, transitively `mcp>=2.0.0,<3.0.0`). Nothing
previously asserted that at runtime, so a deployment that resolved an older SDK line
(observed live: a pod running `mcp` 1.29.0 / `fastmcp` 3.4.5 against v2-targeted source)
only failed as a swallowed `ImportError` deep inside `mcp/child_resilience.py`'s hard
`from mcp.shared.exceptions import MCPError` import — which made
`agent_utilities.mcp.multiplexer` unimportable and silently dropped every fleet
meta-tool (`find_tools`/`list_catalog`/`load_tools`/`unload_tools`/`multiplexer_status`).

`agent_utilities.mcp.protocol_compat.check_mcp_sdk_floor()` closes that gap:

- Reads the `fastmcp` floor from `agent-utilities`'s OWN installed metadata (the `[mcp]`
  extra's PEP 508 marker via `importlib.metadata.requires()`), never a separately-parsed
  `pyproject.toml` — accurate for both a dev checkout and a deployed wheel.
- Derives the `mcp` floor transitively from `fastmcp-slim`'s own installed metadata
  (fastmcp's real runtime dependency) instead of a second, hand-maintained constraint
  that could drift from what fastmcp itself requires.
- Wired into `agent-utilities doctor` as the `mcp_sdk_floor` check (fails with a concrete
  remediation — reinstall/re-lock the `[mcp]` extra — instead of an opaque import crash
  at serve time) and covered by a CI regression test
  (`tests/unit/mcp/test_protocol_compat_sdk_floor.py`) that fails the moment this repo's
  own lock resolves an `mcp`/`fastmcp` pair that no longer satisfies the declared floor.

## The dependency-graph fix: `[tool.uv] override-dependencies`

`fastmcp==4.0.0b1` pins an exact `fastmcp-slim[client,server]==4.0.0b1`. Overriding
`fastmcp-slim` to `fastmcp-slim[client,server]>=4.0.0b1` relaxes pydantic-ai-slim's `<4`
cap project-wide. **Both `client` and `server` extras must be listed** — uv's override
replaces the entire requirement (extras included) for every requester of `fastmcp-slim`,
including `fastmcp`'s own `server`-extra pin that `agent_utilities/mcp/server_factory.py`
needs (`from fastmcp import FastMCP/Context`). An earlier draft of this override listing
only `[client]` silently broke every server-side fastmcp import project-wide; caught by
installing the full `agent-runtime` extra into a real venv before landing.

## Removal condition

Delete `protocol_compat.py`'s call sites and the `override-dependencies` entry together,
the moment `pydantic-ai-slim` ships a release whose `MCPToolset` natively handles the
fastmcp-4 `server/discover` era and whose `mcp.py` reads the SDK v2 field/exception names
directly. `install_mcp_v2_bridge()`'s per-field guard (skip if the attribute already
exists) makes it a no-op automatically the moment fastmcp/pydantic-ai catch up on their
own, so the removal is a cleanup, not a functional change.

## SemVer implication

Retiring the fastmcp-3 default (`[mcp]` now floors on `fastmcp>=4.0.0b1` instead of
`>=3.4.4`) and removing the `mcp-v4` extra name entirely is a **MAJOR** version bump for
`agent-utilities` under SemVer — extras removal and a default-dependency-major bump are
both breaking changes for any consumer pinned to the old floor. au's version stays frozen
per the standing operator directive; the bump is a decision for whoever unfreezes it (see
`reports/deferred/lane-2.1b.md` D-2.1b-2 and the unified program's D-08).
