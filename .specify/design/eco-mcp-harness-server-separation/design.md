# Design Document: The agentic-harness surface (DSTDD, self-model, evaluation, red-team, provenance) ships as a SEPARATE companion MCP server, not bundled into `kg_server.py`

CONCEPT:AU-ECO.mcp.harness-exposure

> `agent_utilities/mcp/harness_server.py:1-20` (module docstring + `_build_server`).

## Decision — `harness_server.py` is its own MCP server process/entrypoint (`agent-utilities-harness`, run via `python -m agent_utilities.mcp.harness_server`), a companion to `kg_server.py`, rather than tools registered onto the KG server

`kg_server.py` exposes graph operations (query/search/ingest) — the platform's core,
highest-traffic surface. `harness_server.py` exposes a functionally distinct
concern: the DSTDD design-governance pipeline (`dstdd_create_design`,
`dstdd_validate_design`, `dstdd_design_to_spec`), task-management ergonomics
(`task_parse_prd`, `task_next`, `task_scope`, …), agent self-model/team-composition
queries, evaluation history, and reliability/red-team/provenance gating tools. The
module docstring states the split explicitly: *"Companion server to `kg_server.py`.
While the KG server exposes graph operations, this server exposes the agentic
harness."* Both are built through the same `create_mcp_server()` factory
(`server_factory.py`) — zero new server-bootstrap machinery — but they are two
separate FastMCP server instances with two separate tool surfaces and, when run over
network transport, two separate ports (the KG server's own docstring shows
`--port 8101` for the harness server as a *different* port than the KG server's
default).

## Rejected alternative — register the harness/DSTDD/evaluation tools directly onto `kg_server.py`

Every harness tool in this module *could* be registered onto the same
`create_mcp_server()` call `kg_server.py` uses — nothing about FastMCP requires a
second process. That was rejected because `kg_server.py` is the platform's
highest-traffic, most latency-sensitive surface (every `graph_*`/`engine_*` call,
from every connected client, funnels through its dispatch core — see
`AU-ECO.mcp.gateway-dispatch-isolation`), while the harness surface is a
lower-frequency, higher-latency governance/evaluation concern (running a red-team
probe suite, an evaluation-history query, a DSTDD validation) that an operator or IDE
often wants reachable independently of whether the primary KG server is even
running. Keeping them as separate servers means a harness-only client does not pull
in the KG server's full tool surface (and vice versa), and an operational issue in
one surface's process does not couple to the other's uptime — the same isolation
motivation `AU-ECO.mcp.gateway-dispatch-isolation` applies at the dispatch level,
applied here at the process/server level instead.

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/harness_server.py`,
  `agent_utilities/mcp/server_factory.py` (shared factory).
- **Backward Compatible**: Yes — additive server; does not change `kg_server.py`'s
  own tool surface.
- **Known weak point**: two servers means two things to keep running/configured in
  `mcp_config.json` for a client that wants both surfaces; there is no single
  combined endpoint for a caller that only wants one connection.
