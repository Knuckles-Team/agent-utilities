# Design Document: Default to `fastmcp>=4.0.0b1` and bridge the two real upstream incompatibilities in our own client-construction path, instead of pinning back to fastmcp 3.x

CONCEPT:AU-ECO.mcp.protocol-compat-bridge

> `pyproject.toml:161-183` (`[project.optional-dependencies].mcp`, `[tool.uv]
> override-dependencies`) and `agent_utilities/mcp/protocol_compat.py:1-30`
> (`install_mcp_v2_bridge`, the `mode="legacy"` bridge).

## Decision — target `fastmcp>=4.0.0b1` / `mcp>=2.0.0` by default and close the two real gaps between it and `pydantic-ai-slim[mcp]` 2.21.0 ourselves, at MCP-client construction time, rather than staying on fastmcp 3.x until upstream catches up

Empirical, live verification (a real `pydantic_ai.mcp.MCPToolset` driven
end-to-end — connect, list tools, call a tool — against a real fastmcp-4 server,
`fastmcp 4.0.0b1` / `mcp 2.0.0` / `pydantic-ai-slim 2.21.0`) found exactly two
upstream gaps, both inside `pydantic_ai.mcp`/`fastmcp`'s own code, not anything this
package calls directly: (1) four camelCase→snake_case protocol-field renames in the
MCP SDK v2 line that `fastmcp`'s own compat shim does not cover
(`PromptsCapability.listChanged` etc., plus the `McpError`→`MCPError` rename), and
(2) `fastmcp.client.Client` defaulting to `mode="auto"`, which leaves
`Client.initialize_result` as `None` and trips `MCPToolset.__aenter__`'s
unconditional assertion. `protocol_compat.py`'s `install_mcp_v2_bridge()` closes
gap (1) with plain properties reading the renamed attributes (the same technique
`fastmcp._compat` itself uses), and construction pins `Client.mode="legacy"` to
close gap (2). Both are applied at every toolset-construction call site in the
package. `pyproject.toml` additionally relaxes `pydantic-ai-slim[mcp]`'s
`fastmcp-slim[client]<4` cap project-wide via `[tool.uv] override-dependencies`,
justified as "conservative rather than a known incompatibility" — the cap predates
fastmcp 4's release and was never validated against it.

## Rejected alternative — pin `fastmcp` back to the 3.x line until `pydantic-ai-slim` ships native fastmcp-4 support

The straightforward, lower-risk-looking alternative is explicit in the source
comment: "Both are bridged by this package at MCP-client construction time, **not
by pinning fastmcp back to 3.x**" (`pyproject.toml:167`). Staying on 3.x avoids
needing any bridge at all — but it means the whole fleet (every MCP server and
every MCP client in the workspace) stays off the current MCP SDK v2 line
indefinitely, waiting on a `pydantic-ai-slim` release with no committed timeline,
and misses whatever fastmcp 4 brings (this repo's own commitment to standardizing
on the pydantic-ai stack makes staying current with its dependency chain a
standing priority, not a one-off choice). Bridging two small, well-understood,
upstream-code-only gaps — verified empirically rather than assumed — was judged
cheaper and more forward-compatible than freezing the whole fleet's MCP stack on
an older major version while waiting for someone else's release.

## Risk Assessment

- **Blast Radius**: `pyproject.toml` (`[mcp]` extra, `[tool.uv]
  override-dependencies`), `agent_utilities/mcp/protocol_compat.py`, every call
  site constructing a `pydantic_ai.mcp.MCPToolset`.
- **Backward Compatible**: Yes — the bridge is purely additive compatibility
  shimming; no behavioural change to tool calls themselves.
- **Known weak point**: the bridge targets `pydantic-ai-slim 2.21.0` specifically
  as "the latest published release as of this writing" — a future
  `pydantic-ai-slim` release that ships native fastmcp-4 support (or changes its
  own internals) needs this bridge re-verified and, per the source comment, "drop
  this [override] the moment pydantic-ai-slim ships a release that natively
  supports fastmcp 4" — nothing currently detects that moment automatically.
