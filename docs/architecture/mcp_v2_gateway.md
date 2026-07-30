# GraphOS MCP v2 compatibility gateway

`mcp_v2_gateway` is a separately packaged Streamable HTTP sidecar for the
2026-07-28 MCP protocol. It exists because GraphOS's in-process FastMCP 3.4.x
environment pins `mcp<2`; the sidecar's own environment installs
`mcp>=2,<3` and cannot share that dependency resolution.

```mermaid
flowchart LR
  C[2026 MCP client] -->|per-request _meta + bearer| V2[MCP v2 gateway]
  V2 -->|same bearer| G[GraphOS FastMCP]
  G --> W[Native WorkItem]
  W --> G --> V2 --> C
```

The sidecar is stateless. `server/discover` and `tools/list` make a fresh,
bearer-forwarded GraphOS call, so the returned catalog retains GraphOS's auth,
tenant, consent, policy, and dynamic tool-visibility decisions. It records no
authorization state and never logs bearer values, endpoint details, tool
arguments, or downstream exception text.

## Tasks extension

The gateway advertises `io.modelcontextprotocol/tasks` only to clients that use
the 2026-07-28 request envelope. A task is created only for the governed
`graph_jobs(action=dispatch)` tool call and only if that *same request* declares
the Tasks extension. GraphOS dispatch durably creates its existing WorkItem;
the gateway immediately reads it through `graph_jobs(action=status)` before
returning `resultType: "task"`. The task id is the existing GraphOS job id, so
there is no task table or second store.

`tasks/get` projects the GraphOS WorkItem status, `tasks/cancel` invokes the
same GraphOS cooperative cancellation action, and `tasks/update` first performs
the same tenant-scoped status read then returns the required empty ack. WorkItems
currently expose no multi-round-trip input requests, so update safely ignores
empty or unknown input responses. Every lifecycle request must declare the
Tasks capability again and carries its bearer to GraphOS.

## SDK seam

The sidecar pins the official MCP Python SDK 2.x, but uses the documented
JSON-RPC dispatcher for this boundary. SDK 2.0's high-level `MCPServer` has a
static `ToolManager` and extension hooks for additive methods; it does not
provide a public override that proxies GraphOS's per-request, authorization
filtered dynamic `tools/list` catalog. Using its static registration would
either cache a catalog across callers or reimplement GraphOS policy, both of
which are unacceptable. The raw dispatcher is therefore intentional and tested
as JSON-RPC conformance behavior; migration to a future official dynamic-proxy
hook can replace only this transport adapter.

Run the focused checks from the repository root:

```bash
PYTHONPATH=. pytest -q tests/mcp_v2_gateway/test_protocol.py
ruff check mcp_v2_gateway tests/mcp_v2_gateway
```
