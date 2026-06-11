# Worked Example: Consuming graph-os over MCP (stdio, HTTP, and the multiplexer)

**What this demonstrates.** The three ways an agent IDE or framework consumes
this platform's MCP surface: `graph-os` as a local stdio child, `graph-os` as a
remote streamable-http child, and many children aggregated behind the
`mcp-multiplexer` — including the per-server resilience overrides
(CONCEPT:ECO-4.34: `max_concurrency`, `pool_size`, `call_timeout`,
`queue_timeout`, breaker overrides) and the `multiplexer_status` health tool.
Related: [Consumption models](../guides/consumption-models.md),
[Building MCP servers](../guides/building-mcp-servers.md).

**Prerequisites (ladder rung).** Zero-infra rung of
[Deployment configurations](../guides/deployment-configurations.md): a Python
environment with `agent-utilities` installed gives you the `graph-os` and
`mcp-multiplexer` console scripts (`pyproject.toml` `[project.scripts]`).

---

## 1. Where the config lives

The multiplexer resolves its config from, in order: `--config <path>`, the
`MCP_CONFIG` env var, then the discovery candidates
`~/.gemini/antigravity/mcp_config.json`,
`~/.config/agent-utilities/mcp_config.json`,
`~/.config/agent-utilities/config.json`, `./mcp_config.json`,
`./workspace/mcp_config.json` (see `_resolve_config_path` in
`agent_utilities/mcp/multiplexer.py`). `${VAR}` references anywhere in the file
are expanded from the environment. A starter file ships as
[`mcp_config.example.json`](https://github.com/Knuckles-Team/agent-utilities/blob/main/mcp_config.example.json).

## 2. graph-os as a direct child — stdio and streamable-http

Both transports come from the same standard server factory
(`agent_utilities/mcp/server_factory.py`: `--transport stdio|streamable-http|sse`,
`--host`, `--port`).

```jsonc
{
  "mcpServers": {
    // Local stdio subprocess — the IDE owns the process lifecycle.
    "graph-os": {
      "command": "uv",
      "args": ["run", "graph-os"],
      "env": {
        "AGENT_ID": "local-developer",
        "WORKSPACE_PATH": "${workspaceFolder}"
      }
    },

    // Remote HTTP — connect to a graph-os started elsewhere with:
    //   graph-os --transport streamable-http --host 0.0.0.0 --port 8765
    "graph-os-remote": {
      "transport": "streamable-http",
      "url": "http://kg-host.arpa:8765/mcp",
      "headers": {"Authorization": "Bearer ${GRAPH_OS_TOKEN}"}
    }
  }
}
```

Transport selection rules (verified in `MCPMultiplexer._start_child`): a child
is remote when it declares a `url` **or** a `transport` of `streamable-http` /
`streamable_http` / `http` / `sse`; otherwise it is a local stdio subprocess
run via `command` + `args` + `env`. SSE is used when `transport` is `sse` or
the URL path ends in `/sse`; any other remote child speaks streamable-http.
`env` values and `headers` values are `${VAR}`-expanded.

## 3. The multiplexer with per-server resilience overrides (ECO-4.34)

A complete, annotated `mcp_config.json` aggregating three children:

```jsonc
{
  "mcpServers": {
    // 1. The KG itself, stdio. Tools surface with the "kg" prefix
    //    (SERVER_NICKNAMES maps "graph-os" -> "kg"), e.g. kg__graph_query.
    "graph-os": {
      "command": "uv",
      "args": ["run", "graph-os"],
      "env": {"AGENT_ID": "multiplexer"},

      // ── ECO-4.34 per-server overrides (each falls back to the global
      //    MCP_CHILD_* env default when omitted) ────────────────────────
      "max_concurrency": 4,        // in-flight call cap   (default: MCP_CHILD_MAX_CONCURRENCY=8; 0 = unlimited)
      "queue_timeout": 10,         // seconds a call may wait for a free slot
                                   //                      (default: MCP_CHILD_QUEUE_TIMEOUT=30)
      "call_timeout": 120,         // per-call answer ceiling in seconds; falls back to
                                   // this entry's "timeout" key, then 300; <=0 disables
      "breaker_threshold": 3,      // consecutive transport failures before the
                                   // circuit opens         (default: MCP_CHILD_BREAKER_THRESHOLD=5; 0 disables)
      "breaker_cooldown": 20,      // seconds open before one half-open probe
                                   //                      (default: MCP_CHILD_BREAKER_COOLDOWN=15)
      "max_restarts": 5,           // crash-restarts allowed inside the window
                                   //                      (default: MCP_CHILD_MAX_RESTARTS=5; 0 disables auto-restart)
      "restart_window": 300        // sliding window, seconds (default: MCP_CHILD_RESTART_WINDOW=300)
    },

    // 2. A remote HTTP child with a session pool: pool_size opens N
    //    round-robin connections for parallel in-flight calls (remote
    //    children only — stdio is single-pipe and always 1 session).
    "egeria-mcp": {
      "transport": "streamable-http",
      "url": "http://egeria-mcp.arpa/mcp",
      "headers": {"Authorization": "Bearer ${EGERIA_MCP_TOKEN}"},
      "pool_size": 3,              // default: MCP_CHILD_POOL_SIZE=1
      "max_concurrency": 6,
      "timeout": 300               // startup + default call ceiling (seconds)
    },

    // 3. A stdio child with tool filtering (fnmatch patterns).
    "repository-manager-mcp": {
      "command": "uvx",
      "args": ["--from", "repository-manager", "repository-manager-mcp"],
      "enabledTools": ["rm_projects*", "rm_git*"],   // whitelist (omit = all)
      "disabledTools": ["rm_git_push*"]              // blacklist on top
    },

    // "disabled": true skips a child entirely; an entry literally named
    // "mcp-multiplexer" is always skipped (self-recursion guard).
    "noisy-experimental-mcp": {
      "command": "uv",
      "args": ["run", "experimental-mcp"],
      "disabled": true
    }
  }
}
```

Run it (stdio toward the IDE, or itself over HTTP):

```bash
mcp-multiplexer --config /path/to/mcp_config.json
mcp-multiplexer --config /path/to/mcp_config.json --transport streamable-http --port 9100
```

Global fallback summary (typed on `AgentConfig`,
`agent_utilities/core/config.py`): `MCP_CHILD_MAX_CONCURRENCY`,
`MCP_CHILD_QUEUE_TIMEOUT`, `MCP_CHILD_POOL_SIZE`, `MCP_CHILD_MAX_RESTARTS`,
`MCP_CHILD_RESTART_WINDOW`, `MCP_CHILD_BREAKER_THRESHOLD`,
`MCP_CHILD_BREAKER_COOLDOWN`. `call_timeout` is the one override with **no**
global env var — its fallback chain is per-server `call_timeout` → per-server
`timeout` → 300 seconds.

When a limit trips, callers get a typed error in the tool result instead of a
hang: `MCPChildBusyError` (slots full past `queue_timeout`),
`MCPChildCallTimeoutError` (child accepted but never answered — the slot is
held until the child truly finishes, so a wedged child applies backpressure),
`MCPChildUnavailableError` (restarting/failed), `MCPChildCircuitOpenError`
(breaker open, failing fast).

## 4. The `multiplexer_status` tool

The multiplexer registers one extra tool of its own. Call:

```json
{"tool": "multiplexer_status", "arguments": {}}
```

**Expected output** (`MCPMultiplexer.status_snapshot()` — one entry per child
from `ChildRuntime.status()`):

```json
{
  "children": {
    "egeria-mcp": {
      "server": "egeria-mcp",
      "state": "up",
      "restart_count": 0,
      "breaker": "closed",
      "sessions": 3,
      "max_concurrency": 6,
      "in_flight": 1,
      "queued": 0
    },
    "graph-os": {
      "server": "graph-os",
      "state": "up",
      "restart_count": 1,
      "breaker": "closed",
      "sessions": 1,
      "max_concurrency": 4,
      "in_flight": 0,
      "queued": 0
    },
    "repository-manager-mcp": {
      "server": "repository-manager-mcp",
      "state": "failed",
      "restart_count": 6,
      "breaker": "open",
      "sessions": 0,
      "max_concurrency": 8,
      "in_flight": 0,
      "queued": 0
    }
  },
  "total_children": 3,
  "total_tools": 87
}
```

`state` is `starting` / `up` / `restarting` / `failed`; `breaker` is `closed` /
`half_open` / `open`. The same health lands on the OS-5.23 metrics registry as
`agent_utilities_mcp_child_calls_total{server,outcome}`,
`agent_utilities_mcp_child_breaker_state{server}`,
`agent_utilities_mcp_child_restarts_total{server}` and
`agent_utilities_mcp_child_queue_depth{server}` (no-ops without the `metrics`
extra).

Tool namespacing: every child tool is re-published under a short, host-aware
prefix (`get_server_prefix`) — well-known servers have fixed nicknames
(`graph-os` → `kg`, `repository-manager-mcp` → `rep`, `tunnel-manager-mcp` →
`tun`, ...; unknown names fall back to their first 5 cleaned characters), and
redundant server-name prefixes are stripped from the tool name itself
(`clean_tool_name`).

---

*Verification: smoke-run against this tree (2026-06-11) for the resilience
layer: `python3 -m pytest tests/test_multiplexer_resilience.py -q` exercises
the per-server override keys, typed errors, restart parking and breaker
behavior. The config-key semantics, transport rules, status shapes, prefix
table and metric names above were verified by reading
`agent_utilities/mcp/multiplexer.py`, `agent_utilities/mcp/child_resilience.py`,
`agent_utilities/mcp/server_factory.py` and `agent_utilities/core/config.py` in
this tree; the `multiplexer_status` values shown are illustrative (no live
multi-child fleet was started), with the field set matching
`ChildRuntime.status()` exactly.*
