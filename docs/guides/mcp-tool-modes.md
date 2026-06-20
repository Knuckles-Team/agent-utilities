# MCP Tool Modes — condensed, verbose, or both

Every fleet MCP server (`agents/*`) can expose its operations as one of two tool
surfaces, selected by a single knob. This guide explains the modes, how the
**verbose 1:1** surface is generated, how API specs feed its **typed** tier, and
how it ties into the one shared XDG config.

## The two surfaces

| Surface | Shape | When to use |
|---|---|---|
| **condensed** (default) | One action-routed tool per domain — `servicenow_cmdb(action, params_json)` — dispatching to many client methods. Small tool count. | Default. Keeps large APIs well under the ~100-tool limit; lowest token/context cost. |
| **verbose** | One named, documented tool **per API-client method** — `servicenow_get_cmdb_instance(...)`. The model selects the exact operation directly. | When you want maximal call accuracy for a specific connector and can afford the larger surface. |

`both` registers both sets at once.

## The knob: `MCP_TOOL_MODE`

```jsonc
// ~/.config/agent-utilities/config.json
{ "mcp_tool_mode": "verbose" }   // condensed | verbose | both  (default: condensed)
```

Read once via `agent_utilities.mcp_utilities.tool_mode()` (backed by
`config.setting`, so it is driven by the one XDG `config.json` — see
[Configuration](configuration.md)). Default `condensed` ⇒ existing deployments are
unchanged.

Every verbose tool is tagged **`verbose`** plus its domain (`cmdb`, `incidents`,
…), so the existing visibility filter still slices the set per request:

```bash
servicenow-mcp --tools tag:verbose          # only the 1:1 tools
# or via env / HTTP query — MCP_ENABLED_TAGS=verbose, ?tags=verbose
```

## How the verbose surface is generated

`register_verbose_tools(mcp, Api, get_client, service=..., manifest=...)`
(in `agent_utilities/mcp/verbose_tools.py`) introspects the **client class** — no
credentials needed at registration; the live client is bound per-call via
`Depends(get_client)`. Fidelity is decided **per method**:

- **Typed tier** — when a normalized *operation manifest* carries the method *with*
  parameters, a fully-typed signature is synthesized (one typed argument per
  parameter, with descriptions). This is the rich surface sourced from an API spec.
- **Introspection fallback** — otherwise the method gets a single `params_json`
  argument (hand-written `**kwargs` clients carry no per-parameter metadata). It is
  still individually named and documented with the method's docstring.

Destructive operations (manifest `http: DELETE`, an explicit `destructive` flag, or
a `delete_/remove_/destroy_/...` method name) gate on a `Context` elicitation
prompt (`ctx_confirm_destructive`) — confirmed interactively on served requests,
allowed by default headless.

## Feeding the typed tier from API specs (sources of truth)

The typed tier is only as good as the **operation manifest** behind it. Always try
to source a machine-readable spec, in this priority order:

1. **OpenAPI / Swagger JSON first** — the richest, most reliable source. Vendor it
   under `<pkg>/specs/` (Swagger 2.0 → convert to OpenAPI 3). This is the gold path
   (see `agents/onetrust-api`, generated from vendored specs).
2. **Crawled documentation site** — when no spec is published, crawl the API docs
   (`web-crawler` / `web-fetch`) and extract operations into OpenAPI.
3. **PDF API spec** — extract text (the KG's PDF pipeline) and structure it into
   OpenAPI.

All three normalize to **OpenAPI 3 → a manifest** (`<pkg>/api/_operation_manifest.py`,
a list of `{operation_id, domain, method, http, path, params, summary, ...}`). The
codegen then emits the client + condensed tools, and `register_verbose_tools` reads
the same manifest for the typed verbose tier. Acquisition + codegen live in the
`api-client-builder` / `agent-package-builder` skills, not in core.

## Wiring it into an agent

```python
from agent_utilities.mcp_utilities import (
    load_config, tool_mode, register_verbose_tools,
)
from my_agent.api_client import Api
from my_agent.auth import get_client

def get_mcp_instance():
    load_config()                      # one XDG config.json, not load_dotenv
    args, mcp, middlewares = create_mcp_server(...)
    mode = tool_mode()
    if mode in ("condensed", "both"):
        ...register the action-routed tools (unchanged)...
    if mode in ("verbose", "both"):
        register_verbose_tools(mcp, Api, get_client, service="my-agent",
                               manifest=OPERATIONS)   # omit if no spec
    ...
```

CONCEPT:ECO-4.82 — MCP tool-mode standardization.
