# Building MCP Servers & API Wrappers

This guide explains how to build MCP (Model Context Protocol) servers and API wrappers using `agent-utilities`. All agents in `agent-packages/agents/*` follow this pattern, giving them a consistent interface and access to powerful context helpers.

## Overview

Every agent in the ecosystem ships up to three server types, all powered by `agent-utilities`:

| Server Type | Entry Point | Purpose |
|---|---|---|
| **Agent Server** | `agent_server.py` | Full graph-orchestrated AI agent with ACP/A2A/AG-UI endpoints |
| **MCP Server** | `mcp_server.py` | Standalone MCP tool server (stdio, SSE, or HTTP transport) |
| **API Wrapper** | `api_wrapper.py` | Python SDK wrapping a third-party REST API (used internally by the MCP server) |

The standard flow is: **API Wrapper** → **MCP Server** → **Agent Server**

---

## Building an MCP Server

### Step 1: Create the Server

Use `create_mcp_server()` from `agent_utilities.mcp_utilities` to bootstrap a fully configured FastMCP server with authentication, middleware, and CLI parsing:

```python
#!/usr/bin/env python
import logging
import os
import warnings

warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="fastmcp")

from agent_utilities.mcp_utilities import create_mcp_server
from fastmcp import Context
from pydantic import Field

__version__ = "1.0.0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bootstrap: creates parser, configures auth, builds middleware stack
args, mcp, middlewares = create_mcp_server(
    name="My Service MCP",
    version=__version__,
    instructions="Tools for managing My Service resources.",
)
```

`create_mcp_server()` gives you:
- **`args`**: Parsed CLI arguments (transport, host, port, auth config, etc.)
- **`mcp`**: A configured `FastMCP` instance with auth provider attached
- **`middlewares`**: Standard middleware stack (error handling, rate limiting, timing, logging)

### Step 2: Register Tools

Use the `@mcp.tool()` decorator with proper annotations and tags:

```python
@mcp.tool(
    annotations={
        "title": "List Resources",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
    tags={"resource_management"},
)
async def list_resources(
    filter_type: str = Field(description="Filter by resource type.", default=""),
    ctx: Context = Field(description="MCP context for progress reporting.", default=None),
) -> dict:
    """List all available resources. Expected return object type: dict"""
    # Use context helpers (see below)
    ctx_log(ctx, logger, "info", f"Listing resources with filter: {filter_type}")
    await ctx_progress(ctx, 0, 100)

    results = my_api.list_resources(filter_type=filter_type)

    await ctx_progress(ctx, 100, 100)
    return {"status": "success", "resources": results}
```

**Tool annotation best practices:**
- `readOnlyHint: True` for GET-like operations
- `destructiveHint: True` for DELETE/destructive operations
- `idempotentHint: True` for operations safe to retry
- Always include `tags` for tool categorization

### Step 3: Start the Server

```python
def mcp_server():
    """Entry point for the MCP server."""
    for mw in middlewares:
        mcp.add_middleware(mw)

    mcp.run(
        transport=args.transport,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    mcp_server()
```

---

## Context Helpers (`ctx_*`)

The `agent_utilities.mcp_utilities` module provides standardized context helpers that make your MCP tools consistent across the ecosystem. **All helpers are safe when `ctx` is `None`** -- they become no-ops in headless/test mode.

```python
from agent_utilities.mcp_utilities import (
    ctx_progress,
    ctx_confirm_destructive,
    ctx_log,
    ctx_set_state,
    ctx_get_state,
    ctx_sample,
)
```

### `ctx_progress(ctx, progress, total=100)`
Report progress to the MCP client:
```python
await ctx_progress(ctx, 0, 100)    # Starting
await ctx_progress(ctx, 50, 100)   # Halfway
await ctx_progress(ctx, 100, 100)  # Done
```

### `ctx_confirm_destructive(ctx, action_description)`
Standard elicitation guard for destructive operations. Asks the user to confirm before proceeding:
```python
if not await ctx_confirm_destructive(ctx, "delete all records"):
    return {"status": "cancelled", "message": "Operation cancelled by user"}
```
- Returns `True` if confirmed or no context available (headless mode)
- Returns `False` if the user cancels

### `ctx_log(ctx, logger, level, message)`
Dual-log to **both** the server-side logger and the MCP client:
```python
ctx_log(ctx, logger, "info", "Processing started")
ctx_log(ctx, logger, "error", f"Failed: {error}")
ctx_log(ctx, logger, "debug", f"Detail: {data}")
```
This ensures diagnostic output is visible in:
- Server process logs (for operators / container logs)
- MCP client log console (for the AI agent / human user)

### `ctx_set_state(ctx, project, key, value)` / `ctx_get_state(ctx, project, key, default=None)`
Store and retrieve session state with namespaced keys to prevent collisions:
```python
await ctx_set_state(ctx, "myservice", "auth_token", token)
token = await ctx_get_state(ctx, "myservice", "auth_token")
```

### `ctx_sample(ctx, prompt, system_prompt=None)`
Ask the client LLM to generate a response (sampling). Only works when the connected MCP client supports sampling:
```python
summary = await ctx_sample(ctx, f"Summarize this data: {data}")
```

---

## Building an API Wrapper

For agents that wrap a REST API, create a clean Python SDK class:

```python
import os
import requests
from agent_utilities.base_utilities import to_boolean

class MyServiceAPI:
    """Python wrapper for the My Service REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_token: str | None = None,
        ssl_verify: bool | None = None,
    ):
        self.base_url = (
            base_url or os.environ.get("SERVICE_URL", "http://localhost:8080")
        ).rstrip("/")
        self.api_token = api_token or os.environ.get("SERVICE_TOKEN", "")
        self.ssl_verify = (
            ssl_verify
            if ssl_verify is not None
            else to_boolean(os.environ.get("SSL_VERIFY", "True"))
        )
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        })
        self.session.verify = self.ssl_verify

    def list_resources(self, filter_type: str = "") -> list[dict]:
        """List all resources."""
        params = {"type": filter_type} if filter_type else {}
        response = self.session.get(f"{self.base_url}/api/resources", params=params)
        response.raise_for_status()
        return response.json()

    def create_resource(self, name: str, data: dict) -> dict:
        """Create a new resource."""
        response = self.session.post(
            f"{self.base_url}/api/resources",
            json={"name": name, **data},
        )
        response.raise_for_status()
        return response.json()
```

**Utilities from `agent_utilities.base_utilities`:**
- `to_boolean(value)` -- Safely convert string env vars to bool (`"True"`, `"1"`, `"yes"` → `True`)
- `to_integer(value)` -- Safely convert string env vars to int
- `expand_env_vars(text)` -- Expand `${VAR}` placeholders in strings

---

## Authentication Options

`create_mcp_server()` supports multiple auth modes via CLI flags:

| Auth Type | Flag | Use Case |
|---|---|---|
| `none` | `--auth-type none` | No authentication (default, local dev) |
| `static` | `--auth-type static` | Hardcoded test tokens |
| `jwt` | `--auth-type jwt --token-jwks-uri ... --token-issuer ... --token-audience ...` | JWT verification via JWKS |
| `oauth-proxy` | `--auth-type oauth-proxy` | OAuth 2.0 proxy (upstream IdP) |
| `oidc-proxy` | `--auth-type oidc-proxy` | OIDC proxy with token delegation |
| `remote-oauth` | `--auth-type remote-oauth` | Remote OAuth with authorization servers |

### Eunomia Policy Enforcement
Add authorization policies to your MCP server:
```bash
my-service-mcp --eunomia-type embedded --eunomia-policy-file mcp_policies.json
```

---

## OpenAPI Import

MCP servers can automatically import tools from an OpenAPI specification:

```bash
my-service-mcp --openapi-file openapi.json --openapi-base-url http://myservice:8080
```

This generates MCP tools for each endpoint in the OpenAPI spec, with proper type annotations and documentation.

---

## Complete Example: `pyproject.toml`

```toml
[project]
name = "my-agent"
version = "1.0.0"
requires-python = ">=3.11,<3.14"
dependencies = [
    "agent-utilities[agent]>=0.2.40",
    "requests>=2.32.0",
]

[project.scripts]
my-agent = "my_agent.agent_server:agent_server"
my-agent-mcp = "my_agent.mcp_server:mcp_server"
```

## Running

```bash
# MCP Server (stdio - for agent consumption)
uv run my-agent-mcp -t stdio

# MCP Server (HTTP - for remote access)
uv run my-agent-mcp -t streamable-http --host 0.0.0.0 --port 8001

# MCP Server (with JWT auth)
uv run my-agent-mcp -t streamable-http --auth-type jwt \
  --token-jwks-uri https://auth.example.com/.well-known/jwks.json \
  --token-issuer https://auth.example.com \
  --token-audience my-service

# Full Agent Server (graph orchestration + web UI)
uv run my-agent --web --port 8080
```

## Next Steps

- See [Creating an Agent](creating-an-agent.md) for the full agent server setup
- See [Architecture](architecture.md) for how MCP tools integrate with the graph orchestration pipeline
- See [Features](features.md) for tool guard behavior and elicitation patterns
