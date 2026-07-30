#!/usr/bin/python
from __future__ import annotations

"""Client-side bridge for the fastmcp-4 / MCP SDK v2 upgrade.

CONCEPT:AU-ECO.mcp.protocol-compat-bridge

`agent-utilities` targets `fastmcp>=4.0.0b1` by default (see the `[mcp]` extra in
`pyproject.toml`), which transitively requires `mcp>=2.0.0,<3.0.0` (the MCP Python
SDK's v2 line). Empirically verified against a live fastmcp-4 server + a real
`pydantic_ai.mcp.MCPToolset` client (fastmcp 4.0.0b1, mcp 2.0.0, pydantic-ai-slim
2.21.0 — the latest published release as of this writing), two upstream gaps break
every real toolset connection unless bridged here. Both gaps are inside
`pydantic_ai.mcp` / `fastmcp`'s own code, not anything this package calls directly,
so they cannot be fixed by changing how *we* invoke the API — only by adapting to
the renamed/defaulted surface until upstream catches up:

1. **`mcp` SDK v2 renamed several protocol fields from camelCase to snake_case**
   (`inputSchema` -> `input_schema`, `mimeType` -> `mime_type`, etc). `fastmcp`
   ships its own deprecation bridge for this (`fastmcp._compat`, a curated table of
   warn-once camelCase properties) and it covers most of what `pydantic_ai.mcp`
   reads — but NOT `PromptsCapability.listChanged` / `ResourcesCapability.listChanged`
   / `ToolsCapability.listChanged` (read unconditionally by
   `ServerCapabilities.from_mcp_sdk` on every `MCPToolset.__aenter__`) or
   `ToolExecution.taskSupport` (read by `MCPToolset.get_tools()` whenever a tool
   advertises `execution` metadata). `mcp.shared.exceptions.McpError` was also
   renamed to `MCPError` — `pydantic_ai.mcp`'s tool-call error handling
   (`except mcp_exceptions.McpError`) still expects the old name. `install_mcp_v2_bridge()`
   closes exactly these four gaps, using the same technique fastmcp uses for the
   rest (a plain property reading the renamed attribute), guarded so it never
   shadows a real upstream fix.
2. **`fastmcp.client.Client` defaults to `mode="auto"`**, which negotiates the
   modern `server/discover` connect era against a fastmcp-4 server and leaves
   `Client.initialize_result` as `None`. `pydantic_ai.mcp.MCPToolset.__aenter__`
   (2.21.0) unconditionally asserts `client.initialize_result is not None`, so
   every connection to a real fastmcp-4 server fails outright unless the
   underlying client is pinned to `mode="legacy"` (today's initialize handshake,
   which populates `initialize_result`). `MCPToolset` does not expose a `mode`
   passthrough for its convenience constructors (bare transport / URL / in-process
   `FastMCP` server / `pydantic_ai.mcp.load_mcp_toolsets`), but `Client.mode` is a
   plain, un-validated instance attribute read lazily at connect time — so
   `force_legacy_protocol_mode()` reaches into an already-constructed
   `MCPToolset.client` (unwrapping `WrapperToolset.wrapped`, e.g.
   `PrefixedToolset` from `load_mcp_toolsets`) and pins it before first use.

Both are temporary, forward-compatible shims: `install_mcp_v2_bridge()` skips any
field pydantic-ai/fastmcp already provide (so a future release that adds proper
support makes this a no-op), and `force_legacy_protocol_mode()` is a one-line
attribute set with no other side effects. Delete this module once
`pydantic-ai-slim` ships a release whose `MCPToolset` handles the fastmcp-4
`server/discover` era natively and whose `mcp.py` reads the SDK v2 field/exception
names directly.
"""

import warnings
from typing import Any

_installed = False


def install_mcp_v2_bridge() -> None:
    """Bridge the MCP SDK v2 attribute renames that `fastmcp._compat` doesn't cover.

    Idempotent. Safe to call from any module that constructs an `MCPToolset` before
    doing so; `agent_utilities.mcp.toolset_factory` calls it at import time so every
    call site in this package gets it for free.
    """
    global _installed
    if _installed:
        return

    from mcp import types as mcp_types
    from mcp.shared import exceptions as mcp_exceptions

    # `mcp.shared.exceptions.McpError` was renamed `MCPError` in SDK v2.
    # `pydantic_ai.mcp` still catches the old name in its tool-call error handling.
    if not hasattr(mcp_exceptions, "McpError") and hasattr(mcp_exceptions, "MCPError"):
        # Assigning through `__dict__` (rather than a plain attribute assignment)
        # keeps this a runtime-only alias that mypy doesn't try to statically
        # unify with the `MCPError` class identity.
        mcp_exceptions.__dict__["McpError"] = mcp_exceptions.MCPError

    # Fields `fastmcp._compat`'s own camelCase bridge table doesn't include, but
    # `pydantic_ai.mcp` still reads unconditionally.
    aliases: dict[type, dict[str, str]] = {
        mcp_types.PromptsCapability: {"listChanged": "list_changed"},
        mcp_types.ResourcesCapability: {"listChanged": "list_changed"},
        mcp_types.ToolsCapability: {"listChanged": "list_changed"},
        mcp_types.ToolExecution: {"taskSupport": "task_support"},
    }
    for cls, mapping in aliases.items():
        model_fields = getattr(cls, "model_fields", {})
        for camel, snake in mapping.items():
            # Never shadow a real attribute — a future SDK/fastmcp release that
            # restores or re-covers the field makes this bridge a no-op.
            if camel in cls.__dict__ or camel in model_fields:
                continue
            setattr(cls, camel, _make_property(cls.__name__, camel, snake))

    _installed = True


def _make_property(cls_name: str, camel: str, snake: str) -> property:
    def getter(self: object) -> object:
        warnings.warn(
            f"Accessing `{cls_name}.{camel}` is deprecated; MCP SDK v2 renamed this "
            f"field to `{snake}`. Update your code to read `.{snake}` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, snake)

    return property(getter)


def force_legacy_protocol_mode(toolset: Any) -> Any:
    """Pin an `MCPToolset` (or a list of toolsets) to `mode="legacy"` before first use.

    Unwraps `WrapperToolset.wrapped` (e.g. `PrefixedToolset`, which
    `pydantic_ai.mcp.load_mcp_toolsets` returns) to find the underlying
    `MCPToolset.client`. Toolsets that aren't MCP-backed (no `.client.mode`) are
    left untouched. Returns `toolset` unchanged for chaining.

    Unwrapping checks `isinstance(target, WrapperToolset)` rather than
    `hasattr(target, "wrapped")`: a bare `unittest.mock.MagicMock` (used
    throughout this package's own test suite to stand in for a toolset)
    answers `hasattr(..., "wrapped")` as `True` for EVERY attribute name and
    hands back a distinct child mock on every access, so a duck-typed unwrap
    loop never terminates against one. `isinstance` only matches the real
    toolset wrapper class.
    """
    if isinstance(toolset, (list, tuple)):
        for item in toolset:
            force_legacy_protocol_mode(item)
        return toolset

    from pydantic_ai.toolsets import WrapperToolset

    target = toolset
    while isinstance(target, WrapperToolset):
        target = target.wrapped

    client = getattr(target, "client", None)
    if client is not None and hasattr(client, "mode"):
        client.mode = "legacy"

    return toolset
