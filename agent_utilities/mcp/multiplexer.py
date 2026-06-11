#!/usr/bin/env python
"""Multi-MCP Server Multiplexer.

Aggregates multiple underlying MCP servers (declared in an ``mcp_config.json``)
into a single unified server, delegating tool calls dynamically based on
prefixed tool names. This speeds up boot times and avoids per-server process
resource contention for clients with tool-count limits.

Built on the standard ``mcp_server.py`` scaffolding: it uses
``create_mcp_server()`` for the standard ``--transport/--host/--port`` args and
middleware, and exposes the aggregated tools through a FastMCP instance so the
multiplexer can be deployed as either a **stdio** or **streamable-http** server.
The proven child-server lifecycle, host-aware prefixing, and enable/disable tool
filtering are preserved (see :class:`MCPMultiplexer`).

CONCEPT:ECO-4.0 — MCP Standardized Interfaces
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import mcp.types
from fastmcp.tools import FunctionTool, ToolResult
from mcp import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession

from agent_utilities.mcp.child_resilience import (
    ChildRuntime,
    MCPChildError,
)

try:  # remote transports — present on modern mcp SDKs
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:  # pragma: no cover - older mcp SDK without streamable-http
    streamablehttp_client = None
try:
    from mcp.client.sse import sse_client
except ImportError:  # pragma: no cover - older mcp SDK without sse
    sse_client = None

# Direct all logs to stderr so stdout remains perfectly clean for stdio JSON-RPC
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mcp_multiplexer")


SERVER_NICKNAMES = {
    "graph-os": "kg",
    "repository-manager-mcp": "rep",
    "tunnel-manager-mcp": "tun",
    "systems-manager-mcp": "sys",
    "container-manager-mcp": "cnt",
    "audio-transcriber-mcp": "aud",
    "archivebox-mcp": "arc",
    "data-science-mcp": "ds",
    "ansible-tower-mcp": "ans",
    "github-mcp": "gh",
    "gitlab-mcp": "gl",
    "home-assistant-mcp": "ha",
    "arr-mcp": "arr",
    "plane-mcp": "pl",
    "mealie-mcp": "mel",
    "postiz-mcp": "pz",
    "owncast-mcp": "oc",
    "jellyfin-mcp": "jf",
    "microsoft-mcp": "ms",
    "listmonk-mcp": "lm",
    "nextcloud-mcp": "nc",
    "atlassian-mcp": "atl",
    "servicenow-mcp": "sn",
    "qbittorrent-mcp": "qb",
    "searxng-mcp": "sx",
    "media-downloader-mcp": "md",
    "stirlingpdf-mcp": "sp",
    "wger-mcp": "wg",
    "uptime-kuma-mcp": "ut",
    "technitium-dns-mcp": "td",
    "caddy-mcp": "cd",
    "keycloak-mcp": "kc",
    "twenty-mcp": "tw",
    "erpnext-mcp": "erp",
    "openbao-mcp": "ob",
    "lgtm-mcp": "lg",
    "mattermost-mcp": "mm",
    "scholarx-mcp": "sx",
    "langfuse-mcp": "lf",
    "portainer-mcp": "pt",
    "legal-peripherals-mcp": "lgl",
    "egeria-mcp": "eg",
    "dr-egeria-mcp": "dre",
}


def get_server_prefix(server_name: str) -> str:
    """Get a clean, short prefix for the server name to respect character limit constraints."""
    if server_name.startswith("systems-manager-mcp-"):
        host = server_name[len("systems-manager-mcp-") :]
        return f"sys_{host}".replace("-", "_")
    if server_name.startswith("container-manager-mcp-"):
        host = server_name[len("container-manager-mcp-") :]
        return f"cnt_{host}".replace("-", "_")
    if server_name in SERVER_NICKNAMES:
        return SERVER_NICKNAMES[server_name]
    # Fallback: clean non-alphanumeric chars and limit to first 5 chars
    clean = "".join(c if (c.isalnum() or c in ("_", "-")) else "_" for c in server_name)
    clean = clean.replace("-", "_")
    return clean[:5].lower().strip("_")


def clean_tool_name(prefix: str, server_name: str, original_tool_name: str) -> str:
    """Removes redundant server/module name prefixes from the tool name and ensures strict length compliance."""
    if server_name.startswith("systems-manager-mcp-"):
        base_server = "systems-manager-mcp"
    elif server_name.startswith("container-manager-mcp-"):
        base_server = "container-manager-mcp"
    else:
        base_server = server_name

    clean_server = base_server.replace("-", "_").lower()
    cleaned = original_tool_name

    # Build potential redundant prefixes to strip from the tool name
    strips = [
        f"{clean_server}_mcp_",
        f"{clean_server}_",
        f"{prefix}_mcp_",
        f"{prefix}_",
    ]

    if base_server.endswith("-mcp"):
        mod_server = base_server[:-4].replace("-", "_").lower()
        strips.append(f"{mod_server}_mcp_")
        strips.append(f"{mod_server}_")

    for s in strips:
        if cleaned.startswith(s):
            cleaned = cleaned[len(s) :]
            break

    # Build the final namespaced candidate
    candidate = f"{prefix}__{cleaned}"

    # Target maximum budget: 44 characters (so client-prefixed name is <= 64 characters)
    if len(candidate) > 44:
        budget = 44 - len(prefix) - 2  # 2 for "__"
        candidate = f"{prefix}__{cleaned[:budget].strip('_')}"

    return candidate


class MCPMultiplexer:
    """Aggregates and proxies multiple MCP servers over a single stdio connection."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.exit_stack = contextlib.AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        # Per-child hardening layer (CONCEPT:ECO-4.34): concurrency limits and
        # bounded queueing live on the ChildRuntime, not the raw session.
        self.children: dict[str, ChildRuntime] = {}
        self.tool_to_server: dict[
            str, tuple[str, str]
        ] = {}  # prefixed_name -> (server_name, original_name)
        self.aggregated_tools: list[mcp.types.Tool] = []

    async def call_proxied_tool(
        self, prefixed_name: str, arguments: dict[str, Any] | None = None
    ) -> mcp.types.CallToolResult:
        """Forward a prefixed tool call to the owning child server's session.

        Looks up the ``(server_name, original_name)`` mapping recorded during
        :meth:`start_children` and forwards the call to that child's live
        ``ClientSession``. Raises if the tool/server is unknown or inactive.
        """
        logger.info(f"Calling tool: {prefixed_name}")
        if prefixed_name not in self.tool_to_server:
            raise ValueError(f"Tool {prefixed_name} is not registered in multiplexer")

        server_name, original_name = self.tool_to_server[prefixed_name]
        runtime = self.children.get(server_name)
        if runtime is None:
            raise RuntimeError(f"Session for server '{server_name}' is not active")

        try:
            # Forward the call through the child's hardened runtime
            # (per-server concurrency limit + bounded queue).
            return await runtime.call_tool(original_name, arguments or {})
        except MCPChildError as e:
            # Typed per-child failure (busy/restarting/failed/circuit-open):
            # surface the error class name so callers can branch on it.
            logger.warning(
                "Tool '%s' on '%s' rejected: %s: %s",
                original_name,
                server_name,
                type(e).__name__,
                e,
            )
            return mcp.types.CallToolResult(
                content=[
                    mcp.types.TextContent(
                        type="text", text=f"{type(e).__name__}: {e}"
                    )
                ],
                isError=True,
            )
        except Exception as e:
            logger.error(
                f"Error calling tool '{original_name}' on '{server_name}': {e}",
                exc_info=True,
            )
            return mcp.types.CallToolResult(
                content=[
                    mcp.types.TextContent(
                        type="text", text=f"Error executing tool: {e}"
                    )
                ],
                isError=True,
            )

    async def _start_child(
        self, server_name: str, cfg: dict
    ) -> tuple[str, ClientSession, list[mcp.types.Tool], dict] | None:
        """Starts a single child server, registers its exit stack on success, and returns its tools and session."""
        command = cfg.get("command")
        url = os.path.expandvars(str(cfg.get("url", "")))
        explicit_transport = str(cfg.get("transport", "")).lower()
        # A child is remote (HTTP) when it declares a ``url`` or an http/sse
        # ``transport``; otherwise it is a local stdio subprocess run via
        # ``command``. Either kind loads transparently from the same config.
        is_remote = bool(url) or explicit_transport in (
            "streamable-http",
            "streamable_http",
            "http",
            "sse",
        )
        if not command and not is_remote:
            logger.warning(
                f"Server '{server_name}' has neither 'command' nor 'url', skipping."
            )
            return None

        args = cfg.get("args", [])
        env = cfg.get("env", None)
        timeout = float(cfg.get("timeout", 300.0))

        # Session-pool sizing (CONCEPT:ECO-4.34): remote children may hold N
        # independent connections for parallel in-flight calls; stdio children
        # are single-pipe and always keep exactly one session.
        from agent_utilities.core.config import config as agent_config

        pool_size = 1
        if is_remote:
            pool_size = max(
                1, int(cfg.get("pool_size") or agent_config.mcp_child_pool_size)
            )

        # Build environment dict with dynamic expansions (stdio children only).
        merged_env = os.environ.copy()
        if env:
            for k, v in env.items():
                merged_env[k] = os.path.expandvars(str(v))

        # Ensure PYTHONPATH and active path are preserved
        if "PYTHONPATH" not in merged_env and "PYTHONPATH" in os.environ:
            merged_env["PYTHONPATH"] = os.environ["PYTHONPATH"]

        # Expand any ${VAR} in remote headers (e.g. auth tokens).
        headers = cfg.get("headers")
        if headers:
            headers = {k: os.path.expandvars(str(v)) for k, v in headers.items()}

        # SSE when explicitly requested or the url path ends in ``/sse``; else
        # streamable-http (the default for a remote child).
        use_sse = explicit_transport == "sse" or url.rstrip("/").endswith("/sse")

        logger.info(
            "Starting child server '%s' (timeout %ss) via %s: %s",
            server_name,
            timeout,
            ("sse" if use_sse else "streamable-http") if is_remote else "stdio",
            url if is_remote else f"{command} {' '.join(args)}",
        )

        child_stack = contextlib.AsyncExitStack()
        try:

            async def _connect_one():
                if is_remote:
                    if use_sse:
                        if sse_client is None:
                            raise RuntimeError(
                                "mcp SDK has no sse_client for SSE transport"
                            )
                        transport = sse_client(url, headers=headers)
                    else:
                        if streamablehttp_client is None:
                            raise RuntimeError(
                                "mcp SDK has no streamablehttp_client for "
                                "streamable-http transport"
                            )
                        transport = streamablehttp_client(url, headers=headers)
                    # streamable-http yields (read, write, get_session_id); sse
                    # yields (read, write). Take the first two streams either way.
                    streams = await child_stack.enter_async_context(transport)
                    read_stream, write_stream = streams[0], streams[1]
                else:
                    server_params = StdioServerParameters(
                        command=command, args=args, env=merged_env
                    )
                    # Connect via stdio transport
                    read_stream, write_stream = await child_stack.enter_async_context(
                        stdio_client(server_params)
                    )

                # Create client session (transport-agnostic from here on)
                session = await child_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )

                await session.initialize()
                return session

            async def _connect_and_init():
                sessions = [await _connect_one() for _ in range(pool_size)]
                tools_result = await sessions[0].list_tools()
                return sessions, tools_result.tools

            sessions, tools = await asyncio.wait_for(
                _connect_and_init(), timeout=timeout
            )

            # Register the child_stack in the main exit_stack so it persists
            await self.exit_stack.enter_async_context(child_stack)

            logger.info(
                "Loaded %d tools from child server '%s' (%d session%s)",
                len(tools),
                server_name,
                len(sessions),
                "" if len(sessions) == 1 else "s",
            )
            return server_name, sessions, tools, cfg

        except TimeoutError:
            logger.error(
                f"Failed to start child server '{server_name}': Timeout of {timeout}s exceeded"
            )
            await child_stack.aclose()
            return None
        except Exception as e:
            logger.error(
                f"Failed to start child server '{server_name}': {e}",
                exc_info=True,
            )
            await child_stack.aclose()
            return None

    async def start_children(self):
        """Parse configuration and start all child processes concurrently."""
        if not self.config_path.exists():
            logger.error(f"Config path {self.config_path} does not exist.")
            return

        try:
            content = self.config_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read config file {self.config_path}: {e}")
            return

        # Attempt to expand env vars using the project's helper
        try:
            from agent_utilities.base_utilities import expand_env_vars

            expanded_content = expand_env_vars(content)
            config_data = json.loads(expanded_content)
        except Exception as e:
            logger.warning(
                f"Could not use base_utilities.expand_env_vars, using fallback parser: {e}"
            )
            try:
                config_data = json.loads(content)
            except Exception as json_err:
                logger.error(f"Failed to parse config as JSON: {json_err}")
                return

        mcp_servers = config_data.get("mcpServers", {})

        tasks = []
        for server_name, cfg in mcp_servers.items():
            # Skip ourselves to avoid self-infinite recursion loops
            if server_name == "mcp-multiplexer" or cfg.get("disabled", False):
                continue
            tasks.append(self._start_child(server_name, cfg))

        if not tasks:
            logger.info("No active child servers configured.")
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                # An exception propagated from asyncio.gather, already logged inside task
                continue

            if not isinstance(result, tuple):
                continue

            server_name, session, tools, cfg = result
            sessions = (
                list(session) if isinstance(session, (list, tuple)) else [session]
            )
            self.sessions[server_name] = sessions[0]

            # Wrap the live session pool in the per-child hardening runtime
            # (CONCEPT:ECO-4.34): bounded concurrency + queue timeout +
            # round-robin dispatch across pooled connections.
            runtime = ChildRuntime(server_name, cfg)
            runtime.adopt_sessions(sessions)
            self.children[server_name] = runtime

            disabled_tools = cfg.get("disabledTools", [])
            enabled_tools = cfg.get("enabledTools", None)

            for tool in tools:
                # 1. Whitelist Check (if enabledTools is defined)
                if enabled_tools is not None:
                    import fnmatch

                    matched = any(
                        fnmatch.fnmatch(tool.name, pat) for pat in enabled_tools
                    )
                    if not matched:
                        logger.info(
                            f"Skipping non-whitelisted tool '{tool.name}' from child server '{server_name}'"
                        )
                        continue

                # 2. Blacklist Check
                if disabled_tools:
                    import fnmatch

                    matched_disabled = any(
                        fnmatch.fnmatch(tool.name, pat) for pat in disabled_tools
                    )
                    if matched_disabled:
                        logger.info(
                            f"Skipping disabled tool '{tool.name}' from child server '{server_name}'"
                        )
                        continue

                prefix = get_server_prefix(server_name)
                prefixed_name = clean_tool_name(prefix, server_name, tool.name)
                self.tool_to_server[prefixed_name] = (server_name, tool.name)

                prefixed_tool = mcp.types.Tool(
                    name=prefixed_name,
                    description=tool.description or "",
                    inputSchema=tool.inputSchema,
                )
                self.aggregated_tools.append(prefixed_tool)


def _resolve_config_path(explicit: str | None) -> Path:
    """Resolve the mcp_config.json path from --config, ``MCP_CONFIG``, or the
    standard discovery candidates."""
    if explicit:
        return Path(explicit)
    if os.environ.get("MCP_CONFIG"):
        return Path(os.environ["MCP_CONFIG"])
    candidates = [
        Path.home() / ".gemini" / "antigravity" / "mcp_config.json",
        Path.home() / ".config" / "agent-utilities" / "mcp_config.json",
        Path.home() / ".config" / "agent-utilities" / "config.json",
        Path("mcp_config.json"),
        Path("workspace/mcp_config.json"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _register_forwarding_tools(mcp, mux: MCPMultiplexer) -> None:
    """Register one FastMCP forwarding tool per aggregated child tool.

    Each tool keeps the multiplexer's prefixed name and the child's published
    input schema; calling it forwards the validated arguments to the owning
    child session via :meth:`MCPMultiplexer.call_proxied_tool`.
    """

    def _make_forwarder(prefixed_name: str):
        async def _forward(**kwargs: Any) -> ToolResult:
            result = await mux.call_proxied_tool(prefixed_name, kwargs)
            return ToolResult(
                content=list(getattr(result, "content", []) or []),
                structured_content=getattr(result, "structuredContent", None),
            )

        return _forward

    for tool in mux.aggregated_tools:
        schema = tool.inputSchema or {"type": "object", "properties": {}}
        mcp.add_tool(
            FunctionTool(
                name=tool.name,
                description=tool.description or "",
                parameters=schema,
                fn=_make_forwarder(tool.name),
            )
        )


def get_mcp_instance():
    """Build the multiplexer's FastMCP server and the aggregation engine.

    Returns ``(args, mcp, mux)``. ``args`` carries the standard
    ``--transport/--host/--port`` options parsed by ``create_mcp_server``; the
    multiplexer-specific ``--config`` is parsed separately so both coexist.
    Child servers are started (and their tools registered) later, inside the
    serving event loop, by :func:`mcp_server`.
    """
    from agent_utilities.mcp.server_factory import create_mcp_server

    # Parse --config without disturbing the factory's own argv parsing.
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument("--config", default=os.environ.get("MCP_CONFIG"))
    cfg_args, _ = cfg_parser.parse_known_args()

    config_path = _resolve_config_path(cfg_args.config)
    logger.info("Using MCP config: %s", config_path)
    mux = MCPMultiplexer(config_path)

    args, mcp, middlewares = create_mcp_server(
        name="mcp-multiplexer",
        version="0.1.0",
        instructions=(
            "Aggregates multiple child MCP servers (declared in mcp_config.json) "
            "into a single unified server. Tools are namespaced by a short, "
            "host-aware server prefix; calls are forwarded to the owning child."
        ),
    )
    for middleware in middlewares:
        mcp.add_middleware(middleware)

    return args, mcp, mux


async def _serve(args, mcp, mux: MCPMultiplexer) -> None:
    """Start child servers, register forwarding tools, and serve — all in one
    event loop so the child ``ClientSession`` objects stay bound to the loop
    that runs the server."""
    try:
        await mux.start_children()
        _register_forwarding_tools(mcp, mux)
        logger.info(
            "Aggregated %d tools from %d child servers. Serving over %s.",
            len(mux.aggregated_tools),
            len(mux.sessions),
            getattr(args, "transport", "stdio"),
        )

        transport = getattr(args, "transport", "stdio")
        host = getattr(args, "host", "0.0.0.0")
        port = int(getattr(args, "port", 8000))

        from agent_utilities.mcp.server_factory import protect_stdio_jsonrpc

        if transport == "stdio":
            protect_stdio_jsonrpc()
            await mcp.run_async(transport="stdio")
        elif transport == "streamable-http":
            await mcp.run_async(transport="streamable-http", host=host, port=port)
        elif transport == "sse":
            await mcp.run_async(transport="sse", host=host, port=port)
        else:
            protect_stdio_jsonrpc()
            await mcp.run_async(transport="stdio")
    finally:
        logger.info("Shutting down multiplexer and child servers...")
        await mux.exit_stack.aclose()


def mcp_server() -> None:
    """mcp-multiplexer entry point (registered as console_scripts).

    Standard ``mcp_server.py`` scaffolding: deployable as a ``stdio`` or
    ``streamable-http`` server via the ``--transport`` flag.
    """
    args, mcp, mux = get_mcp_instance()
    try:
        asyncio.run(_serve(args, mcp, mux))
    except KeyboardInterrupt:
        logger.info("Multiplexer execution interrupted.")


# Back-compat alias — the previous console_scripts entry referenced ``main``.
main = mcp_server


if __name__ == "__main__":
    mcp_server()
