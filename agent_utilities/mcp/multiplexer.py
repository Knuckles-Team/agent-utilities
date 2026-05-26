#!/usr/bin/env python
"""Multi-MCP Server Multiplexer.

Aggregates multiple underlying MCP servers into a single unified MCP server
instance, delegating tool requests dynamically based on prefixed tool names.
This drastically speeds up boot times and avoids process resource contention.

CONCEPT:ECO-4.0 — MCP Standardized Interfaces
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import mcp.types
from mcp import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession
from mcp.server import Server
from mcp.server.stdio import stdio_server

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
        self.tool_to_server: dict[
            str, tuple[str, str]
        ] = {}  # prefixed_name -> (server_name, original_name)
        self.aggregated_tools: list[mcp.types.Tool] = []
        self.server = Server("mcp-multiplexer")
        self._setup_handlers()

    def _setup_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> list[mcp.types.Tool]:
            logger.info("Listing aggregated tools")
            return self.aggregated_tools

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict[str, Any] | None = None
        ) -> mcp.types.CallToolResult:
            logger.info(f"Calling tool: {name}")
            if name not in self.tool_to_server:
                raise ValueError(f"Tool {name} is not registered in multiplexer")

            server_name, original_name = self.tool_to_server[name]
            session = self.sessions.get(server_name)
            if not session:
                raise RuntimeError(f"Session for server '{server_name}' is not active")

            try:
                # Forward the call directly to the child session
                result = await session.call_tool(original_name, arguments or {})
                return result
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

        for server_name, cfg in mcp_servers.items():
            # Skip ourselves to avoid self-infinite recursion loops
            if server_name == "mcp-multiplexer" or cfg.get("disabled", False):
                continue

            command = cfg.get("command")
            if not command:
                logger.warning(f"Server '{server_name}' has no command, skipping.")
                continue

            args = cfg.get("args", [])
            env = cfg.get("env", None)

            # Build environment dict with dynamic expansions
            merged_env = os.environ.copy()
            if env:
                for k, v in env.items():
                    merged_env[k] = os.path.expandvars(str(v))

            # Ensure PYTHONPATH and active path are preserved
            if "PYTHONPATH" not in merged_env and "PYTHONPATH" in os.environ:
                merged_env["PYTHONPATH"] = os.environ["PYTHONPATH"]

            logger.info(
                f"Starting child server '{server_name}': {command} {' '.join(args)}"
            )

            try:
                server_params = StdioServerParameters(
                    command=command, args=args, env=merged_env
                )

                # Connect via stdio transport
                read_stream, write_stream = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )

                # Create client session
                session = await self.exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )

                await session.initialize()

                tools_result = await session.list_tools()
                logger.info(
                    f"Loaded {len(tools_result.tools)} tools from child server '{server_name}'"
                )

                disabled_tools = cfg.get("disabledTools", [])
                enabled_tools = cfg.get("enabledTools", None)

                self.sessions[server_name] = session
                for tool in tools_result.tools:
                    # 1. Whitelist Check (if enabledTools is defined)
                    if enabled_tools is not None:
                        import fnmatch
                        matched = any(fnmatch.fnmatch(tool.name, pat) for pat in enabled_tools)
                        if not matched:
                            logger.info(
                                f"Skipping non-whitelisted tool '{tool.name}' from child server '{server_name}'"
                            )
                            continue

                    # 2. Blacklist Check
                    if disabled_tools:
                        import fnmatch
                        matched_disabled = any(fnmatch.fnmatch(tool.name, pat) for pat in disabled_tools)
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

            except Exception as e:
                logger.error(
                    f"Failed to start/initialize child server '{server_name}': {e}",
                    exc_info=True,
                )

    async def run(self):
        """Run the multiplexer stdio server."""
        try:
            await self.start_children()
            logger.info(
                f"Aggregated {len(self.aggregated_tools)} total tools. Starting stdio server."
            )
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options(),
                    raise_exceptions=True,
                )
        finally:
            logger.info("Shutting down multiplexer and child servers...")
            await self.exit_stack.aclose()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-MCP Server Multiplexer")
    parser.add_argument(
        "--config",
        default=os.environ.get("MCP_CONFIG"),
        help="Path to mcp_config.json file",
    )
    args = parser.parse_args()

    config_path = None
    if args.config:
        config_path = Path(args.config)
    else:
        candidates = [
            Path("/home/genius/.gemini/antigravity/mcp_config.json"),
            Path.home() / ".config" / "agent-utilities" / "mcp_config.json",
            Path.home() / ".config" / "agent-utilities" / "config.json",
            Path("mcp_config.json"),
            Path("workspace/mcp_config.json"),
        ]
        for c in candidates:
            if c.exists():
                config_path = c
                break
        if not config_path:
            config_path = candidates[0]

    logger.info(f"Using MCP config: {config_path}")

    multiplexer = MCPMultiplexer(config_path)
    try:
        asyncio.run(multiplexer.run())
    except KeyboardInterrupt:
        logger.info("Multiplexer execution interrupted.")


if __name__ == "__main__":
    main()
