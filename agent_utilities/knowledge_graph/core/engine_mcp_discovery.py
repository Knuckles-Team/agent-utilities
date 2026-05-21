"""MCP Discovery Mixin — Live tool discovery and freshness verification.

CONCEPT:ECO-4.2 — MCP Server Live Tool Discovery

Provides the ability to connect to MCP servers at ingestion time,
discover their tools via ``list_tools()``, and cache the metadata
in the Knowledge Graph. Supports lazy-refresh verification on
subsequent loads to ensure cached tool metadata stays current.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from .._engine_protocol import _EngineProtocol

    _Base = _EngineProtocol
else:
    _Base = object


logger = logging.getLogger(__name__)


class MCPDiscoveryMixin(_Base):
    """Live MCP server tool discovery and KG cache management.

    CONCEPT:ECO-4.2 — MCP Server Live Tool Discovery

    Enables the ingestion pipeline to:
    1. Parse ``mcp_config.json`` files to extract server entries.
    2. Optionally live-connect to each server to run ``list_tools()``.
    3. Cache discovered tools as ``CallableResource`` nodes in the KG.
    4. Verify freshness of cached metadata on subsequent loads.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_mcp_config(self, config_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse an ``mcp_config.json`` payload into normalized server entries.

        Handles the standard format used across all agent-packages::

            {"mcpServers": {"name": {"command": ..., "args": [...], "env": {...}}}}

        Args:
            config_data: Parsed JSON dictionary from an ``mcp_config.json`` file.

        Returns:
            List of normalized server dicts with keys:
            ``name``, ``command``, ``args``, ``env``, ``tool_flags``,
            ``config_hash``.

        """
        servers: list[dict[str, Any]] = []
        mcp_servers = config_data.get("mcpServers", {})

        for name, entry in mcp_servers.items():
            command = entry.get("command", "")
            args = entry.get("args", [])
            env = entry.get("env", {})
            disabled = entry.get("disabled", False)

            if disabled:
                logger.info("Skipping disabled MCP server '%s'", name)
                continue

            # Extract tool-enable flags (env vars ending in TOOL = "True")
            tool_flags = self._parse_tool_flags(env)

            # Build a deterministic hash of the config for freshness checks
            config_hash = self._compute_config_hash(name, command, args, env)

            servers.append(
                {
                    "name": name,
                    "command": command,
                    "args": args,
                    "env": env,
                    "tool_flags": tool_flags,
                    "config_hash": config_hash,
                    "disabled_tools": entry.get("disabledTools", []),
                }
            )

        return servers

    async def discover_mcp_tools(
        self, server_config: dict[str, Any], timeout: float = 30.0
    ) -> list[dict[str, Any]]:
        """Start an MCP server from its config, call ``list_tools()``, return tool metadata.

        CONCEPT:ECO-4.2 — Live MCP server connection for tool metadata caching.

        This attempts to start the server as a subprocess (using the command/args
        from the config), connect via stdio, and retrieve the tool list. If the
        connection fails, returns an empty list and logs a warning.

        Args:
            server_config: Normalized server dict from :meth:`parse_mcp_config`.
            timeout: Maximum seconds to wait for the server to respond.

        Returns:
            List of tool metadata dicts with keys: ``name``, ``description``,
            ``input_schema``, ``annotations``.

        """
        name = server_config.get("name", "unknown")
        command = server_config.get("command", "")
        args = server_config.get("args", [])

        if not command:
            logger.warning(
                "No command for MCP server '%s' — skipping live discovery", name
            )
            return []

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            logger.debug(
                "mcp package not installed — skipping live discovery for '%s'", name
            )
            return []

        import asyncio
        import os
        import shutil

        tools: list[dict[str, Any]] = []

        env_vars = os.environ.copy()

        # Ensure ~/.local/bin is in PATH for GUI environments that don't source .bashrc
        local_bin = os.path.expanduser("~/.local/bin")
        if local_bin not in env_vars.get("PATH", ""):
            env_vars["PATH"] = f"{local_bin}:{env_vars.get('PATH', '')}".strip(":")

        server_env = server_config.get("env")
        if server_env:
            for k, v in server_env.items():
                env_vars[k] = str(v)

        # Silence FastMCP startup output to prevent stdout pollution breaking JSON-RPC
        env_vars["FASTMCP_SHOW_SERVER_BANNER"] = "false"
        env_vars["FASTMCP_LOG_LEVEL"] = "WARNING"

        # Prevent spawned MCP servers from acquiring a write lock on the knowledge graph
        # They only expose tools, the orchestrator handles writing
        env_vars["LADYBUG_DB_READ_ONLY"] = "1"

        # Resolve command to an absolute path using the updated environment PATH
        resolved_command = shutil.which(command, path=env_vars.get("PATH"))
        if resolved_command:
            command = resolved_command

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env_vars,
        )

        try:
            async with asyncio.timeout(timeout):
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.list_tools()

                        for tool in result.tools:
                            tool_dict: dict[str, Any] = {
                                "name": tool.name,
                                "description": getattr(tool, "description", "") or "",
                                "input_schema": {},
                            }
                            if hasattr(tool, "inputSchema") and tool.inputSchema:
                                tool_dict["input_schema"] = (
                                    tool.inputSchema
                                    if isinstance(tool.inputSchema, dict)
                                    else {}
                                )
                            if hasattr(tool, "annotations") and tool.annotations:
                                tool_dict["annotations"] = (
                                    tool.annotations
                                    if isinstance(tool.annotations, dict)
                                    else str(tool.annotations)
                                )
                            tools.append(tool_dict)

            logger.info(
                "[ECO-4.11] Live discovery for '%s': found %d tools",
                name,
                len(tools),
            )
        except TimeoutError:
            logger.warning(
                "[ECO-4.11] Live discovery for '%s' timed out after %.0fs",
                name,
                timeout,
            )
        except Exception as e:
            logger.warning(
                "[ECO-4.11] Live discovery for '%s' failed: %s",
                name,
                e,
            )

        return tools

    def check_server_freshness(
        self,
        server_name: str,
        config_hash: str,
        max_age_hours: float = 24.0,
    ) -> bool:
        """Check if a server's cached KG data is still fresh.

        Args:
            server_name: The MCP server name (e.g., ``portainer-agent``).
            config_hash: Hash of the current config for change detection.
            max_age_hours: Maximum age in hours before data is considered stale.

        Returns:
            True if the cached data is fresh (no re-ingestion needed).

        """
        if not self.backend:
            return False

        server_id = f"srv:{server_name}"
        try:
            rows = self.backend.execute(
                "MATCH (s:Server {id: $sid}) RETURN s.config_hash AS hash, s.timestamp AS ts",
                {"sid": server_id},
            )
            if not rows:
                return False

            row = rows[0]
            cached_hash = row.get("hash", "")
            cached_ts = row.get("ts", "")

            # Config changed → stale
            if cached_hash != config_hash:
                logger.info(
                    "[ECO-4.11] Server '%s' config hash changed: %s → %s",
                    server_name,
                    cached_hash[:8],
                    config_hash[:8],
                )
                return False

            # Check age
            if cached_ts:
                try:
                    cached_time = time.mktime(
                        time.strptime(cached_ts, "%Y-%m-%dT%H:%M:%SZ")
                    )
                    age_hours = (time.time() - cached_time) / 3600
                    if age_hours > max_age_hours:
                        logger.info(
                            "[ECO-4.11] Server '%s' cache expired (%.1fh old, max %.1fh)",
                            server_name,
                            age_hours,
                            max_age_hours,
                        )
                        return False
                except (ValueError, OverflowError):
                    return False

            return True

        except Exception as e:
            logger.debug("Freshness check failed for '%s': %s", server_name, e)
            return False

    async def verify_mcp_freshness(
        self, server_name: str, server_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Compare KG-cached tools against live server tools.

        Args:
            server_name: The MCP server identifier.
            server_config: Normalized server config dict.

        Returns:
            Dict with keys: ``fresh`` (bool), ``cached_count`` (int),
            ``live_count`` (int), ``changes`` (list of change descriptions).

        """
        result: dict[str, Any] = {
            "fresh": True,
            "cached_count": 0,
            "live_count": 0,
            "changes": [],
        }

        # Get cached tool count
        if self.backend:
            server_id = f"srv:{server_name}"
            try:
                rows = self.backend.execute(
                    "MATCH (s:Server {id: $sid})-[:PROVIDES]->(r:CallableResource) "
                    "RETURN count(r) AS cnt",
                    {"sid": server_id},
                )
                if rows:
                    result["cached_count"] = rows[0].get("cnt", 0)
            except Exception:
                pass  # nosec B110

        # Get live tool count
        live_tools = await self.discover_mcp_tools(server_config, timeout=15.0)
        result["live_count"] = len(live_tools)

        if result["cached_count"] != result["live_count"]:
            result["fresh"] = False
            result["changes"].append(
                f"Tool count changed: {result['cached_count']} → {result['live_count']}"
            )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_flags(env_vars: dict[str, str]) -> list[str]:
        """Extract enabled tool groups from environment variables.

        MCP server configs use env vars like ``DOCKERTOOL=True`` to enable
        tool groups. This extracts those flags into a capabilities list.

        Args:
            env_vars: Environment variable dictionary from the server config.

        Returns:
            List of tool flag names (e.g., ``["docker", "stack", "system"]``).

        """
        flags: list[str] = []
        for key, value in env_vars.items():
            if key.upper().endswith("TOOL") and str(value).lower() in (
                "true",
                "1",
                "yes",
            ):
                # Strip the TOOL suffix and normalize
                flag_name = key[:-4].lower().rstrip("_")
                if flag_name:
                    flags.append(flag_name)
        return sorted(flags)

    @staticmethod
    def _compute_config_hash(
        name: str, command: str, args: list[str], env: dict[str, str]
    ) -> str:
        """Compute a deterministic hash of a server's configuration.

        Used for freshness checks — if the hash changes, the KG cache
        is invalidated and the server is re-ingested.

        """
        payload = json.dumps(
            {
                "name": name,
                "command": command,
                "args": args,
                "env": sorted(env.items()),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
