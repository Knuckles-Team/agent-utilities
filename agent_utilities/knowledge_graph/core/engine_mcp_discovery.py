"""MCP Discovery Mixin — Live tool discovery and freshness verification.

CONCEPT:AU-ECO.mcp.live-server-metadata-cache — MCP Server Live Tool Discovery

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

    CONCEPT:AU-ECO.mcp.live-server-metadata-cache — MCP Server Live Tool Discovery

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
            # Remote (streamable-http / sse) children declare a url + transport
            # instead of a command — capture them so HTTP servers are discoverable
            # rather than mis-treated as empty stdio entries.
            url = entry.get("url", "")
            transport = entry.get("transport", "")
            headers = entry.get("headers", {})
            disabled = entry.get("disabled", False)

            if disabled:
                logger.info("Skipping disabled MCP server '%s'", name)
                continue

            # Extract tool-enable flags (env vars ending in TOOL = "True")
            tool_flags = self._parse_tool_flags(env)

            # Build a deterministic hash of the config for freshness checks
            config_hash = self._compute_config_hash(
                name, command, args, env, url=url, transport=transport
            )

            servers.append(
                {
                    "name": name,
                    "command": command,
                    "args": args,
                    "env": env,
                    "url": url,
                    "transport": transport,
                    "headers": headers,
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

        CONCEPT:AU-ECO.mcp.live-server-metadata-cache — Live MCP server connection for tool metadata caching.

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
        url = server_config.get("url", "")
        transport = str(server_config.get("transport", "")).lower()

        # Remote children (streamable-http / sse) connect over the network instead
        # of spawning a subprocess. Most of the fleet is served this way behind the
        # multiplexer, so without this branch their tools are never discovered.
        if url or transport in ("streamable-http", "streamable_http", "http", "sse"):
            return await self._discover_remote_tools(
                name, url, transport, server_config.get("headers"), timeout
            )

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
                        tools = self._extract_tools(result.tools)

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

    async def _discover_remote_tools(
        self,
        name: str,
        url: str,
        transport: str,
        headers: dict[str, str] | None,
        timeout: float,
    ) -> list[dict[str, Any]]:
        """Discover tools from a remote (streamable-http / sse) MCP server.

        Mirrors the multiplexer's transport handling (CONCEPT:AU-ECO.mcp.profile-differences-from-client): pick sse
        vs streamable-http, attach the optional service-account bearer
        (CONCEPT:AU-OS.identity.so-jwt-protected-children) so jwt-protected children are reachable, then
        ``list_tools()``. A connection/auth failure returns an empty list (logged),
        so an unreachable or 401 server degrades to a tool-less Server node rather
        than aborting the whole toolkit ingest.
        """
        import asyncio
        import os

        try:
            from mcp import ClientSession
        except ImportError:
            logger.debug(
                "mcp package not installed — skipping live discovery for '%s'", name
            )
            return []

        url = os.path.expandvars(url or "")
        if not url:
            logger.warning(
                "Remote MCP server '%s' has no url — skipping live discovery", name
            )
            return []

        hdrs = (
            {k: os.path.expandvars(str(v)) for k, v in headers.items()}
            if headers
            else {}
        )
        # Attach the multiplexer's service-account bearer when configured
        # (opt-in MCP_CLIENT_AUTH=oidc-client-credentials); never overrides a
        # child's own Authorization header; a mint failure degrades to no header.
        try:
            from agent_utilities.mcp.client_credentials import child_auth_header

            _svc = child_auth_header(hdrs)
            if _svc:
                hdrs = {**hdrs, **_svc}
        except Exception:  # noqa: BLE001 - auth is best-effort
            pass

        use_sse = transport == "sse" or url.rstrip("/").endswith("/sse")
        tools: list[dict[str, Any]] = []
        try:
            if use_sse:
                from mcp.client.sse import sse_client

                transport_cm = sse_client(url, headers=hdrs)
            else:
                from mcp.client.streamable_http import streamablehttp_client

                transport_cm = streamablehttp_client(url, headers=hdrs)

            async with asyncio.timeout(timeout):
                async with transport_cm as streams:
                    read_stream, write_stream = streams[0], streams[1]
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        result = await session.list_tools()
                        tools = self._extract_tools(result.tools)

            logger.info(
                "[ECO-4.11] Remote discovery for '%s': found %d tools", name, len(tools)
            )
        except TimeoutError:
            logger.warning(
                "[ECO-4.11] Remote discovery for '%s' timed out after %.0fs",
                name,
                timeout,
            )
        except Exception as e:  # noqa: BLE001 - unreachable/401 → tool-less node
            logger.warning("[ECO-4.11] Remote discovery for '%s' failed: %s", name, e)

        return tools

    @staticmethod
    def _extract_tools(raw_tools: Any) -> list[dict[str, Any]]:
        """Normalize MCP ``list_tools()`` results into tool-metadata dicts.

        Shared by stdio and remote discovery so both transports yield the same
        shape (``name``, ``description``, ``input_schema``, optional
        ``annotations``).
        """
        tools: list[dict[str, Any]] = []
        for tool in raw_tools or []:
            tool_dict: dict[str, Any] = {
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
                "input_schema": {},
            }
            if getattr(tool, "inputSchema", None):
                tool_dict["input_schema"] = (
                    tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
                )
            if getattr(tool, "annotations", None):
                tool_dict["annotations"] = (
                    tool.annotations
                    if isinstance(tool.annotations, dict)
                    else str(tool.annotations)
                )
            tools.append(tool_dict)
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
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str],
        url: str = "",
        transport: str = "",
    ) -> str:
        """Compute a deterministic hash of a server's configuration.

        Used for freshness checks — if the hash changes, the KG cache
        is invalidated and the server is re-ingested. Includes the remote
        ``url``/``transport`` so an endpoint change re-triggers discovery.

        """
        payload = json.dumps(
            {
                "name": name,
                "command": command,
                "args": args,
                "env": sorted(env.items()),
                "url": url,
                "transport": transport,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
