"""MCP Discovery Mixin — Live tool discovery and freshness verification.

CONCEPT:AU-ECO.mcp.live-server-metadata-cache — MCP Server Live Tool Discovery

Provides the ability to connect to MCP servers at ingestion time,
discover their tools via ``list_tools()``, and cache the metadata
in the Knowledge Graph. Supports lazy-refresh verification on
subsequent loads to ensure cached tool metadata stays current.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_MAX_MCP_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_DISCOVERY_TIMEOUT_SECONDS = 300.0
_EPHEMERAL_FRESHNESS_KEY = secrets.token_bytes(32)
_SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:AUTHORIZATION|COOKIE|CREDENTIAL|PASSWORD|SECRET|TOKEN|API_KEY|HMAC_KEY)(?:_|$)",
    re.IGNORECASE,
)
_SERVER_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


@dataclass(frozen=True, slots=True)
class MCPProbePort:
    """Canonical MCP adapter seams consumed by KG discovery and source sync.

    This value contains composition-owned callables only. It neither implements
    another transport nor owns another catalog/source lifecycle.
    """

    probe_declaration: Callable[..., Awaitable[dict[str, Any]]]
    resolve_config_path: Callable[[str | None], Path]
    multiplexer_factory: Callable[[Path], Any]
    run_async: Callable[..., Any]


class MCPDiscoveryError(RuntimeError):
    """Stable live-discovery failure with no child configuration details."""


def _require_mcp_probe_port(engine: Any) -> MCPProbePort:
    """Return the process-composed MCP probe or fail closed."""
    port = getattr(engine, "mcp_probe_port", None)
    if isinstance(port, MCPProbePort):
        return port
    logger.warning("[ECO-4.11] MCP discovery probe is unavailable")
    raise MCPDiscoveryError("mcp_discovery_unavailable")


def _validate_mcp_probe_result(result: Any) -> dict[str, Any]:
    """Require the canonical probe envelope before reading it."""
    if not isinstance(result, dict):
        raise MCPDiscoveryError("mcp_discovery_catalog_invalid")
    return result


class MCPDiscoveryMixin:
    """Live MCP server tool discovery and KG cache management.

    CONCEPT:AU-ECO.mcp.live-server-metadata-cache — MCP Server Live Tool Discovery

    Enables the ingestion pipeline to:
    1. Parse ``mcp_config.json`` files to extract server entries.
    2. Optionally live-connect to each server to run ``list_tools()``.
    3. Cache discovered tools as ``CallableResource`` nodes in the KG.
    4. Verify freshness of cached metadata on subsequent loads.
    """

    backend: Any

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
            ``config_hash``. The hash is a keyed, opaque freshness identity;
            raw credentials, endpoints, commands, arguments, and local paths
            never become graph metadata.

        """
        mcp_servers = self._validate_mcp_config(config_data)
        servers: list[dict[str, Any]] = []
        for raw_name, raw_entry in mcp_servers.items():
            server = self._normalize_mcp_server(raw_name, raw_entry)
            if server is not None:
                servers.append(server)

        return servers

    async def discover_mcp_tools(
        self, server_config: dict[str, Any], timeout: float = 30.0
    ) -> list[dict[str, Any]]:
        """Start an MCP server from its config, call ``list_tools()``, return tool metadata.

        CONCEPT:AU-ECO.mcp.live-server-metadata-cache — Live MCP server connection for tool metadata caching.

        The canonical multiplexer probe owns both stdio and remote transports.
        It applies the same AgentConfig TLS/auth/egress/environment boundary as
        GraphOS and releases the child immediately after ``list_tools()``.

        Args:
            server_config: Normalized server dict from :meth:`parse_mcp_config`.
            timeout: Maximum seconds to wait for the server to respond.

        Returns:
            List of tool metadata dicts with keys: ``name``, ``description``,
            ``input_schema``, ``annotations``.

        """
        try:
            bounded_timeout = float(timeout)
        except (TypeError, ValueError):
            bounded_timeout = 0.0
        if (
            not math.isfinite(bounded_timeout)
            or not 0.001 <= bounded_timeout <= _MAX_DISCOVERY_TIMEOUT_SECONDS
        ):
            raise MCPDiscoveryError("mcp_discovery_timeout_invalid")
        name = str(server_config.get("name") or "")
        if not _SERVER_NAME.fullmatch(name):
            raise MCPDiscoveryError("mcp_discovery_declaration_invalid")

        probe_port = _require_mcp_probe_port(self)
        try:
            result = await probe_port.probe_declaration(
                name,
                server_config,
                timeout=bounded_timeout,
            )
        except Exception as exc:
            logger.warning(
                "[ECO-4.11] MCP discovery rejected (exception_type=%s)",
                type(exc).__name__,
            )
            raise MCPDiscoveryError("mcp_discovery_unavailable") from None
        result = _validate_mcp_probe_result(result)
        if result.get("error") is not None:
            logger.warning("[ECO-4.11] MCP discovery unavailable")
            raise MCPDiscoveryError("mcp_discovery_unavailable")

        tools = result.get("tools")
        if not isinstance(tools, list):
            raise MCPDiscoveryError("mcp_discovery_catalog_invalid")
        normalized: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                raise MCPDiscoveryError("mcp_discovery_catalog_invalid")
            item = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema", {}),
            }
            if "annotations" in tool:
                item["annotations"] = tool["annotations"]
            normalized.append(item)
        logger.info("[ECO-4.11] MCP discovery found %d tools", len(normalized))
        return normalized

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
        if (
            not _SERVER_NAME.fullmatch(str(server_name or ""))
            or not re.fullmatch(r"[0-9a-f]{64}", str(config_hash or ""))
            or not isinstance(max_age_hours, int | float)
            or not math.isfinite(float(max_age_hours))
            or not 0.0 <= float(max_age_hours) <= 8_760.0
        ):
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
                logger.info("[ECO-4.11] MCP freshness identity changed")
                return False

            # Check age
            if cached_ts:
                try:
                    cached_time = time.mktime(
                        time.strptime(cached_ts, "%Y-%m-%dT%H:%M:%SZ")
                    )
                    age_hours = (time.time() - cached_time) / 3600
                    if age_hours > max_age_hours:
                        logger.info("[ECO-4.11] MCP discovery cache expired")
                        return False
                except (ValueError, OverflowError):
                    return False

            return True

        except Exception as exc:
            logger.debug(
                "MCP freshness check failed (exception_type=%s)",
                type(exc).__name__,
            )
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
        if not _SERVER_NAME.fullmatch(str(server_name or "")):
            raise MCPDiscoveryError("mcp_discovery_declaration_invalid")
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
    def _validate_mcp_config(config_data: dict[str, Any]) -> dict[str, Any]:
        """Validate the bounded MCP config envelope and return its catalog."""
        try:
            encoded = json.dumps(
                config_data,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError):
            raise ValueError("MCP configuration is invalid") from None
        if len(encoded) > _MAX_MCP_CONFIG_BYTES or not isinstance(config_data, dict):
            raise ValueError("MCP configuration exceeds its boundary")
        mcp_servers = config_data.get("mcpServers", {})
        if not isinstance(mcp_servers, dict) or len(mcp_servers) > 512:
            raise ValueError("MCP server catalog is invalid")
        return mcp_servers

    @staticmethod
    def _validate_mcp_child(
        raw_name: Any, raw_entry: Any
    ) -> tuple[str, dict[str, Any]]:
        """Validate and shallow-copy one child declaration."""
        if not isinstance(raw_name, str):
            raise ValueError("MCP child declaration is invalid")
        if _SERVER_NAME.fullmatch(raw_name) is None:
            raise ValueError("MCP child declaration is invalid")
        if not isinstance(raw_entry, dict):
            raise ValueError("MCP child declaration is invalid")
        if len(raw_entry) > 128:
            raise ValueError("MCP child declaration is invalid")
        return raw_name, dict(raw_entry)

    @staticmethod
    def _validate_mcp_command(command: Any) -> None:
        """Validate a local child command."""
        if not isinstance(command, str):
            raise ValueError("MCP child declaration is invalid")
        if len(command) > 4_096:
            raise ValueError("MCP child declaration is invalid")
        if "\x00" in command:
            raise ValueError("MCP child declaration is invalid")

    @staticmethod
    def _validate_mcp_args(args: Any) -> None:
        """Validate local child arguments without retaining new references."""
        if not isinstance(args, list):
            raise ValueError("MCP child declaration is invalid")
        if len(args) > 128:
            raise ValueError("MCP child declaration is invalid")
        for value in args:
            if not isinstance(value, str):
                raise ValueError("MCP child declaration is invalid")
            if len(value.encode("utf-8")) > 8_192:
                raise ValueError("MCP child declaration is invalid")
            if "\x00" in value:
                raise ValueError("MCP child declaration is invalid")

    @staticmethod
    def _validate_mcp_env(env: Any) -> None:
        """Validate environment names and bounded values."""
        if not isinstance(env, dict):
            raise ValueError("MCP child declaration is invalid")
        if len(env) > 256:
            raise ValueError("MCP child declaration is invalid")
        for key, value in env.items():
            if not isinstance(key, str):
                raise ValueError("MCP child declaration is invalid")
            if _ENV_NAME.fullmatch(key) is None:
                raise ValueError("MCP child declaration is invalid")
            if len(str(value).encode("utf-8")) > 65_536:
                raise ValueError("MCP child declaration is invalid")

    @staticmethod
    def _validate_mcp_headers(headers: Any) -> None:
        """Validate remote transport headers and bounded values."""
        if not isinstance(headers, dict):
            raise ValueError("MCP child declaration is invalid")
        if len(headers) > 64:
            raise ValueError("MCP child declaration is invalid")
        for key, value in headers.items():
            if not isinstance(key, str):
                raise ValueError("MCP child declaration is invalid")
            if not 1 <= len(key) <= 128:
                raise ValueError("MCP child declaration is invalid")
            if len(str(value).encode("utf-8")) > 16_384:
                raise ValueError("MCP child declaration is invalid")

    @staticmethod
    def _validate_mcp_transport(
        command: Any,
        args: Any,
        env: Any,
        url: Any,
        transport: str,
        headers: Any,
        provider_profile: Any,
    ) -> None:
        """Validate the local/remote transport shape and its shared fields."""
        MCPDiscoveryMixin._validate_mcp_command(command)
        MCPDiscoveryMixin._validate_mcp_args(args)
        MCPDiscoveryMixin._validate_mcp_env(env)
        if not isinstance(url, str):
            raise ValueError("MCP child declaration is invalid")
        if len(url) > 8_192:
            raise ValueError("MCP child declaration is invalid")
        if transport not in {"", "streamable-http", "sse"}:
            raise ValueError("MCP child declaration is invalid")
        MCPDiscoveryMixin._validate_mcp_headers(headers)
        if bool(command) == bool(url):
            raise ValueError("MCP child declaration is invalid")
        if transport and not url:
            raise ValueError("MCP child declaration is invalid")
        if not isinstance(provider_profile, str):
            raise ValueError("MCP child declaration is invalid")
        if (
            provider_profile
            and re.fullmatch(r"[a-z][a-z0-9-]{1,62}", provider_profile) is None
        ):
            raise ValueError("MCP child declaration is invalid")

    @staticmethod
    def _validate_mcp_list(values: Any, *, max_items: int, max_bytes: int) -> None:
        """Validate a bounded list of UTF-8 strings."""
        if not isinstance(values, list):
            raise ValueError("MCP child declaration is invalid")
        if len(values) > max_items:
            raise ValueError("MCP child declaration is invalid")
        for value in values:
            if not isinstance(value, str):
                raise ValueError("MCP child declaration is invalid")
            if len(value.encode("utf-8")) > max_bytes:
                raise ValueError("MCP child declaration is invalid")

    @staticmethod
    def _validate_mcp_metadata(disabled_tools: Any, private_hosts: Any) -> None:
        """Validate tool exclusions and private-host egress declarations."""
        MCPDiscoveryMixin._validate_mcp_list(
            disabled_tools, max_items=2_048, max_bytes=256
        )
        MCPDiscoveryMixin._validate_mcp_list(
            private_hosts, max_items=256, max_bytes=253
        )

    def _normalize_mcp_server(
        self, raw_name: Any, raw_entry: Any
    ) -> dict[str, Any] | None:
        """Validate and normalize one enabled child declaration."""
        name, entry = self._validate_mcp_child(raw_name, raw_entry)
        disabled = entry.get("disabled", False)
        if not isinstance(disabled, bool):
            raise ValueError("MCP child declaration is invalid")
        if disabled:
            return None

        command = entry.get("command", "")
        args = entry.get("args", [])
        env = entry.get("env", {})
        url = entry.get("url", "")
        transport = str(entry.get("transport", "")).lower()
        headers = entry.get("headers", {})
        provider_profile = entry.get("provider_profile", "")
        self._validate_mcp_transport(
            command, args, env, url, transport, headers, provider_profile
        )

        disabled_tools = entry.get("disabledTools", [])
        private_hosts = entry.get("allowed_private_hosts", [])
        self._validate_mcp_metadata(disabled_tools, private_hosts)
        return self._normalized_mcp_server(
            name,
            entry,
            command=command,
            args=args,
            env=env,
            url=url,
            transport=transport,
            headers=headers,
            provider_profile=provider_profile,
            disabled_tools=disabled_tools,
            private_hosts=private_hosts,
        )

    def _normalized_mcp_server(
        self,
        name: str,
        entry: dict[str, Any],
        *,
        command: Any,
        args: Any,
        env: Any,
        url: Any,
        transport: str,
        headers: Any,
        provider_profile: Any,
        disabled_tools: Any,
        private_hosts: Any,
    ) -> dict[str, Any]:
        """Build the public normalized child shape after validation."""
        return {
            "name": name,
            "command": command,
            "args": args,
            "env": env,
            "url": url,
            "transport": transport,
            "headers": headers,
            "provider_profile": provider_profile,
            "tool_flags": self._parse_tool_flags(env),
            "config_hash": self._compute_config_hash(name, entry),
            "disabled_tools": disabled_tools,
            "tls_profile": entry.get("tls_profile", ""),
            "tls_profile_ref": entry.get("tls_profile_ref", ""),
            "allowed_private_hosts": private_hosts,
            "initialization_timeout": entry.get(
                "initialization_timeout", entry.get("timeout", 300.0)
            ),
            "_runtime_materialized_secret_keys": entry.get(
                "_runtime_materialized_secret_keys", []
            ),
            "_runtime_materialization_attestation": entry.get(
                "_runtime_materialization_attestation", ""
            ),
        }

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
    def _compute_config_hash(name: str, declaration: dict[str, Any]) -> str:
        """Return a keyed, opaque freshness identity for one declaration.

        The graph stores only this digest. Location-bearing values are reduced
        to keyed tokens before the outer identity is framed, while sensitive
        fields contribute only their key/presence (never resolved material).
        A production profile uses the configured persistence-identity key; a
        zero-infrastructure development process uses an ephemeral key and will
        conservatively refresh after restart.
        """

        from agent_utilities.security.persistence_privacy import (
            _persistence_identity_key,
        )

        key = _persistence_identity_key() or _EPHEMERAL_FRESHNESS_KEY

        def token(label: str, value: Any) -> str:
            canonical = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            return hmac.new(
                key,
                b"agent-utilities:mcp-freshness-field:v1\x00"
                + label.encode("utf-8")
                + b"\x00"
                + canonical,
                hashlib.sha256,
            ).hexdigest()

        def governed_values(container: Any, *, label: str) -> list[tuple[str, str]]:
            if not isinstance(container, dict):
                return []
            values: list[tuple[str, str]] = []
            for raw_key, value in sorted(container.items()):
                field = str(raw_key)
                marker = (
                    "credential-present"
                    if _SENSITIVE_KEY.search(field.replace("-", "_"))
                    else token(f"{label}:{field}", value)
                )
                values.append((field, marker))
            return values

        payload = {
            "format": "mcp-freshness/v1",
            "server": token("server", name),
            "command": token("command", declaration.get("command", "")),
            "args": token("args", declaration.get("args", [])),
            "environment": governed_values(
                declaration.get("env", {}), label="environment"
            ),
            "url": token("url", declaration.get("url", "")),
            "transport": str(declaration.get("transport", "")).lower(),
            "headers": governed_values(declaration.get("headers", {}), label="header"),
            "tls_profile": token("tls_profile", declaration.get("tls_profile", "")),
            "tls_profile_ref": token(
                "tls_profile_ref", declaration.get("tls_profile_ref", "")
            ),
            "private_hosts": token(
                "private_hosts", declaration.get("allowed_private_hosts", [])
            ),
            "disabled_tools": token(
                "disabled_tools", declaration.get("disabledTools", [])
            ),
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hmac.new(
            key,
            b"agent-utilities:mcp-freshness:v1\x00" + canonical,
            hashlib.sha256,
        ).hexdigest()
