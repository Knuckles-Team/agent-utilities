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
The proven child-server lifecycle, algorithmic collision-free prefixing, and
enable/disable tool filtering are preserved (see :class:`MCPMultiplexer`).

CONCEPT:ECO-4.0 — MCP Standardized Interfaces
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import mcp.types
from fastmcp.tools import FunctionTool, ToolResult
from mcp import StdioServerParameters, stdio_client
from mcp import types as mcp_types  # stable alias: the ``mcp`` param shadows the pkg
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


# Prefixes are derived 100% algorithmically (no per-server lookup table), so any
# MCP server — bundled or third-party — gets a sensible, unique prefix with zero
# code changes. Pin a specific one per server via the ``prefix`` key in its
# mcp_config entry; uniqueness across the fleet is then guaranteed by the
# catalog-aware collision resolver (:meth:`MCPMultiplexer._build_prefix_map`).


# Tokens that carry no identifying signal — stripped before auto-deriving a
# prefix so e.g. "weather-mcp-server" keys off "weather", not "mcp"/"server".
_PREFIX_NOISE_TOKENS = {
    "mcp",
    "server",
    "agent",
    "api",
    "service",
    "srv",
    "tool",
    "tools",
}


def _tokenize_server_name(name: str) -> list[str]:
    """Split a server name into lowercase word tokens, handling separators
    (``-_./:`` etc.) AND camelCase/PascalCase humps (``MyCoolMCP`` →
    my/cool/mcp), so any naming style yields sensible tokens."""
    humped = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return [t.lower() for t in re.split(r"[^A-Za-z0-9]+", humped) if t]


# A trailing host/instance id (r510, rw710, gr1080, …) — kept readable on
# multi-instance servers (e.g. systems-manager-mcp-r510 → sm_r510).
_INSTANCE_ID_RE = re.compile(r"^[a-z]{0,4}\d+[a-z0-9]*$")


def auto_server_prefix(server_name: str) -> str:
    """Algorithmically derive a short, readable prefix for ANY MCP server name —
    no lookup table, so out-of-ecosystem / third-party servers are fully
    supported (CONCEPT:ECO-4.36). Uniqueness across the fleet is guaranteed
    separately by the catalog-aware collision resolver.

    Rules (after dropping noise words mcp/server/agent/api/…):
      • trailing host/instance id → ``<initials>_<id>`` (systems-manager-mcp-r510
        → 'sm_r510'), so multi-instance servers stay distinct and legible;
      • multi-word → initials acronym (container-manager → 'cm', github → 'gith');
      • single-word → a short stem ('leanix' → 'lean')."""
    tokens = _tokenize_server_name(server_name)
    meaningful = [t for t in tokens if t not in _PREFIX_NOISE_TOKENS] or tokens
    if not meaningful:
        return "mcp"
    if len(meaningful) >= 2 and _INSTANCE_ID_RE.match(meaningful[-1]):
        base = meaningful[:-1]
        acronym = "".join(t[0] for t in base) or base[0][:2]
        return f"{acronym}_{meaningful[-1]}"
    if len(meaningful) >= 2:
        acronym = "".join(t[0] for t in meaningful)
        if len(acronym) >= 2:
            return acronym[:5]
    return meaningful[0][:4]


def get_server_prefix(server_name: str, cfg: dict | None = None) -> str:
    """Resolve a server's preferred prefix (uniqueness is enforced later by the
    catalog-aware collision resolver). An explicit ``prefix`` on the server's
    config entry wins; otherwise it is auto-derived from the name."""
    if cfg:
        explicit = cfg.get("prefix")
        if explicit:
            clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(explicit)).strip("_").lower()
            if clean:
                return clean[:10]
    return auto_server_prefix(server_name)


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


def _format_probe_error(exc: BaseException) -> str:
    """Render a probe failure as a concise, useful string. anyio wraps
    transport failures in an ``ExceptionGroup`` whose str() is the opaque
    "unhandled errors in a TaskGroup (N sub-exceptions)" — so unwrap it to the
    distinct leaf causes (e.g. the underlying HTTP 502 / connection error)."""
    leaves: list[str] = []

    def _walk(e: BaseException) -> None:
        subs = getattr(e, "exceptions", None)
        if isinstance(e, BaseExceptionGroup) or (
            subs and isinstance(subs, (list, tuple))
        ):
            for sub in subs:
                _walk(sub)
        else:
            msg = str(e).strip()
            leaves.append(f"{type(e).__name__}: {msg}" if msg else type(e).__name__)

    _walk(exc)
    # de-dup while preserving order
    seen = list(dict.fromkeys(leaves))
    return "; ".join(seen) if seen else f"{type(exc).__name__}: {exc}"


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
        # CONCEPT:ECO-4.36 — dynamic tool gateway state. The catalog is the
        # full set of mountable servers parsed from config WITHOUT spawning
        # them, so find_tools/load_tools know what exists before any child is
        # started. ``_exposed`` tracks prefixed tool names currently registered
        # as live FastMCP tools (so lazy mounts don't double-register).
        self._catalog: dict[str, dict] | None = None
        self._exposed: set[str] = set()
        # CONCEPT:ECO-4.36 — self-catalog: per-server {"tools": [...], "error": str|None}
        # learned by probing each child (connect → list_tools → release), cached so
        # find_tools ranks real fleet-wide tools without holding connections and
        # without depending on the (separately-flaky) KG live discovery.
        self._probe_cache: dict[str, dict] = {}
        # CONCEPT:ECO-4.36 — catalog-aware, collision-free prefix assignment,
        # computed deterministically over the whole server set so similarly
        # named servers (e.g. scholarx/searxng, both preferring "sx") never
        # share a namespace however the fleet scales. Built lazily from catalog.
        self._prefix_map: dict[str, str] | None = None
        self._prefix_reverse: dict[str, str] = {}

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
                    mcp.types.TextContent(type="text", text=f"{type(e).__name__}: {e}")
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

    async def _open_one_session(
        self, server_name: str, cfg: dict, stack: contextlib.AsyncExitStack
    ) -> ClientSession:
        """Open + initialize ONE ``ClientSession`` for a child (stdio or remote),
        entering its transports on ``stack``. Raises on failure. Shared by
        :meth:`_start_child` (session pool) and :meth:`probe_server` (catalog
        probe) so the transport-construction logic lives in one place."""
        command = cfg.get("command")
        url = os.path.expandvars(str(cfg.get("url", "")))
        explicit_transport = str(cfg.get("transport", "")).lower()
        is_remote = bool(url) or explicit_transport in (
            "streamable-http",
            "streamable_http",
            "http",
            "sse",
        )
        if not command and not is_remote:
            raise RuntimeError(
                f"Server '{server_name}' has neither 'command' nor 'url'"
            )

        if is_remote:
            headers = cfg.get("headers")
            if headers:
                headers = {k: os.path.expandvars(str(v)) for k, v in headers.items()}
            use_sse = explicit_transport == "sse" or url.rstrip("/").endswith("/sse")
            if use_sse:
                if sse_client is None:
                    raise RuntimeError("mcp SDK has no sse_client for SSE transport")
                transport = sse_client(url, headers=headers)
            else:
                if streamablehttp_client is None:
                    raise RuntimeError(
                        "mcp SDK has no streamablehttp_client for "
                        "streamable-http transport"
                    )
                transport = streamablehttp_client(url, headers=headers)
            # streamable-http yields (read, write, get_session_id); sse yields
            # (read, write). Take the first two streams either way.
            streams = await stack.enter_async_context(transport)
            read_stream, write_stream = streams[0], streams[1]
        else:
            merged_env = os.environ.copy()
            for k, v in (cfg.get("env") or {}).items():
                merged_env[k] = os.path.expandvars(str(v))
            if "PYTHONPATH" not in merged_env and "PYTHONPATH" in os.environ:
                merged_env["PYTHONPATH"] = os.environ["PYTHONPATH"]
            server_params = StdioServerParameters(
                command=command, args=cfg.get("args", []), env=merged_env
            )
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(server_params)
            )

        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        return session

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

        logger.info(
            "Starting child server '%s' (timeout %ss) via %s: %s",
            server_name,
            timeout,
            "streamable-http/sse" if is_remote else "stdio",
            url if is_remote else f"{command} {' '.join(args)}",
        )

        async def _connect_one(stack: contextlib.AsyncExitStack):
            return await self._open_one_session(server_name, cfg, stack)

        async def _connect(stack: contextlib.AsyncExitStack):
            """One connection generation: full session pool + tool list.

            The stack is owned by the runtime's supervisor task (entered and
            exited there), so each crash/restart cleanly tears down and
            rebuilds every transport of the generation."""
            sessions = [await _connect_one(stack) for _ in range(pool_size)]
            tools_result = await sessions[0].list_tools()
            return sessions, tools_result.tools

        runtime = ChildRuntime(server_name, cfg, connect=_connect)
        try:
            tools = await runtime.start()
        except TimeoutError:
            logger.error(
                f"Failed to start child server '{server_name}': Timeout of {timeout}s exceeded"
            )
            return None
        except Exception as e:
            logger.error(
                f"Failed to start child server '{server_name}': {e}",
                exc_info=True,
            )
            return None

        logger.info(
            "Loaded %d tools from child server '%s' (%d session%s)",
            len(tools),
            server_name,
            pool_size,
            "" if pool_size == 1 else "s",
        )
        return server_name, runtime, tools, cfg

    def load_catalog(self) -> dict[str, dict]:
        """Parse the config once into the mountable-server catalog WITHOUT
        spawning any child (CONCEPT:ECO-4.36).

        Idempotent: the parsed ``{server_name: cfg}`` map is cached on
        ``self._catalog`` and reused. Self (``mcp-multiplexer``) and entries
        flagged ``disabled`` are excluded — they are never mountable.
        """
        if self._catalog is not None:
            return self._catalog

        self._catalog = {}
        if not self.config_path.exists():
            logger.error(f"Config path {self.config_path} does not exist.")
            return self._catalog

        try:
            content = self.config_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read config file {self.config_path}: {e}")
            return self._catalog

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
                return self._catalog

        for server_name, cfg in config_data.get("mcpServers", {}).items():
            # Skip ourselves to avoid self-infinite recursion loops
            if server_name == "mcp-multiplexer" or cfg.get("disabled", False):
                continue
            self._catalog[server_name] = cfg
        return self._catalog

    @staticmethod
    def _server_stem(name: str) -> str:
        """Full cleaned server name used to disambiguate colliding prefixes."""
        clean = "".join(c if (c.isalnum() or c in ("_", "-")) else "_" for c in name)
        return clean.replace("-", "_").lower().strip("_")

    def _build_prefix_map(self) -> None:
        """Assign every catalog server a unique prefix (CONCEPT:ECO-4.36).

        Each server starts from its preferred prefix (:func:`get_server_prefix`
        — config override or auto-derived). Collisions are resolved
        deterministically (sorted order): the first claimant keeps the preferred
        prefix; the rest extend their cleaned stem until unique, falling back to
        a numeric suffix. Guarantees no two servers ever share a prefix, however
        the fleet grows."""
        catalog = self.load_catalog()
        names = sorted(catalog.keys())
        assigned: dict[str, str] = {}
        used: set[str] = set()
        for name in names:
            base = get_server_prefix(name, catalog.get(name))
            if base not in used:
                assigned[name] = base
                used.add(base)
                continue
            stem = self._server_stem(name)
            cand: str | None = None
            for n in range(max(len(base) + 1, 1), len(stem) + 1):
                if stem[:n] not in used:
                    cand = stem[:n]
                    break
            if cand is None:  # stem exhausted — append a numeric suffix
                i = 2
                while f"{base}{i}" in used:
                    i += 1
                cand = f"{base}{i}"
            assigned[name] = cand
            used.add(cand)
        self._prefix_map = assigned
        self._prefix_reverse = {p: n for n, p in assigned.items()}

    def server_prefix(self, server_name: str) -> str:
        """The unique, collision-free prefix for a server (CONCEPT:ECO-4.36).
        Falls back to the bare derived prefix for names outside the catalog."""
        if self._prefix_map is None:
            self._build_prefix_map()
        assert self._prefix_map is not None
        return self._prefix_map.get(server_name) or get_server_prefix(server_name)

    def _tool_enabled(self, server_name: str, tool_name: str) -> bool:
        """Whether a server's tool would actually be exposed on load, applying
        the config's ``enabledTools`` whitelist + ``disabledTools`` blacklist —
        the same filter :meth:`_register_child_result` uses. Lets discovery
        distinguish capable-but-disabled tools from loadable ones."""
        import fnmatch

        cfg = self.load_catalog().get(server_name, {})
        enabled = cfg.get("enabledTools")
        disabled = cfg.get("disabledTools", [])
        if enabled is not None and not any(
            fnmatch.fnmatch(tool_name, pat) for pat in enabled
        ):
            return False
        if disabled and any(fnmatch.fnmatch(tool_name, pat) for pat in disabled):
            return False
        return True

    def _register_child_result(
        self,
        server_name: str,
        payload: Any,
        tools: list[mcp.types.Tool],
        cfg: dict,
    ) -> list[mcp.types.Tool]:
        """Record a freshly started child's runtime, session, and (filtered)
        tools into the aggregation maps. Returns the prefixed ``Tool`` objects
        that were registered for this child.

        Shared by eager :meth:`start_children` and lazy :meth:`mount_child` so
        the enable/disable filtering and prefixing logic lives in one place.
        """
        # The per-child hardening runtime (CONCEPT:ECO-4.34) carries the
        # session pool, concurrency limits, and restart supervisor. Plain
        # session payloads (externally owned connections) are wrapped in a
        # supervisor-less runtime: limits apply, auto-restart does not.
        if isinstance(payload, ChildRuntime):
            runtime = payload
        else:
            sessions = list(payload) if isinstance(payload, list | tuple) else [payload]
            runtime = ChildRuntime(server_name, cfg)
            runtime.adopt_sessions(sessions)
        self.children[server_name] = runtime
        self.sessions[server_name] = runtime.primary_session

        disabled_tools = cfg.get("disabledTools", [])
        enabled_tools = cfg.get("enabledTools", None)

        registered: list[mcp.types.Tool] = []
        for tool in tools:
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

                matched_disabled = any(
                    fnmatch.fnmatch(tool.name, pat) for pat in disabled_tools
                )
                if matched_disabled:
                    logger.info(
                        f"Skipping disabled tool '{tool.name}' from child server '{server_name}'"
                    )
                    continue

            prefix = self.server_prefix(server_name)
            prefixed_name = clean_tool_name(prefix, server_name, tool.name)
            self.tool_to_server[prefixed_name] = (server_name, tool.name)

            prefixed_tool = mcp.types.Tool(
                name=prefixed_name,
                description=tool.description or "",
                inputSchema=tool.inputSchema,
            )
            self.aggregated_tools.append(prefixed_tool)
            registered.append(prefixed_tool)
        return registered

    async def mount_child(self, server_name: str) -> list[mcp.types.Tool]:
        """Start ONE configured child on demand and register its tools
        (CONCEPT:ECO-4.36).

        Idempotent: if the child is already mounted, its already-registered
        prefixed tools are returned without re-spawning. Returns ``[]`` for an
        unknown/unconfigured server. Loop-safe because it is invoked from
        inside the serving event loop (either at boot or from a tool call), so
        the child ``ClientSession`` objects bind to the running loop.
        """
        catalog = self.load_catalog()
        if server_name in self.children:
            return self.prefixed_tools_for_server(server_name)
        cfg = catalog.get(server_name)
        if cfg is None:
            logger.warning(
                "mount_child: server '%s' is not in the catalog", server_name
            )
            return []
        result = await self._start_child(server_name, cfg)
        if not isinstance(result, tuple):
            return []
        s_name, payload, tools, r_cfg = result
        return self._register_child_result(s_name, payload, tools, r_cfg)

    def prefixed_tools_for_server(self, server_name: str) -> list[mcp.types.Tool]:
        """All aggregated prefixed tools currently owned by ``server_name``."""
        names = {
            pn for pn, (srv, _orig) in self.tool_to_server.items() if srv == server_name
        }
        return [t for t in self.aggregated_tools if t.name in names]

    async def start_children(self):
        """Parse configuration and start all child processes concurrently
        (eager mode). Lazy mode uses :meth:`mount_child` instead."""
        catalog = self.load_catalog()
        if not catalog:
            logger.info("No active child servers configured.")
            return

        tasks = [
            self._start_child(server_name, cfg) for server_name, cfg in catalog.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                # An exception propagated from asyncio.gather, already logged inside task
                continue
            if not isinstance(result, tuple):
                continue
            server_name, payload, tools, cfg = result
            self._register_child_result(server_name, payload, tools, cfg)

    # ------------------------------------------------------------------
    # Dynamic tool gateway (CONCEPT:ECO-4.36)
    # ------------------------------------------------------------------

    def _server_for_prefixed(self, prefixed_name: str) -> str | None:
        """Resolve which child server owns a prefixed tool name, even before
        that child is mounted, via the collision-free reverse prefix map.
        Returns None if unknown."""
        if prefixed_name in self.tool_to_server:
            return self.tool_to_server[prefixed_name][0]
        if self._prefix_map is None:
            self._build_prefix_map()
        prefix = prefixed_name.split("__", 1)[0]
        return self._prefix_reverse.get(prefix)

    def _kg_prefixed(self, bare_tool: str) -> str | None:
        """Find the live prefixed name for a knowledge-graph child tool
        (e.g. ``graph_query`` -> ``kg__graph_query``) regardless of how the
        always-on KG server is nicknamed. Requires the KG child to be mounted."""
        suffix = f"__{bare_tool}"
        for prefixed, (_srv, original) in self.tool_to_server.items():
            if original == bare_tool and prefixed.endswith(suffix):
                return prefixed
        return None

    async def _kg_call(self, bare_tool: str, arguments: dict[str, Any]) -> Any:
        """Best-effort call to a knowledge-graph child tool, returning parsed
        JSON (or raw text), or ``None`` if the KG child is unavailable or the
        call fails. Used by discovery so a cold/absent KG degrades gracefully
        rather than raising into the meta-tool."""
        prefixed = self._kg_prefixed(bare_tool)
        if prefixed is None:
            return None
        try:
            result = await self.call_proxied_tool(prefixed, arguments)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("KG call '%s' raised: %s", prefixed, e)
            return None
        if getattr(result, "isError", False):
            return None
        text = "".join(
            getattr(block, "text", "")
            for block in (getattr(result, "content", []) or [])
            if getattr(block, "type", None) == "text"
        )
        if not text:
            return None
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return text

    def _live_tools_for_server(self, server_name: str) -> list[dict]:
        """Raw ``[{name, description, inputSchema}]`` for an already-mounted
        child, reconstructed from the aggregation maps (no reconnect)."""
        out: list[dict] = []
        for prefixed, (srv, original) in self.tool_to_server.items():
            if srv != server_name:
                continue
            tobj = self.tool_object(prefixed)
            out.append(
                {
                    "name": original,
                    "description": (tobj.description if tobj else "") or "",
                    "inputSchema": (tobj.inputSchema if tobj else {}) or {},
                }
            )
        return out

    async def probe_server(
        self, server_name: str, force: bool = False, timeout: float | None = None
    ) -> dict:
        """Probe ONE catalog server for its tool list: connect → list_tools →
        release (CONCEPT:ECO-4.36). Returns (and caches) ``{"tools": [...],
        "error": str|None}``. An already-mounted child reuses its live tools
        instead of reconnecting; an unreachable server records its error string
        (so find_tools/load_tools can report *why* it is unavailable) rather
        than raising."""
        if not force and server_name in self._probe_cache:
            return self._probe_cache[server_name]

        if server_name in self.children:
            info = {"tools": self._live_tools_for_server(server_name), "error": None}
            self._probe_cache[server_name] = info
            return info

        cfg = self.load_catalog().get(server_name)
        if cfg is None:
            info = {"tools": [], "error": "not in catalog"}
            self._probe_cache[server_name] = info
            return info

        probe_to = float(
            timeout
            if timeout is not None
            else cfg.get("probe_timeout", cfg.get("timeout", 10.0))
        )

        async def _probe() -> list[dict]:
            # Enter AND exit the transports within this single coroutine so the
            # anyio cancel scopes are not crossed between tasks. ``wait_for``
            # runs this whole coroutine as one task, so the stack is opened and
            # closed in the same task even on timeout-cancellation. (Wrapping
            # only the connect in wait_for would exit the scope in a different
            # task — "Attempted to exit cancel scope in a different task".)
            async with contextlib.AsyncExitStack() as stack:
                session = await self._open_one_session(server_name, cfg, stack)
                result = await session.list_tools()
                return [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "inputSchema": t.inputSchema or {},
                    }
                    for t in result.tools
                ]

        try:
            tools = await asyncio.wait_for(_probe(), timeout=probe_to)
            info = {"tools": tools, "error": None}
        except TimeoutError:
            info = {"tools": [], "error": f"timeout after {probe_to:g}s"}
        except Exception as e:
            info = {"tools": [], "error": _format_probe_error(e)}
        self._probe_cache[server_name] = info
        return info

    async def probe_catalog(
        self, force: bool = False, timeout: float | None = None
    ) -> dict[str, dict]:
        """Probe every catalog server concurrently (bounded) and cache the
        result, so find_tools can rank the whole fleet's real tools. Cached, so
        only the first call pays the cost; unreachable servers fail fast and are
        recorded, never blocking the reachable ones."""
        catalog = self.load_catalog()
        targets = [s for s in catalog if force or s not in self._probe_cache]
        if not targets:
            return self._probe_cache

        sem = asyncio.Semaphore(16)

        async def _guarded(s: str) -> None:
            async with sem:
                await self.probe_server(s, force=force, timeout=timeout)

        await asyncio.gather(*[_guarded(s) for s in targets], return_exceptions=True)
        return self._probe_cache

    @staticmethod
    def _relevance(query: str, text: str) -> float:
        """Cheap, deterministic token-overlap relevance in [0, 1] — the
        embedding-free backbone so discovery (and its tests) never depend on a
        live model. Semantic scores from the KG are layered on top when present."""
        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not q_tokens:
            return 0.0
        t_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        return len(q_tokens & t_tokens) / len(q_tokens)

    def _server_level_fallback(self) -> list[dict]:
        """When the KG yields no tool-level index (cold/absent KG), still let
        the caller act: surface mountable servers so they can ``load_tools`` by
        server name."""
        out: list[dict] = []
        for server in self.load_catalog():
            out.append(
                {
                    "server": server,
                    "tool": "*",
                    "prefixed_name": None,
                    "description": (
                        f"All tools for '{server}'. KG tool-level discovery "
                        "is unavailable; load the whole server by name."
                    ),
                    "score": 0.0,
                    "mountable": True,
                    "mounted": server in self.children,
                }
            )
        return out

    async def discover_tools(self, query: str, top_k: int | None = None) -> dict:
        """Rank candidate tools across the whole fleet for an NL ``query``
        (CONCEPT:ECO-4.36), without exposing or holding any child.

        Backbone is the self-catalog (:meth:`probe_catalog` — each server's real
        tools, learned by a cached connect→list→release probe), ranked by token
        overlap; KG semantic-search scores are blended in when the KG is warm.
        Returns ``{"results": [...], "unavailable": {server: error}}`` so the
        caller can both pick tools and see which servers couldn't be reached.
        """
        from agent_utilities.core.config import config as agent_config

        if not top_k or top_k <= 0:
            top_k = agent_config.mcp_dynamic_top_k
        catalog = self.load_catalog()
        probe = await self.probe_catalog()

        # Best-effort semantic scores keyed by bare tool name (KG, if warm).
        semantic: dict[str, float] = {}
        search = await self._kg_call(
            "graph_search",
            {"query": query, "mode": "hybrid", "top_k": max(top_k * 3, 20)},
        )
        if isinstance(search, list):
            for hit in search:
                if not isinstance(hit, dict):
                    continue
                name = hit.get("name") or hit.get("tool") or ""
                score = hit.get("score") or hit.get("relevance_score") or 0.0
                if name:
                    try:
                        semantic[name] = max(semantic.get(name, 0.0), float(score))
                    except (ValueError, TypeError):
                        pass

        ranked: list[dict] = []
        unavailable: dict[str, str] = {}
        for server, info in probe.items():
            if info.get("error"):
                unavailable[server] = info["error"]
                continue
            for entry in info.get("tools", []):
                tool = entry["name"]
                # find_tools surfaces only loadable (enabled) tools, so the
                # caller never picks one that load_tools would silently drop.
                # Disabled-but-capable tools remain visible via list_catalog.
                if not self._tool_enabled(server, tool):
                    continue
                desc = entry.get("description", "")
                score = semantic.get(tool, 0.0) + self._relevance(
                    query, f"{tool} {desc}"
                )
                if score <= 0:
                    continue
                prefixed = clean_tool_name(self.server_prefix(server), server, tool)
                ranked.append(
                    {
                        "server": server,
                        "tool": tool,
                        "prefixed_name": prefixed,
                        "description": desc,
                        "score": round(score, 4),
                        "mountable": server in catalog,
                        "mounted": prefixed in self._exposed,
                    }
                )

        ranked.sort(key=lambda r: r["score"], reverse=True)
        results = ranked[:top_k]
        # Nothing matched but reachable servers exist → list them so the caller
        # can still load by server. If every server errored, leave results empty
        # and let ``unavailable`` tell the story.
        if not results and any(not info.get("error") for info in probe.values()):
            results = self._server_level_fallback()
        return {"results": results, "unavailable": unavailable}

    async def list_catalog(self, server: str = "", include_tools: bool = True) -> dict:
        """Browse the WHOLE fleet catalog (CONCEPT:ECO-4.36) — every configured
        server, its tool count + (optionally) tool names, reachability, and
        mount state. This is the flat "show me everything" view; ``find_tools``
        is the semantic "find me the right tool for X" search. Both ride the
        cached self-catalog probe, so the first call may take a few seconds.

        With ``server`` set, drills into one server and returns its full tool
        list with descriptions.
        """
        catalog = self.load_catalog()
        probe = await self.probe_catalog()

        if server:
            if server not in catalog:
                return {"error": f"'{server}' is not in the catalog"}
            info = probe.get(server, {})
            prefix = self.server_prefix(server)
            return {
                "server": server,
                "prefix": prefix,
                "mounted": server in self.children,
                "available": info.get("error") is None,
                "error": info.get("error"),
                "tools": [
                    {
                        "prefixed_name": clean_tool_name(prefix, server, t["name"]),
                        "tool": t["name"],
                        "description": t.get("description", ""),
                        "enabled": self._tool_enabled(server, t["name"]),
                    }
                    for t in info.get("tools", [])
                ],
            }

        servers: list[dict] = []
        for name in catalog:
            info = probe.get(name, {})
            prefix = self.server_prefix(name)
            tool_entries = info.get("tools", [])
            enabled_names: list[str] = []
            disabled_names: list[str] = []
            for t in tool_entries:
                pn = clean_tool_name(prefix, name, t["name"])
                target = (
                    enabled_names
                    if self._tool_enabled(name, t["name"])
                    else disabled_names
                )
                target.append(pn)
            entry = {
                "server": name,
                "prefix": prefix,
                "tool_count": len(tool_entries),
                "enabled_count": len(enabled_names),
                "mounted": name in self.children,
                "available": info.get("error") is None,
            }
            if info.get("error"):
                entry["error"] = info["error"]
            if include_tools:
                entry["tools"] = enabled_names
                if disabled_names:
                    entry["disabled_tools"] = disabled_names
            servers.append(entry)
        return {
            "total_servers": len(servers),
            "total_tools": sum(s["tool_count"] for s in servers),
            "mounted": sorted(self.children.keys()),
            "unavailable": [s["server"] for s in servers if not s["available"]],
            "servers": servers,
        }

    async def resolve_and_mount(
        self,
        tools: list[str] | None = None,
        servers: list[str] | None = None,
    ) -> tuple[list[str], list[str], dict[str, str]]:
        """Mount whatever children are needed to satisfy a ``load_tools``
        request and compute the set of prefixed names to expose.

        Returns ``(mounted_servers, prefixed_names_to_expose, failed)`` where
        ``failed`` maps each server that could not be mounted to a human-readable
        reason (e.g. an unreachable remote server). Does NOT touch FastMCP —
        registration of the live tools (and the list_changed notification) is the
        caller's job so this stays unit-testable.
        """
        requested_tools = list(tools or [])
        target_servers: set[str] = set(servers or [])
        for prefixed in requested_tools:
            owner = self._server_for_prefixed(prefixed)
            if owner:
                target_servers.add(owner)

        mounted: list[str] = []
        failed: dict[str, str] = {}
        for server in sorted(target_servers):
            await self.mount_child(server)
            if server in self.children:
                mounted.append(server)
            else:
                # Mount failed — surface *why* via a targeted probe (cached).
                info = await self.probe_server(server)
                failed[server] = info.get("error") or "could not mount (unreachable?)"

        if requested_tools:
            wanted = set(requested_tools)
        else:
            wanted = set()
            for server in mounted:
                wanted.update(t.name for t in self.prefixed_tools_for_server(server))

        to_expose = [
            name
            for name in sorted(wanted)
            if name in self.tool_to_server and name not in self._exposed
        ]
        return mounted, to_expose, failed

    def tool_object(self, prefixed_name: str) -> mcp.types.Tool | None:
        """The aggregated ``Tool`` object for a prefixed name, if known."""
        for tool in self.aggregated_tools:
            if tool.name == prefixed_name:
                return tool
        return None

    def forget_tool(self, prefixed_name: str) -> str | None:
        """Drop a prefixed tool from the aggregation maps (used by unload).
        Returns the owning server name, if any."""
        owner = None
        mapping = self.tool_to_server.pop(prefixed_name, None)
        if mapping:
            owner = mapping[0]
        self.aggregated_tools = [
            t for t in self.aggregated_tools if t.name != prefixed_name
        ]
        self._exposed.discard(prefixed_name)
        return owner

    def status_snapshot(self) -> dict[str, Any]:
        """Fleet health surface: per-child state, limits, load, restarts."""
        return {
            "children": {
                name: runtime.status()
                for name, runtime in sorted(self.children.items())
            },
            "total_children": len(self.children),
            "total_tools": len(self.aggregated_tools),
        }

    async def aclose(self) -> None:
        """Shut down every child runtime (and any legacy stack registrations)."""
        for runtime in self.children.values():
            await runtime.aclose()
        await self.exit_stack.aclose()


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


def _make_forwarder(mux: MCPMultiplexer, prefixed_name: str):
    """Build the async fn that forwards a prefixed tool call to its child."""

    async def _forward(**kwargs: Any) -> ToolResult:
        result = await mux.call_proxied_tool(prefixed_name, kwargs)
        return ToolResult(
            content=list(getattr(result, "content", []) or []),
            structured_content=getattr(result, "structuredContent", None),
        )

    return _forward


def _register_forwarder(mcp, mux: MCPMultiplexer, tool: mcp.types.Tool) -> bool:
    """Register ONE aggregated child tool as a live FastMCP forwarding tool.

    Idempotent via ``mux._exposed`` so lazy mounts never double-register.
    Returns True if a new tool was added. Shared by eager startup and the
    dynamic ``load_tools`` meta-tool.
    """
    if tool.name in mux._exposed:
        return False
    schema = tool.inputSchema or {"type": "object", "properties": {}}
    mcp.add_tool(
        FunctionTool(
            name=tool.name,
            description=tool.description or "",
            parameters=schema,
            fn=_make_forwarder(mux, tool.name),
        )
    )
    mux._exposed.add(tool.name)
    return True


def _register_status_tool(mcp, mux: MCPMultiplexer) -> None:
    """Register the always-present fleet-health meta-tool (CONCEPT:ECO-4.34)."""

    async def _status() -> ToolResult:
        snapshot = mux.status_snapshot()
        return ToolResult(
            content=[
                mcp_types.TextContent(type="text", text=json.dumps(snapshot, indent=2))
            ],
            structured_content=snapshot,
        )

    mcp.add_tool(
        FunctionTool(
            name="multiplexer_status",
            description=(
                "Health of every aggregated child MCP server: state "
                "(up/restarting/failed), restart count, concurrency "
                "limits, in-flight and queued calls. In dynamic mode also "
                "reflects which children are currently mounted."
            ),
            parameters={"type": "object", "properties": {}},
            fn=_status,
        )
    )


def _register_forwarding_tools(mcp, mux: MCPMultiplexer) -> None:
    """Register a forwarding tool for every aggregated child tool plus the
    status meta-tool. This is the EAGER path (all tools exposed up front)."""
    for tool in mux.aggregated_tools:
        _register_forwarder(mcp, mux, tool)
    _register_status_tool(mcp, mux)


async def _notify_tools_changed(mcp) -> None:
    """Emit ``notifications/tools/list_changed`` so the client re-fetches the
    tool list after a dynamic mount/unmount. Best-effort: a missing request
    context (e.g. no client attached) must not fail the meta-tool."""
    try:
        from fastmcp.server.dependencies import get_context

        await get_context().send_notification(mcp_types.ToolListChangedNotification())
    except Exception as e:  # pragma: no cover - context not always present
        logger.warning("Could not send tools/list_changed notification: %s", e)


def _register_meta_tools(mcp, mux: MCPMultiplexer) -> None:
    """Register the dynamic-gateway meta-tools (CONCEPT:ECO-4.36):
    ``find_tools`` (semantic discovery over the whole fleet), ``list_catalog``
    (flat browse of every server + its tools), ``load_tools`` / ``unload_tools``
    (mount/expose and retract tools at runtime, notifying the client each time),
    plus the status tool."""

    async def _find_tools(query: str, top_k: int = 0) -> ToolResult:
        discovery = await mux.discover_tools(query, top_k=top_k or None)
        results = discovery["results"]
        payload = {
            "query": query,
            "count": len(results),
            "results": results,
            "unavailable": discovery["unavailable"],
        }
        return ToolResult(
            content=[
                mcp_types.TextContent(type="text", text=json.dumps(payload, indent=2))
            ],
            structured_content=payload,
        )

    async def _load_tools(
        tools: list[str] | None = None, servers: list[str] | None = None
    ) -> ToolResult:
        mounted_servers, to_expose, failed = await mux.resolve_and_mount(
            tools=tools, servers=servers
        )
        newly: list[str] = []
        for name in to_expose:
            tool_obj = mux.tool_object(name)
            if tool_obj is not None and _register_forwarder(mcp, mux, tool_obj):
                newly.append(name)
        if newly:
            await _notify_tools_changed(mcp)
        payload = {
            "mounted_servers": mounted_servers,
            "newly_exposed": newly,
            "failed": failed,
            "total_exposed": len(mux._exposed),
        }
        return ToolResult(
            content=[
                mcp_types.TextContent(type="text", text=json.dumps(payload, indent=2))
            ],
            structured_content=payload,
        )

    async def _unload_tools(tools: list[str] | None = None) -> ToolResult:
        removed: list[str] = []
        for name in list(tools or []):
            if name not in mux._exposed:
                continue
            try:
                mcp.local_provider.remove_tool(name)
            except KeyError:
                pass
            mux.forget_tool(name)
            removed.append(name)
        if removed:
            await _notify_tools_changed(mcp)
        payload = {"unloaded": removed, "total_exposed": len(mux._exposed)}
        return ToolResult(
            content=[
                mcp_types.TextContent(type="text", text=json.dumps(payload, indent=2))
            ],
            structured_content=payload,
        )

    async def _list_catalog(server: str = "", include_tools: bool = True) -> ToolResult:
        payload = await mux.list_catalog(server=server, include_tools=include_tools)
        return ToolResult(
            content=[
                mcp_types.TextContent(type="text", text=json.dumps(payload, indent=2))
            ],
            structured_content=payload,
        )

    mcp.add_tool(
        FunctionTool(
            name="find_tools",
            description=(
                "Search the ENTIRE MCP fleet (hundreds of tools across dozens of "
                "servers that are NOT in your current tool list) for the ones "
                "matching a natural-language task. ALWAYS call this FIRST before "
                "concluding a capability is unavailable — most tools are not "
                "loaded yet and only become visible after you load them. Returns "
                "ranked prefixed tool names plus an 'unavailable' map of any "
                "unreachable servers; pass the names you want to load_tools to "
                "make them callable. (Use list_catalog to browse everything.) "
                "The first call probes the fleet (a few seconds); later calls "
                "are cached."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language description of the task or capability needed.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max candidates to return (0 = server default).",
                        "default": 0,
                    },
                },
                "required": ["query"],
            },
            fn=_find_tools,
        )
    )
    mcp.add_tool(
        FunctionTool(
            name="list_catalog",
            description=(
                "Browse the ENTIRE MCP fleet: every configured server with its "
                "tool count, tool names, reachability, and whether it's mounted. "
                "This is the flat 'show me everything available' view (find_tools "
                "is the semantic 'find the right tool for X' search). Pass a "
                "'server' name to drill into just that one and get its full tool "
                "list with descriptions. Then use load_tools to make tools "
                "callable. First call probes the fleet (a few seconds); cached after."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "Optional: a single server to drill into (full tools + descriptions). Empty = list all servers.",
                        "default": "",
                    },
                    "include_tools": {
                        "type": "boolean",
                        "description": "Include each server's tool names in the all-servers view (default true).",
                        "default": True,
                    },
                },
            },
            fn=_list_catalog,
        )
    )
    mcp.add_tool(
        FunctionTool(
            name="load_tools",
            description=(
                "Mount and expose tools at runtime so they become directly "
                "callable. Pass prefixed tool names (from find_tools) via "
                "'tools', and/or whole server names via 'servers' to load all "
                "of a server's tools. Spawns the owning child servers on first "
                "use and notifies the client that the tool list changed. Any "
                "server that can't be reached is reported in the 'failed' map "
                "(with the reason) instead of erroring the whole call."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Prefixed tool names to expose (e.g. 'cnt__cm_container_operations').",
                    },
                    "servers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Server names whose every tool should be exposed (e.g. 'container-manager-mcp').",
                    },
                },
            },
            fn=_load_tools,
        )
    )
    mcp.add_tool(
        FunctionTool(
            name="unload_tools",
            description=(
                "Retract previously loaded tools to reclaim context. Pass the "
                "prefixed tool names to remove; the client is notified that the "
                "tool list changed. Meta-tools and always-on tools are kept."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Prefixed tool names to unload.",
                    }
                },
                "required": ["tools"],
            },
            fn=_unload_tools,
        )
    )
    _register_status_tool(mcp, mux)


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
            "Aggregates many child MCP servers (declared in mcp_config.json) into "
            "one server, tool names namespaced by a short per-server prefix.\n\n"
            "IMPORTANT — in dynamic mode (MCP_MULTIPLEXER_MODE=dynamic) the tools "
            "you see are only a small subset: a few meta-tools plus the always-on "
            "servers. HUNDREDS more tools across dozens of servers exist but are "
            "NOT loaded yet. So whenever you need a capability you don't see in "
            "your current tool list, do NOT assume it's unavailable — use the "
            "meta-tools to reach the rest of the fleet:\n"
            "  • find_tools(query) — semantic search for the right tool by intent\n"
            "  • list_catalog() — browse every server and its tools\n"
            "  • load_tools(tools=[...]] or servers=[...]) — mount the ones you "
            "need; they become directly callable immediately (the tool list "
            "updates live)\n"
            "  • unload_tools(...) — retract tools to reclaim context\n"
            "  • multiplexer_status — health of mounted children\n"
            "Always discover (find_tools/list_catalog) before concluding a tool "
            "doesn't exist."
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
        from agent_utilities.core.config import config as agent_config

        mode = (agent_config.mcp_multiplexer_mode or "eager").lower()
        if mode == "dynamic":
            # CONCEPT:ECO-4.36 — expose only the meta-tools plus the always-on
            # children at boot; everything else is mounted on demand.
            mux.load_catalog()
            always_on = agent_config.mcp_dynamic_always_on or []
            for server_name in always_on:
                tools = await mux.mount_child(server_name)
                for tool in tools:
                    _register_forwarder(mcp, mux, tool)
            _register_meta_tools(mcp, mux)
            logger.info(
                "Dynamic gateway: %d always-on tools from %d child(ren) "
                "(%s) + meta-tools; %d more servers mountable on demand. "
                "Serving over %s.",
                len(mux.aggregated_tools),
                len(mux.children),
                ", ".join(always_on) or "none",
                max(0, len(mux.load_catalog()) - len(mux.children)),
                getattr(args, "transport", "stdio"),
            )
        else:
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
        await mux.aclose()


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
