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

CONCEPT:AU-ECO.mcp.standardized-interfaces — MCP Standardized Interfaces
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import hashlib
import hmac
import importlib.metadata
import json
import logging
import math
import os
import re
import secrets
import stat
import sys
import tempfile
import threading
import weakref
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import mcp.types
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware
from fastmcp.tools import FunctionTool, ToolResult
from mcp import StdioServerParameters, stdio_client
from mcp import types as mcp_types  # stable alias: the ``mcp`` param shadows the pkg
from mcp.client.session import ClientSession

from agent_utilities.core.config import setting
from agent_utilities.mcp.child_resilience import (
    ChildRuntime,
    MCPChildError,
)
from agent_utilities.security.error_surface import public_error_text

streamablehttp_client: Any = None
try:  # remote transports — present on modern mcp SDKs
    from mcp.client.streamable_http import (  # type: ignore[no-redef]
        streamablehttp_client,
    )
except ImportError:  # pragma: no cover - older mcp SDK without streamable-http
    pass
sse_client: Any = None
try:
    from mcp.client.sse import sse_client  # type: ignore[no-redef]
except ImportError:  # pragma: no cover - older mcp SDK without sse
    pass

# Direct all logs to stderr so stdout remains perfectly clean for stdio JSON-RPC
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mcp_multiplexer")
_SESSION_KEY = secrets.token_bytes(32)
_CONFIG_MAX_BYTES = 4 * 1024 * 1024
_MAX_DELEGATED_VALUE_BYTES = 4 * 1024 * 1024
_MAX_DELEGATED_NODES = 16_384
_MAX_DELEGATED_DEPTH = 32
_MAX_DISCOVERED_TOOLS = 2_048
_RUNTIME_CHILD_POLICY_GROUP = "agent_utilities.mcp_child_policies"
_RUNTIME_CHILD_POLICY_RE = re.compile(r"^[a-z][a-z0-9-]{1,62}$")
_RUNTIME_CHILD_POLICY_TRANSPORT_KEYS = frozenset(
    {
        "args",
        "command",
        "env",
        "headers",
        "provider_profile",
        "tls_profile",
        "tls_profile_ref",
        "transport",
        "url",
    }
)
_RUNTIME_CHILD_POLICY_INTERNAL_KEY = "_runtime_child_policy"
_LIVE_MULTIPLEXERS: weakref.WeakSet[Any] = weakref.WeakSet()
_ENV_RUNTIME_REF_RE = re.compile(r"^env://([A-Z][A-Z0-9_]{0,127})$")
_STORE_RUNTIME_REF_RE = re.compile(
    r"^(?:vault|secret)://[A-Za-z0-9][A-Za-z0-9_./#-]{0,511}$"
)
_RUNTIME_REF_RE = re.compile(
    r"^(?:env://[A-Z][A-Z0-9_]{0,127}|"
    r"(?:vault|secret)://[A-Za-z0-9][A-Za-z0-9_./#-]{0,511})$"
)
_ENV_TEMPLATE_RE = re.compile(r"\$\{(?:env:)?([A-Za-z_][A-Za-z0-9_]*)\}")
_SENSITIVE_CONFIG_KEY_RE = re.compile(
    r"(?:^|_)(?:AUTHORIZATION|COOKIE|CREDENTIAL|PASSWORD|SECRET|TOKEN|API_KEY|HMAC_KEY)(?:_|$)",
    re.IGNORECASE,
)
_CHILD_ENV_ALLOWLIST = frozenset(
    {
        "AGENT_UTILITIES_CONFIG_DIR",
        "COMSPEC",
        "APPDATA",
        "HOME",
        "LANG",
        "LC_ALL",
        "LOCALAPPDATA",
        "NO_PROXY",
        "PATH",
        "PATHEXT",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "UV_NATIVE_TLS",
        "WINDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "XDG_STATE_HOME",
    }
)
_PROVIDER_CHILD_ENV_KEYS = frozenset({"AGENT_PROVIDER_PROFILE", "PROVIDER_CONFIGS"})
_PROVIDER_RESOLUTION_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="provider-runtime",
)
_PROVIDER_RESOLUTION_CAPACITY = threading.BoundedSemaphore(8)
_LANGFUSE_PARENT_INGEST_ACTIONS = frozenset(
    {
        "observations_get_many",
        "scores_get_many",
        "sessions_list",
        "trace_get",
        "trace_list",
    }
)


def _sensitive_config_key(name: str) -> bool:
    normalized = str(name).replace("-", "_")
    return bool(_SENSITIVE_CONFIG_KEY_RE.search(normalized))


def _externalized_value(value: Any) -> bool:
    rendered = str(value or "")
    return bool(
        _RUNTIME_REF_RE.fullmatch(rendered) or _ENV_TEMPLATE_RE.search(rendered)
    )


def _resolve_fleet_runtime_reference(reference: str) -> str:
    """Resolve one fleet reference with direct aliases taking precedence.

    ``env://ALIAS`` first reads the live environment projection, which includes
    the runtime-secrets source. Only an unavailable direct alias consults the
    validated ``AgentConfig.mcp_fleet_secret_refs`` mapping. Store references
    resolve through the central runtime secret resolver. No resolved value is
    retained in the catalog or configuration model.
    """

    env_match = _ENV_RUNTIME_REF_RE.fullmatch(reference)
    if env_match is not None:
        alias = env_match.group(1)
        direct = setting(alias)
        if direct not in (None, ""):
            return str(direct)

        from agent_utilities.core.config import config as agent_config

        mappings = getattr(agent_config, "mcp_fleet_secret_refs", None)
        if not isinstance(mappings, dict):
            raise RuntimeError("MCP fleet secret alias mapping is invalid")
        mapped_reference = mappings.get(alias)
        if mapped_reference in (None, ""):
            raise RuntimeError("MCP child runtime reference is unavailable")
        reference = str(mapped_reference)
    elif _STORE_RUNTIME_REF_RE.fullmatch(reference) is None:
        raise RuntimeError("MCP child runtime reference is invalid")

    from agent_utilities.security.cli_secrets import (
        resolve_runtime_secret_reference,
    )

    try:
        return resolve_runtime_secret_reference(reference)
    except Exception:
        raise RuntimeError("MCP child runtime reference is unavailable") from None


def _resolve_runtime_value(
    value: Any,
    *,
    sensitive: bool,
    materialized: bool = False,
) -> str:
    """Resolve externalized catalog values without ever expanding the whole file."""
    rendered = str(value or "")
    if _RUNTIME_REF_RE.fullmatch(rendered):
        rendered = _resolve_fleet_runtime_reference(rendered)
    elif rendered.startswith(("env://", "vault://", "secret://")):
        raise RuntimeError("MCP child runtime reference is invalid")
    else:
        had_template = bool(_ENV_TEMPLATE_RE.search(rendered))

        def replace(match: re.Match[str]) -> str:
            from agent_utilities.core.config import setting

            variable = match.group(1)
            if not sensitive and _sensitive_config_key(variable):
                raise RuntimeError(
                    "Sensitive MCP runtime values require a credential field"
                )
            resolved = setting(variable)
            if resolved in (None, ""):
                raise RuntimeError("MCP child environment reference is unavailable")
            return str(resolved)

        rendered = _ENV_TEMPLATE_RE.sub(replace, rendered)
        if sensitive and not (materialized or had_template):
            raise RuntimeError("MCP child credentials must use runtime references")
    if "\x00" in rendered or "\r" in rendered or "\n" in rendered:
        raise RuntimeError("MCP child runtime value is invalid")
    return rendered


def _assert_bounded_delegated_value(value: Any) -> None:
    """Reject oversized or excessively nested MCP tool arguments."""
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    byte_count = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_DELEGATED_NODES or depth > _MAX_DELEGATED_DEPTH:
            raise ToolError("MCP tool arguments exceed the structural boundary")
        if current is None or isinstance(current, bool | int):
            byte_count += 16
        elif isinstance(current, float):
            if not math.isfinite(current):
                raise ToolError("MCP tool arguments must contain finite numbers")
            byte_count += 16
        elif isinstance(current, str):
            byte_count += len(current.encode("utf-8"))
        elif isinstance(current, list):
            if len(current) > 4_096:
                raise ToolError("MCP tool arguments exceed the collection boundary")
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, dict):
            if len(current) > 4_096:
                raise ToolError("MCP tool arguments exceed the collection boundary")
            for key, item in current.items():
                if not isinstance(key, str) or len(key.encode("utf-8")) > 1_024:
                    raise ToolError("MCP tool argument keys are invalid")
                byte_count += len(key.encode("utf-8"))
                stack.append((item, depth + 1))
        else:
            raise ToolError("MCP tool arguments must be JSON-compatible")
        if byte_count > _MAX_DELEGATED_VALUE_BYTES:
            raise ToolError("MCP tool arguments exceed the size boundary")


def _child_result_payload(result: Any) -> Any:
    """Decode one bounded child result without retaining or logging its body."""

    value = getattr(result, "structuredContent", None)
    if value in (None, {}):
        value = getattr(result, "structured_content", None)
    if value not in (None, {}):
        if isinstance(value, dict) and set(value) == {"result"}:
            value = value["result"]
        if isinstance(value, str):
            if len(value.encode("utf-8")) > _MAX_DELEGATED_VALUE_BYTES:
                raise ToolError("MCP child result exceeds the size boundary")
            value = json.loads(value)
        _assert_bounded_delegated_value(value)
        return value

    texts = [
        str(getattr(item, "text", ""))
        for item in (getattr(result, "content", None) or [])
        if getattr(item, "text", "")
    ]
    rendered = "\n".join(texts)
    if not rendered or len(rendered.encode("utf-8")) > _MAX_DELEGATED_VALUE_BYTES:
        raise ToolError("MCP child result is unavailable for parent ingestion")
    value = json.loads(rendered)
    _assert_bounded_delegated_value(value)
    return value


def _mediate_langfuse_kg_ingestion(
    *,
    child_config: dict[str, Any],
    original_name: str,
    arguments: dict[str, Any],
    result: Any,
) -> None:
    """Persist a Langfuse read only under GraphOS's verified parent authority."""

    from agent_utilities.observability.langfuse_trust import (
        langfuse_parent_kg_ingestion_enabled,
    )

    if not langfuse_parent_kg_ingestion_enabled(child_config):
        return
    if not _runtime_materialized(child_config):
        raise ToolError("Langfuse parent ingestion declaration is not attested")
    if original_name != "langfuse_observability":
        return
    action = str(arguments.get("action") or "")
    if action not in _LANGFUSE_PARENT_INGEST_ACTIONS:
        return
    if bool(getattr(result, "isError", False)) or bool(
        getattr(result, "is_error", False)
    ):
        raise ToolError("Langfuse read failed before parent ingestion")

    # No process identity, child claim, or caller field is accepted here. The
    # graph write inherits the GraphOS request's already-verified session and
    # additionally requires its explicit write scope.
    from agent_utilities.knowledge_graph.core.session import resolve_session

    resolve_session(required_scope="kg:write")
    payload = _child_result_payload(result)
    from langfuse_agent.kg_ingest import ingest_read_result

    ingest_read_result(action, payload)


def _bounded_tool_catalog(raw_tools: Any) -> list[dict[str, Any]]:
    """Project one child catalog into a bounded, JSON-compatible shape.

    The MCP SDK decodes remote responses before returning them to us.  This
    boundary prevents a child from turning that decoded value into an
    unbounded in-memory/KG catalog: tool count, aggregate bytes, nesting,
    collection size, names, descriptions, schemas, and annotations are all
    validated before any caller can cache or register them.
    """

    if (
        not isinstance(raw_tools, list | tuple)
        or len(raw_tools) > _MAX_DISCOVERED_TOOLS
    ):
        raise RuntimeError("MCP child tool catalog exceeded its boundary")
    tools: list[dict[str, Any]] = []
    for tool in raw_tools:
        name = getattr(tool, "name", None)
        description = getattr(tool, "description", "") or ""
        input_schema = getattr(tool, "inputSchema", None) or {}
        annotations = getattr(tool, "annotations", None)
        if annotations is not None and not isinstance(annotations, dict):
            model_dump = getattr(annotations, "model_dump", None)
            if not callable(model_dump):
                raise RuntimeError("MCP child tool catalog is invalid")
            annotations = model_dump(mode="json")
        if (
            not isinstance(name, str)
            or not 1 <= len(name.encode("utf-8")) <= 256
            or any(ord(character) < 32 for character in name)
            or not isinstance(description, str)
            or not isinstance(input_schema, dict)
            or (annotations is not None and not isinstance(annotations, dict))
        ):
            raise RuntimeError("MCP child tool catalog is invalid")
        item: dict[str, Any] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
        }
        if annotations is not None:
            item["annotations"] = annotations
        tools.append(item)
    try:
        _assert_bounded_delegated_value(tools)
    except ToolError:
        raise RuntimeError("MCP child tool catalog exceeded its boundary") from None
    return tools


def _read_catalog_text(path: Path) -> str:
    """Read one regular, non-symlink catalog through a bounded file descriptor."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow and path.is_symlink():
        raise RuntimeError("MCP catalog symlinks are not accepted")
    descriptor = os.open(path, flags | nofollow)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("MCP catalog must be a regular file")
        if metadata.st_size > _CONFIG_MAX_BYTES:
            raise RuntimeError("MCP catalog exceeds its size boundary")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65_536, _CONFIG_MAX_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _CONFIG_MAX_BYTES:
                raise RuntimeError("MCP catalog exceeds its size boundary")
        return b"".join(chunks).decode("utf-8")
    finally:
        os.close(descriptor)


def _validate_externalized_child_secrets(cfg: dict[str, Any]) -> None:
    """Reject credentials committed inline in a persistent child catalog."""
    for container_name in ("env", "headers"):
        values = cfg.get(container_name) or {}
        if not isinstance(values, dict):
            raise RuntimeError("MCP child configuration is invalid")
        for key, value in values.items():
            if container_name == "env" and str(key).upper() in _PROVIDER_CHILD_ENV_KEYS:
                raise RuntimeError(
                    "MCP child provider selection must use provider_profile"
                )
            if _sensitive_config_key(str(key)) and not _externalized_value(value):
                raise RuntimeError("MCP child credentials must use runtime references")


def _selected_child_provider_profile(
    cfg: dict[str, Any], *, is_remote: bool
) -> str | None:
    """Validate one stdio child's deployment-owned provider profile."""

    selected = cfg.get("provider_profile")
    if selected in (None, ""):
        return None
    profile_name = str(selected).strip()
    if (
        profile_name != selected
        or re.fullmatch(r"[a-z][a-z0-9-]{1,62}", profile_name) is None
        or is_remote
    ):
        raise RuntimeError("MCP child provider profile selection is invalid")
    return profile_name


def _materialization_attestation(cfg: dict[str, Any]) -> str:
    """Authenticate one complete process-local child declaration.

    The secret-bearing environment and headers must not be separable from the
    executable, transport destination, trust policy, or parent-only controls
    that consume them.  Signing the complete declaration also makes any
    post-materialization mutation fail closed at the child boundary.
    """

    payload = {
        key: value
        for key, value in cfg.items()
        if key != "_runtime_materialization_attestation"
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hmac.new(_SESSION_KEY, canonical, hashlib.sha256).hexdigest()


def _runtime_materialized(cfg: dict[str, Any]) -> bool:
    """Return whether a runtime-only secret payload was minted in this process."""

    supplied = str(cfg.get("_runtime_materialization_attestation") or "")
    return bool(supplied) and secrets.compare_digest(
        supplied,
        _materialization_attestation(cfg),
    )


def attest_runtime_child_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Mark one in-process child config as safely runtime-materialized.

    Persistent fleet catalogs must carry references rather than credential
    values.  A caller that builds a child config directly from AgentConfig has
    already resolved those references in this process, so it must bind the
    full executable/transport declaration, parent-only controls, sensitive
    field names, and payload to the same per-process attestation used by
    :meth:`MCPMultiplexer.load_catalog` before mounting the child.
    """
    prepared = dict(cfg)
    sensitive_keys = {
        str(key)
        for container_name in ("env", "headers")
        for key in (prepared.get(container_name) or {})
        if _sensitive_config_key(str(key))
    }
    prepared["_runtime_materialized_secret_keys"] = sorted(sensitive_keys)
    prepared["_runtime_materialization_attestation"] = _materialization_attestation(
        prepared
    )
    return prepared


def _load_runtime_child_policy_factory(name: str) -> Any:
    """Load one unambiguous, installed child-policy factory."""

    try:
        matches = tuple(
            importlib.metadata.entry_points(
                group=_RUNTIME_CHILD_POLICY_GROUP,
                name=name,
            )
        )
    except Exception:
        raise RuntimeError("MCP child runtime policy registry is unavailable") from None
    if len(matches) != 1:
        raise RuntimeError("MCP child runtime policy is unavailable")
    try:
        factory = matches[0].load()
    except Exception:
        raise RuntimeError("MCP child runtime policy is unavailable") from None
    if not callable(factory):
        raise RuntimeError("MCP child runtime policy is invalid")
    return factory


def _close_runtime_child_policy(policy: Any) -> None:
    """Close one policy without exposing provider-owned teardown details."""

    try:
        policy.close()
    except Exception:
        logger.error("MCP child runtime policy cleanup failed")


def _prepare_runtime_child_policy(cfg: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Resolve and materialize one optional provider-neutral child policy."""

    selected = cfg.get("runtime_policy")
    if selected in (None, ""):
        return cfg, None
    policy_name = str(selected)
    profile_name = cfg.get("provider_profile")
    if (
        policy_name != selected
        or _RUNTIME_CHILD_POLICY_RE.fullmatch(policy_name) is None
        or not isinstance(profile_name, str)
        or profile_name != profile_name.strip()
        or _RUNTIME_CHILD_POLICY_RE.fullmatch(profile_name) is None
        or any(
            key in cfg
            for key in _RUNTIME_CHILD_POLICY_TRANSPORT_KEYS - {"provider_profile"}
        )
    ):
        raise RuntimeError("MCP child runtime policy selection is invalid")

    from agent_utilities.core.config import config as agent_config

    factory = _load_runtime_child_policy_factory(policy_name)
    try:
        policy = factory(profile_name=profile_name, config=agent_config)
        required = (
            "child_environment",
            "close",
            "fingerprint_catalog",
            "allows_tool",
            "transport_config",
            "verify_before_spawn",
        )
        if not all(callable(getattr(policy, method, None)) for method in required):
            raise RuntimeError("MCP child runtime policy is invalid")
        transport = policy.transport_config()
        if (
            not isinstance(transport, dict)
            or not transport
            or len(transport) > 16
            or not set(transport).issubset(
                _RUNTIME_CHILD_POLICY_TRANSPORT_KEYS - {"provider_profile"}
            )
            or bool(transport.get("command")) == bool(transport.get("url"))
        ):
            raise RuntimeError("MCP child runtime policy transport is invalid")
        prepared = {
            key: value
            for key, value in cfg.items()
            if key not in {"provider_profile", "runtime_policy"}
        }
        prepared.update(transport)
        prepared[_RUNTIME_CHILD_POLICY_INTERNAL_KEY] = policy
        if len(prepared) > 128:
            raise RuntimeError("MCP child runtime policy transport is invalid")
        return prepared, policy
    except Exception:
        if "policy" in locals():
            _close_runtime_child_policy(policy)
        raise RuntimeError("MCP child runtime policy is unavailable") from None


def _request_capabilities() -> frozenset[str] | None:
    """Return verified remote capabilities, or ``None`` for local stdio."""
    try:
        from fastmcp.server.dependencies import get_access_token, get_http_request

        get_http_request()
    except RuntimeError:
        return None
    except Exception:
        raise ToolError("Authenticated HTTP context required") from None
    token = get_access_token()
    if token is None:
        raise ToolError("Authenticated HTTP context required")
    capabilities = {
        str(scope).strip()
        for scope in (getattr(token, "scopes", None) or [])
        if str(scope).strip()
    }
    claims = getattr(token, "claims", None)
    if isinstance(claims, dict):
        try:
            from agent_utilities.core.config import config
            from agent_utilities.security.identity import (
                base_capabilities,
                normalize_identity,
            )

            capabilities.update(
                base_capabilities(
                    normalize_identity(claims),
                    config.identity_group_capability_map,
                )
            )
        except Exception:
            raise ToolError("Verified capability mapping unavailable") from None
    return frozenset(capabilities)


def _require_fleet_capability(kind: str, extra_scopes: list[str] | None = None) -> None:
    """Authorize remote fleet discovery/delegation; local stdio is trusted."""
    capabilities = _request_capabilities()
    if capabilities is None:
        return
    administrative = {"admin", "kg:admin", "mcp:admin"}
    required = (
        {"mcp:discover", "mcp:delegate", *administrative}
        if kind == "discover"
        else {"mcp:delegate", *administrative}
    )
    if not capabilities.intersection(required):
        raise ToolError(f"MCP fleet {kind} capability required")
    if (
        extra_scopes
        and not capabilities.intersection(administrative)
        and not set(extra_scopes).issubset(capabilities)
    ):
        raise ToolError("Child MCP capability scope required")


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


# A trailing neutral instance id (edge101, zone202, …) is kept readable on
# multi-instance servers (e.g. systems-manager-mcp-edge101 → sm_edge101).
_INSTANCE_ID_RE = re.compile(r"^[a-z]{0,4}\d+[a-z0-9]*$")


def auto_server_prefix(server_name: str) -> str:
    """Algorithmically derive a short, readable prefix for ANY MCP server name —
    no lookup table, so out-of-ecosystem / third-party servers are fully
    supported (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog). Uniqueness across the fleet is guaranteed
    separately by the catalog-aware collision resolver.

    Rules (after dropping noise words mcp/server/agent/api/…):
      • trailing host/instance id → ``<initials>_<id>`` (systems-manager-mcp-edge101
        → 'sm_edge101'), so multi-instance servers stay distinct and legible;
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


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two dense vectors (dependency-free; find_tools embeds only
    a handful of short texts, so a pure-Python dot product is cheaper than pulling in a
    numeric dep here)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = sum(x * x for x in a[:n]) ** 0.5
    nb = sum(x * x for x in b[:n]) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _format_probe_error(exc: BaseException) -> str:
    """Render leaf exception types **and their messages** from a probe failure.

    Reporting only ``type(exc).__name__`` (the prior behavior) collapses every
    distinct failure — a DNS error, a TLS failure, and a JWT issuer mismatch —
    into the same bare "RuntimeError", making `find_tools`/`load_tools`/
    `probe_catalog` undiagnosable from the caller's side. The leaf message is
    the actual signal (e.g. an auth server's "issuer mismatch (got X, expected
    Y)"), so it is included, truncated to a bounded length per leaf so one
    verbose exception can't blow out the aggregate catalog response."""
    leaves: list[str] = []

    def _walk(e: BaseException) -> None:
        subs = getattr(e, "exceptions", None)
        if isinstance(e, BaseExceptionGroup) or (
            subs and isinstance(subs, list | tuple)
        ):
            for sub in subs or []:
                _walk(sub)
        else:
            name = type(e).__name__
            msg = str(e).strip()
            leaves.append(f"{name}: {msg[:300]}" if msg else name)

    _walk(exc)
    # de-dup while preserving order
    seen = list(dict.fromkeys(leaves))
    return "; ".join(seen) if seen else type(exc).__name__


def _close_abandoned_provider_projection(
    task: concurrent.futures.Future[Any],
) -> None:
    """Erase a provider projection that completed after its caller left."""

    if task.cancelled():
        return
    try:
        projection = task.result()
    except Exception:
        return
    projection.close()


def _provider_child_sandbox_environment(
    stack: contextlib.AsyncExitStack,
) -> dict[str, str]:
    """Return a private empty home/XDG tree for one provider child."""

    sandbox = tempfile.TemporaryDirectory(prefix="agent-provider-child-")
    stack.callback(sandbox.cleanup)
    root = Path(sandbox.name)
    roots = {
        "config": root / "config",
        "data": root / "data",
        "state": root / "state",
        "cache": root / "cache",
        "runtime": root / "runtime",
        "temp": root / "temp",
    }
    for path in (root, *roots.values()):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            path.chmod(0o700)
        except OSError:
            pass
    return {
        "HOME": str(root),
        "USERPROFILE": str(root),
        "APPDATA": str(roots["config"]),
        "LOCALAPPDATA": str(roots["data"]),
        "XDG_CONFIG_HOME": str(roots["config"]),
        "XDG_DATA_HOME": str(roots["data"]),
        "XDG_STATE_HOME": str(roots["state"]),
        "XDG_CACHE_HOME": str(roots["cache"]),
        "XDG_RUNTIME_DIR": str(roots["runtime"]),
        "AGENT_UTILITIES_CONFIG_DIR": str(roots["config"]),
        "AGENT_UTILITIES_DATA_DIR": str(roots["data"]),
        "AGENT_UTILITIES_CACHE_DIR": str(roots["cache"]),
        "TEMP": str(roots["temp"]),
        "TMP": str(roots["temp"]),
    }


class MCPMultiplexer:
    """Aggregates and proxies multiple MCP servers over a single stdio connection."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.exit_stack = contextlib.AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        # Per-child hardening layer (CONCEPT:AU-ECO.mcp.profile-differences-from-client): concurrency limits and
        # bounded queueing live on the ChildRuntime, not the raw session.
        self.children: dict[str, ChildRuntime] = {}
        self._child_runtime_policies: dict[str, Any] = {}
        self._child_policy_admitted_tools: dict[str, frozenset[str]] = {}
        self._child_catalog_fingerprints: dict[str, str] = {}
        self.tool_to_server: dict[
            str, tuple[str, str]
        ] = {}  # prefixed_name -> (server_name, original_name)
        self.aggregated_tools: list[mcp.types.Tool] = []
        # CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog — dynamic tool gateway state. The catalog is the
        # full set of mountable servers parsed from config WITHOUT spawning
        # them, so find_tools/load_tools know what exists before any child is
        # started. ``_exposed`` tracks prefixed tool names currently registered
        # as live FastMCP tools (so lazy mounts don't double-register).
        self._catalog: dict[str, dict] | None = None
        # ``_exposed`` tracks prefixed tools registered as live FastMCP tools
        # (process-global, so lazy mounts don't double-register). Visibility,
        # however, is PER-SESSION on a shared HTTP server (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog,
        # plan Phase 5): a forwarder is registered once but only listed/callable
        # for sessions that have ``load_tools``-ed it. ``_session_loaded`` maps a
        # session id -> the prefixed names that session has loaded; meta-tools and
        # always-on tools live in ``_global_visible`` and are shown to everyone.
        self._exposed: set[str] = set()
        self._session_loaded: dict[str, set[str]] = {}
        self._global_visible: set[str] = set()
        # CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog — self-catalog: per-server {"tools": [...], "error": str|None}
        # learned by probing each child (connect → list_tools → release), cached so
        # find_tools ranks real fleet-wide tools without holding connections and
        # without depending on the (separately-flaky) KG live discovery.
        self._probe_cache: dict[str, dict] = {}
        # Optional in-process embedder for SEMANTIC find_tools ranking (injected by
        # graph-os via attach_fleet_loader). ``_embed_fn(texts)->list[vector]`` (sync,
        # called off-thread); per-tool embeddings are cached by ``server::tool`` so only
        # the query is embedded per call. Absent ⇒ token-overlap ranking only.
        self._embed_fn: Any = None
        self._tool_embeddings: dict[str, list[float]] = {}
        # Server names never mountable as a child of this multiplexer (self +
        # retired aliases) — set post-construction by the graph-os fleet loader
        # (:func:`attach_fleet_loader`); defaults to just "mcp-multiplexer" via
        # the ``getattr(..., None) or {...}`` fallback in ``load_catalog``.
        self._skip_servers: set[str] | None = None
        # CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog — catalog-aware, collision-free prefix assignment,
        # computed deterministically over the whole server set so similarly
        # named servers (e.g. scholarx/searxng, both preferring "sx") never
        # share a namespace however the fleet scales. Built lazily from catalog.
        self._prefix_map: dict[str, str] | None = None
        self._prefix_reverse: dict[str, str] = {}
        # CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse (Seam 8) — the HOST server's OWN
        # granular tools held back by ``MCP_TOOL_MODE=intent`` (seeded from
        # ``verbose_tools.gated_tool_names`` post-build). Unlike ``_exposed``
        # (fleet forwarders that must be MOUNTED) these are already registered
        # local FastMCP tools that only need a session-visibility flip — no
        # child process, no ``resolve_and_mount``. ``load_tools`` reveals them
        # the same way it reveals a fleet tool: add to ``_session_loaded``.
        self._local_gated: set[str] = set()
        # CONCEPT:AU-ECO.mcp.intent-surface-tool-lifecycle (Seam 8) — session -> tool names
        # ``load_tools(..., auto_unload=True)`` marked for automatic retraction the
        # NEXT time they're called (one-shot: load -> use -> auto-unload), so a
        # long session's tool surface doesn't monotonically grow.
        self._auto_unload: dict[str, set[str]] = {}
        self._catalog_reload_tasks: set[asyncio.Task[Any]] = set()
        self._authority_scope: Any = None
        _LIVE_MULTIPLEXERS.add(self)

    async def call_proxied_tool(
        self, prefixed_name: str, arguments: dict[str, Any] | None = None
    ) -> mcp.types.CallToolResult:
        """Forward a prefixed tool call to the owning child server's session.

        Looks up the ``(server_name, original_name)`` mapping recorded during
        :meth:`start_children` and forwards the call to that child's live
        ``ClientSession``. Raises if the tool/server is unknown or inactive.
        """
        _require_fleet_capability("delegate")
        logger.info("Calling delegated MCP tool")
        _assert_bounded_delegated_value(arguments or {})
        if prefixed_name not in self.tool_to_server:
            raise ValueError("Tool is not registered in multiplexer")

        server_name, original_name = self.tool_to_server[prefixed_name]
        child_config = self.load_catalog().get(server_name)
        if child_config is None:
            raise ToolError("MCP child is no longer enabled")
        configured_scopes = child_config.get("required_scopes", [])
        if isinstance(configured_scopes, str):
            configured_scopes = configured_scopes.split()
        if not isinstance(configured_scopes, list) or not all(
            isinstance(scope, str) and 1 <= len(scope) <= 128
            for scope in configured_scopes
        ):
            raise ToolError("Child MCP scope configuration is invalid")
        _require_fleet_capability("delegate", configured_scopes)
        runtime = self.children.get(server_name)
        if runtime is None:
            raise RuntimeError("MCP child session is not active")
        policy = self._child_runtime_policies.get(server_name)
        if (
            policy is not None
            and original_name
            not in self._child_policy_admitted_tools.get(server_name, frozenset())
        ):
            raise ToolError("MCP child tool is not admitted by runtime policy")

        try:
            # Forward the call through the child's hardened runtime
            # (per-server concurrency limit + bounded queue).
            result = await runtime.call_tool(original_name, arguments or {})
            _mediate_langfuse_kg_ingestion(
                child_config=child_config,
                original_name=original_name,
                arguments=arguments or {},
                result=result,
            )
            return result
        except MCPChildError as e:
            # Typed per-child failure (busy/restarting/failed/circuit-open):
            # surface the error class name so callers can branch on it.
            logger.warning(
                "Child tool call rejected (exception_type=%s)", type(e).__name__
            )
            return mcp.types.CallToolResult(
                content=[mcp.types.TextContent(type="text", text=type(e).__name__)],
                isError=True,
            )
        except Exception as e:
            return mcp.types.CallToolResult(
                content=[mcp.types.TextContent(type="text", text=public_error_text(e))],
                isError=True,
            )

    def _admit_runtime_policy_tools(
        self,
        server_name: str,
        policy: Any,
        tools: list[mcp.types.Tool],
    ) -> list[mcp.types.Tool]:
        """Apply live catalog admission and fingerprinting fail closed."""

        catalog = [
            {
                "annotations": getattr(tool, "annotations", None),
                "inputSchema": tool.inputSchema,
                "name": tool.name,
            }
            for tool in tools
        ]
        try:
            fingerprint = policy.fingerprint_catalog(catalog)
            if not isinstance(fingerprint, str) or not re.fullmatch(
                r"[0-9a-f]{64}", fingerprint
            ):
                raise RuntimeError("invalid fingerprint")
            admitted = [
                tool
                for tool in tools
                if policy.allows_tool(tool.name, getattr(tool, "annotations", None))
            ]
        except Exception:
            raise RuntimeError(
                "MCP child runtime policy rejected live catalog"
            ) from None
        self._child_catalog_fingerprints[server_name] = fingerprint
        self._child_policy_admitted_tools[server_name] = frozenset(
            tool.name for tool in admitted
        )
        return admitted

    async def _open_one_session(
        self, server_name: str, cfg: dict, stack: contextlib.AsyncExitStack
    ) -> ClientSession:
        """Open + initialize ONE ``ClientSession`` for a child (stdio or remote),
        entering its transports on ``stack``. Raises on failure. Shared by
        :meth:`_start_child` (session pool) and :meth:`probe_server` (catalog
        probe) so the transport-construction logic lives in one place."""
        command = cfg.get("command")
        url = _resolve_runtime_value(cfg.get("url", ""), sensitive=False)
        explicit_transport = str(cfg.get("transport", "")).lower()
        if (
            explicit_transport not in {"", "streamable-http", "sse"}
            or bool(command) == bool(url)
            or (explicit_transport and not url)
        ):
            raise RuntimeError("MCP child transport declaration is invalid")
        is_remote = bool(url) or explicit_transport in (
            "streamable-http",
            "sse",
        )
        if not command and not is_remote:
            raise RuntimeError("MCP child requires a command or URL")
        try:
            initialization_timeout = float(
                cfg.get("initialization_timeout", cfg.get("timeout", 300.0))
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("MCP child initialization timeout is invalid") from exc
        if not 0.001 <= initialization_timeout <= 3_600.0:
            raise RuntimeError("MCP child initialization timeout is invalid")
        provider_profile = _selected_child_provider_profile(
            cfg,
            is_remote=is_remote,
        )
        provider_environment: dict[str, str] = {}
        runtime_policy = cfg.get(_RUNTIME_CHILD_POLICY_INTERNAL_KEY)
        if runtime_policy is not None:
            try:
                policy_environment = runtime_policy.child_environment()
            except Exception:
                raise RuntimeError(
                    "MCP child runtime policy environment is unavailable"
                ) from None
            if (
                not isinstance(policy_environment, Mapping)
                or len(policy_environment) > 256
            ):
                raise RuntimeError("MCP child runtime policy environment is invalid")
            for raw_key, raw_value in policy_environment.items():
                key = str(raw_key)
                if (
                    not isinstance(raw_key, str)
                    or not isinstance(raw_value, str)
                    or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", key)
                    or len(raw_value.encode("utf-8")) > 65_536
                    or "\x00" in raw_value
                ):
                    raise RuntimeError(
                        "MCP child runtime policy environment is invalid"
                    )
                provider_environment[key] = raw_value
        if provider_profile is not None:
            from agent_utilities.core.provider_runtime import (
                prepare_provider_runtime_child_environment,
            )

            # Secret backends may perform blocking I/O. Keep resolution off the
            # multiplexer event loop while still failing before process spawn.
            # A bounded slot count prevents timed-out backend calls from
            # exhausting the dedicated resolver executor; late results erase
            # themselves. Do not cancel queued futures on caller timeout: a
            # cancelled concurrent future is marked done before its executor
            # work item is consumed, which would release capacity early and let
            # cancelled items accumulate in ThreadPoolExecutor's internal queue.
            acquired = _PROVIDER_RESOLUTION_CAPACITY.acquire(blocking=False)
            resolution_future: concurrent.futures.Future[Any] | None = None
            try:
                if not acquired:
                    raise RuntimeError("provider resolution capacity unavailable")
                resolution_future = _PROVIDER_RESOLUTION_EXECUTOR.submit(
                    prepare_provider_runtime_child_environment,
                    provider_profile,
                )
                resolution_future.add_done_callback(
                    lambda _future: _PROVIDER_RESOLUTION_CAPACITY.release()
                )
                acquired = False
                prepared_provider = await asyncio.wait_for(
                    asyncio.shield(asyncio.wrap_future(resolution_future)),
                    timeout=initialization_timeout,
                )
            except asyncio.CancelledError:
                if resolution_future is not None:
                    resolution_future.add_done_callback(
                        _close_abandoned_provider_projection
                    )
                elif acquired:
                    _PROVIDER_RESOLUTION_CAPACITY.release()
                raise
            except Exception:
                if resolution_future is not None:
                    resolution_future.add_done_callback(
                        _close_abandoned_provider_projection
                    )
                elif acquired:
                    _PROVIDER_RESOLUTION_CAPACITY.release()
                raise RuntimeError(
                    "MCP child provider profile is unavailable"
                ) from None
            stack.callback(prepared_provider.close)
            provider_environment = dict(prepared_provider.environment)
            provider_environment.update(_provider_child_sandbox_environment(stack))

        if is_remote:
            parsed_url = urlsplit(url)
            if (
                parsed_url.scheme.lower() not in {"http", "https"}
                or not parsed_url.hostname
                or parsed_url.username is not None
                or parsed_url.password is not None
                or parsed_url.fragment
                or parsed_url.query
                or len(url) > 8_192
            ):
                raise RuntimeError("Remote MCP child URL is invalid")
            # Computed here (not just at the transport-pinning site below) so the
            # scheme gate consults the SAME allowlist as the actual DNS-pinned
            # egress: MCP_HTTP_ALLOWED_PRIVATE_HOSTS was already a config field
            # for exactly this (mirroring OIDC_HTTP_ALLOWED_PRIVATE_HOSTS /
            # MODEL_HTTP_ALLOWED_PRIVATE_HOSTS), but this gate never read it —
            # so a deployment that legitimately reaches its MCP fleet over
            # plain HTTP behind a TLS-terminating ingress (MCP_TLS_TERMINATED)
            # could never declare that trust; it always hard-failed here first.
            from agent_utilities.core.config import config as agent_config

            child_private_hosts = cfg.get("allowed_private_hosts", [])
            if not isinstance(child_private_hosts, list):
                raise RuntimeError("Remote MCP child private-host policy is invalid")
            allowed_private_hosts = [
                *agent_config.mcp_http_allowed_private_hosts,
                *(str(value) for value in child_private_hosts),
            ]
            if (
                parsed_url.scheme.lower() == "http"
                and parsed_url.hostname.lower()
                not in {
                    "localhost",
                    "127.0.0.1",
                    "::1",
                    *(host.lower() for host in allowed_private_hosts),
                }
            ):
                raise RuntimeError("Remote MCP child requires HTTPS outside loopback")
            headers = cfg.get("headers")
            if headers:
                if not isinstance(headers, dict) or len(headers) > 64:
                    raise RuntimeError("Remote MCP child headers are invalid")
                validated_headers: dict[str, str] = {}
                materialized = (
                    set(cfg.get("_runtime_materialized_secret_keys") or [])
                    if _runtime_materialized(cfg)
                    else set()
                )
                for key, value in headers.items():
                    name = str(key)
                    lowered = name.lower()
                    if lowered in {
                        "connection",
                        "content-length",
                        "host",
                        "proxy-connection",
                        "te",
                        "trailer",
                        "transfer-encoding",
                        "upgrade",
                    }:
                        raise RuntimeError("Remote MCP child headers are invalid")
                    rendered = _resolve_runtime_value(
                        value,
                        sensitive=_sensitive_config_key(name),
                        materialized=name in materialized,
                    )
                    if (
                        not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}", name)
                        or len(rendered) > 16_384
                        or "\r" in rendered
                        or "\n" in rendered
                    ):
                        raise RuntimeError("Remote MCP child headers are invalid")
                    validated_headers[name] = rendered
                headers = validated_headers

            from agent_utilities.core.http_client import create_async_http_client
            from agent_utilities.core.transport_security import (
                resolve_configured_tls_profile,
            )

            profile_name = str(cfg.get("tls_profile") or "").strip() or None
            profile_ref = str(cfg.get("tls_profile_ref") or "").strip() or None
            trust = resolve_configured_tls_profile(
                "MCP_CHILD",
                profile_name=profile_name,
                profile_ref=profile_ref,
                config=agent_config,
            )
            stack.callback(trust.cleanup)
            if trust.proxy_url:
                raise RuntimeError("Remote MCP child cannot use an inline proxy")

            def _secure_httpx_factory(
                headers: dict[str, str] | None = None,
                timeout: Any = None,
                auth: Any = None,
            ):
                return create_async_http_client(
                    timeout=timeout or 30.0,
                    verify=trust.ssl_context,
                    headers=headers,
                    auth=auth,
                    trust_env=False,
                    follow_redirects=False,
                    pin_egress=True,
                    allowed_private_hosts=allowed_private_hosts,
                )

            # A0 (CONCEPT:AU-OS.identity.so-jwt-protected-children): authenticate jwt-protected children with the
            # multiplexer's service-account bearer. Opt-in via
            # MCP_CLIENT_AUTH=oidc-client-credentials; never overrides a child's
            # own Authorization header; a mint failure aborts the connection.
            # Use a per-request httpx.Auth (not a frozen header): the child's
            # pooled session is long-lived, so a baked-in short-lived token would
            # expire mid-session and wedge calls on a 401 (CONCEPT:AU-OS.identity.so-jwt-protected-children).
            from agent_utilities.mcp.client_credentials import child_auth

            _svc_auth = child_auth(headers)
            use_sse = explicit_transport == "sse" or url.rstrip("/").endswith("/sse")
            if use_sse:
                if sse_client is None:
                    raise RuntimeError("mcp SDK has no sse_client for SSE transport")
                transport = sse_client(
                    url,
                    headers=headers,
                    auth=_svc_auth,
                    httpx_client_factory=_secure_httpx_factory,
                )
            else:
                if streamablehttp_client is None:
                    raise RuntimeError(
                        "mcp SDK has no streamablehttp_client for "
                        "streamable-http transport"
                    )
                transport = streamablehttp_client(
                    url,
                    headers=headers,
                    auth=_svc_auth,
                    httpx_client_factory=_secure_httpx_factory,
                )
            # streamable-http yields (read, write, get_session_id); sse yields
            # (read, write). Take the first two streams either way.
            streams = await stack.enter_async_context(transport)
            read_stream, write_stream = streams[0], streams[1]
        else:
            # `command` is guaranteed set here: the earlier guard raises unless
            # `command` or `is_remote` is truthy, and this is the `not is_remote`
            # branch.
            assert command, "unreachable: non-remote server must have a 'command'"
            command = _resolve_runtime_value(command, sensitive=False)
            raw_args = cfg.get("args", [])
            args = (
                [_resolve_runtime_value(value, sensitive=False) for value in raw_args]
                if isinstance(raw_args, list)
                else raw_args
            )
            configured_env = cfg.get("env") or {}
            if (
                not 1 <= len(command) <= 4_096
                or "\x00" in command
                or not isinstance(args, list)
                or len(args) > 128
                or not all(
                    isinstance(value, str)
                    and len(value) <= 8_192
                    and "\x00" not in value
                    for value in args
                )
                or not isinstance(configured_env, dict)
                or len(configured_env) > 256
            ):
                raise RuntimeError("Local MCP child process configuration is invalid")
            # A child receives only execution/runtime trust variables plus the
            # variables explicitly delegated in its own catalog entry. Copying
            # the entire parent environment leaks unrelated fleet credentials.
            provider_controlled_keys = {key.upper() for key in provider_environment}
            merged_env = {
                key: value
                for key, value in os.environ.items()
                if key.upper() in _CHILD_ENV_ALLOWLIST
                and key.upper() not in provider_controlled_keys
            }
            merged_env.update(provider_environment)
            materialized = (
                set(cfg.get("_runtime_materialized_secret_keys") or [])
                if _runtime_materialized(cfg)
                else set()
            )
            for raw_key, raw_value in configured_env.items():
                key = str(raw_key)
                if key.upper() in (_PROVIDER_CHILD_ENV_KEYS | provider_controlled_keys):
                    raise RuntimeError(
                        "MCP child provider environment is parent-controlled"
                    )
                value = _resolve_runtime_value(
                    raw_value,
                    sensitive=_sensitive_config_key(key),
                    materialized=key in materialized,
                )
                if (
                    not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", key)
                    or len(value) > 65_536
                    or "\x00" in value
                ):
                    raise RuntimeError("Local MCP child environment is invalid")
                merged_env[key] = value
            server_params = StdioServerParameters(
                command=command, args=args, env=merged_env
            )
            if runtime_policy is not None:
                try:
                    runtime_policy.verify_before_spawn()
                except Exception:
                    raise RuntimeError(
                        "MCP child runtime policy pre-spawn verification failed"
                    ) from None
            # The MCP SDK otherwise forwards a child's raw stderr to the parent
            # process. Import and native-loader failures routinely contain
            # interpreter, checkout, and trust-material locations. The parent
            # already emits bounded transport/error codes, so discard that raw
            # channel and keep the sink alive for the complete child generation.
            child_error_sink = stack.enter_context(
                open(os.devnull, "w", encoding="utf-8")
            )
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(server_params, errlog=child_error_sink)
            )

        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        # Cold starts can legitimately exceed 30 seconds on constrained hosts
        # (for example, a Python MCP child importing from a WSL-mounted volume).
        # Keep the handshake bounded, but honor the same per-child connect
        # budget rather than imposing a second, hidden fixed ceiling.
        await asyncio.wait_for(session.initialize(), timeout=initialization_timeout)
        return session

    async def _start_child(
        self, server_name: str, cfg: dict
    ) -> tuple[str, ChildRuntime, list[mcp.types.Tool], dict] | None:
        """Starts a single child server, registers its exit stack on success, and returns its tools and runtime."""
        try:
            cfg, runtime_policy = _prepare_runtime_child_policy(cfg)
        except RuntimeError:
            logger.error("MCP child runtime policy resolution failed")
            return None

        def _close_policy() -> None:
            if runtime_policy is not None:
                _close_runtime_child_policy(runtime_policy)

        command = cfg.get("command")
        url = _resolve_runtime_value(cfg.get("url", ""), sensitive=False)
        explicit_transport = str(cfg.get("transport", "")).lower()
        if (
            explicit_transport not in {"", "streamable-http", "sse"}
            or bool(command) == bool(url)
            or (explicit_transport and not url)
        ):
            logger.error("MCP child transport declaration is invalid")
            _close_policy()
            return None
        # A child is remote (HTTP) when it declares a ``url`` or an http/sse
        # ``transport``; otherwise it is a local stdio subprocess run via
        # ``command``. Either kind loads transparently from the same config.
        is_remote = bool(url) or explicit_transport in (
            "streamable-http",
            "sse",
        )
        if not command and not is_remote:
            logger.warning("MCP child has neither command nor URL; skipping")
            _close_policy()
            return None

        try:
            timeout = float(cfg.get("timeout", 300.0))
        except (TypeError, ValueError):
            timeout = 0.0
        if not 0.001 <= timeout <= 3_600.0:
            logger.error("MCP child timeout is outside the safety boundary")
            _close_policy()
            return None

        # Session-pool sizing (CONCEPT:AU-ECO.mcp.profile-differences-from-client): remote children may hold N
        # independent connections for parallel in-flight calls; stdio children
        # are single-pipe and always keep exactly one session.
        from agent_utilities.core.config import config as agent_config

        pool_size = 1
        if is_remote:
            try:
                pool_size = int(
                    cfg.get("pool_size") or agent_config.mcp_child_pool_size
                )
            except (TypeError, ValueError):
                pool_size = 0
            if not 1 <= pool_size <= 64:
                logger.error("MCP child pool size is outside the safety boundary")
                _close_policy()
                return None

        logger.info(
            "Starting MCP child (transport=%s)", "remote" if is_remote else "stdio"
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
            _bounded_tool_catalog(tools_result.tools)
            tools = list(tools_result.tools)
            if runtime_policy is not None:
                tools = self._admit_runtime_policy_tools(
                    server_name,
                    runtime_policy,
                    tools,
                )
            return sessions, tools

        # Service-authenticated remote children must recycle their session before
        # the bearer's TTL elapses (the result stream is authed once at connect
        # and then wedges on expiry); derive that lifetime from the token TTL.
        session_max_age: float | None = None
        if is_remote:
            from agent_utilities.mcp.client_credentials import service_session_max_age

            session_max_age = service_session_max_age(cfg.get("headers"))
        runtime = ChildRuntime(
            server_name, cfg, connect=_connect, session_max_age=session_max_age
        )
        try:
            tools = await runtime.start()
        except TimeoutError:
            logger.error("MCP child startup timed out")
            _close_policy()
            return None
        except Exception as exc:
            logger.error(
                "Failed to start MCP child (%s)",
                type(exc).__name__,
            )
            _close_policy()
            return None

        if runtime_policy is not None:
            self._child_runtime_policies[server_name] = runtime_policy

        logger.info(
            "Loaded %d tools from MCP child (%d session%s)",
            len(tools),
            pool_size,
            "" if pool_size == 1 else "s",
        )
        return server_name, runtime, tools, cfg

    def load_catalog(self) -> dict[str, dict]:
        """Parse the config once into the mountable-server catalog WITHOUT
        spawning any child (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog).

        Idempotent: the parsed ``{server_name: cfg}`` map is cached on
        ``self._catalog`` and reused. Self (``mcp-multiplexer``) and entries
        flagged ``disabled`` are excluded — they are never mountable.
        """
        if self._catalog is not None:
            return self._catalog

        self._catalog = {}
        config_data: dict[str, Any] = {"mcpServers": {}}
        if self.config_path.exists():
            try:
                content = _read_catalog_text(self.config_path)
            except Exception as exc:
                logger.error("Failed to read MCP config (%s)", type(exc).__name__)
                content = ""
            if content:
                try:
                    # Parse the persistent document literally. Runtime references
                    # are resolved only at the exact child boundary that consumes
                    # them, so secret values never enter this catalog wholesale.
                    config_data = json.loads(content)
                except Exception as exc:
                    logger.error("Failed to parse MCP config (%s)", type(exc).__name__)
                    config_data = {"mcpServers": {}}

        servers = config_data.get("mcpServers") or {}
        if not isinstance(servers, dict) or len(servers) > 512:
            servers = {}
        from agent_utilities.observability.langfuse_trust import (
            LangfuseTrustError,
            is_langfuse_server,
            native_langfuse_mcp_config,
            prepare_langfuse_mcp_config,
        )

        runtime_materialized_configs: set[int] = set()

        has_langfuse = any(
            isinstance(cfg, dict) and is_langfuse_server(str(name), cfg)
            for name, cfg in servers.items()
        )
        if not has_langfuse:
            try:
                native = native_langfuse_mcp_config()
            except LangfuseTrustError as exc:
                native = None
                logger.error(
                    "Native Langfuse MCP disabled: %s configuration invalid (%s)",
                    exc.category,
                    str(exc),
                )
            if native is not None:
                runtime_materialized_configs.add(id(native))
                servers["langfuse-mcp"] = native

        # The host server never mounts itself as a child (avoids self-recursion).
        # Defaults to the retired standalone multiplexer name; graph-os's
        # attach_fleet_loader widens this to include "graph-os".
        skip = getattr(self, "_skip_servers", None) or {"mcp-multiplexer"}
        for server_name, cfg in servers.items():
            if (
                not isinstance(server_name, str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", server_name)
                or not isinstance(cfg, dict)
                or len(cfg) > 128
            ):
                continue
            if server_name in skip or cfg.get("disabled", False):
                continue
            runtime_materialized = id(cfg) in runtime_materialized_configs
            if not runtime_materialized:
                try:
                    _validate_externalized_child_secrets(cfg)
                except RuntimeError:
                    logger.error("MCP child disabled: credential policy violation")
                    continue
            if is_langfuse_server(str(server_name), cfg):
                try:
                    if not runtime_materialized:
                        cfg = prepare_langfuse_mcp_config(cfg)
                    cfg = attest_runtime_child_config(cfg)
                except LangfuseTrustError as exc:
                    logger.error(
                        "Langfuse MCP entry disabled: %s configuration invalid (%s)",
                        exc.category,
                        str(exc),
                    )
                    continue
            self._catalog[str(server_name)] = cfg
        return self._catalog

    def reload_catalog(self) -> dict[str, dict]:
        """Discard runtime-derived fleet state and reparse the current catalog.

        Hot configuration changes must not leave a disabled child callable or a
        credential/TLS change attached to an old process.  Existing forwarder
        registrations remain inert and reusable; their routing entries and
        per-session visibility are cleared until the child is mounted again.
        """
        stale_children = tuple(
            (name, runtime, self._child_runtime_policies.get(name))
            for name, runtime in self.children.items()
        )
        stale_tool_names = set(self.tool_to_server)
        self.children.clear()
        self._child_runtime_policies.clear()
        self._child_policy_admitted_tools.clear()
        self._child_catalog_fingerprints.clear()
        self.sessions.clear()
        self.tool_to_server.clear()
        self.aggregated_tools.clear()
        self._probe_cache.clear()
        self._tool_embeddings.clear()
        self._prefix_map = None
        self._prefix_reverse.clear()
        self._catalog = None
        for loaded in self._session_loaded.values():
            loaded.difference_update(stale_tool_names)
        for loaded in self._auto_unload.values():
            loaded.difference_update(stale_tool_names)

        if stale_children:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # Configuration tooling may run without an event loop.  Calls
                # are already denied by the cleared routing state; normal server
                # shutdown remains the owner of these runtime resources.
                logger.warning("MCP catalog reloaded outside a serving event loop")
                for _name, _runtime, policy in stale_children:
                    if policy is not None:
                        _close_runtime_child_policy(policy)
            else:

                async def _close_stale(runtime: ChildRuntime, policy: Any) -> None:
                    try:
                        await runtime.aclose()
                    finally:
                        if policy is not None:
                            _close_runtime_child_policy(policy)

                for _name, runtime, policy in stale_children:
                    task = loop.create_task(_close_stale(runtime, policy))
                    self._catalog_reload_tasks.add(task)
                    task.add_done_callback(self._catalog_reload_tasks.discard)

        return self.load_catalog()

    @staticmethod
    def _server_stem(name: str) -> str:
        """Full cleaned server name used to disambiguate colliding prefixes."""
        clean = "".join(c if (c.isalnum() or c in ("_", "-")) else "_" for c in name)
        return clean.replace("-", "_").lower().strip("_")

    def _build_prefix_map(self) -> None:
        """Assign every catalog server a unique prefix (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog).

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
        """The unique, collision-free prefix for a server (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog).
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
        # The per-child hardening runtime (CONCEPT:AU-ECO.mcp.profile-differences-from-client) carries the
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
        # `primary_session` is `None` only for a runtime with no adopted sessions
        # yet, which shouldn't happen this far into registration (both branches
        # above guarantee at least one) — but don't put a `None` where a real
        # `ClientSession` is expected if that invariant is ever violated.
        primary = runtime.primary_session
        if primary is not None:
            self.sessions[server_name] = primary

        disabled_tools = cfg.get("disabledTools", [])
        enabled_tools = cfg.get("enabledTools", None)

        registered: list[mcp.types.Tool] = []
        for tool in tools:
            # 1. Whitelist Check (if enabledTools is defined)
            if enabled_tools is not None:
                import fnmatch

                matched = any(fnmatch.fnmatch(tool.name, pat) for pat in enabled_tools)
                if not matched:
                    logger.info("Skipping a non-whitelisted MCP child tool")
                    continue

            # 2. Blacklist Check
            if disabled_tools:
                import fnmatch

                matched_disabled = any(
                    fnmatch.fnmatch(tool.name, pat) for pat in disabled_tools
                )
                if matched_disabled:
                    logger.info("Skipping a disabled MCP child tool")
                    continue

            prefix = self.server_prefix(server_name)
            prefixed_name = clean_tool_name(prefix, server_name, tool.name)
            self.tool_to_server[prefixed_name] = (server_name, tool.name)

            # Preserve _meta (carries FastMCP tags) so downstream consumers — the
            # verbose-tool hold-back, visibility filtering — can read the child's
            # tags off the aggregated tool. (CONCEPT:AU-ECO.multiplexer.condensed-server-load)
            prefixed_tool = mcp.types.Tool(
                name=prefixed_name,
                description=tool.description or "",
                inputSchema=tool.inputSchema,
                _meta=getattr(tool, "meta", None),
            )
            self.aggregated_tools.append(prefixed_tool)
            registered.append(prefixed_tool)
        return registered

    async def mount_child(self, server_name: str) -> list[mcp.types.Tool]:
        """Start ONE configured child on demand and register its tools
        (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog).

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
            logger.warning("Requested MCP child is not in the catalog")
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
    # Dynamic tool gateway (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog)
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
        release (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog). Returns (and caches) ``{"tools": [...],
        "error": str|None}``. An already-mounted child reuses its live tools
        instead of reconnecting; an unreachable server records its error string
        (so find_tools/load_tools can report *why* it is unavailable) rather
        than raising."""
        if not force and server_name in self._probe_cache:
            return self._probe_cache[server_name]

        if server_name in self.children:
            info: dict[str, Any] = {
                "tools": self._live_tools_for_server(server_name),
                "error": None,
            }
            self._probe_cache[server_name] = info
            return info

        cfg = self.load_catalog().get(server_name)
        if cfg is None:
            info = {"tools": [], "error": "not in catalog"}
            self._probe_cache[server_name] = info
            return info

        try:
            probe_to = float(
                timeout
                if timeout is not None
                else cfg.get("probe_timeout", cfg.get("timeout", 10.0))
            )
        except (TypeError, ValueError):
            probe_to = 0.0
        if not 0.001 <= probe_to <= 300.0:
            info = {"tools": [], "error": "invalid probe timeout"}
            self._probe_cache[server_name] = info
            return info

        async def _probe() -> list[dict]:
            # Enter AND exit the transports within this single coroutine so the
            # anyio cancel scopes are not crossed between tasks. ``wait_for``
            # runs this whole coroutine as one task, so the stack is opened and
            # closed in the same task even on timeout-cancellation. (Wrapping
            # only the connect in wait_for would exit the scope in a different
            # task — "Attempted to exit cancel scope in a different task".)
            runtime_cfg, runtime_policy = _prepare_runtime_child_policy(cfg)
            try:
                async with contextlib.AsyncExitStack() as stack:
                    session = await self._open_one_session(
                        server_name, runtime_cfg, stack
                    )
                    result = await session.list_tools()
                    _bounded_tool_catalog(result.tools)
                    tools = list(result.tools)
                    if runtime_policy is not None:
                        tools = self._admit_runtime_policy_tools(
                            server_name,
                            runtime_policy,
                            tools,
                        )
                    return _bounded_tool_catalog(tools)
            finally:
                if runtime_policy is not None:
                    _close_runtime_child_policy(runtime_policy)
                    self._child_policy_admitted_tools.pop(server_name, None)

        try:
            tools = await asyncio.wait_for(_probe(), timeout=probe_to)
            info = {"tools": tools, "error": None}
        except TimeoutError:
            info = {"tools": [], "error": f"timeout after {probe_to:g}s"}
        except Exception as e:
            info = {"tools": [], "error": _format_probe_error(e)}
        self._probe_cache[server_name] = info
        return info

    @classmethod
    async def probe_declaration(
        cls,
        server_name: str,
        declaration: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        """Probe one in-memory declaration through the canonical child boundary.

        KG ingestion and GraphOS fleet discovery intentionally share this entry
        point.  Declarations never pass through an alternate raw MCP client:
        credential externalization, AgentConfig TLS profiles, pinned egress,
        redirect denial, bounded stdio delegation, auth, time, and catalog
        limits are therefore identical on both paths.
        """

        if (
            not isinstance(server_name, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", server_name)
            or not isinstance(declaration, dict)
            or len(declaration) > 128
        ):
            raise RuntimeError("MCP child declaration is invalid")
        materialized = _runtime_materialized(declaration)
        if not materialized:
            _validate_externalized_child_secrets(declaration)
        multiplexer = cls(Path())
        multiplexer._catalog = {server_name: dict(declaration)}
        try:
            return await multiplexer.probe_server(
                server_name,
                force=True,
                timeout=timeout,
            )
        finally:
            await multiplexer.aclose()

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

    async def _embed_semantic_scores(
        self, query: str, probe: dict, semantic: dict[str, float]
    ) -> None:
        """Populate ``semantic[bare_tool] = query↔description cosine`` via the injected
        in-process embedder (graph-os wires its own embedding model here). Per-tool
        embeddings are cached by ``server::tool`` so only the query is embedded per call;
        all embedding runs OFF-THREAD (the embed model is sync/remote) so the event loop
        never blocks. Any failure degrades silently to token-overlap (``semantic`` is
        left as-is). No-op when no embedder is injected."""
        embed = self._embed_fn
        if embed is None:
            return
        tools: list[tuple[str, str]] = []  # (bare_tool, cache_key)
        pending_text: list[str] = []
        pending_key: list[str] = []
        for server, info in probe.items():
            if info.get("error"):
                continue
            for entry in info.get("tools", []):
                tool = entry["name"]
                key = f"{server}::{tool}"
                tools.append((tool, key))
                if key not in self._tool_embeddings:
                    pending_text.append(f"{tool}. {entry.get('description', '')}"[:512])
                    pending_key.append(key)
        try:
            if pending_text:
                vecs = await asyncio.to_thread(embed, pending_text)
                for k, v in zip(pending_key, vecs, strict=False):
                    if v:
                        self._tool_embeddings[k] = list(v)
            qv = (await asyncio.to_thread(embed, [query]))[0]
        except Exception as exc:
            logger.debug(
                "find_tools embedding rerank unavailable; token-overlap only "
                "(exception_type=%s)",
                type(exc).__name__,
            )
            return
        if not qv:
            return
        for tool, key in tools:
            vec = self._tool_embeddings.get(key)
            if vec:
                c = _cosine(qv, vec)
                if c > 0:
                    semantic[tool] = max(semantic.get(tool, 0.0), c)

    async def discover_tools(
        self, query: str, top_k: int | None = None, loaded: set[str] | None = None
    ) -> dict:
        """Rank candidate tools across the whole fleet for an NL ``query``
        (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog), without exposing or holding any child.

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

        # Semantic scores keyed by bare tool name. When graph-os injects an in-process
        # embedder (attach_fleet_loader), rank every probed tool by query↔description
        # cosine similarity (embeddings cached per tool). Absent ⇒ this stays empty and
        # the token-overlap backbone below ranks alone. This is what makes find_tools
        # understand intent ("send a message to a gitlab MR" → the gitlab tools) instead
        # of only matching literal tokens. (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog)
        semantic: dict[str, float] = {}
        await self._embed_semantic_scores(query, probe, semantic)

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
                        "mounted": prefixed
                        in (loaded if loaded is not None else self._exposed),
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
        """Browse configured fleet metadata without unnecessary child starts.

        A server drill-down probes only that server.  A metadata-only fleet
        listing (``include_tools=False``) never probes a child and reports any
        already-cached reachability data.  Only an explicit whole-fleet tool
        listing probes the whole fleet.

        With ``server`` set, drills into one server and returns its full tool
        list with descriptions.
        """
        catalog = self.load_catalog()

        if server:
            if server not in catalog:
                return {"error": f"'{server}' is not in the catalog"}
            info = await self.probe_server(server)
            prefix = self.server_prefix(server)
            result = {
                "server": server,
                "prefix": prefix,
                "mounted": server in self.children,
                "probed": True,
                "available": info.get("error") is None,
                "error": info.get("error"),
            }
            if include_tools:
                result["tools"] = [
                    {
                        "prefixed_name": clean_tool_name(prefix, server, t["name"]),
                        "tool": t["name"],
                        "description": t.get("description", ""),
                        "enabled": self._tool_enabled(server, t["name"]),
                    }
                    for t in info.get("tools", [])
                ]
            return result

        probe = await self.probe_catalog() if include_tools else dict(self._probe_cache)

        servers: list[dict] = []
        for name in catalog:
            probed = name in probe
            info = probe.get(name) or {}
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
                "probed": probed,
                "available": info.get("error") is None if probed else None,
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
            "unavailable": [s["server"] for s in servers if s["available"] is False],
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
            # CONCEPT:AU-ECO.multiplexer.condensed-server-load — a SERVER-level load exposes only the condensed action
            # surface; verbose 1:1 tools stay loadable by EXPLICIT name (via requested_tools)
            # so ``load_tools(servers=[X])`` never floods a session's context with X's whole
            # granular surface. Mirrors the always-on mount's verbose-hold.
            wanted = set()
            for server in mounted:
                for t in self.prefixed_tools_for_server(server):
                    if _tool_is_verbose(t):
                        continue
                    wanted.add(t.name)

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

    def session_loaded(self, session_key: str) -> set[str]:
        """The set of prefixed tools loaded (visible) for one session."""
        return self._session_loaded.setdefault(session_key, set())

    def requested_prefixed(
        self, tools: list[str] | None, servers: list[str] | None
    ) -> list[str]:
        """All catalog prefixed names a ``load_tools`` request resolves to.

        Unlike ``resolve_and_mount`` (which returns only the *newly registerable*
        names), this is the FULL set the requesting session should see — including
        tools another session already registered. Call after ``resolve_and_mount``
        so the owning children are mounted and known.
        """
        wanted: set[str] = set(tools or [])
        for server in servers or []:
            wanted.update(t.name for t in self.prefixed_tools_for_server(server))
        return sorted(n for n in wanted if n in self.tool_to_server)

    def status_snapshot(self) -> dict[str, Any]:
        """Fleet health surface: per-child state, limits, load, restarts."""
        return {
            "children": {
                name: {
                    **runtime.status(),
                    **(
                        {"catalog_fingerprint": self._child_catalog_fingerprints[name]}
                        if name in self._child_catalog_fingerprints
                        else {}
                    ),
                }
                for name, runtime in sorted(self.children.items())
            },
            "total_children": len(self.children),
            "total_tools": len(self.aggregated_tools),
        }

    async def aclose(self) -> None:
        """Shut down every child runtime and direct stack registration."""
        policies = tuple(self._child_runtime_policies.values())
        try:
            for runtime in self.children.values():
                await runtime.aclose()
        finally:
            self._child_runtime_policies.clear()
            self._child_policy_admitted_tools.clear()
            self._child_catalog_fingerprints.clear()
            for policy in policies:
                _close_runtime_child_policy(policy)
        if self._catalog_reload_tasks:
            await asyncio.gather(
                *tuple(self._catalog_reload_tasks), return_exceptions=True
            )
        await self.exit_stack.aclose()


def _resolve_config_path(explicit: str | None) -> Path:
    """Resolve the MCP fleet file from an explicit value or the XDG config."""
    if explicit:
        return Path(explicit)
    if setting("MCP_CONFIG"):
        return Path(setting("MCP_CONFIG"))
    from agent_utilities.core.paths import config_dir

    return config_dir() / "mcp_config.json"


def invalidate_live_catalogs() -> int:
    """Rebuild every live GraphOS fleet catalog after a hot setting change."""
    refreshed = 0
    for multiplexer in tuple(_LIVE_MULTIPLEXERS):
        multiplexer.reload_catalog()
        refreshed += 1
    return refreshed


def _make_forwarder(mux: MCPMultiplexer, prefixed_name: str):
    """Build the async fn that forwards a prefixed tool call to its child."""

    async def _forward(**kwargs: Any) -> ToolResult:
        if mux._authority_scope is None:
            result = await mux.call_proxied_tool(prefixed_name, kwargs)
        else:
            with mux._authority_scope():
                result = await mux.call_proxied_tool(prefixed_name, kwargs)
        if bool(getattr(result, "isError", False)):
            # ``ToolResult`` has no error bit. Returning one here silently
            # converts a child MCP failure into an outer success, so raise the
            # framework's typed error exactly as FastMCP's native proxy does.
            # Keep the public message stable and free of child response data.
            raise ToolError("delegated_child_tool_failed")
        return ToolResult(
            content=list(getattr(result, "content", []) or []),
            structured_content=getattr(result, "structuredContent", None),
        )

    return _forward


def _tool_is_verbose(tool: mcp.types.Tool) -> bool:
    """Whether a child tool is tagged ``verbose`` (FastMCP propagates tags in
    ``_meta``). Verbose 1:1 tools (e.g. graph-os's granular per-action surface)
    are kept in the catalog but NOT auto-exposed by an always-on child — they
    load on demand via ``find_tools``/``load_tools`` to conserve context
    (CONCEPT:AU-ECO.multiplexer.condensed-server-load)."""
    meta = getattr(tool, "meta", None)
    if not isinstance(meta, dict):
        return False
    tags = (meta.get("fastmcp") or {}).get("tags") or []
    return "verbose" in tags


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
    """Register the always-present fleet-health meta-tool (CONCEPT:AU-ECO.mcp.profile-differences-from-client)."""

    async def _status() -> ToolResult:
        _require_fleet_capability("discover")
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


async def _notify_tools_changed(mcp) -> None:
    """Emit ``notifications/tools/list_changed`` so the client re-fetches the
    tool list after a dynamic mount/unmount. Best-effort: a missing request
    context (e.g. no client attached) must not fail the meta-tool."""
    try:
        from fastmcp.server.dependencies import get_context

        await get_context().send_notification(mcp_types.ToolListChangedNotification())
    except Exception as e:  # pragma: no cover - context not always present
        logger.warning(
            "Could not send tools/list_changed notification (exception_type=%s)",
            type(e).__name__,
        )


def _session_key() -> str:
    """Stable per-connection key for session-scoped tool visibility.

    On a shared streamable-http server every client gets its own
    ``Context.session_id``; with no session context (stdio / single-client) all
    requests fall back to one key so behaviour matches the pre-Phase-5 server.
    """
    try:
        from fastmcp.server.dependencies import (
            get_access_token,
            get_context,
            get_http_request,
        )

        sid = get_context().session_id
        if sid:
            return str(sid)
    except Exception:
        pass
    try:
        get_http_request()
    except RuntimeError:
        return "__local_stdio__"
    except Exception:
        return "__invalid_http_context__"
    try:
        token = get_access_token()
    except Exception:
        token = None
    if token is None:
        return "__unauthenticated_http__"
    claims = getattr(token, "claims", None) or {}
    raw = "\x00".join(
        str(value or "")
        for value in (
            getattr(token, "client_id", None),
            claims.get("sub") if isinstance(claims, dict) else None,
            claims.get("tenant_id") if isinstance(claims, dict) else None,
        )
    )
    digest = hashlib.blake2s(
        raw.encode("utf-8"), key=_SESSION_KEY, digest_size=16
    ).hexdigest()
    return f"http_{digest}"


class SessionVisibilityMiddleware(Middleware):
    """Per-session progressive disclosure (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog, plan Phase 5).

    Child forwarders are registered process-globally (once) but a shared server
    must not leak one session's ``load_tools`` to another. This middleware scopes
    ``tools/list`` — and gates ``tools/call`` — to each session's loaded set, plus
    the always-visible meta/always-on tools (``mux._global_visible``). Composes
    with the Eunomia principal filter (both must allow a tool to appear).
    """

    def __init__(self, mux: MCPMultiplexer, mcp: Any = None) -> None:
        self.mux = mux
        self._mcp = mcp

    def _gated(self, name: str) -> bool:
        """Whether ``name`` is subject to session-scoped visibility at all.

        Two independent populations are gated the same way: fleet forwarders
        (``_exposed`` — must be mounted via ``load_tools``) and the host
        server's OWN granular tools held back under ``MCP_TOOL_MODE=intent``
        (``_local_gated`` — already registered locally, just hidden by
        default; CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse). Everything else (native
        tools in neither set, meta-tools) is ungated — always visible.
        """
        return name in self.mux._exposed or name in self.mux._local_gated

    def _visible(self, name: str) -> bool:
        if name in self.mux._global_visible:
            return True
        if not self._gated(name):
            return True
        return name in self.mux._session_loaded.get(_session_key(), set())

    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        return [t for t in tools if self._visible(t.name)]

    async def on_call_tool(self, context, call_next):
        name = getattr(context.message, "name", None)
        # Only gate tools that are actually gated (registered but not globally
        # visible); a tool this session hasn't loaded behaves as "unknown" until
        # load_tools.
        if (
            name
            and self._gated(name)
            and name not in self.mux._global_visible
            and name not in self.mux._session_loaded.get(_session_key(), set())
        ):
            raise ToolError(
                f"Tool '{name}' is not loaded in this session. "
                f"Call load_tools(tools=['{name}']) first."
            )
        result = await call_next(context)
        # CONCEPT:AU-ECO.mcp.intent-surface-tool-lifecycle — a tool loaded with
        # auto_unload=True is retracted right after this (successful) call so a
        # one-shot task doesn't linger in the session's tool list.
        session_key = _session_key()
        auto = self.mux._auto_unload.get(session_key)
        if name and auto and name in auto:
            auto.discard(name)
            self.mux._session_loaded.get(session_key, set()).discard(name)
            if self._mcp is not None:
                await _notify_tools_changed(self._mcp)
        return result


def _tools_with_tag(mcp, tags: list[str] | None) -> set[str]:
    """Names of every registered local FastMCP tool carrying ANY of ``tags``.

    Reuses the same tag vocabulary ``DynamicVisibilityTransform``
    (``server_factory.py``) reads for ``MCP_DISABLED_TAGS`` — a bulk
    "toolset"/domain unload selector (CONCEPT:AU-ECO.mcp.intent-surface-tool-lifecycle) alongside single-tool
    and whole-server unload.
    """
    if not tags:
        return set()
    wanted = {str(t) for t in tags}
    from agent_utilities.mcp.verbose_tools import _provider_tools

    out: set[str] = set()
    for name, tool in _provider_tools(mcp).items():
        tool_tags = getattr(tool, "tags", None)
        if isinstance(tool_tags, set) and tool_tags & wanted:
            out.add(name)
    return out


async def load_session_tools(
    mcp,
    mux: MCPMultiplexer,
    *,
    tools: list[str] | None = None,
    servers: list[str] | None = None,
    auto_unload: bool = False,
) -> dict[str, Any]:
    """Core of the ``load_tools`` meta-tool (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog,
    CONCEPT:AU-ECO.mcp.intent-surface-tool-lifecycle) — standalone so both the meta-tool itself and the
    intent surface's ``manage`` verb can drive it.

    ``auto_unload=True`` marks every name this call newly exposes for automatic
    retraction the NEXT time it is called (load -> use -> auto-unload), so a
    tool pulled in for one task doesn't linger in a long session's surface —
    call again anytime to reload it (nothing is lost, just not kept around).
    """
    # Split out the host server's OWN gated tools (CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse) —
    # already registered locally under MCP_TOOL_MODE=intent, they only need a
    # session-visibility flip, never fleet mounting/resolution.
    requested = list(tools or [])
    local_names = [n for n in requested if n in mux._local_gated]
    fleet_tools = [n for n in requested if n not in mux._local_gated]

    mounted_servers, to_expose, failed = await mux.resolve_and_mount(
        tools=fleet_tools, servers=servers
    )
    # Register forwarders process-globally (once); visibility is per-session.
    for name in to_expose:
        tool_obj = mux.tool_object(name)
        if tool_obj is not None:
            _register_forwarder(mcp, mux, tool_obj)
    # Make the full resolved set visible to THIS session (incl. tools another
    # session already registered).
    session_names = mux.requested_prefixed(fleet_tools, servers) + local_names
    loaded = mux.session_loaded(_session_key())
    newly = [n for n in session_names if n not in loaded]
    loaded.update(session_names)
    if auto_unload and newly:
        mux._auto_unload.setdefault(_session_key(), set()).update(newly)
    if newly:
        await _notify_tools_changed(mcp)
    return {
        "mounted_servers": mounted_servers,
        "newly_exposed": newly,
        "failed": failed,
        "session_total": len(loaded),
        "total_registered": len(mux._exposed) + len(mux._local_gated),
        "auto_unload": bool(auto_unload) and newly,
    }


async def unload_session_tools(
    mcp,
    mux: MCPMultiplexer,
    *,
    tools: list[str] | None = None,
    servers: list[str] | None = None,
    toolsets: list[str] | None = None,
) -> dict[str, Any]:
    """Core of the ``unload_tools`` meta-tool (CONCEPT:AU-ECO.mcp.intent-surface-tool-lifecycle).

    Three unload granularities, unioned: ``tools`` (exact names), ``servers``
    (every tool of a fleet server, OR every one of the HOST's own gated tools
    when a server name matches ``mux._skip_servers``/the host itself — e.g.
    ``servers=["graph-os"]`` unloads the whole condensed surface at once), and
    ``toolsets`` (every tool carrying one of these tags — a domain/toolset
    bulk-unload). Retracts from THIS session only; the forwarder/registration
    stays process-global so another session (or a future ``load_tools`` call
    in this one) is unaffected/instant.
    """
    loaded = mux.session_loaded(_session_key())
    names: set[str] = set(tools or [])
    for server in servers or []:
        if server in (mux._skip_servers or ()):
            names.update(mux._local_gated)
        else:
            names.update(t.name for t in mux.prefixed_tools_for_server(server))
    names.update(_tools_with_tag(mcp, toolsets))

    removed = [n for n in sorted(names) if n in loaded]
    auto = mux._auto_unload.get(_session_key())
    for name in removed:
        loaded.discard(name)
        if auto:
            auto.discard(name)
    if removed:
        await _notify_tools_changed(mcp)
    return {"unloaded": removed, "session_total": len(loaded)}


def _register_meta_tools(mcp, mux: MCPMultiplexer) -> None:
    """Register the dynamic-gateway meta-tools (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog):
    ``find_tools`` (semantic discovery over the whole fleet), ``list_catalog``
    (flat browse of every server + its tools), ``load_tools`` / ``unload_tools``
    (mount/expose and retract tools at runtime — with a load->use->auto-unload
    lifecycle, CONCEPT:AU-ECO.mcp.intent-surface-tool-lifecycle — notifying the client each time), plus
    the status tool."""

    async def _find_tools(query: str, top_k: int = 0) -> ToolResult:
        _require_fleet_capability("discover")
        if not isinstance(query, str) or not 1 <= len(query) <= 4_096:
            raise ToolError("find_tools query is outside the safety boundary")
        if not isinstance(top_k, int) or not 0 <= top_k <= 100:
            raise ToolError("find_tools top_k is outside the safety boundary")
        loaded = mux.session_loaded(_session_key())
        discovery = await mux.discover_tools(query, top_k=top_k or None, loaded=loaded)
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
        tools: list[str] | None = None,
        servers: list[str] | None = None,
        auto_unload: bool = False,
    ) -> ToolResult:
        _require_fleet_capability("delegate")
        if len(tools or []) > 128 or len(servers or []) > 32:
            raise ToolError("load_tools request is outside the safety boundary")
        payload = await load_session_tools(
            mcp, mux, tools=tools, servers=servers, auto_unload=auto_unload
        )
        return ToolResult(
            content=[
                mcp_types.TextContent(type="text", text=json.dumps(payload, indent=2))
            ],
            structured_content=payload,
        )

    async def _unload_tools(
        tools: list[str] | None = None,
        servers: list[str] | None = None,
        toolsets: list[str] | None = None,
    ) -> ToolResult:
        _require_fleet_capability("delegate")
        if (
            len(tools or []) > 128
            or len(servers or []) > 32
            or len(toolsets or []) > 64
        ):
            raise ToolError("unload_tools request is outside the safety boundary")
        payload = await unload_session_tools(
            mcp, mux, tools=tools, servers=servers, toolsets=toolsets
        )
        return ToolResult(
            content=[
                mcp_types.TextContent(type="text", text=json.dumps(payload, indent=2))
            ],
            structured_content=payload,
        )

    async def _list_catalog(server: str = "", include_tools: bool = True) -> ToolResult:
        _require_fleet_capability("discover")
        if not isinstance(server, str) or len(server) > 128:
            raise ToolError("catalog selector is outside the safety boundary")
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
                "of a server's tools (also works for graph-os's OWN granular "
                "tools held back under the condensed intent-surface profile — "
                "pass their bare name, e.g. 'graph_query'). Spawns the owning "
                "child servers on first use and notifies the client that the "
                "tool list changed. Any server that can't be reached is "
                "reported in the 'failed' map instead of erroring the whole "
                "call. Set auto_unload=true for a ONE-SHOT tool: it is "
                "automatically retracted the next time it's called, so a task "
                "you only need once doesn't linger in your tool list — call "
                "load_tools again anytime to bring it back."
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
                    "auto_unload": {
                        "type": "boolean",
                        "description": "Auto-retract these tools after their NEXT call (one-shot use). Default false (stays loaded until unload_tools).",
                        "default": False,
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
                "Retract previously loaded tools to reclaim context — the other "
                "half of the load->use->unload lifecycle (CONCEPT:AU-ECO.mcp.intent-surface-tool-lifecycle). "
                "Three granularities, freely combined: 'tools' (exact names), "
                "'servers' (every tool of a fleet server, or graph-os's WHOLE "
                "condensed surface at once via servers=['graph-os']), and "
                "'toolsets' (every tool carrying one of these tags, e.g. a "
                "domain name — a bulk domain unload). The client is notified "
                "that the tool list changed. Meta-tools and always-on tools are "
                "kept regardless. Nothing is deleted — load_tools brings any of "
                "it straight back."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exact prefixed/local tool names to unload.",
                    },
                    "servers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Server names to unload entirely (fleet server, or 'graph-os' for the whole condensed surface).",
                    },
                    "toolsets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tag/domain names — unload every currently-loaded tool carrying one of these tags.",
                    },
                },
            },
            fn=_unload_tools,
        )
    )
    _register_status_tool(mcp, mux)


def attach_fleet_loader(
    mcp,
    *,
    config_path: str | None = None,
    self_server: str = "graph-os",
    embed_fn=None,
    authority_scope=None,
) -> MCPMultiplexer:
    """Attach on-demand MCP fleet-loading to an EXISTING FastMCP server (graph-os).

    graph-os serves its own KG/engine tools natively (always on). This composes the
    fleet-aggregation engine on top so the SAME server can also reach the rest of the
    MCP fleet (declared in ``mcp_config.json``) on demand — it registers the meta-tools
    ``find_tools`` / ``list_catalog`` / ``load_tools`` / ``unload_tools`` /
    ``multiplexer_status`` plus a per-session progressive-disclosure middleware. Child
    servers are mounted LAZILY (each as an isolated subprocess/HTTP session via
    :class:`~agent_utilities.mcp.child_resilience.ChildRuntime`, with its own breaker +
    concurrency limit) only when a tool is actually loaded, so the base context stays
    small. Returns the :class:`MCPMultiplexer` for lifecycle — call ``await mux.aclose()``
    on shutdown. (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog)

    All wiring here is synchronous (config parse + tool registration + middleware); no
    child is spawned at attach time, so it drops cleanly into graph-os's synchronous
    ``mcp.run(...)`` startup with no event loop.

    ``embed_fn(texts: list[str]) -> list[vector]`` (optional) makes ``find_tools``
    SEMANTIC: graph-os injects its own in-process embedding model so tool suggestions
    rank by query↔description meaning (understands intent) instead of only literal token
    overlap. Per-tool embeddings are cached; it runs off-thread. When absent, discovery
    falls back to the embedding-free token-overlap backbone (never a hard dependency).

    ``authority_scope`` is the host's trusted context manager for stdio process
    authority. GraphOS supplies its existing verified-tool scope; network calls
    keep their request-minted ambient session. The callback is never derived
    from child configuration or caller arguments.
    """
    resolved = _resolve_config_path(config_path or setting("MCP_CONFIG"))
    logger.info("graph-os fleet loader initializing")
    mux = MCPMultiplexer(resolved)
    mux._authority_scope = authority_scope
    # graph-os is the HOST server — never mount it (or the retired standalone
    # multiplexer name) as a child of itself.
    mux._skip_servers = {"mcp-multiplexer", self_server}
    if embed_fn is not None:
        mux._embed_fn = embed_fn
    mux.load_catalog()  # parse the fleet config into the catalog; spawns nothing
    _register_meta_tools(mcp, mux)
    # The always-visible surface: the meta-tools. graph-os's own tools are registered
    # natively by ``register_tool_surface`` and are always on; every OTHER server is
    # mounted on demand and made visible per session by the middleware below.
    mux._global_visible = {
        "find_tools",
        "list_catalog",
        "load_tools",
        "unload_tools",
        "multiplexer_status",
    }
    # CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse (Seam 8) — under MCP_TOOL_MODE=intent,
    # register_tool_surface has already tagged the host's own condensed/verbose
    # tools GATED_TAG; seed the session-visibility gate with those names so
    # load_tools reveals them exactly like a fleet tool (no mounting needed —
    # they are already registered local FastMCP tools, just hidden by default).
    from agent_utilities.mcp.verbose_tools import gated_tool_names

    mux._local_gated = gated_tool_names(mcp)
    # Stash the mux on the server so a local tool (e.g. the ``find`` intent verb,
    # CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse) can best-effort widen its search to the
    # whole fleet catalog without a second multiplexer instance.
    mcp._fleet_mux = mux
    mcp.add_middleware(SessionVisibilityMiddleware(mux, mcp))
    logger.info(
        "graph-os fleet loader ready: %d MCP server(s) mountable on demand via "
        "find_tools/load_tools.",
        len(mux.load_catalog()),
    )
    return mux
