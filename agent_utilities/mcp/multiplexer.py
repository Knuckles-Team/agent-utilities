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
import time
import weakref
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast
from urllib.parse import urlsplit

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware
from fastmcp.tools import FunctionTool, ToolResult
from mcp import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession

from agent_utilities.mcp.protocol_compat import mcp_types_module

if TYPE_CHECKING:
    from mcp_types import CallToolResult as MCPCallToolResult
    from mcp_types import Tool as MCPTool

# The SOLE handle on the MCP protocol types in this module, for two reasons that
# both bite here. (1) MCP SDK v2 re-homed the whole `mcp.types` namespace into the
# standalone `mcp_types` distribution, so `import mcp.types` raises ImportError at
# module scope on an SDK v2 image — and this module is the fleet loader, so that
# takes every child server down with it. `mcp_types_module()` binds whichever the
# installed SDK ships. (2) `mcp` is also a common PARAMETER name in this file (see
# `_register_forwarder`), where it shadows the package outright; a `mcp_types.X`
# attribute chain there resolves against the parameter, not the SDK.
mcp_types = mcp_types_module()

# Remote transports. The MCP SDK v2 line (>=2.0.0, pulled in by the
# `fastmcp>=4.0.0b1` floor of the `[mcp]` extra) renamed
# `streamablehttp_client` -> `streamable_http_client` and replaced its
# headers/auth/httpx_client_factory keywords with a single pre-configured
# `http_client=` — see the call site below. Both names are hard imports: the
# `[mcp]` extra can no longer resolve an SDK that lacks them, so the old
# defensive `try/except ImportError` guards only hid a real breakage (they
# turned the rename into a silent `RuntimeError: mcp SDK has no
# streamablehttp_client` on EVERY remote child, which is the whole deployed
# fleet — `deploy/mcp-fleet.registry.yml` defaults to `streamable-http`).
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client

from agent_utilities.core.capability_contract import Capability
from agent_utilities.core.config import setting
from agent_utilities.core.resource_priority import PriorityClass, priority_scope
from agent_utilities.mcp.child_resilience import (
    ChildRuntime,
    MCPChildError,
)
from agent_utilities.security.error_surface import public_error_text
from agent_utilities.security.log_redaction import redact_for_log

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
# A child catalog is metadata, not a single tool invocation.  It legitimately
# contains hundreds of independently bounded JSON Schemas, so it gets its own
# aggregate structural allowance while retaining the same byte/depth limits.
_MAX_CATALOG_NODES = 131_072
# Skills-over-MCP (CONCEPT:AU-ECO.mcp.skills-over-mcp-provider): a probed
# server's Resources may include ``skill://{name}/SKILL.md`` entries. Bounded
# the same way as the tool catalog so a hostile/misbehaving child cannot force
# an unbounded resource listing into the KG.
_MAX_DISCOVERED_SKILLS = 2_048
_SKILL_RESOURCE_RE = re.compile(r"^skill://(?P<name>[^/]+)/SKILL\.md$")
# CONCEPT:AU-ECO.mcp.cross-process-skill-harvest — a probed child's skill
# *bodies* are read back over the SAME already-open probe session, so a fleet
# skill becomes runnable in graph-os without co-installing the child's package
# (AGENTS.md "Dependency discipline"). Bounded per body and in aggregate: the
# harvest reads attacker-influenced content, so it can never be allowed to
# dominate the probe's latency or memory budget.
_MAX_SKILL_BODY_BYTES = 512 * 1024
_MAX_HARVEST_TOTAL_BYTES = 8 * 1024 * 1024
_SKILL_HARVEST_BUDGET_SEC = 120.0
# A child enforces its OWN request rate limit, and a body harvest is the most
# request-dense thing we ever do to one: probing a fleet child that serves the
# whole shared skill corpus tripped "Rate limit exceeded for client: global"
# after ~50 reads. That is the child correctly defending itself, so the harvest
# BACKS OFF and retries rather than treating a rate-limited read as a permanent
# failure (which would silently strand most of the corpus as un-runnable).
_SKILL_HARVEST_MAX_ATTEMPTS = 5
_SKILL_HARVEST_BACKOFF_SEC = 0.5
_SERVER_DISCOVERY_STOPWORDS = frozenset({"api", "mcp", "manager", "server", "service"})
# Fleet-wide concurrent-probe ceiling (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog), shared
# across overlapping ``probe_catalog`` calls via ``MCPMultiplexer._probe_semaphore``
# (one instance per multiplexer, NOT re-created per call — a per-call semaphore
# would give every new call its own fresh 16 slots regardless of how many probes
# an EARLIER call already left running, defeating the point of a shared cap).
# Measured (see docs/architecture/fleet-catalog-discovery-budget.md): a cold
# stdio child's own connect+handshake already costs ~2.7-2.9s on this hardware
# before any real work, so a 61-server fleet queued 16-wide needed 4 full waves
# to even ATTEMPT every server once — comfortably exceeding any interactive
# budget on its own, before counting genuinely slow/unreachable servers.
_PROBE_CONCURRENCY = 32
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


def _sample_child_health_gauges() -> None:
    """Refresh every live multiplexer's child gauges for a ``/metrics`` scrape.

    CONCEPT:AU-ECO.multiplexer.running-vs-dispatchable-metrics — child health is
    live-object state, not an event stream, so it must be sampled at scrape time
    or the gauges only ever move when somebody happens to call
    ``multiplexer_status``. ``status_snapshot()`` publishes the gauges itself,
    so calling it IS the sample.
    """
    for multiplexer in tuple(_LIVE_MULTIPLEXERS):
        multiplexer.status_snapshot()


def _register_child_health_sampler() -> None:
    """Hook :func:`_sample_child_health_gauges` into the metrics scrape path."""
    try:
        from agent_utilities.observability.gateway_metrics import (
            register_scrape_sampler,
        )

        register_scrape_sampler(_sample_child_health_gauges)
    except Exception as exc:
        logger.warning(
            "Could not register the multiplexer child-health metrics sampler "
            "(exception_type=%s): %s — mounted/dispatchable gauges will only "
            "refresh when multiplexer_status is called.",
            type(exc).__name__,
            redact_for_log(exc),
        )


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


def _assert_bounded_json_value(value: Any, *, max_nodes: int) -> None:
    """Reject oversized or excessively nested JSON-compatible values."""
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    byte_count = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes or depth > _MAX_DELEGATED_DEPTH:
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


def _assert_bounded_delegated_value(value: Any) -> None:
    """Reject oversized or excessively nested MCP tool arguments."""

    _assert_bounded_json_value(value, max_nodes=_MAX_DELEGATED_NODES)


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
        input_schema = getattr(tool, "input_schema", None) or {}
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
        _assert_bounded_json_value(tools, max_nodes=_MAX_CATALOG_NODES)
    except ToolError:
        raise RuntimeError("MCP child tool catalog exceeded its boundary") from None
    return tools


def _tool_catalog_digest(tools: list[MCPTool]) -> str:
    """Return a stable digest of the client-visible portion of one child catalog.

    A child generation already pays for ``tools/list`` as part of its bounded
    connection handshake.  Comparing that result here lets the multiplexer
    refresh only changed forwarding schemas, without polling a provider from
    ordinary tool calls or churning probe/embedding caches after an equivalent
    reconnect.
    """
    catalog = _bounded_tool_catalog(tools)
    for entry, tool in zip(catalog, tools, strict=True):
        meta = getattr(tool, "meta", None)
        if meta is not None:
            _assert_bounded_json_value(meta, max_nodes=_MAX_CATALOG_NODES)
            entry["meta"] = meta
    # A provider is allowed to return its catalog in a different order after a
    # reconnect.  Order is not part of a forwarding schema, so make the
    # no-op comparison insensitive to it and avoid needless cache/host-tool
    # churn on an otherwise equivalent generation.
    catalog.sort(key=lambda entry: entry["name"])
    canonical = json.dumps(
        catalog, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _bounded_skill_catalog(raw_resources: Any) -> list[dict[str, Any]]:
    """Project a probed child's Resources into its Skills-over-MCP subset.

    A fastmcp-4 ``SkillProvider``/``ClaudeSkillsProvider`` exposes each skill
    as ``skill://{name}/SKILL.md`` (+ a sibling ``_manifest`` resource this
    projection ignores — the manifest is fetched on demand, not during probe).
    Non-skill resources are silently dropped: this function answers "which
    skills does this server serve", not "list every resource". Bounded and
    validated exactly like :func:`_bounded_tool_catalog` so a hostile/
    misbehaving child cannot force an unbounded catalog into the KG.
    """
    if not isinstance(raw_resources, list | tuple):
        raise RuntimeError("MCP child resource catalog is invalid")
    if len(raw_resources) > _MAX_DISCOVERED_TOOLS:
        raise RuntimeError("MCP child resource catalog exceeded its boundary")

    skills: list[dict[str, Any]] = []
    for resource in raw_resources:
        uri = getattr(resource, "uri", None)
        uri_text = str(uri) if uri is not None else ""
        match = _SKILL_RESOURCE_RE.match(uri_text)
        if not match:
            continue
        name = match.group("name")
        description = getattr(resource, "description", "") or ""
        if (
            not isinstance(name, str)
            or not 1 <= len(name.encode("utf-8")) <= 256
            or any(ord(character) < 32 for character in name)
            or not isinstance(description, str)
        ):
            raise RuntimeError("MCP child resource catalog is invalid")
        skills.append({"name": name, "uri": uri_text, "description": description})
        if len(skills) > _MAX_DISCOVERED_SKILLS:
            raise RuntimeError("MCP child skill catalog exceeded its boundary")
    try:
        _assert_bounded_json_value(skills, max_nodes=_MAX_CATALOG_NODES)
    except ToolError:
        raise RuntimeError("MCP child skill catalog exceeded its boundary") from None
    return skills


def _resource_body_text(result: Any) -> str:
    """Extract the text payload of one ``resources/read`` result.

    CONCEPT:AU-ECO.mcp.cross-process-skill-harvest — a fastmcp-4
    ``SkillProvider`` serves ``skill://{name}/SKILL.md`` as ``text/markdown``,
    so the body arrives as a ``TextResourceContents``. A binary/blob payload is
    NOT a skill instruction body and is rejected rather than coerced, so a
    child cannot smuggle an unusable resource into the runnable set.
    """
    contents = getattr(result, "contents", None)
    if not isinstance(contents, list | tuple) or not contents:
        raise RuntimeError("skill resource returned no contents")
    parts: list[str] = []
    for item in contents:
        text = getattr(item, "text", None)
        if text is None:
            raise RuntimeError("skill resource body is not text")
        if not isinstance(text, str):
            raise RuntimeError("skill resource body is not text")
        parts.append(text)
    return "\n".join(parts)


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
        logger.debug("Abandoned provider projection failed", exc_info=True)
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
            logger.debug("Provider sandbox permission update failed", exc_info=True)
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
        # Bounded digest/revision state for the schemas currently exposed to
        # GraphOS clients.  A provider reconnect is cheap to compare against
        # this state and does not trigger a fresh provider probe per call.
        self._child_tool_digests: dict[str, str] = {}
        self._child_schema_revisions: dict[str, int] = {}
        # A schema refresh failure is fail-closed for that child only.  The
        # category is intentionally fixed-vocabulary: raw provider details
        # belong only in the server-side redacted log.
        self._child_schema_refresh_errors: dict[str, str] = {}
        # A child recovery runs on a detached supervisor task, where a request
        # context is unsafe to retain or reuse.  Queue changed exposed schemas
        # for each affected client session and deliver its standard MCP
        # ``tools/list_changed`` notification on that session's next request.
        # Entries are bounded by the existing per-session visibility state and
        # disappear with it, so an idle client cannot accumulate revisions.
        self._tool_list_change_revision = 0
        self._pending_tool_list_changes: dict[str, int] = {}
        # Incremented before a hot catalog reload tears down child runtimes so
        # a late callback from an old generation can never repopulate fresh
        # routing state with a stale declaration.
        self._catalog_epoch = 0
        # Set by ``attach_fleet_loader``.  Keeping this optional preserves the
        # standalone probe and unit-test paths, which do not own a FastMCP
        # server or live forwarders.
        self._host_mcp: Any | None = None
        self.tool_to_server: dict[
            str, tuple[str, str]
        ] = {}  # prefixed_name -> (server_name, original_name)
        self.aggregated_tools: list[MCPTool] = []
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
        # without depending on the (separately-flaky) KG live discovery. Every
        # entry carries ``probed_at`` (epoch seconds of the probe that produced
        # it) so a caller can compute truthful staleness instead of a fleet-wide
        # figure silently being served as if it were live (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog).
        self._probe_cache: dict[str, dict] = {}
        # A server probe that hasn't finished when an interactive caller's
        # budget expires is NEVER cancelled — cancelling it would also cancel
        # the ``self._probe_cache`` write at the end of :meth:`probe_server`,
        # which is exactly why an unreachable/slow server used to be re-probed
        # from scratch on EVERY subsequent call (the timed-out result was
        # discarded, not cached). Instead the task keeps running in the
        # background at its own per-server deadline; this map lets a LATER
        # call join the SAME in-flight probe instead of starting a duplicate,
        # and lets :meth:`aclose` await outstanding probes on shutdown.
        self._probe_inflight: dict[str, asyncio.Task] = {}
        # EVERY live probe task, joinable or not. ``_probe_inflight`` answers
        # "can a later call join this?" and therefore holds at most one task per
        # server; a forced re-probe deliberately is not joinable and so never
        # appears there. :meth:`aclose` needs the other question — "what is
        # still running?" — and must see forced probes too, or one can outlive
        # the multiplexer that spawned it.
        self._probe_tasks: set[asyncio.Task] = set()
        # ONE semaphore for the whole instance's lifetime (not re-created per
        # ``probe_catalog`` call) so a probe still running from an earlier
        # (possibly already-returned) call continues to count against the same
        # concurrency cap as a newer call's probes, instead of every call
        # getting its own fresh set of slots.
        self._probe_semaphore = asyncio.Semaphore(_PROBE_CONCURRENCY)
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
        # CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog — the ONE bounded eager posture.
        # Catalog server names (``MCP_ALWAYS_LOAD``) and server-qualified or
        # prefixed tool names (``MCP_ALWAYS_LOAD_TOOLS``) that are mounted on a
        # session's FIRST contact instead of costing a ``find_tools`` round
        # trip. Populated by :func:`attach_fleet_loader` from AgentConfig;
        # empty (fully lazy) for a bare multiplexer.
        self._always_load_servers: list[str] = []
        self._always_load_tool_specs: list[str] = []
        # Per-session "already attempted" marker. Eager mounting runs at most
        # once per session even when every entry failed — a fleet outage must
        # not turn every tools/list into a fresh round of doomed connections.
        self._always_load_done: dict[str, dict[str, Any]] = {}
        _LIVE_MULTIPLEXERS.add(self)
        _register_child_health_sampler()

    async def call_proxied_tool(
        self, prefixed_name: str, arguments: dict[str, Any] | None = None
    ) -> MCPCallToolResult:
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
        if server_name in self._child_schema_refresh_errors:
            return mcp_types.CallToolResult.model_validate(
                {
                    "content": [
                        mcp_types.TextContent(type="text", text="schema_refresh_failed")
                    ],
                    "isError": True,
                }
            )

        try:
            # Forward the call through the child's hardened runtime
            # (per-server concurrency limit + bounded queue).
            revision_before = self._child_schema_revisions.get(server_name, 0)
            result = await runtime.call_tool(original_name, arguments or {})
            # A reconnect can complete while the call above is waiting for the
            # runtime's ready gate.  Do not let that freshly recovered child
            # serve through a route whose outer schema could not be refreshed.
            if server_name in self._child_schema_refresh_errors:
                return mcp_types.CallToolResult.model_validate(
                    {
                        "content": [
                            mcp_types.TextContent(
                                type="text", text="schema_refresh_failed"
                            )
                        ],
                        "isError": True,
                    }
                )
            if (
                self._host_mcp is not None
                and self._child_schema_revisions.get(server_name, 0) != revision_before
            ):
                # This call supplied the request context that observed the
                # reconnect.  A detached recovery queues a durable revision;
                # use this live request to deliver it to this session.
                await self.notify_pending_tools_changed()
            _mediate_langfuse_kg_ingestion(
                child_config=child_config,
                original_name=original_name,
                arguments=arguments or {},
                result=result,
            )
            return result
        except MCPChildError as e:
            # Typed per-child failure (busy/restarting/failed/circuit-open):
            # the CALLER-facing result is deliberately just the class name (so
            # callers can branch on it) — but the server-side log keeps the
            # full message (e.g. which server, timing), which the class name
            # alone drops.
            logger.warning(
                "Child tool call rejected: %s: %s",
                type(e).__name__,
                redact_for_log(e),
            )
            return mcp_types.CallToolResult.model_validate(
                {
                    "content": [
                        mcp_types.TextContent(type="text", text=type(e).__name__)
                    ],
                    "isError": True,
                }
            )
        except Exception as e:
            return mcp_types.CallToolResult.model_validate(
                {
                    "content": [
                        mcp_types.TextContent(type="text", text=public_error_text(e))
                    ],
                    "isError": True,
                }
            )

    def _admit_runtime_policy_tools(
        self,
        server_name: str,
        policy: Any,
        tools: list[MCPTool],
        *,
        record_state: bool = True,
    ) -> list[MCPTool]:
        """Apply live catalog admission and fingerprinting fail closed."""

        catalog = [
            {
                "annotations": getattr(tool, "annotations", None),
                "inputSchema": tool.input_schema,
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
        if record_state:
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
                # D-MTT-1: `_svc_auth` is a local `httpx.Auth` (see
                # `child_auth`'s docstring); `sse_client`'s `auth` param is
                # typed `httpx2.Auth | None` (fastmcp's vendored SDK v2 HTTP
                # client, a distinct package from this repo's own `httpx` —
                # see `agent_utilities/mcp/httpx_boundary.py`). Coerce at
                # this boundary rather than passing the foreign-typed object
                # straight through.
                from agent_utilities.mcp.httpx_boundary import coerce_httpx2_auth

                transport = sse_client(
                    url,
                    headers=headers,
                    auth=coerce_httpx2_auth(_svc_auth),
                    httpx_client_factory=_secure_httpx_factory,
                )
            else:
                # MCP SDK v2 takes the already-built client instead of
                # headers/auth/httpx_client_factory, so the security-hardened
                # client (pinned TLS trust, DNS-pinned egress, no ambient
                # proxy, no redirects) is constructed here and handed over.
                # It is entered on the stack BEFORE the transport so teardown
                # closes the transport first and the client second; the SDK
                # deliberately does not close a caller-provided client.
                http_client = _secure_httpx_factory(headers=headers, auth=_svc_auth)
                await stack.enter_async_context(http_client)
                transport = streamable_http_client(url, http_client=http_client)
            # streamable-http and sse both yield (read, write); SDK v2 dropped
            # streamable-http's third `get_session_id` element. Take the first
            # two streams either way.
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
    ) -> tuple[str, ChildRuntime, list[MCPTool], dict] | None:
        """Starts a single child server, registers its exit stack on success, and returns its tools and runtime."""
        # Preserve the catalog generation that authorized this spawn.  A hot
        # reload may run while the child is handshaking; later callbacks from
        # that retired runtime must not re-populate the new catalog's state.
        catalog_epoch = self._catalog_epoch
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
                    record_state=catalog_epoch == self._catalog_epoch,
                )
            return sessions, tools

        # Service-authenticated remote children must recycle their session before
        # the bearer's TTL elapses (the result stream is authed once at connect
        # and then wedges on expiry); derive that lifetime from the token TTL.
        session_max_age: float | None = None
        if is_remote:
            from agent_utilities.mcp.client_credentials import service_session_max_age

            session_max_age = service_session_max_age(cfg.get("headers"))
        runtime: ChildRuntime

        async def _on_generation(live_tools: list[Any]) -> None:
            await self._refresh_child_tools(
                server_name,
                runtime,
                cfg,
                catalog_epoch,
                cast("list[MCPTool]", live_tools),
            )

        runtime = ChildRuntime(
            server_name,
            cfg,
            connect=_connect,
            session_max_age=session_max_age,
            on_generation=_on_generation,
        )
        try:
            tools = await runtime.start()
        except TimeoutError:
            logger.error("MCP child startup timed out")
            _close_policy()
            return None
        except Exception as exc:
            logger.error(
                "Failed to start MCP child: %s: %s",
                type(exc).__name__,
                redact_for_log(exc),
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
                logger.error(
                    "Failed to read MCP config: %s: %s",
                    type(exc).__name__,
                    redact_for_log(exc),
                )
                content = ""
            if content:
                try:
                    # Parse the persistent document literally. Runtime references
                    # are resolved only at the exact child boundary that consumes
                    # them, so secret values never enter this catalog wholesale.
                    config_data = json.loads(content)
                except Exception as exc:
                    logger.error(
                        "Failed to parse MCP config: %s: %s",
                        type(exc).__name__,
                        redact_for_log(exc),
                    )
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
                # LangfuseTrustError.category AND .reason are both small,
                # fixed-vocabulary labels drawn from a closed set (see the
                # class docstring: "a stable, non-sensitive trust failure" --
                # LangfuseTrustError.__init__ rejects any reason outside its
                # _CATEGORIES map). Read both into locals before logging so
                # this out-of-boundary logger's static exception-redaction
                # check can see neither is the raw exception object, and log
                # .reason verbatim rather than through redact_for_log: that
                # helper is for genuinely sensitive runtime values (paths,
                # endpoints), and hashing an already-safe fixed-vocabulary
                # string only destroys the diagnostic detail this log line
                # exists to carry.
                category = exc.category
                reason = exc.reason
                logger.error(
                    "Native Langfuse MCP disabled: %s configuration invalid (%s)",
                    category,
                    reason,
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
            # Never surface a self-entry as a mountable child, regardless of the name it
            # is filed under. ``skip`` covers it by NAME ("graph-os"), but a fresh
            # ``MCPMultiplexer`` built off the raw config (e.g. by ``_fleet_server_url``)
            # has not had ``attach_fleet_loader`` widen ``skip``. Match by the process's
            # own advertised identity (config-driven) so the gateway's own endpoint is
            # never dialed as if it were a fleet child — it is fronted in-process instead.
            from agent_utilities.base_utilities import (
                is_loopback_url as _is_self_mcp_url,
            )

            if _is_self_mcp_url(str(cfg.get("url") or "")):
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
                    # See the analogous native_langfuse_mcp_config() handler
                    # above: category and reason are both fixed-vocabulary,
                    # non-sensitive labels read into locals before logging
                    # (reason logged verbatim, not through redact_for_log,
                    # for the same reason given there).
                    category = exc.category
                    reason = exc.reason
                    logger.error(
                        "Langfuse MCP entry disabled: %s configuration invalid (%s)",
                        category,
                        reason,
                    )
                    continue
            self._catalog[str(server_name)] = cfg
        return self._catalog

    def reload_catalog(self) -> dict[str, dict]:
        """Discard runtime-derived fleet state and reparse the current catalog.

        Hot configuration changes must not leave a disabled child callable or a
        credential/TLS change attached to an old process.  Mux-owned host
        forwarders are removed too, so a same-named tool on the reloaded child
        cannot retain an obsolete client-visible schema.
        """
        # Invalidate callback closures before tearing down their runtimes.  A
        # delayed reconnect can then cleanly close without resurrecting stale
        # routing or admission state after this catalog has been rebuilt.
        self._catalog_epoch += 1
        stale_children = tuple(
            (name, runtime, self._child_runtime_policies.get(name))
            for name, runtime in self.children.items()
        )
        stale_tool_names = set(self.tool_to_server)
        for prefixed_name in tuple(self._exposed):
            try:
                self._remove_host_forwarder(prefixed_name)
            except Exception as exc:
                # A later load can still repair the local bookkeeping.  Keep
                # the provider failure private while making it visible to the
                # operator; catalog reload must remain fail-soft for siblings.
                logger.error(
                    "Could not remove stale MCP forwarding schema "
                    "(exception_type=%s): %s",
                    type(exc).__name__,
                    redact_for_log(exc),
                )
        # ``_remove_host_forwarder`` discards each normally.  Explicitly
        # converge the marker as well when a legacy/provider failure prevented
        # removal, otherwise a fresh mount would falsely believe its new
        # forwarder had already been registered.
        self._exposed.clear()
        self.children.clear()
        self._child_runtime_policies.clear()
        self._child_policy_admitted_tools.clear()
        self._child_catalog_fingerprints.clear()
        self._child_tool_digests.clear()
        self._child_schema_revisions.clear()
        self._child_schema_refresh_errors.clear()
        self._pending_tool_list_changes.clear()
        self.sessions.clear()
        self.tool_to_server.clear()
        self.aggregated_tools.clear()
        self._probe_cache.clear()
        # Not cancelled (a hot-reload should not visibly break an in-flight
        # discovery call) — just untracked, so a NEW probe_catalog call for
        # the same server starts fresh against the reloaded config rather
        # than joining a probe that may be running against stale credentials.
        self._probe_inflight.clear()
        self._tool_embeddings.clear()
        self._prefix_map = None
        self._prefix_reverse.clear()
        self._catalog = None
        # An eager declaration is configuration-derived.  Its per-session
        # result cannot survive a hot reload or an already-connected session
        # would skip mounting the new declaration indefinitely.
        self._always_load_done.clear()
        if self._host_mcp is not None:
            # ``graph_config set`` calls :func:`invalidate_live_catalogs` for
            # every runtime setting update, including the always-load
            # declarations themselves.  Re-read their validated effective
            # values here so an already-running GraphOS instance applies the
            # new eager posture on the next request rather than only after a
            # process restart.
            self._always_load_servers = _always_load_setting(
                "mcp_always_load", "MCP_ALWAYS_LOAD"
            )
            self._always_load_tool_specs = _always_load_setting(
                "mcp_always_load_tools", "MCP_ALWAYS_LOAD_TOOLS"
            )
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

    def _prefixed_child_tools(
        self,
        server_name: str,
        tools: list[MCPTool],
        cfg: dict,
    ) -> tuple[list[MCPTool], dict[str, str]]:
        """Filter and prefix one child catalog without mutating live state."""
        disabled_tools = cfg.get("disabledTools", [])
        enabled_tools = cfg.get("enabledTools", None)
        registered: list[MCPTool] = []
        originals: dict[str, str] = {}
        for tool in tools:
            if enabled_tools is not None:
                import fnmatch

                if not any(fnmatch.fnmatch(tool.name, pat) for pat in enabled_tools):
                    logger.info("Skipping a non-whitelisted MCP child tool")
                    continue
            if disabled_tools:
                import fnmatch

                if any(fnmatch.fnmatch(tool.name, pat) for pat in disabled_tools):
                    logger.info("Skipping a disabled MCP child tool")
                    continue
            prefix = self.server_prefix(server_name)
            prefixed_name = clean_tool_name(prefix, server_name, tool.name)
            registered.append(
                mcp_types.Tool(
                    name=prefixed_name,
                    description=tool.description or "",
                    input_schema=tool.input_schema,
                    # Preserve _meta (carries FastMCP tags) so downstream
                    # visibility filtering sees the child's real tags.
                    _meta=getattr(tool, "meta", None),
                )
            )
            # ``clean_tool_name`` may strip a redundant server/prefix segment,
            # so the original child name cannot be reconstructed by splitting
            # the forwarded name.  Keep that exact routing target alongside
            # the generated outer schema.
            originals[prefixed_name] = tool.name
        return registered, originals

    def _remove_host_forwarder(self, prefixed_name: str) -> None:
        """Remove one mux-owned FastMCP forwarder without public API churn."""
        host = self._host_mcp
        if host is None:
            self._exposed.discard(prefixed_name)
            return
        provider = getattr(host, "_local_provider", None)
        remove_tool = getattr(provider, "remove_tool", None)
        if not callable(remove_tool):
            raise RuntimeError("FastMCP local provider cannot remove a forwarding tool")
        try:
            remove_tool(prefixed_name)
        except KeyError as exc:
            # FastMCP documents KeyError for an absent tool, which is an
            # idempotent hot-reload outcome only after its private registry
            # confirms that our executable forwarder is gone.  A matching
            # component here makes KeyError an SDK/provider invariant breach;
            # propagate it rather than erasing a live route from mux state.
            components = getattr(provider, "_components", None)
            if not isinstance(components, dict):
                raise RuntimeError(
                    "FastMCP forwarding registry cannot verify an absent tool"
                ) from exc
            if any(
                isinstance(component, FunctionTool) and component.name == prefixed_name
                for component in components.values()
            ):
                raise RuntimeError(
                    "FastMCP reported a forwarding tool absent while it remains registered"
                ) from exc
            logger.info(
                "MCP forwarding tool was already absent during cleanup "
                "(tool_ref=%s, exception_ref=%s)",
                redact_for_log(prefixed_name),
                redact_for_log(exc),
            )
        self._exposed.discard(prefixed_name)

    def _replace_exposed_forwarders(
        self,
        old_tools: dict[str, MCPTool],
        new_tools: dict[str, MCPTool],
    ) -> set[str]:
        """Atomically replace changed exposed schemas in FastMCP's registry.

        FastMCP 4.0.0b1 has no public batch-registration primitive: each
        ``add_tool`` immediately replaces a duplicate in its local provider.
        A two-tool recovery can therefore accept one replacement and fail the
        next.  Stage the complete component registry first, and retain an
        exact SDK-component snapshot so a partial native registration is
        restored with one provider-registry swap rather than a doomed series
        of ``add_tool`` rollback calls.
        """
        exposed = set(old_tools) & self._exposed
        changed = {
            name
            for name in exposed
            if name not in new_tools
            or _tool_catalog_digest([old_tools[name]])
            != _tool_catalog_digest([new_tools[name]])
        }
        if not changed or self._host_mcp is None:
            return set()

        replacements = sorted(name for name in changed if name in new_tools)
        removed = sorted(changed - set(replacements))
        host = self._host_mcp
        provider = getattr(host, "_local_provider", None)
        components = getattr(provider, "_components", None)
        if not isinstance(components, dict):
            raise RuntimeError("FastMCP local component registry is unavailable")

        # ``FunctionTool`` construction validates every child schema before
        # the live registry is touched.  Keep these real SDK objects in the
        # staged mapping — protocol ``Tool`` models are not executable host
        # components and cannot safely stand in for them.
        forwarders = {
            name: _forwarder_component(self, new_tools[name]) for name in replacements
        }
        previous_components = dict(components)
        staged_components = dict(previous_components)
        changed_names = set(changed)
        for key, component in tuple(staged_components.items()):
            if isinstance(component, FunctionTool) and component.name in changed_names:
                staged_components.pop(key)
        for forwarder in forwarders.values():
            staged_components[forwarder.key] = forwarder

        try:
            # Preserve FastMCP's native registration path and any validation
            # it performs.  It is synchronous, so another served request
            # cannot observe an intermediate component map on this event-loop
            # turn.  If a later add fails, restore the exact prior registry
            # below without asking that same failed API to accept a rollback.
            for forwarder in forwarders.values():
                host.add_tool(forwarder)
        except Exception:
            # LocalProvider resolves lookups directly from this mapping in
            # FastMCP 4.0.0b1.  Replacing it restores the prior executable
            # FunctionTool objects atomically even when ``add_tool`` remains
            # unavailable, leaving mux maps untouched below.
            provider._components = previous_components
            raise
        # A removal is committed in the same one-step replacement, so no
        # client can be left with a half-refreshed forwarded tool set.
        provider._components = staged_components
        self._exposed.difference_update(removed)
        return changed

    def _queue_tools_changed(self, changed_names: set[str]) -> None:
        """Record one detached schema refresh for sessions that exposed it.

        A child supervisor has no valid request context: reusing the context
        inherited when it was spawned can write to a completed response stream.
        Keep the notification durable until each affected session makes its
        next request, where :meth:`notify_pending_tools_changed` can use that
        request's own outbound channel.
        """
        if not changed_names:
            return
        affected_sessions = [
            session_key
            for session_key, loaded in self._session_loaded.items()
            if changed_names & loaded
        ]
        if not affected_sessions:
            return
        self._tool_list_change_revision += 1
        for session_key in affected_sessions:
            self._pending_tool_list_changes[session_key] = (
                self._tool_list_change_revision
            )

    async def notify_pending_tools_changed(self) -> bool:
        """Deliver this live session's queued ``tools/list_changed`` event.

        Returning ``False`` retains the revision for a later request; a failed
        notification is never misreported as delivered.  A newer background
        refresh that arrives while the send awaits remains queued after this
        older revision is acknowledged.
        """
        session_key = _session_key()
        revision = self._pending_tool_list_changes.get(session_key)
        if revision is None:
            return True
        if self._host_mcp is None or not await _notify_tools_changed(self._host_mcp):
            return False
        if self._pending_tool_list_changes.get(session_key) == revision:
            self._pending_tool_list_changes.pop(session_key, None)
            self.prune_session_visibility(session_key)
        return True

    def _replace_child_tools(
        self,
        server_name: str,
        tools: list[MCPTool],
        cfg: dict,
    ) -> tuple[list[MCPTool], bool]:
        """Replace cached tools for one child when its live schema changed.

        This runs only at initial mount or after a child connection generation
        has already completed ``tools/list``.  It never performs provider I/O.
        All map mutations are synchronous, and a recovering runtime keeps its
        readiness gate closed until this method returns.
        """
        refreshed, originals = self._prefixed_child_tools(server_name, tools, cfg)
        refreshed_digest = _tool_catalog_digest(refreshed)
        current = self.prefixed_tools_for_server(server_name)
        current_digest = self._child_tool_digests.get(server_name)
        if current_digest is None and current:
            current_digest = _tool_catalog_digest(current)
        if current_digest == refreshed_digest:
            self._child_tool_digests.setdefault(server_name, refreshed_digest)
            return current, False

        current_by_name = {tool.name: tool for tool in current}
        refreshed_by_name = {tool.name: tool for tool in refreshed}
        changed_exposed = self._replace_exposed_forwarders(
            current_by_name, refreshed_by_name
        )

        stale_names = {
            prefixed
            for prefixed, (owner, _original) in self.tool_to_server.items()
            if owner == server_name
        }
        self.tool_to_server = {
            prefixed: target
            for prefixed, target in self.tool_to_server.items()
            if target[0] != server_name
        }
        self.aggregated_tools = [
            tool for tool in self.aggregated_tools if tool.name not in stale_names
        ]
        for tool in refreshed:
            self.tool_to_server[tool.name] = (server_name, originals[tool.name])
        self.aggregated_tools.extend(refreshed)
        self._child_tool_digests[server_name] = refreshed_digest
        self._child_schema_revisions[server_name] = (
            self._child_schema_revisions.get(server_name, 0) + 1
        )
        self._probe_cache.pop(server_name, None)
        embedding_prefix = f"{server_name}::"
        for key in [
            key for key in self._tool_embeddings if key.startswith(embedding_prefix)
        ]:
            self._tool_embeddings.pop(key, None)

        self._queue_tools_changed(changed_exposed)

        removed = stale_names - set(refreshed_by_name)
        if removed:
            for loaded in self._session_loaded.values():
                loaded.difference_update(removed)
            for loaded in self._auto_unload.values():
                loaded.difference_update(removed)
            for session_key in tuple(self._session_loaded):
                self.prune_session_visibility(session_key)
        return refreshed, True

    async def _refresh_child_tools(
        self,
        server_name: str,
        runtime: ChildRuntime,
        cfg: dict,
        catalog_epoch: int,
        tools: list[MCPTool],
    ) -> None:
        """Publish a recovered child generation's schema before it serves calls."""
        if (
            catalog_epoch != self._catalog_epoch
            or self.children.get(server_name) is not runtime
        ):
            # A catalog hot reload already retired this runtime.  Its delayed
            # reconnect must never resurrect stale routing/schema state.
            return
        primary = runtime.primary_session
        if primary is not None:
            self.sessions[server_name] = primary
        try:
            self._replace_child_tools(server_name, tools, cfg)
        except Exception as exc:
            self._child_schema_refresh_errors[server_name] = "schema_refresh_failed"
            logger.error(
                "MCP child schema refresh failed (server=%s, exception_type=%s): %s",
                server_name,
                type(exc).__name__,
                redact_for_log(exc),
            )
        else:
            self._child_schema_refresh_errors.pop(server_name, None)

    def _register_child_result(
        self,
        server_name: str,
        payload: Any,
        tools: list[MCPTool],
        cfg: dict,
    ) -> list[MCPTool]:
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

        registered, _changed = self._replace_child_tools(server_name, tools, cfg)
        return registered

    async def _live_skills_for_server(self, server_name: str) -> list[dict]:
        """Live ``skill://`` resource listing for an already-mounted child
        (CONCEPT:AU-ECO.mcp.skills-over-mcp-provider).

        Unlike tools, a mounted child's Skills-over-MCP resources are never
        cached into an aggregation map at mount time (:meth:`_register_child_result`
        only indexes ``tools``) — so the cold-probe path's ``_probe_skills`` call
        was the ONLY place a server's skills were ever discovered, leaving an
        already-mounted server's skills invisible to ``find``/``find_tools``
        until it happened to be probed cold at least once (D-2.2-2.3-1). Reuses
        the mounted child's live primary session (no reconnect) and the same
        best-effort degrade-to-``[]`` semantics as the cold-probe path.
        """
        session = self.sessions.get(server_name)
        if session is None:
            return []
        return await self._probe_skills(server_name, session)

    async def mount_child(self, server_name: str) -> list[MCPTool]:
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
        catalog_epoch = self._catalog_epoch
        result = await self._start_child(server_name, cfg)
        if not isinstance(result, tuple):
            return []
        s_name, payload, tools, r_cfg = result
        if catalog_epoch != self._catalog_epoch:
            # The async handshake raced a hot reload.  Do not register a child
            # that was authorized only by the retired catalog; its runtime
            # owns real transports and must be closed promptly.
            if isinstance(payload, ChildRuntime):
                await payload.aclose()
            return []
        return self._register_child_result(s_name, payload, tools, r_cfg)

    def prefixed_tools_for_server(self, server_name: str) -> list[MCPTool]:
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

        catalog_epoch = self._catalog_epoch
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
            if catalog_epoch != self._catalog_epoch:
                if isinstance(payload, ChildRuntime):
                    await payload.aclose()
                continue
            self._register_child_result(server_name, payload, tools, cfg)

    # ------------------------------------------------------------------
    # Always-load (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog)
    # ------------------------------------------------------------------

    def always_load_tool_owner(self, spec: str) -> tuple[str | None, str | None]:
        """Resolve one ``MCP_ALWAYS_LOAD_TOOLS`` entry to ``(server, original)``.

        Two accepted forms (see ``AgentConfig.mcp_always_load_tools``):

        * ``"<server>:<tool>"`` — server-qualified ORIGINAL tool name. Resolved
          without consulting the derived prefix map, so a prefix that shifts
          when a new server joins the catalog cannot silently break a
          configured entry. ``original`` is the child's own tool name.
        * ``"<prefix>__<tool>"`` — an aggregated prefixed name; the owner comes
          from the reverse prefix map and ``original`` is ``None`` (the spec
          IS the prefixed name).

        Returns ``(None, None)`` for an entry that resolves to no known server.
        """
        text = str(spec or "").strip()
        if not text:
            return None, None
        if ":" in text:
            server, _, original = text.partition(":")
            server = server.strip()
            original = original.strip()
            return (server or None), (original or None)
        return self._server_for_prefixed(text), None

    def prefixed_for_original(self, server_name: str, original: str) -> str | None:
        """The aggregated prefixed name a child's ORIGINAL tool name maps to.

        Only answerable once ``server_name`` is mounted (``tool_to_server`` is
        populated by :meth:`_register_child_result`); returns ``None`` before
        that, or when the child never registered a tool by that name.
        """
        for prefixed, entry in self.tool_to_server.items():
            if entry == (server_name, original):
                return prefixed
        return None

    def always_load_declared(self) -> bool:
        """True when this multiplexer has any eager always-load declaration."""
        return bool(self._always_load_servers or self._always_load_tool_specs)

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
                    "inputSchema": (tobj.input_schema if tobj else {}) or {},
                }
            )
        return out

    def _cache_probe(self, server_name: str, info: dict) -> dict:
        """Stamp ``probed_at`` (epoch seconds) and store ONE probe result.

        Every write to ``self._probe_cache`` goes through here so age/staleness
        can always be computed truthfully later — ``time.time() - probed_at`` —
        instead of a caller having to guess whether a served result is fresh
        (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog)."""
        info["probed_at"] = time.time()
        self._probe_cache[server_name] = info
        return info

    async def probe_server(
        self, server_name: str, force: bool = False, timeout: float | None = None
    ) -> dict:
        """Probe ONE catalog server for its tool list: connect → list_tools →
        release (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog). Returns (and caches) ``{"tools": [...],
        "error": str|None, "probed_at": float}``. An already-mounted child reuses
        its live tools instead of reconnecting; an unreachable server records its
        error string (so find_tools/load_tools can report *why* it is
        unavailable) rather than raising."""
        if not force and server_name in self._probe_cache:
            return self._probe_cache[server_name]

        if server_name in self.children:
            info: dict[str, Any] = {
                "tools": self._live_tools_for_server(server_name),
                "skills": await self._live_skills_for_server(server_name),
                "error": None,
            }
            return self._cache_probe(server_name, info)

        cfg = self.load_catalog().get(server_name)
        if cfg is None:
            info = {"tools": [], "error": "not in catalog"}
            return self._cache_probe(server_name, info)

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
            return self._cache_probe(server_name, info)

        async def _probe() -> tuple[list[dict], list[dict]]:
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
                    skills = await self._probe_skills(server_name, session)
                    return _bounded_tool_catalog(tools), skills
            finally:
                if runtime_policy is not None:
                    _close_runtime_child_policy(runtime_policy)
                    self._child_policy_admitted_tools.pop(server_name, None)

        try:
            tools, skills = await asyncio.wait_for(_probe(), timeout=probe_to)
            info = {"tools": tools, "skills": skills, "error": None}
        except TimeoutError:
            info = {"tools": [], "skills": [], "error": f"timeout after {probe_to:g}s"}
        except Exception as e:
            info = {"tools": [], "skills": [], "error": _format_probe_error(e)}
        return self._cache_probe(server_name, info)

    async def _probe_skills(self, server_name: str, session: Any) -> list[dict]:
        """Best-effort ``skill://`` resource enumeration for one probed session
        (CONCEPT:AU-ECO.mcp.skills-over-mcp-provider).

        Skills-over-MCP is only a fastmcp-4 server capability (draft MCP
        SEP-2640) — a fastmcp-3 or pre-skills server has no ``skill://``
        resources, and some MCP servers do not implement ``resources/list`` at
        all. Either case degrades to an empty list rather than failing the
        tool probe that already succeeded above.
        """
        try:
            result = await session.list_resources()
        except Exception as exc:  # noqa: BLE001 - resources/list is an OPTIONAL
            # MCP method; a server that doesn't implement it must still
            # contribute the tools its probe already returned. The cause IS
            # logged so a real transport failure stays diagnosable.
            logger.debug(
                "Server %s does not support skill resource discovery: %s: %s",
                server_name,
                type(exc).__name__,
                redact_for_log(exc),
            )
            return []
        try:
            skills = _bounded_skill_catalog(result.resources)
        except Exception as exc:  # noqa: BLE001 - a malformed skill catalog from
            # one server must not fail the tool probe that already succeeded.
            # The cause IS logged.
            logger.warning(
                "Server %s returned an invalid skill resource catalog: %s: %s",
                server_name,
                type(exc).__name__,
                redact_for_log(exc),
            )
            return []
        await self._harvest_skill_bodies(server_name, session, skills)
        return skills

    async def _harvest_skill_bodies(
        self, server_name: str, session: Any, skills: list[dict]
    ) -> None:
        """Read each catalogued ``skill://`` body over the OPEN probe session.

        CONCEPT:AU-ECO.mcp.cross-process-skill-harvest — the fleet's ~62
        ``agents/*`` packages are deliberately NOT co-installed into graph-os's
        serving venv (AGENTS.md "Dependency discipline"), so
        ``resolve_skill_provider_dirs()`` — an in-process
        ``importlib.metadata`` walk — structurally cannot see their skills. But
        each child ALREADY serves them as fastmcp-4 ``skill://{name}/SKILL.md``
        resources, and we ALREADY hold a live session to it. Reading the body
        here is what turns a name-only fleet ``Skill`` node into something
        :func:`~..knowledge_graph.ingestion.skill_workflow_ingest.ingest_runnable_skill`
        can promote to a runnable ``CallableResource``.

        Mutates each entry in place, adding EITHER ``instructions`` (the body)
        OR ``harvest_error`` (a named reason). Never both, and never silently
        neither: an entry without ``instructions`` carries the reason it has
        none, so the downstream promotion fails CLOSED against a named
        precondition instead of quietly skipping the skill.
        """
        harvested_bytes = 0
        deadline = time.monotonic() + _SKILL_HARVEST_BUDGET_SEC
        for entry in skills:
            uri = entry.get("uri") or ""
            if time.monotonic() >= deadline:
                entry["harvest_error"] = (
                    f"skill body harvest budget exceeded after "
                    f"{_SKILL_HARVEST_BUDGET_SEC:g}s"
                )
                continue
            if harvested_bytes >= _MAX_HARVEST_TOTAL_BYTES:
                entry["harvest_error"] = "skill body harvest exceeded its total budget"
                continue
            try:
                body = await self._read_skill_body(session, uri, deadline)
            except Exception as exc:  # noqa: BLE001 - one unreadable skill body
                # must not fail the tool probe that already succeeded. The cause
                # is recorded ON THE ENTRY (so the promotion can name it) AND
                # logged with its traceback — never discarded.
                entry["harvest_error"] = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Server %s could not serve skill body %s (%s)",
                    server_name,
                    entry.get("name", "?"),
                    type(exc).__name__,
                    exc_info=True,
                )
                continue
            encoded = len(body.encode("utf-8"))
            if not body.strip():
                entry["harvest_error"] = "server served an empty skill body"
                continue
            if encoded > _MAX_SKILL_BODY_BYTES:
                entry["harvest_error"] = "skill body exceeded its size boundary"
                continue
            harvested_bytes += encoded
            entry["instructions"] = body

    @staticmethod
    async def _read_skill_body(session: Any, uri: str, deadline: float) -> str:
        """Read one skill body, backing off while the child rate-limits us.

        Retries are bounded by BOTH an attempt count and the caller's harvest
        deadline, so a permanently-failing child costs a fixed amount of time.
        The FINAL failure is re-raised with its original cause intact — the
        caller records and logs it; nothing is swallowed.
        """
        delay = _SKILL_HARVEST_BACKOFF_SEC
        last: Exception | None = None
        for attempt in range(_SKILL_HARVEST_MAX_ATTEMPTS):
            try:
                return _resource_body_text(await session.read_resource(uri))
            except Exception as exc:  # noqa: BLE001 — retried below, then re-raised
                last = exc
                if attempt == _SKILL_HARVEST_MAX_ATTEMPTS - 1:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(delay, remaining))
                delay *= 2
        if last is None:  # pragma: no cover — the loop only exits via a failure
            raise RuntimeError("skill body read failed without a recorded cause")
        raise last

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

    def _settle_probe_task(self, server: str, task: asyncio.Task) -> None:
        """``add_done_callback`` for a background probe: release its in-flight
        slot. ``probe_server`` itself catches every failure mode it knows about
        and always caches a result, so an exception surfacing here is a
        genuinely unexpected bug, not a child's fault — it is logged with its
        cause, never swallowed, rather than left to asyncio's "exception was
        never retrieved" warning."""
        # Only retract the join slot if it is still OURS. A forced probe for the
        # same server runs as its OWN untracked task; popping unconditionally
        # would evict a DIFFERENT, still-running non-forced probe from the join
        # map, so a later call would spawn a duplicate of a probe already in
        # flight and ``aclose`` would no longer know to cancel it.
        if self._probe_inflight.get(server) is task:
            self._probe_inflight.pop(server, None)
        self._probe_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "Unexpected exception from background probe of %s: %s: %s",
                server,
                type(exc).__name__,
                redact_for_log(exc),
            )

    def _ensure_probing(
        self, server: str, *, force: bool, timeout: float | None
    ) -> asyncio.Task:
        """Return the in-flight probe task for ``server``, joining one an
        overlapping call already started instead of spawning a duplicate.

        A forced probe (explicit re-probe, e.g. boot ingestion) always starts
        fresh and is never JOINED by a later caller — ``force`` means "I
        specifically want a new answer now," which a stale in-flight task
        cannot give. It is still registered in ``_probe_tasks`` so
        :meth:`aclose` can cancel it: "not shareable" must not mean "able to
        outlive the multiplexer"."""
        if not force:
            existing = self._probe_inflight.get(server)
            if existing is not None and not existing.done():
                return existing

        async def _run() -> None:
            async with self._probe_semaphore:
                await self.probe_server(server, force=force, timeout=timeout)

        task = asyncio.create_task(_run())
        self._probe_tasks.add(task)
        if not force:
            self._probe_inflight[server] = task
        # `server` is this call's own parameter (not a mutating loop variable), so a
        # plain closure captures it correctly without the default-arg-capture trick —
        # which also lets mypy infer the callback's type against
        # ``Task.add_done_callback``'s single-argument signature.
        task.add_done_callback(lambda t: self._settle_probe_task(server, t))
        return task

    async def probe_catalog(
        self,
        force: bool = False,
        timeout: float | None = None,
        budget: float | None = None,
        servers: list[str] | tuple[str, ...] | None = None,
        priority: PriorityClass | None = None,
    ) -> dict[str, dict]:
        """Probe every catalog server concurrently (bounded) and cache the
        result, so find_tools can rank the whole fleet's real tools.

        ``budget`` bounds how long THIS CALL waits for an answer. It does
        **not** bound how long an individual probe is allowed to run — that
        is each server's own ``probe_timeout``/``timeout``, honored
        independently by :meth:`probe_server`. A server still probing when
        the budget expires is NEVER cancelled: cancelling it would also
        cancel its own ``self._probe_cache`` write, which is exactly what
        used to make an unreachable server pay its full connect cost again
        on every subsequent call, forever, instead of the cache the tool's
        own contract promises ("the first call probes the fleet; later
        calls are cached"). It keeps running toward its own deadline in the
        background — ``_probe_inflight`` lets a later call join that SAME
        probe instead of duplicating it — and each pending server gets an
        honest, individual answer here: a prior (possibly ``stale``,
        age-labelled) result if one exists, or an explicit ``pending``
        marker on its first-ever attempt. Reachable servers are never
        blocked by an unreachable one (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog).

        A long-lived multiplexer (graph-os's fleet loader) relies on
        :meth:`aclose` to cancel whatever is still in-flight at real
        shutdown. A short-lived caller wrapped in ``asyncio.run(...)``
        (e.g. ``source_sync._sync_fleet`` via ``_run_async``) gets the same
        guarantee for free — ``asyncio.run`` cancels every task still on its
        loop when the coroutine it ran returns, this multiplexer's included
        — so no caller needs to await these background probes itself.

        ``priority`` tags this call's probe tasks with a
        :class:`PriorityClass` (ORCH-1.98) for their whole background
        lifetime — pass ``PriorityClass.BACKGROUND_INGESTION`` for a broad
        catalog sweep so it yields shared-resource contention to
        interactive/orchestration work; leave unset to inherit the
        caller's ambient priority (the default — appropriate for a small,
        caller-named set of servers on the interactive path).

        ``servers`` narrows a latency-sensitive first stage without
        creating a second probe path."""
        catalog = self.load_catalog()
        candidates = (
            catalog if servers is None else (s for s in servers if s in catalog)
        )
        targets = [
            server for server in candidates if force or server not in self._probe_cache
        ]
        if not targets:
            return self._probe_cache

        if budget is not None and not 0.001 <= budget <= 300.0:
            raise ValueError("catalog probe budget is outside the safety boundary")

        scope = (
            priority_scope(priority)
            if priority is not None
            else contextlib.nullcontext()
        )
        with scope:
            tasks = {
                self._ensure_probing(server, force=force, timeout=timeout): server
                for server in targets
            }

        if budget is None:
            await asyncio.gather(*tasks, return_exceptions=True)
            return self._probe_cache

        _done, pending = await asyncio.wait(tasks, timeout=budget)
        if not pending:
            return self._probe_cache

        now = time.time()
        result = dict(self._probe_cache)
        for task in pending:
            server = tasks[task]
            prior = self._probe_cache.get(server)
            if prior is not None:
                # Only reachable via an explicit force re-probe (a non-forced
                # target is by definition not yet cached) — serve the last
                # known answer rather than a bare "unavailable", labelled so
                # the caller knows it is not this round's live result.
                stale = dict(prior)
                stale["stale"] = True
                stale["age_s"] = round(now - prior.get("probed_at", now), 3)
                result[server] = stale
            else:
                result[server] = {
                    "tools": [],
                    "error": (
                        f"still probing after {budget:g}s (no result yet) — "
                        "the probe continues in the background; call again shortly"
                    ),
                    "pending": True,
                }
        return result

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

    @staticmethod
    def _priority_catalog_servers(query: str, catalog: Mapping[str, Any]) -> list[str]:
        """Return high-confidence server-name matches for staged discovery."""

        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        ranked: list[tuple[float, int, str]] = []
        for server in catalog:
            identity_tokens = set(re.findall(r"[a-z0-9]+", server.lower()))
            identity_tokens -= _SERVER_DISCOVERY_STOPWORDS
            if not identity_tokens:
                continue
            overlap = query_tokens & identity_tokens
            coverage = len(overlap) / len(identity_tokens)
            if overlap and coverage >= 0.5:
                ranked.append((coverage, len(overlap), server))
        ranked.sort(reverse=True)
        return [server for _coverage, _overlap, server in ranked]

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
        # Tools AND skills, in ONE pass. Skills were previously skipped entirely,
        # so `semantic.get(skill, ...)` in discover_tools always returned 0.0 and
        # skills were ranked on token overlap alone while tools additionally got a
        # cosine term. That is not one capability space: whenever the embedder is
        # warm — the production condition this whole feature exists for — skills
        # were structurally under-ranked against tools for any query where intent
        # similarity matters more than literal token overlap.
        names: list[tuple[str, str]] = []  # (bare_name, cache_key)
        pending_text: list[str] = []
        pending_key: list[str] = []
        for server, info in probe.items():
            if info.get("error"):
                continue
            for kind in ("tools", "skills"):
                for entry in info.get(kind, []) or []:
                    name = entry.get("name")
                    if not name:
                        continue
                    # Namespaced by kind as well as server: a skill and a tool may
                    # legitimately share a name on the same server, and they must
                    # not share one cached embedding.
                    key = f"{server}::{kind}::{name}"
                    names.append((name, key))
                    if key not in self._tool_embeddings:
                        pending_text.append(
                            f"{name}. {entry.get('description', '')}"[:512]
                        )
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
                "find_tools embedding rerank unavailable; token-overlap only: %s: %s",
                type(exc).__name__,
                redact_for_log(exc),
            )
            return
        if not qv:
            return
        for name, key in names:
            vec = self._tool_embeddings.get(key)
            if vec:
                c = _cosine(qv, vec)
                if c > 0:
                    semantic[name] = max(semantic.get(name, 0.0), c)

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
        discovery_timeout = agent_config.mcp_dynamic_discovery_timeout
        loop = asyncio.get_running_loop()
        deadline = loop.time() + discovery_timeout
        priority_servers = self._priority_catalog_servers(query, catalog)
        if priority_servers:
            probe = await self.probe_catalog(
                budget=discovery_timeout,
                servers=priority_servers,
            )
        else:
            probe = {}
        remaining = deadline - loop.time()
        if remaining >= 0.001:
            # The broad fleet-wide sweep (every server, not just the
            # query-relevant ones already probed above) is a background
            # warm-up, not itself the interactive answer — tag it
            # BACKGROUND_INGESTION (ORCH-1.98) so it yields shared-resource
            # contention to interactive/orchestration work instead of
            # competing with it, reusing the ONE existing priority gate
            # rather than inventing a second (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog).
            probe = await self.probe_catalog(
                budget=remaining, priority=PriorityClass.BACKGROUND_INGESTION
            )

        # Semantic scores keyed by bare tool name. When graph-os injects an in-process
        # embedder (attach_fleet_loader), rank every probed tool by query↔description
        # cosine similarity (embeddings cached per tool). Absent ⇒ this stays empty and
        # the token-overlap backbone below ranks alone. This is what makes find_tools
        # understand intent ("send a message to a gitlab MR" → the gitlab tools) instead
        # of only matching literal tokens. (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog)
        semantic: dict[str, float] = {}
        remaining = deadline - loop.time()
        if remaining >= 0.001:
            try:
                await asyncio.wait_for(
                    self._embed_semantic_scores(query, probe, semantic),
                    timeout=remaining,
                )
            except TimeoutError:
                logger.warning("find_tools semantic rerank exceeded its latency budget")

        # One ranked capability space (CONCEPT:AU-KG.retrieval.unified-capability-contract):
        # fleet tools AND fleet-served skill:// resources are scored with the
        # SAME token-overlap+semantic backbone and merged into one ``ranked``
        # list, each item carrying a ``bind`` dict — the exact kwargs
        # `graph_orchestrate` needs to run it — so a caller never has to know
        # in advance whether the winning candidate is a tool or a skill.
        ranked: list[dict] = []
        unavailable: dict[str, str] = {}
        now = time.time()
        for server, info in probe.items():
            if info.get("error"):
                unavailable[server] = info["error"]
                continue
            # Truthful freshness for every surfaced tool/skill (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog):
            # a result served from a probe that ran seconds/minutes ago is
            # still labelled with its real age, not presented as if it were
            # just measured live.
            probe_age = round(now - info.get("probed_at", now), 3)
            is_stale = bool(info.get("stale"))
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
                capability = Capability(
                    kind="tool",
                    id=f"tool_{server}_{tool}",
                    name=tool,
                    description=desc,
                    score=score,
                    server=server,
                    source="fleet_probe",
                )
                ranked.append(
                    {
                        "kind": "tool",
                        "server": server,
                        "tool": tool,
                        "prefixed_name": prefixed,
                        "description": desc,
                        "score": round(score, 4),
                        "mountable": server in catalog,
                        "mounted": prefixed
                        in (loaded if loaded is not None else self._exposed),
                        "bind": capability.to_binding(),
                        "age_s": probe_age,
                        "stale": is_stale,
                    }
                )
            for entry in info.get("skills", []) or []:
                skill = entry.get("name")
                if not skill:
                    continue
                desc = entry.get("description", "")
                score = semantic.get(skill, 0.0) + self._relevance(
                    query, f"{skill} {desc}"
                )
                if score <= 0:
                    continue
                capability = Capability(
                    kind="skill",
                    id=f"skill_{server}_{skill}",
                    name=skill,
                    description=desc,
                    score=score,
                    server=server,
                    source="fleet_probe",
                )
                ranked.append(
                    {
                        "kind": "skill",
                        "server": server,
                        "skill": skill,
                        "uri": entry.get("uri", ""),
                        "description": desc,
                        "score": round(score, 4),
                        "mountable": server in catalog,
                        "mounted": False,
                        "bind": capability.to_binding(),
                        "age_s": probe_age,
                        "stale": is_stale,
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
                # Process-level fact only: the child is spawned. It does NOT
                # mean any of its tools are callable by the CALLING session —
                # that per-tool truth is the "mounted" field inside "tools"
                # below, and it is the ONLY field a caller should read to
                # decide whether it can dispatch a specific tool right now.
                "process_running": server in self.children,
                "probed": True,
                "available": info.get("error") is None,
                "error": info.get("error"),
                "age_s": round(time.time() - info.get("probed_at", time.time()), 3),
            }
            if include_tools:
                result["tools"] = [
                    {
                        "prefixed_name": prefixed_name,
                        "tool": t["name"],
                        "description": t.get("description", ""),
                        "enabled": self._tool_enabled(server, t["name"]),
                        # Session-scoped dispatch truth (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog):
                        # derived from the SAME predicate the dispatch gate
                        # (SessionVisibilityMiddleware) enforces, so this can
                        # never claim a tool is usable when a call would
                        # actually be rejected.
                        "mounted": self.tool_dispatchable(prefixed_name),
                    }
                    for t in info.get("tools", [])
                    for prefixed_name in (clean_tool_name(prefix, server, t["name"]),)
                ]
            return result

        if include_tools:
            from agent_utilities.core.config import config as agent_config

            # A whole-fleet browse is a background sweep, not a targeted
            # interactive lookup: bound it by the same interactive discovery
            # budget as find_tools (a server that never answers must not hang
            # this call indefinitely) and tag it BACKGROUND_INGESTION so it
            # yields to interactive/orchestration work (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog).
            probe = await self.probe_catalog(
                budget=agent_config.mcp_dynamic_discovery_timeout,
                priority=PriorityClass.BACKGROUND_INGESTION,
            )
        else:
            probe = dict(self._probe_cache)

        now = time.time()
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
                # Process-level fact only (the child is spawned) — NOT a claim
                # that any tool is callable by the caller's own session. See
                # "dispatchable_tools" for the truthful, session-scoped answer.
                "process_running": name in self.children,
                "probed": probed,
                "available": info.get("error") is None if probed else None,
            }
            if probed:
                entry["pending"] = bool(info.get("pending"))
                entry["stale"] = bool(info.get("stale"))
                if "probed_at" in info:
                    entry["age_s"] = round(now - info["probed_at"], 3)
            if info.get("error"):
                entry["error"] = info["error"]
            if include_tools:
                entry["tools"] = enabled_names
                if disabled_names:
                    entry["disabled_tools"] = disabled_names
                # Session-scoped dispatch truth (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog):
                # the subset of `tools` the CALLING session could actually
                # invoke right now, derived from the same predicate the
                # dispatch gate enforces (MCPMultiplexer.tool_dispatchable).
                entry["dispatchable_tools"] = [
                    pn for pn in enabled_names if self.tool_dispatchable(pn)
                ]
            servers.append(entry)
        return {
            "total_servers": len(servers),
            "total_tools": sum(s["tool_count"] for s in servers),
            "servers_running": sorted(self.children.keys()),
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
        ``failed`` maps each server OR each explicitly requested tool name that
        could not be resolved to a human-readable reason (e.g. an unreachable
        remote server, or a tool the owning server never actually registered —
        disabled by config, renamed, or dropped by its own admission policy).
        A requested tool that does not end up resolvable is NEVER silently
        left out of both ``to_expose`` and ``failed`` — the caller must be
        told exactly which of its requested names didn't make it, so the
        reported state can never claim a tool is loadable when the dispatcher
        cannot actually reach it. Does NOT touch FastMCP — registration of
        the live tools (and the list_changed notification) is the caller's
        job so this stays unit-testable.
        """
        requested_tools = list(tools or [])
        target_servers: set[str] = set(servers or [])
        unresolved_tools: list[str] = []
        for prefixed in requested_tools:
            owner = self._server_for_prefixed(prefixed)
            if owner:
                target_servers.add(owner)
            else:
                unresolved_tools.append(prefixed)

        mounted: list[str] = []
        failed: dict[str, str] = {
            name: "tool is not present in the fleet catalog"
            for name in unresolved_tools
        }
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
        # A requested tool whose owning server mounted but that never actually
        # registered (disabled by config, or dropped by the child's runtime
        # admission policy) must not vanish silently — it belongs in `failed`,
        # not in a phantom `newly_exposed`/`mounted` claim.
        for prefixed in requested_tools:
            if prefixed in self.tool_to_server or prefixed in failed:
                continue
            owner = self._server_for_prefixed(prefixed)
            if owner is not None and owner in failed:
                continue  # already explained by the server-level failure
            failed[prefixed] = (
                "tool is not registered by its owning server "
                "(disabled by config or rejected by its runtime policy)"
            )
        return mounted, to_expose, failed

    def tool_object(self, prefixed_name: str) -> MCPTool | None:
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

    def is_serving(self) -> bool:
        """Whether this instance's catalog actually names at least one server.

        CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog — checks truthiness
        (non-empty), not merely ``self._catalog is not None``: calling
        :meth:`tool_dispatchable` on ANY instance — including a
        freshly-constructed, never-served one (what ``source_sync``/a fleet
        harvest builds standalone, D-SH-6, ``reports/deferred/
        lane-skill-harvest.md``) — reaches :meth:`_server_for_prefixed`,
        which lazily calls :meth:`load_catalog` as a side effect via
        :meth:`_build_prefix_map`. That side effect turns ``self._catalog``
        from ``None`` into (at least) ``{}``, so an ``is not None`` check
        would already read ``True`` by the time this runs — it is the
        catalog being genuinely EMPTY (verified live in-pod: a harvest's
        throwaway instance resolved zero servers for both real probed tool
        names and an invented one) that distinguishes a non-serving instance,
        not whether ``load_catalog`` merely ran.
        """
        return bool(self._catalog)

    def tool_dispatchable(
        self, prefixed_name: str, *, session_key: str | None = None
    ) -> bool:
        """Whether ``prefixed_name`` is ACTUALLY callable right now for a session.

        Single source of truth for "can this session dispatch this tool" —
        the exact predicate :class:`SessionVisibilityMiddleware` enforces at
        the ``tools/call`` gate. Every status-reporting surface (``list_catalog``,
        ``find_tools``, ``load_tools``) MUST derive its "mounted"/"loaded" claims
        from this one function instead of re-deriving the same logic against a
        parallel bookkeeping structure (e.g. ``server in self.children``, which
        only reflects the CHILD PROCESS being up, not whether THIS session may
        call one of its tools) — that divergence is exactly the control-plane
        truthfulness bug this exists to close: a caller must never be told a
        tool is usable when the dispatch gate would reject it.

        D-SH-6 (``reports/deferred/lane-skill-harvest.md``): on a
        freshly-constructed instance that never loaded a catalog
        (:meth:`is_serving` is ``False``), ``_global_visible``/
        ``_local_gated``/``_server_for_prefixed`` all resolve nothing for
        EVERY name — including an invented one — so this used to fall
        through to the final "unknown to our bookkeeping" branch
        unconditionally, re-creating the exact mounted-vs-callable lie the
        reconciliation gate exists to close, one layer up. That final
        branch's premise (an out-of-band native host tool the real dispatch
        middleware would also allow through) only holds for an instance
        that is actually serving; a non-serving instance has no dispatch
        middleware running at all, so there is nothing to defer to — refuse
        rather than default-open.
        """
        if prefixed_name in self._global_visible:
            return True
        if prefixed_name in self._local_gated:
            key = session_key if session_key is not None else _session_key()
            return prefixed_name in self._session_loaded.get(key, set())
        # ``_server_for_prefixed`` resolves ownership from the catalog's
        # collision-free prefix map, which works even BEFORE the owning child
        # is mounted — so this branch also catches a probed-but-never-mounted
        # fleet tool, not just an already-mounted one.
        if self._server_for_prefixed(prefixed_name) is not None:
            # A KNOWN fleet tool (its owning server is at least catalogued,
            # whether or not it has been mounted yet) is only dispatchable
            # once ``load_tools`` actually registered a live forwarder for it
            # (``_exposed``) AND this session has it loaded. Being merely
            # catalogued/mountable is not enough — that gap (catalog says it
            # exists vs. a forwarder was ever built) is exactly what let
            # ``list_catalog``/``find_tools`` claim a tool was usable before
            # any session had loaded it.
            if prefixed_name not in self._exposed:
                return False
            key = session_key if session_key is not None else _session_key()
            return prefixed_name in self._session_loaded.get(key, set())
        if not self.is_serving():
            return False
        # Unknown to the multiplexer's own bookkeeping entirely — e.g. a
        # native host tool registered directly on the FastMCP server outside
        # the progressive-disclosure surface. Nothing here can gate it, so it
        # is unconditionally callable, matching how the dispatch middleware
        # (which only ever sees already-registered tool names) treats it.
        return True

    def prune_session_visibility(self, session_key: str) -> None:
        """Drop empty per-session visibility state after explicit retraction."""
        if not self._auto_unload.get(session_key):
            self._auto_unload.pop(session_key, None)
        if not self._session_loaded.get(session_key):
            self._auto_unload.pop(session_key, None)
            # Keep an empty visibility record only while a detached recovery
            # still owes this client a schema-removal notification. The record
            # is removed by ``notify_pending_tools_changed`` after delivery.
            if session_key not in self._pending_tool_list_changes:
                self._session_loaded.pop(session_key, None)

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

    def _mounted_tool_counts(self) -> dict[str, int]:
        """Live forwarding tools currently registered, per child server.

        ``_exposed`` holds the prefixed names registered as real FastMCP tools;
        ``tool_to_server`` resolves each back to its child. In dynamic mode a
        catalogued-but-unmounted child legitimately counts zero.
        """
        counts: dict[str, int] = {name: 0 for name in self.children}
        for prefixed in self._exposed:
            entry = self.tool_to_server.get(prefixed)
            if entry is not None:
                counts[entry[0]] = counts.get(entry[0], 0) + 1
        return counts

    def status_snapshot(self) -> dict[str, Any]:
        """Fleet health surface: per-child state, limits, load, restarts.

        CONCEPT:AU-ECO.multiplexer.running-vs-dispatchable-metrics — this is the
        ONE producer of child health, rendered two ways: the
        ``multiplexer_status`` tool returns the dict, and the Prometheus child
        gauges are published from the very same dict on the way out, so the
        scrape and the tool can never disagree. ``mounted_tools`` makes the
        process-up / tools-callable distinction explicit in the payload instead
        of leaving it implied by a child's absence from ``children``.
        """
        mounted = self._mounted_tool_counts()
        snapshot = {
            "children": {
                name: {
                    **runtime.status(),
                    "mounted_tools": mounted.get(name, 0),
                    "catalog_revision": self._child_schema_revisions.get(name, 0),
                    **(
                        {"catalog_fingerprint": self._child_catalog_fingerprints[name]}
                        if name in self._child_catalog_fingerprints
                        else {}
                    ),
                    **(
                        {
                            "catalog_refresh_error": self._child_schema_refresh_errors[
                                name
                            ]
                        }
                        if name in self._child_schema_refresh_errors
                        else {}
                    ),
                }
                for name, runtime in sorted(self.children.items())
            },
            "total_children": len(self.children),
            "total_tools": len(self.aggregated_tools),
        }
        try:
            from agent_utilities.observability.gateway_metrics import (
                publish_multiplexer_child_gauges,
            )

            publish_multiplexer_child_gauges(snapshot)
        except Exception as exc:
            # Metrics must never break the health surface — but never silently:
            # a health snapshot that stopped feeding the scrape is exactly the
            # kind of blind spot these gauges exist to remove.
            logger.warning(
                "Could not publish multiplexer child gauges (exception_type=%s): %s",
                type(exc).__name__,
                redact_for_log(exc),
            )
        return snapshot

    async def aclose(self) -> None:
        """Shut down every child runtime and direct stack registration."""
        # Background catalog probes (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog) are
        # deliberately left running past an interactive call's own budget so a
        # later call can benefit from their result — but they must not outlive
        # the multiplexer itself. Cancel them here; each already carries its
        # own try/except around the connect, so cancellation is a clean abort,
        # not a leaked subprocess/session.
        # ``_probe_tasks``, not ``_probe_inflight`` — the latter only holds the
        # JOINABLE probe per server, so cancelling from it would leave a forced
        # re-probe running after the multiplexer it belongs to is gone.
        inflight = tuple(self._probe_tasks)
        for task in inflight:
            task.cancel()
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
        self._probe_inflight.clear()
        self._probe_tasks.clear()
        policies = tuple(self._child_runtime_policies.values())
        try:
            for runtime in self.children.values():
                await runtime.aclose()
        finally:
            self._child_runtime_policies.clear()
            self._child_policy_admitted_tools.clear()
            self._child_catalog_fingerprints.clear()
            self._child_tool_digests.clear()
            self._child_schema_revisions.clear()
            self._child_schema_refresh_errors.clear()
            self._pending_tool_list_changes.clear()
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
        if bool(getattr(result, "is_error", False)):
            # ``ToolResult`` has no error bit. Returning one here silently
            # converts a child MCP failure into an outer success, so raise the
            # framework's typed error exactly as FastMCP's native proxy does.
            # Keep the public message stable and free of child response data.
            raise ToolError("delegated_child_tool_failed")
        return ToolResult(
            content=list(getattr(result, "content", []) or []),
            structured_content=getattr(result, "structured_content", None),
        )

    return _forward


def _tool_is_verbose(tool: MCPTool) -> bool:
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


def _forwarder_component(mux: MCPMultiplexer, tool: MCPTool) -> FunctionTool:
    """Build the executable FastMCP component for one aggregated child tool."""
    schema = tool.input_schema or {"type": "object", "properties": {}}
    return FunctionTool(
        name=tool.name,
        description=tool.description or "",
        parameters=schema,
        fn=_make_forwarder(mux, tool.name),
    )


def _register_forwarder(mcp, mux: MCPMultiplexer, tool: MCPTool) -> bool:
    """Register ONE aggregated child tool as a live FastMCP forwarding tool.

    Idempotent via ``mux._exposed`` so lazy mounts never double-register.
    Schema replacement is deliberately handled by
    :meth:`MCPMultiplexer._replace_exposed_forwarders`, where the complete
    FastMCP registry can be staged atomically. Returns True if FastMCP was
    asked to register a tool. Shared by eager startup and the dynamic
    ``load_tools`` meta-tool.
    """
    if tool.name in mux._exposed:
        return False
    mcp.add_tool(_forwarder_component(mux, tool))
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


async def _notify_tools_changed(mcp) -> bool:
    """Emit ``notifications/tools/list_changed`` so the client re-fetches the
    tool list after a dynamic mount/unmount.

    Returns whether the push actually reached the client. A missing request
    context (e.g. eager startup with no client attached yet) is an expected,
    silent no-op — but a LIVE client that rejected or dropped the notification
    is a real failure and must not be swallowed into a log line only the
    server operator sees: a caller (``load_tools``/``unload_tools``) that
    reported success while the client's tool list silently went stale is
    exactly the control-plane-truthfulness gap this exists to prevent
    (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog). Callers surface the
    returned flag to the agent so it can tell its own tool list may be stale
    instead of discovering it only via a later "no such tool" failure.
    """
    from fastmcp.server.dependencies import get_context

    try:
        context = get_context()
    except RuntimeError:
        # No active request context at all — not a client-facing failure.
        return False
    try:
        await context.send_notification(mcp_types.ToolListChangedNotification())
    except Exception as exc:
        logger.warning(
            "tools/list_changed notification failed to reach the client: %s: %s",
            type(exc).__name__,
            redact_for_log(exc),
        )
        return False
    return True


# Well-known ``_meta`` key an in-process caller can set on every ``tools/call``
# it issues (``fastmcp.Client.call_tool(..., meta={_LOCAL_SESSION_META_KEY: id})``)
# to explicitly identify which logical session it belongs to. Same wire key as
# ``agent_utilities.observability.correlation.SESSION_HEADER`` so a caller that
# already threads a correlation/session id through ``correlation.inject()`` /
# ``current_carrier()`` gets multiplexer session isolation for free — imported
# lazily below (not at module scope) purely to avoid an eager cross-package
# import at multiplexer load time, not because of any real cycle.
_LOCAL_SESSION_META_KEY = "x-session-id"
_MAX_LOCAL_SESSION_ID_BYTES = 256


def _explicit_local_session_key() -> str | None:
    """A caller-declared session id for a NON-HTTP (stdio/in-memory) request.

    CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog — D-W2-6: for stdio/in-memory
    transports the underlying MCP SDK (fastmcp 4.0.0b1 / mcp 2.0.0) gives NO
    stable per-connection identity at all: ``Context.session_id``, the
    low-level ``ServerSession``, and its ``Connection`` are all reconstructed
    fresh on EVERY single request, even within the same open client connection
    (verified empirically — not merely a docstring claim). That is why
    :func:`_session_key`'s non-HTTP branch cannot derive an ambient per-connection
    key the way the HTTP branch does.

    Multiple logically-distinct local callers sharing ONE process (e.g. an
    orchestrator dispatching several concurrent internal task sessions against
    the SAME in-process multiplexer/FastMCP instance) therefore used to
    collapse onto the single ``"__local_stdio__"`` bucket and see each other's
    loaded tools — a real process-global visibility leak, not a hypothetical
    one (reproduced by ``test_per_session_disclosure_isolation`` /
    ``test_list_catalog_mounted_matches_dispatch_reality_across_sessions``).

    The fix is a caller-supplied session id carried in the standard MCP
    request ``_meta`` (``Client.call_tool(..., meta={...})`` is a first-class,
    documented FastMCP mechanism for exactly this kind of contextual data —
    unlike ``Context.session_id`` it is NOT reconstructed per request, it is
    literally provided by the caller on each call). This is deliberately only
    consulted from the non-HTTP branch of :func:`_session_key`: a real HTTP
    session's key is derived from the AUTHENTICATED connection/token, and must
    never be overridable by caller-declared request metadata (that would let
    one HTTP caller simply claim another's session id). Within a single
    trusted process, callers are not adversarial to each other in this way —
    this only ever adds isolation between COOPERATING local callers, it can
    never be used to cross the HTTP trust boundary.
    """
    try:
        from fastmcp.server.dependencies import get_context

        request_context = get_context().request_context
        meta = request_context.meta if request_context is not None else None
    except Exception:
        return None
    if not isinstance(meta, Mapping):
        return None
    raw = meta.get(_LOCAL_SESSION_META_KEY)
    if not isinstance(raw, str) or not raw:
        return None
    encoded = raw.encode("utf-8")
    if len(encoded) > _MAX_LOCAL_SESSION_ID_BYTES or any(
        ord(character) < 32 for character in raw
    ):
        return None
    digest = hashlib.blake2s(encoded, key=_SESSION_KEY, digest_size=16).hexdigest()
    return f"local_{digest}"


def _session_key() -> str:
    """Stable per-connection key for session-scoped tool visibility.

    On a shared streamable-http server every client gets its own
    ``Context.session_id``; with no session context (stdio / single-client) all
    requests fall back to one key so behaviour matches the pre-Phase-5 server —
    UNLESS the caller explicitly declares its own session id (see
    :func:`_explicit_local_session_key`), in which case that takes priority so
    concurrent local callers in one process are not forced to share a bucket.

    The HTTP-context check MUST run before consulting ``Context.session_id``:
    fastmcp 4's ``Context.session_id`` no longer raises when there is no real
    (HTTP) session — for stdio/in-memory transports it silently mints a fresh
    UUID on every single call (no stable ``connection`` to cache it on), which
    would otherwise give every call in the same local session a different key.
    """
    try:
        from fastmcp.server.dependencies import (
            get_access_token,
            get_context,
            get_http_request,
        )

        get_http_request()
    except RuntimeError:
        return _explicit_local_session_key() or "__local_stdio__"
    except Exception as exc:  # noqa: BLE001 — deliberate DEBUG: this is a per-request CONTROL-FLOW probe ("is there an HTTP request context?"), not an error path. Every stdio/local call takes it, so WARNING here would emit one line per request. The cause is preserved (interpolated) and the outcome is encoded in the returned key.
        logger.debug(
            "No HTTP request context; trying next key source: %s: %s",
            type(exc).__name__,
            redact_for_log(exc),
        )
        return _explicit_local_session_key() or "__invalid_http_context__"
    try:
        sid = get_context().session_id
        if sid:
            return str(sid)
    except Exception as exc:  # noqa: BLE001 — deliberate DEBUG: same per-request probe as above, one rung down the key-source cascade (session_id -> token -> unauthenticated). Absence is the NORMAL case for an unauthenticated caller, not a failure; the cause is preserved and the cascade continues below.
        logger.debug(
            "HTTP context present but no session_id; falling back to token key: %s: %s",
            type(exc).__name__,
            redact_for_log(exc),
        )
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

    def _visible(self, name: str) -> bool:
        # Delegates to the SAME single-source-of-truth predicate every
        # status-reporting tool now uses (:meth:`MCPMultiplexer.tool_dispatchable`)
        # so what a session is TOLD it can call and what it can ACTUALLY call
        # are structurally the same computation, never two that can drift.
        return self.mux.tool_dispatchable(name)

    async def _ensure_always_loaded(self) -> None:
        """Eagerly mount the configured always-load set for this session
        (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog).

        Runs here rather than at ``attach_fleet_loader`` time because that is
        synchronous (it drops into graph-os's ``mcp.run(...)`` with no event
        loop) and children must bind to the SERVING loop. This is the first
        point inside it, so "loaded by default when graph-os first connects"
        holds for the client's very first ``tools/list``.

        Fail-soft is the whole contract: :func:`ensure_always_loaded` never
        raises, and this is a belt-and-braces second guard, because an eager
        convenience must never be able to break a request.
        """
        if not self.mux.always_load_declared():
            return
        try:
            await ensure_always_loaded(self._mcp, self.mux)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - never fail the request
            logger.error(
                "always-load pass could not run for this session; the fleet "
                "remains reachable via find_tools/load_tools "
                "(exception_type=%s): %s",
                type(exc).__name__,
                redact_for_log(exc),
            )

    async def on_list_tools(self, context, call_next):
        await self._ensure_always_loaded()
        tools = await call_next(context)
        # A child recovery is detached from the request that originally
        # mounted it.  Its schema update is queued until this real client
        # request can safely carry MCP's standard list-changed notification.
        await self.mux.notify_pending_tools_changed()
        return [t for t in tools if self._visible(t.name)]

    async def on_call_tool(self, context, call_next):
        # Before the dispatch gate: a client that calls an always-load tool
        # without a preceding tools/list must still find it dispatchable.
        await self._ensure_always_loaded()
        # A detached recovery can remove a tool between the client's cached
        # tools/list and this call.  Send that session's queued standard
        # invalidation before the gate rejects the stale name; otherwise the
        # ToolError would skip this method's later notification point and
        # strand the client on the obsolete catalog indefinitely.
        await self.mux.notify_pending_tools_changed()
        name = getattr(context.message, "name", None)
        # A tool this session hasn't loaded behaves as "unknown" until load_tools.
        if name and not self.mux.tool_dispatchable(name):
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
            self.mux.prune_session_visibility(session_key)
            if self._mcp is not None:
                await _notify_tools_changed(self._mcp)
        await self.mux.notify_pending_tools_changed()
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
    session_key = _session_key()
    loaded = mux.session_loaded(session_key)
    newly = [n for n in session_names if n not in loaded]
    loaded.update(session_names)
    if auto_unload and newly:
        mux._auto_unload.setdefault(session_key, set()).update(newly)
    # Truthfully report whether the client was actually told its tool list
    # changed. A caller whose client doesn't (yet) know a newly_exposed name
    # will see calls to it fail client-side ("no such tool") even though the
    # server-side registration succeeded — ``notified: False`` here is the
    # signal that distinguishes that from a genuine dispatch failure.
    notified = await _notify_tools_changed(mcp) if newly else True
    return {
        "mounted_servers": mounted_servers,
        "newly_exposed": newly,
        "failed": failed,
        "session_total": len(loaded),
        "total_registered": len(mux._exposed) + len(mux._local_gated),
        "auto_unload": bool(auto_unload) and newly,
        "notified": notified,
    }


class AlwaysLoadResult(TypedDict):
    """The result contract of one session's always-load pass.

    A named shape rather than a bare ``dict`` because three separate consumers
    read these keys — the middleware, ``multiplexer_status``-style reporting,
    and the regression tests — and a producer/consumer key drift here would
    silently report an always-load server as mounted when it is not
    (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog).
    """

    #: Servers whose child actually mounted, in declaration order.
    mounted_servers: list[str]
    #: Prefixed tool names newly made visible to THIS session.
    exposed: list[str]
    #: Entry (server name or tool spec) -> why it fell back to lazy discovery.
    degraded: dict[str, str]
    #: Whether the client was actually told its tool list changed.
    notified: NotRequired[bool]


def _empty_always_load_result() -> AlwaysLoadResult:
    return {"mounted_servers": [], "exposed": [], "degraded": {}}


async def _perform_always_load(
    mcp, mux: MCPMultiplexer, session_key: str
) -> AlwaysLoadResult:
    """One session's eager always-load pass. Every step is individually
    fail-soft (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog).

    Each declared server is mounted on its OWN try/except so a server that is
    missing from the catalog, unreachable, or crash-looping degrades to the
    normal lazy path and is reported in ``degraded`` — it can never prevent the
    remaining always-load entries from mounting, and it can never propagate out
    of a ``tools/list``. This is not hypothetical: a fastmcp-version mismatch
    has put dozens of fleet pods into a crash loop at once, and eager-loading
    them must not take graph-os down with them.
    """
    degraded: dict[str, str] = {}
    mounted: list[str] = []
    expose: set[str] = set()

    # Group the tool-level specs by owning server so each server is mounted once
    # whether it was named wholesale, per-tool, or both.
    per_server_tools: dict[str, list[tuple[str, str | None]]] = {}
    for spec in mux._always_load_tool_specs:
        server, original = mux.always_load_tool_owner(spec)
        if not server:
            degraded[spec] = "tool is not resolvable to any catalog server"
            logger.error(
                "always-load tool spec could not be resolved to a fleet server; "
                "it will remain lazily discoverable only (spec=%s)",
                redact_for_log(spec),
            )
            continue
        per_server_tools.setdefault(server, []).append((spec, original))

    whole = [str(s).strip() for s in mux._always_load_servers if str(s).strip()]
    for server in list(dict.fromkeys([*whole, *per_server_tools])):
        try:
            await mux.mount_child(server)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - eager mount must fail soft
            degraded[server] = _format_probe_error(exc)
            logger.error(
                "always-load server failed to mount and is DEGRADED to lazy "
                "discovery; graph-os continues serving without it "
                "(server=%s, error=%s)",
                server,
                degraded[server],
            )
            continue
        if server not in mux.children:
            degraded[server] = "could not mount (not in catalog, or unreachable)"
            logger.error(
                "always-load server did not mount and is DEGRADED to lazy "
                "discovery; graph-os continues serving without it (server=%s)",
                server,
            )
            continue
        mounted.append(server)
        if server in whole:
            # Mirror the server-level ``load_tools`` contract: expose the
            # condensed action surface only, never the verbose 1:1 tools —
            # always-load exists to save a round trip, not to flood context.
            for tool in mux.prefixed_tools_for_server(server):
                if _tool_is_verbose(tool):
                    continue
                expose.add(tool.name)
        for spec, original in per_server_tools.get(server, ()):
            prefixed = (
                mux.prefixed_for_original(server, original)
                if original is not None
                else (spec if spec in mux.tool_to_server else None)
            )
            if prefixed is None:
                degraded[spec] = (
                    "tool is not registered by its owning server "
                    "(disabled by config or rejected by its runtime policy)"
                )
                logger.error(
                    "always-load tool is absent from its mounted server and is "
                    "DEGRADED to lazy discovery (spec=%s)",
                    redact_for_log(spec),
                )
                continue
            expose.add(prefixed)

    for name in sorted(expose):
        tool_obj = mux.tool_object(name)
        if tool_obj is not None and name not in mux._exposed:
            _register_forwarder(mcp, mux, tool_obj)
    loaded = mux.session_loaded(session_key)
    newly = [n for n in sorted(expose) if n in mux.tool_to_server and n not in loaded]
    loaded.update(newly)
    notified = await _notify_tools_changed(mcp) if newly else True
    result: AlwaysLoadResult = {
        "mounted_servers": mounted,
        "exposed": newly,
        "degraded": degraded,
        "notified": notified,
    }
    if degraded:
        logger.warning(
            "graph-os always-load completed DEGRADED: %d of %d entries "
            "unavailable and left to lazy discovery",
            len(degraded),
            len(mounted) + len(degraded),
        )
    else:
        logger.info(
            "graph-os always-load ready: %d server(s), %d tool(s) pre-mounted",
            len(mounted),
            len(newly),
        )
    return result


async def ensure_always_loaded(
    mcp, mux: MCPMultiplexer, *, session_key: str | None = None
) -> AlwaysLoadResult:
    """Mount the configured always-load servers/tools for THIS session, once.

    Idempotent per session and safe under concurrency: the first caller runs
    the pass while any concurrent caller awaits the same future, so a client
    that fires ``tools/list`` and a ``tools/call`` back to back cannot start two
    mounting passes. A pass that raises (or is cancelled) never poisons the
    session — the marker is cleared so a later call can retry.

    NEVER raises. The caller is a middleware on the serving hot path; an eager
    convenience must not be able to fail a request
    (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog).
    """
    key = session_key or _session_key()
    pending = mux._always_load_done.get(key)
    if pending is not None:
        inflight = pending.get("future")
        if isinstance(inflight, asyncio.Future) and not inflight.done():
            try:
                return cast(AlwaysLoadResult, await asyncio.shield(inflight))
            except asyncio.CancelledError:
                raise
            except BaseException:  # noqa: BLE001 - never fail the request
                return _empty_always_load_result()
        settled = pending.get("result")
        if isinstance(settled, dict):
            return cast(AlwaysLoadResult, settled)
        return _empty_always_load_result()

    loop = asyncio.get_running_loop()
    barrier: asyncio.Future = loop.create_future()
    record: dict[str, Any] = {"future": barrier, "result": None}
    mux._always_load_done[key] = record
    try:
        result = await _perform_always_load(mcp, mux, key)
    except asyncio.CancelledError:
        mux._always_load_done.pop(key, None)
        if not barrier.done():
            barrier.cancel()
        raise
    except BaseException as exc:  # noqa: BLE001 - eager mount must fail soft
        logger.error(
            "graph-os always-load pass failed entirely; every declared server "
            "remains reachable through find_tools/load_tools "
            "(exception_type=%s): %s",
            type(exc).__name__,
            redact_for_log(exc),
        )
        result = _empty_always_load_result()
        result["degraded"] = {"*": _format_probe_error(exc)}
    record["result"] = result
    record["future"] = None
    if not barrier.done():
        barrier.set_result(result)
    return result


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
    session_key = _session_key()
    loaded = mux.session_loaded(session_key)
    names: set[str] = set(tools or [])
    for server in servers or []:
        if server in (mux._skip_servers or ()):
            names.update(mux._local_gated)
        else:
            names.update(t.name for t in mux.prefixed_tools_for_server(server))
    names.update(_tools_with_tag(mcp, toolsets))

    removed = [n for n in sorted(names) if n in loaded]
    auto = mux._auto_unload.get(session_key)
    for name in removed:
        loaded.discard(name)
        if auto:
            auto.discard(name)
    mux.prune_session_visibility(session_key)
    notified = await _notify_tools_changed(mcp) if removed else True
    return {"unloaded": removed, "session_total": len(loaded), "notified": notified}


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
                "tool count, tool names, and reachability. 'process_running' "
                "means the server's child process is up — it does NOT mean a "
                "tool is callable by YOU yet. Per-tool 'mounted' (drill-down) "
                "and 'dispatchable_tools' (all-servers view) are the truthful, "
                "session-scoped answer to 'can I call this right now' — call "
                "load_tools first if a tool you want isn't in either. This is "
                "the flat 'show me everything available' view (find_tools "
                "is the semantic 'find the right tool for X' search). Pass a "
                "'server' name to drill into just that one and get its full tool "
                "list with descriptions. First call probes the fleet (a few "
                "seconds); cached after."
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
                "tool list changed — check the response's 'notified' field: "
                "if false, your OWN client may not have refreshed its tool "
                "list yet, so a newly_exposed name can still fail with "
                "'no such tool' until you retry. Any server or specific tool "
                "that can't be reached/registered is reported in the 'failed' "
                "map (never silently dropped) instead of erroring the whole "
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


def _always_load_setting(field: str, alias: str) -> list[str]:
    """Read one always-load list from the effective configuration.

    The typed ``AgentConfig`` field is the source of truth — that is what
    carries the shipped defaults and the validated coercion — but it is parsed
    once, so a LIVE ``setting()`` value (a runtime ``graph_config set``, a
    ``monkeypatch.setenv``) takes precedence when present. Accepts a real list,
    a JSON array, or a comma-separated string, so ``MCP_ALWAYS_LOAD=a,b`` in a
    pod env is as valid as a JSON array in ``config.json``.

    Never raises: an unreadable or malformed value degrades to fully-lazy and
    is logged (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog).
    """
    raw: Any = None
    try:
        raw = setting(alias)
    except Exception as exc:  # noqa: BLE001 - configuration must not fail attach
        logger.error(
            "always-load setting %s unreadable (exception_type=%s): %s",
            alias,
            type(exc).__name__,
            redact_for_log(exc),
        )
        raw = None
    if raw is None:
        try:
            from agent_utilities.core.config import config as agent_config

            raw = getattr(agent_config, field, None)
        except Exception as exc:  # noqa: BLE001 - configuration must not fail attach
            logger.error(
                "always-load field %s unreadable (exception_type=%s): %s",
                field,
                type(exc).__name__,
                redact_for_log(exc),
            )
            return []
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text[:1] == "[":
            try:
                raw = json.loads(text)
            except ValueError:
                logger.error("always-load setting %s is not valid JSON", alias)
                return []
        else:
            raw = text.split(",")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set)):
        logger.error("always-load setting %s is not a list", alias)
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


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
    # Keep the host solely for lifecycle replacement of mux-owned forwarding
    # schemas after a child generation recovers.  Standalone mux/probe paths
    # deliberately leave this unset.
    mux._host_mcp = mcp
    mux._authority_scope = authority_scope
    # graph-os is the HOST server — never mount it (or the retired standalone
    # multiplexer name) as a child of itself.
    mux._skip_servers = {"mcp-multiplexer", self_server}
    if embed_fn is not None:
        mux._embed_fn = embed_fn
    mux.load_catalog()  # parse the fleet config into the catalog; spawns nothing
    # CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog — the always-load declaration is READ here
    # (synchronously, no I/O) but ACTED ON in the serving loop, on a session's
    # first request, by ``SessionVisibilityMiddleware``. Nothing is spawned at
    # attach time, so a broken always-load server cannot fail startup.
    mux._always_load_servers = _always_load_setting(
        "mcp_always_load", "MCP_ALWAYS_LOAD"
    )
    mux._always_load_tool_specs = _always_load_setting(
        "mcp_always_load_tools", "MCP_ALWAYS_LOAD_TOOLS"
    )
    if mux.always_load_declared():
        logger.info(
            "graph-os always-load declared: %d server(s), %d tool(s); mounted on "
            "a session's first request, fail-soft to lazy discovery",
            len(mux._always_load_servers),
            len(mux._always_load_tool_specs),
        )
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
