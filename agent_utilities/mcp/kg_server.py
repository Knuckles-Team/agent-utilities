#!/usr/bin/python
"""Knowledge Graph MCP Server — Thin wrapper over IntelligenceGraphEngine.

CONCEPT:AU-ECO.mcp.knowledge-graph-exposure — Knowledge Graph MCP Exposure

Exposes the internal Knowledge Graph as MCP tools for external agents
(Claude Code, Antigravity IDE, OpenCode, Devin) to query, search, and
ingest data into the shared unified KG.

Architecture:
    This module reuses the existing ``create_mcp_server()`` infrastructure
    from ``agent_utilities.mcp.server_factory`` — zero new abstractions.
    All tools delegate to ``IntelligenceGraphEngine`` methods that already
    exist in the 15-phase pipeline.

Security:
    - Read-only by default for external agents.
    - Write access requires ``kg:write`` scope via MCP auth.
    - Every write carries provenance: ``agent_id``, ``session_id``,
      ``workspace_path`` for multi-agent traceability.

Usage:
    # Start as stdio MCP server (default):
    graph-os --transport stdio

    # Start as HTTP transport:
    graph-os --transport streamable-http --host 127.0.0.1 --port 8004

Cross-IDE Discovery:
    Register in ``~/.config/agent-utilities/mcp_config.json``::

        {
          "mcpServers": {
            "graph-os": {
              "command": "graph-os",
              "args": ["--transport", "stdio"]
            }
          }
        }
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypedDict

from agent_utilities._version import __version__
from agent_utilities.core.config import setting
from agent_utilities.security.identifiers import validate_identifier

logger = logging.getLogger(__name__)


REGISTERED_TOOLS: dict[str, Any] = {}


def _build_dummy_request(path_params=None, json_body=None):
    from starlette.requests import Request

    scope: dict[str, Any] = {
        "type": "http",
        "path_params": path_params or {},
        "query_string": b"",
        "headers": [],
    }
    req = Request(scope)
    if json_body is not None:

        async def mock_json():
            return json_body

        # Intentional instance-level override of Request.json for this dummy/mock
        # request (there is no other way to fake a request body without a real ASGI
        # receive channel) — not a real Request whose .json() must stay bound.
        req.json = mock_json  # type: ignore[method-assign]
    return req


# Server-side authority for stdio MCP, minted once from configured runtime
# secret-reference/OAuth2 identity. Network requests receive their session from middleware and
# never fall back to this process authority.
_PROCESS_SESSION: Any = None
_PROCESS_SESSION_REFRESH_LOCK = threading.Lock()
_PROCESS_AUTHORITY_STOP = threading.Event()
_PROCESS_AUTHORITY_THREAD: threading.Thread | None = None

# D-SNV-5 follow-up: guards :func:`authority_keepalive_scope` against starting a
# second renewal loop for the same lease when one guarded scope nests inside
# another on the same task tree (for example an MCP tool dispatch whose tool body
# itself calls ``Orchestrator.execute_agent`` for a sub-delegation). contextvars
# propagate across ``await``, ``asyncio.ensure_future``/``create_task``, and
# ``asyncio.to_thread`` (which copies the current context into the worker thread),
# so the guard is visible to a nested scope even when the nesting crosses a
# to_thread boundary.
_AUTHORITY_KEEPALIVE_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_authority_keepalive_active", default=False
)

_CALLER_AUTHORITY_FIELDS = frozenset({"_actor", "_roles", "_tenant"})


def _reject_caller_authority(kwargs: dict[str, Any]) -> None:
    """Reject legacy tool fields that attempted to self-assert authority."""
    if _CALLER_AUTHORITY_FIELDS.intersection(kwargs):
        raise PermissionError("Caller-supplied graph authority is forbidden")


class UnsupportedToolFieldError(ValueError):
    """U-74: a caller (generically the REST twin, which forwards the raw JSON
    body as kwargs with no schema validation) supplied a field the target
    tool's signature does not accept.

    ``POST /engine/tenants`` with ``{"action": "list", "connection":
    "default"}`` used to reach ``_engine_domain_tool(action, params_json,
    graph)`` as an unfiltered ``**body`` call, raise a raw ``TypeError:
    _engine_domain_tool() got an unexpected keyword argument 'connection'``,
    and surface as an opaque HTTP 500 — indistinguishable from a real server
    fault. This is a distinct exception type (rather than reusing the
    existing ``_missing_required`` ``ValueError``) so the generic REST
    endpoint factory below can map it to a deterministic 4xx instead of the
    default 500, without changing behavior for any other error class."""


def _validate_tool_kwargs_against_signature(
    tool_name: str, tool_func: Any, kwargs: dict
) -> None:
    """Fail closed on a field the tool's own signature does not declare,
    instead of forwarding it into the call and letting a bare ``TypeError``
    surface as an internal error (U-74). A tool that declares ``**kwargs`` is
    exempted — it explicitly accepts arbitrary fields."""
    import inspect

    try:
        parameters = inspect.signature(tool_func).parameters
    except (TypeError, ValueError):  # noqa: BLE001 — signature introspection
        # failing (e.g. a builtin/C-implemented callable with no inspectable
        # signature) is not this guard's problem to solve: this validation is
        # purely an ADDITIONAL fail-fast check layered in front of the real
        # call, so skipping it here only forgoes the nicer 4xx and falls back
        # to today's behavior (any real failure still surfaces normally from
        # the call itself, e.g. as the existing bare-TypeError-to-500 path).
        return
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return
    unknown = sorted(set(kwargs) - set(parameters))
    if unknown:
        raise UnsupportedToolFieldError(
            f"Tool {tool_name!r} does not accept field(s): {', '.join(unknown)}."
        )


def _resolve_verified_scope_actor(session):
    """Resolve the ambient actor for `verified_tool_session_scope`, if any.

    Returns ``None`` when no actor is bound. Raises when a bound,
    authenticated actor disagrees with the session's own actor.
    """
    from ..security.brain_context import IdentityRequiredError, current_actor

    try:
        actor = current_actor()
    except IdentityRequiredError:
        return None
    if actor is not None and actor.authenticated and actor != session.actor:
        raise PermissionError("Verified actor and GraphSession authority differ")
    return actor


@contextlib.contextmanager
def verified_tool_session_scope():
    """Scope one served tool call to middleware/process-minted authority.

    Identity, tenant, audience, and policy revision are validated before tool
    dispatch. The error surface deliberately omits principal, tenant, token,
    endpoint, and policy values.
    """
    from ..knowledge_graph.core.session import current_session, use_session
    from ..security.brain_context import use_actor

    ambient = current_session()
    session = ambient or _PROCESS_SESSION
    if session is None:
        raise PermissionError("Verified GraphSession required")
    try:
        session.engine_verified_context()
    except PermissionError:
        raise PermissionError("Verified GraphSession authority is incomplete") from None

    actor = _resolve_verified_scope_actor(session)

    with contextlib.ExitStack() as stack:
        if ambient is None:
            stack.enter_context(use_session(session))
        if actor is None or actor != session.actor:
            stack.enter_context(use_actor(session.actor))
        yield session


async def _execute_tool(tool_name: str, **kwargs) -> Any:
    tool_func = REGISTERED_TOOLS.get(tool_name)
    if not tool_func:
        raise ValueError(f"Tool {tool_name} not registered")

    import inspect

    _reject_caller_authority(kwargs)
    # U-74: fail closed on a field the tool's signature does not declare
    # (e.g. a caller reusing the query/write `connection` selector against a
    # lifecycle tool that is action-only) BEFORE invocation, rather than
    # forwarding it and letting a raw TypeError surface as an opaque 500.
    _validate_tool_kwargs_against_signature(tool_name, tool_func, kwargs)

    # Tool functions declare params as ``name: T = Field(default=...)``. When the tool is
    # invoked through FastMCP, the schema layer resolves those defaults. Calling the raw
    # function directly here (internal callers, the REST gateway, tests) does NOT — so any
    # omitted param would be bound to the raw ``FieldInfo`` object, later blowing up with
    # "'FieldInfo' object has no attribute 'replace'" / "not JSON serializable". Resolve
    # FieldInfo defaults for omitted params so direct invocation matches the MCP behavior.
    _missing_required: list[str] = []
    try:
        from pydantic.fields import FieldInfo
        from pydantic_core import PydanticUndefined

        for _name, _param in inspect.signature(tool_func).parameters.items():
            if _name in kwargs:
                continue
            _default = _param.default
            if isinstance(_default, FieldInfo):
                _resolved = _default.default
                if _resolved is not PydanticUndefined:
                    kwargs[_name] = _resolved
                elif getattr(_default, "default_factory", None) is not None:
                    kwargs[_name] = _default.default_factory()  # type: ignore[misc, call-arg]
                else:
                    # Required param (Field with no default) omitted: without this the
                    # raw FieldInfo would bind and later blow up deep in the tool with a
                    # cryptic "'FieldInfo' object has no attribute 'strip'". Fail loud
                    # with the actual missing-arg name instead.
                    _missing_required.append(_name)
    except Exception:  # noqa: BLE001 — never let default-resolution break dispatch
        pass
    if _missing_required:
        raise ValueError(
            f"Tool {tool_name!r} missing required argument(s): "
            f"{', '.join(_missing_required)}."
        )

    import asyncio

    # Dispatch isolation (CONCEPT:AU-ECO.mcp.gateway-dispatch-isolation): most graph_*/
    # engine_* tools are SYNC and do blocking engine I/O. Running them inline blocks the ONE
    # gateway asyncio loop, so a single hung/misbehaving tool call (an uncompiled engine
    # surface, a bad action, a wedged backend) freezes the whole graph-os child and
    # disconnects EVERY connected MCP client. Run sync tools on a worker thread and bound
    # every call with a timeout so a hung tool FAILS LOUD and frees the loop instead of taking
    # the gateway down. The timeout is > the delegation wall-clock so execute_agent isn't
    # killed. Threads propagate the current contextvars (actor/session) via to_thread.
    _TOOL_CALL_TIMEOUT_S = 320.0

    # Dispatch isolation (CONCEPT:AU-ECO.mcp.gateway-dispatch-isolation): most graph_*/
    # engine_* tools are SYNC and do blocking engine I/O. Running them inline blocks the ONE
    # gateway asyncio loop, so a single hung/misbehaving tool call (an uncompiled engine
    # surface, a bad action, a wedged backend) freezes the whole graph-os child and
    # disconnects EVERY connected MCP client. Run sync tools on a worker thread and bound
    # every call with a timeout so a hung tool FAILS LOUD and frees the loop instead of taking
    # the gateway down. The timeout is > the delegation wall-clock so execute_agent isn't
    # killed. Threads propagate the current contextvars (actor/session) via to_thread.
    _TOOL_CALL_TIMEOUT_S = 320.0

    async def _run() -> Any:
        if inspect.iscoroutinefunction(tool_func):
            return await asyncio.wait_for(
                tool_func(**kwargs), timeout=_TOOL_CALL_TIMEOUT_S
            )
        return await asyncio.wait_for(
            asyncio.to_thread(tool_func, **kwargs), timeout=_TOOL_CALL_TIMEOUT_S
        )

    async def _guarded() -> Any:
        try:
            active_session = await _ensure_process_authority_current()
            with verified_tool_session_scope():
                # D-SNV-5: a dispatch may run for the whole _TOOL_CALL_TIMEOUT_S
                # window, but authority is otherwise checked only once, at entry.
                # authority_keepalive_scope gives a renewable (server-minted
                # process/client-credentials) session a background keepalive for
                # the dispatch's duration so a long delegation renews its own
                # authority instead of failing closed mid-flight — the SAME
                # primitive Orchestrator.execute_agent opens directly, so a
                # nested execute_agent call here (a tool that itself delegates)
                # does not start a second renewal loop. A caller-presented
                # bearer JWT has no credential_lease and is never proactively
                # renewed here — that would be forging authority the server
                # does not hold.
                async with authority_keepalive_scope(active_session):
                    return await _run()
        except TimeoutError:
            return {
                "error": (
                    f"tool {tool_name!r} exceeded the {_TOOL_CALL_TIMEOUT_S:.0f}s dispatch "
                    "timeout and was abandoned; the gateway stayed responsive (fail-loud "
                    "dispatch isolation)."
                ),
                "tool": tool_name,
                "degraded": True,
            }

    # CONCEPT:AU-ORCH.scheduling.resource-priority-edict — an MCP tool call is the
    # INTERACTIVE entry boundary (a live Claude / end-user request). Tag the whole
    # dispatch INTERACTIVE so its engine reads — including a delegation's RAG
    # context-compilation GetNodeProperties point-reads, which run on the ``to_thread``
    # worker that inherits this context — carry the top QoS class and claim the engine's
    # reserved read lane ahead of a saturating background-ingestion write storm. Tag ONLY
    # when the context is UNTAGGED: a re-entrant call from a delegated agent (ORCHESTRATION)
    # or a background task (BACKGROUND_INGESTION) keeps its own, lower class — never upgraded.
    from agent_utilities.core.resource_priority import (
        PriorityClass,
        current_priority,
        priority_scope,
    )

    if current_priority() is None:
        with priority_scope(PriorityClass.INTERACTIVE):
            return await _guarded()
    return await _guarded()


def build_native_graphos_toolset(tool_names: list[str], *, toolset_id: str) -> Any:
    """Bind registered GraphOS tools for one governed in-process delegation.

    Native delegation must not connect GraphOS back to its own HTTP endpoint or
    call raw registered functions directly.  Each generated PydanticAI tool
    preserves the registered function's schema but dispatches through
    :func:`_execute_tool`, which reuses the verified caller session, rejects
    caller-supplied authority, resolves FastMCP defaults, and preserves bounded
    dispatch isolation.  The marker is consumed by the mandatory identity-policy
    wrapper before a specialist receives the toolset.
    """

    from pydantic_ai import Tool
    from pydantic_ai.toolsets.function import FunctionToolset

    if not tool_names or len(tool_names) != len(set(tool_names)):
        raise ValueError("native GraphOS tool names must be non-empty and unique")
    if not toolset_id or len(toolset_id) > 128:
        raise ValueError("native GraphOS toolset id is invalid")

    registered: list[tuple[str, Any]] = []
    for name in tool_names:
        if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name or "") is None:
            raise ValueError("native GraphOS tool name is invalid")
        function = REGISTERED_TOOLS.get(name)
        if function is None:
            raise RuntimeError("requested native GraphOS tool is unavailable")
        registered.append((name, function))

    tools: list[Any] = []
    for name, function in registered:
        schema_source = Tool(function, name=name)

        async def dispatch(_tool_name: str = name, **kwargs: Any) -> Any:
            return await _execute_tool(_tool_name, **kwargs)

        tools.append(
            Tool.from_schema(
                dispatch,
                name=name,
                description=schema_source.description,
                json_schema=schema_source.function_schema.json_schema,
                sequential=schema_source.sequential,
            )
        )

    return FunctionToolset(
        tools,
        id=toolset_id,
        metadata={"graphos_native": True},
    )


def get_existing_disabled_batch(
    engine, node_ids: list[str], *, label: str = "CallableResource"
) -> dict[str, bool]:
    """Resolve many nodes' prior ``disabled`` flag in ONE engine round trip.

    The boot skill-ingestion loop (:func:`_ingest_skill_capabilities`) used to
    call a since-removed singular per-id helper once per skill file — N engine
    round trips for N skills against the out-of-process engine, the
    per-element-loop shape the engine's own design rule forbids ("batch,
    never per-element"). This resolves every id's prior ``disabled`` flag in
    a single ``query_cypher`` call (falling back to the in-memory
    ``graph_compute`` cache per id first, when that cache is available).

    The query is scoped to ``label`` — the verified label of every id the
    caller is passing. The default, ``:CallableResource``, is the original
    (and still sole default) caller's label: the skill runnable-resource ids
    built in :func:`_ingest_skill_capabilities` (see ``ingest_runnable_skill``'s
    ``engine._upsert_node("CallableResource", resource_id, ...)``).
    :func:`_ingest_capabilities`'s MCP-config and native-tool loops pass
    ``label="MCPServer"``/``label="NativeTool"`` respectively. An unlabeled
    ``MATCH (n)`` here would clone every node's property blob in the whole
    graph on every boot; the label makes it an indexed lookup instead. Kept
    to exactly one label per call (no unlabeled fallback) so this stays the
    single round trip the batching contract above — and
    ``test_boot_skill_ingest_batches_existing_disabled_lookup`` — require.

    Fail-closed: a lookup that could not complete (an exception from the
    in-memory cache or ``query_cypher``) marks every id still unresolved at
    that point ``True`` (disabled) in the returned mapping — never omitted,
    since call sites read a missing key as "not disabled". A genuinely absent
    id (query executed successfully, found nothing) is left absent, exactly
    as before — that is a brand-new node with no prior state, not a failure.
    """
    safe_label = validate_identifier(label, kind="label")
    result: dict[str, bool] = {}
    remaining = list(dict.fromkeys(node_ids))  # de-dupe, preserve order
    if not remaining:
        return result
    remaining = _disabled_batch_cache_lookup(engine, remaining, result)
    if not remaining:
        return result
    _disabled_batch_engine_lookup(engine, safe_label, remaining, result)
    return result


def _disabled_batch_cache_lookup(
    engine, node_ids: list[str], result: dict[str, bool]
) -> list[str]:
    """Resolve as many ids as possible from the in-memory graph-compute cache.

    Mutates ``result`` in place for ids found in the cache. Returns the ids
    still unresolved (for the caller to fall through to the engine query).
    Fails closed: on any lookup error, marks every id passed in as disabled
    in ``result`` and returns an empty list.
    """
    try:
        if hasattr(engine, "graph_compute") and hasattr(engine.graph_compute, "graph"):
            graph = engine.graph_compute.graph
            still_remaining = []
            for node_id in node_ids:
                if node_id in graph:
                    result[node_id] = bool(graph.nodes[node_id].get("disabled", False))
                else:
                    still_remaining.append(node_id)
            return still_remaining
    except Exception as exc:  # noqa: BLE001 — surfaced as fail-closed below
        logger.error(
            "get_existing_disabled_batch: in-memory cache lookup failed — "
            "failing closed for %d id(s): %s",
            len(node_ids),
            type(exc).__name__,
        )
        for node_id in node_ids:
            result[node_id] = True
        return []
    return node_ids


def _disabled_batch_engine_lookup(
    engine, safe_label: str, node_ids: list[str], result: dict[str, bool]
) -> None:
    """Resolve the remaining ids via one ``query_cypher`` round trip.

    Mutates ``result`` in place. Fails closed: on any query error, marks
    every id passed in as disabled in ``result``.
    """
    try:
        # Re-validated here (not just trusted via the ``safe_label`` name
        # from the caller) so this interpolation site is safe by
        # construction on its own — an invalid label falls through to the
        # same fail-closed handling as any other lookup error below.
        safe_label = validate_identifier(safe_label, kind="label")
        res = engine.query_cypher(
            f"MATCH (n:{safe_label}) WHERE n.id IN $node_ids "
            "RETURN n.id AS id, n.disabled AS disabled",
            {"node_ids": node_ids},
        )
        if not isinstance(res, list):
            raise TypeError(f"expected a list of rows, got {type(res).__name__}")
    except Exception as exc:  # noqa: BLE001 — surfaced as fail-closed below
        logger.error(
            "get_existing_disabled_batch(%d ids) lookup failed — failing "
            "closed (treating every unresolved id as disabled): %s",
            len(node_ids),
            type(exc).__name__,
        )
        for node_id in node_ids:
            result[node_id] = True
        return
    for row in res:
        if isinstance(row, dict) and row.get("id"):
            result[str(row["id"])] = bool(row.get("disabled", False))


def safe_json_load(s: Any) -> Any:
    if hasattr(s, "model_dump"):
        return s.model_dump()
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception as exc:  # noqa: BLE001 — non-JSON string is a normal input, not a failure
            logger.debug(
                "safe_json_load: input is not JSON, returned as-is: %s",
                type(exc).__name__,
            )
    return s


def _parse_skill_md_frontmatter(content: str) -> dict[str, Any]:
    """Extract the YAML frontmatter block from a SKILL.md's raw content.

    Falls back to a line-by-line ``key: value`` scan when the block is not
    valid YAML. Returns ``{}`` when there is no frontmatter block at all.
    """
    import re

    import yaml

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return {}
    try:
        return yaml.safe_load(match.group(1)) or {}
    except Exception:
        metadata: dict[str, Any] = {}
        for line in match.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                metadata[k.strip()] = v.strip()
        return metadata


def _skill_record_from_metadata(
    metadata: dict[str, Any], path_obj: Any
) -> dict[str, Any]:
    """Build the skill-registration record from parsed frontmatter metadata."""
    name = metadata.get("name") or path_obj.parent.name
    description = metadata.get("description") or ""
    domain = metadata.get("domain") or (
        path_obj.parent.parent.name if len(path_obj.parts) > 2 else ""
    )
    tags = metadata.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    return {
        "id": name,
        "name": name,
        "description": description,
        "domain": domain,
        "tags": tags,
        "enabled": True,
        "file_path": f"skill://{name}",
    }


def _parse_skill_md(path: Any) -> dict[str, Any]:
    """Parse YAML frontmatter from a SKILL.md file."""
    from pathlib import Path

    path_obj = Path(path)
    try:
        content = path_obj.read_text(encoding="utf-8", errors="ignore")
        metadata = _parse_skill_md_frontmatter(content)
        return _skill_record_from_metadata(metadata, path_obj)
    except Exception as e:
        logger.error("Failed to parse SKILL.md: %s", e)
        name = path_obj.parent.name
        return {
            "id": name,
            "name": name,
            "description": "",
            "domain": "",
            "tags": [],
            "enabled": True,
            "file_path": f"skill://{name}",
        }


def get_toggle_states_batch(
    engine: Any, items: list[tuple[str, str]]
) -> dict[tuple[str, str], bool]:
    """Resolve many ``(item_type, item_id)`` toggle states in ONE round trip.

    DEFECT B fix: ``get_tools_endpoint`` used to call a single-item toggle read
    once per rendered item — one synchronous Cypher round trip each. Measured
    inventory on the production pod: 254 skill files + 68 skill-graph files +
    31 builtin tools + 66 MCP servers = 350+ sequential engine round trips in
    a single request (it did not return within 90s, nor within 180s). This
    batches every id the caller is about to render into ONE query.

    Engine facts this function must respect (both confirmed live against the
    deployed engine — getting either wrong makes the batch silently match
    nothing):

    1. ``STARTS WITH`` with a ``$param`` operand does not parse on the
       deployed engine. This uses ``IN`` with an explicit id list instead —
       index-servable via the engine's node-id fast path, O(items rendered)
       rather than O(all preferences), and already the pattern used by the
       sibling batching helper :func:`get_existing_disabled_batch`.
    2. The row-governance layer (``secured_reads.row_node_ids``) requires
       every returned row to carry an identity under ``id``/``node_id``/
       ``n.id``/``_id`` — this projects ``p.id AS id`` so a real match is not
       rejected by governance and silently reported as "enabled" (see
       the data-loss note above for what this caused).

    Fail-open on a query error (an id with no resolvable state defaults to
    enabled=True), matching the previous per-item
    default — this function only changes the ROUND-TRIP COUNT and the
    governance projection, not the toggle default semantics.
    """
    seen = list(dict.fromkeys(items))  # de-dupe, preserve order
    if not engine or not seen:
        return dict.fromkeys(seen, True)

    pref_id_by_key = {key: f"preference:toggle:{key[0]}:{key[1]}" for key in seen}
    pref_ids = list(pref_id_by_key.values())
    res = _toggle_batch_query(engine, pref_ids, seen)
    if res is None:
        return dict.fromkeys(seen, True)

    value_by_pref_id: dict[str, Any] = {}
    for row in res:
        if isinstance(row, dict) and row.get("id"):
            value_by_pref_id[str(row["id"])] = row.get("value")

    result: dict[tuple[str, str], bool] = {}
    for key in seen:
        value = value_by_pref_id.get(pref_id_by_key[key])
        result[key] = True if value is None else value == "enabled"
    return result


def _toggle_batch_query(
    engine: Any, pref_ids: list[str], seen: list[tuple[str, str]]
) -> list[Any] | None:
    """Run the batched ``Preference`` lookup for :func:`get_toggle_states_batch`.

    Fail-open: returns ``None`` on any query error, so the caller defaults
    every requested item to ``enabled=True``.
    """
    try:
        res = engine.query_cypher(
            "MATCH (p:Preference) WHERE p.id IN $pref_ids "
            "RETURN p.id AS id, p.value AS value",
            {"pref_ids": pref_ids},
        )
        if not isinstance(res, list):
            raise TypeError(f"expected a list of rows, got {type(res).__name__}")
    except Exception as exc:
        logger.error(
            "get_toggle_states_batch(%d items) failed — defaulting every "
            "item to enabled=True: %s",
            len(seen),
            exc,
        )
        return None
    return res


_TOGGLE_NODE_ID_PREFIX: dict[str, str] = {
    "mcp_server": "mcp_server_",
    "builtin_tool": "native_tool_",
    "skill": "skill_",
    "skill_workflow": "skill_workflow_",
    "skill_graph": "skill_graph_",
}


def _sync_toggle_node_state(engine, node_id: str, enabled: bool) -> None:
    """Mirror a toggle's new state onto the node itself (engine + cache)."""
    engine.query_cypher(
        "MATCH (n) WHERE n.id = $node_id SET n.disabled = $disabled",
        {"node_id": node_id, "disabled": not enabled},
    )
    # Also update in-memory graph cache if active
    if (
        hasattr(engine, "graph_compute")
        and engine.graph_compute
        and hasattr(engine.graph_compute, "graph")
        and node_id in engine.graph_compute.graph.nodes
    ):
        engine.graph_compute.graph.nodes[node_id]["disabled"] = not enabled


def set_toggle_state(engine, item_type: str, item_id: str, enabled: bool):
    """Set the toggle state of an item in the KG."""
    if not engine:
        return
    pref_id = f"preference:toggle:{item_type}:{item_id}"
    try:
        from datetime import datetime

        engine.add_node(
            pref_id,
            "Preference",
            {
                "category": "toggle_state",
                "value": "enabled" if enabled else "disabled",
                "timestamp": datetime.now().isoformat(),
                "is_permanent": True,
            },
        )
        # Also update the actual node in the graph for real-time sync
        prefix = _TOGGLE_NODE_ID_PREFIX.get(item_type, "")
        node_id = f"{prefix}{item_id}" if prefix else ""
        if node_id:
            _sync_toggle_node_state(engine, node_id, enabled)
    except Exception as exc:
        logger.error("Failed to save toggle state: %s", exc)


from starlette.requests import Request
from starlette.responses import JSONResponse

from agent_utilities.security.error_surface import public_error_payload


def _external_failure_payload(
    exc: BaseException, *, code: str = "operation_failed"
) -> dict[str, str]:
    """Return a correlation-safe public error without exception details.

    Driver and tool exceptions routinely embed credentials, endpoints, local
    paths, queries, or request payloads.  External surfaces receive only a
    stable code/message and an opaque correlation identifier.  The matching
    log entry deliberately records the exception *type* only.
    """

    return public_error_payload(exc, logger=logger, code=code)


def _external_error_response(
    exc: BaseException, *, status_code: int = 500, code: str = "operation_failed"
) -> JSONResponse:
    """Build the canonical exception-safe REST error response."""

    return JSONResponse(
        _external_failure_payload(exc, code=code), status_code=status_code
    )


class _ToolsPayload(TypedDict):
    """The catalog body :func:`get_tools_endpoint` serialises.

    Named rather than ``dict[str, Any]`` so the producer/consumer seam is
    typed: the handler, its tests, and the webui contract all agree on this
    key set instead of rediscovering it from the return statement.

    FIX LANE (collapse-tool-endpoints): the original five list keys are
    UNCHANGED (same names, same per-item field names) — nothing that reads
    this route's JSON body needs to change. ``section_status`` is new and
    purely additive (CONCEPT:AU-KG.ingest.fleet-catalog-relational-tables):
    each of the five sections below now degrades independently on its own
    read failure (``"unavailable"``) instead of the whole request failing
    closed, and this map is how a caller tells "genuinely zero items" apart
    from "this section's source could not be read this time".
    """

    mcp_tools: list[dict[str, Any]]
    builtin_tools: list[dict[str, Any]]
    skills: list[dict[str, Any]]
    skill_graphs: list[dict[str, Any]]
    skill_workflows: list[dict[str, Any]]
    section_status: dict[str, str]


# Bound on how many pages of one fleet-catalog ``kind`` this route will drain
# via registry_api's own keyset-paginated ``_authorized_page`` (100 rows per
# page, see ``registry_api._MAX_LIMIT``) before giving up on that section for
# this request. Mirrors the same defensive drain-cap idea
# ``agent_webui.api_extensions._read_fleet_catalog`` already applies to the
# identical read path (its own comment there measured ~9 pages to drain 841
# ``skills`` rows) — 25 pages is headroom above that observed size without
# letting one pathological catalog hang this request forever.
_TOOLS_CATALOG_DRAIN_MAX_PAGES = 25


def _read_catalog_kind_sync(
    kind: str, *, require_discovery_binding: bool
) -> list[dict[str, Any]]:
    """Drain one fleet-catalog ``kind`` through registry_api's OWN
    tenant/principal-scoped, fail-closed authorized-read path — the exact
    same private functions ``agent_webui.api_extensions._read_fleet_catalog``
    already reuses in-process for ``/api/enhanced/tools`` (see that
    function's docstring). This never re-derives tenant scoping, redaction,
    or SQL construction; it is a thin synchronous drain loop on top of
    ``_authorized_page``.

    Synchronous and blocking (a unix-socket engine RPC per page) by design:
    the caller, :func:`_build_tools_payload_sync`, already runs entirely
    inside a worker thread via ``asyncio.to_thread`` from
    :func:`get_tools_endpoint` — calling registry_api's own ASYNC wrapper
    (``_offload_catalog_call``, which itself does ``asyncio.to_thread``)
    from here would require a running event loop that this thread does not
    have. Calling the sync ``_authorized_page``/``_authorized_count``
    directly is therefore both correct and simpler here.

    Raises whatever ``_require_catalog_authority``/``_authorized_page``
    raise (``PermissionError``, ``registry_api.CatalogUnavailable``, or any
    other exception the engine surfaces) — the caller is responsible for
    catching this per-section and recording ``section_status``, matching
    every other section's independent-degrade contract in this function.
    """
    from ..gateway.registry_api import (
        _KIND_SPECS,
        _MAX_LIMIT,
        _authorized_page,
        _get_catalog_engine,
        _require_catalog_authority,
        _row_key,
    )

    tenant, principal, grant_digests = _require_catalog_authority(
        require_discovery_binding=require_discovery_binding
    )
    engine = _get_catalog_engine()
    spec = _KIND_SPECS[kind]
    rows: list[dict[str, Any]] = []
    after: tuple[str, str] | None = None
    for _page_num in range(_TOOLS_CATALOG_DRAIN_MAX_PAGES):
        page = _authorized_page(
            kind,
            tenant=tenant,
            principal=principal,
            grant_digests=grant_digests,
            query="",
            after=after,
            limit=_MAX_LIMIT,
            engine=engine,
        )
        if not page:
            break
        rows.extend(page)
        if len(page) < _MAX_LIMIT:
            break
        after = _row_key(spec, page[-1])
    return rows


def _gather_mcp_catalog_entries() -> tuple[list[tuple[str, dict[str, Any]]], str]:
    """Gather (name, catalog_row) pairs for the fleet catalog's ``servers`` kind.

    Section 1 of :func:`_build_tools_payload_sync` — see that function's
    docstring for why ``mcp_tools`` reads the SQL fleet catalog now.
    """
    try:
        server_rows = _read_catalog_kind_sync(
            "servers", require_discovery_binding=False
        )
        mcp_entries = [
            (str(row.get("name") or ""), row) for row in server_rows if row.get("name")
        ]
        return mcp_entries, "ok"
    except Exception as e:
        logger.error("Failed to read the fleet-catalog 'servers' kind: %s", e)
        return [], "unavailable"


def _gather_builtin_tool_stems() -> tuple[list[str], str]:
    """Gather built-in agent tool file stems. No catalog equivalent exists."""
    try:
        tools_dir = Path(__file__).resolve().parents[1] / "tools"
        builtin_stems: list[str] = []
        if tools_dir.exists() and tools_dir.is_dir():
            for f in tools_dir.glob("*.py"):
                if f.name.startswith("_"):
                    continue
                builtin_stems.append(f.stem)
        return builtin_stems, "ok"
    except Exception as e:
        logger.error("Failed to scan built-in tools directory: %s", e)
        return [], "unavailable"


def _gather_skill_and_workflow_entries(
    workspace_root: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Gather Skill / Skill-Workflow entries by parsing SKILL.md files.

    Stays filesystem-sourced (not the fleet catalog) — see
    :func:`_build_tools_payload_sync`'s docstring for the domain/tags and
    freshness gap that rules the catalog out for this section.
    """
    skill_entries: list[dict[str, Any]] = []
    workflow_entries: list[dict[str, Any]] = []
    try:
        univ_skills_dir = (
            workspace_root
            / "agent-packages"
            / "skills"
            / "universal-skills"
            / "universal_skills"
            if workspace_root is not None
            else None
        )
        if univ_skills_dir is not None and univ_skills_dir.exists():
            for p in univ_skills_dir.glob("**/SKILL.md"):
                skill_info = _parse_skill_md(p)
                if "workflows" in p.parts:
                    skill_info["type"] = "Skill Workflow"
                    workflow_entries.append(skill_info)
                else:
                    skill_info["type"] = "Agent Skill"
                    skill_entries.append(skill_info)
        return skill_entries, workflow_entries, "ok"
    except Exception as e:
        logger.error("Failed to scan the universal-skills corpus: %s", e)
        return [], [], "unavailable"


def _gather_skill_graph_entries(
    workspace_root: Path | None,
) -> tuple[list[dict[str, Any]], str]:
    """Gather Skill-Graph entries by parsing SKILL.md files.

    Stays filesystem-sourced — see :func:`_build_tools_payload_sync`'s
    docstring for why (no reliable ingestion sync for this package).
    """
    graph_entries: list[dict[str, Any]] = []
    try:
        graphs_dir = (
            workspace_root
            / "agent-packages"
            / "skills"
            / "skill-graphs"
            / "skill_graphs"
            if workspace_root is not None
            else None
        )
        if graphs_dir is not None and graphs_dir.exists():
            for p in graphs_dir.glob("**/SKILL.md"):
                skill_info = _parse_skill_md(p)
                skill_info["type"] = "Skill Graph"
                graph_entries.append(skill_info)
        return graph_entries, "ok"
    except Exception as e:
        logger.error("Failed to scan the skill-graphs corpus: %s", e)
        return [], "unavailable"


def _resolve_tool_payload_toggle_states(
    engine: Any,
    mcp_entries: list[tuple[str, dict[str, Any]]],
    builtin_stems: list[str],
    workflow_entries: list[dict[str, Any]],
    skill_entries: list[dict[str, Any]],
    graph_entries: list[dict[str, Any]],
) -> dict[tuple[str, str], bool]:
    """One batched toggle-state round trip for every item about to render.

    See :func:`_build_tools_payload_sync`'s docstring for why ``mcp_tools``
    reads this Preference-node store even though it is catalog-sourced now.
    """
    toggle_keys: list[tuple[str, str]] = (
        [("mcp_server", name) for name, _row in mcp_entries]
        + [("builtin_tool", stem) for stem in builtin_stems]
        + [("skill_workflow", info["id"]) for info in workflow_entries]
        + [("skill", info["id"]) for info in skill_entries]
        + [("skill_graph", info["id"]) for info in graph_entries]
    )
    return get_toggle_states_batch(engine, toggle_keys)


def _mcp_tools_section(
    mcp_entries: list[tuple[str, dict[str, Any]]],
    toggle_states: dict[tuple[str, str], bool],
) -> list[dict[str, Any]]:
    """Build the ``mcp_tools`` payload rows from catalog entries + toggle state."""
    mcp_tools: list[dict[str, Any]] = []
    for name, row in mcp_entries:
        mcp_enabled = toggle_states[("mcp_server", name)]
        if not row.get("enabled", True):
            mcp_enabled = False
        transport = str(row.get("transport") or "")
        is_stdio = transport == "stdio"
        mcp_tools.append(
            {
                "name": name,
                "type": "MCP Server",
                "launch_mode": "subprocess" if is_stdio else "remote",
                # The catalog never stores the raw command/args (privacy —
                # see fleet_catalog_tables' module docstring); these stayed
                # opaque presence markers even before this migration.
                "command": "[configured]" if is_stdio else "",
                "args": ["[configured]"] if is_stdio else [],
                "status": "active" if mcp_enabled else "disabled",
                "enabled": mcp_enabled,
            }
        )
    return mcp_tools


def _build_tools_payload_sync(
    engine: Any, workspace_root: Path | None
) -> _ToolsPayload:
    """Synchronous body of :func:`get_tools_endpoint` — file I/O + ONE batched engine round trip.

    DEFECT A/B fix: this used to be inlined directly in the ``async def``
    handler, issuing a single-item toggle read per rendered item (350+
    sequential, BLOCKING ``query_cypher`` round trips on the production pod —
    254 skill files + 68 skill-graph files + 31 builtin tools + 66 MCP
    servers — enough that the request never returned within 180s). Every one
    of those blocking calls ran directly on the single asyncio event loop,
    starving every other request on the worker (reproduced live: concurrent
    static-asset requests timed out at the 25s ceiling while this request was
    in flight; idle baseline for those same assets is 44-112ms).

    Fixed two ways:
    1. This whole function is now synchronous, blocking, file-I/O-and-engine
       heavy code, run via ``asyncio.to_thread`` from the async endpoint
       below — matching the existing ``_execute_tool``/``asyncio.to_thread``
       pattern already used elsewhere in this file — so it never blocks the
       event loop.
    2. It gathers every ``(item_type, item_id)`` pair it is about to render
       FIRST, then resolves every toggle state in ONE
       :func:`get_toggle_states_batch` call instead of N per-item calls.

    FIX LANE (collapse-tool-endpoints) — SQL fleet catalog as the single
    source of truth: this used to build every section from a fresh
    config/filesystem scan, a second inventory of the SAME MCP/skill fleet
    that ``/api/registry/*`` and the webui BFF already read from the SQL
    fleet-catalog tables (``agent_utilities.knowledge_graph.core.
    fleet_catalog_tables``). Evidence-based per section:

    - ``mcp_tools`` (despite the key name, this has always been a list of
      *servers*, one per configured ``mcpServers`` entry — never individual
      MCP tools) now reads the catalog's ``servers`` kind
      (``mcp_servers`` table). That table is written from the SAME
      multiplexer config map (``MCPMultiplexer.load_catalog()``) this used
      to re-parse from ``mcp_config.json`` directly
      (:func:`~..knowledge_graph.core.fleet_catalog_tables.
      write_fleet_catalog`), so this is a genuine single-source collapse
      with no fidelity loss: ``command``/``args`` were already opaque
      presence markers (``"[configured]"``), never real values, and the
      catalog derives the same ``launch_mode`` split from ``transport``
      that this used to derive from ``cfg.get("command")``.
    - ``skills``/``skill_workflows``/``skill_graphs``/``builtin_tools``
      stay on their existing filesystem/KG-native sources — investigated
      and deliberately NOT moved:
        * ``builtin_tools`` has no catalog table at all. The fleet catalog
          models MCP servers/tools/prompts/resources and skills-over-MCP;
          these are native, in-process Python callables under
          ``agent_utilities/tools/*.py``, never MCP-discovered and never
          written to any catalog table.
        * ``skills``/``skill_workflows`` (local ``universal-skills``
          corpus) — the catalog's ``skills`` table CAN represent an
          individual skill's id/name/description/enabled (written by
          :func:`~..knowledge_graph.ingestion.skill_workflow_ingest.
          ingest_atomic_skills`/``ingest_skill_workflows``), but it does
          NOT store ``domain`` or ``tags`` — both real fields on this
          route's existing per-item shape, sourced from each ``SKILL.md``'s
          frontmatter. There is also no live-freshness guarantee: catalog
          rows are only as current as the last ingestion pass (an
          on-demand action or the package-install-triggered watermarked
          leg), while this filesystem glob always reflects the corpus as
          it exists on disk right now. Moving these two sections would
          silently blank ``domain``/``tags`` and could show a stale/absent
          item for anything added since the last ingest — exactly the
          "fabricate or silently drop" failure mode this fix lane was
          told to avoid, so they stay filesystem-sourced.
        * ``skill_graphs`` — the catalog schema supports this
          (``skill_type="graph"``), but unlike the atomic-skill/workflow
          legs, nothing ingests the ``skill-graphs`` package on any
          automatic/scheduled trigger (only a manual, explicit-``root``
          on-demand action reaches it) — in a typical deployment those
          catalog rows are simply absent. Serving this section from the
          catalog today would silently show an empty list where the
          on-disk corpus is real and current, so it also stays
          filesystem-sourced.
    """

    mcp_entries, mcp_status = _gather_mcp_catalog_entries()
    builtin_stems, builtin_status = _gather_builtin_tool_stems()
    skill_entries, workflow_entries, skills_status = _gather_skill_and_workflow_entries(
        workspace_root
    )
    graph_entries, graphs_status = _gather_skill_graph_entries(workspace_root)
    section_status: dict[str, str] = {
        "mcp_tools": mcp_status,
        "builtin_tools": builtin_status,
        "skills": skills_status,
        "skill_workflows": skills_status,
        "skill_graphs": graphs_status,
    }

    # ── ONE batched engine round trip for every toggle state ───────────────
    # Still the Preference-node toggle store, for EVERY section including the
    # now-catalog-sourced ``mcp_tools`` — this is deliberate, not an
    # oversight: ``POST /api/tools/toggle`` (``toggle_tool_endpoint`` below)
    # writes user enable/disable preference to this SAME store, keyed by
    # ``(item_type, item_id)``. The fleet-catalog row's own ``enabled``
    # column reflects the SERVER's configured ``disabled`` flag, not this
    # per-user toggle preference — reading catalog ``enabled`` here instead
    # would make toggling a server in the UI silently stop being reflected
    # on the next GET. The catalog row's own ``enabled`` is still honored as
    # an additional AND term below (a server force-disabled in config stays
    # disabled even if the toggle preference says otherwise), preserving the
    # original ``cfg.get("disabled")`` override semantics.
    toggle_states = _resolve_tool_payload_toggle_states(
        engine,
        mcp_entries,
        builtin_stems,
        workflow_entries,
        skill_entries,
        graph_entries,
    )

    mcp_tools = _mcp_tools_section(mcp_entries, toggle_states)

    builtin_tools = [
        {
            "name": stem,
            "type": "Built-in Tool",
            "file_path": f"tool://{stem}",
            "status": "enabled"
            if toggle_states[("builtin_tool", stem)]
            else "disabled",
            "enabled": toggle_states[("builtin_tool", stem)],
        }
        for stem in builtin_stems
    ]

    workflows = []
    for skill_info in workflow_entries:
        skill_info["enabled"] = toggle_states[("skill_workflow", skill_info["id"])]
        workflows.append(skill_info)

    skills = []
    for skill_info in skill_entries:
        skill_info["enabled"] = toggle_states[("skill", skill_info["id"])]
        skills.append(skill_info)

    graphs = []
    for skill_info in graph_entries:
        skill_info["enabled"] = toggle_states[("skill_graph", skill_info["id"])]
        graphs.append(skill_info)

    return {
        "mcp_tools": mcp_tools,
        "builtin_tools": builtin_tools,
        "skills": sorted(skills, key=lambda x: x.get("name", "").lower()),
        "skill_graphs": sorted(graphs, key=lambda x: x.get("name", "").lower()),
        "skill_workflows": sorted(workflows, key=lambda x: x.get("name", "").lower()),
        "section_status": section_status,
    }


async def get_tools_endpoint(request: Request) -> JSONResponse:
    """Retrieve all MCP tools, built-in tools, skills, skill graphs, and workflows categorized."""
    from ..knowledge_graph.core.session import resolve_session

    resolve_session(required_scope="kg:read")

    engine = _get_engine()
    workspace_value = (setting("WORKSPACE_PATH", "") or "").strip()
    workspace_root = Path(workspace_value) if workspace_value else None

    # DEFECT A fix: this endpoint used to call ``engine.query_cypher`` (a
    # plain blocking ``def``) directly and synchronously from inside an
    # ``async def`` handler, blocking the single-threaded asyncio event loop
    # for the whole request — starving every other request on the worker,
    # including static files (reproduced live). Move the blocking work off
    # the loop via ``asyncio.to_thread``, matching ``_execute_tool``'s
    # existing pattern in this file.
    payload = await asyncio.to_thread(_build_tools_payload_sync, engine, workspace_root)
    return JSONResponse(payload)


async def toggle_tool_endpoint(request: Request) -> JSONResponse:
    """Toggle the enabled status of an item (mcp_server, mcp_tool, builtin_tool, skill, etc.) in the graph."""
    from ..knowledge_graph.core.session import resolve_session

    resolve_session(required_scope="kg:write")
    try:
        data = await request.json()
    except Exception:
        data = {}

    item_type = data.get("type")
    item_id = data.get("id")
    enabled = data.get("enabled", True)

    if not item_type or not item_id:
        return JSONResponse(
            {"error": "Missing 'type' or 'id' in request body"}, status_code=400
        )

    engine = _get_engine()
    # DEFECT A audit: same blocking-call anti-pattern as `get_tools_endpoint`
    # — `set_toggle_state` calls `engine.add_node`/`engine.query_cypher`
    # (plain blocking `def`s) directly from this `async def` handler. Move it
    # off the loop via `asyncio.to_thread`, matching `_execute_tool`'s
    # existing pattern in this file.
    await asyncio.to_thread(set_toggle_state, engine, item_type, item_id, enabled)
    return JSONResponse(
        {"status": "success", "type": item_type, "id": item_id, "enabled": enabled}
    )


# ── Canonical tool ⇄ REST parity map ────────────────────────────────────────
# Single source of truth: every action-routed MCP tool in ``REGISTERED_TOOLS``
# has exactly one collapsed action-routed REST twin (POST, JSON body carries the
# ``action`` and its args). Granular CRUD sub-routes (``/graph/write/node`` etc.)
# are layered on top for fine-grained HTTP clients, but this map guarantees that
# anything callable over MCP is also callable over REST and vice versa. The
# parity contract test (tests/unit/test_gateway_mcp_parity.py) asserts this map
# stays in lockstep with REGISTERED_TOOLS so the two surfaces never drift.
ACTION_TOOL_ROUTES: dict[str, str] = {
    "graph_query": "/graph/query",
    "graph_ask": "/graph/ask",
    "graph_table": "/graph/table",
    "graph_search": "/graph/search",
    "graph_search_synthesis": "/graph/search-synthesis",
    "graph_code_nav": "/graph/code-nav",
    "graph_document_tree": "/graph/document-tree",
    "graph_write": "/graph/write",
    "graph_ingest": "/graph/ingest",
    "graph_analyze": "/graph/analyze",
    "graph_code": "/graph/code",
    "graph_research": "/graph/research",
    "graph_evaluate": "/graph/evaluate",
    "graph_explain": "/graph/explain",
    "graph_observe": "/graph/observe",
    "graph_orchestrate": "/graph/orchestrate",
    "graph_config": "/graph/config",
    "graph_configure": "/graph/configure",
    "graph_context": "/graph/context",
    "graph_feedback": "/graph/feedback",
    "graph_sessions": "/graph/sessions",
    "graph_goals": "/graph/goals",
    "graph_message": "/graph/message",
    "graph_reach": "/graph/reach",
    "graph_bus": "/graph/bus",
    "graph_secret": "/graph/secret",
    "document_process": "/document/process",
    "source_connector": "/connector/source",
    "graph_writeback": "/graph/writeback",
    "spec_ticket": "/spec/ticket",
    "concept_registry": "/concept/registry",
    "source_sync": "/source/sync",
    "source_drain": "/source/drain",
    "graph_etl": "/graph/etl",
    "ontology_property_types": "/ontology/property-types",
    "ontology_value_types": "/ontology/value-types",
    "ontology_interface": "/ontology/interface",
    "ontology_sampling_profile": "/ontology/sampling-profiles",
    "ontology_model_profile": "/ontology/model-profiles",
    "ontology_function": "/ontology/function",
    "ontology_derive": "/ontology/derive",
    "ontology_link_materialize": "/ontology/link-materialize",
    "ontology_leanix_sync": "/ontology/leanix-sync",
    "ontology_classification_claims": "/ontology/classification-claims",
    "graph_data_prep": "/data/prep",
    "ontology_repository_provenance": "/ontology/repository-provenance",
    "graph_ontology": "/graph/ontology",
    "object_edits": "/object/edits",
    "object_index": "/object/index",
    "object_permissioning": "/object/permissioning",
    "object_set": "/object/set",
    "graph_share": "/graph/share",
    "usage_query": "/usage/query",
    "ingest_sessions": "/usage/ingest-sessions",
    "research_artifact": "/research/artifact",
    "graph_loops": "/graph/loops",
    "graph_schedules": "/graph/schedules",
    "graph_feeds": "/graph/feeds",
    "graph_sandbox": "/graph/sandbox",
    "graph_runvcs": "/graph/runvcs",
    "graph_claims": "/graph/claims",
    "graph_candidate_claims": "/graph/candidate-claims",
    "skill_classify": "/skill/classify",
}

# Immutable seed used by deterministic catalog generators. Runtime registrars
# extend ``ACTION_TOOL_ROUTES`` with their own twins, but a generator must never
# inherit routes left behind by an earlier server build in the same process.
BASE_ACTION_TOOL_ROUTES = MappingProxyType(dict(ACTION_TOOL_ROUTES))


def _is_engine_dispatch_client_error(parsed: Any) -> bool:
    """U-74 (GOC-83-W05): does a parsed ``engine_<domain>`` dispatch result
    represent a CALLER-caused parameter mistake, rather than a real result or
    a genuine server-side/engine-side failure?

    ``engine_tools._dispatch`` is the ONE dispatcher every ``engine_<domain>``
    tool shares; it never raises for an unknown action name, an unknown/
    duplicate/wrong-type/missing parameter to the target EG method, or a
    malformed ``params_json`` — it catches all of those and returns a JSON
    STRING error payload instead, by design, so an MCP tool caller always gets
    data back (never an exception it has to unwrap). That is the right
    contract for the MCP surface, but the generic REST twin
    (:func:`_make_tool_endpoint`) unconditionally wrapped ANY such string in
    ``{"status": "success", ...}`` at HTTP 200 — a client-caused parameter
    mistake was indistinguishable, over REST, from a real result. This
    recognizes the two client-error shapes ``_dispatch`` emits (an unknown/
    non-callable action name — a bare ``error: str``; or an
    ``error.code == "invalid_request"`` structured payload from
    ``public_error_json`` — unknown/duplicate/wrong-type/missing parameter, or
    an undecodable ``params_json``), scoped ONLY to ``engine_*`` REST
    responses (checked by the caller) so no other tool's REST status-code
    contract changes. A ``dependency_unavailable`` (engine unreachable) or
    unclassified ``operation_failed`` payload is NOT a client mistake and is
    deliberately left at 200, matching prior behavior exactly.
    """
    if not isinstance(parsed, dict):
        return False
    error = parsed.get("error")
    if isinstance(error, str):
        # `_dispatch`'s two pre-call, action-name-level rejections: {"error":
        # "unknown action ...", "actions": [...]} / {"error": "engine_<domain>
        # has no callable action ..."} — both are a caller supplying an action
        # name the domain doesn't have, i.e. exactly as client-caused as an
        # unsupported top-level field.
        return True
    if isinstance(error, dict) and error.get("code") == "invalid_request":
        # `public_error_json(..., code="invalid_request")` — raised for an
        # undecodable `params_json`, or a `TypeError` from calling the target
        # EG method with an unknown/duplicate/wrong-type/missing argument.
        return True
    return False


def _make_tool_endpoint(tool_name: str):
    """Build a thin REST handler that dispatches a JSON body to an MCP tool.

    Both the MCP tool surface and the REST surface funnel through
    :func:`_execute_tool` against the shared in-process engine, so a handler is
    just: parse body → execute tool → wrap result. This factory is the canonical
    adapter; per-tool endpoints below that need bespoke parsing keep their own
    definitions, but every tool in :data:`ACTION_TOOL_ROUTES` without one is
    served by this.
    """

    is_engine_domain_tool = tool_name.startswith("engine_")

    async def _handler(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            res = await _execute_tool(tool_name, **body)
            parsed = safe_json_load(res)
            # U-74 (GOC-83-W05): an `engine_<domain>` action-name or
            # parameter mistake never raises (see `_is_engine_dispatch_
            # client_error`'s docstring) — surface it as a deterministic 4xx
            # instead of an HTTP 200 that hides a caller-caused failure behind
            # `"status": "success"`. Scoped to `engine_*` tools only; every
            # other tool's REST status-code contract is unchanged.
            if is_engine_domain_tool and _is_engine_dispatch_client_error(parsed):
                return JSONResponse(
                    {"status": "failed", "result": parsed}, status_code=400
                )
            if (
                tool_name == "graph_sessions"
                and isinstance(parsed, dict)
                and isinstance(parsed.get("evidence"), dict)
                and parsed["evidence"].get("ready") is False
            ):
                # Fleet health/topology are supervisory evidence, not ordinary
                # session CRUD. Preserve the shared fail-closed HTTP signal
                # while returning the exact typed evidence body used by MCP.
                return JSONResponse(
                    {"status": "unavailable", "result": parsed}, status_code=503
                )
            return JSONResponse({"status": "success", "result": parsed})
        except UnsupportedToolFieldError as e:
            # U-74: a caller-supplied field the tool doesn't accept is a
            # client-side schema mismatch, not a server fault — deterministic
            # 4xx instead of the generic 500 every other exception maps to.
            return _external_error_response(e, status_code=400, code="invalid_request")
        except Exception as e:
            return _external_error_response(e)

    _handler.__name__ = f"{tool_name}_endpoint"
    return _handler


#: The ``graph_query`` MCP tool's own documented parameters (KG-2.134 /
#: ``agent_utilities/mcp/tools/query_tools.py``'s ``graph_query`` signature).
#: Kept as an explicit allowlist so this REST twin never blind-splats an
#: arbitrary request body into ``_execute_tool`` (LANE 9 / U-74 follow-up):
#: an unrecognized field becomes an immediate, clean 4xx here instead of
#: reaching ``_execute_tool`` at all.
_GRAPH_QUERY_TOOL_FIELDS = frozenset(
    {
        "as_of",
        "connection",
        "cypher",
        "graph",
        "include_epistemic",
        "params",
        "reference_id",
        "scope",
    }
)


async def graph_query_endpoint(request: Request) -> JSONResponse:
    """REST twin of the ``graph_query`` MCP tool.

    LANE 9 fix: the tool's real parameter is ``cypher`` (see
    ``_GRAPH_QUERY_TOOL_FIELDS`` / the ``graph_query`` tool signature), but a
    plausible, naturally-expected wire name for "the query string" is
    ``query`` — and that name is not a caller mistake in this codebase: it is
    the genuine field name of the *different*, already-correct
    ``POST /api/graph/execute_cypher`` route (agent-webui's
    ``execute_cypher``, whose target ``QueryMixin.query_cypher`` really does
    take a ``query`` kwarg — see
    ``agent_utilities/knowledge_graph/orchestration/engine_query.py``), which
    ``CypherReplView.tsx``/``TemporalGraphView.tsx``/``GraphView.tsx`` all
    call. To stay compatible with a client that assumes wire-name parity
    across these two Cypher-shaped routes, ``query`` is accepted here as an
    alias for ``cypher`` — mapping at this boundary, rather than renaming the
    tool's own ``cypher`` parameter (which every existing internal caller of
    the ``graph_query`` MCP tool relies on) or forcing every REST client onto
    one spelling.

    Precedence when both are supplied: identical values collapse to one
    (no ambiguity); different values are a client error returned as a
    deterministic 4xx rather than silently preferring either field.

    This endpoint does not forward the raw request body into
    ``_execute_tool`` — only ``_GRAPH_QUERY_TOOL_FIELDS`` (plus the ``query``
    alias) are ever passed through, so a genuinely unknown field fails fast
    as a clean 4xx here instead of reaching the tool dispatch (and, on a
    build predating the ``_execute_tool``-internal
    ``_validate_tool_kwargs_against_signature`` guard, the authority/session
    bootstrap that precedes it) only to 500 later.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return JSONResponse(
            {"status": "error", "message": "request body must be a JSON object"},
            status_code=400,
        )

    result = _graph_query_request_kwargs(body)
    if not isinstance(result, dict):
        payload, status_code = result
        return JSONResponse(payload, status_code=status_code)
    kwargs = result

    try:
        res = await _execute_tool("graph_query", **kwargs)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except UnsupportedToolFieldError as e:
        # Defense-in-depth: `_GRAPH_QUERY_TOOL_FIELDS` is kept in sync with
        # the tool's real signature above, so this should be unreachable —
        # but if it ever drifts, still surface the client-caused 4xx rather
        # than the generic 500 below (U-74).
        return _external_error_response(e, status_code=400, code="invalid_request")
    except Exception as e:
        return _external_error_response(e)


def _graph_query_request_kwargs(
    body: dict[str, Any],
) -> dict[str, Any] | tuple[dict[str, Any], int]:
    """Validate + normalize a ``graph_query`` REST body into tool kwargs.

    Handles the ``query``/``cypher`` aliasing documented on
    :func:`graph_query_endpoint`. Returns ``kwargs`` (a dict) on success, or
    ``(payload, status_code)`` for a 4xx the caller should return verbatim.
    The two are distinguished by the caller with ``isinstance(result, dict)``.
    """
    query_val = body.get("query")
    cypher_val = body.get("cypher")
    if query_val is not None and cypher_val is not None and query_val != cypher_val:
        return (
            {
                "status": "error",
                "message": (
                    "both 'query' and 'cypher' were supplied with different "
                    "values; send exactly one (or identical values in both)."
                ),
            },
            400,
        )

    unknown = sorted(set(body) - _GRAPH_QUERY_TOOL_FIELDS - {"query"})
    if unknown:
        return (
            {
                "status": "error",
                "message": f"Unsupported field(s): {', '.join(unknown)}.",
            },
            400,
        )

    kwargs = {k: v for k, v in body.items() if k in _GRAPH_QUERY_TOOL_FIELDS}
    if "cypher" not in kwargs and query_val is not None:
        kwargs["cypher"] = query_val
    return kwargs


async def graph_search_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool("graph_search", **body)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except UnsupportedToolFieldError as e:
        # U-74: same deterministic-4xx treatment as `_make_tool_endpoint` —
        # this hand-written endpoint predates that factory and was never
        # updated to catch this exception subclass specially, so a caller
        # field the `graph_search` tool doesn't accept fell through to the
        # generic 500 below instead. `graph_search`'s own wire field names
        # (`query`, `mode`, `top_k`, ...) already match its documented tool
        # parameters 1:1 — see `graph_search`'s signature in
        # `agent_utilities/mcp/tools/query_tools.py` — so unlike
        # `graph_query`/`cypher` there is no latent name mismatch here; only
        # the missing status-code mapping needed fixing.
        return _external_error_response(e, status_code=400, code="invalid_request")
    except Exception as e:
        return _external_error_response(e)


async def graph_write_endpoint(request: Request) -> JSONResponse:
    """POST /graph/write — collapsed, typed dispatch for every ``graph_write``
    action. Covers the six actions that used to have their own granular
    routes (``add_node``, ``add_edge``, ``delete_edge``, ``bulk_ingest``,
    ``log_chat``, ``register_execution`` — formerly
    ``/graph/write/{node,edge,bulk,chat,execution}``) plus every other
    action the tool accepts (``delete_node``, ``register_external_graph``,
    ``compare_and_set``, ``store_memory``, ``recall_memory``,
    ``recall_media``, ``submit_sdd``, ``check_loop``) — see
    ``GraphWriteAction`` below for the full discriminated union. The body is
    validated against that union instead of forwarded blind (``**body``), so
    an unrecognized/malformed ``action`` is a clean 400, never a 500, and
    FastAPI documents every action's real shape (mounted via
    ``add_api_route(..., response_model=GraphToolResponse)`` in
    ``_mount_rest_routes``, not the raw Starlette ``add_route`` most other
    handlers in this file still use).
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        action_model = _GRAPH_WRITE_ACTION_ADAPTER.validate_python(body)
    except ValidationError as e:
        return _external_error_response(e, status_code=400, code="invalid_request")
    try:
        res = await _dispatch_graph_write_action(action_model)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except UnsupportedToolFieldError as e:
        # U-74, same class of fix as `graph_search_endpoint` above.
        return _external_error_response(e, status_code=400, code="invalid_request")
    except Exception as e:
        return _external_error_response(e)


graph_ingest_endpoint = _make_tool_endpoint("graph_ingest")


graph_analyze_endpoint = _make_tool_endpoint("graph_analyze")


#: The graph_mine actions with a natural-body REST twin (CONCEPT:EG-KG.mining.frequent-itemset-mining).
#: Each mounts ``POST /api/mining/<action>`` dispatching the SAME
#: ``_execute_tool("graph_mine", action=...)`` core as the MCP verb — surface
#: parity is a build gate, so the MCP action + its REST twin ship together.
MINING_ACTIONS = (
    "associate",
    "cluster",
    "anomaly",
    "classify_fit",
    "classify_predict",
    "reduce",
    "sequence",
    "forecast",
    "text",
    "subgraph",
    "entity_resolve",
    "causal_impact",
    "process",
    "root_cause",
    "risk_propagation",
    "ontology_gap",
    "retrieval_quality",
    "community",
)


#: The graph_learn actions with a natural-body REST twin (CONCEPT:EG-KG.graphlearn.link-predictor).
#: Each mounts ``POST /api/graphlearn/<action>`` dispatching the SAME
#: ``_execute_tool("graph_learn", action=...)`` core as the MCP verb — surface parity.
GRAPHLEARN_ACTIONS = ("fit", "predict")


#: The graph_mine_deep actions with a natural-body REST twin (CONCEPT:AU-KG.mining.dsm-forecast-delegation —
#: Phase-6 heavy-dep delegation to data-science-mcp). Each mounts
#: ``POST /api/mining/deep/<action>`` dispatching the SAME
#: ``_execute_tool("graph_mine_deep", action=...)`` core as the MCP verb — surface parity.
DEEP_MINING_ACTIONS = (
    "deep_forecast",
    "deep_classify",
    "autoencoder_anomaly",
    "xgboost",
    "embed",
)


async def _read_json_body(request: Request) -> Any:
    """Read a REST body using the gateway's permissive JSON fallback."""

    try:
        return await request.json()
    except Exception:
        return {}


async def _run_json_endpoint(
    request: Request,
    tool_name: str,
    kwargs_factory: Callable[[Any], dict[str, Any]],
    *,
    require_object: bool = False,
) -> JSONResponse:
    """Dispatch a JSON REST adapter through the shared tool/error boundary."""

    body = await _read_json_body(request)
    if require_object and not isinstance(body, dict):
        return JSONResponse(
            {"status": "error", "message": "body must be a JSON object"},
            status_code=400,
        )
    try:
        res = await _execute_tool(tool_name, **kwargs_factory(body))
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as exc:
        return _external_error_response(exc)


def _action_body_kwargs(body: dict[str, Any], action: str) -> dict[str, Any]:
    """Build the natural-body payload used by action-routed REST adapters."""

    graph = body.pop("graph", "") or ""
    return {"action": action, "params_json": json.dumps(body), "graph": graph}


def _query_top_k_kwargs(body: Any, action: str) -> dict[str, Any]:
    return {
        "action": action,
        "query": body.get("query", ""),
        "top_k": int(body.get("top_k", 10)),
    }


def _search_mode_top_k_kwargs(body: Any, mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "query": body.get("query", ""),
        "top_k": int(body.get("top_k", 10)),
    }


def _change_coupling_kwargs(body: Any) -> dict[str, Any]:
    return {
        "action": "change_coupling",
        "target": body.get("repo", ""),
        "depth": int(body.get("min_support", 3)),
    }


def _target_query_kwargs(
    body: Any,
    *,
    action: str,
    target_field: str,
    query_field: str,
    target_default: str = "",
    query_default: str = "",
    top_k_default: int = 10,
) -> dict[str, Any]:
    return {
        "action": action,
        "target": body.get(target_field, target_default),
        "query": body.get(query_field, query_default),
        "top_k": int(body.get("top_k", top_k_default)),
    }


def _code_evolution_kwargs(body: Any) -> dict[str, Any]:
    return _target_query_kwargs(
        body,
        action="code_evolution",
        target_field="mode",
        query_field="target",
        target_default="file",
        top_k_default=20,
    )


def _context_kwargs(body: Any) -> dict[str, Any]:
    return _target_query_kwargs(
        body, action="context", target_field="target", query_field="query"
    )


def _code_context_kwargs(body: Any) -> dict[str, Any]:
    intent = str(body.get("intent", "how"))
    if body.get("cross_repo"):
        intent = f"{intent}+xrepo"
    return {
        "action": "code_context",
        "query": body.get("query", ""),
        "target": intent,
        "node_id": body.get("node_id", ""),
        "top_k": int(body.get("top_k", 10)),
        "depth": int(body.get("depth", 2)),
    }


def _explain_kwargs(body: Any) -> dict[str, Any]:
    domain = str(body.get("domain", ""))
    intent = str(body.get("intent", ""))
    return {
        "action": "explain",
        "query": body.get("query", ""),
        "target": f"{domain}:{intent}" if domain else intent,
        "node_id": body.get("node_id", ""),
        "top_k": int(body.get("top_k", 10)),
        "depth": int(body.get("depth", 2)),
    }


def _make_action_body_endpoint(tool_name: str, action: str):
    """Build a JSON-object action endpoint for a tool with natural parameters."""

    async def _endpoint(request: Request) -> JSONResponse:
        return await _run_json_endpoint(
            request,
            tool_name,
            lambda body: _action_body_kwargs(body, action),
            require_object=True,
        )

    return _endpoint


def _make_mining_deep_endpoint(action: str):
    """Build the REST twin for one ``graph_mine_deep`` action (CONCEPT:AU-KG.mining.dsm-forecast-delegation).

    ``POST /api/mining/deep/<action>`` accepts a natural body (``x``/``values``/
    ``source``, ``y``, ``writeback``, algo kwargs, ...) plus an optional ``graph``,
    and dispatches the SAME ``_execute_tool("graph_mine_deep", action=<action>, ...)``
    core the MCP verb uses — the delegated call to data-science-mcp and the KG
    foldback happen once, in that one core.
    """
    return _make_action_body_endpoint("graph_mine_deep", action)


def _make_graphlearn_endpoint(action: str):
    """Build the REST twin for one ``graph_learn`` action (CONCEPT:EG-KG.graphlearn.link-predictor).

    ``POST /api/graphlearn/<action>`` accepts a natural body (the action's kwargs,
    e.g. ``{node_label, direction, degree, epochs, writeback, ...}`` for fit,
    ``{model, node_label, top_k|candidate_pairs, writeback, ...}`` for predict) plus an
    optional ``graph``, and dispatches the SAME
    ``_execute_tool("graph_learn", action=<action>, ...)`` core as the MCP verb.
    """
    return _make_action_body_endpoint("graph_learn", action)


def _make_mining_endpoint(action: str):
    """Build the REST twin for one ``graph_mine`` action (CONCEPT:EG-KG.mining.frequent-itemset-mining).

    ``POST /api/mining/<action>`` accepts a natural mining body (the action's
    kwargs, e.g. ``{transactions|source,...}`` for associate, ``{features|source,
    algorithm,...}`` for cluster, ``{features|values|source,algorithm,...}`` for
    anomaly, ``{x|source,y,algorithm,...}`` for classify_fit, ``{model,x|source,...}``
    for classify_predict, ``{x|source,algorithm,n_components,...}`` for reduce,
    ``{sequences|source,min_support,algorithm,...}`` for sequence,
    ``{values,algorithm,horizon,...}`` for forecast,
    ``{docs|source,algorithm,k,...}`` for text,
    ``{label,min_support,max_edges,algorithm,...}`` for subgraph,
    ``{records|vectors|source,threshold,...}`` for entity_resolve,
    ``{series,control,intervention_index,...}`` for causal_impact,
    ``{traces,process_id,...}`` for process,
    ``{nodes,edges,scores,symptom,...}`` for root_cause,
    ``{nodes,edges,seed,...}`` for risk_propagation,
    ``{label,...}`` for ontology_gap,
    ``{traces,k,...}`` for retrieval_quality,
    ``{label,algorithm,...}`` for community) plus an
    optional ``graph``, and dispatches the SAME
    ``_execute_tool("graph_mine", action=<action>, ...)`` core as the MCP verb.
    """
    return _make_action_body_endpoint("graph_mine", action)


def _make_action_endpoint(tool_name: str):
    """Build an action-routed REST endpoint for a focused analyze-suite tool — the REST
    twin of the MCP tool, dispatching through the same ``_execute_tool`` core (KG-2.257)."""

    async def _endpoint(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            res = await _execute_tool(tool_name, **body)
            return JSONResponse({"status": "success", "result": safe_json_load(res)})
        except UnsupportedToolFieldError as e:
            # U-74, same class of fix as `graph_search_endpoint` above: this
            # factory blind-splats the body the same way, so any tool it
            # backs (graph_code/research/evaluate/explain/observe) shared the
            # missing 4xx mapping.
            return _external_error_response(e, status_code=400, code="invalid_request")
        except Exception as e:
            return _external_error_response(e)

    return _endpoint


graph_code_endpoint = _make_action_endpoint("graph_code")
graph_research_endpoint = _make_action_endpoint("graph_research")
graph_evaluate_endpoint = _make_action_endpoint("graph_evaluate")
graph_explain_endpoint = _make_action_endpoint("graph_explain")
graph_observe_endpoint = _make_action_endpoint("graph_observe")


async def graph_orchestrate_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool("graph_orchestrate", **body)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_configure_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool("graph_configure", **body)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


def _to_json_str(val: Any) -> str:
    if isinstance(val, dict | list):
        return json.dumps(val)
    return str(val) if val is not None else ""


# 1. Granular Graph Query endpoints
async def graph_query_federated_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_query",
            cypher=body.get("cypher", ""),
            params=_to_json_str(body.get("params", {})),
            scope="federated",
            reference_id=body.get("reference_id", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


# 2. Granular Graph Search endpoints
def _make_granular_search_endpoint(
    mode: str, *, include_top_k: bool = True
) -> Callable[[Request], JSONResponse]:
    """Build one mode-fixed ``graph_search`` REST adapter.

    The per-mode URLs remain distinct, but all of them share the same JSON
    parsing, async dispatch, authority propagation, success envelope, and
    public error boundary. ``discover`` intentionally omits ``top_k`` because
    its historical adapter never forwarded that field to the tool.
    """

    def kwargs_factory(body: Any) -> dict[str, Any]:
        if include_top_k:
            return _search_mode_top_k_kwargs(body, mode)
        return {"query": body.get("query", ""), "mode": mode}

    async def _endpoint(request: Request) -> JSONResponse:
        return await _run_json_endpoint(request, "graph_search", kwargs_factory)

    endpoint_name = f"graph_search_{mode}_endpoint"
    _endpoint.__name__ = endpoint_name
    _endpoint.__qualname__ = endpoint_name
    _endpoint.__doc__ = (
        f"REST twin of graph_search mode={mode!r}; the route fixes the mode "
        "while preserving the underlying tool's request and response boundary."
    )
    return _endpoint


graph_search_concept_endpoint = _make_granular_search_endpoint("concept")
graph_search_analogy_endpoint = _make_granular_search_endpoint("analogy")
graph_search_memory_endpoint = _make_granular_search_endpoint("memory")
graph_search_discover_endpoint = _make_granular_search_endpoint(
    "discover", include_top_k=False
)
graph_search_dci_endpoint = _make_granular_search_endpoint("dci")


# 3. Collapsed Graph Write endpoint (POST + DELETE /graph/write)
#
# CONSOLIDATION: this used to be six separate granular routes — POST
# /graph/write/node, POST/DELETE /graph/write/edge, POST /graph/write/bulk,
# POST /graph/write/chat, POST /graph/write/execution — each a thin
# hand-written Starlette handler reading a handful of ``body.get`` keys.
# Collapsed into the SAME action-routed ``POST /graph/write`` the base
# endpoint already exposed (plus a ``DELETE /graph/write`` twin for
# ``delete_edge`` — see ``graph_write_delete_edge_endpoint`` below), now
# dispatched through a real Pydantic discriminated union
# (``GraphWriteAction``) instead of ``**body`` passthrough, so every action
# gets its own validated shape AND FastAPI documents it.
#
# The six "primary" variants below (``_AddNodeAction`` ..
# ``_RegisterExecutionAction``) extend the already-merged per-route models in
# ``agent_utilities.gateway.schemas.graph_ingest`` (imported, not redefined)
# — each adds the ``action`` discriminator plus ``connection``/``graph``,
# which the granular routes never forwarded even though ``graph_write``
# resolves them generically for EVERY action (``_resolve_target_engines``/
# ``bound_to_graph`` run before the action dispatch, not just for
# ``bulk_ingest``). ``_BulkIngestAction`` additionally restores
# ``idempotency_key``/``evidence``/``upsert`` — the real defect this
# consolidation fixes: the deleted ``graph_write_bulk_endpoint`` forwarded
# ONLY ``nodes``, always taking the non-idempotent ``BatchUpdate``
# (``upsert=True``) path even when a caller supplied an idempotency key, so a
# retried bulk write could double-write.
#
# ``_OtherGraphWriteAction`` is the 7th union member and covers every action
# that never had its own granular route (``delete_node``,
# ``register_external_graph``, ``compare_and_set``, ``store_memory``,
# ``recall_memory``, ``recall_media``, ``submit_sdd``, ``check_loop``) — it
# IS ``GraphWriteRequest`` itself (imported, not redefined; the already
# merged, ``extra="allow"``, full-passthrough model), with ``action``
# narrowed to a Literal of exactly those eight values (Pydantic discriminated
# unions support several tag values mapping to one member). This preserves
# the pre-consolidation base route's full action vocabulary byte-for-byte
# instead of silently dropping it down to only the six actions collapsed
# here.
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, ValidationError, field_validator

from agent_utilities.gateway.schemas.graph_ingest import (
    GraphToolResponse,
    GraphWriteBulkRequest,
    GraphWriteChatRequest,
    GraphWriteEdgeDeleteRequest,
    GraphWriteEdgeRequest,
    GraphWriteExecutionRequest,
    GraphWriteNodeRequest,
    GraphWriteRequest,
)

_CONNECTION_FIELD_DESCRIPTION = (
    "Named backend connection to write to (default = primary). Use a "
    "registered connection name, or 'all'/a comma-separated list to mirror "
    "the same write to several backends. Applies to every action (resolved "
    "generically before the action-specific dispatch) — not just "
    "bulk_ingest, which is all the granular routes ever exposed this on."
)
_GRAPH_FIELD_DESCRIPTION = (
    "Explicit physical engine graph to write to, independent of "
    "'connection'. Empty = the caller's own bound graph. Requires exactly "
    "one resolved 'connection' — never combinable with connection='all'/a "
    "list. Applies to every action, not just bulk_ingest."
)


def _coerce_properties_str_to_dict(v: Any) -> Any:
    """``mode="before"`` validator shared by ``_AddNodeAction``/
    ``_AddEdgeAction``: accept an already-JSON-encoded string for
    ``properties`` (the shape the base ``/graph/write`` route has
    historically taken there — see ``test_tiny_profile_serves_kg_over_
    gateway_with_zero_containers``) IN ADDITION to a plain JSON object,
    without widening the field's declared type away from the inherited
    ``dict[str, Any]`` (a wider ``dict | str`` annotation here would violate
    Liskov substitution against ``GraphWriteNodeRequest``/
    ``GraphWriteEdgeRequest``'s own ``properties: dict[str, Any]``, which
    mypy correctly rejects). An empty string normalizes to ``{}``; any other
    string is JSON-decoded (a non-dict/invalid JSON string is a clean 400 via
    the surrounding discriminated-union validation, not a silent pass).
    """
    if isinstance(v, str):
        return json.loads(v) if v.strip() else {}
    return v


class _AddNodeAction(GraphWriteNodeRequest):
    """``POST /graph/write``, ``action='add_node'`` — replaces
    ``POST /graph/write/node``.
    """

    action: Literal["add_node"] = Field(
        description="Fixed discriminator for this variant: 'add_node'."
    )
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "JSON object of node properties, OR an already-JSON-encoded "
            "string (both accepted; forwarded to the graph_write tool as a "
            "JSON-encoded string either way — the base /graph/write route "
            "has historically taken a raw pre-encoded string here, so both "
            "forms are supported for compatibility)."
        ),
        json_schema_extra={"examples": [{"label": "example"}]},
    )
    connection: str = Field(default="", description=_CONNECTION_FIELD_DESCRIPTION)
    graph: str = Field(default="", description=_GRAPH_FIELD_DESCRIPTION)

    _coerce_properties = field_validator("properties", mode="before")(
        _coerce_properties_str_to_dict
    )


class _AddEdgeAction(GraphWriteEdgeRequest):
    """``POST /graph/write``, ``action='add_edge'`` — replaces
    ``POST /graph/write/edge``.
    """

    action: Literal["add_edge"] = Field(
        description="Fixed discriminator for this variant: 'add_edge'."
    )
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "JSON object of edge properties, OR an already-JSON-encoded "
            "string (both accepted)."
        ),
    )
    connection: str = Field(default="", description=_CONNECTION_FIELD_DESCRIPTION)
    graph: str = Field(default="", description=_GRAPH_FIELD_DESCRIPTION)

    _coerce_properties = field_validator("properties", mode="before")(
        _coerce_properties_str_to_dict
    )


class _DeleteEdgeAction(GraphWriteEdgeDeleteRequest):
    """``action='delete_edge'`` — reachable via ``POST /graph/write`` (this
    variant) AND via ``DELETE /graph/write``
    (``graph_write_delete_edge_endpoint`` below, which validates the same
    ``GraphWriteEdgeDeleteRequest`` shape and hard-codes this action) —
    replaces ``DELETE /graph/write/edge``. Both are kept so neither an
    action-field-first caller nor a REST-verb-first caller loses the
    capability.
    """

    action: Literal["delete_edge"] = Field(
        description="Fixed discriminator for this variant: 'delete_edge'."
    )
    connection: str = Field(default="", description=_CONNECTION_FIELD_DESCRIPTION)
    graph: str = Field(default="", description=_GRAPH_FIELD_DESCRIPTION)


class _BulkIngestAction(GraphWriteBulkRequest):
    """``POST /graph/write``, ``action='bulk_ingest'`` — replaces
    ``POST /graph/write/bulk``.

    REGRESSION FIX: the deleted granular route forwarded ONLY ``nodes``,
    silently discarding ``idempotency_key``/``evidence``/``upsert``/
    ``connection``/``graph`` and always taking the non-idempotent
    ``BatchUpdate(upsert=True)`` path. This variant forwards all of them.
    """

    action: Literal["bulk_ingest"] = Field(
        description="Fixed discriminator for this variant: 'bulk_ingest'."
    )
    idempotency_key: str = Field(
        default="",
        description=(
            "Caller-owned idempotency key for this exact batch. Non-empty "
            "(or a non-empty 'evidence') routes the batch onto the engine's "
            "durably-idempotent ApplyChangeEnvelopes path, scoped by "
            "(tenant, graph, idempotency_key) — a replay reports "
            "'status':'skipped', never silently re-reported as fresh "
            "success. Empty uses the lighter BatchUpdate path, which has no "
            "per-call idempotency key."
        ),
    )
    evidence: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Evidence records ({'object_id','modality','locus',"
            "'content_digest'}) attached to the first node in 'nodes'. "
            "Non-empty routes the batch onto ApplyChangeEnvelopes instead "
            "of the lighter BatchUpdate path."
        ),
    )
    upsert: bool = Field(
        default=True,
        description=(
            "On the BatchUpdate (light) path only: True (default) MERGEs "
            "onto an existing id (idempotent); False INSERTs (a repeated "
            "edge becomes an additional parallel edge rather than "
            "replacing the prior one)."
        ),
    )
    connection: str = Field(default="", description=_CONNECTION_FIELD_DESCRIPTION)
    graph: str = Field(default="", description=_GRAPH_FIELD_DESCRIPTION)


class _LogChatAction(GraphWriteChatRequest):
    """``POST /graph/write``, ``action='log_chat'`` — replaces
    ``POST /graph/write/chat``.
    """

    action: Literal["log_chat"] = Field(
        description="Fixed discriminator for this variant: 'log_chat'."
    )
    connection: str = Field(default="", description=_CONNECTION_FIELD_DESCRIPTION)
    graph: str = Field(default="", description=_GRAPH_FIELD_DESCRIPTION)


class _RegisterExecutionAction(GraphWriteExecutionRequest):
    """``POST /graph/write``, ``action='register_execution'`` — replaces
    ``POST /graph/write/execution``.
    """

    action: Literal["register_execution"] = Field(
        description="Fixed discriminator for this variant: 'register_execution'."
    )
    connection: str = Field(default="", description=_CONNECTION_FIELD_DESCRIPTION)
    graph: str = Field(default="", description=_GRAPH_FIELD_DESCRIPTION)


class _OtherGraphWriteAction(GraphWriteRequest):
    """``POST /graph/write`` for every action never given its own granular
    route: ``delete_node``, ``register_external_graph``,
    ``compare_and_set``, ``store_memory``, ``recall_memory``,
    ``recall_media``, ``submit_sdd``, ``check_loop``. ``GraphWriteRequest``
    itself (imported, not redefined) already declares every field these
    actions read, with ``extra='allow'`` full passthrough — this subclass
    only narrows ``action`` to a Literal of those eight values so the
    discriminated union can tag-match it.
    """

    action: Literal[
        "delete_node",
        "register_external_graph",
        "compare_and_set",
        "store_memory",
        "recall_memory",
        "recall_media",
        "submit_sdd",
        "check_loop",
    ] = Field(
        description=(
            "One of: delete_node, register_external_graph, compare_and_set, "
            "store_memory, recall_memory, recall_media, submit_sdd, "
            "check_loop. See GraphWriteRequest's own field docs for which "
            "fields each of these reads."
        )
    )


GraphWriteAction = Annotated[
    _AddNodeAction
    | _AddEdgeAction
    | _DeleteEdgeAction
    | _BulkIngestAction
    | _LogChatAction
    | _RegisterExecutionAction
    | _OtherGraphWriteAction,
    Field(discriminator="action"),
]

_GRAPH_WRITE_ACTION_ADAPTER: TypeAdapter[Any] = TypeAdapter(GraphWriteAction)


async def _dispatch_graph_write_action(action_model: Any) -> Any:
    """Invoke ``_execute_tool("graph_write", ...)`` for one validated
    ``GraphWriteAction``. The six explicit branches mirror, field-for-field,
    what the now-deleted granular ``/graph/write/*`` routes used to forward
    (plus ``connection``/``graph``, and the ``bulk_ingest`` defect fix — see
    the class docstrings above); ``_OtherGraphWriteAction`` forwards its
    full body exactly as the pre-consolidation ``**body`` passthrough did.
    """
    if isinstance(action_model, _AddNodeAction):
        return await _execute_tool(
            "graph_write",
            action="add_node",
            node_id=action_model.node_id,
            node_type=action_model.node_type,
            properties=_to_json_str(action_model.properties),
            connection=action_model.connection,
            graph=action_model.graph,
        )
    if isinstance(action_model, _AddEdgeAction):
        return await _execute_tool(
            "graph_write",
            action="add_edge",
            source_id=action_model.source_id,
            target_id=action_model.target_id,
            rel_type=action_model.rel_type,
            properties=_to_json_str(action_model.properties),
            connection=action_model.connection,
            graph=action_model.graph,
        )
    if isinstance(action_model, _DeleteEdgeAction):
        return await _execute_tool(
            "graph_write",
            action="delete_edge",
            source_id=action_model.source_id,
            target_id=action_model.target_id,
            rel_type=action_model.rel_type,
            connection=action_model.connection,
            graph=action_model.graph,
        )
    if isinstance(action_model, _BulkIngestAction):
        return await _execute_tool(
            "graph_write",
            action="bulk_ingest",
            nodes=_to_json_str(action_model.nodes),
            idempotency_key=action_model.idempotency_key,
            evidence=_to_json_str(action_model.evidence),
            upsert=action_model.upsert,
            connection=action_model.connection,
            graph=action_model.graph,
        )
    if isinstance(action_model, _LogChatAction):
        return await _execute_tool(
            "graph_write",
            action="log_chat",
            agent_id=action_model.agent_id,
            properties=action_model.content,
            connection=action_model.connection,
            graph=action_model.graph,
        )
    if isinstance(action_model, _RegisterExecutionAction):
        return await _execute_tool(
            "graph_write",
            action="register_execution",
            agent_id=action_model.agent_id,
            connection=action_model.connection,
            graph=action_model.graph,
        )
    # _OtherGraphWriteAction: full passthrough parity with the
    # pre-consolidation **body forwarding.
    return await _execute_tool("graph_write", **action_model.model_dump())


async def graph_write_delete_node_endpoint(request: Request) -> JSONResponse:
    try:
        node_id = request.path_params.get("node_id", "")
        # Same DEFECT C field-name bug as `_AddNodeAction`'s dispatch above:
        # the tool parameter is ``node_id``, not ``id``.
        res = await _execute_tool("graph_write", action="delete_node", node_id=node_id)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except UnsupportedToolFieldError as e:
        return _external_error_response(e, status_code=400, code="invalid_request")
    except Exception as e:
        return _external_error_response(e)


async def graph_write_delete_edge_endpoint(request: Request) -> JSONResponse:
    """DELETE /graph/write — action='delete_edge' (replaces
    DELETE /graph/write/edge). Kept as a dedicated DELETE handler on the
    collapsed base path — see ``_DeleteEdgeAction``'s docstring above for
    why ``action='delete_edge'`` is ALSO reachable via POST.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        payload = GraphWriteEdgeDeleteRequest.model_validate(body or {})
    except ValidationError as e:
        return _external_error_response(e, status_code=400, code="invalid_request")
    try:
        res = await _execute_tool(
            "graph_write",
            action="delete_edge",
            source_id=payload.source_id,
            target_id=payload.target_id,
            rel_type=payload.rel_type,
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except UnsupportedToolFieldError as e:
        return _external_error_response(e, status_code=400, code="invalid_request")
    except Exception as e:
        return _external_error_response(e)


async def graph_write_external_endpoint(request: Request) -> JSONResponse:
    return await _run_json_endpoint(
        request,
        "graph_write",
        lambda body: {
            "action": "register_external_graph",
            "endpoint_url": body.get("endpoint_url", ""),
            "graph_type": body.get("graph_type", ""),
        },
    )


async def graph_write_memory_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_write",
            action="store_memory",
            agent_id=body.get("agent_id", ""),
            node_type=body.get("memory_type", ""),
            properties=body.get("content", ""),
            nodes=_to_json_str(body.get("tags", [])),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_write_memory_recall_endpoint(request: Request) -> JSONResponse:
    return await _run_json_endpoint(
        request,
        "graph_write",
        lambda body: {
            "action": "recall_memory",
            "properties": body.get("query", ""),
            "node_type": body.get("memory_type", ""),
        },
    )


async def graph_ontology_sync_packages_endpoint(request: Request) -> JSONResponse:
    """REST twin of ``graph_ontology action='sync_packages'`` (CONCEPT:AU-KG.ontology.federation-runtime).

    Federation: load every ontology ``.ttl`` contributed by installed fleet
    packages (``agent_utilities.ontology_providers``) through the shared ontology
    load path. Mirrors the generic ``POST /graph/ontology`` action twin as an
    explicit convenience route.
    """
    try:
        res = await _execute_tool("graph_ontology", action="sync_packages")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_ontology_publish_stardog_endpoint(request: Request) -> JSONResponse:
    """REST twin of ``graph_ontology action='publish_stardog'`` (CONCEPT:AU-KG.ontology.stardog-catalog-overwrite).

    Push the platform's authoritative bundled TBox to a Stardog triplestore, overwriting
    the target named graph by default so an updated ontology updates the catalog.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_ontology",
            action="publish_stardog",
            named_graph=body.get("named_graph", ""),
            overwrite=bool(body.get("overwrite", True)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_ontology_import_stardog_endpoint(request: Request) -> JSONResponse:
    """REST twin of ``graph_ontology action='import_stardog'`` (CONCEPT:AU-KG.ontology.stardog-catalog-import).

    Consume the TBox already living in a Stardog database / named graph back into the
    engine, activating it for reasoning.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_ontology",
            action="import_stardog",
            named_graph=body.get("named_graph", ""),
            activate=bool(body.get("activate", True)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_write_sdd_endpoint(request: Request) -> JSONResponse:
    return await _run_json_endpoint(
        request,
        "graph_write",
        lambda body: {
            "action": "submit_sdd",
            "agent_id": body.get("agent_id", ""),
            "properties": body.get("content", ""),
        },
    )


# 4. Granular Graph Ingest endpoints
async def graph_ingest_submit_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_ingest",
            action="ingest",
            target_path=_to_json_str(body.get("target_path", "")),
            max_depth=int(body.get("max_depth", 3)),
            agent_id=body.get("agent_id", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_ingest_corpus_endpoint(request: Request) -> JSONResponse:
    return await _run_json_endpoint(
        request,
        "graph_ingest",
        lambda body: {
            "action": "corpus",
            "corpus_name": body.get("corpus_name", ""),
            "base_path": body.get("base_path", ""),
            "description": body.get("description", ""),
        },
    )


async def graph_ingest_jobs_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_ingest", action="jobs")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def connector_sources_endpoint(request: Request) -> JSONResponse:
    """List registered document-source connectors (CONCEPT:AU-ECO.connector.factory-ingestion-adaptor)."""
    try:
        res = await _execute_tool("source_connector", action="list")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def connector_run_endpoint(request: Request) -> JSONResponse:
    """Build + drain a document-source connector into the KG (CONCEPT:AU-ECO.connector.document-source-framework–4.29)."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "source_connector",
            action="run",
            source_type=body.get("source_type", ""),
            config=body.get("config", {}) or {},
            connector_id=body.get("connector_id", ""),
            contextual=bool(body.get("contextual", True)),
            incremental=bool(body.get("incremental", True)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_ingest_job_status_endpoint(request: Request) -> JSONResponse:
    try:
        job_id = request.path_params.get("job_id", "")
        res = await _execute_tool("graph_ingest", action="job_status", job_id=job_id)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_ingest_rebuild_indexes_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_ingest", action="rebuild_indexes")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_ingest_observe_endpoint(request: Request) -> JSONResponse:
    return await _run_json_endpoint(
        request,
        "graph_ingest",
        lambda body: {
            "action": "observe",
            "target_path": body.get("target_path", ""),
            "agent_id": body.get("agent_id", ""),
        },
    )


async def graph_ingest_materialize_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_ingest", action="materialize")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_ingest_materialize_source_endpoint(request: Request) -> JSONResponse:
    """Persist an enterprise source extractor (camunda/aris/egeria) into the KG.

    Body: ``{"category": "camunda", "config": {...}}`` — ``category`` is the
    extractor key (required); ``config`` is an optional extractor-config dict.
    """
    try:
        body = await request.json()
        category = body.get("category") or body.get("corpus_name") or ""
        config = body.get("config")
        res = await _execute_tool(
            "graph_ingest",
            action="materialize_source",
            corpus_name=category,
            description=json.dumps(config) if isinstance(config, dict) else "",
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_ingest_sync_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_ingest", action="sync")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_ingest_reflect_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_ingest", action="reflect")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_ingest_agent_toolkit_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_ingest",
            action="agent_toolkit",
            target_path=_to_json_str(body.get("sources", [])),
            description=body.get("agent_card_path", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_ingest_knowledge_pack_endpoint(request: Request) -> JSONResponse:
    return await _run_json_endpoint(
        request,
        "graph_ingest",
        lambda body: {
            "action": "ingest_knowledge_pack",
            "target_path": body.get("target_path", ""),
        },
    )


# 5. Granular Graph Analyze endpoints
async def graph_analyze_synthesize_endpoint(request: Request) -> JSONResponse:
    return await _run_json_endpoint(
        request, "graph_research", lambda body: _query_top_k_kwargs(body, "synthesize")
    )


async def graph_analyze_process_writeback_endpoint(request: Request) -> JSONResponse:
    """Push KG process intelligence INTO Camunda instances / ARIS models.

    Body: ``{"target": "both|camunda|aris", "query": "id1,id2"}`` —
    ``target`` is the writeback scope (default ``both``); ``query`` is an
    optional comma-separated list of BusinessProcess node ids to limit to.
    """
    return await _run_json_endpoint(
        request,
        "graph_analyze",
        lambda body: {
            "action": "process_writeback",
            "target": body.get("target", "both"),
            "query": body.get("query", ""),
        },
    )


async def graph_analyze_deep_extract_endpoint(request: Request) -> JSONResponse:
    return await _run_json_endpoint(
        request,
        "graph_research",
        lambda body: _query_top_k_kwargs(body, "deep_extract"),
    )


async def graph_analyze_background_research_endpoint(request: Request) -> JSONResponse:
    return await _run_json_endpoint(
        request,
        "graph_research",
        lambda body: _query_top_k_kwargs(body, "background_research"),
    )


async def graph_analyze_relevance_sweep_endpoint(request: Request) -> JSONResponse:
    return await _run_json_endpoint(
        request,
        "graph_research",
        lambda body: _query_top_k_kwargs(body, "relevance_sweep"),
    )


async def graph_analyze_blast_radius_endpoint(request: Request) -> JSONResponse:
    try:
        node_id = request.query_params.get("id", "")
        depth = int(request.query_params.get("depth", "2"))
        res = await _execute_tool(
            "graph_code", action="blast_radius", node_id=node_id, depth=depth
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_inspect_endpoint(request: Request) -> JSONResponse:
    try:
        target = request.query_params.get("target", "")
        res = await _execute_tool("graph_analyze", action="inspect", target=target)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_call_graph_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=call_graph (CONCEPT:EG-KG.compute.type-scope-resolved-call): the
    type/scope-resolved call/inheritance graph for a symbol. ``id`` = symbol id;
    ``direction`` = callees | callers | inherits."""
    try:
        node_id = request.query_params.get("id", "")
        direction = request.query_params.get("direction") or request.query_params.get(
            "target", "callees"
        )
        res = await _execute_tool(
            "graph_code", action="call_graph", node_id=node_id, target=direction
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_similar_code_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=similar_code (CONCEPT:EG-KG.compute.model-free-similar-code): a
    symbol's model-free MinHash/LSH near-clone neighbours (embedder-free).
    ``id`` = symbol id; ``top_k`` optional."""
    try:
        node_id = request.query_params.get("id", "")
        top_k = int(request.query_params.get("top_k", "10"))
        res = await _execute_tool(
            "graph_code", action="similar_code", node_id=node_id, top_k=top_k
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_routes_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=routes (CONCEPT:AU-KG.compute.http-route-graph): the HTTP route
    graph — each Route, its handler, and the Service that serves it."""
    try:
        res = await _execute_tool("graph_code", action="routes")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_change_coupling_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=change_coupling (CONCEPT:AU-KG.ingest.mine-git-history-files): mine a
    repo's git history into FILE_CHANGES_WITH edges. Body: ``{repo, min_support?}``."""
    return await _run_json_endpoint(request, "graph_code", _change_coupling_kwargs)


async def graph_analyze_code_evolution_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=code_evolution (CONCEPT:AU-KG.enrichment.query-ingested-commit-history): query the
    ingested commit-history graph for codebase evolution. Body:
    ``{mode?, target?, top_k?}`` — mode = file|owners|hotspots|coupled,
    target = file path / subsystem path substring."""
    return await _run_json_endpoint(request, "graph_code", _code_evolution_kwargs)


async def graph_analyze_adr_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=adr (CONCEPT:AU-KG.compute.adr-crud): ADR CRUD. Body:
    ``{title?, status?, decision?}`` — title creates, empty lists."""
    return await _run_json_endpoint(
        request,
        "graph_code",
        lambda body: {
            "action": "adr",
            "query": body.get("title", ""),
            "target": body.get("status", ""),
            "node_id": body.get("decision", ""),
        },
    )


async def graph_analyze_harness_gate_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=harness_gate (CONCEPT:AU-AHE.evaluation.parity-surpass-scoreboard): validate a
    candidate harness-evolution state against the concentration/no-regression/pathology
    SHACL gate. Body: ``{edits:[…], variants?:[…], pathologies?:[…]}``."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        import json as _json

        res = await _execute_tool(
            "graph_evaluate", action="harness_gate", query=_json.dumps(body)
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_code_context_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_code action=code_context (CONCEPT:AU-KG.retrieval.synthesized-cited-answer): the
    synthesized, cited codebase Q&A. Body: ``{query, intent?(how|usage|impact),
    node_id?, top_k?, depth?, cross_repo?}``."""
    return await _run_json_endpoint(request, "graph_code", _code_context_kwargs)


async def graph_analyze_explain_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_explain action=explain (CONCEPT:AU-KG.retrieval.route-question-its-domain): the universal
    context plane. Body: ``{query, domain?, intent?, node_id?, top_k?, depth?}`` —
    routes to the domain provider (code | ops | …) and returns the cited answer."""
    return await _run_json_endpoint(request, "graph_explain", _explain_kwargs)


async def graph_analyze_cross_repo_usages_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=cross_repo_usages (CONCEPT:AU-KG.retrieval.every-usage-published-symbol): every
    usage of a published symbol across the fleet, grouped by repo. ``symbol`` /
    ``query`` = the symbol name; ``top_k`` optional."""
    try:
        symbol = request.query_params.get("symbol") or request.query_params.get(
            "query", ""
        )
        top_k = int(request.query_params.get("top_k", "200"))
        res = await _execute_tool(
            "graph_code", action="cross_repo_usages", query=symbol, top_k=top_k
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def _run_graph_code_scope_endpoint(request: Request, action: str) -> JSONResponse:
    try:
        scope = request.query_params.get("scope") or request.query_params.get(
            "target", ""
        )
        top_k = int(request.query_params.get("top_k", "10"))
        res = await _execute_tool(
            "graph_code", action=action, target=scope, top_k=top_k
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_code_metrics_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=code_metrics (CONCEPT:AU-KG.retrieval.god-nodes-communities): Graphify-
    style god nodes / communities / surprising connections over the :Code subgraph.
    ``scope`` (or ``target``) = optional file_path/source_system substring;
    ``top_k`` = section sizes."""
    return await _run_graph_code_scope_endpoint(request, "code_metrics")


async def graph_analyze_arch_report_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=arch_report (CONCEPT:AU-KG.retrieval.architecture-report): the
    regenerable architecture report (GRAPH_REPORT.md analog) as Markdown + metrics.
    ``scope`` (or ``target``) = optional substring; ``top_k`` = section sizes."""
    return await _run_graph_code_scope_endpoint(request, "arch_report")


async def graph_analyze_context_endpoint(request: Request) -> JSONResponse:
    return await _run_json_endpoint(request, "graph_explain", _context_kwargs)


async def graph_analyze_evaluate_alpha_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_evaluate", action="evaluate_alpha", target=body.get("target", "")
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_evaluate_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_evaluate", action="evaluate", target=body.get("target", "")
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_evolve_model_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_evaluate", action="evolve_model")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_forecast_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_evaluate", action="forecast")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_causal_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_evaluate", action="causal")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_invariant_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_evaluate", action="invariant")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_security_scan_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_analyze", action="security_scan", target=body.get("target", "")
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


# 7. Granular Graph Configure endpoints
async def graph_configure_secret_endpoint(request: Request) -> JSONResponse:
    return await _run_json_endpoint(
        request,
        "graph_configure",
        lambda body: {
            "action": "set_secret",
            "config_key": body.get("config_key", ""),
            "config_value": body.get("config_value", ""),
        },
    )


async def graph_configure_vault_sync_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_configure action=vault_sync (CONCEPT:AU-OS.deployment.vault-first-routine-genesis)."""
    return await _run_json_endpoint(
        request,
        "graph_configure",
        lambda body: {
            "action": "vault_sync",
            "config_key": body.get("config_key", ""),
            "config_value": body.get("config_value", ""),
        },
    )


async def graph_configure_register_mcp_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_configure",
            action="register_mcp",
            config_key=body.get("config_key", ""),
            config_value=_to_json_str(body.get("config_value", {})),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_configure_install_hooks_endpoint(request: Request) -> JSONResponse:
    return await _run_json_endpoint(
        request,
        "graph_configure",
        lambda body: {
            "action": "install_hooks",
            "config_value": body.get("config_value", ""),
        },
    )


async def graph_configure_uninstall_hooks_endpoint(request: Request) -> JSONResponse:
    return await _run_json_endpoint(
        request,
        "graph_configure",
        lambda body: {
            "action": "uninstall_hooks",
            "config_value": body.get("config_value", ""),
        },
    )


async def graph_configure_doctor_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_configure", action="doctor")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


# Default agent identity for provenance tracking
_AGENT_ID = setting("AGENT_ID", f"mcp-client-{uuid.uuid4().hex}")
_SESSION_ID = setting("SESSION_ID", uuid.uuid4().hex)


_ENGINE_LOCK = threading.Lock()


_EXTRACTION_MANAGER: Any = None


def _get_extraction_manager(engine: Any) -> Any:
    """Lazily build the single GPU-slot extraction job manager (KG-2.65)."""
    global _EXTRACTION_MANAGER
    if _EXTRACTION_MANAGER is None:
        from ..knowledge_graph.extraction.job_manager import ExtractionJobManager

        _EXTRACTION_MANAGER = ExtractionJobManager(engine)
    return _EXTRACTION_MANAGER


def _bind_ontology_package_sync(value: Any) -> None:
    """Bind the current ontology adapter once per engine instance."""
    from agent_utilities.mcp.tools.ontology_tools import _sync_package_ontologies

    if getattr(value, "_ontology_package_sync", None) is not _sync_package_ontologies:
        value._ontology_package_sync = _sync_package_ontologies


def _get_engine():
    """Lazily initialize and return the IntelligenceGraphEngine singleton.

    Thread-safe double-checked locking prevents concurrent runtime callers from
    racing a second authority into existence. Direct GraphOS startup resolves
    this engine synchronously only through the bounded packaged-skill readiness
    barrier; noncritical bootstrap work remains asynchronous.
    (CONCEPT:EG-KG.storage.nonblocking-checkpoint)
    """
    from agent_utilities.core.paths import ensure_dirs
    from agent_utilities.knowledge_graph.backends import create_backend
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    def _register_runtime_authorities(value: Any) -> Any:
        # Registration is process-owned startup state.  The served callers can
        # only resolve these recorded adapters, never select one from request
        # data. Keep the ontology binding in ``finally`` so one optional
        # registration failure cannot suppress the other capability.
        try:
            try:
                from agent_utilities.mcp.tools.data_prep_tools import (
                    register_process_data_prep_runtime,
                )

                register_process_data_prep_runtime(value)
            finally:
                # Keep ontology package application at the composition
                # boundary. Lower-layer ingestion consumes this bound
                # capability and never imports the MCP adapter itself.
                _bind_ontology_package_sync(value)
        except Exception:  # noqa: BLE001 - optional adapters must not block boot
            logger.warning(
                "runtime optional capability registration deferred",
                exc_info=True,
            )
        return value

    engine = IntelligenceGraphEngine.get_active()
    if engine is not None:
        return _register_runtime_authorities(engine)

    with _ENGINE_LOCK:
        engine = IntelligenceGraphEngine.get_active()
        if engine is not None:
            return _register_runtime_authorities(engine)
        # First-run: ensure XDG dirs exist and create backend
        ensure_dirs()

        def _factory():
            backend = create_backend()
            return IntelligenceGraphEngine(
                backend=backend,
                defer_background_start=True,
            )

        return _register_runtime_authorities(
            IntelligenceGraphEngine.get_or_create(factory=_factory)
        )


# ── CONCEPT:AU-KG.backend.multi-connection-registry — Named multi-connection graph registry ────────────────
_CONNECTION_REGISTRY = None
_REGISTRY_LOCK = threading.Lock()


def get_connection_registry():
    """Process-wide :class:`ConnectionRegistry` singleton.

    The reserved ``"default"`` target resolves only the process-active authority;
    registry construction never creates or seeds a second engine. Reference-only
    ``config.external_graph_connectors`` and ``config.kg_connections`` are
    registered on first build.
    """
    global _CONNECTION_REGISTRY
    if _CONNECTION_REGISTRY is not None:
        return _CONNECTION_REGISTRY
    with _REGISTRY_LOCK:
        if _CONNECTION_REGISTRY is not None:
            return _CONNECTION_REGISTRY
        from agent_utilities.knowledge_graph.core.connection_registry import (
            ConnectionRegistry,
        )

        registry = ConnectionRegistry()
        # Seed reference-only external sources first, then let an explicit
        # KG_CONNECTIONS declaration with the same alias take precedence.
        _seed_connection_registry(registry)
        _CONNECTION_REGISTRY = registry
        return _CONNECTION_REGISTRY


def _seed_external_graph_connectors(registry: Any) -> None:
    """Register reference-only external sources from config, best-effort per item."""
    from agent_utilities.core.config import config as _cfg

    for declared in _cfg.external_graph_connectors or []:
        value = (
            declared.model_dump() if hasattr(declared, "model_dump") else dict(declared)
        )
        name = str(value.pop("name", "") or "")
        if not name:
            continue
        value["role"] = "read"
        try:
            registry.register(name, value)
        except Exception as exc:  # noqa: BLE001 — one bad declaration never blocks the rest
            logger.warning(
                "Skipping invalid external source declaration: %s",
                type(exc).__name__,
            )


def _seed_kg_connections(registry: Any) -> None:
    """Register explicit KG_CONNECTIONS declarations from config, best-effort per item."""
    from agent_utilities.core.config import config as _cfg

    for spec in _cfg.kg_connections or []:
        spec = dict(spec)
        name = spec.pop("name", "")
        if name:
            try:
                registry.register(name, spec)
            except Exception as e:  # noqa: BLE001 — one bad declaration never blocks the rest
                logger.warning(
                    "Skipping invalid graph connection declaration: %s",
                    type(e).__name__,
                )


def _seed_connection_registry(registry: Any) -> None:
    """Seed a fresh :class:`ConnectionRegistry` from config, best-effort.

    An explicit ``KG_CONNECTIONS`` declaration takes precedence over a
    reference-only external source registered under the same alias, since
    external sources are seeded first. Config-less environments (the whole
    seeding step raises) leave the registry with nothing seeded.
    """
    try:
        _seed_external_graph_connectors(registry)
        _seed_kg_connections(registry)
    except Exception as exc:  # noqa: BLE001 — config-less environments
        logger.debug(
            "Graph connection declarations were not seeded (%s)",
            type(exc).__name__,
        )


def _resolve_target_engines(
    target: Any,
) -> tuple[list[tuple[str, Any]], dict[str, str], bool]:
    """Resolve a tool ``target`` into live engines for execution.

    Returns ``(entries, errors, fanout)`` where ``entries`` is a list of
    ``(name, engine)`` to run against and ``errors`` maps any name that could not
    be resolved to its error string. For a non-fan-out target, resolution errors
    propagate (fail-loud); for fan-out they are captured into ``errors`` so one
    bad connection never aborts the others (partial-success contract).
    """
    registry = get_connection_registry()
    names, fanout = registry.resolve_names(target)
    entries: list[tuple[str, Any]] = []
    errors: dict[str, str] = {}
    for name in names:
        if fanout:
            engine, err = registry.safe_get_engine(name)
            if err is not None:
                errors[name] = err
            else:
                entries.append((name, engine))
        else:
            entries.append((name, registry.get_engine(name)))
    return entries, errors, fanout


def _resolve_read_engines(
    target: Any,
) -> tuple[list[tuple[str, Any]], dict[str, str], bool]:
    """Resolve a READ tool's ``target`` into engines, unioning content graphs.

    CONCEPT:AU-KG.ingest.unified-query-routing — preserve unified query under ingestion graph routing. When
    routing is on and the caller did NOT pin an explicit target, content lives
    spread across per-source graphs (``code:*`` / ``src:*`` / …) that the single
    default engine cannot see. This resolver returns one engine per active content
    graph (plus the default) with ``fanout=True``, so the existing fan-out machinery
    unions them and a node written to ``code:X`` stays findable via the normal
    ``graph_search`` / ``graph_query`` path. An explicit ``target`` (a named
    connection, ``"all"``, a list) defers to the standard connection resolver
    unchanged, and with routing off this is byte-for-byte ``_resolve_target_engines``.
    """
    from agent_utilities.knowledge_graph.core import ingest_routing

    is_implicit_default = target is None or (
        isinstance(target, str) and target.strip().lower() in ("", "default")
    )
    if not is_implicit_default:
        return _resolve_target_engines(target)

    read_graphs = ingest_routing.read_graph_targets()
    if len(read_graphs) <= 1:
        # Nothing routed yet → stay on the fast single default-graph path.
        return _resolve_target_engines(target)

    from agent_utilities.knowledge_graph.core.shard_topology import default_graph_name

    default_graph = default_graph_name()
    entries: list[tuple[str, Any]] = []
    errors: dict[str, str] = {}
    # CONCEPT:AU-KG.backend.fanout-dedup — de-duplicate fan-out targets by the engine's actual bound
    # graph so the SAME backend (e.g. ``__commons__``) is never queried more than
    # once. Without this a query for nodes that live only in the default graph is
    # answered identically by every target, and an aggregation row (no node id to
    # dedup on) is repeated once per graph. Key on the backend's ``graph_name``,
    # falling back to ``id(engine)`` so two engines over one store collapse to one.
    seen_backends: set[Any] = set()

    def _backend_key(engine: Any) -> Any:
        gname = getattr(getattr(engine, "backend", None), "graph_name", None)
        return gname if gname is not None else id(engine)

    for gname in read_graphs:
        if gname == default_graph:
            eng: Any = _get_engine()
            name = "default"
        else:
            eng, err = ingest_routing.safe_engine_for_graph(gname)
            if err is not None:
                errors[gname] = err
                continue
            name = gname
        key = _backend_key(eng)
        if key in seen_backends:
            continue
        seen_backends.add(key)
        entries.append((name, eng))
    return entries, errors, True


class GraphSelectionError(ValueError):
    """Base for an explicit ``graph`` selection that cannot be honored.

    CONCEPT:AU-KG.backend.explicit-graph-selection — ``connection`` (a backend
    alias, CONCEPT:AU-KG.backend.multi-connection-registry, resolved by
    :class:`~agent_utilities.knowledge_graph.core.connection_registry.ConnectionRegistry`)
    and ``graph`` (a physical engine graph, the thing ``engine_tenants(action="list")``/
    the engine's own ``ListGraphs`` returns) are two independent axes. Every raise
    of this family is FAIL-CLOSED: the caller gets a typed error, never a silent
    fallback to a default graph and never a union across graphs.
    """


class GraphNotFoundError(GraphSelectionError, LookupError):
    """The requested ``graph`` is absent from the engine's own catalog."""


class GraphSelectionConflictError(GraphSelectionError):
    """The ``graph``/``connection`` combination is ambiguous or unsupported.

    Raised instead of silently picking one interpretation — e.g. an explicit
    ``graph`` alongside a fan-out ``connection`` (``'all'``/a list), or an
    explicit ``graph`` alongside a non-native (external/read-only) connection
    that has no physical-graph concept of its own.
    """


def resolve_explicit_graph(
    entries: list[tuple[str, Any]], graph: str, *, fanout: bool
) -> list[tuple[str, Any]]:
    """Validate an explicit ``graph`` against the already-resolved connection(s).

    Returns ``entries`` unchanged (no-op) when ``graph`` is empty — existing
    ``connection``-only behavior is untouched. When ``graph`` is set:

    * a fan-out ``connection`` (``entries`` has more than one, or ``fanout`` is
      True) is rejected — an explicit graph requires exactly one connection,
      never a union (CONCEPT:AU-KG.backend.explicit-graph-selection).
    * a non-``"default"`` connection is rejected — only the native (epistemic-
      graph) connection has a multi-graph concept; naming a graph for e.g. an
      external Neo4j connection would otherwise be silently ignored (a silent
      pick of that connection's own implicit graph), which is exactly the
      defect this function exists to prevent.
    * existence is checked, best-effort, against the SAME catalog
      ``engine_tenants(action="list")``/``ListGraphs`` already exposes
      (``engine.graph_compute.client.tenants.list()``) — never a parallel
      authorization mechanism. The actual per-call authorization remains the
      engine's own RBAC/RLS (``crates/eg-core/src/isolation.rs::check_access``),
      evaluated server-side on every request this binds into, regardless of
      whether this probe ran.
    """
    graph = graph.strip() if isinstance(graph, str) else ""
    if not graph:
        return entries
    if fanout or len(entries) != 1:
        raise GraphSelectionConflictError(
            "an explicit graph requires exactly one resolved connection, "
            "never a fan-out/union"
        )
    name, engine = entries[0]
    if name != "default":
        raise GraphSelectionConflictError(
            f"connection {name!r} has no physical-graph concept; explicit "
            "graph selection is supported only on the default connection"
        )
    _validate_graph_exists_in_catalog(engine, graph)
    return entries


def _validate_graph_exists_in_catalog(engine: Any, graph: str) -> None:
    """Best-effort existence check for ``graph`` against the engine's catalog.

    A degraded/unavailable catalog probe must never itself deny or (worse)
    silently permit — the engine's own RBAC/RLS is the real authorization
    boundary either way, so this just skips the check in that case and lets
    the actual call surface whatever the engine decides.
    """
    tenants = getattr(
        getattr(getattr(engine, "graph_compute", None), "client", None),
        "tenants",
        None,
    )
    list_graphs = getattr(tenants, "list", None)
    if not callable(list_graphs):
        return
    try:
        catalog = list_graphs() or []
    except Exception:  # noqa: BLE001 — best-effort probe; see docstring
        return
    names = {
        row.get("name") for row in catalog if isinstance(row, dict) and row.get("name")
    }
    if graph not in names:
        raise GraphNotFoundError(
            f"graph {graph!r} is not present in the engine catalog"
        )


@contextlib.contextmanager
def bound_to_graph(graph: str) -> Any:
    """Bind an explicit physical graph onto the verified session for one call.

    A no-op when ``graph`` is empty. Otherwise reuses the SAME sanctioned
    session-retargeting primitive already used by
    ``agent_utilities.knowledge_graph.pipeline`` and
    ``agent_utilities.orchestration.approval`` —
    :meth:`~agent_utilities.knowledge_graph.core.session.GraphSession.with_graph`
    scoped for the call's duration via
    :func:`~agent_utilities.knowledge_graph.core.session.use_session`. This
    introduces no new authority: the engine's own RBAC/RLS
    (``crates/eg-core/src/isolation.rs::check_access``) evaluates
    ``(agent_id, graph, action)`` on every RPC issued while bound, independent
    of what this function does — binding only says which graph the request
    NAMES, never what it is ALLOWED to touch.
    """
    graph = graph.strip() if isinstance(graph, str) else ""
    if not graph:
        yield
        return
    from agent_utilities.knowledge_graph.core.session import (
        current_session,
        use_session,
    )

    session = current_session()
    if session is None:
        raise GraphSelectionConflictError(
            "a verified session is required to bind an explicit graph"
        )
    with use_session(session.with_graph(graph)):
        yield


#: Per-target wall-clock budget (seconds) for a fan-out (``target='all'`` or a
#: multi-target list). One slow/unreachable backend must not stall the whole set;
#: override live via ``graph_configure set_config GRAPH_FANOUT_TIMEOUT`` (KG-2.63).
DEFAULT_FANOUT_TIMEOUT_S = 30.0

#: Per-target wall-clock budget (seconds) for an IMPLICIT-default read fan-out —
#: a ``graph_search``/``graph_query`` call with no explicit ``target`` that
#: resolves (CONCEPT:AU-KG.ingest.unified-query-routing) to the routed
#: content-graph union: ``default`` + every active ``code:*``/``src:*`` graph,
#: which can be dozens of per-repo connections and often includes idle/
#: unreachable ones. Using the full ``DEFAULT_FANOUT_TIMEOUT_S`` there means one
#: unreachable ``code:<repo>`` backend blocks the common no-target call for up to
#: 30s each, flooding the result with "timed out" entries. A short budget keeps
#: the default call fast — an unreachable graph is skipped, not waited on — while
#: an explicit ``target='all'``/list (a deliberate cross-repo search) keeps the
#: full ``DEFAULT_FANOUT_TIMEOUT_S``.
DEFAULT_CONTENT_FANOUT_TIMEOUT_S = 3.0


def fanout_error_label(exc: BaseException) -> str:
    """Classify a per-target fan-out failure as retryable-degraded or rejected.

    BUG-048: ``fanout_execute`` (below) and the "primary target" call sites
    that deliberately bypass its concurrency machinery but must mirror its
    error handling (e.g. ``graph_query``'s ``default``-connection fast path
    in ``query_tools.py``) reduce a raised exception to a bare string outside
    the ``OperationResult``/``OperationError`` schema ``public_error_payload``
    covers. Without this, a breaker-open on one fan-out target collapsed to
    the SAME ``"target_operation_failed"`` label as a genuinely rejected
    query — indistinguishable to a caller deciding whether to retry, exactly
    the defect BUG-048 fixed at the single-target boundary. Reuses
    :func:`agent_utilities.security.error_surface.is_engine_degraded` — the
    one place this breaker-vs-rejection vocabulary is defined — rather than
    reimplementing it a third time.
    """
    from agent_utilities.security.error_surface import is_engine_degraded

    return "target_degraded" if is_engine_degraded(exc) else "target_operation_failed"


def fanout_execute(entries, fn, *, timeout=None):
    """Run ``fn(name, engine)`` for every fan-out target CONCURRENTLY under a shared
    per-target wall-clock timeout, so one slow/unreachable backend can't stall the
    others (CONCEPT:AU-KG.backend.multi-connection-registry).

    Returns ``(results, errors)`` keyed by connection name. A target that exceeds the
    budget (or raises) lands in ``errors`` while the rest still return — the
    partial-success contract the sequential loop violated by blocking on the slowest.
    A raised target is labeled via :func:`fanout_error_label` — ``"target_degraded"``
    for a breaker-open/transport-down failure (retryable), ``"target_operation_failed"``
    for everything else (BUG-048) — never the exception text itself.

    B-18: ``concurrent.futures.ThreadPoolExecutor.submit`` does NOT propagate the
    calling thread's :mod:`contextvars` context — unlike ``asyncio.to_thread``
    (which the async tool endpoints above use to reach this synchronous helper
    in the first place), a plain worker thread starts with a FRESH, empty
    context. Without this, the ambient authenticated ``GraphSession`` (and any
    narrowing a caller applies via ``bound_to_graph`` around ``fn``) is
    invisible inside every fan-out target, and each one fails closed with
    ``SessionRequiredError`` regardless of which graph it targets. Capturing
    ``contextvars.copy_context()`` once PER submission (never reused across
    concurrent ``Context.run`` calls, which is not reentrant) and running
    ``fn`` inside that copy restores the ambient session per target while
    keeping each target's own narrowing (if any) isolated from its siblings.
    """
    import concurrent.futures
    import contextvars

    if timeout is None:
        timeout = float(setting("GRAPH_FANOUT_TIMEOUT", DEFAULT_FANOUT_TIMEOUT_S))
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    if not entries:
        return results, errors
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(entries)))
    futures = {
        ex.submit(contextvars.copy_context().run, fn, name, engine): name
        for name, engine in entries
    }
    done, not_done = concurrent.futures.wait(futures, timeout=timeout)
    for fut in done:
        name = futures[fut]
        try:
            results[name] = fut.result()
        except Exception as exc:  # noqa: BLE001 — partial-success contract
            logger.warning(
                "Graph fan-out target failed (exception_type=%s)",
                type(exc).__name__,
            )
            errors[name] = fanout_error_label(exc)
    for fut in not_done:
        errors[futures[fut]] = "target_timeout"
    # Never block on a hung backend's thread; let it finish in the background.
    ex.shutdown(wait=False, cancel_futures=True)
    return results, errors


def _provenance_props(agent_id: str | None = None) -> dict[str, Any]:
    """Build persistence-safe provenance without host or principal material."""
    from agent_utilities.security.persistence_privacy import persistence_reference

    return {
        "agent_ref": persistence_reference(
            "agent", agent_id or _AGENT_ID, namespace="mcp-provenance"
        ),
        "session_ref": persistence_reference(
            "session", _SESSION_ID, namespace="mcp-provenance"
        ),
        "timestamp": datetime.now(UTC).isoformat(),
        "source": "mcp",
    }


def _neutral_capability_name(value: object, *, fallback_ref: str) -> str:
    """Return a bounded service alias, never an arbitrary config key."""
    from agent_utilities.security.persistence_privacy import sanitize_for_persistence

    rendered = str(value or "").strip().lower()
    sanitized, report = sanitize_for_persistence(rendered)
    if not report.changed and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,62}", rendered):
        return rendered
    return f"external-{fallback_ref.rsplit('_', 1)[-1][:12]}"


def _mcp_capability_declaration(
    server_name: object, server_details: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Project one MCP runtime declaration into privacy-safe KG metadata."""
    from agent_utilities.knowledge_graph.core.source_sync import (
        derive_capability_synonyms,
    )
    from agent_utilities.security.persistence_privacy import persistence_reference

    server_ref = persistence_reference(
        "mcp_server", server_name, namespace="capability-ingestion"
    )
    neutral_name = _neutral_capability_name(server_name, fallback_ref=server_ref)
    configuration_ref = persistence_reference(
        "mcp_configuration",
        json.dumps(server_details, sort_keys=True, separators=(",", ":")),
        namespace=server_ref,
    )
    capabilities = [
        str(value).lower()
        for value in server_details.get("capabilities", [])
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,62}", str(value))
    ]
    return (
        f"mcp_server:{server_ref}",
        {
            "name": neutral_name,
            "server_ref": server_ref,
            "configuration_ref": configuration_ref,
            "capabilities": sorted(set(capabilities)),
            "synonyms": derive_capability_synonyms(neutral_name),
        },
    )


def _ontology_system():
    """Return an OntologySystem bound to the live engine store (or offline).

    Module-level so the ontology/object tool group registers from
    mcp/tools/ontology_tools.py instead of a _build_server closure.
    """
    from agent_utilities.knowledge_graph.facade import KnowledgeGraph

    try:
        engine = _get_engine()
    except Exception:  # pragma: no cover - defensive
        engine = None
    backend = getattr(engine, "backend", None) if engine is not None else None
    kg = KnowledgeGraph()
    if backend is not None:
        kg._store = backend
    return kg.ontology


class GraphOSStartupReadinessError(RuntimeError):
    """Stable, environment-free failure raised before GraphOS starts serving."""


def _read_skill_capability(skill_md) -> tuple[str, str, str, str | None]:
    """Read one bounded skill declaration without retaining its discovery path.

    Returns ``(name, description, instructions, skill_type)`` — ``skill_type``
    is the frontmatter's own ``skill_type`` declaration (or ``None`` when
    absent), threaded through to :func:`~agent_utilities.knowledge_graph.
    ingestion.skill_workflow_ingest.ingest_runnable_skill` so classification is
    a stored column rather than a value dropped on the floor at read time
    (CONCEPT:AU-KG.ingest.fleet-catalog-relational-tables).
    """
    path = Path(skill_md)
    payload = path.read_bytes()
    if not payload or len(payload) > 512 * 1024:
        raise ValueError("skill declaration size is invalid")
    content = payload.decode("utf-8")
    frontmatter, instructions = _parse_skill_capability_frontmatter(content)
    fallback_name = path.parent.name
    name = str(frontmatter.get("name") or fallback_name).strip()
    description = str(frontmatter.get("description") or "").strip()
    raw_skill_type = frontmatter.get("skill_type")
    skill_type = str(raw_skill_type).strip().lower() or None if raw_skill_type else None
    if not name or not instructions.strip():
        raise ValueError("skill declaration is incomplete")
    return name, description, instructions, skill_type


def _parse_skill_capability_frontmatter(content: str) -> tuple[dict, str]:
    """Split a skill declaration's YAML frontmatter from its instructions body.

    Returns ``(frontmatter, instructions)``. When there is no ``---``-delimited
    frontmatter block, ``frontmatter`` is empty and ``instructions`` is the
    whole content, unchanged.
    """
    import yaml

    frontmatter: dict = {}
    instructions = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) == 3:
            parsed = yaml.safe_load(parts[1].strip()) or {}
            if not isinstance(parsed, dict):
                raise ValueError("skill frontmatter must be an object")
            frontmatter = parsed
            instructions = parts[2].strip()
    return frontmatter, instructions


def _ingest_skill_capabilities(
    engine,
    provider: str,
    skills_path,
    *,
    include_names: frozenset[str] | None = None,
    skip_names: frozenset[str] = frozenset(),
) -> int:
    """Persist provider skills as runnable resources without retaining paths.

    The prior-``disabled`` lookup used to be one ``get_existing_disabled``
    engine round trip PER skill file inside this loop — for the full fleet
    skill catalog (hundreds of ``SKILL.md`` files under
    ``resolve_skill_provider_dirs()``) that is exactly the per-element engine
    call this codebase's own design rule forbids ("batch, never per-element;
    N elements in a loop = N round-trips = catastrophic" — see
    ``epistemic-graph`` AGENTS.md). It was the dominant contributor to the
    measured GraphOS cold-boot incident (2026-08-16): hundreds of sequential
    ``HasNode``/``GetNodeProperties``/``BatchUpdate`` round trips against a
    contended engine, each taking 1-15s under load. Every candidate's local
    ``SKILL.md`` is now parsed first (cheap, no engine call), then the
    "already ingested / disabled" state for the WHOLE batch is resolved with
    ONE :func:`get_existing_disabled_batch` call before any per-skill write.
    """
    from agent_utilities.core.providers import is_skill_graph_reference_path

    root = Path(skills_path)
    if not root.is_dir():
        return 0

    skill_files = (
        [root / "SKILL.md"]
        if (root / "SKILL.md").is_file()
        else sorted(
            skill_md
            for skill_md in root.rglob("SKILL.md")
            if not is_skill_graph_reference_path(skill_md, root)
        )
    )

    declarations = _collect_skill_declarations(
        skill_files, include_names=include_names, skip_names=skip_names
    )
    if not declarations:
        return 0

    disabled_by_resource = get_existing_disabled_batch(
        engine, [declaration[4] for declaration in declarations]
    )

    total = len(declarations)
    # Make an in-progress boot pass observable: this loop was previously
    # indistinguishable, in the container logs, from a hung process — an
    # operator saw only individual engine-op trace lines with no running
    # count or total, exactly the ambiguity that turned the 2026-08-16
    # cold-start incident into an 11-minute unattributed stall before the
    # startup probe killed the container. A bounded item count up front plus
    # a periodic "N/total" line lets "still working" be told apart from
    # "stuck" without reading engine wire traces.
    logger.info("GraphOS ingesting %d %s skill(s) from %s", total, provider, root)
    return _write_skill_declarations(
        engine,
        declarations,
        provider=provider,
        disabled_by_resource=disabled_by_resource,
    )


def _collect_skill_declarations(
    skill_files: list[Path],
    *,
    include_names: frozenset[str] | None,
    skip_names: frozenset[str],
) -> list[tuple[Path, str, str, str, str, str | None]]:
    """Parse candidate ``SKILL.md`` files into declarations, skipping bad ones.

    A malformed skill's parse failure is logged (stage="declaration") and
    that skill excluded — never blocks the batch (see
    :func:`_ingest_skill_capabilities`'s docstring for why this is a
    separate pass from the write loop).
    """
    from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
        skill_reference,
    )
    from agent_utilities.security.persistence_privacy import persistence_reference

    declarations: list[tuple[Path, str, str, str, str, str | None]] = []
    for skill_md in skill_files:
        fallback_name = skill_md.parent.name
        try:
            name, description, instructions, skill_type = _read_skill_capability(
                skill_md
            )
            if include_names is not None and name not in include_names:
                continue
            if name in skip_names:
                continue
            skill_slug = skill_reference(name).removeprefix("skill://")
            resource_id = f"resource:skill:{skill_slug}"
            declarations.append(
                (skill_md, name, description, instructions, resource_id, skill_type)
            )
        except Exception as exc:  # noqa: BLE001 - one malformed skill cannot block boot
            # ``exc.args[0]`` (not ``str(exc)``/``exc`` itself, and no
            # ``exc_info=True``) preserves the real cause for the operator
            # (test_boot_skill_failure_log_uses_neutral_reference) while
            # satisfying the served-boundary exception-surface gate. The
            # skill's discovery PATH is never in this message (``_read_
            # skill_capability`` raises path-free ValueErrors/YAML parser
            # errors), and the skill's own identity is already redacted via
            # ``persistence_reference`` above.
            logger.error(
                "Failed to ingest %s (stage=%s %s: %s)",
                persistence_reference(
                    "skill", fallback_name, namespace="skill-provider-ingest"
                ),
                "declaration",
                type(exc).__name__,
                exc.args[0] if exc.args else "",
            )
    return declarations


def _write_skill_declarations(
    engine,
    declarations: list[tuple[Path, str, str, str, str, str | None]],
    *,
    provider: str,
    disabled_by_resource: dict[str, bool],
) -> int:
    """Write each parsed skill declaration as a runnable resource.

    Logs an "N/total" progress line periodically (see
    :func:`_ingest_skill_capabilities`'s docstring for why). A malformed or
    failing write is logged (stage="write") and skipped — never blocks the
    batch.
    """
    from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
        ingest_runnable_skill,
    )
    from agent_utilities.security.persistence_privacy import persistence_reference

    total = len(declarations)
    ingested = 0
    for index, (
        skill_md,
        name,
        description,
        instructions,
        resource_id,
        skill_type,
    ) in enumerate(declarations, start=1):
        fallback_name = skill_md.parent.name
        try:
            ingest_runnable_skill(
                engine,
                name=name,
                description=description,
                instructions=instructions,
                provider=provider,
                disabled=disabled_by_resource.get(resource_id, False),
                skill_type=skill_type,
            )
            ingested += 1
            if index % 25 == 0 or index == total:
                logger.info(
                    "GraphOS skill ingestion progress: %d/%d (%s)",
                    index,
                    total,
                    provider,
                )
        except Exception as exc:  # noqa: BLE001 - one malformed skill cannot block boot
            logger.error(
                "Failed to ingest %s (stage=%s %s: %s)",
                persistence_reference(
                    "skill", fallback_name, namespace="skill-provider-ingest"
                ),
                "write",
                type(exc).__name__,
                exc.args[0] if exc.args else "",
            )
    return ingested


def _bundled_skill_contract() -> tuple[Path, dict[str, str]]:
    """Load the exact current packaged-skill digest contract."""
    from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
        runnable_skill_digest,
    )
    from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard
    from agent_utilities.skills import BUNDLED_SKILLS

    if len(BUNDLED_SKILLS) != 13 or len(set(BUNDLED_SKILLS)) != 13:
        raise GraphOSStartupReadinessError("graphos_bundled_skills_unready")
    root = Path(__file__).resolve().parents[1] / "skills"
    guard = PersistencePrivacyGuard()
    expected: dict[str, str] = {}
    try:
        for bundled_name in BUNDLED_SKILLS:
            name, _description, instructions, _skill_type = _read_skill_capability(
                root / bundled_name / "SKILL.md"
            )
            if name != bundled_name:
                raise ValueError("bundled skill identity mismatch")
            body, _privacy = guard.sanitize_text(instructions.strip())
            if not body:
                raise ValueError("bundled skill body is empty")
            expected[bundled_name] = runnable_skill_digest(body)
    except Exception as exc:
        # Only the exception type is recorded in the log (never its raw message
        # or traceback, D-LR-2); the real cause still propagates to the caller
        # via the chained `from exc` on the re-raise below.
        logger.error(
            "GraphOS packaged-skill readiness check failed (%s)",
            type(exc).__name__,
        )
        raise GraphOSStartupReadinessError("graphos_bundled_skills_unready") from exc
    return root, expected


def _query_bundled_skill_rows(
    engine: Any, expected_digests: dict[str, str]
) -> list[dict[str, Any]] | None:
    """Run the bundled-skill readiness probe query.

    Returns ``None`` (never raises) when the probe cannot run at all, or on
    a graph that does not exist yet — a first boot, or the first boot after
    a tenant claim starts scoping this process to a new tenant graph, where
    the engine answers "Graph '<name>' not found" rather than an empty
    result. That is the correct answer to "nothing is ready", not a
    failure, and treating it as fatal makes the server unable to perform
    the very ingestion that would create the graph. A genuine engine fault
    still surfaces from the caller's own use of the (empty) result.
    """
    query = getattr(engine, "query_cypher", None)
    if not callable(query):
        return None
    try:
        return query(
            "MATCH (n:CallableResource) WHERE n.name IN $names "
            "RETURN n.id AS id, n.name AS name, n.resource_type AS rtype, "
            "n.system_prompt AS system_prompt, "
            "n.instruction_digest AS instruction_digest, "
            "n.source_ref AS source_ref, n.runnable_bound AS runnable_bound",
            {"names": sorted(expected_digests)},
        )
    except Exception as exc:
        logger.info(
            "bundled-skill readiness probe found no existing skill graph "
            "(%s); treating every bundled skill as not yet ingested",
            exc,
        )
        return None


def _group_bundled_skill_candidates(
    rows: list[dict[str, Any]] | None, expected_digests: dict[str, str]
) -> dict[str, list[dict[str, Any]]]:
    """Bucket readiness-probe rows by skill name, ignoring unrequested names."""
    candidates: dict[str, list[dict[str, Any]]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        if name in expected_digests:
            candidates.setdefault(name, []).append(row)
    return candidates


def _bundled_skill_row_matches_contract(
    row: dict[str, Any], name: str, expected_digest: str, expected_ref: str
) -> bool:
    """Does this one candidate row satisfy the exact ready contract for ``name``?"""
    from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
        runnable_skill_digest,
    )

    body = str(row.get("system_prompt") or "").strip()
    digest = str(row.get("instruction_digest") or "")
    return (
        row.get("id") == f"resource:skill:{name}"
        and row.get("rtype") == "AGENT_SKILL"
        and row.get("runnable_bound") is True
        and row.get("source_ref") == expected_ref
        and bool(body)
        and digest == expected_digest
        and runnable_skill_digest(body) == digest
    )


def _is_bundled_skill_ready(
    name: str, expected_digest: str, matches: list[dict[str, Any]]
) -> bool:
    """Is any candidate row for ``name`` exactly the expected ready contract?"""
    from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
        skill_reference,
    )

    if len(matches) > 1:
        # Readiness asks "is a correct node present", and every check below
        # pins the exact node id `resource:skill:<name>`, so a second row can
        # never sneak past them. Requiring exactly ONE row instead conflated
        # "more than one row came back" with "not ready", which left a skill
        # permanently unready and — because this is a HARD startup gate —
        # kept graph-os from serving at all. Log the duplication as the
        # hygiene problem it is, then evaluate the rows on their merits.
        logger.warning(
            "bundled skill %r resolved to %d nodes; readiness is decided by "
            "the exact id resource:skill:%s",
            name,
            len(matches),
            name,
        )
    expected_ref = skill_reference(name)
    return any(
        _bundled_skill_row_matches_contract(row, name, expected_digest, expected_ref)
        for row in matches
    )


def _ready_bundled_skill_names(
    engine: Any, expected_digests: dict[str, str]
) -> frozenset[str]:
    """Return exact packaged skills already ready for delegated execution."""
    rows = _query_bundled_skill_rows(engine, expected_digests)
    candidates = _group_bundled_skill_candidates(rows, expected_digests)
    ready = {
        name
        for name, expected_digest in expected_digests.items()
        if _is_bundled_skill_ready(name, expected_digest, candidates.get(name, []))
    }
    return frozenset(ready)


def _ensure_bundled_skills_ready(engine: Any) -> dict[str, Any]:
    """Synchronously establish the packaged delegation contract before serving."""
    from agent_utilities.skills import BUNDLED_SKILLS

    try:
        root, expected = _bundled_skill_contract()
        ready_before = _ready_bundled_skill_names(engine, expected)
        missing = frozenset(BUNDLED_SKILLS) - ready_before
        ingested = 0
        if missing:
            ingested = _ingest_skill_capabilities(
                engine,
                "agent-utilities",
                root,
                include_names=missing,
            )
        ready_after = _ready_bundled_skill_names(engine, expected)
        if ready_after != frozenset(BUNDLED_SKILLS):
            # Name the skills that did not become ready. A bare count tells an
            # operator that something is wrong but not which thing, and this is a
            # HARD startup gate — the difference decides whether the server runs.
            logger.error(
                "GraphOS packaged-skill readiness incomplete "
                "(ready_before=%d ingested=%d ready_after=%d required=%d) "
                "not_ready=%s",
                len(ready_before),
                ingested,
                len(ready_after),
                len(BUNDLED_SKILLS),
                sorted(frozenset(BUNDLED_SKILLS) - ready_after),
            )
    except GraphOSStartupReadinessError:
        raise
    except Exception as exc:
        # The internal agent_utilities.* log is inside the process-wide
        # log_privacy.py sanitization boundary (paths/endpoints/emails
        # redacted, message preserved), so it carries the real exception here
        # for diagnosability. D-LR-2 still holds for the EXTERNAL boundary
        # below: the /health payload has no such sanitizer, so the "error"
        # field there stays type-only.
        logger.error(
            "GraphOS packaged-skill readiness check failed: %s",
            exc,
        )
        return {
            "required": len(BUNDLED_SKILLS),
            "already_ready": 0,
            "ingested": 0,
            "ready": 0,
            "not_ready": sorted(BUNDLED_SKILLS),
            # Unlike the logger.error above (an agent_utilities.* logger,
            # already inside the process-wide privacy boundary), this dict is
            # published via _set_bundled_skill_readiness for the /health HTTP
            # surface (agent_utilities/observability/runtime_health.py's
            # _check_bundled_skills) -- an external caller, so only the
            # exception TYPE is exposed here, never its raw text (D-LR-2).
            "error": type(exc).__name__,
        }
    return {
        "required": len(BUNDLED_SKILLS),
        "already_ready": len(ready_before),
        "ingested": ingested,
        "ready": len(ready_after),
        "not_ready": sorted(frozenset(BUNDLED_SKILLS) - ready_after),
    }


def _ingest_capabilities(engine, *, skip_skill_names: frozenset[str] = frozenset()):
    """Natively ingest MCP configurations, Native Tools, and Skills into the KG on startup."""
    _ingest_mcp_config_capabilities(engine)
    _ingest_native_tool_capabilities(engine)
    _ingest_skill_provider_capabilities(engine, skip_skill_names)

    # Fleet tool schemas stay lazy.  Startup has already materialized each MCP
    # server declaration above; probing every child here would launch the whole
    # fleet and contend with an operator's targeted ``list_catalog`` call.
    # Explicit ``source_sync(source="fleet")`` remains the governed full-scan
    # path when an operator wants every live tool schema elevated into the KG.


def _load_mcp_config_servers() -> dict[str, Any] | None:
    """Read + parse ``mcp_config.json``'s ``mcpServers`` map, or ``None`` if absent.

    Raises on a config file that exists but fails validation (oversized, not
    JSON, or not the expected shape) — the caller's boot-time try/except logs
    and skips this whole ingestion step on any of those.
    """
    import json

    import platformdirs

    APP_NAME = "agent-utilities"
    APP_AUTHOR = "knuckles-team"
    cfg_dir = Path(platformdirs.user_config_path(APP_NAME, APP_AUTHOR))
    mcp_config_path = cfg_dir / "mcp_config.json"
    if not mcp_config_path.is_file() or mcp_config_path.is_symlink():
        return None
    payload = mcp_config_path.read_bytes()
    if len(payload) > 4 * 1024 * 1024:
        raise ValueError("MCP configuration exceeds its ingestion bound")
    data = json.loads(payload)
    mcp_servers = data.get("mcpServers", {})
    if not isinstance(mcp_servers, dict):
        raise ValueError("MCP server registry must be an object")
    return mcp_servers


def _build_mcp_server_declarations(
    mcp_servers: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Build ``(node_id, declaration)`` pairs for every valid server entry."""
    declarations: list[tuple[str, dict[str, Any]]] = []
    for server_name, server_details in mcp_servers.items():
        if not isinstance(server_details, dict):
            continue
        node_id, declaration = _mcp_capability_declaration(server_name, server_details)
        declarations.append((node_id, declaration))
    return declarations


def _ingest_mcp_server_declarations(
    engine: Any, declarations: list[tuple[str, dict[str, Any]]]
) -> int:
    """Batch-resolve prior ``disabled`` state, then write every server node.

    One batched round trip for every server's prior ``disabled`` flag
    instead of one query per server (was the dominant source of the "slow
    engine call" warnings at boot).
    """
    disabled_by_id = get_existing_disabled_batch(
        engine,
        [node_id for node_id, _declaration in declarations],
        label="MCPServer",
    )
    ingested = 0
    for node_id, declaration in declarations:
        engine.add_node(
            node_id,
            "MCPServer",
            {**declaration, "disabled": disabled_by_id.get(node_id, False)},
        )
        ingested += 1
    return ingested


def _ingest_mcp_config_capabilities(engine: Any) -> None:
    """Section 1 of :func:`_ingest_capabilities`: ``mcp_config.json`` -> ``MCPServer`` nodes."""
    try:
        mcp_servers = _load_mcp_config_servers()
        if mcp_servers is None:
            return
        declarations = _build_mcp_server_declarations(mcp_servers)
        ingested = _ingest_mcp_server_declarations(engine, declarations)
        logger.info("Ingested %d MCP capability declarations", ingested)
    except Exception as exc:
        logger.error("Failed to ingest MCP configuration: %s", exc)


def _discover_native_tool_entries(
    tools_package: Any,
) -> list[tuple[str, dict[str, Any]]]:
    """Import every non-package module under ``tools_package`` and collect its
    agentic-versioned functions as ``(node_id, properties)`` pairs.
    """
    import importlib
    import inspect
    import pkgutil

    from agent_utilities.security.persistence_privacy import sanitize_for_persistence

    prefix = tools_package.__name__ + "."
    tool_entries: list[tuple[str, dict[str, Any]]] = []
    for _importer, modname, ispkg in pkgutil.iter_modules(
        tools_package.__path__, prefix
    ):
        if ispkg:
            continue
        try:
            module = importlib.import_module(modname)
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if not hasattr(obj, "__agentic_version__"):
                    continue
                node_id = f"native_tool_{name}"
                description, _privacy = sanitize_for_persistence(
                    (obj.__doc__ or "")[:8192]
                )
                tool_entries.append(
                    (
                        node_id,
                        {
                            "name": name,
                            "description": str(description),
                            "version": obj.__agentic_version__,
                            "module": modname,
                        },
                    )
                )
        except Exception as exc:  # noqa: BLE001 — per-module best-effort skip; the outer scan already logs failures
            logger.debug(
                "Failed to ingest a native-tool module: %s", type(exc).__name__
            )
    return tool_entries


def _ingest_native_tool_capabilities(engine: Any) -> None:
    """Section 2 of :func:`_ingest_capabilities`: scan ``agent_utilities.tools`` -> ``NativeTool`` nodes."""
    try:
        import agent_utilities.tools

        tool_entries = _discover_native_tool_entries(agent_utilities.tools)
        # One batched round trip for every native tool's prior ``disabled``
        # flag instead of one query per tool.
        disabled_by_id = get_existing_disabled_batch(
            engine,
            [node_id for node_id, _properties in tool_entries],
            label="NativeTool",
        )
        for node_id, properties in tool_entries:
            engine.add_node(
                node_id,
                "NativeTool",
                {**properties, "disabled": disabled_by_id.get(node_id, False)},
            )
        logger.info("Ingested Native Tools")
    except Exception as exc:
        logger.error("Failed to scan native tools: %s", exc)


def _ingest_skill_provider_capabilities(
    engine: Any, skip_skill_names: frozenset[str]
) -> None:
    """Section 3 of :func:`_ingest_capabilities`: every skill provider's ``SKILL.md`` files."""
    try:
        from agent_utilities.core.config import config
        from agent_utilities.core.providers import resolve_skill_provider_dirs

        sources = resolve_skill_provider_dirs()
        if config.custom_skills_directory:
            sources.append(("configured-overlay", Path(config.custom_skills_directory)))
        ingested = sum(
            _ingest_skill_capabilities(
                engine,
                provider,
                root,
                skip_names=skip_skill_names,
            )
            for provider, root in sources
        )
        if ingested:
            logger.info("Ingested %d runnable skills", ingested)
    except Exception as e:
        logger.error("Failed to ingest skills: %s", e)


# ── Boot hydration plan (ingestion-hydration-program.md §3) ─────────────────
#
# ``_ingest_capabilities`` above (mcp_config.json / native tools / skills) is
# the ORIGINAL boot hydration; the two helpers below extend it with the
# capability legs Phases C and E built but never wired to a boot call. Each is
# its own best-effort, exception-isolated step — same shape as the ontology
# federation sync already nested inside :func:`_start_engine_bootstrap`'s
# background thread — so a failure in one never skips, or blocks serving for,
# the others. Fleet tool-schema ingestion (Phase A) and the mcp_config.json
# router (Phase B) need no boot call here: A rides its own hourly
# ``deploy/schedules.yml`` cadence (``fleet-tool-schema-sync``) and B rides the
# always-on codebase sweep's ``_route_classified_artifacts`` fan-out.


def _record_boot_hydration_step(
    engine: Any, name: str, priority: int, status: str
) -> None:
    """Persist the small boot plan state when the active engine accepts nodes.

    The record is deliberately stable per step, not a new unbounded node for
    every process start.  It gives operators a durable answer to "which boot
    hydration phase last ran?" while keeping every actual ingest on its owned
    incremental/checkpointed implementation.
    """
    add_node = getattr(engine, "add_node", None)
    if not callable(add_node):
        return
    try:
        add_node(
            f"boot-hydration:{name}",
            "HydrationPlanStep",
            {
                "name": name,
                "priority": priority,
                "status": status,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
    except Exception:  # noqa: BLE001 - observability must not stop hydration
        logger.debug("boot hydration plan record failed for %s", name, exc_info=True)


def _hydrate_code_and_configured_connectors(engine: Any) -> None:
    """Queue the lowest-priority checkpointed hydration work.

    Code uses the existing breadth ingest (which performs its git-SHA pre-skip)
    and connectors use ``sweep_all_sources`` (which prechecks the signed
    provider contract before queue publication).  Empty configured roots are a
    valid no-op for a packaged/tiny deployment; no guessed workstation path is
    ever scanned.
    """
    from agent_utilities.core.config import config
    from agent_utilities.core.workspace_config import workspace_project_roots
    from agent_utilities.knowledge_graph.assimilation.breadth_ingest import (
        run_breadth_ingest,
    )
    from agent_utilities.knowledge_graph.core.source_sync import sweep_all_sources

    library_roots = [p for p in config.kg_breadth_library_roots.split(",") if p]
    repo_roots = [p for p in config.kg_breadth_repo_roots.split(",") if p]
    if not library_roots and not repo_roots:
        repo_roots = workspace_project_roots()
    if library_roots or repo_roots:
        run_breadth_ingest(engine, library_roots=library_roots, repo_roots=repo_roots)
    # This is intentionally enqueue-only.  ``sweep_all_sources`` rejects known
    # unavailable providers before creating work, so boot never spends an engine
    # lease on a connector that is guaranteed to fail.
    sweep_all_sources(engine, mode="delta", enqueue=True, priority=3)


def _enqueue_fleet_tool_schema_hydration(engine: Any) -> None:
    """Queue the live 65+ server tool-schema probe as priority-one boot work.

    MCP declarations are cheap and synchronous; the live schemas are network
    work and belong on the durable connector lane.  A stable target lets the
    WorkItem queue deduplicate restarts in the same hour without its O(N)
    target scan. A later hour gets a fresh delta probe.

    ``task_type="capability_hydration"`` (CONCEPT:AU-ORCH.scheduling.acquisition-lane-fairness), NOT the
    generic ``connector_sync`` the */20m fleet sweep uses for every OTHER connector.
    Both ride the same ``connectors`` lane (same soft-timeout envelope), but a
    distinct type lets the worker pool reserve this job a claim floor
    (:func:`agent_utilities.knowledge_graph.core.engine_tasks.start_task_workers`)
    instead of it only ever landing on a worker the moment one of potentially
    dozens of concurrently-running legacy connector syncs happens to free up —
    the proven starvation mode where priority alone could not preempt
    already-running work.
    """
    submit = getattr(engine, "submit_task", None)
    if not callable(submit):
        return
    job_id = submit(
        target_path="fleet",
        is_codebase=False,
        provenance={"sync_mode": "delta", "boot_hydration": True},
        task_type="capability_hydration",
        priority=1,
        skip_dedupe=True,
        job_id=f"boot:fleet-tool-schemas:{datetime.now(UTC):%Y%m%d%H}",
    )
    logger.info("Queued fleet MCP tool-schema boot hydration: %s", job_id)


def _run_boot_hydration_plan(
    engine: Any, *, skip_skill_names: frozenset[str] = frozenset()
) -> None:
    """Run GraphOS boot hydration in its fixed resource-priority order.

    1. bounded GraphOS/fleet tool metadata, then runnable skills and MCP declarations;
    2. prompts/agent templates;
    3. package ontologies; and
    4. codebases and configured connectors through their durable delta queues.

    Each step is isolated so a failed optional source cannot prevent later
    priority classes from making progress.
    """
    steps = (
        ("fleet_tool_schemas", 1, lambda: _enqueue_fleet_tool_schema_hydration(engine)),
        ("graphos_tool_surface", 1, lambda: _ingest_self_tool_surface_at_boot(engine)),
        (
            "capabilities",
            1,
            lambda: _ingest_capabilities(engine, skip_skill_names=skip_skill_names),
        ),
        ("prompts", 2, _ingest_prompts_at_boot),
        ("ontologies", 3, lambda: _sync_ontologies_at_boot(engine)),
        (
            "code_and_connectors",
            4,
            lambda: _hydrate_code_and_configured_connectors(engine),
        ),
    )
    for name, priority, step in steps:
        _record_boot_hydration_step(engine, name, priority, "running")
        try:
            step()
        except Exception:  # noqa: BLE001 - each plan leg is independently retryable
            _record_boot_hydration_step(engine, name, priority, "failed")
            logger.error("Boot hydration step %s failed", name, exc_info=True)
        else:
            _record_boot_hydration_step(engine, name, priority, "completed")
    _record_hydration_manifest(engine)


def _record_hydration_manifest(engine: Any) -> None:
    """Build, sign and persist the boot hydration manifest.

    CONCEPT:AU-KG.audit.hydration-manifest-signed / CONCEPT:AU-KG.audit.hydration-absent-vs-hidden

    This is the one production call site of
    :mod:`agent_utilities.knowledge_graph.ingestion.hydration_manifest`. Without
    it the module was a 912-line orphan: its signing and its serving-vs-service
    two-authority cross-check were correct but ran on no live path, so the
    absent-vs-hidden ambiguity it exists to resolve stayed unresolved for every
    real deployment (and ``scripts/check_surface_parity.py`` flagged it as an
    unexposed capability). Running it HERE — immediately after every plan leg has
    reported completed/failed — is what makes the manifest a truthful record of
    what that boot actually hydrated, rather than a snapshot of an arbitrary
    later moment.

    Best-effort by design, exactly like :func:`_record_boot_hydration_step`:
    boot hydration must never be blocked by its own audit record. A missing
    release-signing key is the normal case for a dev checkout and degrades to a
    debug line rather than a failed boot.
    """
    from agent_utilities.knowledge_graph.ingestion.hydration_manifest import (
        build_hydration_manifest,
        persist_hydration_manifest,
        sign_hydration_manifest,
    )

    try:
        manifest = build_hydration_manifest()
        signed = sign_hydration_manifest(manifest)
        persist_hydration_manifest(engine, signed)
    except Exception as exc:  # noqa: BLE001 - the audit record must never block
        # boot hydration itself. The cause IS logged (exc.args[0], never
        # str()/repr() -- test_record_hydration_manifest_never_blocks_boot
        # asserts the real message reaches the log) so a persistently
        # unsignable/unbuildable manifest is diagnosable rather than silent.
        logger.debug(
            "boot hydration manifest not recorded: %s",
            exc.args[0] if exc.args else type(exc).__name__,
        )
    else:
        logger.info(
            "Recorded signed hydration manifest (generated_at=%s)",
            manifest.generated_at,
        )


def _ingest_prompts_at_boot() -> None:
    """Hydrate the ``:Prompt`` corpus at boot (Phase C → Phase F wiring).

    :func:`agent_utilities.agent.registry_builder.ingest_prompts_to_graph` is
    content-hash incremental (CONCEPT:EG-KG.storage.nonblocking-checkpoint,
    ``DeltaManifest`` category ``"prompt_base"``), so calling it on every boot
    is cheap: the first boot upserts every prompt, every later boot skips the
    unchanged ones. This runs inside the background bootstrap thread, which
    has no running event loop, so ``asyncio.run`` is the correct, safe way to
    drive the coroutine (mirrors the existing synchronous callers in
    ``registry_builder.py``/``package_install_ingest.py``) — never blocks
    graph-os startup or serving, and a failure here is isolated and logged,
    never raised into the caller.
    """
    try:
        from agent_utilities.agent.registry_builder import ingest_prompts_to_graph

        asyncio.run(ingest_prompts_to_graph())
        logger.info("Ingested prompt-base library at boot (Phase C hydration)")
    except Exception as exc:
        logger.error("Prompt-base boot ingestion failed: %s", exc)


def _graphos_self_tool_surface() -> list[dict[str, Any]]:
    """In-process snapshot of graph-os's own registered MCP tool surface.

    CONCEPT:AU-KG.ingest.self-tool-surface — the exact shape
    :func:`~agent_utilities.knowledge_graph.ingestion.engine.register_self_tool_surface_provider`
    expects: a plain, synchronous, zero-argument callable that reads the
    already-built ``REGISTERED_TOOLS`` dict (populated by
    ``register_tool_surface`` during :func:`_build_server`, which always runs
    before this process starts serving). Reading a module-global dict is the
    entire implementation — no network call, no MCP round-trip, no self-probe.
    """
    return [
        {
            "name": name,
            "description": (getattr(func, "__doc__", None) or "").strip(),
        }
        for name, func in sorted(REGISTERED_TOOLS.items())
    ]


def _ingest_self_tool_surface_at_boot(engine: Any) -> None:
    """Queue graph-os's own ~95 ``:MCPServer``/``:Tool`` nodes at boot.

    CONCEPT:AU-KG.ingest.self-tool-surface (Phase E → Phase F wiring). The
    provider is registered synchronously because workers share this process,
    while the native ChangeEnvelope materialization runs as a fenced,
    checkpointable priority-one WorkItem. Queue publication therefore cannot
    hold up prompts, ontologies, or later boot-plan checkpoints when the
    operational graph is cold or write-contended.
    """
    try:
        from agent_utilities.knowledge_graph.ingestion.engine import (
            register_self_tool_surface_provider,
        )

        register_self_tool_surface_provider(_graphos_self_tool_surface)
        submit = getattr(engine, "submit_task", None)
        if not callable(submit):
            return
        surface_digest = hashlib.sha256(
            json.dumps(
                _graphos_self_tool_surface(),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:16]
        job_id = submit(
            target_path="graph-os",
            is_codebase=False,
            provenance={"boot_hydration": True},
            task_type="self_tool_surface",
            priority=1,
            skip_dedupe=True,
            job_id=f"boot:self-tool-surface:{surface_digest}",
        )
        logger.info("Queued self tool-surface boot hydration: %s", job_id)
    except Exception as exc:
        logger.error("Self tool-surface boot enqueue failed: %s", exc)


def _sync_ontologies_at_boot(engine: Any) -> None:
    """Load package ontologies after runnable resources are available.

    CONCEPT:AU-KG.ontology.integrity-bootstrap — ``activate_graph()`` runs FIRST
    and unconditionally, so the dedicated ontology graph's SHACL/ICV integrity
    policy is registered even on a boot with zero federated ontology content
    to load (which would otherwise never reach the ``load()``/``_load_axioms``
    chokepoint that also performs activation). Idempotent; safe every boot.
    """
    sync_packages = engine._ontology_package_sync
    from agent_utilities.knowledge_graph.ontology.lifecycle import OntologyLifecycle

    lc = OntologyLifecycle(engine=engine)
    activation = lc.activate_graph()
    if not activation.get("activated") and activation.get("reason") not in (
        "no engine RDF surface",
    ):
        logger.error(
            "Ontology graph activation failed at boot: %s", activation.get("reason")
        )

    report = sync_packages(lc)
    if report.get("providers_loaded"):
        logger.info(
            "Ontology federation: loaded %d package ontolog(ies) at boot",
            report["providers_loaded"],
        )


def _set_readiness_authority(session: Any) -> None:
    """Hand the readiness probe the process's own verified authority."""

    from agent_utilities.observability.runtime_health import set_readiness_authority

    set_readiness_authority(session)


def _mint_process_session(transport: str) -> Any:
    """Mint the process's verified graph authority.

    Tiny packaged-local stdio uses an in-memory asymmetric authority. Every
    other topology resolves a token reference or performs OAuth2 client
    credentials, then validates the result through the same JWKS path as HTTP.
    The authority scopes background engine bootstrap for every transport and is
    additionally used for stdio tool calls, which have no request Authorization
    header.
    """
    from agent_utilities.core.config import config
    from agent_utilities.security.request_identity import (
        acquire_process_identity_token,
        local_process_authority_enabled,
        mint_actor_from_token_sync,
        mint_graph_session,
        mint_local_process_session,
    )

    if transport == "stdio" and local_process_authority_enabled(config):
        session = mint_local_process_session()
    else:
        token = acquire_process_identity_token(config)
        actor = mint_actor_from_token_sync(token)
        from agent_utilities.security.brain_context import CredentialLease

        expires_at = getattr(actor, "credential_expires_at", None)
        if expires_at is None:
            raise RuntimeError("Graph process identity has no bounded expiry")
        actor = replace(
            actor,
            credential_lease=CredentialLease(int(expires_at)),
        )
        session = mint_graph_session(actor)
    session.engine_verified_context()
    logger.info("Verified graph process authority minted")
    return session


def _same_process_authority(left: Any, right: Any) -> bool:
    """Return whether a renewed token preserves the original authority."""
    fields = (
        "actor_id",
        "actor_type",
        "roles",
        "tenant_id",
        "authenticated",
        "groups",
    )
    return all(
        getattr(left, name, None) == getattr(right, name, None) for name in fields
    )


def _renewed_process_actor(config: Any) -> Any:
    from agent_utilities.security.request_identity import (
        acquire_process_identity_token,
        local_process_authority_enabled,
        mint_actor_from_token_sync,
        mint_local_process_session,
    )

    if local_process_authority_enabled(config):
        return mint_local_process_session().actor
    token = acquire_process_identity_token(config)
    try:
        return mint_actor_from_token_sync(token)
    finally:
        del token


def _refresh_process_authority(session: Any) -> Any:
    """Renew one process lease without replacing captured sessions.

    External identities reacquire and validate their configured token. Tiny
    packaged-local stdio remints its in-memory asymmetric proof instead; it
    never falls through to an external-token lookup it cannot satisfy. After
    validation, only the bounded expiry is copied into the shared in-memory
    lease. Identity, roles, tenant, route, and policy may not change.
    """
    from agent_utilities.core.config import config
    from agent_utilities.knowledge_graph.core.session import SessionExpiredError

    lease = getattr(getattr(session, "actor", None), "credential_lease", None)
    if lease is None:
        raise RuntimeError("Graph process authority is not renewable")
    with _PROCESS_SESSION_REFRESH_LOCK:
        try:
            session.ensure_authority_current(minimum_ttl_seconds=30)
            return session
        except SessionExpiredError:  # noqa: BLE001 — expected: falls through to the renewal path below
            pass
        renewed_actor = _renewed_process_actor(config)
        if not _same_process_authority(session.actor, renewed_actor):
            raise RuntimeError("Graph process authority changed during renewal")
        expires_at = getattr(renewed_actor, "credential_expires_at", None)
        if expires_at is None or int(expires_at) <= int(time.time()) + 30:
            raise RuntimeError("Graph process authority renewal is too short-lived")
        lease.renew(int(expires_at))
        session.ensure_authority_current(minimum_ttl_seconds=30)
        return session


async def _ensure_process_authority_current() -> Any:
    """Ensure request/process authority without blocking the MCP event loop.

    D-SNV-5: an ambient session whose actor carries a renewable
    ``credential_lease`` (a server-minted process/client-credentials identity
    — never a caller-presented bearer JWT, which has no ``credential_lease``
    and stays exactly as fail-closed as before) is proactively renewed here
    the same way the stdio ``_PROCESS_SESSION`` fallback already was. This is
    the dispatch-entry check only; :func:`_keep_process_authority_current`
    covers the rest of a long-running dispatch.
    """
    from agent_utilities.knowledge_graph.core.session import (
        SessionExpiredError,
        current_session,
    )

    ambient = current_session()
    if ambient is not None:
        if getattr(getattr(ambient, "actor", None), "credential_lease", None) is None:
            # Not server-renewable: unchanged fail-closed behavior, no
            # minimum-TTL headroom requirement — this must never mask a
            # caller's real credential expiry.
            ambient.ensure_authority_current()
            return ambient
        try:
            ambient.ensure_authority_current(minimum_ttl_seconds=30)
        except SessionExpiredError:
            ambient = await asyncio.to_thread(_refresh_process_authority, ambient)
        if ambient is None:
            # `_refresh_process_authority` is typed `Any` (it renews in place and
            # returns the same session), so this should be unreachable in
            # practice — but if it ever did return nothing, failing closed with
            # the same PermissionError the no-ambient-session branch below uses
            # is correct, not an opaque AttributeError on the next line.
            raise PermissionError("Verified GraphSession required")
        ambient.ensure_authority_current(minimum_ttl_seconds=30)
        return ambient
    session = _PROCESS_SESSION
    if session is None:
        raise PermissionError("Verified GraphSession required")
    try:
        session.ensure_authority_current(minimum_ttl_seconds=30)
    except SessionExpiredError:
        session = await asyncio.to_thread(_refresh_process_authority, session)
    session.ensure_authority_current(minimum_ttl_seconds=30)
    return session


async def _keep_process_authority_current(session: Any) -> None:
    """Background keepalive for one in-flight dispatch under a renewable session.

    A tool dispatch may run for the whole ``_TOOL_CALL_TIMEOUT_S`` window
    (``_execute_tool``), but authority was previously checked only once, at
    entry (D-SNV-5: a real 192s ServiceNow delegation died mid-flight with
    ``SessionExpiredError`` because nothing renewed it after that). This loop
    renews the SAME mutable ``CredentialLease`` the dispatch's ambient
    ``GraphSession.actor`` already holds, so every downstream authority check
    (``GraphSession.ensure_authority_current`` at every engine boundary, e.g.
    ``graph_compute.py``'s ``_invoke_at``) sees it transparently — no session
    object is replaced and no verification is weakened; a caller-presented
    bearer JWT (no lease) never reaches this function at all.

    Runs only for the lifetime the caller's :func:`authority_keepalive_scope`
    is open (started and cancelled there) — never a free-running background
    task.
    """
    from agent_utilities.knowledge_graph.core.session import SessionExpiredError

    lease = session.actor.credential_lease
    try:
        while True:
            seconds_left = lease.expires_at - int(time.time())
            await asyncio.sleep(max(1.0, min(30.0, seconds_left - 30.0)))
            try:
                await asyncio.to_thread(_refresh_process_authority, session)
            except SessionExpiredError:
                # Fail-closed: the next real authority check (the deep engine
                # call already in flight, or the next one) raises for real.
                # D-SWG-3: log it here too so the keepalive giving up is visible
                # at the moment it happens, not only inferred later from a
                # downstream failure.
                logger.warning(
                    "Delegation authority keepalive stopping: session lease "
                    "already expired; the next authority check will fail closed"
                )
                return
            except Exception as exc:  # noqa: BLE001 — keepalive is best-effort renewal; a hard failure surfaces at the next real authority check either way
                logger.error(
                    "Delegation authority keepalive renewal failed (exception_type=%s)",
                    type(exc).__name__,
                )
    except asyncio.CancelledError:
        # D-SWG-3: expected, normal shutdown — the caller's
        # authority_keepalive_scope cancels this loop when the wrapped call
        # finishes. Logged at debug (routine, high-frequency) rather than
        # dropped silently.
        logger.debug("Delegation authority keepalive cancelled (scope exited)")


@contextlib.asynccontextmanager
async def authority_keepalive_scope(session: Any | None = None) -> AsyncIterator[None]:
    """Renew a renewable session's authority for the duration of the wrapped call.

    CONCEPT:AU-ORCH.execution.delegation-hot-path-authority — keep a long delegation authorized for its whole run, on every entrypoint, not just MCP dispatch.

    Every delegation entrypoint — the MCP ``_execute_tool`` dispatch, the
    ``agent-webui``/REST gateway, the messaging router (Telegram/Mattermost), the
    autonomous ``agent_dispatch_worker``, ``org_runtime``, a governed dynamic
    workflow, and the parallel engine — converges on the single function
    :meth:`agent_utilities.orchestration.manager.Orchestrator.execute_agent`.
    A background keepalive wired ONLY into ``_execute_tool`` (the original
    D-SNV-5 fix) therefore covered MCP-dispatched delegations only; anything that
    reached ``execute_agent`` some other way still expired mid-flight with
    ``SessionExpiredError``. This context manager is the one reusable primitive
    both ``_execute_tool`` and ``Orchestrator.execute_agent`` open, so every
    surface inherits the renewal by construction instead of each caller
    reimplementing it.

    Security posture is unchanged from the original ``_execute_tool``-only fix:

    * Only a server-minted, renewable ``credential_lease`` is ever proactively
      renewed. A caller-presented bearer JWT carries no ``credential_lease`` and
      is never touched here — it stays exactly as fail-closed as before.
    * The SAME mutable :class:`~agent_utilities.security.brain_context.CredentialLease`
      object the session already holds is renewed in place, so every downstream
      authority check (``GraphSession.ensure_authority_current``, e.g.
      ``graph_compute.py``'s ``_invoke_at``) sees it transparently — no session
      object is replaced and the expiry check itself is never weakened or
      lengthened.
    * ``session`` defaults to the ambient :func:`GraphSession
      <agent_utilities.knowledge_graph.core.session.current_session>` (falling
      back to the stdio ``_PROCESS_SESSION``, same as ``verified_tool_session_scope``)
      resolved WITHOUT raising when neither is set — this helper only ever adds
      renewal; it must never introduce a new precondition. A caller with no
      ambient/process session at all fails exactly as it already would, at the
      first real engine boundary (``SessionRequiredError``), not here.

    Idempotent/reentrant via :data:`_AUTHORITY_KEEPALIVE_ACTIVE`: a scope nested
    inside another already-open scope on the same task tree (a nested MCP
    dispatch that itself calls ``execute_agent``, or vice versa) is a no-op —
    the outer scope already renews the same lease, and the extra loop would only
    add redundant token round-trips, not extra safety.
    """
    if _AUTHORITY_KEEPALIVE_ACTIVE.get():
        yield
        return

    if session is None:
        from agent_utilities.knowledge_graph.core.session import current_session

        session = current_session() or _PROCESS_SESSION
    if session is None:
        yield
        return

    lease = getattr(getattr(session, "actor", None), "credential_lease", None)
    if lease is None:
        yield
        return

    guard_token = _AUTHORITY_KEEPALIVE_ACTIVE.set(True)
    keepalive = asyncio.ensure_future(_keep_process_authority_current(session))
    try:
        yield
    finally:
        keepalive.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await keepalive
        _AUTHORITY_KEEPALIVE_ACTIVE.reset(guard_token)


def _process_authority_refresh_loop(session: Any) -> None:
    """Keep a renewable process lease current for all captured worker sessions."""
    lease = getattr(getattr(session, "actor", None), "credential_lease", None)
    if lease is None:
        return
    while not _PROCESS_AUTHORITY_STOP.is_set():
        seconds_left = lease.expires_at - int(time.time())
        if seconds_left > 30:
            _PROCESS_AUTHORITY_STOP.wait(min(60.0, max(1.0, seconds_left - 30.0)))
            continue
        try:
            _refresh_process_authority(session)
        except Exception as exc:  # noqa: BLE001 - retry; expiry remains fail-closed
            logger.error(
                "Graph process authority renewal failed (exception_type=%s)",
                type(exc).__name__,
            )
            _PROCESS_AUTHORITY_STOP.wait(5.0)


def _start_process_authority_supervisor(session: Any) -> None:
    """Start the sole external-authority renewal supervisor when required."""
    global _PROCESS_AUTHORITY_THREAD
    _stop_process_authority_supervisor()
    _PROCESS_AUTHORITY_STOP.clear()
    if getattr(getattr(session, "actor", None), "credential_lease", None) is None:
        return
    thread = threading.Thread(
        target=_process_authority_refresh_loop,
        args=(session,),
        daemon=True,
        name="GraphProcessAuthority",
    )
    _PROCESS_AUTHORITY_THREAD = thread
    thread.start()


def _stop_process_authority_supervisor() -> None:
    """Stop and forget the process-authority supervisor."""
    global _PROCESS_AUTHORITY_THREAD
    _PROCESS_AUTHORITY_STOP.set()
    thread = _PROCESS_AUTHORITY_THREAD
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=2.0)
    _PROCESS_AUTHORITY_THREAD = None


_BUNDLED_SKILL_READINESS: dict[str, Any] = {}
# A 9–10 GiB four-shard graph can legitimately need just over five minutes to
# rebuild its lazy-open indexes on slower storage. Keep this within the
# deployment's ten-minute startup probe while avoiding a pointless restart at
# the former five-minute boundary, which merely repeated the cold-open work.
_ENGINE_MATERIALIZATION_TIMEOUT_SECONDS = 540.0
# Materialization progresses in durable page batches measured in seconds on a
# large graph.  Four manifest reads per second added ~1,200 Python→native calls
# to a five-minute cold open without improving correctness.  One authoritative
# poll per second keeps readiness latency bounded while leaving the engine's
# foreground read lane available for the materializer and health probes.
_ENGINE_MATERIALIZATION_POLL_SECONDS = 1.0


def _set_bundled_skill_readiness(report: dict[str, Any]) -> None:
    """Publish packaged-skill readiness so /health can report it.

    Readiness no longer gates boot, so it MUST be observable at runtime —
    otherwise "serving degraded" is indistinguishable from "fully ready" to
    anything outside the process, which is the silent-failure pattern this
    codebase keeps getting bitten by.
    """
    _BUNDLED_SKILL_READINESS.clear()
    _BUNDLED_SKILL_READINESS.update(report)


def bundled_skill_readiness() -> dict[str, Any]:
    """The last packaged-skill readiness report (empty before bootstrap runs)."""
    return dict(_BUNDLED_SKILL_READINESS)


def _resolve_materialization_handles(engine: Any) -> tuple[Any, Any, str]:
    """Resolve the engine's native ``(query_cypher, list_graphs, graph_name)``.

    ``query_cypher``/``list_graphs`` come back ``None`` when the engine
    doesn't participate in native lifecycle (lightweight test engines and
    non-native backends have neither ``client`` nor ``graph_name``) or has
    no graph name at all — the caller treats either as "not_applicable".

    GraphOS owns the high-level IntelligenceGraphEngine; the lifecycle
    manifest belongs to its native GraphComputeEngine authority. Test tools
    and lower-level callers may pass that authority directly.
    """
    native_engine = getattr(engine, "graph_compute", None) or engine
    client = getattr(native_engine, "client", None)
    graph_name = str(getattr(native_engine, "graph_name", "") or "")
    query_cypher = getattr(native_engine, "query_cypher", None)
    tenants = getattr(client, "tenants", None)
    list_graphs = getattr(tenants, "list", None)
    if not graph_name or not callable(query_cypher) or not callable(list_graphs):
        return None, None, graph_name
    return query_cypher, list_graphs, graph_name


def _engine_read_probe_status(query_cypher: Any) -> str:
    """Run the one bounded read that both triggers and probes materialization.

    Returns ``"complete"`` (the read succeeded), ``"partial"`` (the read hit
    ``PARTIAL_MATERIALIZATION`` — the caller should keep polling the
    manifest), or ``"absent"`` (the graph does not exist). Any other
    exception propagates unchanged.
    """
    try:
        query_cypher("MATCH (n) RETURN n.id AS id LIMIT 1")
    except Exception as exc:
        # Control-flow only: this text never reaches a log or a caller, so it
        # is read via `exc.args` (never `str(exc)`/`repr(exc)`) to stay clear
        # of the served-boundary exception-surface policy on principle.
        detail = str(exc.args[0]) if exc.args else ""
        if "PARTIAL_MATERIALIZATION" in detail:
            return "partial"
        if "not found" in detail.lower():
            return "absent"
        raise
    return "complete"


def _resolve_manifest_entry(
    list_graphs: Any, graph_name: str, manifest_visible: bool | None
) -> tuple[dict[str, Any] | None, bool | None]:
    """Resolve this graph's manifest entry from ``list_graphs()``.

    Respects the "already known hidden" cache — once a poll has found the
    graph absent from the catalog (RLS-filtered), later polls skip
    re-querying the whole catalog and go straight to the read-probe
    fallback. Returns ``(entry, manifest_visible)``.
    """
    if manifest_visible is False:
        return None, manifest_visible
    entries = list_graphs() or []
    entry = next(
        (
            value
            for value in entries
            if (value.get("name") if isinstance(value, dict) else None) == graph_name
        ),
        None,
    )
    return entry, isinstance(entry, dict)


def _handle_materialized_manifest_entry(
    entry: dict[str, Any],
    graph_name: str,
    last_progress: tuple[str, int | None, int | None] | None,
) -> tuple[dict[str, Any] | None, tuple[str, int | None, int | None] | None]:
    """Interpret one polled, catalog-visible manifest entry.

    Returns ``(result, updated_last_progress)`` where ``result`` is the
    barrier's terminal return value once materialization is complete, or
    ``None`` to keep polling. Raises on a ``failed`` materialization phase.
    """
    phase = str(entry.get("materialization") or "unknown")
    valid = entry.get("valid") is True
    cursor = entry.get("completeness_cursor")
    node_offset = cursor.get("node_offset") if isinstance(cursor, dict) else None
    edge_offset = cursor.get("edge_offset") if isinstance(cursor, dict) else None
    progress = (phase, node_offset, edge_offset)
    if progress != last_progress:
        logger.info(
            "Epistemic graph materialization progress "
            "(graph=%s phase=%s node_offset=%s edge_offset=%s)",
            graph_name,
            phase,
            node_offset,
            edge_offset,
        )
        last_progress = progress
    if phase == "complete" and valid:
        logger.info(
            "Epistemic graph materialization ready "
            "(graph=%s node_offset=%s edge_offset=%s)",
            graph_name,
            node_offset,
            edge_offset,
        )
        return dict(entry), last_progress
    if phase == "failed":
        raise RuntimeError(
            "epistemic graph materialization failed "
            f"(graph={graph_name!r}, cursor={cursor!r})"
        )
    return None, last_progress


def _handle_hidden_manifest_entry(
    query_cypher: Any, graph_name: str
) -> dict[str, Any] | None:
    """Handle a poll where the graph's manifest entry is not catalog-visible.

    RLS may intentionally hide a protected graph (for example
    ``__secrets__``) from the catalog while still authorizing this
    process-scoped graph view. In that case the same bounded read that
    triggered materialization is the authoritative completion probe. Do not
    weaken catalog RLS merely to make boot observable. Returns a terminal
    result dict, or ``None`` to keep polling.
    """
    status = _engine_read_probe_status(query_cypher)
    if status == "absent":
        return {"graph": graph_name, "materialization": "absent"}
    if status == "complete":
        logger.info(
            "Epistemic graph materialization ready via authorized "
            "read probe (graph=%s; catalog manifest hidden)",
            graph_name,
        )
        return {
            "graph": graph_name,
            "materialization": "complete",
            "valid": True,
            "manifest_visible": False,
        }
    return None


def _initial_materialization_probe(
    engine: Any,
) -> tuple[dict[str, Any] | None, Any, Any, str]:
    """Resolve native handles, then run the initial bounded-read probe.

    Returns ``(early_result, query_cypher, list_graphs, graph_name)``.
    ``early_result`` is non-``None`` when the caller should return it
    immediately (``not_applicable`` / ``absent`` / ``complete``) rather than
    entering the manifest poll loop.
    """
    query_cypher, list_graphs, graph_name = _resolve_materialization_handles(engine)
    if query_cypher is None or list_graphs is None:
        early = {"graph": graph_name or None, "materialization": "not_applicable"}
        return early, query_cypher, list_graphs, graph_name

    status = _engine_read_probe_status(query_cypher)
    if status == "absent":
        return (
            {"graph": graph_name, "materialization": "absent"},
            query_cypher,
            list_graphs,
            graph_name,
        )
    if status == "complete":
        return (
            {"graph": graph_name, "materialization": "complete", "valid": True},
            query_cypher,
            list_graphs,
            graph_name,
        )
    return None, query_cypher, list_graphs, graph_name


def _poll_materialization_step(
    list_graphs: Any,
    query_cypher: Any,
    graph_name: str,
    last_progress: tuple[str, int | None, int | None] | None,
    manifest_visible: bool | None,
) -> tuple[
    dict[str, Any] | None,
    tuple[str, int | None, int | None] | None,
    bool | None,
]:
    """Run one manifest-poll iteration of :func:`_wait_for_engine_materialization`.

    Returns ``(result, last_progress, manifest_visible)``; ``result`` is the
    barrier's terminal return value once resolved, or ``None`` to keep
    polling.
    """
    entry, manifest_visible = _resolve_manifest_entry(
        list_graphs, graph_name, manifest_visible
    )
    if isinstance(entry, dict):
        result, last_progress = _handle_materialized_manifest_entry(
            entry, graph_name, last_progress
        )
        return result, last_progress, manifest_visible
    result = _handle_hidden_manifest_entry(query_cypher, graph_name)
    return result, last_progress, manifest_visible


def _wait_for_engine_materialization(
    engine: Any,
    *,
    timeout_seconds: float = _ENGINE_MATERIALIZATION_TIMEOUT_SECONDS,
    poll_seconds: float = _ENGINE_MATERIALIZATION_POLL_SECONDS,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Wait for a lazy-open graph to become complete before boot writes begin.

    The epistemic engine deliberately serves its catalog before a cold graph has
    finished paging into memory.  Reads and writes against that graph correctly
    fail with ``PARTIAL_MATERIALIZATION`` until its durable manifest is both
    ``complete`` and ``valid``.  GraphOS boot hydration and task workers are not
    ordinary retrying callers: starting them during that interval can discard
    their one-shot startup work.  This barrier starts lazy-open with one bounded
    read, then observes the authoritative ``ListGraphs`` manifest until it is
    safe to perform any boot mutation.

    Lightweight test engines and non-native backends have neither ``client`` nor
    ``graph_name`` and therefore do not participate in this native lifecycle.
    A graph absent from a new empty engine is likewise left for normal creation.
    """
    early_result, query_cypher, list_graphs, graph_name = (
        _initial_materialization_probe(engine)
    )
    if early_result is not None:
        return early_result

    logger.info(
        "GraphOS waiting for epistemic graph materialization before hydration "
        "(graph=%s timeout_seconds=%g)",
        graph_name,
        timeout_seconds,
    )
    shutdown = stop_event or _PROCESS_AUTHORITY_STOP
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    last_progress: tuple[str, int | None, int | None] | None = None
    manifest_visible: bool | None = None
    while True:
        result, last_progress, manifest_visible = _poll_materialization_step(
            list_graphs, query_cypher, graph_name, last_progress, manifest_visible
        )
        if result is not None:
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "epistemic graph did not become completely materialized before "
                f"GraphOS boot hydration (graph={graph_name!r}, "
                f"timeout_seconds={timeout_seconds:g}, last={last_progress!r})"
            )
        if shutdown.wait(max(0.0, poll_seconds)):
            raise InterruptedError(
                "GraphOS shutdown cancelled the graph materialization barrier"
            )


def _engine_bootstrap_is_client(engine: Any, fallback_role: str) -> bool:
    """Return whether either the requested or elected engine role is client."""
    requested_role = (
        (getattr(engine, "_daemon_role", None) or fallback_role).strip().lower()
    )
    effective_role = (
        (getattr(engine, "_effective_role", None) or requested_role).strip().lower()
    )
    return "client" in {requested_role, effective_role}


def _run_enabled_boot_hydration(
    engine: Any,
    *,
    client_role: bool,
    background_sync_enabled: bool,
    skip_skill_names: frozenset[str],
) -> None:
    """Hydrate only on the elected background-sync host."""
    if client_role or not background_sync_enabled:
        return
    _run_boot_hydration_plan(engine, skip_skill_names=skip_skill_names)


def _start_engine_bootstrap(session: Any) -> None:
    """Establish engine/skill readiness, then start noncritical services."""
    from agent_utilities.core.config import config
    from agent_utilities.knowledge_graph.core.engine_tasks import (
        _authorized_background_thread,
        _require_verified_background_session,
        daemon_role,
    )
    from agent_utilities.knowledge_graph.core.session import use_session
    from agent_utilities.security.brain_context import use_actor
    from agent_utilities.skills import BUNDLED_SKILLS

    verified_session = _require_verified_background_session(session)
    with (
        use_actor(verified_session.actor),
        use_session(verified_session),
    ):
        engine = _get_engine()
        # Correctness gate: unlike missing packaged skills, a partially
        # materialized graph cannot safely accept one-shot boot hydration or
        # worker claims.  Let this failure stop startup so the orchestrator can
        # retry the process instead of advertising a silently incomplete graph.
        _wait_for_engine_materialization(engine)
    try:
        with (
            use_actor(verified_session.actor),
            use_session(verified_session),
        ):
            readiness = _ensure_bundled_skills_ready(engine)
    except Exception as exc:
        # Packaged-skill readiness is a CAPABILITY concern, not a correctness or
        # security one, so it must not decide whether graph-os serves at all. A
        # server that refuses to boot because some bundled skills did not ingest
        # takes down every unrelated tool, the health surface, and the operator's
        # ability to diagnose the very problem — the failure mode is far worse
        # than running degraded. Record it, surface it in /health, keep serving.
        # The LOG line preserves the real cause (an operator needs to see WHICH
        # packaged skill failed and why, not just "SERVING DEGRADED" for every
        # distinct cause — HANDOFF-2026-07-22 turned exactly this omission into
        # an hours-long dead end); ``exc.args[0]`` (not ``str(exc)``/``exc``
        # itself, and no ``exc_info=True``) keeps the served-boundary
        # exception-surface gate satisfied. The `/health`-published readiness
        # dict below is a DIFFERENT, wider-audience surface and stays
        # type-only (D-LR-2).
        logger.error(
            "GraphOS packaged-skill bootstrap failed; SERVING DEGRADED (%s: %s)",
            type(exc).__name__,
            exc.args[0] if exc.args else "",
        )
        _set_bundled_skill_readiness(
            {
                "required": len(BUNDLED_SKILLS),
                "ready": 0,
                "not_ready": sorted(BUNDLED_SKILLS),
                # See _ensure_bundled_skills_ready's identical comment: this
                # dict is published for the /health HTTP surface, not logged,
                # so only the exception TYPE is exposed here (D-LR-2).
                "error": type(exc).__name__,
            }
        )
        return

    _set_bundled_skill_readiness(readiness)
    if readiness.get("not_ready"):
        logger.error(
            "GraphOS is SERVING DEGRADED: %d/%d packaged skills ready, not_ready=%s",
            readiness.get("ready", 0),
            readiness.get("required", 0),
            readiness.get("not_ready"),
        )
    logger.info(
        "GraphOS packaged-skill readiness established (%d/%d)",
        readiness["ready"],
        readiness["required"],
    )
    # An explicit client role is a hard serving-plane boundary. In particular,
    # stale KG_LOOP/maintenance settings must not turn a network-facing GraphOS
    # process into an autonomous scheduler or queue worker. The requested role
    # captured on the engine wins over any later process-environment mutation.
    client_role = _engine_bootstrap_is_client(engine, daemon_role())
    if not client_role:
        # BUG-295 (NE-009/NE-020): the daemon role is the one that runs the
        # unified scheduler (start_daemons() below), whose every tick reads
        # and writes the isolated `__control__` graph under THIS process's
        # own verified identity. Admit that identity into the narrow
        # control:system RBAC role idempotently, once per process, before
        # the scheduler can fire a single tick against it. Auto-at-boot by
        # design (see system_rbac_admission's module docstring, "Auto-
        # admission at boot"); degrades honestly — never crashes the
        # process and never claims success it did not confirm. Until an
        # operator seeds the missing NE-021 provisioner credential (or the
        # engine is unreachable), this logs the exact actionable cause once
        # per 30s backoff window and the scheduler keeps failing exactly as
        # visibly as it does today, but now with a diagnosis attached.
        try:
            from agent_utilities.security.system_rbac_admission import (
                ensure_system_principal_access,
            )

            ensure_system_principal_access(verified_session.actor.actor_id)
        except Exception as exc:  # noqa: BLE001 - must never block/crash boot
            logger.error(
                "system-principal control-graph admission not confirmed; "
                "the scheduler will keep failing until this is resolved "
                "(%s: %s)",
                type(exc).__name__,
                exc,
            )
    start_daemons = getattr(engine, "start_background_daemons", None)
    if not client_role and callable(start_daemons):
        start_daemons()

    def _bootstrap_engine() -> None:
        try:
            if (
                not client_role
                and engine
                and engine.backend
                and not getattr(engine.backend, "read_only", False)
            ):
                engine.start_task_workers()
            # The listener barrier already reconciled bundled skills.  Continue
            # broader discovery via the durable, fixed-priority plan without
            # blocking serving.
            _run_enabled_boot_hydration(
                engine,
                client_role=client_role,
                background_sync_enabled=config.knowledge_graph_sync_background,
                skip_skill_names=frozenset(BUNDLED_SKILLS),
            )
        except Exception as exc:
            logger.error("KG engine background bootstrap failed: %s", exc)

    try:
        _authorized_background_thread(
            verified_session,
            _bootstrap_engine,
            name="KGEngineBootstrap",
        ).start()
    except Exception as exc:
        # Packaged delegation is already ready. Optional workers, provider
        # discovery, and ontology federation remain retryable operational work.
        logger.error("GraphOS noncritical bootstrap launch failed: %s", exc)


def _build_server(
    bootstrap: bool = True,
    *,
    tool_profile: str | None = None,
    canonical_surface: bool = False,
):
    """Build the KG MCP server with all tools registered.

    Args:
        bootstrap: Whether this is a directly served process. The caller starts
            background engine bootstrap only after process identity is minted.
            The API gateway calls this with ``bootstrap=False`` (via
            :func:`ensure_tools_registered`) because it owns the engine/daemon
            lifecycle itself and only needs ``REGISTERED_TOOLS`` populated so the
            centralized REST handlers can dispatch.
        tool_profile: Explicit tool mode for deterministic catalog generation.
            ``None`` uses the configured runtime mode.
        canonical_surface: Register every condensed domain regardless of
            deployment toggles. This is reserved for catalog/gate construction;
            served processes continue to honor their configured toggles.
    """
    from agent_utilities.mcp.server_factory import create_mcp_server

    is_readonly = False

    def _check_readonly():
        if is_readonly:
            return json.dumps(
                {
                    "error": "Knowledge Graph is currently in READ-ONLY mode due to database lock contention. "
                    "Write operations and ingestion are disabled until the other process releases the lock."
                }
            )
        return None

    # In embedded mode (bootstrap=False, e.g. the API gateway populating
    # REGISTERED_TOOLS) do NOT parse the host process's argv — pass an empty
    # command line so the factory uses defaults instead of choking on unrelated
    # flags (pytest/uvicorn args) with SystemExit.
    args, mcp, middlewares = create_mcp_server(
        name="graph-os",
        version=__version__,
        instructions=(
            "Knowledge Graph MCP Server for agent-utilities. "
            "Provides access to the shared unified Knowledge Graph that powers "
            "the 5-pillar agent architecture (ORCH, KG, AHE, ECO, OS). "
            "Use kg_query for Cypher queries, kg_search for semantic search, "
            "kg_analyze for LLM-powered cross-reference analysis, "
            "and kg_ingest_* for adding data.\n\n"
            "graph-os is ALSO the MCP fleet gateway: its own KG/engine tools are "
            "always on, and it can load ANY other MCP server (declared in "
            "mcp_config.json) ON DEMAND. Hundreds more tools across dozens of "
            "servers exist but are NOT loaded yet — so when you need a capability "
            "you don't see, do NOT assume it's unavailable; use the fleet meta-tools:\n"
            "  • find_tools(query) — semantic search for the right tool by intent\n"
            "  • list_catalog() — browse every mountable server and its tools\n"
            "  • load_tools(tools=[...] or servers=[...]) — mount them; they become "
            "directly callable immediately (the tool list updates live)\n"
            "  • unload_tools(...) — retract tools to reclaim context\n"
            "  • multiplexer_status — health of mounted children\n"
            "Always discover (find_tools/list_catalog) before concluding a tool "
            "doesn't exist.\n\n"
            "EXCEPTION — the always-load set (MCP_ALWAYS_LOAD / "
            "MCP_ALWAYS_LOAD_TOOLS): a short operator-chosen list of core servers "
            "and individual tools is mounted EAGERLY on your first request, so it "
            "is already in your tool list and needs no find_tools/load_tools hop. "
            "Its absence is therefore meaningful — if an always-load tool is NOT "
            "listed, that server is genuinely degraded (eager mounting fails soft), "
            "not merely undiscovered; multiplexer_status says which and why. "
            "Everything OUTSIDE that set still follows the discover-first rule "
            "above. Inspect or change the set with "
            "graph_config(action='get'/'describe'/'set', key='MCP_ALWAYS_LOAD')."
        ),
        command_args=None if bootstrap else [],
        transport_choices=("stdio", "streamable-http"),
    )

    # Unauthenticated liveness + readiness for HTTP deployments (CONCEPT:AU-OS.deployment.liveness-vs-readiness-split).
    # Both dispatch into the ONE shared health-check core
    # (``observability.runtime_health.collect_health``) also used by the REST
    # gateway's ``/health``/``/health/ready`` and by ``graph_configure(action=
    # "health")`` — never a second implementation that can drift.
    #
    # ``/health`` is LIVENESS: it always answers 200 (this process itself is up
    # and answering requests) even when the body reports "unhealthy" — a
    # /health is a dependency-free, status-only liveness signal. The readiness
    # twin executes the truthful bounded collector on its reserved control lane,
    # but returns only ready/not_ready because both routes are intentionally
    # unauthenticated for kubelet. Detailed component data stays behind
    # graph_configure(action="health") and authenticated dashboard surfaces.
    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:  # noqa: ARG001
        return JSONResponse({"status": "ok"}, headers={"Cache-Control": "no-store"})

    @mcp.custom_route("/health/ready", methods=["GET"])
    async def readiness_check(request: Request) -> JSONResponse:  # noqa: ARG001
        from agent_utilities.observability.runtime_health import (
            collect_health_async,
            is_overall_healthy,
        )

        report = await collect_health_async()
        ready = is_overall_healthy(report)
        return JSONResponse(
            {"status": "ready" if ready else "not_ready"},
            status_code=200 if ready else 503,
            headers={"Cache-Control": "no-store"},
        )

    # ARD registry surface (CONCEPT:AU-ECO.mcp.eco-serves-two-ard/ECO-4.97) — the graph-os twin of the
    # gateway routes in server/routers/ard.py. This is the container the deploy
    # mechanic restarts, so it must answer the well-known + search paths too. Both
    # delegate into the same ecosystem.ard_* core to stay in lockstep with the gateway.
    @mcp.custom_route("/.well-known/ai-catalog.json", methods=["GET"])
    async def ard_ai_catalog(request: Request) -> JSONResponse:  # noqa: ARG001
        from agent_utilities.ecosystem.ard_registry import build_ai_catalog

        return JSONResponse(build_ai_catalog())

    @mcp.custom_route("/search", methods=["POST"])
    async def ard_search_route(request: Request) -> JSONResponse:
        from agent_utilities.ecosystem.ard_federation import ArdFederationRelay

        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 — malformed body ⇒ empty query, not a 500
            body = {}
        query = body.get("query") or {}
        text = str(query.get("text") or body.get("text") or "")
        types = ((query.get("filter") or {}).get("type")) or None
        page_size = int(body.get("pageSize") or 5)
        result = ArdFederationRelay().federated_search(
            text,
            types=types,
            page_size=page_size,
            mode=body.get("federationMode"),
            via=body.get("via") or [],
        )
        return JSONResponse(result)

    # ═══ Grouped action-routed tools ═══

    from agent_utilities.mcp.tools import (
        register_agent_execution_tools,
        register_analysis_tools,
        register_analyze_suite_tools,
        register_argument_tools,
        register_audit_tools,
        register_bus_tools,
        register_candidate_claim_tools,
        register_claim_tools,
        register_compliance_tools,
        register_config_tools,
        register_data_prep_tools,
        register_domain_ops_tools,
        register_durable_tools,
        register_engine_surface_tools,
        register_engine_tools,
        register_epistemic_tools,
        register_evolution_tools,
        register_governance_tools,
        register_graph_engineering_tools,
        register_incident_tools,
        register_job_tools,
        register_mcp_apps_tools,
        register_media_sidecar_tools,
        register_ontology_tools,
        register_ops_causal_tools,
        register_query_tools,
        register_reach_tools,
        register_rlm_tools,
        register_secret_tools,
        register_state_tools,
        register_workflow_tools,
        register_write_ingest_tools,
    )
    from agent_utilities.mcp.verbose_tools import register_tool_surface, tool_mode

    # graph-os is an action-routed wrapper over the API gateway's action core. The
    # condensed surface is the per-domain action tools (gated by `<DOMAIN>TOOL`); the
    # verbose surface is one 1:1 tool per gateway CRUD action, both dispatching through
    # the same `_execute_tool` core. register_tool_surface owns the MCP_TOOL_MODE
    # selection (intent default / condensed / verbose / both) for both.
    register_tool_surface(
        mcp,
        service="graph-os",
        registrars=[
            register_query_tools,
            register_write_ingest_tools,
            register_analysis_tools,
            register_agent_execution_tools,
            register_analyze_suite_tools,
            register_state_tools,
            register_ontology_tools,
            register_reach_tools,
            register_bus_tools,
            register_candidate_claim_tools,
            register_claim_tools,
            register_secret_tools,
            register_config_tools,
            register_data_prep_tools,
            register_engine_tools,
            register_engine_surface_tools,
            register_domain_ops_tools,
            register_evolution_tools,
            register_governance_tools,
            register_ops_causal_tools,
            register_graph_engineering_tools,
            register_audit_tools,
            register_epistemic_tools,
            register_incident_tools,
            register_job_tools,
            register_media_sidecar_tools,
            register_compliance_tools,
            register_rlm_tools,
            register_workflow_tools,
            register_argument_tools,
            register_durable_tools,
        ],
        verbose_register=register_graphos_verbose_tools,
        mode_override=tool_profile,
        force_condensed_registration=canonical_surface,
    )

    # CONCEPT:AU-ECO.ui.mcp-apps-host — the MCP Apps entry-point tool + its
    # ui:// resource (agent_utilities/mcp/tools/mcp_apps.py). Registered
    # directly, not through register_tool_surface: it has no `action` param
    # to condense/verbose-split (a single-purpose tool + a resource, not an
    # action-routed dispatcher), so it doesn't fit that harness's contract.
    register_mcp_apps_tools(mcp)

    # CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse (Seam 8, Phases 2-3) — the ADDITIONAL, small
    # "ask/find/write/act/manage/why" intent surface, selected by the default
    # MCP_TOOL_MODE=intent. Every granular tool the
    # condensed surface just registered stays reachable via load_tools (verbose_tools
    # tagged them GATED_TAG); the intent verbs dispatch through the SAME
    # REGISTERED_TOOLS/_execute_tool core.
    if (tool_profile or tool_mode()) == "intent":
        from agent_utilities.mcp.tools.intent_tools import register_intent_tools

        register_intent_tools(mcp)

    return args, mcp, middlewares


def register_graphos_verbose_tools(mcp) -> None:
    """Register graph-os's verbose 1:1 surface — one tool per gateway CRUD action.

    Each tool is a thin 1:1 alias that dispatches through the same ``_execute_tool``
    action core as the condensed ``graph_*`` tools and the REST gateway (no second
    implementation). Operations come from the generated
    ``_graphos_action_manifest.GRAPHOS_ACTIONS``; each is tagged ``{"verbose", <tool>}``
    so the visibility transform can slice them. CONCEPT:AU-ECO.mcp.tool-mode-standardization.
    """
    import json as _json

    from pydantic import Field

    from agent_utilities.mcp._graphos_action_manifest import GRAPHOS_ACTIONS
    from agent_utilities.mcp.verbose_tools import tool_mode

    # In ``both`` mode — and, since D-WS-1, in ``verbose`` mode too — the condensed
    # action tools are also registered (register_tool_surface now always registers
    # the condensed registrars so the dispatch core / REGISTERED_TOOLS is never left
    # empty; see verbose_tools.register_tool_surface). A single-op (action=None)
    # verbose tool shares the condensed tool's NAME, so skip it to avoid overwriting
    # the condensed tool's FastMCP component with a verbose-tagged duplicate whose
    # schema (bare ``params_json``) doesn't match what REGISTERED_TOOLS actually
    # dispatches for that name.
    skip_single_op = tool_mode() in ("both", "verbose")

    def _make(tool_name: str, action: str | None):
        # The low-level engine_<domain> tools (CONCEPT:AU-ECO.mcp.full-api-mcp-surface) are generic
        # action-routed dispatchers that take method kwargs as a single
        # ``params_json`` string (they cannot accept **kwargs — FastMCP rejects
        # VAR_KEYWORD). So forward params_json verbatim instead of spreading it.
        is_engine = tool_name.startswith("engine_")

        async def _verbose_op(
            params_json: str = Field(
                default="{}",
                description="JSON object of arguments for this operation.",
            ),
        ) -> Any:
            if is_engine:
                return await _execute_tool(
                    tool_name, action=action, params_json=params_json or "{}"
                )
            kwargs = _json.loads(params_json) if params_json else {}
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            if action is not None:
                kwargs.setdefault("action", action)
            return await _execute_tool(tool_name, **kwargs)

        return _verbose_op

    from agent_utilities.mcp.verbose_tools import GRANULAR_TAG

    for op in GRAPHOS_ACTIONS:
        if op["action"] is None and skip_single_op:
            continue
        fn = _make(op["tool"], op["action"])
        fn.__name__ = op["name"]
        fn.__doc__ = (
            f"graph-os {op['tool']} — action '{op['action']}' "
            "(1:1 over the action core)."
            if op["action"]
            else f"graph-os {op['tool']} (single operation)."
        )
        mcp.tool(name=op["name"], tags={"verbose", op["tool"], GRANULAR_TAG})(fn)


# ══════════════════════════════════════════════════════════════════


def ensure_tools_registered() -> None:
    """Idempotently register all ``graph_*`` tools into ``REGISTERED_TOOLS``.

    The centralized REST handlers (and the API gateway that mounts them via
    :func:`_mount_rest_routes`) dispatch through ``REGISTERED_TOOLS`` using
    :func:`_execute_tool`. Building the MCP server populates that dict as a side
    effect; we discard the throwaway FastMCP instance and skip the engine
    bootstrap (``bootstrap=False``) because the gateway owns the engine/daemon
    lifecycle and the handlers resolve the engine lazily via ``_get_engine()``.
    """
    if REGISTERED_TOOLS:
        return
    _build_server(bootstrap=False)


def _mount_rest_routes(app, prefix: str = "") -> None:
    """Mount the full Knowledge Graph REST surface onto ``app``.

    ``app`` is any Starlette/FastAPI application exposing ``add_route``. Every
    path is prepended with ``prefix`` (the API gateway mounts these under
    ``/api``). Handlers dispatch through ``REGISTERED_TOOLS`` — call
    :func:`ensure_tools_registered` first.

    This is the single source of truth for the KG REST route table. The
    ``graph-os`` MCP server itself is now a thin FastMCP wrapper (MCP tools
    only); the REST API is served centrally by ``agent_utilities.gateway`` so the
    table never drifts between the two.
    """
    from agent_utilities.core.sessions import (
        cancel_goal,
        cancel_session_run,
        create_goal,
        delete_session,
        get_all_sessions,
        get_goal_iterations,
        get_session_details,
        list_goals,
        submit_session_reply,
    )
    from agent_utilities.gateway.schemas.graph_analyze import (
        SearchDiscoverRequest,
        SearchQueryTopKRequest,
        SearchTextResponse,
    )

    def route(path: str, handler, methods: list[str]) -> None:
        app.add_route(prefix + path, handler, methods=methods)

    def route_typed(
        path: str,
        handler,
        methods: list[str],
        *,
        response_model: type,
        summary: str,
        description: str,
        request_model: type | Any,
        request_body_required: bool = True,
    ) -> None:
        """Like ``route()`` but mounted via FastAPI's
        ``add_api_route(..., response_model=...)`` when ``app`` supports it
        (every production caller — see ``build_agent_app``) so the route is
        visible to ``app.openapi()``. ``scripts/check_openapi_coverage.py``
        measures exactly this gap for every OTHER route in this file, which
        still uses the raw ``route()``/``add_route`` helper above; this is
        the first route in ``_mount_rest_routes`` to close it. Falls back to
        the plain Starlette ``add_route`` for a bare-Starlette/test-double
        ``app`` that only implements ``add_route`` (mirrors the existing
        ``add_api_route``-vs-``add_route`` guard in
        ``agent_utilities.gateway.graph_api.register_graph_routes``'s
        ``/metrics`` mount) — undocumented in that fallback case, but still
        callable.

        ``request_model`` may be a ``BaseModel`` subclass or a typing
        construct (e.g. an ``Annotated[Union[...], Field(discriminator=...)]``
        alias) — the handler itself parses the body manually (so a
        malformed/unrecognized ``action`` is a controlled 400, not FastAPI's
        default 422), so this only feeds ``openapi_extra`` a real JSON Schema
        for documentation; it does not change request parsing.
        """
        if not hasattr(app, "add_api_route"):
            app.add_route(prefix + path, handler, methods=methods)
            return
        if hasattr(request_model, "model_json_schema"):
            body_schema = request_model.model_json_schema()
        else:
            body_schema = TypeAdapter(request_model).json_schema()
        app.add_api_route(
            prefix + path,
            handler,
            methods=methods,
            response_model=response_model,
            summary=summary,
            description=description,
            openapi_extra={
                "requestBody": {
                    "required": request_body_required,
                    "content": {"application/json": {"schema": body_schema}},
                }
            },
        )

    # ── Sessions & goals (durable Starlette handlers in core.sessions) ──
    route("/sessions", get_all_sessions, ["GET"])
    route("/sessions/{session_id}", get_session_details, ["GET"])
    route("/sessions/{session_id}", delete_session, ["DELETE"])
    route("/sessions/{session_id}/reply", submit_session_reply, ["POST"])
    route("/sessions/{session_id}/cancel", cancel_session_run, ["POST"])
    route("/goals", create_goal, ["POST"])
    route("/goals", list_goals, ["GET"])
    route("/goals/{goal_id}/iterations", get_goal_iterations, ["GET"])
    route("/goals/{goal_id}/cancel", cancel_goal, ["POST"])

    # ── Tools introspection / toggles ──
    route("/tools", get_tools_endpoint, ["GET"])
    route("/tools/toggle", toggle_tool_endpoint, ["POST"])

    # ── Bilateral graph execution (action-routed) ──
    route("/graph/query", graph_query_endpoint, ["POST"])
    route("/graph/search", graph_search_endpoint, ["POST"])
    # Collapsed, typed graph_write dispatch (CONSOLIDATION: see the
    # `GraphWriteAction` discriminated union above graph_write_endpoint's
    # definition) — the first FastAPI-documented route in this file.
    route_typed(
        "/graph/write",
        graph_write_endpoint,
        ["POST"],
        response_model=GraphToolResponse,
        summary="Write a node/edge or run another graph_write action",
        description=(
            "Collapsed, action-routed graph_write endpoint. Validates the "
            "body against a discriminated union on 'action' covering "
            "add_node, add_edge, delete_edge, bulk_ingest, log_chat, "
            "register_execution (formerly separate granular routes under "
            "/graph/write/{node,edge,bulk,chat,execution}, now removed), "
            "plus every other graph_write action (delete_node, "
            "register_external_graph, compare_and_set, store_memory, "
            "recall_memory, recall_media, submit_sdd, check_loop). See the "
            "GraphWriteAction union's member models for the exact per-action "
            "request shape."
        ),
        request_model=GraphWriteAction,
    )
    route_typed(
        "/graph/write",
        graph_write_delete_edge_endpoint,
        ["DELETE"],
        response_model=GraphToolResponse,
        summary="Delete an edge (graph_write action=delete_edge)",
        description=(
            "Deletes one edge identified by source_id/target_id/rel_type. "
            "Equivalent to POST /graph/write with action='delete_edge'; "
            "kept as a dedicated DELETE verb on the same collapsed path so "
            "a REST-verb-first caller does not lose the capability the "
            "removed DELETE /graph/write/edge granular route had."
        ),
        request_model=GraphWriteEdgeDeleteRequest,
    )
    route("/graph/ingest", graph_ingest_endpoint, ["POST"])
    route("/graph/analyze", graph_analyze_endpoint, ["POST"])
    route("/graph/code", graph_code_endpoint, ["POST"])
    route("/graph/research", graph_research_endpoint, ["POST"])
    route("/graph/evaluate", graph_evaluate_endpoint, ["POST"])
    route("/graph/explain", graph_explain_endpoint, ["POST"])
    route("/graph/observe", graph_observe_endpoint, ["POST"])
    route("/graph/orchestrate", graph_orchestrate_endpoint, ["POST"])
    route("/graph/configure", graph_configure_endpoint, ["POST"])

    # ── Granular query ──
    route("/graph/query/federated", graph_query_federated_endpoint, ["POST"])

    # ── Granular search ──
    # These mode-fixed adapters share one implementation, while each remains a
    # separately documented route with the same request/response schemas that
    # describe its existing wire behavior.
    for _path, _handler, _summary, _description, _request_model in (
        (
            "/graph/search/concept",
            graph_search_concept_endpoint,
            "Search concepts",
            "Search the knowledge graph using concept retrieval.",
            SearchQueryTopKRequest,
        ),
        (
            "/graph/search/analogy",
            graph_search_analogy_endpoint,
            "Search by analogy",
            "Search the knowledge graph for analogous concepts.",
            SearchQueryTopKRequest,
        ),
        (
            "/graph/search/memory",
            graph_search_memory_endpoint,
            "Search memories",
            "Search retained graph memories.",
            SearchQueryTopKRequest,
        ),
        (
            "/graph/search/discover",
            graph_search_discover_endpoint,
            "Discover graph capabilities",
            "Discover ingested graph capabilities matching a query.",
            SearchDiscoverRequest,
        ),
        (
            "/graph/search/dci",
            graph_search_dci_endpoint,
            "Search with DCI",
            "Search the knowledge graph using DCI retrieval.",
            SearchQueryTopKRequest,
        ),
    ):
        route_typed(
            _path,
            _handler,
            ["POST"],
            response_model=SearchTextResponse,
            summary=_summary,
            description=_description,
            request_model=_request_model,
            request_body_required=False,
        )

    # ── Granular write (out of this consolidation's scope — see kg_server.py's
    # collapsed-write comment block above graph_write_endpoint) ──
    route("/graph/write/node/{node_id}", graph_write_delete_node_endpoint, ["DELETE"])
    route("/graph/write/external", graph_write_external_endpoint, ["POST"])
    route("/graph/write/memory", graph_write_memory_endpoint, ["POST"])
    route("/graph/write/memory/recall", graph_write_memory_recall_endpoint, ["POST"])
    # CONCEPT:AU-KG.ontology.federation-runtime — federation: explicit twin for ontology package-sync.
    route(
        "/graph/ontology/sync-packages",
        graph_ontology_sync_packages_endpoint,
        ["POST"],
    )
    # CONCEPT:AU-KG.ontology.stardog-catalog-overwrite / stardog-catalog-import — Stardog catalog twins.
    route(
        "/graph/ontology/publish-stardog",
        graph_ontology_publish_stardog_endpoint,
        ["POST"],
    )
    route(
        "/graph/ontology/import-stardog",
        graph_ontology_import_stardog_endpoint,
        ["POST"],
    )
    route("/graph/write/sdd", graph_write_sdd_endpoint, ["POST"])

    # ── Granular ingest ──
    route("/graph/ingest/submit", graph_ingest_submit_endpoint, ["POST"])
    route("/graph/ingest/corpus", graph_ingest_corpus_endpoint, ["POST"])
    route("/graph/ingest/jobs", graph_ingest_jobs_endpoint, ["GET"])
    route("/connector/sources", connector_sources_endpoint, ["GET"])
    route("/connector/run", connector_run_endpoint, ["POST"])
    route("/graph/ingest/job/{job_id}", graph_ingest_job_status_endpoint, ["GET"])
    route(
        "/graph/ingest/rebuild-indexes", graph_ingest_rebuild_indexes_endpoint, ["POST"]
    )
    route("/graph/ingest/observe", graph_ingest_observe_endpoint, ["POST"])
    route("/graph/ingest/materialize", graph_ingest_materialize_endpoint, ["POST"])
    route(
        "/graph/ingest/materialize-source",
        graph_ingest_materialize_source_endpoint,
        ["POST"],
    )
    route("/graph/ingest/sync", graph_ingest_sync_endpoint, ["POST"])
    route("/graph/ingest/reflect", graph_ingest_reflect_endpoint, ["POST"])
    route("/graph/ingest/agent-toolkit", graph_ingest_agent_toolkit_endpoint, ["POST"])
    route(
        "/graph/ingest/knowledge-pack", graph_ingest_knowledge_pack_endpoint, ["POST"]
    )

    # ── Granular analyze ──
    route("/graph/analyze/synthesize", graph_analyze_synthesize_endpoint, ["POST"])
    route(
        "/graph/analyze/process-writeback",
        graph_analyze_process_writeback_endpoint,
        ["POST"],
    )
    route("/graph/analyze/deep-extract", graph_analyze_deep_extract_endpoint, ["POST"])
    route(
        "/graph/analyze/background-research",
        graph_analyze_background_research_endpoint,
        ["POST"],
    )
    route(
        "/graph/analyze/relevance-sweep",
        graph_analyze_relevance_sweep_endpoint,
        ["POST"],
    )
    route("/graph/analyze/blast-radius", graph_analyze_blast_radius_endpoint, ["GET"])
    route("/graph/analyze/inspect", graph_analyze_inspect_endpoint, ["GET"])
    route("/graph/analyze/call-graph", graph_analyze_call_graph_endpoint, ["GET"])
    route("/graph/analyze/similar-code", graph_analyze_similar_code_endpoint, ["GET"])
    route("/graph/analyze/routes", graph_analyze_routes_endpoint, ["GET"])
    route(
        "/graph/analyze/change-coupling",
        graph_analyze_change_coupling_endpoint,
        ["POST"],
    )
    route(
        "/graph/analyze/code-evolution",
        graph_analyze_code_evolution_endpoint,
        ["POST"],
    )
    route("/graph/analyze/adr", graph_analyze_adr_endpoint, ["POST"])
    route("/graph/analyze/harness-gate", graph_analyze_harness_gate_endpoint, ["POST"])
    route("/graph/analyze/code-context", graph_analyze_code_context_endpoint, ["POST"])
    route("/graph/analyze/code-metrics", graph_analyze_code_metrics_endpoint, ["GET"])
    route("/graph/analyze/arch-report", graph_analyze_arch_report_endpoint, ["GET"])
    route("/graph/analyze/explain", graph_analyze_explain_endpoint, ["POST"])
    route(
        "/graph/analyze/cross-repo-usages",
        graph_analyze_cross_repo_usages_endpoint,
        ["GET"],
    )
    route("/graph/analyze/context", graph_analyze_context_endpoint, ["POST"])
    route(
        "/graph/analyze/evaluate-alpha", graph_analyze_evaluate_alpha_endpoint, ["POST"]
    )
    route("/graph/analyze/evaluate", graph_analyze_evaluate_endpoint, ["POST"])
    route("/graph/analyze/evolve-model", graph_analyze_evolve_model_endpoint, ["POST"])
    route("/graph/analyze/forecast", graph_analyze_forecast_endpoint, ["POST"])
    route("/graph/analyze/causal", graph_analyze_causal_endpoint, ["POST"])
    route("/graph/analyze/invariant", graph_analyze_invariant_endpoint, ["POST"])
    route(
        "/graph/analyze/security-scan", graph_analyze_security_scan_endpoint, ["POST"]
    )

    # ── Granular configure ──
    route("/graph/configure/secret", graph_configure_secret_endpoint, ["POST"])
    route("/graph/configure/vault-sync", graph_configure_vault_sync_endpoint, ["POST"])
    route(
        "/graph/configure/register-mcp", graph_configure_register_mcp_endpoint, ["POST"]
    )
    route(
        "/graph/configure/install-hooks",
        graph_configure_install_hooks_endpoint,
        ["POST"],
    )
    route(
        "/graph/configure/uninstall-hooks",
        graph_configure_uninstall_hooks_endpoint,
        ["POST"],
    )
    route("/graph/configure/doctor", graph_configure_doctor_endpoint, ["POST"])

    # ── Collapsed action-routed twins (full MCP⇄REST parity) ──
    # The core graph_* tools above already have bespoke endpoints; every
    # other MCP tool in ACTION_TOOL_ROUTES (context, feedback, hydrate, sessions,
    # goals, document_process, source_connector, ontology_*, object_*) is served
    # by the generic factory so the REST surface reaches everything MCP can.
    _bespoke_action_tools = {
        "graph_query",
        "graph_search",
        "graph_write",
        "graph_ingest",
        "graph_analyze",
        "graph_orchestrate",
        "graph_configure",
        # graph_mine has a bespoke endpoint (natural mining body → the same
        # _execute_tool core) mounted below (CONCEPT:EG-KG.mining.frequent-itemset-mining).
        "graph_mine",
        # graph_learn likewise has bespoke natural-body twins (CONCEPT:EG-KG.graphlearn.link-predictor).
        "graph_learn",
        # graph_mine_deep likewise has bespoke natural-body twins (CONCEPT:AU-KG.mining.dsm-forecast-delegation).
        "graph_mine_deep",
    }
    for _tool, _path in ACTION_TOOL_ROUTES.items():
        if _tool in _bespoke_action_tools:
            continue
        route(_path, _make_tool_endpoint(_tool), ["POST"])

    # Data-mining REST twins (CONCEPT:EG-KG.mining.frequent-itemset-mining) — one natural-body
    # /api/mining/<action> endpoint per graph_mine action (the full 18-action surface —
    # see MINING_ACTIONS), each dispatching the SAME graph_mine _execute_tool core
    # (surface parity).
    if "graph_mine" in ACTION_TOOL_ROUTES:
        for _mine_action in MINING_ACTIONS:
            route(
                f"/mining/{_mine_action}", _make_mining_endpoint(_mine_action), ["POST"]
            )

    # Graph-learning REST twins (CONCEPT:EG-KG.graphlearn.link-predictor) — one
    # natural-body /api/graphlearn/<action> endpoint per graph_learn action (fit|predict),
    # each dispatching the SAME graph_learn _execute_tool core (surface parity).
    if "graph_learn" in ACTION_TOOL_ROUTES:
        for _gl_action in GRAPHLEARN_ACTIONS:
            route(
                f"/graphlearn/{_gl_action}",
                _make_graphlearn_endpoint(_gl_action),
                ["POST"],
            )

    # Deep-mining delegation REST twins (CONCEPT:AU-KG.mining.dsm-forecast-delegation — Phase 6) — one
    # natural-body /api/mining/deep/<action> endpoint per graph_mine_deep action
    # (deep_forecast|deep_classify|autoencoder_anomaly|xgboost|embed), each
    # dispatching the SAME graph_mine_deep _execute_tool core (surface parity).
    if "graph_mine_deep" in ACTION_TOOL_ROUTES:
        for _deep_action in DEEP_MINING_ACTIONS:
            route(
                f"/mining/deep/{_deep_action}",
                _make_mining_deep_endpoint(_deep_action),
                ["POST"],
            )


_FLEET_EMBED_MODEL: Any = None


def _fleet_embed_fn():
    """Return a sync batch-embed callable ``(texts) -> list[vector]`` for find_tools'
    semantic tool ranking, backed by graph-os's own embedding model (built lazily +
    cached on first use). The model is remote (vLLM) and sync, so the fleet loader calls
    this OFF-THREAD. Any construction/inference failure is swallowed by the caller, which
    then degrades to token-overlap ranking — so this never blocks fleet loading."""

    def _embed(texts):
        global _FLEET_EMBED_MODEL
        if _FLEET_EMBED_MODEL is None:
            from agent_utilities.core.embedding_utilities import create_embedding_model

            _FLEET_EMBED_MODEL = create_embedding_model()
        model = _FLEET_EMBED_MODEL
        batch = getattr(model, "get_text_embedding_batch", None)
        if callable(batch):
            return batch(list(texts))
        return [model.get_text_embedding(t) for t in texts]

    return _embed


def _configure_graphos_otel() -> None:
    """Activate the canonical metadata-only OTLP pipeline when configured."""

    if not setting("ENABLE_OTEL", False):
        return
    try:
        from agent_utilities.observability.custom_observability import setup_otel

        setup_otel(service_name="graph-os")
    except Exception as exc:  # noqa: BLE001 - observability cannot prevent serving
        logger.warning(
            "GraphOS OTLP setup failed; trace export is disabled (exception_type=%s)",
            type(exc).__name__,
        )


def _configure_telemetry_engine_otel() -> None:
    """Eagerly start the standard-env-var OTLP trace pipeline (X2).

    CONCEPT:AU-OS.observability.telemetry-observability — independent of
    ``_configure_graphos_otel``'s ``ENABLE_OTEL``-gated Logfire/Langfuse
    pipeline: :class:`~agent_utilities.observability.TelemetryEngine` self-gates
    purely on the standard ``OTEL_EXPORTER_OTLP_ENDPOINT``/``OTEL_SERVICE_NAME``/
    ``OTEL_TRACES_EXPORTER`` vars (falling back to ``EPISTEMIC_GRAPH_OBS_ADDR``),
    so no new env var is introduced here. This is the ONE process-bootstrap
    call site that activates it for the graph-os MCP server — never per-request.
    """
    try:
        from agent_utilities.observability import get_telemetry_engine

        configured = get_telemetry_engine().is_otel_configured()
        logger.info(
            "GraphOS OTLP trace export (standard env vars): %s",
            "enabled" if configured else "disabled",
        )
    except Exception as exc:  # noqa: BLE001 - observability cannot prevent serving
        logger.warning(
            "GraphOS TelemetryEngine OTel setup failed (exception_type=%s)",
            type(exc).__name__,
        )


def _preflight_mcp_sdk_floor() -> None:
    """Fail startup loudly when the installed MCP SDK is below the declared floor.

    CONCEPT:AU-ECO.mcp.protocol-compat-bridge — closes D-OB-18.

    The category defect this exists for is that source-vs-installed divergence was
    INVISIBLE: graph-os source targeting fastmcp 4 / mcp 2 ran for weeks on an image
    that shipped fastmcp 3.4.5 / mcp 1.29.0, and the only symptom was a single ERROR
    log line as `attach_fleet_loader` lost every fleet meta-tool to
    ``ImportError: cannot import name 'MCPError'``. A rebuilt image with no assertion
    just resets that clock, so the assertion runs here, at startup, as well as at
    image-build time.

    Enforcement is a hook, not a hardcoded policy: ``MCP_SDK_FLOOR_ENFORCE`` selects
    ``error`` (default — refuse to start, because a graph-os that silently loses its
    fleet surface is worse than one that will not come up) or ``warn`` (log and
    continue, for an operator who is knowingly running a mismatched pair during a
    migration).
    """
    from agent_utilities.mcp.protocol_compat import check_mcp_sdk_floor

    result = check_mcp_sdk_floor()
    if result["ok"] is True:
        logger.info("MCP SDK floor OK: %s", result["detail"])
        return
    if result["ok"] is None:
        logger.warning("MCP SDK floor check skipped: %s", result["detail"])
        return

    mode = str(setting("MCP_SDK_FLOOR_ENFORCE", "error") or "error").strip().lower()
    message = (
        f"installed MCP SDK does not satisfy the declared [mcp] floor: {result['detail']}. "
        "The runtime image and this source tree have diverged — rebuild the image "
        "(docker/graphos-unified.Dockerfile) so its dependency closure matches the "
        "source it serves. Set MCP_SDK_FLOOR_ENFORCE=warn to start anyway."
    )
    if mode == "warn":
        logger.error("graph-os starting with a mismatched MCP SDK: %s", message)
        return
    raise RuntimeError(message)


async def _write_refreshed_fleet_catalog(catalog, configs, bindings):
    """Bridge the served GraphOS mux into source-sync's canonical writer."""
    from agent_utilities.knowledge_graph.core.source_sync import (
        write_fleet_catalog_snapshot,
    )

    return await asyncio.to_thread(
        write_fleet_catalog_snapshot,
        _get_engine(),
        catalog,
        configs=configs,
        discovery_bindings=bindings,
    )


def mcp_server() -> None:
    """``graph-os`` MCP server entry point (registered as console_scripts).

    Thin FastMCP wrapper following the standard ``mcp_server.py`` template: it
    serves ONLY the MCP tool surface, over ``stdio`` or ``streamable-http``,
    selected by the standard ``--transport/--host/--port`` args
    from :func:`create_mcp_server`. The REST API (``/graph/*``, ``/sessions``,
    ``/goals``, ``/tools``) is centralized in the API gateway
    (``agent_utilities.gateway``) — see :func:`_mount_rest_routes`.
    """
    global _PROCESS_SESSION
    from agent_utilities.core.config import load_config

    load_config()  # resolve settings through the one shared XDG config.json
    _preflight_mcp_sdk_floor()
    _configure_graphos_otel()
    _configure_telemetry_engine_otel()
    os.environ["IS_KG_SERVER"] = "true"
    args, mcp, middlewares = _build_server()

    # Apply the middleware stack assembled by the factory.
    for middleware in middlewares:
        mcp.add_middleware(middleware)

    # Fold in the MCP fleet-loader (retires the standalone mcp-multiplexer): graph-os's
    # own tools stay always-on; this adds find_tools/load_tools/... so the SAME server
    # reaches the rest of the MCP fleet on demand. Attached AFTER the factory middlewares
    # so per-session tool visibility runs with identity/auth already applied. Only for a
    # directly-served process — the embedded API-gateway build owns no serving loop.
    # The six meta-tools this attaches (find_tools/list_catalog/load_tools/
    # unload_tools/refresh_mcp_server/multiplexer_status) plus the
    # session-visibility middleware are
    # MODE-INDEPENDENT infrastructure — they are the only way to reach anything
    # the active MCP_TOOL_MODE holds back, so they must be present under intent,
    # condensed, verbose AND both. A failure here is therefore NOT survivable:
    # the previous `except Exception: logger.error(...)` downgraded it to a log
    # line and served a silently wrong surface (an SDK-rename ImportError in
    # child_resilience left graph-os exposing 118 ungated tools with no
    # load_tools at all). Fail loud, preserving __cause__.
    # CONCEPT:AU-ECO.mcp.fleet-meta-tools-always-on
    try:
        from agent_utilities.mcp.multiplexer import attach_fleet_loader

        # Inject graph-os's own embedding model so find_tools ranks fleet tools by
        # query↔description MEANING (semantic), not just literal token overlap.
        fleet_mux = attach_fleet_loader(
            mcp,
            embed_fn=_fleet_embed_fn(),
            authority_scope=verified_tool_session_scope,
            catalog_writer=_write_refreshed_fleet_catalog,
        )
    except Exception as exc:
        raise RuntimeError(
            "graph-os fleet loader attach failed: the fleet meta-tools "
            "(find_tools/list_catalog/load_tools/unload_tools/"
            "refresh_mcp_server/multiplexer_status) "
            "and the session-visibility middleware could not be registered, so the "
            "served tool surface would be wrong under every MCP_TOOL_MODE."
        ) from exc

    transport = getattr(args, "transport", "stdio")
    host = getattr(args, "host", "127.0.0.1")
    port = int(getattr(args, "port", 8000))

    bootstrap_session = _mint_process_session(transport)
    _PROCESS_SESSION = bootstrap_session if transport == "stdio" else None
    _start_process_authority_supervisor(bootstrap_session)
    # Readiness probes the live fleet/goal authority, which is a real graph read
    # and therefore needs a bound session. `_PROCESS_SESSION` is deliberately
    # None on network transports (it is a stdio fallback, and must not become a
    # way for a request path to pick up identity it never authenticated), so
    # readiness gets its own narrowly-scoped handle on the process authority.
    # Without this the probe measured its own missing identity instead of the
    # authority, reported the goal store `unavailable`, and held /health/ready
    # at 503 forever on every served deployment.
    _set_readiness_authority(bootstrap_session)

    co_service_supervisor = None
    try:
        logger.info("Starting graph-os MCP server (transport=%s)", transport)

        from agent_utilities.mcp.server_factory import mcp_network_run_kwargs
        from agent_utilities.security.request_identity import (
            apply_served_security_profile,
        )

        # Network transports serve many clients at once: enforce server-validated
        # identity + tenant scoping, or fail loud (CONCEPT:AU-OS.identity.authenticated-identity-enforcement). No-op for stdio.
        apply_served_security_profile(
            transport,
            transport_auth_configured=(
                str(getattr(args, "auth_type", "none") or "none").lower() != "none"
            ),
        )

        # Stdout purity on the stdio transport needs no call here: it is owned
        # fd-level by the MCP SDK's own ``stdio_server()`` for the scope of the
        # later stdio-serve call below (see the "Stdio JSON-RPC purity" note in
        # server_factory.py) — that covers every co-service thread started
        # below too, since they share this process's file-descriptor table for
        # as long as serving blocks. The residual window before that call
        # claims fd 1 (engine bootstrap, co-service startup, this function
        # itself) is covered by the static "no print() in the served package"
        # gate (``scripts/check_no_stdout_writes.py``), not a runtime patch.
        # No-op for network transports either way (they don't own stdout as a
        # protocol channel).

        # Bind the minted process session (+ its verified actor) as ambient
        # authority before engine bootstrap. An explicit client role remains a
        # hard serving-plane boundary; this entrypoint never promotes itself to
        # the host that owns maintenance, workers, or autonomous loops.
        from agent_utilities.core.config import config
        from agent_utilities.knowledge_graph.core.session import use_session
        from agent_utilities.mcp.co_service_supervisor import start_co_services
        from agent_utilities.security.brain_context import use_actor

        with use_actor(bootstrap_session.actor), use_session(bootstrap_session):
            _start_engine_bootstrap(bootstrap_session)

            # Self-composing co-services, phase 2: messaging now that a real engine
            # exists. Credentials keep outbound sending available, but the explicit
            # MESSAGING_INTAKE_ENABLED deployment intent (false by default) is the
            # only way this request container may enter the shared native lease
            # boundary. When ENABLE_WEB_UI is true, the packaged agent-webui is
            # started in-process by this same supervisor as a separately bound,
            # independently restartable co-service.
            co_service_supervisor = start_co_services(
                bootstrap_session,
                _get_engine(),
                messaging_intake_enabled=config.messaging_intake_enabled,
            )

        if transport == "stdio":
            mcp.run(transport="stdio")
        elif transport == "streamable-http":
            mcp.run(
                transport="streamable-http",
                host=host,
                port=port,
                **mcp_network_run_kwargs(args),
            )
        else:
            raise ValueError("graph-os transport must be 'stdio' or 'streamable-http'")
    finally:
        if co_service_supervisor is not None:
            co_service_supervisor.stop_all()
        _PROCESS_SESSION = None
        _stop_process_authority_supervisor()
        # Best-effort teardown of any lazily-mounted fleet children.
        if fleet_mux is not None:
            try:
                asyncio.run(fleet_mux.aclose())
            except Exception as exc:  # noqa: BLE001 — best-effort teardown of a lazily-mounted fleet child at process exit
                logger.debug("fleet loader close failed: %s", type(exc).__name__)


if __name__ == "__main__":
    mcp_server()
