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
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agent_utilities._version import __version__
from agent_utilities.core.config import setting

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

_CALLER_AUTHORITY_FIELDS = frozenset({"_actor", "_roles", "_tenant"})


def _reject_caller_authority(kwargs: dict[str, Any]) -> None:
    """Reject legacy tool fields that attempted to self-assert authority."""
    if _CALLER_AUTHORITY_FIELDS.intersection(kwargs):
        raise PermissionError("Caller-supplied graph authority is forbidden")


@contextlib.contextmanager
def verified_tool_session_scope():
    """Scope one served tool call to middleware/process-minted authority.

    Identity, tenant, audience, and policy revision are validated before tool
    dispatch. The error surface deliberately omits principal, tenant, token,
    endpoint, and policy values.
    """
    from ..knowledge_graph.core.session import current_session, use_session
    from ..security.brain_context import (
        IdentityRequiredError,
        current_actor,
        use_actor,
    )

    ambient = current_session()
    session = ambient or _PROCESS_SESSION
    if session is None:
        raise PermissionError("Verified GraphSession required")
    try:
        session.engine_verified_context()
    except PermissionError:
        raise PermissionError("Verified GraphSession authority is incomplete") from None

    try:
        actor = current_actor()
    except IdentityRequiredError:
        actor = None
    if actor is not None and actor.authenticated and actor != session.actor:
        raise PermissionError("Verified actor and GraphSession authority differ")

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
            await _ensure_process_authority_current()
            with verified_tool_session_scope():
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


def get_existing_disabled(engine, node_id: str) -> bool:
    try:
        # 1. Try in-memory graph cache first
        if hasattr(engine, "graph_compute") and hasattr(engine.graph_compute, "graph"):
            if node_id in engine.graph_compute.graph:
                return engine.graph_compute.graph.nodes[node_id].get("disabled", False)
        # 2. Try Cypher match as a fallback
        res = engine.query_cypher(
            "MATCH (n) WHERE n.id = $node_id RETURN n.disabled AS disabled",
            {"node_id": node_id},
        )
        if res and isinstance(res, list) and len(res) > 0:
            return bool(res[0].get("disabled", False))
    except Exception:
        pass
    return False


def safe_json_load(s: Any) -> Any:
    if hasattr(s, "model_dump"):
        return s.model_dump()
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:
            pass
    return s


def _parse_skill_md(path: Any) -> dict[str, Any]:
    """Parse YAML frontmatter from a SKILL.md file."""
    import re
    from pathlib import Path

    import yaml

    path_obj = Path(path)
    try:
        content = path_obj.read_text(encoding="utf-8", errors="ignore")
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        metadata: dict[str, Any] = {}
        if match:
            try:
                metadata = yaml.safe_load(match.group(1)) or {}
            except Exception:
                for line in match.group(1).splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        metadata[k.strip()] = v.strip()

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
    except Exception as e:
        logger.error("Failed to parse SKILL.md (%s)", type(e).__name__)
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


def get_toggle_state(engine, item_type: str, item_id: str) -> bool:
    """Check if an item is enabled or disabled in the KG."""
    if not engine:
        return True
    pref_id = f"preference:toggle:{item_type}:{item_id}"
    try:
        res = engine.query_cypher(
            "MATCH (p:Preference) WHERE p.id = $pref_id RETURN p.value as value",
            {"pref_id": pref_id},
        )
        if res and len(res) > 0:
            return res[0].get("value") == "enabled"
    except Exception as exc:
        logger.error(
            "Failed to query toggle state (exception_type=%s)",
            type(exc).__name__,
        )
    return True  # Enabled by default


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
        node_id = ""
        if item_type == "mcp_server":
            node_id = f"mcp_server_{item_id}"
        elif item_type == "builtin_tool":
            node_id = f"native_tool_{item_id}"
        elif item_type == "skill":
            node_id = f"skill_{item_id}"
        elif item_type == "skill_workflow":
            node_id = f"skill_workflow_{item_id}"
        elif item_type == "skill_graph":
            node_id = f"skill_graph_{item_id}"

        if node_id:
            engine.query_cypher(
                "MATCH (n) WHERE n.id = $node_id SET n.disabled = $disabled",
                {"node_id": node_id, "disabled": not enabled},
            )
            # Also update in-memory graph cache if active
            if (
                hasattr(engine, "graph_compute")
                and engine.graph_compute
                and hasattr(engine.graph_compute, "graph")
            ):
                if node_id in engine.graph_compute.graph.nodes:
                    engine.graph_compute.graph.nodes[node_id]["disabled"] = not enabled
    except Exception as exc:
        logger.error(
            "Failed to save toggle state (exception_type=%s)",
            type(exc).__name__,
        )


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


async def get_tools_endpoint(request: Request) -> JSONResponse:
    """Retrieve all MCP tools, built-in tools, skills, skill graphs, and workflows categorized."""
    import json
    from pathlib import Path

    from ..knowledge_graph.core.session import resolve_session

    resolve_session(required_scope="kg:read")

    engine = _get_engine()

    # 1. MCP Tools
    mcp_tools = []
    # Try different config paths
    config_paths = [
        Path.home() / ".config" / "agent-utilities" / "mcp_config.json",
        Path.home() / ".config" / "agent-utilities" / "config.json",
        Path("workspace/mcp_config.json"),
    ]
    config_path = None
    for cp in config_paths:
        if cp.exists():
            config_path = cp
            break

    if config_path:
        try:
            mcp_data = json.loads(config_path.read_text(encoding="utf-8"))
            mcp_servers = mcp_data.get("mcpServers", {})
            if (
                not mcp_servers
                and "mcp_config" in mcp_data
                and isinstance(mcp_data["mcp_config"], dict)
            ):
                mcp_servers = mcp_data["mcp_config"].get("mcpServers", {})
            for name, cfg in mcp_servers.items():
                mcp_enabled = get_toggle_state(engine, "mcp_server", name)
                if cfg.get("disabled", False):
                    mcp_enabled = False
                mcp_tools.append(
                    {
                        "name": name,
                        "type": "MCP Server",
                        "launch_mode": "subprocess" if cfg.get("command") else "remote",
                        "command": "[configured]" if cfg.get("command") else "",
                        "args": ["[configured]"] if cfg.get("args") else [],
                        "status": "active" if mcp_enabled else "disabled",
                        "enabled": mcp_enabled,
                    }
                )
        except Exception as e:
            logger.error("Failed to parse MCP config (%s)", type(e).__name__)

    # 2. Built-in Agent Tools
    builtin_tools = []
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    if tools_dir.exists() and tools_dir.is_dir():
        for f in tools_dir.glob("*.py"):
            if f.name.startswith("_"):
                continue
            builtin_enabled = get_toggle_state(engine, "builtin_tool", f.stem)
            builtin_tools.append(
                {
                    "name": f.stem,
                    "type": "Built-in Tool",
                    "file_path": f"tool://{f.stem}",
                    "status": "enabled" if builtin_enabled else "disabled",
                    "enabled": builtin_enabled,
                }
            )

    # 3. Skills & Workflows
    skills = []
    workflows = []
    workspace_value = (setting("WORKSPACE_PATH", "") or "").strip()
    workspace_root = Path(workspace_value) if workspace_value else None
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
                skill_info["enabled"] = get_toggle_state(
                    engine, "skill_workflow", skill_info["id"]
                )
                workflows.append(skill_info)
            else:
                skill_info["type"] = "Agent Skill"
                skill_info["enabled"] = get_toggle_state(
                    engine, "skill", skill_info["id"]
                )
                skills.append(skill_info)

    # 4. Skill Graphs
    graphs = []
    graphs_dir = (
        workspace_root / "agent-packages" / "skills" / "skill-graphs" / "skill_graphs"
        if workspace_root is not None
        else None
    )
    if graphs_dir is not None and graphs_dir.exists():
        for p in graphs_dir.glob("**/SKILL.md"):
            skill_info = _parse_skill_md(p)
            skill_info["type"] = "Skill Graph"
            skill_info["enabled"] = get_toggle_state(
                engine, "skill_graph", skill_info["id"]
            )
            graphs.append(skill_info)

    return JSONResponse(
        {
            "mcp_tools": mcp_tools,
            "builtin_tools": builtin_tools,
            "skills": sorted(skills, key=lambda x: x.get("name", "").lower()),
            "skill_graphs": sorted(graphs, key=lambda x: x.get("name", "").lower()),
            "skill_workflows": sorted(
                workflows, key=lambda x: x.get("name", "").lower()
            ),
        }
    )


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
    set_toggle_state(engine, item_type, item_id, enabled)
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
    "ontology_function": "/ontology/function",
    "ontology_derive": "/ontology/derive",
    "ontology_link_materialize": "/ontology/link-materialize",
    "ontology_leanix_sync": "/ontology/leanix-sync",
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
}

# Immutable seed used by deterministic catalog generators. Runtime registrars
# extend ``ACTION_TOOL_ROUTES`` with their own twins, but a generator must never
# inherit routes left behind by an earlier server build in the same process.
BASE_ACTION_TOOL_ROUTES = MappingProxyType(dict(ACTION_TOOL_ROUTES))


def _make_tool_endpoint(tool_name: str):
    """Build a thin REST handler that dispatches a JSON body to an MCP tool.

    Both the MCP tool surface and the REST surface funnel through
    :func:`_execute_tool` against the shared in-process engine, so a handler is
    just: parse body → execute tool → wrap result. This factory is the canonical
    adapter; per-tool endpoints below that need bespoke parsing keep their own
    definitions, but every tool in :data:`ACTION_TOOL_ROUTES` without one is
    served by this.
    """

    async def _handler(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            res = await _execute_tool(tool_name, **body)
            return JSONResponse({"status": "success", "result": safe_json_load(res)})
        except Exception as e:
            return _external_error_response(e)

    _handler.__name__ = f"{tool_name}_endpoint"
    return _handler


async def graph_query_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool("graph_query", **body)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_search_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool("graph_search", **body)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_write_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool("graph_write", **body)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_ingest_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool("graph_ingest", **body)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool("graph_analyze", **body)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


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


def _make_mining_deep_endpoint(action: str):
    """Build the REST twin for one ``graph_mine_deep`` action (CONCEPT:AU-KG.mining.dsm-forecast-delegation).

    ``POST /api/mining/deep/<action>`` accepts a natural body (``x``/``values``/
    ``source``, ``y``, ``writeback``, algo kwargs, ...) plus an optional ``graph``,
    and dispatches the SAME ``_execute_tool("graph_mine_deep", action=<action>, ...)``
    core the MCP verb uses — the delegated call to data-science-mcp and the KG
    foldback happen once, in that one core.
    """

    async def _endpoint(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            return JSONResponse(
                {"status": "error", "message": "body must be a JSON object"},
                status_code=400,
            )
        graph = body.pop("graph", "") or ""
        try:
            res = await _execute_tool(
                "graph_mine_deep",
                action=action,
                params_json=json.dumps(body),
                graph=graph,
            )
            return JSONResponse({"status": "success", "result": safe_json_load(res)})
        except Exception as e:
            return _external_error_response(e)

    return _endpoint


def _make_graphlearn_endpoint(action: str):
    """Build the REST twin for one ``graph_learn`` action (CONCEPT:EG-KG.graphlearn.link-predictor).

    ``POST /api/graphlearn/<action>`` accepts a natural body (the action's kwargs,
    e.g. ``{node_label, direction, degree, epochs, writeback, ...}`` for fit,
    ``{model, node_label, top_k|candidate_pairs, writeback, ...}`` for predict) plus an
    optional ``graph``, and dispatches the SAME
    ``_execute_tool("graph_learn", action=<action>, ...)`` core as the MCP verb.
    """

    async def _endpoint(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            return JSONResponse(
                {"status": "error", "message": "body must be a JSON object"},
                status_code=400,
            )
        graph = body.pop("graph", "") or ""
        try:
            res = await _execute_tool(
                "graph_learn",
                action=action,
                params_json=json.dumps(body),
                graph=graph,
            )
            return JSONResponse({"status": "success", "result": safe_json_load(res)})
        except Exception as exc:  # noqa: BLE001 — canonical safe error surface
            return _external_error_response(exc)

    return _endpoint


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
    ``{label,min_support,max_edges,algorithm,...}`` for subgraph) plus an
    optional ``graph``, and dispatches the SAME
    ``_execute_tool("graph_mine", action=<action>, ...)`` core as the MCP verb.
    """

    async def _endpoint(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            return JSONResponse(
                {"status": "error", "message": "body must be a JSON object"},
                status_code=400,
            )
        graph = body.pop("graph", "") or ""
        try:
            res = await _execute_tool(
                "graph_mine",
                action=action,
                params_json=json.dumps(body),
                graph=graph,
            )
            return JSONResponse({"status": "success", "result": safe_json_load(res)})
        except Exception as e:
            return _external_error_response(e)

    return _endpoint


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
async def graph_search_concept_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_search",
            query=body.get("query", ""),
            mode="concept",
            top_k=int(body.get("top_k", 10)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_search_analogy_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_search",
            query=body.get("query", ""),
            mode="analogy",
            top_k=int(body.get("top_k", 10)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_search_memory_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_search",
            query=body.get("query", ""),
            mode="memory",
            top_k=int(body.get("top_k", 10)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_search_discover_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_search", query=body.get("query", ""), mode="discover"
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_search_dci_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_search",
            query=body.get("query", ""),
            mode="dci",
            top_k=int(body.get("top_k", 10)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


# 3. Granular Graph Write endpoints
async def graph_write_node_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_write",
            action="add_node",
            id=body.get("node_id", ""),
            node_type=body.get("node_type", ""),
            properties=_to_json_str(body.get("properties", {})),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_write_delete_node_endpoint(request: Request) -> JSONResponse:
    try:
        node_id = request.path_params.get("node_id", "")
        res = await _execute_tool("graph_write", action="delete_node", id=node_id)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_write_edge_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_write",
            action="add_edge",
            source_id=body.get("source_id", ""),
            target_id=body.get("target_id", ""),
            rel_type=body.get("rel_type", ""),
            properties=_to_json_str(body.get("properties", {})),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_write_delete_edge_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_write",
            action="delete_edge",
            source_id=body.get("source_id", ""),
            target_id=body.get("target_id", ""),
            rel_type=body.get("rel_type", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_write_external_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_write",
            action="register_external_graph",
            endpoint_url=body.get("endpoint_url", ""),
            graph_type=body.get("graph_type", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_write_bulk_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_write",
            action="bulk_ingest",
            nodes=_to_json_str(body.get("nodes", [])),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


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
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_write",
            action="recall_memory",
            properties=body.get("query", ""),
            node_type=body.get("memory_type", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


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


async def graph_write_chat_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_write",
            action="log_chat",
            agent_id=body.get("agent_id", ""),
            properties=body.get("content", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_write_sdd_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_write",
            action="submit_sdd",
            agent_id=body.get("agent_id", ""),
            properties=body.get("content", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_write_execution_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_write",
            action="register_execution",
            agent_id=body.get("agent_id", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


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
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_ingest",
            action="corpus",
            corpus_name=body.get("corpus_name", ""),
            base_path=body.get("base_path", ""),
            description=body.get("description", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


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
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_ingest",
            action="observe",
            target_path=body.get("target_path", ""),
            agent_id=body.get("agent_id", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


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
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_ingest",
            action="ingest_knowledge_pack",
            target_path=body.get("target_path", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


# 5. Granular Graph Analyze endpoints
async def graph_analyze_synthesize_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_research",
            action="synthesize",
            query=body.get("query", ""),
            top_k=int(body.get("top_k", 10)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_process_writeback_endpoint(request: Request) -> JSONResponse:
    """Push KG process intelligence INTO Camunda instances / ARIS models.

    Body: ``{"target": "both|camunda|aris", "query": "id1,id2"}`` —
    ``target`` is the writeback scope (default ``both``); ``query`` is an
    optional comma-separated list of BusinessProcess node ids to limit to.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_analyze",
            action="process_writeback",
            target=body.get("target", "both"),
            query=body.get("query", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_deep_extract_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_research",
            action="deep_extract",
            query=body.get("query", ""),
            top_k=int(body.get("top_k", 10)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_background_research_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_research",
            action="background_research",
            query=body.get("query", ""),
            top_k=int(body.get("top_k", 10)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_relevance_sweep_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_research",
            action="relevance_sweep",
            query=body.get("query", ""),
            top_k=int(body.get("top_k", 10)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


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
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_code",
            action="change_coupling",
            target=body.get("repo", ""),
            depth=int(body.get("min_support", 3)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_code_evolution_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=code_evolution (CONCEPT:AU-KG.enrichment.query-ingested-commit-history): query the
    ingested commit-history graph for codebase evolution. Body:
    ``{mode?, target?, top_k?}`` — mode = file|owners|hotspots|coupled,
    target = file path / subsystem path substring."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_code",
            action="code_evolution",
            target=body.get("mode", "file"),
            query=body.get("target", ""),
            top_k=int(body.get("top_k", 20)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_adr_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=adr (CONCEPT:AU-KG.compute.adr-crud): ADR CRUD. Body:
    ``{title?, status?, decision?}`` — title creates, empty lists."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_code",
            action="adr",
            query=body.get("title", ""),
            target=body.get("status", ""),
            node_id=body.get("decision", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


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
    """REST twin of graph_analyze action=code_context (CONCEPT:AU-KG.retrieval.synthesized-cited-answer): the
    synthesized, cited codebase Q&A. Body: ``{query, intent?(how|usage|impact),
    node_id?, top_k?, depth?, cross_repo?}``."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        intent = str(body.get("intent", "how"))
        if body.get("cross_repo"):
            intent = f"{intent}+xrepo"
        res = await _execute_tool(
            "graph_code",
            action="code_context",
            query=body.get("query", ""),
            target=intent,
            node_id=body.get("node_id", ""),
            top_k=int(body.get("top_k", 10)),
            depth=int(body.get("depth", 2)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_explain_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_explain action=explain (CONCEPT:AU-KG.retrieval.route-question-its-domain): the universal
    context plane. Body: ``{query, domain?, intent?, node_id?, top_k?, depth?}`` —
    routes to the domain provider (code | ops | …) and returns the cited answer."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        domain = str(body.get("domain", ""))
        intent = str(body.get("intent", ""))
        target = f"{domain}:{intent}" if domain else intent
        res = await _execute_tool(
            "graph_explain",
            action="explain",
            query=body.get("query", ""),
            target=target,
            node_id=body.get("node_id", ""),
            top_k=int(body.get("top_k", 10)),
            depth=int(body.get("depth", 2)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


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


async def graph_analyze_code_metrics_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=code_metrics (CONCEPT:AU-KG.retrieval.god-nodes-communities): Graphify-
    style god nodes / communities / surprising connections over the :Code subgraph.
    ``scope`` (or ``target``) = optional file_path/source_system substring;
    ``top_k`` = section sizes."""
    try:
        scope = request.query_params.get("scope") or request.query_params.get(
            "target", ""
        )
        top_k = int(request.query_params.get("top_k", "10"))
        res = await _execute_tool(
            "graph_code", action="code_metrics", target=scope, top_k=top_k
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_arch_report_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=arch_report (CONCEPT:AU-KG.retrieval.architecture-report): the
    regenerable architecture report (GRAPH_REPORT.md analog) as Markdown + metrics.
    ``scope`` (or ``target``) = optional substring; ``top_k`` = section sizes."""
    try:
        scope = request.query_params.get("scope") or request.query_params.get(
            "target", ""
        )
        top_k = int(request.query_params.get("top_k", "10"))
        res = await _execute_tool(
            "graph_code", action="arch_report", target=scope, top_k=top_k
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_analyze_context_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_explain",
            action="context",
            target=body.get("target", ""),
            query=body.get("query", ""),
            top_k=int(body.get("top_k", 10)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


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
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_configure",
            action="set_secret",
            config_key=body.get("config_key", ""),
            config_value=body.get("config_value", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_configure_vault_sync_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_configure action=vault_sync (CONCEPT:AU-OS.deployment.vault-first-routine-genesis)."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_configure",
            action="vault_sync",
            config_key=body.get("config_key", ""),
            config_value=body.get("config_value", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


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
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_configure",
            action="install_hooks",
            config_value=body.get("config_value", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


async def graph_configure_uninstall_hooks_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_configure",
            action="uninstall_hooks",
            config_value=body.get("config_value", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return _external_error_response(e)


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

    engine = IntelligenceGraphEngine.get_active()
    if engine is not None:
        return engine

    with _ENGINE_LOCK:
        engine = IntelligenceGraphEngine.get_active()
        if engine is not None:
            return engine
        # First-run: ensure XDG dirs exist and create backend
        ensure_dirs()

        def _factory():
            backend = create_backend()
            return IntelligenceGraphEngine(backend=backend)

        return IntelligenceGraphEngine.get_or_create(factory=_factory)


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
        try:
            from agent_utilities.core.config import config as _cfg

            for declared in _cfg.external_graph_connectors or []:
                value = (
                    declared.model_dump()
                    if hasattr(declared, "model_dump")
                    else dict(declared)
                )
                name = str(value.pop("name", "") or "")
                if not name:
                    continue
                value["role"] = "read"
                try:
                    registry.register(name, value)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Skipping invalid external source declaration (%s)",
                        type(exc).__name__,
                    )

            for spec in _cfg.kg_connections or []:
                spec = dict(spec)
                name = spec.pop("name", "")
                if name:
                    try:
                        registry.register(name, spec)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "Skipping invalid graph connection declaration (%s)",
                            type(e).__name__,
                        )
        except Exception as exc:  # noqa: BLE001 — config-less environments
            logger.debug(
                "Graph connection declarations were not seeded (%s)",
                type(exc).__name__,
            )
        _CONNECTION_REGISTRY = registry
        return _CONNECTION_REGISTRY


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


def fanout_execute(entries, fn, *, timeout=None):
    """Run ``fn(name, engine)`` for every fan-out target CONCURRENTLY under a shared
    per-target wall-clock timeout, so one slow/unreachable backend can't stall the
    others (CONCEPT:AU-KG.backend.multi-connection-registry).

    Returns ``(results, errors)`` keyed by connection name. A target that exceeds the
    budget (or raises) lands in ``errors`` while the rest still return — the
    partial-success contract the sequential loop violated by blocking on the slowest.
    """
    import concurrent.futures

    if timeout is None:
        timeout = float(setting("GRAPH_FANOUT_TIMEOUT", DEFAULT_FANOUT_TIMEOUT_S))
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    if not entries:
        return results, errors
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(entries)))
    futures = {ex.submit(fn, name, engine): name for name, engine in entries}
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
            errors[name] = "target_operation_failed"
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


def _read_skill_capability(skill_md) -> tuple[str, str, str]:
    """Read one bounded skill declaration without retaining its discovery path."""
    import yaml

    path = Path(skill_md)
    payload = path.read_bytes()
    if not payload or len(payload) > 512 * 1024:
        raise ValueError("skill declaration size is invalid")
    content = payload.decode("utf-8")
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
    fallback_name = path.parent.name
    name = str(frontmatter.get("name") or fallback_name).strip()
    description = str(frontmatter.get("description") or "").strip()
    if not name or not instructions.strip():
        raise ValueError("skill declaration is incomplete")
    return name, description, instructions


def _ingest_skill_capabilities(
    engine,
    provider: str,
    skills_path,
    *,
    include_names: frozenset[str] | None = None,
    skip_names: frozenset[str] = frozenset(),
) -> int:
    """Persist provider skills as runnable resources without retaining paths."""
    from agent_utilities.core.providers import is_skill_graph_reference_path
    from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
        ingest_runnable_skill,
        skill_reference,
    )

    root = Path(skills_path)
    if not root.is_dir():
        return 0

    ingested = 0
    skill_files = (
        [root / "SKILL.md"]
        if (root / "SKILL.md").is_file()
        else sorted(
            skill_md
            for skill_md in root.rglob("SKILL.md")
            if not is_skill_graph_reference_path(skill_md, root)
        )
    )
    for skill_md in skill_files:
        skill_dir = skill_md.parent
        fallback_name = skill_dir.name
        stage = "declaration"
        try:
            name, description, instructions = _read_skill_capability(skill_md)
            if include_names is not None and name not in include_names:
                continue
            if name in skip_names:
                continue
            skill_slug = skill_reference(name).removeprefix("skill://")
            resource_id = f"resource:skill:{skill_slug}"
            stage = "state"
            disabled = get_existing_disabled(engine, resource_id)
            stage = "write"
            ingest_runnable_skill(
                engine,
                name=name,
                description=description,
                instructions=instructions,
                provider=provider,
                disabled=disabled,
            )
            ingested += 1
        except Exception as exc:  # noqa: BLE001 - one malformed skill cannot block boot
            from agent_utilities.security.persistence_privacy import (
                persistence_reference,
            )

            logger.error(
                "Failed to ingest %s (stage=%s %s: %s)",
                persistence_reference(
                    "skill", fallback_name, namespace="skill-provider-ingest"
                ),
                stage,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
    return ingested


def _bundled_skill_contract() -> tuple[Path, dict[str, str]]:
    """Load the exact current packaged-skill digest contract."""
    from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
        runnable_skill_digest,
    )
    from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard
    from agent_utilities.skills import BUNDLED_SKILLS

    if len(BUNDLED_SKILLS) != 10 or len(set(BUNDLED_SKILLS)) != 10:
        raise GraphOSStartupReadinessError("graphos_bundled_skills_unready")
    root = Path(__file__).resolve().parents[1] / "skills"
    guard = PersistencePrivacyGuard()
    expected: dict[str, str] = {}
    try:
        for bundled_name in BUNDLED_SKILLS:
            name, _description, instructions = _read_skill_capability(
                root / bundled_name / "SKILL.md"
            )
            if name != bundled_name:
                raise ValueError("bundled skill identity mismatch")
            body, _privacy = guard.sanitize_text(instructions.strip())
            if not body:
                raise ValueError("bundled skill body is empty")
            expected[bundled_name] = runnable_skill_digest(body)
    except Exception as exc:
        # Same reasoning as _start_engine_bootstrap: the exception class alone is
        # not diagnosable. Keep the message and the chained traceback so the
        # actual reason a skill could not be ingested is visible in the log.
        logger.error(
            "GraphOS packaged-skill readiness check failed (%s: %s)",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        raise GraphOSStartupReadinessError("graphos_bundled_skills_unready") from exc
    return root, expected


def _ready_bundled_skill_names(
    engine: Any, expected_digests: dict[str, str]
) -> frozenset[str]:
    """Return exact packaged skills already ready for delegated execution."""
    from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
        runnable_skill_digest,
        skill_reference,
    )

    query = getattr(engine, "query_cypher", None)
    if not callable(query):
        return frozenset()
    try:
        rows = query(
            "MATCH (n:CallableResource) WHERE n.name IN $names "
            "RETURN n.id AS id, n.name AS name, n.resource_type AS rtype, "
            "n.system_prompt AS system_prompt, "
            "n.instruction_digest AS instruction_digest, "
            "n.source_ref AS source_ref, n.runnable_bound AS runnable_bound",
            {"names": sorted(expected_digests)},
        )
    except Exception as exc:
        # This is a READINESS PROBE: "which bundled skills are already ingested".
        # On a graph that does not exist yet — a first boot, or the first boot
        # after a tenant claim starts scoping this process to a new tenant graph
        # — the engine answers "Graph '<name>' not found" rather than an empty
        # result. That is the correct answer to "nothing is ready", not a
        # failure, and treating it as fatal makes the server unable to perform
        # the very ingestion that would create the graph. Report none-ready and
        # let the caller ingest; a genuine engine fault still surfaces there.
        logger.info(
            "bundled-skill readiness probe found no existing skill graph "
            "(%s: %s); treating every bundled skill as not yet ingested",
            type(exc).__name__,
            exc,
        )
        return frozenset()
    candidates: dict[str, list[dict[str, Any]]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        if name in expected_digests:
            candidates.setdefault(name, []).append(row)

    ready: set[str] = set()
    for name, expected_digest in expected_digests.items():
        matches = candidates.get(name, [])
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
        for row in matches:
            body = str(row.get("system_prompt") or "").strip()
            digest = str(row.get("instruction_digest") or "")
            if (
                row.get("id") == f"resource:skill:{name}"
                and row.get("rtype") == "AGENT_SKILL"
                and row.get("runnable_bound") is True
                and row.get("source_ref") == expected_ref
                and body
                and digest == expected_digest
                and runnable_skill_digest(body) == digest
            ):
                ready.add(name)
                break
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
        # Same reasoning as _start_engine_bootstrap: the exception class alone is
        # not diagnosable. Keep the message and the chained traceback so the
        # actual reason a skill could not be ingested is visible in the log.
        logger.error(
            "GraphOS packaged-skill readiness check failed (%s: %s)",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return {
            "required": len(BUNDLED_SKILLS),
            "already_ready": 0,
            "ingested": 0,
            "ready": 0,
            "not_ready": sorted(BUNDLED_SKILLS),
            "error": f"{type(exc).__name__}: {exc}",
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
    import importlib
    import inspect
    import json
    import pkgutil
    from pathlib import Path

    import platformdirs

    from agent_utilities.security.persistence_privacy import (
        sanitize_for_persistence,
    )

    # 1. mcp_config.json
    try:
        APP_NAME = "agent-utilities"
        APP_AUTHOR = "knuckles-team"
        cfg_dir = Path(platformdirs.user_config_path(APP_NAME, APP_AUTHOR))
        mcp_config_path = cfg_dir / "mcp_config.json"

        if mcp_config_path.is_file() and not mcp_config_path.is_symlink():
            payload = mcp_config_path.read_bytes()
            if len(payload) > 4 * 1024 * 1024:
                raise ValueError("MCP configuration exceeds its ingestion bound")
            data = json.loads(payload)
            mcp_servers = data.get("mcpServers", {})
            if not isinstance(mcp_servers, dict):
                raise ValueError("MCP server registry must be an object")
            ingested = 0
            for server_name, server_details in mcp_servers.items():
                if not isinstance(server_details, dict):
                    continue
                node_id, declaration = _mcp_capability_declaration(
                    server_name, server_details
                )
                disabled = get_existing_disabled(engine, node_id)
                engine.add_node(
                    node_id,
                    "MCPServer",
                    {**declaration, "disabled": disabled},
                )
                ingested += 1
            logger.info("Ingested %d MCP capability declarations", ingested)
    except Exception as exc:
        logger.error(
            "Failed to ingest MCP configuration (exception_type=%s)",
            type(exc).__name__,
        )

    # 2. Native Tools
    try:
        import agent_utilities.tools

        prefix = agent_utilities.tools.__name__ + "."
        for importer, modname, ispkg in pkgutil.iter_modules(
            agent_utilities.tools.__path__, prefix
        ):
            if not ispkg:
                try:
                    module = importlib.import_module(modname)
                    for name, obj in inspect.getmembers(module, inspect.isfunction):
                        if hasattr(obj, "__agentic_version__"):
                            node_id = f"native_tool_{name}"
                            disabled = get_existing_disabled(engine, node_id)
                            description, _privacy = sanitize_for_persistence(
                                (obj.__doc__ or "")[:8192]
                            )
                            engine.add_node(
                                node_id,
                                "NativeTool",
                                {
                                    "name": name,
                                    "description": str(description),
                                    "version": obj.__agentic_version__,
                                    "module": modname,
                                    "disabled": disabled,
                                },
                            )
                except Exception as exc:
                    logger.debug(
                        "Failed to ingest a native-tool module (exception_type=%s)",
                        type(exc).__name__,
                    )
        logger.info("Ingested Native Tools")
    except Exception as exc:
        logger.error(
            "Failed to scan native tools (exception_type=%s)",
            type(exc).__name__,
        )

    # 3. Skills
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
        logger.error("Failed to ingest skills (%s)", type(e).__name__)

    # Fleet tool schemas stay lazy.  Startup has already materialized each MCP
    # server declaration above; probing every child here would launch the whole
    # fleet and contend with an operator's targeted ``list_catalog`` call.
    # Explicit ``source_sync(source="fleet")`` remains the governed full-scan
    # path when an operator wants every live tool schema elevated into the KG.


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


def _refresh_process_authority(session: Any) -> Any:
    """Renew one external process lease without replacing captured sessions.

    The token exists only inside this call. After validation, only its bounded
    expiry is copied into the shared in-memory lease. Identity, roles, tenant,
    route, and policy may not change during renewal.
    """
    from agent_utilities.core.config import config
    from agent_utilities.knowledge_graph.core.session import SessionExpiredError
    from agent_utilities.security.request_identity import (
        acquire_process_identity_token,
        mint_actor_from_token_sync,
    )

    lease = getattr(getattr(session, "actor", None), "credential_lease", None)
    if lease is None:
        raise RuntimeError("Graph process authority is not renewable")
    with _PROCESS_SESSION_REFRESH_LOCK:
        try:
            session.ensure_authority_current(minimum_ttl_seconds=30)
            return session
        except SessionExpiredError:
            pass
        token = acquire_process_identity_token(config)
        renewed_actor = mint_actor_from_token_sync(token)
        del token
        if not _same_process_authority(session.actor, renewed_actor):
            raise RuntimeError("Graph process authority changed during renewal")
        expires_at = getattr(renewed_actor, "credential_expires_at", None)
        if expires_at is None or int(expires_at) <= int(time.time()) + 30:
            raise RuntimeError("Graph process authority renewal is too short-lived")
        lease.renew(int(expires_at))
        session.ensure_authority_current(minimum_ttl_seconds=30)
        return session


async def _ensure_process_authority_current() -> Any:
    """Ensure request/process authority without blocking the MCP event loop."""
    from agent_utilities.knowledge_graph.core.session import (
        SessionExpiredError,
        current_session,
    )

    ambient = current_session()
    if ambient is not None:
        ambient.ensure_authority_current()
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
    if getattr(getattr(session, "actor", None), "credential_lease", None) is None:
        return
    _stop_process_authority_supervisor()
    _PROCESS_AUTHORITY_STOP.clear()
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


def _start_engine_bootstrap(session: Any) -> None:
    """Establish critical skill readiness, then start noncritical services."""
    from agent_utilities.knowledge_graph.core.engine_tasks import (
        _authorized_background_thread,
        _require_verified_background_session,
    )
    from agent_utilities.knowledge_graph.core.session import use_session
    from agent_utilities.security.brain_context import use_actor
    from agent_utilities.skills import BUNDLED_SKILLS

    verified_session = _require_verified_background_session(session)
    try:
        with (
            use_actor(verified_session.actor),
            use_session(verified_session),
        ):
            engine = _get_engine()
            readiness = _ensure_bundled_skills_ready(engine)
    except Exception as exc:
        # Log the cause, not just its class. Reporting only `exception_type=X`
        # leaves an operator with nothing actionable — every distinct failure
        # (a missing symbol, an unreachable engine, a denied capability) reads
        # identically as "graphos_bundled_skills_unready", and `raise ... from
        # None` then discards the chained traceback too. The message and the
        # original traceback are what make the next failure diagnosable.
        # Packaged-skill readiness is a CAPABILITY concern, not a correctness or
        # security one, so it must not decide whether graph-os serves at all. A
        # server that refuses to boot because some bundled skills did not ingest
        # takes down every unrelated tool, the health surface, and the operator's
        # ability to diagnose the very problem — the failure mode is far worse
        # than running degraded. Record it, surface it in /health, keep serving.
        logger.error(
            "GraphOS packaged-skill bootstrap failed; SERVING DEGRADED (%s: %s)",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        _set_bundled_skill_readiness(
            {
                "required": len(BUNDLED_SKILLS),
                "ready": 0,
                "not_ready": sorted(BUNDLED_SKILLS),
                "error": f"{type(exc).__name__}: {exc}",
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

    def _bootstrap_engine() -> None:
        try:
            if (
                engine
                and engine.backend
                and not getattr(engine.backend, "read_only", False)
            ):
                engine.start_task_workers()
            # The listener barrier already reconciled the exact packaged skill
            # contract. Continue all broader discovery asynchronously, but do
            # not rewrite or overlay those ten resources a second time.
            _ingest_capabilities(
                engine,
                skip_skill_names=frozenset(BUNDLED_SKILLS),
            )
            try:
                from agent_utilities.knowledge_graph.ontology.lifecycle import (
                    OntologyLifecycle,
                )
                from agent_utilities.mcp.tools.ontology_tools import (
                    _sync_package_ontologies,
                )

                report = _sync_package_ontologies(OntologyLifecycle(engine=engine))
                if report.get("providers_loaded"):
                    logger.info(
                        "Ontology federation: loaded %d package ontolog(ies) at boot",
                        report["providers_loaded"],
                    )
            except Exception as exc:
                logger.error(
                    "Ontology federation sync at boot failed (exception_type=%s)",
                    type(exc).__name__,
                )
        except Exception as exc:
            logger.error(
                "KG engine background bootstrap failed (exception_type=%s)",
                type(exc).__name__,
            )

    try:
        _authorized_background_thread(
            verified_session,
            _bootstrap_engine,
            name="KGEngineBootstrap",
        ).start()
    except Exception as exc:
        # Packaged delegation is already ready. Optional workers, provider
        # discovery, and ontology federation remain retryable operational work.
        logger.error(
            "GraphOS noncritical bootstrap launch failed (exception_type=%s)",
            type(exc).__name__,
        )


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
            "doesn't exist."
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
    # downstream dependency being down must NOT make kubelet kill/restart an
    # otherwise-fine process (that would crash-loop the pod for something a
    # restart cannot fix). The body is the truthful, loud signal: real engine
    # reachability + circuit-breaker state, plus every configured co-service/
    # dependency — never a stub that reports "ok" regardless of reality.
    #
    # ``/health/ready`` is READINESS: the SAME report, but its overall status
    # maps onto the HTTP status code (200 healthy / 503 unhealthy) so kubelet
    # pulls this pod out of Service routing while it's genuinely broken,
    # without touching the process.
    #
    # Both stay status-only-by-default in the sense that only non-secret detail
    # (counts, booleans, resolved modes, platform ids) is ever included — no
    # endpoint DSNs, tokens, or credentials.
    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:  # noqa: ARG001
        from agent_utilities.observability.runtime_health import collect_health

        report = await asyncio.to_thread(collect_health)
        return JSONResponse(report, headers={"Cache-Control": "no-store"})

    @mcp.custom_route("/health/ready", methods=["GET"])
    async def readiness_check(request: Request) -> JSONResponse:  # noqa: ARG001
        from agent_utilities.observability.runtime_health import (
            collect_health,
            is_overall_healthy,
        )

        report = await asyncio.to_thread(collect_health)
        status_code = 200 if is_overall_healthy(report) else 503
        return JSONResponse(
            report, status_code=status_code, headers={"Cache-Control": "no-store"}
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
        register_compliance_tools,
        register_domain_ops_tools,
        register_engine_surface_tools,
        register_engine_tools,
        register_epistemic_tools,
        register_evolution_tools,
        register_governance_tools,
        register_graph_engineering_tools,
        register_incident_tools,
        register_job_tools,
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
            register_secret_tools,
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
            register_compliance_tools,
            register_rlm_tools,
            register_workflow_tools,
            register_argument_tools,
        ],
        verbose_register=register_graphos_verbose_tools,
        mode_override=tool_profile,
        force_condensed_registration=canonical_surface,
    )

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

    # In ``both`` mode the condensed action tools are also registered; a single-op
    # (action=None) verbose tool shares the condensed tool's NAME, so skip it to
    # avoid overwriting the (untagged) condensed tool with a verbose-tagged
    # duplicate. In verbose-only mode there is no condensed tool — keep them.
    skip_single_op = tool_mode() == "both"

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

    def route(path: str, handler, methods: list[str]) -> None:
        app.add_route(prefix + path, handler, methods=methods)

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
    route("/graph/write", graph_write_endpoint, ["POST"])
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
    route("/graph/search/concept", graph_search_concept_endpoint, ["POST"])
    route("/graph/search/analogy", graph_search_analogy_endpoint, ["POST"])
    route("/graph/search/memory", graph_search_memory_endpoint, ["POST"])
    route("/graph/search/discover", graph_search_discover_endpoint, ["POST"])
    route("/graph/search/dci", graph_search_dci_endpoint, ["POST"])

    # ── Granular write ──
    route("/graph/write/node", graph_write_node_endpoint, ["POST"])
    route("/graph/write/node/{node_id}", graph_write_delete_node_endpoint, ["DELETE"])
    route("/graph/write/edge", graph_write_edge_endpoint, ["POST"])
    route("/graph/write/edge", graph_write_delete_edge_endpoint, ["DELETE"])
    route("/graph/write/external", graph_write_external_endpoint, ["POST"])
    route("/graph/write/bulk", graph_write_bulk_endpoint, ["POST"])
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
    route("/graph/write/chat", graph_write_chat_endpoint, ["POST"])
    route("/graph/write/sdd", graph_write_sdd_endpoint, ["POST"])
    route("/graph/write/execution", graph_write_execution_endpoint, ["POST"])

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
    # /api/mining/<action> endpoint per graph_mine action (associate|cluster|anomaly|
    # classify_fit|classify_predict|reduce), each dispatching the SAME graph_mine
    # _execute_tool core (surface parity).
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
    _configure_graphos_otel()
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
    fleet_mux = None
    try:
        from agent_utilities.mcp.multiplexer import attach_fleet_loader

        # Inject graph-os's own embedding model so find_tools ranks fleet tools by
        # query↔description MEANING (semantic), not just literal token overlap.
        fleet_mux = attach_fleet_loader(
            mcp,
            embed_fn=_fleet_embed_fn(),
            authority_scope=verified_tool_session_scope,
        )
    except Exception as exc:
        logger.error(
            "graph-os fleet loader attach failed; fleet tools disabled "
            "(exception_type=%s)",
            type(exc).__name__,
        )

    transport = getattr(args, "transport", "stdio")
    host = getattr(args, "host", "127.0.0.1")
    port = int(getattr(args, "port", 8000))

    bootstrap_session = _mint_process_session(transport)
    _PROCESS_SESSION = bootstrap_session if transport == "stdio" else None
    _start_process_authority_supervisor(bootstrap_session)

    co_service_supervisor = None
    try:
        logger.info("Starting graph-os MCP server (transport=%s)", transport)

        from agent_utilities.mcp.server_factory import (
            mcp_network_run_kwargs,
            protect_stdio_jsonrpc,
        )
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

        # Stdout purity BEFORE any co-service can log a single line. On stdio,
        # stdout IS the JSON-RPC channel — this monkeypatches builtins.print /
        # warnings.showwarning process-wide, so it protects every co-service
        # thread started below too, not just this one. No-op for network
        # transports (they don't own stdout as a protocol channel).
        if transport == "stdio":
            protect_stdio_jsonrpc()

        # Self-composing co-services, phase 1: the KG host daemon must be decided
        # BEFORE the engine is constructed below so it can win (or lose) the
        # host-lock race as the first constructor (see co_service_supervisor's
        # module docstring for the exact "client role + no live host" detection).
        from agent_utilities.mcp.co_service_supervisor import (
            bring_up_host_daemon_if_needed,
            start_co_services,
        )

        bring_up_host_daemon_if_needed()
        _start_engine_bootstrap(bootstrap_session)

        # Self-composing co-services, phase 2: messaging (config-detected — real
        # platform credentials present) now that a real engine exists; agent-webui
        # is reported (ENABLE_WEB_UI) but is an external Node frontend, never
        # started in-process.
        co_service_supervisor = start_co_services(bootstrap_session, _get_engine())

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
            except Exception as exc:
                logger.debug(
                    "fleet loader close failed (exception_type=%s)",
                    type(exc).__name__,
                )


if __name__ == "__main__":
    mcp_server()
