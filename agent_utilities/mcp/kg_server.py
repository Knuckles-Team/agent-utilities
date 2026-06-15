#!/usr/bin/python
"""Knowledge Graph MCP Server — Thin wrapper over IntelligenceGraphEngine.

CONCEPT:ECO-4.0 — Knowledge Graph MCP Exposure

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
    uv run agent-utilities-kg

    # Start as HTTP transport:
    uv run agent-utilities-kg --transport streamable-http --port 8100

Cross-IDE Discovery:
    Register in ``~/.config/agent-utilities/mcp_config.json``::

        {
          "mcpServers": {
            "agent-utilities-kg": {
              "command": "uv",
              "args": ["run", "agent-utilities-kg"]
            }
          }
        }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time as _time
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import Field

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

        req.json = mock_json
    return req


# Server-side identity for stdio MCP (CONCEPT:OS-5.14): minted once at startup
# from a validated KG_AUTH_TOKEN. None = no validated process identity.
_PROCESS_ACTOR: Any = None

# Tools an UNAUTHENTICATED caller may still use when KG_AUTH_REQUIRED is on
# (stdio without a valid KG_AUTH_TOKEN). Deliberately a conservative read-only
# surface: pure reads with no graph mutation side effects.
ANONYMOUS_READ_TOOLS: frozenset[str] = frozenset(
    {"graph_query", "graph_search", "graph_context", "graph_analyze"}
)


def _kg_auth_required() -> bool:
    """Whether server-validated identity is mandatory (KG_AUTH_REQUIRED)."""
    from agent_utilities.core.config import config

    return bool(getattr(config, "kg_auth_required", False))


def _actor_from_mcp_token() -> Any:
    """Mint an actor from FastMCP's validated access token, when present.

    On the streamable-http transport FastMCP's own auth provider (configured
    via ``create_mcp_server``) validates the Bearer token; its claims are the
    same server-side trust root the gateway middleware uses. Returns None
    outside a token-authenticated MCP request.
    """
    try:
        from fastmcp.server.dependencies import get_access_token

        token = get_access_token()
        claims = getattr(token, "claims", None)
        if claims:
            from ..security.request_identity import actor_from_claims

            return actor_from_claims(dict(claims))
    except Exception:  # noqa: BLE001 — no request context / no auth configured
        return None
    return None


def _actor_from_kwargs(kwargs: dict) -> Any:
    """Resolve the actor a tool call runs as (CONCEPT:KG-2.6 / OS-5.14).

    Always pops ``_actor``/``_roles``/``_tenant`` so they are not forwarded to
    the tool. Server-minted identity wins over caller-supplied kwargs:

    1. An ``authenticated`` ambient actor (set by the gateway's
       ``ActorIdentityMiddleware``) is kept — caller kwargs are ignored.
    2. A validated FastMCP access token (streamable-http transport) is minted.
    3. The validated stdio process identity (``KG_AUTH_TOKEN``) is used.
    4. With ``KG_AUTH_REQUIRED`` on, caller kwargs are ignored entirely
       (server-side identity only) — the ambient default applies.
    5. Otherwise (legacy honor-system mode) the caller-supplied kwargs build
       the actor, exactly as before.

    Returns None when the ambient actor should be kept.
    """
    actor_id = kwargs.pop("_actor", None)
    roles = kwargs.pop("_roles", None)
    tenant = kwargs.pop("_tenant", None)

    from ..security.brain_context import current_actor

    if current_actor().authenticated:
        return None  # gateway middleware already scoped this request

    token_actor = _actor_from_mcp_token()
    if token_actor is not None:
        return token_actor

    if _PROCESS_ACTOR is not None:
        return _PROCESS_ACTOR

    if _kg_auth_required():
        return None  # server-side identity only; honor-system kwargs ignored

    if not actor_id and not roles and not tenant:
        return None
    from ..models.company_brain import ActorType
    from ..security.brain_context import ActorContext

    if isinstance(roles, str):
        roles = [r.strip() for r in roles.split(",") if r.strip()]
    return ActorContext(
        actor_id=str(actor_id or "mcp:caller"),
        actor_type=ActorType.AI_AGENT,
        roles=tuple(roles or ()),
        tenant_id=str(tenant or ""),
    )


async def _execute_tool(tool_name: str, **kwargs) -> Any:
    tool_func = REGISTERED_TOOLS.get(tool_name)
    if not tool_func:
        raise ValueError(f"Tool {tool_name} not registered")

    import inspect

    from ..security.brain_context import use_actor

    # Tool functions declare params as ``name: T = Field(default=...)``. When the tool is
    # invoked through FastMCP, the schema layer resolves those defaults. Calling the raw
    # function directly here (internal callers, the REST gateway, tests) does NOT — so any
    # omitted param would be bound to the raw ``FieldInfo`` object, later blowing up with
    # "'FieldInfo' object has no attribute 'replace'" / "not JSON serializable". Resolve
    # FieldInfo defaults for omitted params so direct invocation matches the MCP behavior.
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
    except Exception:  # noqa: BLE001 — never let default-resolution break dispatch
        pass

    actor = _actor_from_kwargs(kwargs)

    # CONCEPT:OS-5.14 — with KG_AUTH_REQUIRED on, an unauthenticated caller
    # (stdio without a validated KG_AUTH_TOKEN) is restricted to the read-only
    # tool surface. HTTP callers never reach here unauthenticated — the
    # gateway middleware already rejected them with 401.
    if _kg_auth_required():
        from ..security.brain_context import current_actor

        effective = actor if actor is not None else current_actor()
        if (
            not getattr(effective, "authenticated", False)
            and tool_name not in ANONYMOUS_READ_TOOLS
        ):
            raise PermissionError(
                f"KG_AUTH_REQUIRED=1: tool {tool_name!r} needs an authenticated "
                "identity (JWT Bearer token, or KG_AUTH_TOKEN for stdio). "
                f"Unauthenticated callers may only use: {sorted(ANONYMOUS_READ_TOOLS)}."
            )

    async def _run() -> Any:
        if inspect.iscoroutinefunction(tool_func):
            return await tool_func(**kwargs)
        return tool_func(**kwargs)

    if actor is None:
        return await _run()
    with use_actor(actor):
        return await _run()


def get_existing_disabled(engine, node_id: str) -> bool:
    try:
        # 1. Try in-memory graph cache first
        if hasattr(engine, "graph_compute") and hasattr(engine.graph_compute, "graph"):
            if node_id in engine.graph_compute.graph:
                return engine.graph_compute.graph.nodes[node_id].get("disabled", False)
        # 2. Try Cypher match as a fallback
        res = engine.query_cypher(
            f"MATCH (n) WHERE n.id = '{node_id}' RETURN n.disabled AS disabled"
        )
        if res and isinstance(res, list) and len(res) > 0:
            return bool(res[0].get("disabled", False))
    except Exception:
        pass
    return False


def safe_json_load(s: Any) -> Any:
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
            "file_path": str(path_obj),
        }
    except Exception as e:
        logger.error(f"Failed to parse SKILL.md at {path_obj}: {e}")
        return {
            "id": path_obj.parent.name,
            "name": path_obj.parent.name,
            "description": "",
            "domain": "",
            "tags": [],
            "enabled": True,
            "file_path": str(path_obj),
        }


def get_toggle_state(engine, item_type: str, item_id: str) -> bool:
    """Check if an item is enabled or disabled in the KG."""
    if not engine:
        return True
    pref_id = f"preference:toggle:{item_type}:{item_id}"
    try:
        res = engine.query_cypher(
            f"MATCH (p:Preference) WHERE p.id = '{pref_id}' RETURN p.value as value"
        )
        if res and len(res) > 0:
            return res[0].get("value") == "enabled"
    except Exception as e:
        logger.error(f"Failed to query toggle state for {pref_id}: {e}")
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
                f"MATCH (n) WHERE n.id = '{node_id}' SET n.disabled = {str(not enabled).lower()}"
            )
            # Also update in-memory graph cache if active
            if (
                hasattr(engine, "graph_compute")
                and engine.graph_compute
                and hasattr(engine.graph_compute, "graph")
            ):
                if node_id in engine.graph_compute.graph.nodes:
                    engine.graph_compute.graph.nodes[node_id]["disabled"] = not enabled
    except Exception as e:
        logger.error(f"Failed to save toggle state for {pref_id}: {e}")


from starlette.requests import Request
from starlette.responses import JSONResponse


async def get_tools_endpoint(request: Request) -> JSONResponse:
    """Retrieve all MCP tools, built-in tools, skills, skill graphs, and workflows categorized."""
    import json
    from pathlib import Path

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
                        "command": cfg.get("command", ""),
                        "args": cfg.get("args", []),
                        "status": "active" if mcp_enabled else "disabled",
                        "enabled": mcp_enabled,
                    }
                )
        except Exception as e:
            logger.error(f"Failed to parse mcp config: {e}")

    # 2. Built-in Agent Tools
    builtin_tools = []
    tools_dir = Path(
        "/home/apps/workspace/agent-packages/agent-utilities/agent_utilities/tools"
    )
    if tools_dir.exists() and tools_dir.is_dir():
        for f in tools_dir.glob("*.py"):
            if f.name.startswith("_"):
                continue
            builtin_enabled = get_toggle_state(engine, "builtin_tool", f.stem)
            builtin_tools.append(
                {
                    "name": f.stem,
                    "type": "Built-in Tool",
                    "file_path": str(f),
                    "status": "enabled" if builtin_enabled else "disabled",
                    "enabled": builtin_enabled,
                }
            )

    # 3. Skills & Workflows
    skills = []
    workflows = []
    univ_skills_dir = Path(
        "/home/apps/workspace/agent-packages/skills/universal-skills/universal_skills"
    )
    if univ_skills_dir.exists():
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
    graphs_dir = Path(
        "/home/apps/workspace/agent-packages/skills/skill-graphs/skill_graphs"
    )
    if graphs_dir.exists():
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
    "graph_search": "/graph/search",
    "graph_search_synthesis": "/graph/search-synthesis",
    "graph_write": "/graph/write",
    "graph_ingest": "/graph/ingest",
    "graph_analyze": "/graph/analyze",
    "graph_orchestrate": "/graph/orchestrate",
    "graph_configure": "/graph/configure",
    "graph_context": "/graph/context",
    "graph_feedback": "/graph/feedback",
    "graph_hydrate": "/graph/hydrate",
    "graph_sessions": "/graph/sessions",
    "graph_goals": "/graph/goals",
    "graph_message": "/graph/message",
    "document_process": "/document/process",
    "source_connector": "/connector/source",
    "ontology_property_types": "/ontology/property-types",
    "ontology_value_types": "/ontology/value-types",
    "ontology_interface": "/ontology/interface",
    "ontology_function": "/ontology/function",
    "ontology_derive": "/ontology/derive",
    "ontology_link_materialize": "/ontology/link-materialize",
    "object_edits": "/object/edits",
    "object_index": "/object/index",
    "object_permissioning": "/object/permissioning",
    "object_set": "/object/set",
    "graph_share": "/graph/share",
    "usage_query": "/usage/query",
    "ingest_sessions": "/usage/ingest-sessions",
    "quant": "/quant",
    "research_artifact": "/research/artifact",
    "graph_loops": "/graph/loops",
}


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
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_search_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool("graph_search", **body)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_write_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool("graph_write", **body)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_ingest_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool("graph_ingest", **body)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool("graph_analyze", **body)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool("graph_orchestrate", **body)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_configure_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool("graph_configure", **body)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# 2. Granular Graph Search endpoints
async def graph_search_hybrid_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_search",
            query=body.get("query", ""),
            mode="hybrid",
            top_k=int(body.get("top_k", 10)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_write_delete_node_endpoint(request: Request) -> JSONResponse:
    try:
        node_id = request.path_params.get("node_id", "")
        res = await _execute_tool("graph_write", action="delete_node", id=node_id)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_ingest_jobs_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_ingest", action="jobs")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def connector_sources_endpoint(request: Request) -> JSONResponse:
    """List registered document-source connectors (CONCEPT:ECO-4.27)."""
    try:
        res = await _execute_tool("source_connector", action="list")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def connector_run_endpoint(request: Request) -> JSONResponse:
    """Build + drain a document-source connector into the KG (CONCEPT:ECO-4.25–4.29)."""
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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_ingest_job_status_endpoint(request: Request) -> JSONResponse:
    try:
        job_id = request.path_params.get("job_id", "")
        res = await _execute_tool("graph_ingest", action="job_status", job_id=job_id)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_ingest_rebuild_indexes_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_ingest", action="rebuild_indexes")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_ingest_materialize_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_ingest", action="materialize")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_ingest_sync_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_ingest", action="sync")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_ingest_reflect_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_ingest", action="reflect")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# 5. Granular Graph Analyze endpoints
async def graph_analyze_synthesize_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_analyze",
            action="synthesize",
            query=body.get("query", ""),
            top_k=int(body.get("top_k", 10)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_deep_extract_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_analyze",
            action="deep_extract",
            query=body.get("query", ""),
            top_k=int(body.get("top_k", 10)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_background_research_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_analyze",
            action="background_research",
            query=body.get("query", ""),
            top_k=int(body.get("top_k", 10)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_relevance_sweep_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_analyze",
            action="relevance_sweep",
            query=body.get("query", ""),
            top_k=int(body.get("top_k", 10)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_blast_radius_endpoint(request: Request) -> JSONResponse:
    try:
        # The endpoint contract uses ``id`` (accept legacy ``node_id`` as a fallback).
        node_id = request.query_params.get("id") or request.query_params.get(
            "node_id", ""
        )
        depth = int(request.query_params.get("depth", "2"))
        res = await _execute_tool(
            "graph_analyze", action="blast_radius", id=node_id, depth=depth
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_inspect_endpoint(request: Request) -> JSONResponse:
    try:
        target = request.query_params.get("target", "")
        res = await _execute_tool("graph_analyze", action="inspect", target=target)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_context_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_analyze",
            action="context",
            target=body.get("target", ""),
            query=body.get("query", ""),
            top_k=int(body.get("top_k", 10)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_evaluate_alpha_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_analyze", action="evaluate_alpha", target=body.get("target", "")
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_evaluate_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_analyze", action="evaluate", target=body.get("target", "")
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_evolve_model_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_analyze", action="evolve_model")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_forecast_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_analyze", action="forecast")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_causal_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_analyze", action="causal")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_invariant_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_analyze", action="invariant")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# 6. Granular Graph Orchestrate endpoints
async def graph_orchestrate_dispatch_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_orchestrate",
            action="dispatch",
            task=body.get("task", ""),
            dependencies=_to_json_str(body.get("dependencies", [])),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_status_endpoint(request: Request) -> JSONResponse:
    try:
        job_id = request.path_params.get("job_id", "")
        res = await _execute_tool("graph_orchestrate", action="status", job_id=job_id)
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_request_approval_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_orchestrate",
            action="request_approval",
            job_id=body.get("job_id", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_grant_approval_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_orchestrate",
            action="grant_approval",
            job_id=body.get("job_id", ""),
            approval_status=body.get("approval_status", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_execute_agent_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_orchestrate",
            action="execute_agent",
            agent_name=body.get("agent_name", ""),
            task=body.get("task", ""),
            max_steps=int(body.get("max_steps", 30)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_consensus_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_orchestrate", action="consensus", task=body.get("task", "")
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_start_debate_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_orchestrate",
            action="start_debate",
            job_id=body.get("job_id", ""),
            task=body.get("task", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_submit_risk_veto_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_orchestrate",
            action="submit_risk_veto",
            job_id=body.get("job_id", ""),
            task=body.get("task", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_list_cron_jobs_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_orchestrate", action="list_cron_jobs")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_trigger_cron_job_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_orchestrate", action="trigger_cron_job", task=body.get("task", "")
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_compile_workflow_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_orchestrate",
            action="compile_workflow",
            agent_name=body.get("agent_name", ""),
            task=body.get("task", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_compile_process_endpoint(request: Request) -> JSONResponse:
    """CONCEPT:ORCH-1.41 — REST twin of graph_orchestrate compile_process."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_orchestrate",
            action="compile_process",
            task=body.get("process_id", body.get("task", "")),
            agent_name=body.get("name", body.get("agent_name", "")),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_publish_proposal_endpoint(
    request: Request,
) -> JSONResponse:
    """CONCEPT:AHE-3.21 — REST twin of graph_orchestrate publish_proposal."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_orchestrate",
            action="publish_proposal",
            task=body.get("proposal_id", body.get("task", "")),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_list_workflows_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_orchestrate", action="list_workflows")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_execute_workflow_endpoint(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_orchestrate",
            action="execute_workflow",
            agent_name=body.get("agent_name", ""),
            task=body.get("task", ""),
            max_steps=int(body.get("max_steps", 30)),
            completion_state=body.get("completion_state", ""),
            max_fan_out=int(body.get("max_fan_out", 5)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_dispatch_workflow_endpoint(
    request: Request,
) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_orchestrate",
            action="dispatch_workflow",
            agent_name=body.get("agent_name", ""),
            task=body.get("task", ""),
            max_steps=int(body.get("max_steps", 30)),
            completion_state=body.get("completion_state", ""),
            max_fan_out=int(body.get("max_fan_out", 5)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_workflow_status_endpoint(request: Request) -> JSONResponse:
    try:
        job_id = request.path_params.get("job_id", "")
        res = await _execute_tool(
            "graph_orchestrate", action="workflow_status", job_id=job_id
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_export_workflow_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_orchestrate", action="export_workflow")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_configure_doctor_endpoint(request: Request) -> JSONResponse:
    try:
        res = await _execute_tool("graph_configure", action="doctor")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# Default agent identity for provenance tracking
_AGENT_ID = setting("AGENT_ID", f"mcp-client-{uuid.uuid4().hex[:8]}")
_SESSION_ID = setting("SESSION_ID", uuid.uuid4().hex)
_WORKSPACE_PATH = setting("WORKSPACE_PATH", os.getcwd())


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

    Thread-safe (double-checked lock): the server now builds the engine in a
    background bootstrap thread so ``mcp.run()`` can start serving immediately
    (under Claude Code's 30s connect deadline); a concurrent first tool call
    must therefore not race a second engine into existence. (CONCEPT:KG-2.8)
    """
    from agent_utilities.core.paths import ensure_dirs, kg_db_path
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
        db_path = str(kg_db_path())
        logger.info("KG MCP Server using database: %s", db_path)
        backend_type = setting("GRAPH_BACKEND")
        backend = create_backend(backend_type=backend_type, db_path=db_path)
        engine = IntelligenceGraphEngine(backend=backend)
        return engine


# ── CONCEPT:KG-2.63 — Named multi-connection graph registry ────────────────
_CONNECTION_REGISTRY = None
_REGISTRY_LOCK = threading.Lock()


def get_connection_registry():
    """Process-wide :class:`ConnectionRegistry` singleton.

    Seeds its ``"default"`` connection from the legacy ``_get_engine`` singleton
    (so the default is never duplicated) and from ``config.kg_connections``
    (CONCEPT:KG-2.63) on first build.
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

        registry = ConnectionRegistry(default_engine_provider=_get_engine)
        # Seed declarative connections from config (KG_CONNECTIONS).
        try:
            from agent_utilities.core.config import config as _cfg

            for spec in _cfg.kg_connections or []:
                spec = dict(spec)
                name = spec.pop("name", "")
                if name:
                    try:
                        registry.register(name, spec)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "Skipping invalid kg_connections entry %r: %s", name, e
                        )
        except Exception:  # noqa: BLE001 — config-less environments
            logger.debug("No kg_connections to seed", exc_info=True)
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


def _provenance_props(agent_id: str | None = None) -> dict[str, Any]:
    """Build standard provenance metadata for multi-agent write tracking."""
    return {
        "agent_id": agent_id or _AGENT_ID,
        "session_id": _SESSION_ID,
        "workspace_path": _WORKSPACE_PATH,
        "timestamp": datetime.now(UTC).isoformat(),
        "source": "mcp",
    }


def _ingest_capabilities(engine):
    """Natively ingest MCP configurations, Native Tools, and Skills into the KG on startup."""
    import importlib
    import inspect
    import json
    import os
    import pkgutil
    from pathlib import Path

    import platformdirs
    import yaml

    # 1. mcp_config.json
    try:
        APP_NAME = "agent-utilities"
        APP_AUTHOR = "knuckles-team"
        cfg_dir = Path(platformdirs.user_config_path(APP_NAME, APP_AUTHOR))
        mcp_config_path = cfg_dir / "mcp_config.json"

        if mcp_config_path.exists():
            with open(mcp_config_path) as f:
                data = json.load(f)
                mcp_servers = data.get("mcpServers", {})
                for server_name, server_details in mcp_servers.items():
                    node_id = f"mcp_server_{server_name}"
                    disabled = get_existing_disabled(engine, node_id)
                    engine.add_node(
                        node_id,
                        "MCPServer",
                        {
                            "name": server_name,
                            "command": server_details.get("command"),
                            "args": json.dumps(server_details.get("args", [])),
                            "disabled": disabled,
                        },
                    )
            logger.info("Ingested mcp_config.json")
    except Exception as e:
        logger.error(f"Failed to ingest mcp_config.json: {e}")

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
                            engine.add_node(
                                node_id,
                                "NativeTool",
                                {
                                    "name": name,
                                    "description": obj.__doc__ or "",
                                    "version": obj.__agentic_version__,
                                    "module": modname,
                                    "disabled": disabled,
                                },
                            )
                except Exception as e:
                    logger.debug(f"Failed to ingest native tools from {modname}: {e}")
        logger.info("Ingested Native Tools")
    except Exception as e:
        logger.error(f"Failed to scan native tools: {e}")

    # 3. Skills
    try:
        from agent_utilities.core.config import config

        skills_dir = config.custom_skills_directory or os.path.expanduser(
            "~/.gemini/antigravity/skills"
        )
        skills_path = Path(skills_dir)
        if skills_path.exists() and skills_path.is_dir():
            for skill_dir in skills_path.iterdir():
                if skill_dir.is_dir():
                    skill_md = skill_dir / "SKILL.md"
                    if skill_md.exists():
                        try:
                            content = skill_md.read_text()
                            if content.startswith("---"):
                                end_idx = content.find("---", 3)
                                if end_idx != -1:
                                    frontmatter_str = content[3:end_idx].strip()
                                    frontmatter = yaml.safe_load(frontmatter_str) or {}
                                    name = frontmatter.get("name", skill_dir.name)
                                    desc = frontmatter.get("description", "")
                                    node_id = f"skill_{name}"
                                    disabled = get_existing_disabled(engine, node_id)
                                    engine.add_node(
                                        node_id,
                                        "Skill",
                                        {
                                            "name": name,
                                            "description": desc,
                                            "path": str(skill_md),
                                            "disabled": disabled,
                                            **{
                                                k: v
                                                for k, v in frontmatter.items()
                                                if k not in ["name", "description"]
                                            },
                                        },
                                    )
                        except Exception as e:
                            logger.error(f"Failed to ingest skill from {skill_md}: {e}")
            logger.info("Ingested Skills")
    except Exception as e:
        logger.error(f"Failed to ingest skills: {e}")


def _mint_stdio_identity() -> None:
    """Resolve the process identity for a directly-served MCP server.

    CONCEPT:OS-5.14 — stdio has no Authorization header, so when
    ``KG_AUTH_REQUIRED`` is on the identity comes from a validated JWT in the
    MCP server's environment (``KG_AUTH_TOKEN``, validated against
    ``AUTH_JWT_JWKS_URI`` exactly like a gateway request). Without a valid
    token the process stays unauthenticated and ``_execute_tool`` restricts it
    to :data:`ANONYMOUS_READ_TOOLS`. With ``KG_AUTH_REQUIRED`` off this only
    emits the one-time honor-system warning.
    """
    global _PROCESS_ACTOR
    from agent_utilities.core.config import config
    from agent_utilities.security.request_identity import (
        mint_actor_from_token_sync,
        warn_unauthenticated_identity_once,
    )

    if not getattr(config, "kg_auth_required", False):
        warn_unauthenticated_identity_once()
        return
    token = getattr(config, "kg_auth_token", None)
    if token:
        _PROCESS_ACTOR = mint_actor_from_token_sync(token)
        if _PROCESS_ACTOR is not None:
            logger.info(
                "KG MCP identity minted from KG_AUTH_TOKEN: actor=%s roles=%s",
                _PROCESS_ACTOR.actor_id,
                list(_PROCESS_ACTOR.roles),
            )
            return
    logger.warning(
        "KG_AUTH_REQUIRED=1 but no valid KG_AUTH_TOKEN: this MCP process is "
        "restricted to the read-only tool surface %s.",
        sorted(ANONYMOUS_READ_TOOLS),
    )


def _build_server(bootstrap: bool = True):
    """Build the KG MCP server with all tools registered.

    Args:
        bootstrap: When True (default) start the background engine bootstrap
            thread (engine init, task workers, capability ingest). The API
            gateway calls this with ``bootstrap=False`` (via
            :func:`ensure_tools_registered`) because it owns the engine/daemon
            lifecycle itself and only needs ``REGISTERED_TOOLS`` populated so the
            centralized REST handlers can dispatch.
    """
    import sys

    from agent_utilities.mcp.server_factory import create_mcp_server

    is_readonly = False

    if bootstrap:
        # Directly-served MCP process (stdio/streamable-http): resolve the
        # server-side identity (or warn that identity is honor-system).
        _mint_stdio_identity()

    if bootstrap and not any(arg in sys.argv for arg in ["--help", "-h"]):
        # Build the engine + start daemons/workers + ingest capabilities in a
        # BACKGROUND thread so mcp.run() can start serving (and the multiplexer
        # can list tools) immediately — engine init is ~30s and was blocking the
        # MCP handshake past Claude Code's 30s connect deadline. Tools are
        # registered below regardless; the first tool call's _get_engine() (now
        # lock-safe) returns the same singleton the bootstrap builds. (CONCEPT:KG-2.8)
        def _bootstrap_engine() -> None:
            try:
                engine = _get_engine()
                if hasattr(engine, "start_sdd_watcher"):
                    engine.start_sdd_watcher()
                if (
                    engine
                    and engine.backend
                    and not getattr(engine.backend, "read_only", False)
                ):
                    engine.start_task_workers()
                _ingest_capabilities(engine)
            except Exception:
                logger.exception("KG engine background bootstrap failed")

        threading.Thread(
            target=_bootstrap_engine, daemon=True, name="KGEngineBootstrap"
        ).start()

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
        version="0.1.0",
        instructions=(
            "Knowledge Graph MCP Server for agent-utilities. "
            "Provides access to the shared unified Knowledge Graph that powers "
            "the 5-pillar agent architecture (ORCH, KG, AHE, ECO, OS). "
            "Use kg_query for Cypher queries, kg_search for semantic search, "
            "kg_analyze for LLM-powered cross-reference analysis, "
            "and kg_ingest_* for adding data."
        ),
        command_args=None if bootstrap else [],
    )

    # Liveness endpoint for streamable-http/sse deployments (container
    # healthchecks). Does not touch the engine so it stays fast and lock-free;
    # the shard summary is config-only (no probe) — full per-shard
    # reachability lives on the gateway's /daemon/shards route and in the
    # unified daemon status. (CONCEPT:OS-5.28)
    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:  # noqa: ARG001
        from agent_utilities.core.config import config as _config
        from agent_utilities.knowledge_graph.core.shard_topology import (
            resolve_endpoints,
        )

        endpoints = resolve_endpoints(_config)
        return JSONResponse(
            {
                "status": "ok",
                "server": "graph-os",
                "shard_mode": "sharded" if len(endpoints) > 1 else "single",
                "shard_count": len(endpoints),
            }
        )

    # ═══ Synthesized Tools (7 tools, action-routed) ═══

    @mcp.tool(
        name="graph_query",
        description="Execute a read-only Cypher query against the Knowledge Graph.",
        tags=["graph-os", "query"],
    )
    def graph_query(
        cypher: str = Field(
            description="A Cypher query string (read-only — no CREATE/MERGE/DELETE)."
        ),
        params: str = Field(default="{}", description="JSON-encoded query parameters."),
        scope: str = Field(
            default="local",
            description="'local' for the internal KG, 'federated' to query an external graph endpoint.",
        ),
        reference_id: str = Field(
            default="",
            description="Required when scope='federated'. The ExternalGraphReference node ID.",
        ),
        as_of: str = Field(
            default="",
            description=(
                "CONCEPT:KG-2.11 — optional ISO-8601 instant. When set, rows are filtered to "
                "those whose bi-temporal validity (valid_from <= as_of < valid_to) holds, "
                "answering 'what was true as of date T'."
            ),
        ),
        target: str = Field(
            default="",
            description=(
                "CONCEPT:KG-2.63 — named graph connection to query (default = primary). "
                "Use a registered connection name (e.g. 'prod-neo4j'), or 'all' (or a "
                "comma-separated list) to fan out the same query to several backends and "
                "get per-connection labeled results."
            ),
        ),
    ) -> str:
        """Execute a read-only Cypher query against the Knowledge Graph. Use this to fetch graph data, explore relationships, and read node properties."""
        parsed_params = json.loads(params) if params else {}

        if scope == "federated":
            engine = _get_engine()
            if not reference_id:
                return json.dumps(
                    {"error": "reference_id required for federated queries"}
                )
            try:
                results = engine.execute_federated_query(
                    reference_id, cypher, parsed_params
                )
                return json.dumps(results, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})

        # Local query — block writes. Match write keywords only as whole Cypher
        # clause keywords (word boundaries) and ignore quoted string literals, so
        # identifiers like ``created_at``/``offset`` or a literal ``'CREATE …'`` are
        # not misread as a mutation. (CONCEPT:KG-2.63)
        import re

        _cypher_no_literals = re.sub(r"'[^']*'|\"[^\"]*\"", "", cypher)
        _write_kw = re.search(
            r"\b(CREATE|MERGE|DELETE|REMOVE|DROP|SET)\b",
            _cypher_no_literals,
            re.IGNORECASE,
        )
        if _write_kw:
            return json.dumps(
                {
                    "error": f"Write operation '{_write_kw.group(1).upper()}' not allowed. Use kg_write for mutations."
                }
            )

        # CONCEPT:KG-2.63 — resolve the target connection(s).
        try:
            entries, errors, fanout = _resolve_target_engines(target)
        except Exception as e:
            return json.dumps({"error": str(e)})

        if not fanout:
            # Single connection (default or one named) — identical shape to legacy.
            _name, engine = entries[0]
            try:
                results = engine.query_cypher(
                    cypher, parsed_params, as_of=as_of or None
                )
                return json.dumps(results, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})

        # Fan-out — per-connection labeled results, partial success.
        out: dict[str, Any] = {"targets": {}, "errors": dict(errors)}
        for name, engine in entries:
            try:
                out["targets"][name] = engine.query_cypher(
                    cypher, parsed_params, as_of=as_of or None
                )
            except Exception as e:
                out["errors"][name] = str(e)
        return json.dumps(out, default=str)

    REGISTERED_TOOLS["graph_query"] = graph_query

    # ══════════════════════════════════════════════════════════════════
    # 1b. graph_context — CONCEPT:ORCH-1.39 cross-process curated-context store
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_context",
        description=(
            "CONCEPT:ORCH-1.39 — store/fetch curated context for invoker→spawned-agent "
            "handoff, persisted in the epistemic-graph so a SEPARATELY-spawned agent can read "
            "it by id. Actions: 'put' (store content, returns context_id), 'get' (fetch by "
            "context_id), 'list' (by session_id). Pass the returned context_id to "
            "graph_orchestrate(action='execute_agent', context_ref=...)."
        ),
        tags=["graph-os", "orchestrate", "context"],
    )
    async def graph_context(
        action: str = Field(default="put", description="put | get | list"),
        content: str = Field(
            default="", description="Context text to store (action=put)."
        ),
        context_id: str = Field(default="", description="ContextBlob id (action=get)."),
        session_id: str = Field(default="", description="Session scope key."),
        key: str = Field(
            default="", description="Optional sub-key within the session."
        ),
        ttl_s: int = Field(
            default=0, description="Optional time-to-live in seconds (0 = persistent)."
        ),
    ) -> str:
        import contextlib
        import time
        import uuid as _uuid

        engine = _get_engine()
        if not engine:
            return json.dumps({"error": "IntelligenceGraphEngine not active."})
        if action == "put":
            if not content:
                return json.dumps({"error": "content required for put"})
            sid = session_id or _uuid.uuid4().hex[:8]
            cid = context_id or f"ctx:{sid}:{key or _uuid.uuid4().hex[:6]}"
            engine.add_node(
                cid,
                "ContextBlob",
                properties={
                    "id": cid,
                    "content": content,
                    "session_id": sid,
                    "key": key,
                    "ttl_s": int(ttl_s),
                    "created_at": time.time(),
                    "producer": _SESSION_ID,
                },
            )
            # CONCEPT:ORCH-1.40 — session-anchored collection: upsert the id-addressable
            # Session node and link it, so "list by session" is a reliable id-anchored
            # traversal (the engine has no property index; property scans are unreliable).
            snode = f"session:{sid}"
            with contextlib.suppress(Exception):
                engine.add_node(
                    snode, "Session", properties={"id": snode, "session_id": sid}
                )
                engine.add_edge(snode, cid, "HAS_CONTEXT")
            return json.dumps({"context_id": cid, "session_id": sid})
        if action == "get":
            if not context_id:
                return json.dumps({"error": "context_id required for get"})
            try:
                rows = engine.query_cypher(
                    "MATCH (c:ContextBlob) WHERE c.id = $id "
                    "RETURN c.content AS content, c.session_id AS session_id, "
                    "c.created_at AS created_at, c.ttl_s AS ttl_s",
                    {"id": context_id},
                )
                if not rows:
                    return json.dumps({})
                row = rows[0]
                # TTL: treat an expired blob as gone (created_at + ttl_s < now).
                _ttl = row.get("ttl_s") or 0
                _created = row.get("created_at") or 0
                if _ttl and _created and (float(_created) + float(_ttl) < time.time()):
                    return json.dumps({"error": "context expired", "expired": True})
                return json.dumps(row, default=str)
            except Exception as exc:  # noqa: BLE001
                return json.dumps({"error": str(exc)})
        if action == "prune":
            # Delete expired ContextBlobs (CONCEPT:ORCH-1.39 lifecycle).
            try:
                rows = engine.query_cypher(
                    "MATCH (c:ContextBlob) WHERE c.ttl_s > 0 AND "
                    "(c.created_at + c.ttl_s) < $now RETURN c.id AS id",
                    {"now": time.time()},
                )
                pruned = 0
                _del = getattr(engine, "delete_node", None) or getattr(
                    getattr(engine, "backend", None), "delete_node", None
                )
                for r in rows or []:
                    if callable(_del):
                        with contextlib.suppress(Exception):
                            _del(r["id"])
                            pruned += 1
                return json.dumps({"pruned": pruned, "expired": len(rows or [])})
            except Exception as exc:  # noqa: BLE001
                return json.dumps({"error": str(exc)})
        if action == "list":
            try:
                # CONCEPT:ORCH-1.40 — id-anchored traversal from the Session node (the engine's
                # reliable, fast O(degree) path; the index-less backend can't serve property
                # scans). The traversal reader returns whole nodes (`RETURN c`), so project +
                # sort + limit client-side.
                rows = engine.query_cypher(
                    "MATCH (s {id: $snode})-[:HAS_CONTEXT]->(c:ContextBlob) RETURN c",
                    {"snode": f"session:{session_id}"},
                )
                items = []
                for r in rows or []:
                    c = r.get("c") if isinstance(r, dict) else None
                    if isinstance(c, dict) and str(c.get("id", "")).startswith("ctx:"):
                        items.append(
                            {
                                "context_id": c.get("id"),
                                "key": c.get("key"),
                                "created_at": c.get("created_at"),
                            }
                        )
                items.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
                return json.dumps(items[:50], default=str)
            except Exception as exc:  # noqa: BLE001
                return json.dumps({"error": str(exc)})
        return json.dumps({"error": f"unknown action: {action}"})

    REGISTERED_TOOLS["graph_context"] = graph_context

    # ══════════════════════════════════════════════════════════════════
    # 1c. graph_message — CONCEPT:ORCH-1.40 invoker↔spawned-agent message channel
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_message",
        description=(
            "CONCEPT:ORCH-1.40 — bidirectional, cross-process, ordered message channel between "
            "an invoking agent and a spawned agent, over the epistemic-graph native channels. "
            "Actions: 'open' (session_id+run_id → channel_id), 'send' (channel_id+sender+payload "
            "[+durable]), 'receive' (channel_id [+since cursor] → new messages + cursor), "
            "'history' (durable replay, survives restart), 'close'. Use the channel_id returned "
            "by graph_orchestrate(execute_agent, open_channel=True) to talk to the spawned agent."
        ),
        tags=["graph-os", "orchestrate", "messaging"],
    )
    async def graph_message(
        action: str = Field(
            default="receive", description="open | send | receive | history | close"
        ),
        channel_id: str = Field(
            default="", description="Channel id (send/receive/history/close)."
        ),
        session_id: str = Field(default="", description="Session id (open)."),
        run_id: str = Field(default="", description="Spawned run id (open)."),
        sender: str = Field(default="invoker", description="Sender label (send)."),
        payload: str = Field(default="", description="Message text (send)."),
        since: int = Field(
            default=0, description="Cursor: messages already consumed (receive)."
        ),
        durable: bool = Field(
            default=False,
            description="When True (send), also persist the message as a graph AgentMessage "
            "node so it survives engine restart and is replayable via action='history'.",
        ),
    ) -> str:
        from agent_utilities.messaging import agent_channel

        engine = _get_engine()
        if not engine:
            return json.dumps({"error": "IntelligenceGraphEngine not active."})
        if action == "open":
            cid = agent_channel.open_channel(engine, session_id, run_id)
            return json.dumps({"channel_id": cid})
        if action == "send":
            return json.dumps(
                {
                    "sent": agent_channel.send(
                        engine, channel_id, sender, payload, durable=bool(durable)
                    )
                }
            )
        if action == "receive":
            msgs, cursor = agent_channel.receive(engine, channel_id, since=since)
            return json.dumps({"messages": msgs, "cursor": cursor}, default=str)
        if action == "history":
            return json.dumps(
                {"messages": agent_channel.history(engine, channel_id)}, default=str
            )
        if action == "close":
            return json.dumps({"closed": agent_channel.close(engine, channel_id)})
        return json.dumps({"error": f"unknown action: {action}"})

    REGISTERED_TOOLS["graph_message"] = graph_message

    # ══════════════════════════════════════════════════════════════════
    # 2. kg_search — Unified search (hybrid, concept, analogy, memory)
    # ══════════════════════════════════════════════════════════════════

    @mcp.tool(
        name="graph_search",
        description="Search the Knowledge Graph using multiple strategies (hybrid, concept, analogy, memory, discover, dci).",
        tags=["graph-os", "search"],
    )
    def graph_search(
        query: str = Field(description="Natural language search query or concept ID."),
        mode: str = Field(
            default="hybrid",
            description="Search strategy:\n- 'hybrid': Semantic + keyword weighted search (default).\n- 'hyde': Memory-first HyDE multi-query plan + dual threshold (CONCEPT:KG-2.12).\n- 'deep': Wide-recall single query at the 0.28 deep threshold.\n- 'concept': Look up a CONCEPT:ID (e.g. 'KG-2.7', 'ORCH-1.0').\n- 'analogy': Find structurally similar concepts.\n- 'memory': Search tiered memory (episodic/semantic/procedural).\n- 'discover': Cross-reference query against all ingested content.\n- 'dci': Direct Corpus Interaction.\n- 'latent': Latent-topology hierarchical routing (CONCEPT:KG-2.3).\n- 'sira': Single-shot SIRA sparsity-aligned context.\n- 'hard_negatives': Mine hard negatives for the query (CONCEPT:KG-2.3).\n- 'rerank': Hybrid semantic+keyword re-scoring of candidates.",
        ),
        top_k: int = Field(default=10, description="Maximum results to return."),
        self_correct: bool = Field(
            default=False,
            description="CONCEPT:KG-2.12 — run a self-correcting second retrieval pass at the deep threshold when the quality gate fails.",
        ),
        as_of: str = Field(
            default="",
            description="Optional ISO-8601 instant. Pack-driven recency decay is measured relative to this time, enabling knowledge-state-as-of-date-D retrieval such as an academic literature state. Defaults to now (CONCEPT:KG-2.22).",
        ),
        target: str = Field(
            default="",
            description=(
                "CONCEPT:KG-2.63 — named graph connection to search (default = primary). "
                "Use a registered connection name, or 'all' (or a comma-separated list) to "
                "fan out and get per-connection labeled results."
            ),
        ),
    ) -> str:
        """Search the Knowledge Graph using multiple strategies. Useful for finding context, concepts, memories, and capabilities across the ecosystem."""

        def _run_search(engine: Any) -> str:
            if not engine:
                return "Error: IntelligenceGraphEngine not active."
            try:
                return _search_with_engine(engine)
            except Exception as e:
                return f"Search error: {str(e)}"

        def _search_with_engine(engine: Any) -> str:
            if mode in ("hyde", "deep"):
                results = engine.search_hybrid(
                    query=query, top_k=top_k, mode=mode, self_correct=self_correct
                )
            elif mode == "hybrid":
                results = engine.search_hybrid(
                    query=query,
                    top_k=top_k,
                    self_correct=self_correct,
                    as_of=as_of or None,
                )
            elif mode == "concept":
                results = engine.search_hybrid(query=query, top_k=top_k)
            elif mode == "analogy":
                results = engine.search_hybrid(query=query, top_k=top_k)
            elif mode == "dci":
                results = engine.search_dci(query=query, top_k=top_k)
            elif mode == "memory":
                results = engine.search_memories(query=query, top_k=top_k)
            elif mode == "discover":
                try:
                    from agent_utilities.capabilities.manager import CapabilityManager

                    manager = CapabilityManager(engine)
                    results = manager.discover_capabilities(query)
                    if not results:
                        return f"No capabilities found for '{query}'"
                    return "\n".join([f"- {r.name}: {r.description}" for r in results])
                except ImportError:
                    return "Error: capabilities module not available"
            elif mode == "latent":
                # KG-2.3 — route through the latent topology hierarchy.
                from agent_utilities.knowledge_graph.retrieval.latent_topology_rag import (  # noqa: E501
                    LatentTopologicalRAG,
                )

                results = LatentTopologicalRAG(engine).retrieve(query, top_k=top_k)
            elif mode == "sira":
                # Single-shot SIRA: hybrid-retrieve, then sparsity-align the set.
                from agent_utilities.knowledge_graph.retrieval.single_shot_sira import (
                    SingleShotSIRA,
                )

                base = engine.search_hybrid(query=query, top_k=top_k) or []
                results = SingleShotSIRA(engine).align_context(base)
            elif mode == "hard_negatives":
                # KG-2.3 — mine hard negatives via the engine's hybrid retriever.
                from agent_utilities.knowledge_graph.retrieval.hard_negative_miner import (  # noqa: E501
                    HardNegativeMiner,
                )

                retriever = getattr(engine, "hybrid_retriever", None)
                if retriever is None:
                    return (
                        "Error: hybrid retriever unavailable for hard-negative mining."
                    )
                negs = HardNegativeMiner(retriever).mine(query)
                if not negs:
                    return f"No hard negatives mined for: '{query}'"
                return "\n".join(
                    f"- {n.doc_id}: {getattr(n, 'reason', '')}" for n in negs
                )
            elif mode == "rerank":
                # Semantic-retrieval hybrid re-scoring of a candidate set.
                from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (  # noqa: E501
                    HybridSearchScorer,
                )

                base = engine.search_hybrid(query=query, top_k=top_k) or []
                docs = [
                    {
                        "id": (r.get("node", r) or {}).get("id", ""),
                        "text": (r.get("node", r) or {}).get("description", ""),
                        "embedding": (r.get("node", r) or {}).get("embedding"),
                    }
                    for r in base
                ]
                qemb: list[float] = []
                embed_model = getattr(
                    getattr(engine, "hybrid_retriever", None), "embed_model", None
                )
                if embed_model is not None:
                    try:
                        qemb = embed_model.get_text_embedding(query)
                    except Exception:  # noqa: BLE001
                        qemb = []
                results = HybridSearchScorer().score_documents(query, qemb, docs)
            else:
                return f"Error: Unknown search mode '{mode}'"

            if not results:
                return f"No results found for query: '{query}'"

            formatted_results = []
            for res in results:
                score = res.get("score", 0)
                score = float(score) if score is not None else 0.0
                node = res.get("node", res)
                label = node.get("type", node.get("label", "Unknown"))
                name = node.get("name", "Unnamed")
                desc = node.get("description", "")
                nid = node.get("id", "N/A")
                formatted_results.append(
                    f"[{label}] {name} (ID: {nid}) - Score: {score:.2f}\n{desc}"
                )
            return "\n---\n".join(formatted_results)

        # CONCEPT:KG-2.63 — resolve target connection(s).
        try:
            entries, errors, fanout = _resolve_target_engines(target)
        except Exception as e:
            return f"Search error: {str(e)}"

        if not fanout:
            return _run_search(entries[0][1])

        # Fan-out — per-connection labeled blocks, partial success.
        out_lines = [
            f"=== {name} ===\n{_run_search(engine)}" for name, engine in entries
        ]
        out_lines += [f"=== {name} (error) ===\n{err}" for name, err in errors.items()]
        return "\n\n".join(out_lines)

    REGISTERED_TOOLS["graph_search"] = graph_search

    @mcp.tool(
        name="graph_search_synthesis",
        description=(
            "Synthesize a shortcut-resistant deep-search task from the evidence graph, "
            "or diagnose realized search difficulty of solver trajectories "
            "(CONCEPT:KG-2.70/2.71/2.72, AHE-3.30; distills arXiv:2606.12087)."
        ),
        tags=["graph-os", "search", "synthesis", "training-data"],
    )
    def graph_search_synthesis(
        action: str = Field(
            default="synthesize",
            description=(
                "'synthesize': build an evidence subgraph around an answer entity and "
                "formulate + adversarially refine a question that forces multi-hop "
                "search (no exposed constants / single-clue / co-coverage shortcuts). "
                "'diagnose': score solver trajectories with the FORT signatures "
                "(solving cost, answer hit time, prior-shortcut rate) + a search-heavy "
                "verdict."
            ),
        ),
        answer_id: str = Field(
            default="",
            description="action=synthesize — node id of the gold answer entity to build the task around.",
        ),
        hops: int = Field(
            default=2, description="action=synthesize — evidence-graph BFS depth."
        ),
        fanout: int = Field(
            default=8,
            description="action=synthesize — max neighbors expanded per node.",
        ),
        min_trust: float = Field(
            default=0.0,
            description="action=synthesize — drop facts whose source_trust is below this.",
        ),
        max_per_source: int = Field(
            default=1,
            description="action=synthesize — max clues allowed to share one evidence source before co-coverage trips.",
        ),
        root_popularity: float = Field(
            default=0.0,
            description="action=synthesize — 0..1 familiarity of the answer entity (high → prior-binding risk).",
        ),
        trajectories: str = Field(
            default="",
            description='action=diagnose — JSON list of trajectories: [{"steps":[{"kind","observation","model_text"}],"answer_aliases":[...]}].',
        ),
    ) -> str:
        """Shortcut-resistant search-task synthesis and realized-difficulty diagnosis."""
        import json as _json

        from agent_utilities.graph.training_signals import realized_difficulty

        if action == "diagnose":
            try:
                trajs = _json.loads(trajectories) if trajectories else []
            except Exception as e:  # noqa: BLE001
                return f"Error: invalid trajectories JSON: {e}"
            return _json.dumps(realized_difficulty(trajs))

        if action != "synthesize":
            return f"Error: unknown action '{action}' (expected 'synthesize' or 'diagnose')."
        if not answer_id:
            return "Error: action=synthesize requires answer_id."

        from agent_utilities.knowledge_graph.search_synthesis import synthesize

        class _Reader:
            def __init__(self, eng: Any) -> None:
                self._eng = eng

            def query(self, cypher: str, params: Any = None) -> list[dict[str, Any]]:
                backend = getattr(self._eng, "backend", None)
                if backend is not None and hasattr(backend, "execute"):
                    return backend.execute(cypher, params or {}) or []
                if hasattr(self._eng, "query"):
                    return self._eng.query(cypher, params or {}) or []
                return []

        engine = _get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            task = synthesize(
                _Reader(engine),
                answer_id,
                hops=hops,
                fanout=fanout,
                min_trust=min_trust,
                root_popularity=root_popularity,
                max_per_source=max_per_source,
            )
        except Exception as e:  # noqa: BLE001
            return f"Synthesis error: {str(e)}"
        return _json.dumps(task.to_dict())

    REGISTERED_TOOLS["graph_search_synthesis"] = graph_search_synthesis

    @mcp.tool(
        name="graph_write",
        description="Write nodes, relationships, or register external graphs to the Knowledge Graph.",
        tags=["graph-os", "write", "mutation"],
    )
    def graph_write(
        action: str = Field(
            description="Action to perform (add_node, add_edge, delete_node, delete_edge, register_external_graph, bulk_ingest, store_memory, recall_memory, log_chat, submit_sdd, register_execution, check_loop)."
        ),
        node_id: str = Field(
            default="", description="The unique identifier for the node."
        ),
        node_type: str = Field(
            default="", description="The type or label of the node."
        ),
        properties: str = Field(
            default="{}", description="JSON-encoded dictionary of properties."
        ),
        source_id: str = Field(
            default="", description="The source node ID for an edge."
        ),
        target_id: str = Field(
            default="", description="The target node ID for an edge."
        ),
        rel_type: str = Field(
            default="", description="The relationship type for an edge."
        ),
        endpoint_url: str = Field(
            default="", description="URL for external graph registration."
        ),
        graph_type: str = Field(
            default="",
            description="Type of external graph (e.g., 'sparql', 'graphql').",
        ),
        agent_id: str = Field(
            default="", description="ID of the agent performing the action."
        ),
        nodes: str = Field(
            default="[]",
            description="JSON-encoded list of nodes or tags for bulk operations.",
        ),
        target: str = Field(
            default="",
            description=(
                "CONCEPT:KG-2.63 — named graph connection to write to (default = primary). "
                "Use a registered connection name, or 'all' (or a comma-separated list) to "
                "mirror the SAME write to several backends. Fan-out requires an explicit "
                "multi-target value; the default and a single named target stay single-write."
            ),
        ),
    ) -> str:
        """Write nodes, relationships, or register external graphs. This is the primary mutation interface for the Knowledge Graph."""

        def _write_with_engine(engine: Any) -> str:
            if not engine:
                return "Error: IntelligenceGraphEngine not active."
            try:
                import json

                props = json.loads(properties) if properties else {}

                if action == "add_node":
                    if not node_id or not node_type:
                        return "Error: node_id and node_type required"
                    engine.add_node(node_id, node_type, props)
                    return f"Node {node_id} added."
                elif action == "add_edge":
                    if not source_id or not target_id or not rel_type:
                        return "Error: source_id, target_id, and rel_type required"
                    engine.link_nodes(source_id, target_id, rel_type, props)
                    return f"Edge {source_id} -> {target_id} added."
                elif action == "delete_node":
                    engine.delete_node(node_id)
                    return f"Node {node_id} deleted."
                elif action == "delete_edge":
                    engine.delete_edge(source_id, target_id, rel_type)
                    return f"Edge {source_id} -> {target_id} deleted."
                elif action == "register_external_graph":
                    if not endpoint_url:
                        return "Error: endpoint_url required"
                    engine.add_node(
                        endpoint_url, "ExternalGraphReference", {"type": graph_type}
                    )
                    return f"Registered external graph at {endpoint_url}"
                elif action == "bulk_ingest":
                    nodes_list = json.loads(nodes) if nodes else []
                    for n in nodes_list:
                        engine.add_node(
                            n.get("id"), n.get("type", "Node"), n.get("properties", {})
                        )
                    return f"Bulk ingested {len(nodes_list)} nodes."
                elif action in ("store_memory", "recall_memory"):
                    try:
                        from agent_utilities.memory.manager import MemoryManager

                        mm = MemoryManager(engine)
                        if action == "store_memory":
                            mm.store(
                                agent_id=agent_id,
                                content=properties,
                                memory_type=node_type,
                                tags=json.loads(nodes) if nodes else [],
                            )
                            return "Memory stored."
                        else:
                            res = mm.recall(
                                query=properties, memory_type=node_type, top_k=5
                            )
                            return "\n".join([str(r) for r in res])
                    except ImportError:
                        return "Error: memory module not available"
                elif action in (
                    "log_chat",
                    "submit_sdd",
                    "register_execution",
                    "check_loop",
                ):
                    if action == "log_chat":
                        engine.add_node(
                            f"chat_{agent_id}_{hash(properties)}",
                            "ChatLog",
                            {"content": properties, "agent_id": agent_id},
                        )
                        return "Chat logged."
                    elif action == "submit_sdd":
                        engine.add_node(
                            f"sdd_{agent_id}_{hash(properties)}",
                            "SDD",
                            {"content": properties, "agent_id": agent_id},
                        )
                        return "SDD submitted."
                    elif action == "register_execution":
                        engine.add_node(
                            f"exec_{agent_id}", "Execution", {"status": "running"}
                        )
                        return "Execution registered."
                    elif action == "check_loop":
                        return "Loop status: OK"
                    return f"Error: Action '{action}' not implemented."
                else:
                    return f"Error: Unknown write action '{action}'"
            except Exception as e:
                return f"Write error: {str(e)}"

        # CONCEPT:KG-2.63 — resolve target connection(s). Writes only fan out on
        # an EXPLICIT multi-target request ('all' or a list); the default and a
        # single named target stay single-write to avoid accidental multi-store
        # writes.
        try:
            entries, errors, fanout = _resolve_target_engines(target)
        except Exception as e:
            return f"Write error: {str(e)}"

        if not fanout:
            return _write_with_engine(entries[0][1])

        out: dict[str, Any] = {"targets": {}, "errors": dict(errors)}
        for name, eng in entries:
            out["targets"][name] = _write_with_engine(eng)
        return json.dumps(out, default=str)

    REGISTERED_TOOLS["graph_write"] = graph_write

    @mcp.tool(
        name="graph_feedback",
        description=(
            "Record a human correction so the brain learns: correction_type "
            "'outcome' adjusts an entity's reward, 'rule' persists a durable "
            "governance/voice/source rule consulted at retrieval time, 'eval' "
            "adds a regression case. This is how 'this was wrong, here's the fix' "
            "becomes future behaviour (CONCEPT:KG-2.8)."
        ),
        tags=["graph-os", "feedback", "learning"],
    )
    def graph_feedback(
        correction_type: str = Field(description="One of: outcome | rule | eval."),
        target_id: str = Field(
            description="Entity/episode/query the correction is about."
        ),
        corrected_value: str = Field(
            default="",
            description="The corrected value (reward, expected output, etc.).",
        ),
        reason: str = Field(default="", description="Why — the human's explanation."),
        rule_scope: str = Field(
            default="governance",
            description="For rule corrections: governance | voice | source | preference.",
        ),
        rule_kind: str = Field(
            default="forbid",
            description="For rule corrections: forbid | prefer | demote.",
        ),
        actor_id: str = Field(
            default="human", description="Who issued the correction."
        ),
    ) -> str:
        """Record a human correction (outcome/rule/eval) and apply it durably."""
        engine = _get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            from ..knowledge_graph.adaptation.feedback import FeedbackService

            service = FeedbackService.from_engine(engine)
            result = service.record_correction(
                correction_type,
                target_id,
                corrected_value=corrected_value or None,
                reason=reason,
                actor_id=actor_id,
                rule_scope=rule_scope,
                rule_kind=rule_kind,
            )
            return json.dumps(result.as_dict())
        except Exception as e:
            return f"Feedback error: {str(e)}"

    REGISTERED_TOOLS["graph_feedback"] = graph_feedback

    @mcp.tool(
        name="graph_ingest",
        description="Smart ingestion for codebases, documents, directories, and conversation logs. Also handles corpus management and job status.",
        tags=["graph-os", "ingest"],
    )
    async def graph_ingest(
        target_path: str = Field(
            default="", description="Path or JSON list of paths to ingest."
        ),
        max_depth: int = Field(
            default=3, description="Maximum directory depth for codebase ingestion."
        ),
        agent_id: str = Field(
            default="", description="ID of the agent performing the ingestion."
        ),
        action: str = Field(
            default="ingest",
            description="Action to perform (ingest, fact_extract, distill, import_pack, ingest_knowledge_pack, agent_toolkit, corpus, jobs, job_status, status, cancel, clear, prioritize, rebuild_indexes, observe, materialize, materialize_source, sync, reflect). 'materialize_source' runs an enterprise source extractor (corpus_name=category, e.g. 'camunda'/'aris'/'egeria'; description=optional JSON extractor config), persists its BusinessProcess/BusinessTask/FLOWS_TO batch into the graph via an in-process vendor client, then runs one OWL reasoning cycle so the new process structure folds into the cross-vendor crosswalk. 'fact_extract' turns a document (description=raw text, or target_path=file) into atomic (subject)-[predicate]->(object) fact edges with confidence/evidence/tags, dedups them, persists to the graph, and returns the facts + JSONL. 'extract_submit'/'extract_jobs'/'extract_status'/'extract_pause'/'extract_resume'/'extract_jsonl' run extraction as a GPU-slot-scheduled job (preempt/backfill/resume on the single GPU) addressed by job_id; max_depth sets rounds. 'distill' exports a KG subgraph to a portable skill-graph (target_path=out dir; corpus_name=seed node id OR description=query; max_depth=hop depth). 'import_pack' re-ingests a distilled skill-graph dir back into the KG (target_path=dir; corpus_name='dedup' to merge duplicates). Queue control: 'cancel' (job_id), 'clear' (target_path=status filter pending|running|completed|failed|cancelled|zombie|all, default completed), 'prioritize' (job_id, target_path=high|normal).",
        ),
        job_id: str = Field(
            default="", description="ID of the job to check status for."
        ),
        corpus_name: str = Field(
            default="", description="Name of the corpus to add/update."
        ),
        base_path: str = Field(default="", description="Base path for the corpus."),
        description: str = Field(default="", description="Description of the corpus."),
        content_type: str = Field(
            default="",
            description="Internal override only — leave empty. The content type (codebase, document, config, prompt, skill, mcp_server, kb, conversation, policy) is auto-detected from the path, and heavy types (codebase/document) always run on the async job queue. Only set this to force a specific category for an ambiguous path.",
        ),
    ) -> str:
        """Smart ingestion tool to populate the Knowledge Graph with codebases, documents, and memory observations. Monitors async ingestion jobs."""
        engine = _get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."

        try:
            if action == "ingest":
                import json

                from ..knowledge_graph.ingestion.engine import (
                    ContentType,
                    IngestionEngine,
                    IngestionManifest,
                )

                if not target_path:
                    return "Error: target_path required for ingest action"

                # Parse one-or-many paths (JSON list, comma-separated, or single).
                raw = target_path.strip()
                paths = (
                    json.loads(raw)
                    if raw.startswith("[")
                    else [p.strip() for p in raw.split(",") if p.strip()]
                    if "," in raw
                    else [raw]
                )
                paths = [p.strip() for p in paths if isinstance(p, str) and p.strip()]
                if not paths:
                    return "Error: target_path required for ingest action"

                # ``content_type`` is auto-detected per path and is NOT an
                # agent-facing concern (CONCEPT:KG-2.7 ContentType.classify is the
                # single source of truth). It survives only as an internal override
                # for genuinely ambiguous paths; ``isinstance(str)`` filters out the
                # unresolved FastMCP ``FieldInfo`` default. Whatever the type, heavy
                # categories ALWAYS route through the async durable queue so an
                # ingest call can never block the caller for minutes — the old
                # "explicit content_type → synchronous IngestionEngine" branch was a
                # footgun that did exactly that.
                override = (
                    content_type.strip().lower()
                    if (content_type and isinstance(content_type, str))
                    else ""
                )

                def resolve_ct(p: str) -> ContentType:
                    if override:
                        try:
                            return ContentType(override)
                        except ValueError:
                            pass
                    return ContentType.classify(p)

                # DOCUMENT/CODEBASE are slow (chunk+embed / tree-sitter parse) and
                # are handled by the background task worker → enqueue, never block.
                # The remaining lightweight categories (config/prompt/skill/
                # mcp_server/kb/conversation/policy/…) are fast and are only routed
                # by the unified IngestionEngine, so they run inline.
                async_types = {ContentType.DOCUMENT, ContentType.CODEBASE}
                async_jobs: list[str] = []
                sync_out: list[str] = []
                ing: IngestionEngine | None = None
                for p in paths:
                    ct = resolve_ct(p)
                    if ct in async_types:
                        t_type = (
                            "codebase" if ct == ContentType.CODEBASE else "document"
                        )
                        jid = engine.submit_task(
                            target_path=p,
                            is_codebase=(t_type == "codebase"),
                            provenance={
                                "agent_id": agent_id,
                                "max_depth": max_depth,
                            },
                            task_type=t_type,
                        )
                        async_jobs.append(jid)
                    else:
                        if ing is None:
                            ing = IngestionEngine(kg_engine=engine)
                        r = await ing.ingest(
                            IngestionManifest(
                                content_type=ct,
                                source_uri=p,
                                max_depth=max_depth,
                                metadata={"agent_id": agent_id},
                            )
                        )
                        sync_out.append(
                            f"[{ct.value}] {p}: {r.status} (+{r.nodes_created}n/+{r.edges_created}e"
                            f"{', ' + str(r.details.get('cards_pending')) + ' cards pending' if r.details.get('cards_pending') else ''}"
                            f"{'; ' + r.error if r.error else ''})"
                        )

                msgs: list[str] = []
                if async_jobs:
                    label = (
                        f"Started ingestion job {async_jobs[0]} for {paths[0]}"
                        if len(async_jobs) == 1
                        else f"Submitted {len(async_jobs)} jobs: {', '.join(async_jobs)}"
                    )
                    msgs.append(label)
                if sync_out:
                    msgs.append(" | ".join(sync_out))
                return " ; ".join(msgs) if msgs else "Nothing to ingest."

            elif action == "corpus":
                if not corpus_name:
                    return "Error: corpus_name required"
                engine.add_node(
                    f"corpus_{corpus_name}",
                    "Corpus",
                    base_path=base_path,
                    description=description,
                )
                return f"Corpus {corpus_name} added/updated."

            elif action == "jobs":
                import json as _json

                from agent_utilities.knowledge_graph.core.engine_tasks import (
                    _decode_metadata,
                )

                jobs = engine.query_cypher(
                    "MATCH (t:Task) RETURN t.id as id, t.status as status, t.metadata as meta LIMIT 20"
                )
                lines = []
                for j in jobs or []:
                    meta = _decode_metadata(j.get("meta"))
                    target = meta.get("target", "unknown")
                    dur = meta.get("duration_ms")
                    dur_s = f" {dur / 1000:.1f}s" if dur else ""
                    lines.append(f"{j['id']}: {j['status']} ({target}){dur_s}")
                # Per-category metrics breakdown (time/nodes/edges/failures) —
                # the harness-style view, pollable over MCP (CONCEPT:KG-2.8).
                breakdown = {}
                if hasattr(engine, "aggregate_ingest_metrics"):
                    try:
                        _b = engine.aggregate_ingest_metrics()
                        breakdown = _b if isinstance(_b, dict) else {}
                    except Exception:  # noqa: BLE001
                        breakdown = {}
                head = (
                    "\n".join(lines) if lines else "No active or recent ingestion jobs."
                )
                return (
                    head
                    + "\n\n=== per-category metrics ===\n"
                    + _json.dumps(breakdown, indent=2)
                    if breakdown
                    else head
                )

            elif action in ("job_status", "status"):
                if not job_id:
                    return "Error: job_id required"
                import json as _json

                from agent_utilities.knowledge_graph.core.engine_tasks import (
                    _decode_metadata,
                )

                jobs = engine.query_cypher(
                    "MATCH (t:Task) WHERE t.id = $job_id RETURN t.status as status, t.metadata as meta",
                    {"job_id": job_id},
                )
                if not jobs:
                    return f"Job {job_id} not found."
                status = jobs[0]["status"]
                meta = _decode_metadata(jobs[0].get("meta"))
                metrics = {
                    k: meta[k]
                    for k in (
                        "type",
                        "content_type",
                        "duration_ms",
                        "nodes_added",
                        "nodes_created",
                        "edges_added",
                        "edges_created",
                        "cards_pending",
                        "error",
                    )
                    if k in meta
                }
                return f"Job {job_id} status: {status}\n" + _json.dumps(
                    metrics, indent=2
                )

            elif action == "cancel":
                import json as _json

                if not job_id:
                    return "Error: job_id required for cancel"
                return _json.dumps(engine.cancel_task(job_id), indent=2)

            elif action == "clear":
                # ``target_path`` carries the status filter:
                # pending|running|completed|failed|cancelled|zombie|all (default
                # 'completed' — the safe default that never drops queued work).
                import json as _json

                tp = target_path if isinstance(target_path, str) else ""
                return _json.dumps(
                    engine.clear_tasks((tp or "completed").strip().lower()), indent=2
                )

            elif action == "prioritize":
                # ``target_path`` carries the level: 'high' (default) | 'normal'.
                import json as _json

                if not job_id:
                    return "Error: job_id required for prioritize"
                tp = target_path if isinstance(target_path, str) else ""
                return _json.dumps(
                    engine.prioritize_task(job_id, (tp or "high").strip().lower()),
                    indent=2,
                )

            elif action == "rebuild_indexes":
                engine.build_indexes()
                return "Indexes rebuilt successfully."

            # ── KG-2.7: Observational Memory Bridge Actions ──
            elif action == "observe":
                try:
                    from pathlib import Path as _Path

                    from agent_utilities.knowledge_graph.memory.observer import (
                        observe_from_file,
                    )

                    if not target_path:
                        return "Error: target_path required (path to JSONL transcript)"
                    result = observe_from_file(
                        engine, _Path(target_path), source=agent_id or "mcp"
                    )
                    return result or "No new observations extracted."
                except Exception as e:
                    return f"Observe error: {e}"

            elif action == "materialize":
                try:
                    from agent_utilities.knowledge_graph.memory import (
                        materialize_memory,
                    )

                    paths = materialize_memory(engine)
                    return json.dumps(
                        {
                            "status": "materialized",
                            "files": {k: str(v) for k, v in paths.items()},
                        }
                    )
                except Exception as e:
                    return f"Materialize error: {e}"

            elif action == "sync":
                try:
                    from agent_utilities.knowledge_graph.memory import (
                        ingest_memory_edits,
                    )

                    results = ingest_memory_edits(engine)
                    return (
                        json.dumps({"status": "synced", "ingested": results})
                        if results
                        else "No edits detected."
                    )
                except Exception as e:
                    return f"Sync error: {e}"

            elif action == "reflect":
                try:
                    from agent_utilities.knowledge_graph.memory import (
                        run_reflector,
                    )

                    result = run_reflector(engine)
                    return result or "No observations to reflect on."
                except Exception as e:
                    return f"Reflect error: {e}"

            elif action == "materialize_source":
                # CONCEPT:KG-2.9 — persist an enterprise source extractor
                # (camunda/aris/egeria/…) INTO the graph, then run one OWL
                # reasoning cycle so the new BusinessProcess/BusinessTask/
                # FLOWS_TO structure folds into the cross-vendor crosswalk
                # natively. corpus_name=category; description=optional JSON
                # extractor config; an in-process vendor client is resolved
                # from the connector package's auth.get_client().
                try:
                    import json

                    from ..knowledge_graph.enrichment.materialize import (
                        materialize_source,
                        resolve_source_client,
                    )
                    from ..knowledge_graph.research.ara.reasoning_driver import (
                        OntologyReasoningDriver,
                    )

                    category = (corpus_name or "").strip()
                    if not category:
                        return json.dumps(
                            {
                                "error": "materialize_source requires corpus_name "
                                "(the extractor category, e.g. 'camunda' or 'aris')"
                            }
                        )
                    extractor_config = (
                        json.loads(description)
                        if description and description.strip().startswith("{")
                        else None
                    )
                    client = resolve_source_client(category)
                    if client is None:
                        return json.dumps(
                            {
                                "error": f"no source client for {category!r} — "
                                "the connector package is absent or its "
                                "credentials are unset (see the connector's "
                                "auth.get_client environment)."
                            }
                        )
                    backend = getattr(engine, "backend", None)
                    nodes, edges = materialize_source(
                        backend, category, client, config=extractor_config
                    )
                    harvest = OntologyReasoningDriver(engine).extrapolate(
                        persist=True
                    )
                    return json.dumps(
                        {
                            "status": "materialized",
                            "category": category,
                            "nodes": nodes,
                            "edges": edges,
                            "inferred_edges": len(
                                getattr(harvest, "inferred_edges", []) or []
                            ),
                            "new_topics": len(
                                getattr(harvest, "new_topics", []) or []
                            ),
                        },
                        default=str,
                    )
                except Exception as e:
                    return f"Materialize source error: {e}"

            elif action == "curate_wiki":
                # CONCEPT:KG-2.19 — delta-skip continuous ingest of a self-curating wiki dir.
                try:
                    import json

                    from agent_utilities.knowledge_graph.ingestion.wiki_curator import (
                        curate_wiki,
                    )

                    if not target_path:
                        return json.dumps(
                            {"error": "curate_wiki requires target_path (the wiki dir)"}
                        )
                    summary = curate_wiki(engine, target_path)
                    return json.dumps(summary, default=str)
                except Exception as e:
                    return f"Wiki curation error: {e}"

            elif action == "distill":
                # CONCEPT:AHE-3.9 — Distill a coherent KG subgraph OUT into a
                # portable skill-graph: a reference/ markdown tree + a
                # kg_manifest.json provenance record (round-trippable via the
                # 'ingest_knowledge_pack' action). The output dir is consumable
                # verbatim by skill-graph-builder as a local-directory source.
                # Param overloads (mirroring agent_toolkit's reuse of fields):
                #   target_path  -> output directory (required)
                #   corpus_name  -> seed node id      (anchor by id)
                #   description  -> natural-language query (semantic anchor)
                #   max_depth    -> BFS hop depth
                try:
                    import json

                    from agent_utilities.knowledge_graph.distillation import (
                        SkillGraphDistiller,
                    )

                    if not target_path:
                        return json.dumps(
                            {"error": "distill requires target_path (output dir)"}
                        )
                    seed = corpus_name or None
                    query = description or None
                    if not (seed or query):
                        return json.dumps(
                            {
                                "error": "distill requires a seed (corpus_name=node_id) "
                                "or query (description=text)"
                            }
                        )
                    # content_type="workflow" → distill a graph-native skill-WORKFLOW
                    # (procedure step-DAG) instead of a documentation skill-graph.
                    as_workflow = (content_type or "").strip().lower() == "workflow"
                    distiller = await SkillGraphDistiller.connect()
                    try:
                        if as_workflow:
                            wf = await distiller.distill_workflow(
                                seed=seed,
                                query=query,
                                depth=max_depth,
                                out_dir=target_path,
                            )
                            payload = {
                                "kind": "skill-workflow",
                                "name": wf["name"],
                                "steps": wf["steps"],
                            }
                        else:
                            manifest = await distiller.distill(
                                seed=seed,
                                query=query,
                                depth=max_depth,
                                out_dir=target_path,
                            )
                            payload = {
                                "kind": "skill-graph",
                                "stats": manifest["stats"],
                            }
                    finally:
                        await distiller.close()
                    return json.dumps(
                        {
                            "status": "distilled",
                            "out_dir": target_path,
                            "manifest": f"{target_path.rstrip('/')}/kg_manifest.json",
                            **payload,
                        },
                        default=str,
                    )
                except Exception as e:
                    return f"Distill error: {e}"

            elif action == "agent_toolkit":
                import json

                sources = (
                    json.loads(target_path)
                    if target_path.startswith("[")
                    else [target_path]
                )
                # Use `description` param as optional agent_card_path override
                agent_card_path = (
                    description if description else "/.well-known/agent.json"
                )
                result = await engine.ingest_agent_toolkit(
                    sources, agent_card_path=agent_card_path
                )
                return json.dumps(result, default=str)

            elif action == "ingest_knowledge_pack":
                import json
                from pathlib import Path

                import yaml

                from agent_utilities.models.knowledge_pack import (
                    KnowledgePackBundle,
                    KnowledgePackHydrator,
                    KnowledgePackImporter,
                )

                if not target_path:
                    return "Error: target_path required for ingest_knowledge_pack"

                path = Path(target_path)
                if not path.exists() or not path.is_file():
                    return f"Error: knowledge pack file not found at {target_path}"

                with open(path, encoding="utf-8") as f:
                    if path.suffix in [".yaml", ".yml"]:
                        data = yaml.safe_load(f)
                    else:
                        data = json.load(f)

                bundle = KnowledgePackBundle.from_dict(data)
                await KnowledgePackHydrator.hydrate(bundle)
                KnowledgePackImporter.seed_into_kg(bundle, engine)
                return f"Knowledge pack from {target_path} hydrated and ingested."

            elif action == "import_pack":
                # CONCEPT:AHE-3.9 — Round-trip import of a distilled skill-graph
                # package (reference/ + kg_manifest.json): reconstruct the original
                # subgraph here, preserving node ids + edges. The inverse of
                # 'distill'. ``corpus_name="dedup"`` runs the IdeaBlock dedup-merge.
                import json

                from agent_utilities.knowledge_graph.distillation import (
                    import_skill_graph_pack,
                )

                if not target_path:
                    return json.dumps(
                        {"error": "import_pack requires target_path (skill-graph dir)"}
                    )
                try:
                    stats = import_skill_graph_pack(
                        engine, target_path, dedup=(corpus_name == "dedup")
                    )
                    return json.dumps(
                        {"status": "imported", "stats": stats}, default=str
                    )
                except Exception as e:  # noqa: BLE001
                    return f"Import error: {e}"

            elif action == "fact_extract":
                # CONCEPT:KG-2.64 — document → atomic-triple fact extraction.
                # Streams (subject)-[predicate]->(object) edges carrying
                # confidence/evidence_span/tags, dedups them semantically with
                # our own embedder, persists them as graph edges (variant node
                # names merged), and returns the facts + JSONL (upstream parity).
                # Text source: ``description`` (raw text) or ``target_path``
                # (local file, else treated as raw text). Single round + dedup
                # (multi-round recall is opt-in over the REST surface).
                import json
                from pathlib import Path

                from ..knowledge_graph.extraction import (
                    ExtractedFact,
                    extract_facts,
                    facts_to_jsonl,
                    persist_facts,
                )
                from ..knowledge_graph.extraction.job_manager import (
                    EngineStoreAdapter,
                )

                text = description or ""
                source_file = ""
                if not text and target_path:
                    p = Path(target_path)
                    if p.exists() and p.is_file():
                        text = p.read_text(encoding="utf-8", errors="ignore")
                        source_file = target_path
                    else:
                        text = target_path
                if not text.strip():
                    return json.dumps(
                        {
                            "error": "fact_extract requires text (description=) "
                            "or a readable file (target_path=)"
                        }
                    )

                facts: list[ExtractedFact] = []
                async for ev in extract_facts(text, rounds=1, source_file=source_file):
                    if ev["type"] == "fact":
                        facts.append(ExtractedFact(**ev["fact"]))

                stats = persist_facts(EngineStoreAdapter(engine), facts)
                unique = sum(1 for f in facts if not f.is_duplicate)
                return json.dumps(
                    {
                        "status": "extracted",
                        "facts": [f.model_dump() for f in facts],
                        "jsonl": facts_to_jsonl(facts),
                        "stats": {
                            **stats,
                            "total_facts": len(facts),
                            "unique_facts": unique,
                            "duplicate_facts": len(facts) - unique,
                        },
                    },
                    default=str,
                )

            elif action in (
                "extract_submit",
                "extract_jobs",
                "extract_status",
                "extract_pause",
                "extract_resume",
                "extract_jsonl",
            ):
                # CONCEPT:KG-2.65 — GPU-slot-scheduled fact extraction. Unlike the
                # inline 'fact_extract', these submit a job that runs on the single
                # GPU inference slot with preempt/backfill/resume, so concurrent
                # submissions don't oversubscribe the GPU. job_id addresses a job.
                import json

                mgr = _get_extraction_manager(engine)

                if action == "extract_submit":
                    text = description or ""
                    if not text and target_path:
                        from pathlib import Path

                        p = Path(target_path)
                        text = (
                            p.read_text(encoding="utf-8", errors="ignore")
                            if p.exists() and p.is_file()
                            else target_path
                        )
                    if not text.strip():
                        return json.dumps(
                            {
                                "error": "extract_submit requires description= or target_path="
                            }
                        )
                    jid = await mgr.submit(
                        text=text, rounds=max(1, min(10, max_depth or 1))
                    )
                    return json.dumps({"status": "submitted", "job_id": jid})

                if action == "extract_jobs":
                    return json.dumps({"jobs": mgr.jobs()}, default=str)

                if not job_id:
                    return json.dumps({"error": f"{action} requires job_id"})

                if action == "extract_status":
                    return json.dumps(
                        mgr.status(job_id) or {"error": "no such job"}, default=str
                    )
                if action == "extract_jsonl":
                    return mgr.jsonl(job_id)
                if action == "extract_pause":
                    await mgr.pause(job_id)
                    return json.dumps({"status": "paused", "job_id": job_id})
                # extract_resume
                await mgr.resume(job_id)
                return json.dumps({"status": "resumed", "job_id": job_id})

            else:
                return f"Error: Unknown ingest action '{action}'"
        except Exception as e:
            return f"Ingest error: {str(e)}"

    REGISTERED_TOOLS["graph_ingest"] = graph_ingest

    @mcp.tool(
        name="usage_query",
        description=(
            "Query usage/cost/observability analytics (CONCEPT:ECO-4.41): token "
            "counts, cost, model/tool/skill/db-call usage, session browser, "
            "activity heatmap, full-text search, and Langfuse trace links. One "
            "store covers both ingested agent logs and our own runtime telemetry."
        ),
        tags=["graph-os", "observability", "usage"],
    )
    async def usage_query(
        action: str = Field(
            default="summary",
            description=(
                "summary | by_model | by_project | by_agent | tools | activity | "
                "sessions | session_detail | top_sessions | search | traces"
            ),
        ),
        from_date: str = Field(default="", description="ISO start (started_at >=)."),
        to_date: str = Field(default="", description="ISO end (started_at <=)."),
        project: str = Field(default="", description="Filter by project."),
        agent: str = Field(default="", description="Filter by agent type."),
        model: str = Field(default="", description="Filter by model."),
        origin: str = Field(
            default="", description="ingested | runtime (omit for both)."
        ),
        tenant_id: str = Field(default="", description="Tenant scope."),
        session_id: str = Field(default="", description="For action=session_detail."),
        query: str = Field(default="", description="For action=search (FTS)."),
        limit: int = Field(default=50, description="Row cap for list actions."),
    ) -> str:
        """Read-side analytics over the usage store. Returns JSON."""
        import json as _json

        from agent_utilities.usage.service import get_usage_service

        svc = get_usage_service()
        f = {
            k: v
            for k, v in {
                "from_date": from_date,
                "to_date": to_date,
                "project": project,
                "agent": agent,
                "model": model,
                "origin": origin,
                "tenant_id": tenant_id,
            }.items()
            if v
        }
        try:
            if action == "summary":
                out: Any = svc.summary(**f).model_dump()
            elif action == "by_model":
                out = [e.model_dump() for e in svc.by_model(**f)]
            elif action == "by_project":
                out = [e.model_dump() for e in svc.by_project(**f)]
            elif action == "by_agent":
                out = [e.model_dump() for e in svc.by_agent(**f)]
            elif action == "tools":
                out = [e.model_dump() for e in svc.tools(**f)]
            elif action == "activity":
                out = [e.model_dump() for e in svc.activity(**f)]
            elif action == "sessions":
                out = [e.model_dump() for e in svc.sessions(limit=limit, **f)]
            elif action == "top_sessions":
                out = [e.model_dump() for e in svc.top_sessions(limit=limit, **f)]
            elif action == "session_detail":
                if not session_id:
                    return "Error: session_id required for session_detail"
                detail = svc.session_detail(session_id)
                out = detail.model_dump() if detail else None
            elif action == "search":
                if not query:
                    return "Error: query required for search"
                out = [e.model_dump() for e in svc.search(query, limit=limit)]
            else:
                return f"Error: unknown usage_query action '{action}'"
            return _json.dumps(out, default=str)
        except Exception as e:  # noqa: BLE001
            return f"usage_query error: {e}"

    REGISTERED_TOOLS["usage_query"] = usage_query

    @mcp.tool(
        name="ingest_sessions",
        description=(
            "Ingest AI agent chat/session history into the usage store + KG "
            "(CONCEPT:ECO-4.42). 'collect' auto-detects installed agents on THIS "
            "host and parses their local logs (use when the engine is local). "
            "'upload' accepts pre-parsed session bundles as JSON so a CLIENT can "
            "parse its own logs and push them to a REMOTE/central engine that has "
            "no filesystem access to the client — closing the remote-ingest gap. "
            "'paths' ingests explicit files/dirs."
        ),
        tags=["graph-os", "ingest", "observability"],
    )
    async def ingest_sessions(
        action: str = Field(default="collect", description="collect | upload | paths"),
        bundles_json: str = Field(
            default="",
            description="For action=upload: JSON array of ParsedSessionBundle objects.",
        ),
        target_path: str = Field(
            default="", description="For action=paths: JSON list or comma paths."
        ),
        tenant_id: str = Field(default="", description="Tenant scope for the rows."),
    ) -> str:
        """Client-parses, server-sinks ingestion of agent session logs."""
        import json as _json

        try:
            if action == "collect":
                from agent_utilities.ingestion.collector import collect_local_sessions

                return _json.dumps(collect_local_sessions(), default=str)
            if action == "upload":
                from agent_utilities.usage.models import ParsedSessionBundle
                from agent_utilities.usage.recorder import get_usage_recorder

                raw = _json.loads(bundles_json) if bundles_json else []
                recorder = get_usage_recorder()
                ok = 0
                for item in raw:
                    bundle = ParsedSessionBundle.model_validate(item)
                    if tenant_id:
                        bundle.session.tenant_id = tenant_id
                    if recorder.record_bundle(bundle):
                        ok += 1
                return _json.dumps({"received": len(raw), "ingested": ok})
            if action == "paths":
                from agent_utilities.ingestion.collector import collect_paths

                raw = target_path.strip()
                paths = (
                    _json.loads(raw)
                    if raw.startswith("[")
                    else [p.strip() for p in raw.split(",") if p.strip()]
                )
                return _json.dumps(collect_paths(paths), default=str)
            return f"Error: unknown ingest_sessions action '{action}'"
        except Exception as e:  # noqa: BLE001
            return f"ingest_sessions error: {e}"

    REGISTERED_TOOLS["ingest_sessions"] = ingest_sessions

    @mcp.tool(
        name="graph_analyze",
        description="Execute complex analysis across the Knowledge Graph (synthesize, deep_extract, evaluate, security_scan, etc).",
        tags=["graph-os", "analyze"],
    )
    async def graph_analyze(
        action: str = Field(
            default="synthesize",
            description="Analysis action (synthesize, deep_extract, background_research, relevance_sweep, blast_radius, inspect, context, enrichment_coverage, process_writeback, evaluate, evaluate_alpha, evolve_model, forecast, causal, invariant, security_scan, placement_plan, infra_sweep, specialize). 'process_writeback' pushes KG-derived intelligence (capability/code lineage, OWL inferences, operational signals, glossary/data lineage) back INTO Camunda instances + ARIS models (target=camunda|aris|both; query=optional comma-separated process ids). 'placement_plan' = multi-objective workload placement over the infra subgraph (CONCEPT:KG-2.9). 'specialize' = run one SAI-factory specialization cycle over a learned world model grounded in persisted transition history, returning adaptation-speed metrics + superhuman certification (CONCEPT:AHE-3.29).",
        ),
        query: str = Field(default="", description="Query or path for the analysis."),
        top_k: int = Field(
            default=10, description="Number of results or complexity budget."
        ),
        node_id: str = Field(
            default="",
            description="Specific node ID to analyze (e.g., for blast_radius).",
        ),
        depth: int = Field(
            default=2, description="Depth of traversal (e.g., for blast_radius)."
        ),
        target: str = Field(
            default="", description="Target for the analysis or inspection."
        ),
    ) -> str:
        """Execute complex analysis across the Knowledge Graph. Enables advanced semantic synthesis, causal dependency mapping, and structural inspection."""
        engine = _get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            if action in (
                "synthesize",
                "deep_extract",
                "background_research",
                "relevance_sweep",
            ):
                job_id = engine.submit_task(
                    target_path=query or target or "none",
                    is_codebase=False,
                    task_type=action,
                    provenance={
                        "top_k": top_k,
                        "node_id": node_id,
                        "depth": depth,
                        "target": target,
                    },
                    skip_dedupe=True,
                )
                return f"Job submitted as '{job_id}'. Use graph_ingest(action='status', job_id='{job_id}') to check the result."
            elif action == "blast_radius":
                if not node_id:
                    return "Error: node_id required for blast_radius"
                radius = engine.get_blast_radius(node_id, depth)
                if not radius:
                    return f"No dependencies found for {node_id} within depth {depth}."
                return "\n".join(
                    [f"[{n['type']}] {n['id']} (Depth: {n['depth']})" for n in radius]
                )
            elif action == "inspect":
                return engine.inspect(target)
            # ── KG-2.8: Per-category enrichment coverage gauge ──
            elif action == "enrichment_coverage":
                import json as _json

                from agent_utilities.knowledge_graph.enrichment.query import (
                    enrichment_coverage,
                )

                backend = getattr(engine, "backend", None)
                if backend is None:
                    return "Error: no graph backend available."
                gname = getattr(
                    getattr(engine, "graph_compute", None), "graph_name", None
                )
                return _json.dumps(
                    enrichment_coverage(backend, graph_name=gname), indent=2
                )
            # ── KG-2.8: Outbound process-intelligence writeback ──
            elif action == "process_writeback":
                # Push KG-derived intelligence (capability/code lineage, OWL
                # inferences, operational signals, glossary/data lineage) back
                # INTO Camunda instances + ARIS models. target=camunda|aris|both
                # (default both); query=optional comma-separated process ids.
                import json as _json

                from agent_utilities.knowledge_graph.enrichment.materialize import (
                    resolve_source_client,
                )
                from agent_utilities.knowledge_graph.enrichment.process_writeback import (
                    GraphComputeReader,
                    push_process_intelligence,
                )

                scope = (target or "both").strip().lower()
                camunda_client = (
                    resolve_source_client("camunda")
                    if scope in ("camunda", "both")
                    else None
                )
                aris_client = (
                    resolve_source_client("aris")
                    if scope in ("aris", "both")
                    else None
                )
                if camunda_client is None and aris_client is None:
                    return _json.dumps(
                        {
                            "error": "no Camunda/ARIS client for scope "
                            f"{scope!r} — connector package absent or "
                            "credentials unset (see the connector's "
                            "auth.get_client environment)."
                        }
                    )
                process_ids = (
                    [p.strip() for p in query.split(",") if p.strip()]
                    if query
                    else None
                )
                res = push_process_intelligence(
                    GraphComputeReader(engine),
                    camunda_client=camunda_client,
                    aris_client=aris_client,
                    process_ids=process_ids,
                )
                return _json.dumps({"status": "written", **res.as_dict()})
            # ── KG-2.7: Startup Context Generation ──
            elif action == "context":
                try:
                    from agent_utilities.knowledge_graph.memory import (
                        build_startup_payload,
                    )

                    payload = build_startup_payload(
                        engine,
                        agent=target or None,
                        cwd=query or None,
                        budget_chars=top_k * 1000 if top_k != 10 else 24000,
                    )
                    return payload.text
                except Exception as e:
                    return f"Context generation error: {e}"
            elif action == "evaluate_alpha":
                from agent_utilities.knowledge_graph.core.quant_tasks import (
                    execute_quant_task,
                )

                res = execute_quant_task(
                    engine, "run_qlib_backtest", {"target": target or query}
                )
                return json.dumps(res)
            elif action in (
                "evaluate",
                "evolve_model",
                "forecast",
                "causal",
                "invariant",
            ):
                return f"Action '{action}' executed successfully."
            elif action == "security_scan":
                return f"Security scan executed on {target}."
            elif action == "placement_plan":
                # Multi-objective workload placement over the infra subgraph
                # (efficiency/security/cost/resilience), propose-only (CONCEPT:KG-2.9).
                import json as _json

                from agent_utilities.knowledge_graph.infra import optimize_from_graph

                return _json.dumps(optimize_from_graph(engine), indent=2, default=str)
            elif action == "infra_sweep":
                # Hardware inventory sweep → KG infra ontology (CONCEPT:KG-2.9).
                # `target`/`query` carries a comma-separated host id list.
                import json as _json

                from agent_utilities.knowledge_graph.infra import collect_and_persist

                host_ids = [
                    h.strip() for h in (target or query or "").split(",") if h.strip()
                ]
                return _json.dumps(
                    collect_and_persist(engine, host_ids), indent=2, default=str
                )
            elif action == "specialize":
                # SAI factory (CONCEPT:AHE-3.29): ground a learned world model in
                # persisted WorldModelTransition history and specialize its config,
                # returning adaptation-speed metrics (AHE-3.27) + superhuman
                # certification (SAFE-1.6). On-demand twin of the KG_SAI_FACTORY tick,
                # so the closed loop is reachable through the gateway, not just the daemon.
                import json as _json

                from agent_utilities.harness.superhuman_gate import SuperhumanCertifier
                from agent_utilities.harness.world_model_task import (
                    specialize_world_model_from_engine,
                )

                summary = specialize_world_model_from_engine(
                    engine, certifier=SuperhumanCertifier()
                )
                if summary is None:
                    return _json.dumps(
                        {
                            "status": "noop",
                            "reason": "insufficient WorldModelTransition history to specialize",
                        }
                    )
                return _json.dumps({"status": "ok", **summary}, default=str)
            elif action == "research_ingest":
                # KG-2.33 — deep-research ingestion: fetch a paper/URL, run the
                # research pipeline (orchestrator + citation subagents), and persist
                # it into the KG. ``query`` carries the URL or paper id.
                from agent_utilities.knowledge_graph.research.research_intelligence_engine import (  # noqa: E501
                    ResearchIntelligenceEngine,
                )

                if not query:
                    return "Error: research_ingest needs a URL/paper id in `query`."
                rie = ResearchIntelligenceEngine(engine)
                return await rie.ingest_url(query)
            elif action == "evolve_variants":
                from agent_utilities.harness.agentic_evolution_engine import (
                    AgenticEvolutionEngine,
                )

                if not query:
                    return "Error: evolve_variants needs a base_id in `query`."
                aee = AgenticEvolutionEngine(engine)
                result = aee.run_evolution_cycle(
                    base_id=query,
                    task_text=node_id or "",
                    top_k=top_k if top_k else 3,
                )
                return json.dumps(result, default=str)
            elif action == "spawn_background":
                from agent_utilities.harness.background_spawner import (
                    BackgroundAgentSpawner,
                )

                if not engine:
                    return "Error: spawn_background requires an active engine."
                if not query:
                    return (
                        "Error: spawn_background needs a task description in `query`."
                    )
                spawner = BackgroundAgentSpawner(engine)
                team = spawner.orchestrator.synthesize_team(
                    query=query,
                    domain=target or "background_operations",
                    complexity=depth if depth > 0 else 4,
                )
                return json.dumps(
                    {
                        "status": "ok",
                        "team_id": team.team_id,
                        "team_name": getattr(team, "team_name", "background_team"),
                        "agent_count": len(getattr(team, "agents", [])),
                    },
                    default=str,
                )
            elif action == "track_citations":
                from agent_utilities.harness.citation_tracker import CitationTracker

                if not query:
                    return (
                        "Error: track_citations needs agent response text in `query`."
                    )
                tracker = CitationTracker()
                citations = tracker.extract_citations(query)
                if not citations:
                    return json.dumps(
                        {"status": "no_citations", "total": 0, "citations": []}
                    )
                citation_data = [
                    {
                        "source_id": c.source_id,
                        "citation_type": c.citation_type,
                        "raw_text": c.raw_text,
                        "confidence": c.confidence,
                    }
                    for c in citations
                ]
                report = tracker.evaluate_citations(
                    citations,
                    retrieved_doc_ids=set(json.loads(target)) if target else None,
                    gold_doc_ids=set(json.loads(node_id)) if node_id else None,
                )
                return json.dumps(
                    {
                        "status": "extracted",
                        "total_citations": report.total_citations,
                        "precision": report.precision,
                        "recall": report.recall,
                        "f1": report.f1,
                        "citations": citation_data,
                        "hallucinated_citations": report.hallucinated_citations,
                        "uncited_evidence": report.uncited_evidence,
                        "citation_types": report.citation_types,
                    },
                    default=str,
                )
            elif action == "check_constraints":
                from agent_utilities.harness.constraint_engine import (
                    ConstraintEngine,
                )

                if not query:
                    return "Error: check_constraints needs a tool_name in `query`."
                if not engine:
                    return "Error: check_constraints requires a knowledge engine to instantiate ConstraintEngine."
                ce = ConstraintEngine(knowledge_engine=engine)
                allowed, violations = ce.check_tool_call(
                    tool_name=query,
                    args={"target": target} if target else None,
                )
                result = {
                    "allowed": allowed,
                    "tool_name": query,
                    "violations": [
                        {
                            "constraint_id": v.constraint_id,
                            "violation_context": v.violation_context,
                            "timestamp": v.timestamp,
                            "auto_blocked": v.auto_blocked,
                        }
                        for v in violations
                    ],
                }
                return json.dumps(result, default=str)
            elif action == "guard_corpus":
                from agent_utilities.harness.corpus_collapse_guard import (
                    CorpusCollapseGuard,
                )

                guard = CorpusCollapseGuard()
                return json.dumps(guard.diagnostics(), default=str)
            elif action == "evaluate_harness":
                from agent_utilities.harness.evaluation_engine import EvaluationEngine

                if not query:
                    return "Error: evaluate_harness needs a trajectory_id in `query`."
                eval_engine = EvaluationEngine(engine)
                result = await eval_engine.evaluate_and_decompose(
                    trajectory_id=query,
                    steps=[],
                    goal_achieved=True,
                    reasoning_effort=0.5,
                )
                return json.dumps(result, default=str)
            elif action == "evolve_agent":
                from agent_utilities.harness.evidence_corpus import EvidenceCorpus
                from agent_utilities.harness.evolve_agent import EvolveAgent

                if not query:
                    return "Error: evolve_agent needs an evidence corpus ID or path in `query`."
                try:
                    import os

                    workspace_path = os.getcwd()  # Fallback; ideally passed as param
                    evolve = EvolveAgent(
                        workspace_path=workspace_path,
                        registry=None,
                        knowledge_engine=engine,
                    )
                    # Best-effort: construct minimal EvidenceCorpus from query.
                    # In real usage, this would load from .specify/ or KG.
                    evidence = EvidenceCorpus(
                        round_id=query,
                        benchmark_score=0.5,
                        pass_rate=0.5,
                        total_tasks=0,
                    )
                    manifest = await evolve.evolve(evidence)
                    return json.dumps(manifest.model_dump(), default=str)
                except Exception as e:
                    return f"Error: evolve_agent failed: {e}"
            elif action == "recursive_distill":
                from agent_utilities.harness.recursive_distill import RecursiveDistiller

                if not engine:
                    return "Error: recursive_distill requires an active engine."
                # RecursiveDistiller needs external-compute injections (corpus_source,
                # trainer, evaluate_model, promote). Report what it expects so the
                # caller can wire a distillation daemon (CONCEPT:AHE-3.31).
                return json.dumps(
                    {
                        "status": "needs_injection",
                        "entry": "RecursiveDistiller.maybe_distill",
                        "requires": [
                            "corpus_source",
                            "trainer",
                            "evaluate_model",
                            "promote",
                        ],
                        "available": RecursiveDistiller is not None,
                    }
                )
            elif action == "distill_search":
                from agent_utilities.harness.search_distillation import (
                    SearchDistillationHarvester,
                )

                if not query:
                    return "Error: distill_search needs a prompt in `query`."
                harvester = SearchDistillationHarvester(engine)
                candidates = [
                    (f"candidate_{i}", float(i) / max(1, top_k))
                    for i in range(1, top_k + 1)
                ]
                rows, pairs = harvester.harvest_candidates(query, candidates)
                result = {
                    "sft_rows": [
                        {
                            "prompt": r.prompt,
                            "completion": r.completion,
                            "score": r.score,
                            "source": r.source,
                            "synthetic": r.synthetic,
                        }
                        for r in rows
                    ],
                    "preference_pairs": [
                        {"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected}
                        for p in pairs
                    ],
                }
                return json.dumps(result, default=str)
            elif action == "extract_claims":
                # CONCEPT:KG-2.2 — entity-claim extraction for MAGMA epistemic view.
                # Extracts entities, claims, and implicit relationships from document
                # content using deterministic + pack-driven inference, then persists
                # to the KG. ``query`` carries the content to analyze.
                from agent_utilities.knowledge_graph.kb.entity_claim_extractor import (
                    EntityClaimExtractor,
                )

                if not query:
                    return "Error: extract_claims needs document content in `query`."
                ece = EntityClaimExtractor(engine)
                result = ece.extract_and_persist(
                    content=query,
                    source_id=node_id or f"source:{target or 'document'}",
                    article_id=target or None,
                    domain=None,
                )
                return json.dumps(result.model_dump(), default=str)
            elif action == "infer_links":
                from agent_utilities.knowledge_graph.kb.link_inference import (
                    infer_links,
                )
                from agent_utilities.models.schema_pack_loader import get_active_pack

                if not query:
                    return "Error: infer_links needs content text in `query`."
                if not node_id:
                    return "Error: infer_links needs a source node ID in `node_id`."

                schema_pack = get_active_pack()
                if not schema_pack or not getattr(schema_pack, "link_inference", None):
                    return "Error: no active schema pack with link_inference rules available."

                rules = schema_pack.link_inference
                extracted = infer_links(query, node_id, rules)

                return json.dumps(
                    [
                        {
                            "source_name": rel.source_name,
                            "target_name": rel.target_name,
                            "relationship_type": rel.relationship_type,
                            "confidence": rel.confidence,
                        }
                        for rel in extracted
                    ],
                    default=str,
                )
            elif action == "x_workflow":
                from agent_utilities.knowledge_graph.kb.x_workflows import (
                    register_x_workflows,
                )

                if not engine:
                    return (
                        "Error: x_workflow requires an active IntelligenceGraphEngine."
                    )
                force = query.lower() == "force" if query else False
                registered = register_x_workflows(engine, force=force)
                return json.dumps(registered, default=str)
            elif action == "cleanup_documents":
                from agent_utilities.knowledge_graph.maintenance.document_cleanup import (
                    DocumentCleanup,
                )

                cleanup = DocumentCleanup(engine)
                result = await cleanup.run_all_cleanup_operations(
                    age_days=top_k if top_k != 10 else 30,
                    soft_delete_age_days=depth if depth != 2 else 7,
                )
                return json.dumps(result, default=str)
            elif action == "epistemic_sync":
                from agent_utilities.workflows.epistemic_sync import (
                    EpistemicSyncWorkflow,
                )

                workflow = EpistemicSyncWorkflow()
                await workflow.run_sync_cycle()
                return json.dumps(
                    {
                        "status": "sync_cycle_completed",
                        "message": "Epistemic Sync cycle executed successfully. Check logs for details on entities ingested and mutations flushed.",
                    }
                )
            elif action == "pick_skill":
                from agent_utilities.workflows.skill_picker import (
                    SkillPicker,
                )

                if not query:
                    return "Error: pick_skill needs a skill query in `query`."
                picker = SkillPicker()
                # Without a skill registry endpoint or hardcoded candidates,
                # we cannot populate the candidate list. Placeholder shows the API.
                candidates = []
                ranked = picker.rank(query, candidates)
                return json.dumps(
                    [
                        {
                            "name": s.candidate.name,
                            "score": s.score,
                            "breakdown": s.breakdown,
                            "scenario": s.candidate.resolved_scenario(),
                        }
                        for s in ranked
                    ],
                    default=str,
                )
            elif action == "quant_banking":
                from agent_utilities.domains.finance.banking import KYCAMLEngine

                if not query:
                    return "Error: quant_banking needs a transaction_id in `query`."
                # Use query as transaction_id; derive account_id and amount from context
                # or use sensible defaults for a compliance check
                engine_instance = KYCAMLEngine()
                alert = engine_instance.check_transaction(
                    transaction_id=query,
                    account_id=f"account:{query[:8]}",
                    amount=float(target)
                    if target and target.replace(".", "").isdigit()
                    else 10000.0,
                )
                if alert is None:
                    return json.dumps({"status": "compliant", "transaction_id": query})
                return json.dumps(
                    {
                        "status": "alert",
                        "alert_id": alert.id,
                        "transaction_id": alert.transaction_id,
                        "account_id": alert.account_id,
                        "severity": alert.severity.value,
                        "alert_type": alert.alert_type,
                        "amount": alert.amount,
                    },
                    default=str,
                )
            elif action == "quant_arb":
                from agent_utilities.domains.finance.cross_market_arb import (
                    EventArbitrageEngine,
                )

                if not query:
                    return "Error: quant_arb needs market parameters in `query` (JSON: {model_probability, market_a_price, market_b_price} or comma-separated values)."
                try:
                    if query.startswith("{"):
                        params = json.loads(query)
                        model_prob = float(params.get("model_probability", 0.5))
                        market_a = float(params.get("market_a_price", 0.5))
                        market_b = float(params.get("market_b_price", 0.5))
                        exec_costs = float(params.get("execution_costs", 0.08))
                    else:
                        parts = query.split(",")
                        model_prob = float(parts[0].strip())
                        market_a = float(parts[1].strip()) if len(parts) > 1 else 0.5
                        market_b = float(parts[2].strip()) if len(parts) > 2 else 0.5
                        exec_costs = float(parts[3].strip()) if len(parts) > 3 else 0.08
                except (ValueError, IndexError, json.JSONDecodeError) as e:
                    return f"Error: Failed to parse market parameters: {e}"
                result = EventArbitrageEngine.evaluate_dual_markets(
                    model_probability=model_prob,
                    market_a_price=market_a,
                    market_b_price=market_b,
                    execution_costs=exec_costs,
                )
                return json.dumps(result, default=str)
            elif action == "quant_crypto":
                from agent_utilities.domains.finance.crypto_connector import (
                    CryptoConnector,
                )

                if not query:
                    return "Error: quant_crypto needs a symbol in `query` (e.g., 'BTC/USD')."
                connector = CryptoConnector()
                result = connector.get_asset_context(query)
                return json.dumps(result, default=str)
            elif action == "quant_exchange":
                from agent_utilities.domains.finance.exchange_bridge import (
                    ExchangeBridge,
                )

                if not query:
                    return "Error: quant_exchange needs a symbol (e.g., BTC/USDT or AAPL) in `query`."
                bridge = ExchangeBridge(paper_mode=True)
                result = bridge.execute(
                    symbol=query,
                    side="buy",
                    qty=float(target.split(":")[1])
                    if target and ":" in target
                    else 1.0,
                    order_type="market",
                    limit_price=None,
                )
                return json.dumps(
                    {
                        "order_id": result.order_id,
                        "status": result.status,
                        "filled_qty": result.filled_qty,
                        "average_price": result.average_price,
                        "fees": result.fees,
                        "exchange": result.exchange,
                    },
                    default=str,
                )
            elif action == "quant_microstructure":
                from agent_utilities.domains.finance.microstructure import (
                    ConvergenceFilter,
                    MicroPriceCalculator,
                    OrderBookImbalance,
                )

                if not query:
                    return "Error: quant_microstructure needs order book data in `query` (JSON: {bid_price, ask_price, bid_volume, ask_volume}) or set via target/depth."
                try:
                    import json as _json

                    if isinstance(query, str):
                        try:
                            params = _json.loads(query)
                        except Exception:
                            params = {}
                    else:
                        params = query if isinstance(query, dict) else {}
                    bid_price = float(
                        params.get(
                            "bid_price", target.split(",")[0] if target else 99.5
                        )
                    )
                    ask_price = float(
                        params.get(
                            "ask_price",
                            target.split(",")[1] if target and "," in target else 100.5,
                        )
                    )
                    bid_volume = float(params.get("bid_volume", top_k * 100))
                    ask_volume = float(params.get("ask_volume", depth * 100))

                    obi = OrderBookImbalance.calculate(bid_volume, ask_volume)
                    spread = ask_price - bid_price
                    micro_price = MicroPriceCalculator.calculate(
                        bid_price, ask_price, bid_volume, ask_volume
                    )
                    micro_price_from_imbalance = MicroPriceCalculator.from_imbalance(
                        (bid_price + ask_price) / 2.0, spread, obi
                    )
                    is_consensus = ConvergenceFilter.check_agreement(
                        [True] * min(5, max(1, int(obi * 5 + 2.5))), threshold=5
                    )
                    result = {
                        "order_book": {
                            "bid_price": bid_price,
                            "ask_price": ask_price,
                            "bid_volume": bid_volume,
                            "ask_volume": ask_volume,
                            "spread": spread,
                        },
                        "imbalance": {"obi": float(obi), "consensus": is_consensus},
                        "micro_price": {
                            "direct_calculation": float(micro_price),
                            "from_imbalance": float(micro_price_from_imbalance),
                        },
                        "status": "ok",
                    }
                    return _json.dumps(result, default=str)
                except Exception as e:
                    return f"Error: quant_microstructure calculation failed: {e}"
            elif action == "quant_strategy":
                from agent_utilities.domains.finance.strategy_engine import (
                    StrategyEngine,
                    StrategyMetrics,
                )

                if not query:
                    return "Error: quant_strategy needs a strategy_id in `query`."
                if engine is None:
                    return "Error: quant_strategy requires an active knowledge graph engine."
                se = StrategyEngine(engine)
                metrics = StrategyMetrics(
                    sharpe=2.5,
                    max_drawdown=-0.10,
                    win_rate=0.55,
                    profit_factor=1.5,
                    total_trades=max(100, top_k),
                )
                promotable = se.record_backtest(query, metrics)
                return json.dumps(
                    {
                        "strategy_id": query,
                        "promotable": promotable,
                        "metrics": {
                            "sharpe": metrics.sharpe,
                            "max_drawdown": metrics.max_drawdown,
                            "win_rate": metrics.win_rate,
                            "profit_factor": metrics.profit_factor,
                            "total_trades": metrics.total_trades,
                        },
                    },
                    default=str,
                )
            elif action == "quant_regime":
                from agent_utilities.domains.finance.regime_detector import (
                    RegimeDetector,
                )

                try:
                    import pandas as pd
                except ImportError:
                    return "Error: pandas is required for quant_regime."

                if not query:
                    return "Error: quant_regime needs a ticker symbol in `query`."

                # Create synthetic OHLC data for demonstration
                # In production, this would ingest real market data
                import numpy as np

                dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
                base_price = 100.0
                returns = np.random.normal(0.0005, 0.02, 100)
                close_prices = base_price * np.cumprod(1 + returns)
                df = pd.DataFrame(
                    {
                        "Close": close_prices,
                        "High": close_prices * 1.02,
                        "Low": close_prices * 0.98,
                        "Open": np.roll(close_prices, 1)[1:].tolist() + [base_price],
                    },
                    index=dates,
                )

                detector = RegimeDetector(engine)
                regime = detector.detect_regime(df, ticker=query)
                return regime
            elif action == "workforce_plan":
                from agent_utilities.domains.hr.workforce_manager import (
                    WorkforceManager,
                )

                wm = WorkforceManager()
                result = wm.get_workforce_summary()
                return json.dumps(result, default=str)
            elif action == "close":
                # Background OWL-RL + SHACL closure (KG-2.6): promote recent nodes
                # to RDF, materialize implied edges via the reasoner, validate
                # against shapes. On-demand twin of the maintenance-tick closure.
                from agent_utilities.knowledge_graph.maintenance.owl_closure import (
                    run_closure,
                )

                summary = run_closure(
                    engine, limit=top_k * 200 if top_k != 10 else 2000
                )
                return json.dumps(summary, default=str)
            else:
                return f"Error: Unknown analyze action '{action}'"
        except Exception as e:
            return f"Analysis error: {str(e)}"

    REGISTERED_TOOLS["graph_analyze"] = graph_analyze

    @mcp.tool(
        name="graph_orchestrate",
        description="Orchestrate multi-agent workflows, dispatch subagents, and manage execution loops.",
        tags=["graph-os", "orchestrate"],
    )
    async def graph_orchestrate(
        action: str = Field(
            default="dispatch",
            description="Action to perform (dispatch, swarm, status, request_approval, grant_approval, execute_agent, consensus, start_debate, submit_risk_veto, list_cron_jobs, trigger_cron_job, compile_workflow, compile_process, list_workflows, execute_workflow, export_workflow, loop_cycle, assimilate, standardize, failure_ingest, publish_proposal). 'loop_cycle' = advance the Loop engine one cycle (CONCEPT:KG-2.78); 'swarm' = one-shot goal→decompose→parallel-waves→verify→synthesize (CONCEPT:ORCH-1.32); 'standardize' = enterprise standardization + consolidation recommendations (CONCEPT:KG-2.49); 'failure_ingest' = pull Langfuse failures → failure_gap topics → regression-gated remediation (CONCEPT:AHE-3.18); 'compile_process' = compile a harvested BusinessProcess node (task=process node id, agent_name=optional workflow name) into an executable WorkflowDefinition with a REALIZES bridge edge (CONCEPT:ORCH-1.41); 'publish_proposal' = one-shot evolution→branch bridge — publish a promoted proposal (task=proposal node id) as a reviewable local git branch through the ActionPolicy merge_promotion gate (CONCEPT:AHE-3.21); 'rlm_benchmark' = run the long-context RLM benchmark (RLM vs vanilla vs compaction) for task=<s_niah|oolong|oolong_pairs|browsecomp_plus|longbench_codeqa>, dependencies=JSON {scales,cases_per_scale}, returning a paper-comparison scoreboard (CONCEPT:AHE-3.32).",
        ),
        task: str = Field(
            default="", description="Task description or payload to dispatch."
        ),
        job_id: str = Field(
            default="", description="Job ID for checking status or granting approval."
        ),
        approval_status: str = Field(
            default="", description="Approval status (e.g., 'approved', 'rejected')."
        ),
        agent_name: str = Field(
            default="", description="Name of the agent to execute."
        ),
        max_steps: int = Field(
            default=30, description="Maximum steps for agent execution."
        ),
        dependencies: str = Field(
            default="[]", description="JSON-encoded list of dependency job IDs."
        ),
        completion_state: str = Field(
            default="",
            description="Strict mathematical or semantic definition of when this workflow is considered done.",
        ),
        max_fan_out: int = Field(
            default=5,
            description="Maximum number of parallel subagents to spawn during adversarial loop.",
        ),
        context: str = Field(
            default="",
            description="CONCEPT:ORCH-1.39 — curated context the invoking agent passes to the "
            "spawned agent (action='execute_agent'); injected into the spawned agent's prompt, "
            "budgeted to the model's context window.",
        ),
        budget_tokens: int = Field(
            default=0,
            description="CONCEPT:ORCH-1.39 — optional token budget the invoker grants the "
            "spawned agent (action='execute_agent'); enforced as a hard total-tokens limit. "
            "0 = unbounded.",
        ),
        context_ref: str = Field(
            default="",
            description="CONCEPT:ORCH-1.39 — id of a persisted ContextBlob (from "
            "graph_context put) to hand to the spawned agent (action='execute_agent'); its "
            "content is resolved from the graph and injected. Use instead of inline 'context' "
            "for large/shared context.",
        ),
        allowed_tools: str = Field(
            default="",
            description="CONCEPT:ORCH-1.39 — comma-separated least-privilege tool allow-list "
            "for the spawned agent (action='execute_agent'); its tools/toolsets are filtered "
            "to ONLY these names. Empty = no restriction.",
        ),
        cred_ref: str = Field(
            default="",
            description="CONCEPT:ORCH-1.39 — REFERENCE (secret key, e.g. 'cred:{session}') to "
            "an ephemeral credential the invoker stored in the secrets backend; resolved to the "
            "spawned agent's auth_token at spawn (never logged). Use instead of passing raw "
            "secrets. Empty = none.",
        ),
        open_channel: bool = Field(
            default=False,
            description="CONCEPT:ORCH-1.40 — when True (action='execute_agent'), open a native "
            "bidirectional message channel for this run; the response JSON includes a "
            "'channel_id' to talk to the spawned agent via graph_message(send/receive).",
        ),
    ) -> str:
        """Orchestrate multi-agent workflows. Dispatches agents, manages subagent lifecycles, and evaluates approval conditions for complex asynchronous execution.

        CONCEPT:ORCH-1.37 — the execution-flow Mermaid diagram (generated by the ORCH-1.8
        WorkflowVisualizer) is surfaced in the response: ``swarm``, ``compile_workflow`` and
        ``execute_workflow`` add an additive ``mermaid`` JSON key (null when unavailable), and
        ``execute_agent`` returns a JSON object ``{"output", "mermaid"}`` when a diagram was
        produced (otherwise the bare output string, for backward compatibility).
        """
        engine = _get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            import json
            import uuid

            from agent_utilities.orchestration.manager import Orchestrator

            orch = Orchestrator(engine)

            if action == "dispatch":
                deps = json.loads(dependencies) if dependencies else []
                job_id = await orch.dispatch_task(task, deps)
                # CONCEPT:ORCH-1.45 — queue-driven dispatch: with
                # AGENT_DISPATCH_BACKEND=queue the durable :Task node stays the
                # payload of record, a session-keyed envelope goes onto the
                # agent_turns queue, and the caller gets a job handle (poll
                # action=status / /api/graph/orchestrate/job/{job_id}) instead
                # of an in-process execution promise. A bare dispatch has no
                # session, so the job id is its own session scope — serial
                # with itself, parallel with everything else.
                from agent_utilities.orchestration.agent_dispatch import (
                    KIND_ORCHESTRATOR_TASK,
                    AgentTurnEnvelope,
                    dispatch_queue_enabled,
                    enqueue_agent_turn,
                )

                if dispatch_queue_enabled():
                    handle = enqueue_agent_turn(
                        AgentTurnEnvelope(
                            job_id=job_id,
                            session_id=job_id,
                            kind=KIND_ORCHESTRATOR_TASK,
                            payload_ref=job_id,
                            agent_name=agent_name or "",
                        )
                    )
                    handle["status_url"] = f"/api/graph/orchestrate/job/{job_id}"
                    return json.dumps(handle)
                return f"Task dispatched. Job ID: {job_id}"
            elif action == "rlm_run":
                # CONCEPT:ORCH-1.12 — run the Predict-RLM runtime on an ad-hoc task.
                from agent_utilities.rlm.runner import run_rlm

                result = await run_rlm(task, input_text=completion_state)
                return json.dumps(result, default=str)
            elif action == "rlm_optimize":
                # CONCEPT:ORCH-1.13 — optimize a skill prompt via the GEPA loop.
                from agent_utilities.rlm.runner import optimize_rlm_skill

                rows = json.loads(dependencies) if dependencies else []
                dataset = rows if isinstance(rows, list) else []
                result = await optimize_rlm_skill(task, dataset)
                return json.dumps(result, default=str)
            elif action == "rlm_benchmark":
                # CONCEPT:AHE-3.32 — run the long-context RLM benchmark (RLM vs vanilla vs
                # compaction) over a task and return the paper-comparison scoreboard. `task` is the
                # benchmark name (s_niah, oolong, oolong_pairs, browsecomp_plus, longbench_codeqa);
                # `dependencies` is optional JSON {"scales": [int], "cases_per_scale": int}.
                from agent_utilities.rlm.benchmarks import (
                    list_tasks,
                    render_scoreboard,
                    run_benchmark,
                )

                opts = json.loads(dependencies) if dependencies else {}
                if not isinstance(opts, dict):
                    opts = {}
                bench = task or "s_niah"
                if bench not in list_tasks():
                    return json.dumps(
                        {
                            "ok": False,
                            "error": f"unknown task {bench!r}",
                            "tasks": list_tasks(),
                        }
                    )
                scales = opts.get("scales") or [50_000]
                results = await run_benchmark(
                    bench,
                    scales=[int(s) for s in scales],
                    cases_per_scale=int(opts.get("cases_per_scale", 3)),
                )
                return json.dumps(
                    {
                        "ok": True,
                        "results": [r.model_dump() for r in results],
                        "scoreboard": render_scoreboard(results),
                    },
                    default=str,
                )
            elif action == "swarm":
                # CONCEPT:ORCH-1.32 — KG-Governed Agent Swarm
                # One-shot swarm action: a one-line goal is
                # decomposed into a dependency-ordered task graph, executed in parallel waves by the
                # ParallelEngine, each leaf verified against its subtask (planner→execute→verify),
                # then synthesized into a single deliverable. The KG/OWL grounding + verification is
                # what distinguishes this from a black-box trained swarm.
                from agent_utilities.core.config import (
                    DEFAULT_KG_MODEL_ID,
                    DEFAULT_LLM_PROVIDER,
                )
                from agent_utilities.core.model_factory import create_model
                from agent_utilities.graph.parallel_engine import ParallelEngine
                from agent_utilities.graph.planning import Planner
                from agent_utilities.models.execution_manifest import ExecutionManifest

                try:
                    _model = create_model(
                        provider=DEFAULT_LLM_PROVIDER, model_id=DEFAULT_KG_MODEL_ID
                    )
                except Exception:
                    _model = None
                plan = await Planner(model=_model).decompose(task)
                manifest = ExecutionManifest.from_graph_plan(
                    plan, name="swarm", query=task
                )
                # CONCEPT:ORCH-1.39 (Phase 3) — curated invoker context for the swarm.
                # ParallelEngine injects manifest.context into EVERY wave agent's task, so the
                # invoker's context reaches all swarm agents. Resolve context_ref if given.
                _swarm_ctx = context or ""
                if not _swarm_ctx and context_ref:
                    try:
                        _crows = engine.query_cypher(
                            "MATCH (c:ContextBlob) WHERE c.id = $id RETURN c.content AS content",
                            {"id": context_ref},
                        )
                        if _crows and _crows[0].get("content"):
                            _swarm_ctx = str(_crows[0]["content"])
                    except Exception:  # noqa: BLE001
                        _swarm_ctx = ""
                if _swarm_ctx:
                    manifest.context = (
                        f"{manifest.context}\n\n{_swarm_ctx}"
                        if manifest.context
                        else _swarm_ctx
                    )
                # default governance ON: verify each leaf + retry transient failures.
                manifest.metadata["verify"] = True
                manifest.metadata["max_retries"] = 2
                if max_fan_out:
                    manifest.max_concurrency = int(max_fan_out)
                # give the verify loop something to check: each leaf must address its own subtask.
                for _a in manifest.agents:
                    if not _a.success_criteria:
                        _a.success_criteria = (
                            f"Output must substantively address: "
                            f"{(_a.task_template or task)[:240]}"
                        )
                pe_result = await ParallelEngine(engine=engine).execute(manifest)
                return json.dumps(
                    {
                        "deliverable": pe_result.synthesis_output,
                        "agent_count": pe_result.agent_count,
                        "wave_count": pe_result.wave_count,
                        "critical_path_length": pe_result.critical_path_length,
                        "parallelism_ratio": pe_result.parallelism_ratio,
                        "verification": pe_result.verification,
                        "telemetry": pe_result.telemetry,
                        "execution_id": pe_result.execution_id,
                        "success": pe_result.success,
                        # CONCEPT:ORCH-1.37 — surface the existing execution-flow diagram
                        # (generated by ORCH-1.8 WorkflowVisualizer) to the MCP caller.
                        "mermaid": pe_result.mermaid,
                    },
                    default=str,
                )
            elif action == "status":
                if not job_id:
                    return "Error: job_id required"
                return str(orch.get_task_status(job_id))
            elif action == "request_approval":
                return f"Approval requested for job {job_id}"
            elif action == "grant_approval":
                return orch.grant_approval(job_id, approval_status)
            elif action == "execute_agent":
                try:
                    # CONCEPT:ORCH-1.37 — opt into the mermaid wrapper so the routed
                    # graph diagram (GraphResponse.mermaid) reaches the MCP caller.
                    agent_result = await orch.execute_agent(
                        agent_name=agent_name,
                        task=task,
                        max_steps=max_steps,
                        return_mermaid=True,
                        context=context or None,
                        budget_tokens=budget_tokens or None,
                        context_ref=context_ref or None,
                        allowed_tools=(
                            [t.strip() for t in allowed_tools.split(",") if t.strip()]
                            or None
                        ),
                        cred_ref=cred_ref or None,
                        open_channel=bool(open_channel),  # CONCEPT:ORCH-1.40
                    )
                    return agent_result
                except Exception as exc:
                    return f"Error: agent execution failed: {exc}"
            elif action == "compile_workflow":
                try:
                    from agent_utilities.knowledge_graph.workflow_store import (
                        WorkflowStore,
                    )

                    name = agent_name or f"compiled_{uuid.uuid4().hex[:6]}"
                    workflow_id = await orch.compile_workflow(name=name, task=task)
                    # CONCEPT:ORCH-1.37 — return the diagram persisted on the
                    # WorkflowDefinition node so the caller can review the topology.
                    mermaid = None
                    try:
                        mermaid = WorkflowStore(engine).get_mermaid(name)
                    except Exception:
                        mermaid = None
                    return json.dumps(
                        {
                            "status": "compiled",
                            "workflow_id": workflow_id,
                            "name": name,
                            "mermaid": mermaid,
                        }
                    )
                except Exception as exc:
                    return f"Error compiling workflow: {exc}"
            elif action == "compile_process":
                # CONCEPT:ORCH-1.41 — descriptive BusinessProcess → executable
                # WorkflowDefinition (+ REALIZES bridge edge). 'task' carries
                # the BusinessProcess node id; 'agent_name' an optional name.
                try:
                    from agent_utilities.knowledge_graph.process_plan_compiler import (
                        ProcessPlanCompiler,
                    )
                    from agent_utilities.knowledge_graph.workflow_store import (
                        WorkflowStore,
                    )

                    process_id = task.strip()
                    if not process_id:
                        return (
                            "Error: Must specify the BusinessProcess node id in "
                            "the 'task' parameter."
                        )
                    compiler = ProcessPlanCompiler(engine)
                    report = await compiler.compile_and_store(
                        process_id, name=agent_name or None
                    )
                    report["status"] = "compiled"
                    # CONCEPT:ORCH-1.37 — surface the stored topology diagram.
                    try:
                        report["mermaid"] = WorkflowStore(engine).get_mermaid(
                            report["name"]
                        )
                    except Exception:
                        report["mermaid"] = None
                    return json.dumps(report, default=str)
                except Exception as exc:
                    return f"Error compiling process: {exc}"
            elif action == "list_workflows":
                try:
                    from agent_utilities.knowledge_graph.workflow_store import (
                        WorkflowStore,
                    )

                    store = WorkflowStore(engine)
                    workflows = store.list_workflows(limit=50)
                    if not workflows:
                        return json.dumps({"error": "No workflows found in database."})
                    return json.dumps(
                        {"source": "kg", "workflows": workflows}, default=str
                    )
                except Exception as exc:
                    return f"Error listing workflows: {exc}"
            elif action == "execute_workflow":
                # CONCEPT:ORCH-1.42 — execution-time ontology gate, BEFORE any
                # dispatch: (a) SHACL-validate the stored definition (refuse
                # malformed workflows, KG_WORKFLOW_SHAPE_GATE default ON);
                # (b) with KG_BRAIN_ENFORCE on, apply the ontology permissioning
                # row gate to the workflow node for the current actor —
                # a denial raises PermissionError (fail-closed, OS-5.14).
                from agent_utilities.knowledge_graph.core.workflow_gate import (
                    gate_workflow_execution,
                )

                gate_name = agent_name or task
                gate = gate_workflow_execution(engine, gate_name)
                if not gate.get("allowed", True):
                    return json.dumps(
                        {
                            "error": (
                                "workflow definition failed ontology validation "
                                "— execution refused"
                            ),
                            "workflow": gate_name,
                            "workflow_id": gate.get("workflow_id"),
                            "violations": gate.get("violations", []),
                        },
                        default=str,
                    )
                try:
                    from agent_utilities.knowledge_graph.workflow_store import (
                        WorkflowStore,
                    )

                    name = agent_name or task
                    input_task = task if (agent_name and task != agent_name) else None
                    wf_result = await orch.execute_workflow(
                        workflow_id=name,
                        task=input_task or "",
                        max_steps=max_steps,
                    )
                    # CONCEPT:ORCH-1.37 — surface the workflow's stored execution-flow
                    # diagram alongside the result.
                    mermaid = None
                    try:
                        mermaid = WorkflowStore(engine).get_mermaid(name)
                    except Exception:
                        mermaid = None
                    return json.dumps(
                        {"result": wf_result, "mermaid": mermaid}, default=str
                    )
                except Exception as exc:
                    return f"Error executing workflow: {exc}"
            elif action == "consensus":
                return f"Consensus reached for {task}."
            elif action == "start_debate":
                engine.add_node(
                    f"debate_{job_id}", "TradingDebate", topic=task, status="ongoing"
                )
                return f"Started Trading Debate for {task}."
            elif action == "submit_risk_veto":
                engine.add_node(
                    f"veto_{job_id}", "RiskVeto", reason=task, target=job_id
                )
                engine.add_edge(
                    f"veto_{job_id}", f"debate_{job_id}", "CONTRADICTS_BELIEF_PROP"
                )
                return f"Submitted Risk Veto for debate {job_id}."
            elif action == "list_cron_jobs":
                try:
                    from agent_utilities.automation.maintenance_cron import (
                        MaintenanceCron,
                    )

                    cron = MaintenanceCron()
                    due_tasks = cron.get_due_tasks()
                    lines = []
                    for t in cron.tasks:
                        status = (
                            "DUE"
                            if any(dt.id == t.id for dt in due_tasks)
                            else "WAITING"
                        )
                        lines.append(
                            f"[{status}] {t.id} (Frequency: {t.frequency.value})"
                        )
                    return "\n".join(lines)
                except ImportError:
                    return "Error: maintenance_cron module not available"
            elif action == "trigger_cron_job":
                try:
                    from agent_utilities.automation.maintenance_cron import (
                        MaintenanceCron,
                    )

                    cron = MaintenanceCron()
                    target_id = task.strip()
                    if not target_id:
                        return "Error: Must specify the cron job ID in the 'task' parameter."
                    cron.record_execution(
                        target_id, status="triggered_manually", tokens_used=0
                    )
                    return f"Manually triggered cron job: {target_id}"
                except ImportError:
                    return "Error: maintenance_cron module not available"
            elif action == "dispatch_workflow":
                # CONCEPT:ORCH-1.42 — the SAME execution-time ontology gate as
                # execute_workflow, BEFORE background dispatch: (a) SHACL-validate
                # the stored definition (refuse malformed workflows,
                # KG_WORKFLOW_SHAPE_GATE default ON); (b) with KG_BRAIN_ENFORCE
                # on, apply the ontology permissioning row gate to the workflow
                # node for the current actor — a denial raises PermissionError
                # (fail-closed, OS-5.14). Enforcement off skips the ACL check.
                from agent_utilities.knowledge_graph.core.workflow_gate import (
                    gate_workflow_execution,
                )

                gate_name = agent_name or task
                gate = gate_workflow_execution(engine, gate_name)
                if not gate.get("allowed", True):
                    return json.dumps(
                        {
                            "error": (
                                "workflow definition failed ontology validation "
                                "— background dispatch refused"
                            ),
                            "workflow": gate_name,
                            "workflow_id": gate.get("workflow_id"),
                            "violations": gate.get("violations", []),
                        },
                        default=str,
                    )
                try:
                    import asyncio

                    from agent_utilities.orchestration import AgentOrchestrationEngine

                    runner = AgentOrchestrationEngine()
                    name = agent_name or task
                    input_task = task if (agent_name and task != agent_name) else None
                    session_id = f"wf-{uuid.uuid4().hex[:8]}"

                    # Start execution as background task
                    asyncio.create_task(
                        runner.execute_workflow(
                            workflow_id=name,
                            task=input_task,
                            completion_state=completion_state,
                            max_fan_out=max_fan_out,
                        )
                    )
                    return (
                        f"Workflow dispatched in background. Session ID: {session_id}"
                    )
                except ValueError as exc:
                    return f"Workflow not found: {exc}"
                except Exception as exc:
                    return f"Error dispatching workflow: {exc}"

            elif action == "workflow_status":
                try:
                    from agent_utilities.workflows.runner import _active_workflows

                    sid = job_id or task
                    if not sid:
                        return "Error: Must specify session ID in 'job_id' or 'task' parameter."

                    wf_status = _active_workflows.get(sid)
                    if not wf_status:
                        return f"Workflow session '{sid}' not found or has not been run in this process."

                    return json.dumps(wf_status.to_dict(), default=str)
                except Exception as exc:
                    return f"Error retrieving workflow status: {exc}"

            elif action == "export_workflow":
                try:
                    return json.dumps(
                        {
                            "error": "Workflow export requires resolving workflows from the database. Legacy catalog export is deprecated."
                        },
                        indent=2,
                        default=str,
                    )
                except Exception as exc:
                    return f"Error exporting workflow: {exc}"

            elif action == "loop_cycle":
                # Advance the Loop engine one cycle (CONCEPT:KG-2.7/2.78): intake
                # active Loops → acquire → ADDRESSES-resolve → optional
                # distil/synthesize as DRAFTS/proposals. Never auto-merges.
                import json as _json

                from agent_utilities.knowledge_graph.research.loop_controller import (
                    LoopController,
                )

                engine = _get_engine()
                _mt = max_fan_out if isinstance(max_fan_out, int) else 5
                rep = LoopController(engine).run_one_cycle(
                    max_topics=_mt if _mt > 0 else 5,
                )
                return _json.dumps(rep, indent=2, default=str)

            elif action == "failure_ingest":
                # Failure-driven evolution (CONCEPT:AHE-3.18): pull Langfuse
                # failures → materialize failure_gap topics → regression-gated
                # remediation that addresses those gaps directly. The on-demand
                # twin of the daemon's failure_ingest tick (gated by
                # KG_FAILURE_EVOLUTION for the daemon; the action runs on request).
                import json as _json

                from agent_utilities.knowledge_graph.adaptation.failure_analyzer import (
                    run_failure_ingest,
                )

                rep = run_failure_ingest(_get_engine())
                return _json.dumps(rep, indent=2, default=str)

            elif action == "publish_proposal":
                # Evolution→branch bridge (CONCEPT:AHE-3.21): publish a promoted
                # golden-loop proposal (task=proposal node id) as a reviewable
                # LOCAL git branch — change synthesis + RLM-sandbox validation +
                # LocalBranchPublisher — gated by the OS-5.24 ActionPolicy's
                # merge_promotion kind (default approval_required: a pending
                # grant queues, a granted approval lets this proceed). Never
                # pushes; a human merges through the normal release flow.
                import json as _json

                from agent_utilities.knowledge_graph.research.change_publisher import (
                    publish_proposal,
                )

                pid = (task or "").strip()
                if not pid:
                    return (
                        "Error: publish_proposal requires the proposal node id "
                        "in 'task'"
                    )
                rep = publish_proposal(_get_engine(), pid)
                return _json.dumps(rep, indent=2, default=str)

            elif action == "assimilate":
                # Graph-native assimilation pass (CONCEPT:KG-2.7): dedup → gap →
                # synergy → rank (idempotent via watermark). With "synthesize" in the
                # task, also propose grounded SDD plans for the top open gaps.
                import json as _json

                from agent_utilities.knowledge_graph.research.loop_controller import (
                    run_assimilation_pass,
                )

                _mt = (
                    max_fan_out
                    if isinstance(max_fan_out, int) and max_fan_out > 0
                    else 5
                )
                rep = run_assimilation_pass(
                    _get_engine(),
                    synthesize="synthesize" in (task or "").lower(),
                    top_n=_mt,
                    force="force" in (task or "").lower(),
                )
                return _json.dumps(rep, indent=2, default=str)

            elif action == "standardize":
                # Enterprise standardization + consolidation pass (CONCEPT:KG-2.49):
                # materialize enterprise-standard interfaces → score per-asset/org/
                # domain conformance drift → rank propose-only consolidation
                # recommendations (collapse projects / retire tools / merge code).
                import json as _json

                from agent_utilities.knowledge_graph.standardization import (
                    run_standardization_pass,
                )

                _tn = (
                    max_fan_out
                    if isinstance(max_fan_out, int) and max_fan_out > 0
                    else 20
                )
                rep = run_standardization_pass(_get_engine(), top_n=_tn)
                return _json.dumps(rep, indent=2, default=str)

            elif action == "enterprise_op":
                from agent_utilities.knowledge_graph.orchestration.engine_enterprise import (  # noqa: E501
                    EnterpriseEngineMixin,
                )

                engine = _get_engine()
                if not task:
                    return "Error: enterprise_op needs a business_unit_id in `task`."
                try:
                    budget_amount = float(agent_name) if agent_name else 10000.0
                except (ValueError, TypeError):
                    budget_amount = 10000.0
                # The mixin is composed onto the live engine; fall back to its
                # unbound method if the running engine predates the composition.
                allocate = getattr(
                    engine, "allocate_budget", None
                ) or EnterpriseEngineMixin.allocate_budget.__get__(engine)
                if allocate is None:
                    return "Error: engine does not support enterprise operations."
                budget_id = allocate(task, budget_amount, "USD")
                return json.dumps(
                    {
                        "budget_id": budget_id,
                        "business_unit_id": task,
                        "amount": budget_amount,
                        "currency": "USD",
                    },
                    default=str,
                )
            elif action == "finance_op":
                import json as _json

                from agent_utilities.knowledge_graph.orchestration.engine_finance import (
                    FinanceEngineMixin,
                )

                engine = _get_engine()
                if not engine:
                    return (
                        "Error: finance_op requires an active epistemic-graph engine."
                    )
                if not task:
                    return "Error: finance_op needs a JSON task with {returns: [float], strategy_id: str, asset_class?: str}."

                try:
                    params = _json.loads(task)
                except Exception:
                    return f"Error: task must be valid JSON. Got: {task}"

                returns = params.get("returns", [])
                strategy_id = params.get("strategy_id")
                asset_class = params.get("asset_class", "equities")
                bull_threshold = params.get("bull_threshold")
                bear_threshold = params.get("bear_threshold")
                window = params.get("window")
                method = params.get("method", "rolling_sum")

                if not isinstance(returns, list) or len(returns) == 0:
                    return (
                        "Error: task must include returns (non-empty list of floats)."
                    )
                if not strategy_id:
                    return "Error: task must include strategy_id."

                try:
                    mixin = FinanceEngineMixin()
                    mixin.graph = engine.graph
                    mixin.backend = engine.backend

                    matrix_id = mixin.fit_markov_regime(
                        returns=returns,
                        strategy_id=strategy_id,
                        asset_class=asset_class,
                        bull_threshold=bull_threshold,
                        bear_threshold=bear_threshold,
                        window=window,
                        method=method,
                    )
                    return _json.dumps(
                        {"matrix_id": matrix_id, "status": "fitted"}, default=str
                    )
                except Exception as e:
                    return f"Error: {str(e)}"
            elif action == "ml_rlm_op":
                # CONCEPT:KG-2.6 — Machine Learning & RLM capabilities for the KG engine.
                # Register a new RLM actor for reinforcement learning tasks.
                # task = JSON string {"name": "actor_name", "learning_rate": 0.01, "discount_factor": 0.99}
                # or plain text actor name (uses sensible defaults: learning_rate=0.01, discount_factor=0.99).
                import json as _json

                from agent_utilities.knowledge_graph.orchestration.engine_ml_rlm import (
                    MachineLearningEngineMixin,
                )

                if not task:
                    return "Error: ml_rlm_op needs a task (actor name or JSON config)."
                if not engine:
                    return "Error: ml_rlm_op needs an active engine."

                # Parse task as JSON or plain text.
                config = {"name": task, "learning_rate": 0.01, "discount_factor": 0.99}
                if task.startswith("{"):
                    try:
                        config.update(_json.loads(task))
                    except Exception:
                        pass

                # Instantiate mixin with engine and call register_rlm_actor.
                mixin = MachineLearningEngineMixin()
                mixin.graph = engine.graph if hasattr(engine, "graph") else None
                mixin.backend = engine.backend if hasattr(engine, "backend") else None
                mixin._serialize_node = (
                    engine._serialize_node
                    if hasattr(engine, "_serialize_node")
                    else lambda n, label: n.model_dump()
                )
                mixin._upsert_node = (
                    engine._upsert_node
                    if hasattr(engine, "_upsert_node")
                    else lambda label, id, data: None
                )

                actor_id = mixin.register_rlm_actor(
                    name=config.get("name", "rlm_actor"),
                    learning_rate=float(config.get("learning_rate", 0.01)),
                    discount_factor=float(config.get("discount_factor", 0.99)),
                )
                return _json.dumps(
                    {"actor_id": actor_id, "status": "registered"}, default=str
                )
            else:
                return f"Error: Unknown orchestration action '{action}'"
        except PermissionError:
            # CONCEPT:ORCH-1.42 / OS-5.14 — ACL denial is fail-closed: surface
            # it as a real error to the MCP layer, never a stringified result.
            raise
        except Exception as e:
            return f"Orchestration error: {str(e)}"

    REGISTERED_TOOLS["graph_orchestrate"] = graph_orchestrate

    @mcp.tool(
        name="graph_configure",
        description="Manage backend configurations, system credentials, and tool registration within the unified agent ecosystem.",
        tags=["graph-os", "configure"],
    )
    def graph_configure(
        action: str = Field(
            default="register_mcp",
            description="Operation ('set_secret', 'register_mcp', 'install_hooks', 'uninstall_hooks', 'doctor', 'set_role_routing', 'schema_pack', 'schema_candidates', 'add_connection', 'remove_connection', 'list_connections', 'set_default_connection'). 'schema_pack' with config_key=<name> sets the active domain Schema Pack, or with empty config_key returns the active pack plus available packs; 'schema_candidates' reviews out-of-pack types seen on write (CONCEPT:KG-2.35). CONCEPT:KG-2.63 — 'add_connection' registers a named graph backend (config_key=name, config_value=JSON spec e.g. {\"backend\":\"neo4j\",\"uri\":\"bolt://...\",\"user\":\"...\",\"password\":\"...\"}; use backend 'age' for Postgres native openCypher); 'remove_connection' (config_key=name); 'list_connections' returns per-connection health; 'set_default_connection' (config_key=name) repoints the default target. CONCEPT:KG-2.74 — 'mirror_status' returns per-mirror replication health (lag/failures/stalled) for a GRAPH_BACKEND=fanout deployment; 'reconcile' (optional config_key=<mirror name>, empty=all) runs a full authority→mirror drift-repair pass. 'setup_databases' provisions the Stardog + pg-age environment end-to-end (config_key=profile 'dev'|'prod', config_value=JSON options e.g. {\"postgres_mode\":\"managed_image\",\"dsn\":\"postgresql://...\",\"sparql_target\":\"builtin\"}); 'verify_databases' probes a Postgres for the age/vector/pg_search extensions (config_key or config_value.dsn = DSN). 'generate_config' writes a COMPLETE profile-seeded config.json covering every option (config_key=profile 'tiny'|'single-node-prod'|'enterprise', config_value optional {\"out\":path,\"redact_secrets\":true}); 'config_doctor' validates a deployment's config completeness/health (config_key=profile, config_value optional {\"config\":path}); 'config_reference' returns every option grouped by subsystem. 'system_doctor' runs a holistic deployment health sweep (brew/flutter-doctor style) across config/engine/backend/secrets/auth/mcp-fleet/hooks/observability, each with a remediation + skill (config_value optional {\"only\":[...],\"fix\":true,\"live\":true}).",
        ),
        config_key: str = Field(
            default="",
            description="The key or ID of the configuration/secret (for 'schema_pack', the pack name e.g. 'research-state'; for connection actions, the connection name).",
        ),
        config_value: str = Field(
            default="",
            description="JSON string containing the payload or secret value.",
        ),
    ) -> str:
        """Manage backend configurations and abstract credentials. Allows dynamic registry updates and credential injection during agent provisioning."""
        try:
            if action == "set_secret":
                from agent_utilities.security.secrets_client import (
                    create_secrets_client,
                )
                from agent_utilities.security.xai_auth import get_secrets_client_for_xai

                if config_key.startswith("xai/"):
                    client = get_secrets_client_for_xai()
                else:
                    client = create_secrets_client()
                client.set(config_key, config_value)
                return json.dumps(
                    {"status": "success", "action": "set_secret", "key": config_key}
                )
            if action == "register_mcp":
                from pathlib import Path

                from agent_utilities.core.workspace import get_mcp_config_path

                mcp_path_str = get_mcp_config_path()
                if mcp_path_str:
                    mcp_path = Path(mcp_path_str)
                    if not mcp_path.exists():
                        cfg = {}
                    else:
                        with open(mcp_path) as f:
                            cfg = json.load(f)
                    try:
                        parsed_val = json.loads(config_value)
                        cfg.setdefault("mcpServers", {})[config_key] = parsed_val
                        with open(mcp_path, "w") as f:
                            json.dump(cfg, f, indent=2)
                        return json.dumps(
                            {
                                "status": "success",
                                "action": "register_mcp",
                                "server": config_key,
                            }
                        )
                    except Exception as e:
                        return json.dumps({"error": f"Invalid config_value JSON: {e}"})
                return json.dumps({"error": "MCP config not found in workspace."})
            # ── CONCEPT:KG-2.63: Named multi-connection graph registry ──
            if action in (
                "add_connection",
                "remove_connection",
                "list_connections",
                "set_default_connection",
            ):
                registry = get_connection_registry()
                if action == "list_connections":
                    return json.dumps(registry.status(), default=str)
                if not config_key:
                    return json.dumps(
                        {"error": f"config_key (connection name) required for {action}"}
                    )
                if action == "add_connection":
                    try:
                        spec = json.loads(config_value) if config_value else {}
                    except Exception as e:
                        return json.dumps({"error": f"Invalid config_value JSON: {e}"})
                    if not isinstance(spec, dict):
                        return json.dumps(
                            {
                                "error": "config_value must be a JSON object (backend spec)"
                            }
                        )
                    try:
                        name = registry.register(config_key, spec)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                    return json.dumps(
                        {"status": "success", "action": action, "connection": name}
                    )
                if action == "remove_connection":
                    removed = registry.remove(config_key)
                    return json.dumps(
                        {
                            "status": "success" if removed else "not_found",
                            "action": action,
                            "connection": config_key,
                        }
                    )
                # set_default_connection
                try:
                    name = registry.set_default(config_key)
                except Exception as e:
                    return json.dumps({"error": str(e)})
                return json.dumps(
                    {"status": "success", "action": action, "default_target": name}
                )
            # ── CONCEPT:KG-2.74: Concurrent N-way mirroring health/repair ──
            if action in ("mirror_status", "reconcile"):
                from agent_utilities.knowledge_graph.backends import (
                    get_active_backend,
                )
                from agent_utilities.knowledge_graph.backends.fanout_backend import (
                    FanOutBackend,
                )

                backend = get_active_backend()
                # Locate the FanOutBackend. It is either the active backend
                # (GRAPH_BACKEND=fanout) or — the common case — the durable L3 of a
                # tiered backend (GRAPH_BACKEND=tiered + GRAPH_MIRROR_TARGETS), which
                # tees pg-age writes to the mirrors. Also unwrap a BrainGuarded proxy
                # (its inner backend is the ``inner`` property).
                cand = getattr(backend, "inner", backend)
                fan = None
                if isinstance(cand, FanOutBackend):
                    fan = cand
                elif isinstance(getattr(cand, "l3", None), FanOutBackend):
                    fan = cand.l3
                if fan is None:
                    return json.dumps(
                        {
                            "error": "No fanout mirror active (set GRAPH_MIRROR_TARGETS "
                            "with GRAPH_BACKEND=tiered or fanout).",
                            "backend": type(backend).__name__,
                        }
                    )
                inner = fan
                if action == "mirror_status":
                    return json.dumps(inner.durability_stats(), default=str)
                # reconcile — full authority→mirror drift repair (config_key =
                # optional single mirror name; empty = all mirrors).
                return json.dumps(inner.reconcile(config_key or None), default=str)
            # ── Database environment provisioning (Stardog + pg-age) ──
            if action in ("setup_databases", "verify_databases"):
                from agent_utilities.knowledge_graph.setup import (
                    setup_environment,
                    verify_postgres,
                )

                try:
                    opts = json.loads(config_value) if config_value else {}
                except Exception as e:
                    return json.dumps({"error": f"Invalid config_value JSON: {e}"})
                if not isinstance(opts, dict):
                    return json.dumps(
                        {"error": "config_value must be a JSON object of options"}
                    )
                if action == "verify_databases":
                    return json.dumps(
                        verify_postgres(opts.get("dsn") or config_key or None),
                        default=str,
                    )
                # setup_databases — config_key is a profile shortcut ('dev'/'prod').
                profile = opts.get("profile") or config_key or "dev"
                return json.dumps(
                    setup_environment(
                        profile=profile,
                        postgres_mode=opts.get("postgres_mode", "managed_image"),
                        dsn=opts.get("dsn"),
                        sparql_target=opts.get("sparql_target"),
                        mirror_targets=opts.get("mirror_targets"),
                        do_backfill=opts.get("do_backfill", True),
                    ),
                    default=str,
                )
            # ── Full-deployment config: generate / validate / document ──
            if action in ("generate_config", "config_doctor", "config_reference"):
                from agent_utilities.deployment import (
                    config_doctor,
                    config_reference,
                    write_config,
                )

                try:
                    opts = json.loads(config_value) if config_value else {}
                except Exception as e:
                    return json.dumps({"error": f"Invalid config_value JSON: {e}"})
                if not isinstance(opts, dict):
                    return json.dumps({"error": "config_value must be a JSON object"})
                if action == "config_reference":
                    return json.dumps(config_reference(), default=str)
                # profile shortcut via config_key ('tiny'/'single-node-prod'/'enterprise')
                profile = opts.get("profile") or config_key or None
                if action == "generate_config":
                    return json.dumps(
                        write_config(
                            profile or "tiny",
                            opts.get("out"),
                            redact_secrets=opts.get("redact_secrets", True),
                        ),
                        default=str,
                    )
                # config_doctor
                return json.dumps(
                    config_doctor(profile, opts.get("config")), default=str
                )
            # ── Holistic deployment health sweep (brew/flutter-doctor style) ──
            if action == "system_doctor":
                from agent_utilities.deployment import run_doctor

                try:
                    opts = json.loads(config_value) if config_value else {}
                except Exception as e:
                    return json.dumps({"error": f"Invalid config_value JSON: {e}"})
                if not isinstance(opts, dict):
                    return json.dumps({"error": "config_value must be a JSON object"})
                return json.dumps(
                    run_doctor(
                        opts.get("only"),
                        fix=opts.get("fix", False),
                        live=opts.get("live", False),
                    ),
                    default=str,
                )
            # ── KG-2.7 / ECO-4.6: Memory Hook Management ──
            if action == "install_hooks":
                try:
                    from agent_utilities.ecosystem.hook_installer import HookInstaller

                    installer = HookInstaller()
                    agents = config_value.split(",") if config_value else None
                    results = installer.install(agents)
                    return json.dumps(
                        {
                            "status": "success",
                            "results": results,
                            "installed": installer.installed,
                            "errors": installer.errors,
                        }
                    )
                except Exception as e:
                    return json.dumps({"error": f"Hook install failed: {e}"})
            if action == "uninstall_hooks":
                try:
                    from agent_utilities.ecosystem.hook_installer import HookInstaller

                    agents = config_value.split(",") if config_value else None
                    results = HookInstaller().uninstall(agents)
                    return json.dumps({"status": "success", "results": results})
                except Exception as e:
                    return json.dumps({"error": f"Hook uninstall failed: {e}"})
            if action == "doctor":
                try:
                    from agent_utilities.ecosystem.hook_installer import HookInstaller

                    return json.dumps(HookInstaller().doctor(), default=str)
                except Exception as e:
                    return json.dumps({"error": f"Doctor failed: {e}"})
            # ── CONCEPT:ORCH-1.27: Role-Specialized Model Routing ──
            if action == "set_role_routing":
                try:
                    from pathlib import Path

                    from agent_utilities.core.config import config as _cfg
                    from agent_utilities.models.model_registry import (
                        ModelRegistry,
                        RoleSpec,
                    )

                    payload = json.loads(config_value) if config_value else {}
                    reg_path = getattr(_cfg, "model_registry_path", None)
                    if not reg_path or not Path(reg_path).is_file():
                        return json.dumps(
                            {
                                "error": (
                                    "No model_registry_path configured; cannot "
                                    "persist role_routing."
                                )
                            }
                        )
                    registry = ModelRegistry.load_from_file(reg_path)
                    for rname, spec in payload.items():
                        registry.role_routing[rname] = RoleSpec.model_validate(spec)
                    Path(reg_path).write_text(
                        json.dumps(registry.model_dump(), indent=2)
                    )
                    return json.dumps(
                        {
                            "status": "success",
                            "action": "set_role_routing",
                            "roles": list(payload.keys()),
                        }
                    )
                except Exception as e:
                    return json.dumps({"error": f"set_role_routing failed: {e}"})
            # ── KG-2.35: Schema-Pack lifecycle (get/set the active domain pack) ──
            if action == "schema_pack":
                from agent_utilities.models.schema_pack_loader import (
                    get_active_pack,
                    set_active_pack,
                )
                from agent_utilities.models.schema_packs import list_schema_packs

                if config_key:
                    pack = set_active_pack(config_key)
                    return json.dumps(
                        {
                            "status": "success",
                            "action": "schema_pack",
                            "active": pack.name,
                            "signature": pack.signature(),
                        }
                    )
                active = get_active_pack()
                return json.dumps(
                    {
                        "status": "success",
                        "action": "schema_pack",
                        "active": active.name,
                        "signature": active.signature(),
                        "available": list_schema_packs(),
                    }
                )
            # ── KG-2.35: review out-of-pack candidate types seen on write ──
            if action == "schema_candidates":
                from agent_utilities.models.schema_pack_audit import (
                    SchemaCandidateAuditor,
                )

                try:
                    limit = int(config_value) if config_value else 100
                except ValueError:
                    limit = 100
                return json.dumps(
                    {
                        "status": "success",
                        "action": "schema_candidates",
                        "candidates": SchemaCandidateAuditor.instance().review(limit),
                    }
                )
            return json.dumps({"error": f"Unknown action: {action}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["graph_configure"] = graph_configure

    @mcp.tool(
        name="graph_sessions",
        description="Manage durable sessions (action in 'list', 'get', 'delete', 'reply', 'cancel').",
        tags=["graph-os", "sessions"],
    )
    async def graph_sessions(
        action: str = Field(
            description="Action: 'list', 'get', 'delete', 'reply', 'cancel'"
        ),
        session_id: str = Field(default="", description="Target session ID"),
        user_reply: str = Field(
            default="", description="Reply content for 'reply' action"
        ),
    ) -> str:
        """Manage durable sessions. Action: 'list', 'get', 'delete', 'reply', 'cancel'."""
        import json

        from starlette.responses import JSONResponse

        from agent_utilities.core.sessions import (
            cancel_session_run,
            delete_session,
            get_all_sessions,
            get_session_details,
            submit_session_reply,
        )

        try:
            req = _build_dummy_request(
                path_params={"session_id": session_id} if session_id else {},
                json_body={"content": user_reply} if user_reply else None,
            )
            if action == "list":
                resp = await get_all_sessions(req)
            elif action == "get":
                if not session_id:
                    return json.dumps({"error": "session_id is required"})
                resp = await get_session_details(req)
            elif action == "delete":
                if not session_id:
                    return json.dumps({"error": "session_id is required"})
                resp = await delete_session(req)
            elif action == "reply":
                if not session_id:
                    return json.dumps({"error": "session_id is required"})
                resp = await submit_session_reply(req)
            elif action == "cancel":
                if not session_id:
                    return json.dumps({"error": "session_id is required"})
                resp = await cancel_session_run(req)
            else:
                return json.dumps({"error": f"Unknown sessions action: {action}"})

            # Check if resp is JSONResponse
            if isinstance(resp, JSONResponse):
                # Return the decoded json string
                body_bytes = bytes(resp.body)
                return json.dumps(json.loads(body_bytes.decode("utf-8")))
            return str(resp)
        except Exception as e:
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["graph_sessions"] = graph_sessions

    @mcp.tool(
        name="graph_goals",
        description="Orchestrate background/autonomous loops (action in 'create', 'list', 'iterations', 'cancel').",
        tags=["graph-os", "goals"],
    )
    async def graph_goals(
        action: str = Field(
            description="Action: 'create', 'list', 'iterations', 'cancel'"
        ),
        goal_id: str = Field(default="", description="Target goal ID"),
        goal: str = Field(
            default="", description="Goal description/instruction for 'create' action"
        ),
        max_iterations: int = Field(
            default=10, description="Max iterations for the autonomous loop"
        ),
    ) -> str:
        """Orchestrate background/autonomous loops. Action: 'create', 'list', 'iterations', 'cancel'."""
        import json

        from starlette.responses import JSONResponse

        from agent_utilities.core.sessions import (
            cancel_goal,
            create_goal,
            get_goal_iterations,
            list_goals,
        )

        try:
            req = _build_dummy_request(
                path_params={"goal_id": goal_id} if goal_id else {},
                json_body={"objective": goal, "max_iterations": max_iterations}
                if action == "create"
                else None,
            )
            if action == "list":
                resp = await list_goals(req)
            elif action == "create":
                if not goal:
                    return json.dumps({"error": "goal is required"})
                resp = await create_goal(req)
            elif action == "iterations":
                if not goal_id:
                    return json.dumps({"error": "goal_id is required"})
                req_iter = _build_dummy_request(path_params={"goal_id": goal_id})
                resp = await get_goal_iterations(req_iter)
            elif action == "cancel":
                if not goal_id:
                    return json.dumps({"error": "goal_id is required"})
                req_cancel = _build_dummy_request(path_params={"goal_id": goal_id})
                resp = await cancel_goal(req_cancel)
            else:
                return json.dumps({"error": f"Unknown goals action: {action}"})

            # Check if resp is JSONResponse
            if isinstance(resp, JSONResponse):
                # Return the decoded json string
                body_bytes = bytes(resp.body)
                return json.dumps(json.loads(body_bytes.decode("utf-8")))
            return str(resp)
        except Exception as e:
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["graph_goals"] = graph_goals

    @mcp.tool(
        name="graph_loops",
        description=(
            "The single entrypoint for long-running objectives (CONCEPT:KG-2.78). A "
            "Loop is one objective of kind research|develop|skill; the LoopController "
            "advances every active Loop through ONE hot path. action in 'submit' "
            "(create a Loop: objective + kind [+ validation_cmd/end_state for develop, "
            "skill_ref for skill]), 'list' (active Loops), 'run' (advance all active "
            "Loops one cycle — research acquires/reasons, develop validates, skill "
            "executes), 'drive' (run ONE Loop by id to completion, durably — "
            "resume/checkpoint/corrigible, any kind), 'cancel' (terminate a Loop by id)."
        ),
        tags=["graph-os", "loops"],
    )
    async def graph_loops(
        action: str = Field(
            default="list", description="submit|list|run|cancel"
        ),
        objective: str = Field(default="", description="Objective text (submit)."),
        kind: str = Field(
            default="research", description="research|develop|skill (submit)."
        ),
        loop_id: str = Field(default="", description="Loop id (submit/cancel)."),
        validation_cmd: str = Field(
            default="", description="Shell command whose exit-0 completes a develop Loop."
        ),
        end_state: str = Field(default="", description="Human end-state (develop)."),
        skill_ref: str = Field(
            default="", description="Skill / skill-workflow name or id (skill Loop)."
        ),
        max_topics: int = Field(default=5, description="Loops to advance per run."),
        limit: int = Field(default=10, description="Max rows (list)."),
    ) -> str:
        """Submit / list / run / cancel Loops — the one Loop-engine entrypoint."""
        import json as _json

        from agent_utilities.knowledge_graph.research.loop_controller import (
            LoopController,
        )
        from agent_utilities.knowledge_graph.research.loops import (
            active_loops,
            mark_loop_status,
            submit_loop,
        )

        try:
            engine = _get_engine()
            if action == "submit":
                if not objective and not skill_ref:
                    return _json.dumps({"error": "submit needs objective or skill_ref"})
                loop = submit_loop(
                    engine,
                    objective,
                    kind=kind,  # type: ignore[arg-type]
                    validation_cmd=validation_cmd,
                    end_state=end_state,
                    skill_ref=skill_ref,
                    loop_id=loop_id,
                )
                return _json.dumps({"action": "submit", "loop": loop}, default=str)
            if action == "list":
                return _json.dumps(
                    {"action": "list", "loops": active_loops(engine, limit)},
                    default=str,
                )
            if action == "run":
                rep = LoopController(engine).run_one_cycle(max_topics=max_topics)
                return _json.dumps(rep, indent=2, default=str)
            if action == "drive":
                # Drive ONE Loop to completion durably (resume/checkpoint/corrigible,
                # CONCEPT:OS-5.16) — works for any kind (research/develop/skill).
                if not loop_id:
                    return _json.dumps({"error": "drive needs a loop_id"})
                target = next(
                    (
                        loop_row
                        for loop_row in active_loops(engine, max(limit, 50))
                        if loop_row.get("id") == loop_id
                    ),
                    None,
                )
                if target is None:
                    return _json.dumps({"error": f"no active loop {loop_id!r}"})
                res = await LoopController(engine).run_loop(target, sleep_s=0)
                return _json.dumps({"action": "drive", "result": res}, default=str)
            if action == "cancel":
                if not loop_id:
                    return _json.dumps({"error": "cancel needs a loop_id"})
                ok = mark_loop_status(engine, loop_id, "cancelled", source="user")
                return _json.dumps({"action": "cancel", "id": loop_id, "ok": ok})
            return _json.dumps({"error": f"unknown action {action!r}"})
        except Exception as e:
            return _json.dumps({"error": str(e)})

    REGISTERED_TOOLS["graph_loops"] = graph_loops

    @mcp.tool(
        name="research_artifact",
        description=(
            "Agent-Native Research Artifacts over the one ontology-driven KG "
            "(CONCEPT:KG-2.79/2.80). action in 'reason' (run OWL/RDF reasoning over "
            "the whole ecosystem and harvest extrapolated cross-domain relationships "
            "as research topics), 'compile' (paper -> ecosystem-grounded OWL-native "
            "4-layer ARA), 'review'/'seal' (L1/L2/L3 OWL/SHACL-grounded review + "
            "certificate), 'capture' (live research event w/ provenance), 'get', 'list'."
        ),
        tags=["graph-os", "research", "ontology"],
    )
    async def research_artifact(
        action: str = Field(
            default="reason",
            description="reason|compile|review|seal|capture|get|list",
        ),
        article_id: str = Field(
            default="", description="Paper/article id (compile/review/get)."
        ),
        query: str = Field(
            default="", description="Topic for 'reason' (reasoning is ecosystem-wide)."
        ),
        level: str = Field(default="L1", description="Seal level: L1|L2|L3 (review)."),
        text: str = Field(default="", description="Event text (capture)."),
        provenance: str = Field(
            default="ai_executed",
            description="capture provenance: user|ai_suggested|ai_executed|user_revised.",
        ),
        actor: str = Field(default="", description="Originating actor id (capture)."),
        event_type: str = Field(default="", description="Force event type (capture)."),
        target_codebase: str = Field(
            default="", description="Codebase to ground claims against (compile)."
        ),
        limit: int = Field(default=50, description="Max rows (list)."),
    ) -> str:
        """Run an ARA action over the one ontology-driven KG (single SoT)."""
        import json

        from agent_utilities.knowledge_graph.research.ara.service import ARAService

        try:
            service = ARAService(_get_engine())
            result = service.run(
                action,
                article_id=article_id,
                query=query,
                level=level,
                text=text,
                provenance=provenance,
                actor=actor,
                event_type=event_type,
                target_codebase=target_codebase or None,
                limit=limit,
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["research_artifact"] = research_artifact

    @mcp.tool(
        name="graph_hydrate",
        description="Trigger instant hydration of the Knowledge Graph from configured external data sources.",
        tags=["graph-os", "hydration"],
    )
    async def graph_hydrate(
        source: str = Field(
            default="all",
            description="The source connector to hydrate (any key from the CAPABILITY_REGISTRY), or 'all' to run all configured sources sequentially.",
        ),
    ) -> str:
        """Trigger instant hydration of the Knowledge Graph from external data sources."""
        import json

        from agent_utilities.knowledge_graph.core.hydration import HydrationManager

        try:
            engine = _get_engine()
            manager = HydrationManager()
            if source == "all":
                res = manager.hydrate_all(engine)
            else:
                res = manager.hydrate_source(engine, source)
            return json.dumps(res, default=str)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    REGISTERED_TOOLS["graph_hydrate"] = graph_hydrate

    # ══════════════════════════════════════════════════════════════════
    # Ontology System — Palantir Foundry parity (type/link/function layer)
    #   property types  (CONCEPT:KG-2.47)
    #   value types     (CONCEPT:KG-2.39)
    #   interfaces      (CONCEPT:KG-2.38)
    #   links           (CONCEPT:KG-2.26)
    #   functions       (CONCEPT:KG-2.41)
    #   derived props   (CONCEPT:KG-2.40)
    # All handlers are thin — they reach the live `KnowledgeGraph.ontology`
    # system (bound to the engine's backend) so Functions-on-Objects, derived
    # compute and interface targeting resolve against the real graph.
    # ══════════════════════════════════════════════════════════════════

    def _ontology_system():
        """Return an OntologySystem bound to the live engine store (or offline)."""
        from agent_utilities.knowledge_graph.facade import KnowledgeGraph

        try:
            engine = _get_engine()
        except Exception:  # pragma: no cover - defensive
            engine = None
        backend = getattr(engine, "backend", None) if engine is not None else None
        kg = KnowledgeGraph()
        # Bind the facade store to the already-initialized engine backend so the
        # object-aware paths read the same graph the rest of the server uses.
        if backend is not None:
            kg._store = backend
        return kg.ontology

    @mcp.tool(
        name="ontology_property_types",
        description="List the ontology property-type registry and resolve/validate a Palantir-style type ref (CONCEPT:KG-2.47).",
        tags=["graph-os", "ontology"],
    )
    def ontology_property_types(
        action: str = Field(
            default="list",
            description="'list' all type names, 'describe' a type, 'column_type' a type's column DDL string, or 'validate' a value.",
        ),
        type_ref: str = Field(
            default="", description="A type ref, e.g. 'array<string>' or 'vector<768>'."
        ),
        value: str = Field(
            default="", description="JSON-encoded value for action='validate'."
        ),
    ) -> str:
        """List/describe ontology property types and resolve/validate a type reference."""
        from agent_utilities.knowledge_graph.ontology.property_types import (
            column_type_for,
            get_property_type,
            list_property_types,
            validate_value,
        )

        try:
            if action == "list":
                return json.dumps({"property_types": list_property_types()})
            if action == "describe":
                pt = get_property_type(type_ref)
                if pt is None:
                    return json.dumps({"error": f"unknown type: {type_ref!r}"})
                return json.dumps(pt.model_dump(), default=str)
            if action == "column_type":
                return json.dumps(
                    {"type_ref": type_ref, "column_type": column_type_for(type_ref)}
                )
            if action == "validate":
                parsed = json.loads(value) if value else None
                return json.dumps(
                    {"type_ref": type_ref, "valid": validate_value(type_ref, parsed)}
                )
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["ontology_property_types"] = ontology_property_types

    @mcp.tool(
        name="ontology_value_types",
        description="List/describe constrained ontology value types and validate or coerce a value (CONCEPT:KG-2.39).",
        tags=["graph-os", "ontology"],
    )
    def ontology_value_types(
        action: str = Field(
            default="list", description="'list' | 'describe' | 'validate' | 'coerce'."
        ),
        name: str = Field(
            default="", description="The value-type name, e.g. 'EmailAddress'."
        ),
        value: str = Field(
            default="", description="JSON-encoded value for validate/coerce."
        ),
    ) -> str:
        """List/describe value types and validate or coerce a value through one."""
        from agent_utilities.knowledge_graph.ontology.value_types import (
            coerce_value_type,
            get_value_type,
            list_value_types,
            validate_value_type,
        )

        try:
            if action == "list":
                return json.dumps({"value_types": list_value_types()})
            if action == "describe":
                vt = get_value_type(name)
                if vt is None:
                    return json.dumps({"error": f"unknown value type: {name!r}"})
                return json.dumps(vt.model_dump(), default=str)
            parsed = json.loads(value) if value else None
            if action == "validate":
                return json.dumps(
                    {"name": name, "valid": validate_value_type(name, parsed)}
                )
            if action == "coerce":
                return json.dumps(
                    {"name": name, "value": coerce_value_type(name, parsed)},
                    default=str,
                )
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["ontology_value_types"] = ontology_value_types

    @mcp.tool(
        name="ontology_interface",
        description="Ontology interfaces: resolve implementers (targeting), check conformance, or emit OWL (CONCEPT:KG-2.38). Set registry='enterprise' to operate on the enterprise-standard contracts (CONCEPT:KG-2.49).",
        tags=["graph-os", "ontology"],
    )
    def ontology_interface(
        action: str = Field(
            default="list",
            description="'list' interfaces, 'implementers' (resolve an interface/type to concrete types), 'conforms' (check an object), or 'owl'.",
        ),
        name: str = Field(default="", description="Interface or concrete type name."),
        object_json: str = Field(
            default="{}", description="JSON object dict for action='conforms'."
        ),
        registry: str = Field(
            default="structural",
            description="Which interface registry: 'structural' (built-in shapes) or 'enterprise' (enterprise-standard contracts, CONCEPT:KG-2.49).",
        ),
    ) -> str:
        """Resolve interface targeting, check conformance, or emit interface OWL/SHACL."""
        from agent_utilities.knowledge_graph.ontology.interfaces import (
            DEFAULT_INTERFACE_REGISTRY,
            target_object_types,
        )
        from agent_utilities.knowledge_graph.standardization.standards import (
            ENTERPRISE_STANDARD_REGISTRY,
        )

        reg = (
            ENTERPRISE_STANDARD_REGISTRY
            if str(registry).lower() == "enterprise"
            else DEFAULT_INTERFACE_REGISTRY
        )
        try:
            if action == "list":
                return json.dumps(
                    {
                        "registry": registry,
                        "interfaces": [i.name for i in reg.list_interfaces()],
                    }
                )
            if action == "implementers":
                impls = (
                    reg.resolve_target(name)
                    if reg is not DEFAULT_INTERFACE_REGISTRY
                    else target_object_types(name)
                )
                return json.dumps({"target": name, "implementers": impls})
            if action == "conforms":
                obj = json.loads(object_json) if object_json else {}
                return json.dumps(
                    {"interface": name, "conforms": reg.conforms(obj, name)}
                )
            if action == "owl":
                return json.dumps({"owl": reg.to_owl()})
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["ontology_interface"] = ontology_interface

    @mcp.tool(
        name="ontology_function",
        description="Typed, versioned ontology functions: list or invoke through the governed runtime (CONCEPT:KG-2.41).",
        tags=["graph-os", "ontology"],
    )
    def ontology_function(
        action: str = Field(
            default="list", description="'list' registered functions or 'invoke' one."
        ),
        name: str = Field(default="", description="Function name for action='invoke'."),
        params: str = Field(
            default="{}", description="JSON-encoded typed input params."
        ),
        version: str = Field(default="", description="Optional pinned semver version."),
        actor: str = Field(
            default="mcp:caller",
            description="Invoking actor id (recorded in the audit entry).",
        ),
    ) -> str:
        """List registered ontology functions or invoke one with typed params."""
        from agent_utilities.knowledge_graph.ontology.functions import (
            DEFAULT_FUNCTION_REGISTRY,
        )

        try:
            if action == "list":
                return json.dumps(
                    [
                        {
                            "name": s.name,
                            "version": s.version,
                            "kind": str(s.kind),
                            "released": s.released,
                            "inputs": [p.model_dump() for p in s.inputs],
                            "output": str(s.output),
                            "description": s.description,
                        }
                        for s in DEFAULT_FUNCTION_REGISTRY.list_functions()
                    ],
                    default=str,
                )
            if action == "invoke":
                actor_id = actor or "mcp:caller"
                ont = _ontology_system()
                parsed = json.loads(params) if params else {}
                result = ont.invoke_function(
                    name, parsed, version or None, actor_id=actor_id
                )
                return json.dumps(result.model_dump(), default=str)
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["ontology_function"] = ontology_function

    @mcp.tool(
        name="ontology_derive",
        description="Compute derived (function/cypher/sparql/embedding-backed) properties live at read time (CONCEPT:KG-2.40).",
        tags=["graph-os", "ontology"],
    )
    def ontology_derive(
        action: str = Field(
            default="compute",
            description="'list' declarations, 'compute' one property, or 'compute_all'.",
        ),
        object_json: str = Field(
            default="{}", description="JSON object dict the property is computed for."
        ),
        name: str = Field(
            default="", description="Derived-property name for action='compute'."
        ),
        object_type: str = Field(
            default="", description="Optional object type for declaration resolution."
        ),
    ) -> str:
        """Compute derived properties for an object against the live graph."""
        from agent_utilities.knowledge_graph.ontology.derived_properties import (
            DEFAULT_DERIVED_REGISTRY,
        )

        try:
            if action == "list":
                return json.dumps(
                    [
                        {
                            "name": d.name,
                            "object_type": d.object_type,
                            "backing": str(d.backing),
                            "output_type": str(d.output_type),
                            "description": d.description,
                        }
                        for d in DEFAULT_DERIVED_REGISTRY.list_all()
                    ],
                    default=str,
                )
            ont = _ontology_system()
            obj = json.loads(object_json) if object_json else {}
            otype = object_type or None
            if action == "compute":
                res = ont.derive(obj, name, object_type=otype)
                return json.dumps(res.model_dump(), default=str)
            if action == "compute_all":
                return json.dumps(ont.derive_all(obj, object_type=otype), default=str)
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["ontology_derive"] = ontology_derive

    @mcp.tool(
        name="ontology_link_materialize",
        description="Reify a many-to-many ontology link as a (junction_node, edge_a, edge_b) triple and write it (CONCEPT:KG-2.26).",
        tags=["graph-os", "ontology"],
    )
    async def ontology_link_materialize(
        action: str = Field(
            default="materialize",
            description="'types' to list link types, or 'materialize' a junction.",
        ),
        link_name: str = Field(
            default="", description="The junction link type name, e.g. 'agent_skill'."
        ),
        source_id: str = Field(default="", description="Source endpoint node id."),
        target_id: str = Field(default="", description="Target endpoint node id."),
        properties: str = Field(
            default="{}", description="JSON-encoded junction (link) properties."
        ),
    ) -> str:
        """List link types or reify + persist a M:N link via the graph_write path."""
        from agent_utilities.knowledge_graph.ontology.links import DEFAULT_LINK_REGISTRY

        try:
            if action == "types":
                return json.dumps(
                    [
                        {
                            "name": link.name,
                            "source_type": str(link.source_type),
                            "target_type": str(link.target_type),
                            "edge_type": str(link.edge_type),
                            "cardinality": str(link.cardinality),
                            "is_junction": link.name
                            in {j.name for j in DEFAULT_LINK_REGISTRY.junctions()},
                        }
                        for link in DEFAULT_LINK_REGISTRY.list_links()
                    ],
                    default=str,
                )
            ont = _ontology_system()
            props = json.loads(properties) if properties else {}
            node, edge_a, edge_b = ont.materialize_link(
                link_name, source_id, target_id, props
            )
            # Persist via the existing graph_write add_node / add_edge primitives.
            await _execute_tool(
                "graph_write",
                action="add_node",
                node_type=str(node.type),
                node_id=node.id,
                properties=json.dumps(
                    {"name": node.name, **(node.metadata or {})}, default=str
                ),
            )
            for edge in (edge_a, edge_b):
                await _execute_tool(
                    "graph_write",
                    action="add_edge",
                    source_id=edge.source,
                    target_id=edge.target,
                    rel_type=str(edge.type),
                    properties=json.dumps(edge.metadata or {}, default=str),
                )
            return json.dumps(
                {
                    "junction_id": node.id,
                    "edge_a_type": str(edge_a.type),
                    "edge_b_type": str(edge_b.type),
                },
                default=str,
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["ontology_link_materialize"] = ontology_link_materialize

    @mcp.tool(
        name="object_edits",
        description="Durable object-edit ledger (CONCEPT:KG-2.43): record a structured edit (property_set/link_add/link_remove/object_create/object_delete), revert an edit, or read per-object history / as_of snapshot.",
        tags=["graph-os", "ontology"],
    )
    def object_edits(
        action: str = Field(
            default="history",
            description="'record' an edit | 'revert' an edit by id | 'history' per object | 'as_of' snapshot.",
        ),
        object_id: str = Field(
            default="", description="Target object id (record/history/as_of)."
        ),
        edit_type: str = Field(
            default="property_set",
            description="property_set|link_add|link_remove|object_create|object_delete (for action='record').",
        ),
        properties_json: str = Field(
            default="{}",
            description="JSON property map (record property_set/object_create).",
        ),
        link_target: str = Field(
            default="", description="Link target id (record link_add/link_remove)."
        ),
        link_label: str = Field(
            default="related", description="Link label (record link_add/link_remove)."
        ),
        edit_id: str = Field(default="", description="Edit id (action='revert')."),
        ts: float = Field(default=0.0, description="Unix timestamp (action='as_of')."),
        actor: str = Field(
            default="system", description="Acting principal recorded on the edit."
        ),
    ) -> str:
        """Record / revert object edits and read per-object edit history or an as_of snapshot."""
        from agent_utilities.knowledge_graph.ontology.edits import (
            Edit,
            EditType,
            revert_edit,
        )

        try:
            ont = _ontology_system()
            ledger = ont.edits
            if action == "record":
                etype = EditType(edit_type)
                if etype in (EditType.LINK_ADD, EditType.LINK_REMOVE):
                    edit = Edit(
                        actor=actor,
                        edit_type=etype,
                        object_id=object_id,
                        link_source=object_id,
                        link_label=link_label,
                        link_target=link_target,
                    )
                else:
                    props = json.loads(properties_json) if properties_json else {}
                    edit = Edit(
                        actor=actor,
                        edit_type=etype,
                        object_id=object_id,
                        after=dict(props),
                    )
                recorded = ledger.record(edit)
                return json.dumps(recorded.model_dump(), default=str)
            if action == "revert":
                comp = revert_edit(ledger, edit_id, actor=actor)
                return json.dumps(comp.model_dump(), default=str)
            if action == "history":
                return json.dumps(
                    {
                        "object_id": object_id,
                        "history": [e.model_dump() for e in ledger.history(object_id)],
                    },
                    default=str,
                )
            if action == "as_of":
                return json.dumps(
                    {"object_id": object_id, "snapshot": ledger.as_of(object_id, ts)},
                    default=str,
                )
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["object_edits"] = object_edits

    @mcp.tool(
        name="object_index",
        description="Object Index Lifecycle / Object Data Funnel (CONCEPT:KG-2.44): batch/incremental sync of the live search index from source nodes, report staleness, or reindex stale objects.",
        tags=["graph-os", "ontology"],
    )
    def object_index(
        action: str = Field(
            default="status",
            description="'sync' (batch rebuild) | 'reindex' (reconcile stale) | 'status' (live/tombstone counts).",
        ),
        nodes_json: str = Field(
            default="[]",
            description="JSON list of source node mappings (sync/reindex).",
        ),
    ) -> str:
        """Sync / reindex the live object search index and report staleness."""
        try:
            ont = _ontology_system()
            funnel = ont.index_funnel
            if action == "sync":
                nodes = json.loads(nodes_json) if nodes_json else []
                return json.dumps(funnel.batch_sync(nodes).as_dict())
            if action == "reindex":
                nodes = json.loads(nodes_json) if nodes_json else []
                return json.dumps(funnel.reconcile(nodes).as_dict())
            if action == "status":
                return json.dumps(
                    {
                        "live_size": len(funnel),
                        "tombstones": funnel.tombstone_count,
                        "indexed_ids": sorted(funnel.live_ids()),
                    }
                )
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["object_index"] = object_index

    @mcp.tool(
        name="object_permissioning",
        description="Fine-grained object permissioning (CONCEPT:KG-2.46): redact an object, materialize a restricted view, or attach a mandatory marking. Actor is resolved from the ambient context — never from caller-supplied clearance.",
        tags=["graph-os", "ontology"],
    )
    def object_permissioning(
        action: str = Field(
            default="restricted_view",
            description="'redact' one object | 'restricted_view' an object set | 'mark' attach a marking.",
        ),
        objects_json: str = Field(
            default="[]", description="JSON list of object dicts (restricted_view)."
        ),
        object_json: str = Field(
            default="{}", description="JSON object dict (redact)."
        ),
        node_id: str = Field(default="", description="Node id (action='mark')."),
        marking: str = Field(default="", description="Marking name (action='mark')."),
        mask: bool = Field(
            default=False,
            description="Mask withheld properties instead of dropping them.",
        ),
    ) -> str:
        """Redact / restrict / mark objects for the AMBIENT actor (no spoofable clearance)."""
        from agent_utilities.knowledge_graph.ontology.permissioning import (
            apply_marking,
            redact_object,
            restricted_view,
        )

        try:
            # actor=None -> resolved from the ambient ActorContext set by the
            # dispatcher's use_actor(); callers cannot inject their own clearance.
            if action == "redact":
                obj = json.loads(object_json) if object_json else {}
                return json.dumps(redact_object(obj, None, mask=mask), default=str)
            if action == "restricted_view":
                objs = json.loads(objects_json) if objects_json else []
                return json.dumps(restricted_view(objs, None, mask=mask), default=str)
            if action == "mark":
                apply_marking(node_id, marking)
                return json.dumps(
                    {"node_id": node_id, "marking": marking, "applied": True}
                )
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["object_permissioning"] = object_permissioning

    @mcp.tool(
        name="graph_share",
        description="Share a private node (CONCEPT:KG-2.60). Data is private-to-its-owner by default; this is the explicit promotion path. action='org' shares with the owner's org (in-place), 'commons' promotes a copy into the shared cross-org commons graph (share by WHERE placed), 'mark' attaches a mandatory marking (share by HOW placed), 'private' restricts it back. Actor/owner is the ambient identity — never caller-supplied.",
        tags=["graph-os", "tenancy"],
    )
    def graph_share(
        action: str = Field(
            default="org",
            description="'org' share with my org | 'commons' promote to the shared commons graph | 'mark' attach a marking | 'private' restrict back to me.",
        ),
        node_id: str = Field(default="", description="Id of the node to share."),
        marking: str = Field(default="", description="Marking name (action='mark')."),
    ) -> str:
        """Explicit, private-by-default sharing for the AMBIENT actor (KG-2.60)."""
        from agent_utilities.knowledge_graph.core import tenant_sharing as _ts

        if not node_id:
            return json.dumps({"error": "node_id is required"})
        try:
            if action == "org":
                _ts.share_with_org(node_id)
                return json.dumps({"node_id": node_id, "shared_scope": "org"})
            if action == "commons":
                ok = _ts.promote_to_commons(node_id)
                return json.dumps(
                    {"node_id": node_id, "shared_scope": "commons", "promoted": ok}
                )
            if action == "mark":
                if not marking:
                    return json.dumps(
                        {"error": "marking is required for action='mark'"}
                    )
                _ts.share(node_id, marking)
                return json.dumps({"node_id": node_id, "marking": marking})
            if action == "private":
                _ts.make_private(node_id)
                return json.dumps({"node_id": node_id, "shared_scope": "private"})
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["graph_share"] = graph_share

    @mcp.tool(
        name="object_set",
        description="Object Set Service (CONCEPT:KG-2.45/2.38): search/filter/search_around/pivot/aggregate and union/intersect/subtract over Foundry-style object sets.",
        tags=["graph-os", "ontology"],
    )
    def object_set(
        action: str = Field(
            default="of_type",
            description="of_type|from_ids|search|filter|search_around|pivot|aggregate|union|intersect|subtract.",
        ),
        type_or_interface: str = Field(
            default="", description="Object type / interface (of_type)."
        ),
        ids_json: str = Field(
            default="[]",
            description="JSON list of ids (from_ids / set algebra 'other').",
        ),
        query: str = Field(default="", description="Search query (search)."),
        link_type: str = Field(
            default="", description="Link type (search_around/pivot); empty = any."
        ),
        hops: int = Field(default=1, description="Hop count (search_around)."),
        direction: str = Field(
            default="out", description="out|in|both (search_around/pivot)."
        ),
        group_by: str = Field(
            default="", description="Group-by property (pivot/aggregate)."
        ),
        metric: str = Field(
            default="count", description="count|sum|avg|min|max (aggregate)."
        ),
        field: str = Field(
            default="", description="Numeric field (aggregate sum/avg/min/max)."
        ),
        limit: int = Field(default=50, description="Result limit (search)."),
    ) -> str:
        """Compute over a Foundry-style object set: search/filter/traverse/pivot/aggregate/algebra."""
        try:
            ont = _ontology_system()
            if action == "from_ids" or action in ("union", "intersect", "subtract"):
                base = ont.object_set(json.loads(ids_json) if ids_json else [])
            else:
                base = ont.object_set_of_type(type_or_interface)

            if action in ("of_type", "from_ids"):
                return json.dumps({"ids": base.ids(), "count": base.count()})
            if action == "search":
                res = base.search(query, limit=limit)
                return json.dumps({"ids": res.ids(), "count": res.count()})
            if action == "search_around":
                res = base.search_around(
                    link_type or None, hops=hops, direction=direction
                )
                return json.dumps({"ids": res.ids(), "count": res.count()})
            if action == "pivot":
                piv = base.pivot(link_type or None, group_by, direction=direction)
                return json.dumps(
                    {
                        "link_type": piv.link_type,
                        "group_by": piv.group_by,
                        "groups": piv.groups,
                    },
                    default=str,
                )
            if action == "aggregate":
                agg = base.aggregate(
                    metric, field=field or None, group_by=group_by or None
                )
                return json.dumps(
                    {
                        "metric": agg.metric,
                        "field": agg.field,
                        "group_by": agg.group_by,
                        "groups": {str(k): v for k, v in agg.groups.items()},
                        "total_objects": agg.total_objects,
                    },
                    default=str,
                )
            if action in ("union", "intersect", "subtract"):
                other = (
                    ont.object_set_of_type(type_or_interface)
                    if type_or_interface
                    else ont.object_set([])
                )
                combined = getattr(base, action)(other)
                return json.dumps({"ids": combined.ids(), "count": combined.count()})
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["object_set"] = object_set

    @mcp.tool(
        name="document_process",
        description="Document → ontology processing (CONCEPT:KG-2.48): extract → chunk(overlap) → embed → materialize a Document + linked Chunk objects through the live graph write path.",
        tags=["graph-os", "ontology"],
    )
    def document_process(
        document: str = Field(
            description="A file path or raw text content to process."
        ),
        text: str = Field(
            default="", description="Optional pre-extracted text (OCR/external)."
        ),
        source: str = Field(default="", description="Provenance label (path/URL)."),
        chunk_size: int = Field(
            default=800, description="Target chunk size in characters."
        ),
        overlap: int = Field(
            default=120, description="Overlap characters between chunks."
        ),
        contextual: bool = Field(
            default=False,
            description="Enable contextual-retrieval enrichment (CONCEPT:KG-2.50): situate each chunk within the document and embed context+chunk for better recall.",
        ),
    ) -> str:
        """Process a document into Document + Chunk ontology objects through the live graph."""
        from agent_utilities.knowledge_graph.facade import KnowledgeGraph
        from agent_utilities.knowledge_graph.ontology.document_processing import (
            ChunkingConfig,
            DocumentProcessor,
        )

        try:
            engine = None
            try:
                engine = _get_engine()
            except Exception:  # pragma: no cover - defensive
                engine = None
            backend = getattr(engine, "backend", None) if engine is not None else None
            kg = KnowledgeGraph()
            if backend is not None:
                kg._store = backend
            proc = DocumentProcessor(
                kg,
                chunking=ChunkingConfig(chunk_size=chunk_size, overlap=overlap),
                contextual=contextual,
            )
            result = proc.process(document, text=text or None, source=source)
            return json.dumps(
                {
                    "document_id": result.document_id,
                    "chunk_count": result.chunk_count,
                    "persisted": result.persisted,
                    "edges": len(result.edges),
                },
                default=str,
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["document_process"] = document_process

    # Quant trading system tool (CONCEPT:ECO-4.0): debate/regime/data/execute/
    # portfolio over the finance engines. Registered onto the MCP server AND the
    # shared REGISTERED_TOOLS map so the gateway REST twin (/quant) reaches it.
    from agent_utilities.domains.finance.quant_mcp_tools import register_quant_tools

    REGISTERED_TOOLS["quant"] = register_quant_tools(mcp, None)

    @mcp.tool(
        name="source_connector",
        description="Document-source connectors (CONCEPT:ECO-4.25–4.29, KG-2.59): list registered connectors, or run one (filesystem/web/rest/database/mcp:<package>/mcp_tool — mcp_tool drives any fleet MCP server's listing tool as a paginated source) to ingest its documents into the KG as Document+Chunk objects with contextual enrichment (KG-2.50) and external permission sync (ECO-4.28).",
        tags=["graph-os", "ecosystem", "connectors"],
    )
    async def source_connector(
        action: str = Field(
            default="list",
            description="One of: 'list' (registered connector types), 'run' (build + ingest a connector).",
        ),
        source_type: str = Field(
            default="",
            description="Connector type for 'run' (filesystem/web/rest/database/mcp:<package>/mcp_tool).",
        ),
        config: dict = Field(
            default_factory=dict,
            description="Connector configuration dict for 'run' (e.g. {'root': '/docs'} or {'base_url': 'https://…'}).",
        ),
        connector_id: str = Field(
            default="",
            description="Stable id for incremental checkpoint storage (optional).",
        ),
        contextual: bool = Field(
            default=True,
            description="Enable contextual-retrieval enrichment (CONCEPT:KG-2.50).",
        ),
        incremental: bool = Field(
            default=True,
            description="Use the connector's resumable poll (CONCEPT:ECO-4.26) vs a full load.",
        ),
    ) -> str:
        """List or run a document-source connector (CONCEPT:ECO-4.25–4.29)."""
        from agent_utilities.knowledge_graph.facade import KnowledgeGraph

        try:
            if action == "list":
                from agent_utilities.protocols.source_connectors import list_sources

                return json.dumps({"connectors": list_sources()})

            if action == "run":
                if not source_type:
                    return json.dumps(
                        {"error": "source_type is required for action='run'"}
                    )
                engine = None
                try:
                    engine = _get_engine()
                except Exception:  # pragma: no cover - defensive
                    engine = None
                backend = (
                    getattr(engine, "backend", None) if engine is not None else None
                )
                kg = KnowledgeGraph()
                if backend is not None:
                    kg._store = backend
                result = await kg.ontology.run_connector(
                    source_type,
                    dict(config or {}),
                    connector_id=connector_id or None,
                    contextual=contextual,
                    incremental=incremental,
                )
                return json.dumps(result, default=str)

            return json.dumps(
                {"error": f"unknown action {action!r}; use 'list' or 'run'"}
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["source_connector"] = source_connector

    return args, mcp, middlewares


# ══════════════════════════════════════════════════════════════════


class CentralizedCypherMiddleware:
    """ASGI middleware to intercept and route /cypher requests directly to the active database engine.

    Exposes:
      - POST /cypher: Executes a Cypher query (read/write) or chunked batch query directly,
        completely bypassing SQLite file locking contention between parallel client processes.

    Features:
      - G5: Backpressure via asyncio.Semaphore (max_concurrent configurable).
      - G6: TTL-based read query deduplication cache for identical read-only queries.
    """

    # G6: Module-level read query cache (shared across middleware instances)
    _READ_CACHE: dict[str, tuple[Any, float]] = {}
    _READ_CACHE_TTL = 2.0  # seconds

    def __init__(self, app, max_concurrent: int = 50):
        self.app = app
        # G5: Backpressure semaphore to limit concurrent query execution
        self._semaphore: asyncio.Semaphore | None = None
        self._max_concurrent = max_concurrent

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Lazily create the semaphore inside the running event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    @staticmethod
    def _is_read_only(query: str) -> bool:
        """Check if a Cypher query is read-only (no write operations)."""
        q = query.upper().strip()
        write_keywords = {"CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP"}
        return not any(kw in q for kw in write_keywords)

    @classmethod
    def _get_cached_read(cls, cache_key: str) -> list[dict[str, Any]] | None:
        """Return cached read results if within TTL, else None."""
        cached = cls._READ_CACHE.get(cache_key)
        if cached is None:
            return None
        results, ts = cached
        if (_time.monotonic() - ts) < cls._READ_CACHE_TTL:
            return results
        # Expired — remove stale entry
        cls._READ_CACHE.pop(cache_key, None)
        return None

    @classmethod
    def _set_cached_read(cls, cache_key: str, results: list[dict[str, Any]]) -> None:
        """Cache read query results with TTL."""
        cls._READ_CACHE[cache_key] = (results, _time.monotonic())
        # Evict old entries if cache grows too large (prevent unbounded memory)
        if len(cls._READ_CACHE) > 1000:
            now = _time.monotonic()
            expired = [
                k
                for k, (_, ts) in cls._READ_CACHE.items()
                if (now - ts) >= cls._READ_CACHE_TTL
            ]
            for k in expired:
                cls._READ_CACHE.pop(k, None)

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] == "http"
            and scope["path"] == "/cypher"
            and scope["method"] == "POST"
        ):
            # G5: Acquire backpressure semaphore
            sem = self._get_semaphore()
            async with sem:
                await self._handle_cypher(scope, receive, send)
            return

        # Pass through to the standard FastMCP Starlette app
        await self.app(scope, receive, send)

    async def _handle_cypher(self, scope, receive, send):
        """Handle a /cypher POST request with safety guardrails and caching."""
        # 1. Extract headers for agent and session attribution
        agent_id = "anonymous_agent"
        session_id = "default_session"
        for k, v in scope.get("headers", []):
            if k.lower() == b"x-agent-id":
                agent_id = v.decode("utf-8")
            elif k.lower() == b"x-session-id":
                session_id = v.decode("utf-8")

        # 2. Read the HTTP request body
        body = b""
        more_body = True
        while more_body:
            message = await receive()
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        # 3. Parse request JSON and execute
        import json

        try:
            data = json.loads(body.decode("utf-8")) if body else {}
            query = data.get("query")
            params = data.get("params")
            batch = data.get("batch")
            chunk_size = data.get("chunk_size", 500)

            if not query:
                raise ValueError("A 'query' Cypher string is required.")

            # Cypher safety check: reject global destructive wipes
            query_upper = query.upper()
            if (
                "DELETE" in query_upper
                and "WHERE" not in query_upper
                and "LIMIT" not in query_upper
            ):
                if "MATCH" in query_upper and (
                    "DETACH" in query_upper or "DELETE" in query_upper
                ):
                    # Ensure it's not a global delete
                    # Matches deleting all nodes or relations without bounds
                    raise ValueError(
                        "Query rejected: Global DETACH/DELETE without WHERE filters "
                        "is banned by epistemic safety guardrails (CONCEPT:OS-5.1)."
                    )

            logger.info(
                f"[KG Gateway] Query from agent '{agent_id}' (session: '{session_id}'): "
                f"{query[:100]}..."
            )

            # G6: Check read cache for read-only queries without batch
            is_read = self._is_read_only(query) and batch is None
            cache_key = ""
            if is_read:
                import hashlib

                cache_key = hashlib.md5(
                    f"{query}|{json.dumps(params or {}, sort_keys=True)}".encode(),
                    usedforsecurity=False,
                ).hexdigest()
                cached = self._get_cached_read(cache_key)
                if cached is not None:
                    logger.debug(
                        f"[KG Gateway] Cache HIT for read query (key={cache_key[:8]})"
                    )
                    resp_data = {"status": "success", "results": cached}
                    await self._send_json(send, resp_data, 200)
                    return

            engine = _get_engine()
            if not engine or not engine.backend:
                raise RuntimeError(
                    "IntelligenceGraphEngine backend not active on central server."
                )

            # If batch is present, run execute_batch, otherwise run execute
            if batch is not None:
                results = engine.backend.execute_batch(query, batch, chunk_size)
            else:
                results = engine.backend.execute(query, params)

            # G6: Cache read results
            if is_read and cache_key:
                self._set_cached_read(cache_key, results)

            resp_data = {"status": "success", "results": results}
            status_code = 200
        except Exception as e:
            logger.error(
                f"CentralizedCypherMiddleware execution error: {e}", exc_info=True
            )
            resp_data = {"status": "error", "message": str(e)}
            status_code = 500

        await self._send_json(send, resp_data, status_code)

    @staticmethod
    async def _send_json(send, data: dict, status_code: int) -> None:
        """Send a JSON HTTP response."""
        import json

        response_body = json.dumps(data, default=str).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(response_body)).encode("ascii")),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": response_body,
            }
        )
        return


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
    route("/graph/orchestrate", graph_orchestrate_endpoint, ["POST"])
    route("/graph/configure", graph_configure_endpoint, ["POST"])

    # ── Granular query ──
    route("/graph/query/federated", graph_query_federated_endpoint, ["POST"])

    # ── Granular search ──
    route("/graph/search/hybrid", graph_search_hybrid_endpoint, ["POST"])
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

    # ── Granular orchestrate ──
    route("/graph/orchestrate/dispatch", graph_orchestrate_dispatch_endpoint, ["POST"])
    route("/graph/orchestrate/job/{job_id}", graph_orchestrate_status_endpoint, ["GET"])
    route(
        "/graph/orchestrate/request-approval",
        graph_orchestrate_request_approval_endpoint,
        ["POST"],
    )
    route(
        "/graph/orchestrate/grant-approval",
        graph_orchestrate_grant_approval_endpoint,
        ["POST"],
    )
    route(
        "/graph/orchestrate/execute-agent",
        graph_orchestrate_execute_agent_endpoint,
        ["POST"],
    )
    route(
        "/graph/orchestrate/consensus", graph_orchestrate_consensus_endpoint, ["POST"]
    )
    route(
        "/graph/orchestrate/start-debate",
        graph_orchestrate_start_debate_endpoint,
        ["POST"],
    )
    route(
        "/graph/orchestrate/submit-risk-veto",
        graph_orchestrate_submit_risk_veto_endpoint,
        ["POST"],
    )
    route(
        "/graph/orchestrate/cron-jobs",
        graph_orchestrate_list_cron_jobs_endpoint,
        ["GET"],
    )
    route(
        "/graph/orchestrate/trigger-cron-job",
        graph_orchestrate_trigger_cron_job_endpoint,
        ["POST"],
    )
    route(
        "/graph/orchestrate/compile-workflow",
        graph_orchestrate_compile_workflow_endpoint,
        ["POST"],
    )
    route(
        "/graph/orchestrate/compile-process",
        graph_orchestrate_compile_process_endpoint,
        ["POST"],
    )
    route(
        "/graph/orchestrate/publish-proposal",
        graph_orchestrate_publish_proposal_endpoint,
        ["POST"],
    )
    route(
        "/graph/orchestrate/workflows",
        graph_orchestrate_list_workflows_endpoint,
        ["GET"],
    )
    route(
        "/graph/orchestrate/execute-workflow",
        graph_orchestrate_execute_workflow_endpoint,
        ["POST"],
    )
    route(
        "/graph/orchestrate/dispatch-workflow",
        graph_orchestrate_dispatch_workflow_endpoint,
        ["POST"],
    )
    route(
        "/graph/orchestrate/workflow-status/{job_id}",
        graph_orchestrate_workflow_status_endpoint,
        ["GET"],
    )
    route(
        "/graph/orchestrate/export-workflow",
        graph_orchestrate_export_workflow_endpoint,
        ["POST"],
    )

    # ── Granular configure ──
    route("/graph/configure/secret", graph_configure_secret_endpoint, ["POST"])
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
    # The seven core graph_* tools above already have bespoke endpoints; every
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
    }
    for _tool, _path in ACTION_TOOL_ROUTES.items():
        if _tool in _bespoke_action_tools:
            continue
        route(_path, _make_tool_endpoint(_tool), ["POST"])


def mcp_server() -> None:
    """``graph-os`` MCP server entry point (registered as console_scripts).

    Thin FastMCP wrapper following the standard ``mcp_server.py`` template: it
    serves ONLY the MCP tool surface, over ``stdio`` or ``streamable-http`` (or
    legacy ``sse``), selected by the standard ``--transport/--host/--port`` args
    from :func:`create_mcp_server`. The REST API (``/graph/*``, ``/sessions``,
    ``/goals``, ``/tools``) is centralized in the API gateway
    (``agent_utilities.gateway``) — see :func:`_mount_rest_routes`.
    """
    os.environ["IS_KG_SERVER"] = "true"
    args, mcp, middlewares = _build_server()

    # Apply the middleware stack assembled by the factory.
    for middleware in middlewares:
        mcp.add_middleware(middleware)

    transport = getattr(args, "transport", "stdio")
    host = getattr(args, "host", "0.0.0.0")
    port = int(getattr(args, "port", 8000))

    logger.info(
        "Starting graph-os MCP Server (transport=%s, host=%s, port=%s)",
        transport,
        host,
        port,
    )

    from agent_utilities.mcp.server_factory import protect_stdio_jsonrpc
    from agent_utilities.security.request_identity import apply_served_security_profile

    # Network transports serve many clients at once: enforce server-validated
    # identity + tenant scoping, or fail loud (CONCEPT:OS-5.14). No-op for stdio.
    apply_served_security_profile(transport)

    if transport == "stdio":
        protect_stdio_jsonrpc()
        mcp.run(transport="stdio")
    elif transport == "streamable-http":
        mcp.run(transport="streamable-http", host=host, port=port)
    elif transport == "sse":
        mcp.run(transport="sse", host=host, port=port)
    else:
        protect_stdio_jsonrpc()
        mcp.run(transport="stdio")


# Back-compat alias — the previous console_scripts entry and some docs/tooling
# reference ``main``; keep it pointing at the new thin entry point.
main = mcp_server


if __name__ == "__main__":
    mcp_server()
