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


# Server-side identity for stdio MCP (CONCEPT:AU-OS.identity.authenticated-identity-enforcement): minted once at startup
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
    """Resolve the actor a tool call runs as (CONCEPT:AU-KG.research.research-pipeline-runner / OS-5.14).

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

    # CONCEPT:AU-OS.identity.authenticated-identity-enforcement — with KG_AUTH_REQUIRED on, an unauthenticated caller
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
    "graph_hydrate": "/graph/hydrate",
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
    "quant": "/quant",
    "research_artifact": "/research/artifact",
    "graph_loops": "/graph/loops",
    "graph_schedules": "/graph/schedules",
    "graph_feeds": "/graph/feeds",
    "graph_sandbox": "/graph/sandbox",
    "graph_runvcs": "/graph/runvcs",
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
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

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
    """List registered document-source connectors (CONCEPT:AU-ECO.connector.factory-ingestion-adaptor)."""
    try:
        res = await _execute_tool("source_connector", action="list")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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


async def graph_analyze_call_graph_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=call_graph (CONCEPT:EG-KG.compute.type-scope-resolved-call): the
    type/scope-resolved call/inheritance graph for a symbol. ``id`` = symbol id;
    ``direction`` = callees | callers | inherits."""
    try:
        node_id = request.query_params.get("id") or request.query_params.get(
            "node_id", ""
        )
        direction = request.query_params.get("direction") or request.query_params.get(
            "target", "callees"
        )
        res = await _execute_tool(
            "graph_analyze", action="call_graph", node_id=node_id, target=direction
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_similar_code_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=similar_code (CONCEPT:EG-KG.compute.model-free-similar-code): a
    symbol's model-free MinHash/LSH near-clone neighbours (embedder-free).
    ``id`` = symbol id; ``top_k`` optional."""
    try:
        node_id = request.query_params.get("id") or request.query_params.get(
            "node_id", ""
        )
        top_k = int(request.query_params.get("top_k", "10"))
        res = await _execute_tool(
            "graph_analyze", action="similar_code", node_id=node_id, top_k=top_k
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_routes_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=routes (CONCEPT:AU-KG.compute.http-route-graph): the HTTP route
    graph — each Route, its handler, and the Service that serves it."""
    try:
        res = await _execute_tool("graph_analyze", action="routes")
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_change_coupling_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=change_coupling (CONCEPT:AU-KG.ingest.mine-git-history-files): mine a
    repo's git history into FILE_CHANGES_WITH edges. Body: ``{repo, min_support?}``."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_analyze",
            action="change_coupling",
            target=body.get("repo", ""),
            depth=int(body.get("min_support", 3)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
            "graph_analyze",
            action="code_evolution",
            target=body.get("mode", "file"),
            query=body.get("target", ""),
            top_k=int(body.get("top_k", 20)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_adr_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=adr (CONCEPT:AU-KG.compute.adr-crud): ADR CRUD. Body:
    ``{title?, status?, decision?}`` — title creates, empty lists."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_analyze",
            action="adr",
            query=body.get("title", ""),
            target=body.get("status", ""),
            node_id=body.get("decision", ""),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
            "graph_analyze", action="harness_gate", query=_json.dumps(body)
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
            "graph_analyze",
            action="code_context",
            query=body.get("query", ""),
            target=intent,
            node_id=body.get("node_id", ""),
            top_k=int(body.get("top_k", 10)),
            depth=int(body.get("depth", 2)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_analyze_explain_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action=explain (CONCEPT:AU-KG.retrieval.route-question-its-domain): the universal
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
            "graph_analyze",
            action="explain",
            query=body.get("query", ""),
            target=target,
            node_id=body.get("node_id", ""),
            top_k=int(body.get("top_k", 10)),
            depth=int(body.get("depth", 2)),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
            "graph_analyze", action="cross_repo_usages", query=symbol, top_k=top_k
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
            "graph_analyze", action="code_metrics", target=scope, top_k=top_k
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
            "graph_analyze", action="arch_report", target=scope, top_k=top_k
        )
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
    """CONCEPT:AU-ORCH.planning.compile-process-rest-twin — REST twin of graph_orchestrate compile_process."""
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
    """CONCEPT:AU-AHE.harness.publish-proposal-rest — REST twin of graph_orchestrate publish_proposal."""
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


async def graph_orchestrate_distill_skills_endpoint(
    request: Request,
) -> JSONResponse:
    """CONCEPT:AU-KG.ontology.connector-agnostic-proposal/2.83 — REST twin of graph_orchestrate distill_skills.

    Connector → skill synthesis: turn the mapped processes of ALL connected
    systems (egeria/leanix/aris/camunda) into propose-only atomic-skill +
    skill-workflow PROPOSALS, connector-agnostic over the ontology. Pass
    ``{"draft": true}`` to also render reviewable SKILL.md staging artifacts.
    Dispatches into the SAME action core as the MCP tool.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_orchestrate",
            action="distill_skills",
            task="draft" if body.get("draft") else "",
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


async def graph_orchestrate_synthesize_org_endpoint(request: Request) -> JSONResponse:
    """CONCEPT:AU-ORCH.org.recruiter — REST twin of graph_orchestrate synthesize_org."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_orchestrate",
            action="synthesize_org",
            task=body.get("task", "") or body.get("goal", ""),
            dependencies=body.get("dependencies", "[]"),
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def graph_orchestrate_run_org_endpoint(request: Request) -> JSONResponse:
    """CONCEPT:AU-ORCH.org.work-item-dag — REST twin of graph_orchestrate run_org."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        res = await _execute_tool(
            "graph_orchestrate",
            action="run_org",
            task=body.get("task", "") or body.get("goal", ""),
            dependencies=body.get("dependencies", "[]"),
            max_steps=int(body.get("max_steps", 20)),
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
    must therefore not race a second engine into existence. (CONCEPT:EG-KG.storage.nonblocking-checkpoint)
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


# ── CONCEPT:AU-KG.backend.multi-connection-registry — Named multi-connection graph registry ────────────────
_CONNECTION_REGISTRY = None
_REGISTRY_LOCK = threading.Lock()


def get_connection_registry():
    """Process-wide :class:`ConnectionRegistry` singleton.

    Seeds its ``"default"`` connection from the legacy ``_get_engine`` singleton
    (so the default is never duplicated) and from ``config.kg_connections``
    (CONCEPT:AU-KG.backend.multi-connection-registry) on first build.
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
    if not (is_implicit_default and ingest_routing.routing_enabled()):
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
        except Exception as e:  # noqa: BLE001 — partial-success contract
            errors[name] = str(e)
    for fut in not_done:
        errors[futures[fut]] = (
            f"timed out after {timeout:.0f}s (target slow/unreachable)"
        )
    # Never block on a hung backend's thread; let it finish in the background.
    ex.shutdown(wait=False, cancel_futures=True)
    return results, errors


def _provenance_props(agent_id: str | None = None) -> dict[str, Any]:
    """Build standard provenance metadata for multi-agent write tracking."""
    return {
        "agent_id": agent_id or _AGENT_ID,
        "session_id": _SESSION_ID,
        "workspace_path": _WORKSPACE_PATH,
        "timestamp": datetime.now(UTC).isoformat(),
        "source": "mcp",
    }


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

    from agent_utilities.knowledge_graph.core.source_sync import (
        derive_capability_synonyms,
        sync_source,
    )

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
                            "synonyms": derive_capability_synonyms(server_name),
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

    # 4. Fleet MCP-server tools → Tool capability nodes (CONCEPT:AU-KG.ontology.capability-node-aliases-lexical).
    # Steps 1-3 ingest graph-os's OWN servers/native-tools/skills; the ~62 fleet
    # MCP servers' tools were never elevated, so the KG lacked the fleet
    # capability vocabulary the classification gate and dispatcher specialist
    # routing query. This probes the served multiplexer catalog and writes each
    # fleet tool as a Tool node. Native/default-on; unreachable servers are
    # skipped. Goes through the same `source_sync` path as the MCP/REST surface.
    try:
        result = sync_source(engine, "fleet", mode="full")
        logger.info("Ingested fleet capabilities: %s", result)
    except Exception as e:
        logger.error(f"Failed to ingest fleet capabilities: {e}")


def _mint_stdio_identity() -> None:
    """Resolve the process identity for a directly-served MCP server.

    CONCEPT:AU-OS.identity.authenticated-identity-enforcement — stdio has no Authorization header, so when
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
        # lock-safe) returns the same singleton the bootstrap builds. (CONCEPT:EG-KG.storage.nonblocking-checkpoint)
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
                # CONCEPT:AU-KG.ontology.federation-runtime — federation: load every ontology contributed by
                # installed fleet packages into the live reasoner at boot, alongside
                # the bundled TBox distribution. Failure-isolated so a bad or absent
                # provider never blocks engine bootstrap.
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
                except Exception:
                    logger.exception("Ontology federation sync at boot failed")
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
    )

    # Liveness endpoint for streamable-http/sse deployments (container
    # healthchecks). Does not touch the engine so it stays fast and lock-free;
    # the shard summary is config-only (no probe) — full per-shard
    # reachability lives on the gateway's /daemon/shards route and in the
    # unified daemon status. (CONCEPT:AU-OS.scaling.shard-topology-visibility-per)
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

    # ═══ Synthesized Tools (7 tools, action-routed) ═══

    from agent_utilities.mcp.tools import (
        register_analysis_tools,
        register_analyze_suite_tools,
        register_bus_tools,
        register_engine_surface_tools,
        register_engine_tools,
        register_ontology_tools,
        register_query_tools,
        register_reach_tools,
        register_secret_tools,
        register_state_tools,
        register_write_ingest_tools,
    )
    from agent_utilities.mcp.verbose_tools import register_tool_surface

    # graph-os is an action-routed wrapper over the API gateway's action core. The
    # condensed surface is the per-domain action tools (gated by `<DOMAIN>TOOL`); the
    # verbose surface is one 1:1 tool per gateway CRUD action, both dispatching through
    # the same `_execute_tool` core. register_tool_surface owns the MCP_TOOL_MODE
    # selection (condensed default / verbose / both) for both.
    register_tool_surface(
        mcp,
        service="graph-os",
        registrars=[
            register_query_tools,
            register_write_ingest_tools,
            register_analysis_tools,
            register_analyze_suite_tools,
            register_state_tools,
            register_ontology_tools,
            register_reach_tools,
            register_bus_tools,
            register_secret_tools,
            register_engine_tools,
            register_engine_surface_tools,
        ],
        verbose_register=register_graphos_verbose_tools,
    )

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
        mcp.tool(name=op["name"], tags={"verbose", op["tool"]})(fn)


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
                        "is banned by epistemic safety guardrails (CONCEPT:AU-OS.config.secrets-authentication)."
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
    # CONCEPT:AU-KG.ontology.federation-runtime — federation: explicit twin for ontology package-sync.
    route(
        "/graph/ontology/sync-packages",
        graph_ontology_sync_packages_endpoint,
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
        "/graph/orchestrate/distill-skills",
        graph_orchestrate_distill_skills_endpoint,
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
        "/graph/orchestrate/synthesize-org",
        graph_orchestrate_synthesize_org_endpoint,
        ["POST"],
    )
    route(
        "/graph/orchestrate/run-org",
        graph_orchestrate_run_org_endpoint,
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


def mcp_server() -> None:
    """``graph-os`` MCP server entry point (registered as console_scripts).

    Thin FastMCP wrapper following the standard ``mcp_server.py`` template: it
    serves ONLY the MCP tool surface, over ``stdio`` or ``streamable-http`` (or
    legacy ``sse``), selected by the standard ``--transport/--host/--port`` args
    from :func:`create_mcp_server`. The REST API (``/graph/*``, ``/sessions``,
    ``/goals``, ``/tools``) is centralized in the API gateway
    (``agent_utilities.gateway``) — see :func:`_mount_rest_routes`.
    """
    from agent_utilities.core.config import load_config

    load_config()  # resolve settings through the one shared XDG config.json
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
        fleet_mux = attach_fleet_loader(mcp, embed_fn=_fleet_embed_fn())
    except Exception:
        logger.exception("graph-os fleet loader attach failed (fleet tools disabled)")

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
    # identity + tenant scoping, or fail loud (CONCEPT:AU-OS.identity.authenticated-identity-enforcement). No-op for stdio.
    apply_served_security_profile(transport)

    try:
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
    finally:
        # Best-effort teardown of any lazily-mounted fleet children.
        if fleet_mux is not None:
            try:
                asyncio.run(fleet_mux.aclose())
            except Exception:
                logger.debug("fleet loader aclose failed", exc_info=True)


# Back-compat alias — the previous console_scripts entry and some docs/tooling
# reference ``main``; keep it pointing at the new thin entry point.
main = mcp_server


if __name__ == "__main__":
    mcp_server()
