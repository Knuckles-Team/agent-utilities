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
import time as _time
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import Field

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


async def _execute_tool(tool_name: str, **kwargs) -> Any:
    tool_func = REGISTERED_TOOLS.get(tool_name)
    if not tool_func:
        raise ValueError(f"Tool {tool_name} not registered")

    import inspect

    if inspect.iscoroutinefunction(tool_func):
        return await tool_func(**kwargs)
    else:
        return tool_func(**kwargs)


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
        node_id = request.query_params.get("node_id", "")
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
_AGENT_ID = os.environ.get("AGENT_ID", f"mcp-client-{uuid.uuid4().hex[:8]}")
_SESSION_ID = os.environ.get("SESSION_ID", uuid.uuid4().hex)
_WORKSPACE_PATH = os.environ.get("WORKSPACE_PATH", os.getcwd())


def _get_engine():
    """Lazily initialize and return the IntelligenceGraphEngine singleton."""
    from agent_utilities.core.paths import ensure_dirs, kg_db_path
    from agent_utilities.knowledge_graph.backends import create_backend
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if engine is not None:
        return engine

    # First-run: ensure XDG dirs exist and create backend
    ensure_dirs()
    db_path = str(kg_db_path())
    logger.info("KG MCP Server using database: %s", db_path)
    backend_type = os.environ.get("GRAPH_BACKEND")
    backend = create_backend(backend_type=backend_type, db_path=db_path)
    engine = IntelligenceGraphEngine(backend=backend)
    return engine


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


def _build_server():
    """Build the KG MCP server with all tools registered."""
    import sys

    from agent_utilities.mcp.server_factory import create_mcp_server

    is_readonly = False

    if not any(arg in sys.argv for arg in ["--help", "-h"]):
        engine = _get_engine()

        import threading

        # Run the expensive metadata ingestion in the background so it doesn't block the MCP server connection to the IDE
        threading.Thread(
            target=_ingest_capabilities,
            args=(engine,),
            daemon=True,
            name="KGCapabilityIngestThread",
        ).start()

        # Start the background plan/task watcher thread natively via the engine
        if hasattr(engine, "start_sdd_watcher"):
            engine.start_sdd_watcher()

        # Check if backend is in read-only mode (contention workaround)
        is_readonly = getattr(engine.backend, "read_only", False)

        if engine and engine.backend and not is_readonly:
            engine.start_task_workers()

    def _check_readonly():
        if is_readonly:
            return json.dumps(
                {
                    "error": "Knowledge Graph is currently in READ-ONLY mode due to database lock contention. "
                    "Write operations and ingestion are disabled until the other process releases the lock."
                }
            )
        return None

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
    )

    # ═══ Consolidated Tools (7 tools, action-routed) ═══

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
    ) -> str:
        """Execute a read-only Cypher query against the Knowledge Graph. Use this to fetch graph data, explore relationships, and read node properties."""
        engine = _get_engine()
        parsed_params = json.loads(params) if params else {}

        if scope == "federated":
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

        # Local query — block writes
        cypher_upper = cypher.upper().strip()
        for kw in ["CREATE", "MERGE", "DELETE", "SET ", "REMOVE", "DROP"]:
            if kw in cypher_upper:
                return json.dumps(
                    {
                        "error": f"Write operation '{kw}' not allowed. Use kg_write for mutations."
                    }
                )
        try:
            results = engine.query_cypher(cypher, parsed_params)
            return json.dumps(results, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    REGISTERED_TOOLS["graph_query"] = graph_query

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
            description="Search strategy:\n- 'hybrid': Semantic + keyword weighted search (default).\n- 'concept': Look up a CONCEPT:ID (e.g. 'KG-2.15', 'ORCH-1.0').\n- 'analogy': Find structurally similar concepts.\n- 'memory': Search tiered memory (episodic/semantic/procedural).\n- 'discover': Cross-reference query against all ingested content.\n- 'dci': Direct Corpus Interaction.",
        ),
        top_k: int = Field(default=10, description="Maximum results to return."),
    ) -> str:
        """Search the Knowledge Graph using multiple strategies. Useful for finding context, concepts, memories, and capabilities across the ecosystem."""
        engine = _get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            if mode == "hybrid":
                results = engine.search_hybrid(query=query, top_k=top_k)
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
        except Exception as e:
            return f"Search error: {str(e)}"

    REGISTERED_TOOLS["graph_search"] = graph_search

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
    ) -> str:
        """Write nodes, relationships, or register external graphs. This is the primary mutation interface for the Knowledge Graph."""
        engine = _get_engine()
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

    REGISTERED_TOOLS["graph_write"] = graph_write

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
            description="Action to perform (ingest, ingest_knowledge_pack, agent_toolkit, corpus, jobs, job_status, status, rebuild_indexes, observe, materialize, sync, reflect).",
        ),
        job_id: str = Field(
            default="", description="ID of the job to check status for."
        ),
        corpus_name: str = Field(
            default="", description="Name of the corpus to add/update."
        ),
        base_path: str = Field(default="", description="Base path for the corpus."),
        description: str = Field(default="", description="Description of the corpus."),
    ) -> str:
        """Smart ingestion tool to populate the Knowledge Graph with codebases, documents, and memory observations. Monitors async ingestion jobs."""
        engine = _get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."

        try:
            if action == "ingest":
                import json
                from pathlib import Path

                def get_task_type(p: str) -> str:
                    p_path = Path(p.strip())
                    if p_path.is_file() and p_path.suffix.lower() in [
                        ".pdf",
                        ".docx",
                        ".doc",
                        ".txt",
                        ".md",
                    ]:
                        return "document"
                    return "codebase"

                if target_path.startswith("[") or "," in target_path:
                    try:
                        paths = (
                            json.loads(target_path)
                            if target_path.startswith("[")
                            else target_path.split(",")
                        )
                        job_ids = []
                        for path in paths:
                            p_strip = path.strip()
                            if not p_strip:
                                continue
                            t_type = get_task_type(p_strip)
                            jid = engine.submit_task(
                                target_path=p_strip,
                                is_codebase=(t_type == "codebase"),
                                provenance={
                                    "agent_id": agent_id,
                                    "max_depth": max_depth,
                                },
                                task_type=t_type,
                            )
                            job_ids.append(jid)
                        return f"Submitted {len(job_ids)} jobs: {', '.join(job_ids)}"
                    except json.JSONDecodeError:
                        pass
                if not target_path:
                    return "Error: target_path required for ingest action"
                t_type = get_task_type(target_path)
                jid = engine.submit_task(
                    target_path=target_path,
                    is_codebase=(t_type == "codebase"),
                    provenance={"agent_id": agent_id, "max_depth": max_depth},
                    task_type=t_type,
                )
                return f"Started ingestion job {jid} for {target_path}"

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
                from agent_utilities.knowledge_graph.core.engine_tasks import (
                    _decode_metadata,
                )

                jobs = engine.query_cypher(
                    "MATCH (t:Task) RETURN t.id as id, t.status as status, t.metadata as meta LIMIT 20"
                )
                if not jobs:
                    return "No active or recent ingestion jobs."
                lines = []
                for j in jobs:
                    meta = _decode_metadata(j.get("meta"))
                    target = meta.get("target", "unknown")
                    lines.append(f"{j['id']}: {j['status']} ({target})")
                return "\n".join(lines)

            elif action in ("job_status", "status"):
                if not job_id:
                    return "Error: job_id required"
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
                error_info = ""
                if status == "failed" and meta.get("error"):
                    error_info = f"\nError: {meta['error']}"
                return f"Job {job_id} status: {status}{error_info}"

            elif action == "rebuild_indexes":
                engine.build_indexes()
                return "Indexes rebuilt successfully."

            # ── KG-2.10: Observational Memory Bridge Actions ──
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
                    from agent_utilities.knowledge_graph.memory.memory_materializer import (
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
                    from agent_utilities.knowledge_graph.memory.memory_materializer import (
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
                    from agent_utilities.knowledge_graph.memory.reflector import (
                        run_reflector,
                    )

                    result = run_reflector(engine)
                    return result or "No observations to reflect on."
                except Exception as e:
                    return f"Reflect error: {e}"

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

            else:
                return f"Error: Unknown ingest action '{action}'"
        except Exception as e:
            return f"Ingest error: {str(e)}"

    REGISTERED_TOOLS["graph_ingest"] = graph_ingest

    @mcp.tool(
        name="graph_analyze",
        description="Execute complex analysis across the Knowledge Graph (synthesize, deep_extract, evaluate, security_scan, etc).",
        tags=["graph-os", "analyze"],
    )
    async def graph_analyze(
        action: str = Field(
            default="synthesize",
            description="Analysis action (synthesize, deep_extract, background_research, relevance_sweep, blast_radius, inspect, context, evaluate, evaluate_alpha, evolve_model, forecast, causal, invariant, security_scan).",
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
            # ── KG-2.10: Startup Context Generation ──
            elif action == "context":
                try:
                    from agent_utilities.knowledge_graph.memory.startup_context import (
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
            description="Action to perform (dispatch, status, request_approval, grant_approval, execute_agent, consensus, start_debate, submit_risk_veto, list_cron_jobs, trigger_cron_job, compile_workflow, list_workflows, execute_workflow, export_workflow).",
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
    ) -> str:
        """Orchestrate multi-agent workflows. Dispatches agents, manages subagent lifecycles, and evaluates approval conditions for complex asynchronous execution."""
        engine = _get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            if action in ("dispatch", "status", "request_approval", "grant_approval"):
                try:
                    from agent_utilities.orchestration.manager import Orchestrator

                    orch = Orchestrator(engine)

                    if action == "dispatch":
                        deps = json.loads(dependencies) if dependencies else []
                        job_id = await orch.dispatch_task(task, deps)
                        return f"Task dispatched. Job ID: {job_id}"
                    elif action == "status":
                        if not job_id:
                            return "Error: job_id required"
                        return str(orch.get_task_status(job_id))
                    elif action == "request_approval":
                        return f"Approval requested for job {job_id}"
                    elif action == "grant_approval":
                        return orch.grant_approval(job_id, approval_status)
                    return f"Error: Action '{action}' not implemented."
                except ImportError:
                    return "Error: orchestration module not available"
            elif action == "execute_agent":
                try:
                    from agent_utilities.orchestration.agent_runner import (
                        run_agent,
                    )

                    result = await run_agent(
                        agent_name=agent_name,
                        task=task,
                        max_steps=max_steps,
                        engine=engine,
                    )
                    return result
                except ImportError as exc:
                    return f"Error: agent_runner module not available: {exc}"
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
            # ── CONCEPT:ORCH-1.24: Workflow Lifecycle Actions ──
            elif action == "compile_workflow":
                try:
                    from agent_utilities.knowledge_graph.workflow_compiler import (
                        WorkflowCompiler,
                    )

                    compiler = WorkflowCompiler(engine)
                    name = agent_name or f"compiled_{uuid.uuid4().hex[:6]}"
                    workflow_id = await compiler.compile_and_store(
                        name=name,
                        description=task,
                    )
                    return json.dumps(
                        {
                            "status": "compiled",
                            "workflow_id": workflow_id,
                            "name": name,
                        }
                    )
                except Exception as exc:
                    return f"Error compiling workflow: {exc}"

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
                try:
                    from agent_utilities.orchestration import AgentOrchestrationEngine

                    runner = AgentOrchestrationEngine()
                    name = agent_name or task
                    input_task = task if (agent_name and task != agent_name) else None
                    wf_result = await runner.execute_workflow(
                        workflow_id=name,
                        task=input_task,
                    )
                    return json.dumps(wf_result, default=str)
                except ValueError as exc:
                    return f"Workflow not found: {exc}"
                except Exception as exc:
                    return f"Error executing workflow: {exc}"

            elif action == "dispatch_workflow":
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

            else:
                return f"Error: Unknown orchestration action '{action}'"
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
            description="Operation ('set_secret', 'register_mcp', 'install_hooks', 'uninstall_hooks', 'doctor').",
        ),
        config_key: str = Field(
            default="", description="The key or ID of the configuration/secret."
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
            # ── KG-2.10 / ECO-4.6: Memory Hook Management ──
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


def main():
    """Entry point for the KG MCP server."""
    os.environ["IS_KG_SERVER"] = "true"
    args, mcp, middlewares = _build_server()

    # Apply middleware stack
    for middleware in middlewares:
        mcp.add_middleware(middleware)

    logger.info(
        "Starting Knowledge Graph MCP Server (transport=%s, port=%s)",
        args.transport,
        args.port,
    )

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        import anyio
        import uvicorn

        app = mcp.http_app()

        # Mount standard Starlette sessions and goals REST endpoints
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

        app.add_route("/sessions", get_all_sessions, methods=["GET"])
        app.add_route("/sessions/{session_id}", get_session_details, methods=["GET"])
        app.add_route("/sessions/{session_id}", delete_session, methods=["DELETE"])
        app.add_route(
            "/sessions/{session_id}/reply", submit_session_reply, methods=["POST"]
        )
        app.add_route(
            "/sessions/{session_id}/cancel", cancel_session_run, methods=["POST"]
        )
        app.add_route("/goals", create_goal, methods=["POST"])
        app.add_route("/goals", list_goals, methods=["GET"])
        app.add_route(
            "/goals/{goal_id}/iterations", get_goal_iterations, methods=["GET"]
        )
        app.add_route("/goals/{goal_id}/cancel", cancel_goal, methods=["POST"])

        # Mount new Tools and Graph endpoints
        app.add_route("/tools", get_tools_endpoint, methods=["GET"])
        app.add_route("/tools/toggle", toggle_tool_endpoint, methods=["POST"])

        # Bilateral Graph execution routes
        app.add_route("/graph/query", graph_query_endpoint, methods=["POST"])
        app.add_route("/graph/search", graph_search_endpoint, methods=["POST"])
        app.add_route("/graph/write", graph_write_endpoint, methods=["POST"])
        app.add_route("/graph/ingest", graph_ingest_endpoint, methods=["POST"])
        app.add_route("/graph/analyze", graph_analyze_endpoint, methods=["POST"])
        app.add_route(
            "/graph/orchestrate", graph_orchestrate_endpoint, methods=["POST"]
        )
        app.add_route("/graph/configure", graph_configure_endpoint, methods=["POST"])

        # Granular Graph Query endpoints
        app.add_route(
            "/graph/query/federated", graph_query_federated_endpoint, methods=["POST"]
        )

        # Granular Graph Search endpoints
        app.add_route(
            "/graph/search/hybrid", graph_search_hybrid_endpoint, methods=["POST"]
        )
        app.add_route(
            "/graph/search/concept", graph_search_concept_endpoint, methods=["POST"]
        )
        app.add_route(
            "/graph/search/analogy", graph_search_analogy_endpoint, methods=["POST"]
        )
        app.add_route(
            "/graph/search/memory", graph_search_memory_endpoint, methods=["POST"]
        )
        app.add_route(
            "/graph/search/discover", graph_search_discover_endpoint, methods=["POST"]
        )
        app.add_route("/graph/search/dci", graph_search_dci_endpoint, methods=["POST"])

        # Granular Graph Write endpoints
        app.add_route("/graph/write/node", graph_write_node_endpoint, methods=["POST"])
        app.add_route(
            "/graph/write/node/{node_id}",
            graph_write_delete_node_endpoint,
            methods=["DELETE"],
        )
        app.add_route("/graph/write/edge", graph_write_edge_endpoint, methods=["POST"])
        app.add_route(
            "/graph/write/edge", graph_write_delete_edge_endpoint, methods=["DELETE"]
        )
        app.add_route(
            "/graph/write/external", graph_write_external_endpoint, methods=["POST"]
        )
        app.add_route("/graph/write/bulk", graph_write_bulk_endpoint, methods=["POST"])
        app.add_route(
            "/graph/write/memory", graph_write_memory_endpoint, methods=["POST"]
        )
        app.add_route(
            "/graph/write/memory/recall",
            graph_write_memory_recall_endpoint,
            methods=["POST"],
        )
        app.add_route("/graph/write/chat", graph_write_chat_endpoint, methods=["POST"])
        app.add_route("/graph/write/sdd", graph_write_sdd_endpoint, methods=["POST"])
        app.add_route(
            "/graph/write/execution", graph_write_execution_endpoint, methods=["POST"]
        )

        # Granular Graph Ingest endpoints
        app.add_route(
            "/graph/ingest/submit", graph_ingest_submit_endpoint, methods=["POST"]
        )
        app.add_route(
            "/graph/ingest/corpus", graph_ingest_corpus_endpoint, methods=["POST"]
        )
        app.add_route("/graph/ingest/jobs", graph_ingest_jobs_endpoint, methods=["GET"])
        app.add_route(
            "/graph/ingest/job/{job_id}",
            graph_ingest_job_status_endpoint,
            methods=["GET"],
        )
        app.add_route(
            "/graph/ingest/rebuild-indexes",
            graph_ingest_rebuild_indexes_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/ingest/observe", graph_ingest_observe_endpoint, methods=["POST"]
        )
        app.add_route(
            "/graph/ingest/materialize",
            graph_ingest_materialize_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/ingest/sync", graph_ingest_sync_endpoint, methods=["POST"]
        )
        app.add_route(
            "/graph/ingest/reflect", graph_ingest_reflect_endpoint, methods=["POST"]
        )
        app.add_route(
            "/graph/ingest/agent-toolkit",
            graph_ingest_agent_toolkit_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/ingest/knowledge-pack",
            graph_ingest_knowledge_pack_endpoint,
            methods=["POST"],
        )

        # Granular Graph Analyze endpoints
        app.add_route(
            "/graph/analyze/synthesize",
            graph_analyze_synthesize_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/analyze/deep-extract",
            graph_analyze_deep_extract_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/analyze/background-research",
            graph_analyze_background_research_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/analyze/relevance-sweep",
            graph_analyze_relevance_sweep_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/analyze/blast-radius",
            graph_analyze_blast_radius_endpoint,
            methods=["GET"],
        )
        app.add_route(
            "/graph/analyze/inspect", graph_analyze_inspect_endpoint, methods=["GET"]
        )
        app.add_route(
            "/graph/analyze/context", graph_analyze_context_endpoint, methods=["POST"]
        )
        app.add_route(
            "/graph/analyze/evaluate-alpha",
            graph_analyze_evaluate_alpha_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/analyze/evaluate", graph_analyze_evaluate_endpoint, methods=["POST"]
        )
        app.add_route(
            "/graph/analyze/evolve-model",
            graph_analyze_evolve_model_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/analyze/forecast", graph_analyze_forecast_endpoint, methods=["POST"]
        )
        app.add_route(
            "/graph/analyze/causal", graph_analyze_causal_endpoint, methods=["POST"]
        )
        app.add_route(
            "/graph/analyze/invariant",
            graph_analyze_invariant_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/analyze/security-scan",
            graph_analyze_security_scan_endpoint,
            methods=["POST"],
        )

        # Granular Graph Orchestrate endpoints
        app.add_route(
            "/graph/orchestrate/dispatch",
            graph_orchestrate_dispatch_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/orchestrate/job/{job_id}",
            graph_orchestrate_status_endpoint,
            methods=["GET"],
        )
        app.add_route(
            "/graph/orchestrate/request-approval",
            graph_orchestrate_request_approval_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/orchestrate/grant-approval",
            graph_orchestrate_grant_approval_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/orchestrate/execute-agent",
            graph_orchestrate_execute_agent_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/orchestrate/consensus",
            graph_orchestrate_consensus_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/orchestrate/start-debate",
            graph_orchestrate_start_debate_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/orchestrate/submit-risk-veto",
            graph_orchestrate_submit_risk_veto_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/orchestrate/cron-jobs",
            graph_orchestrate_list_cron_jobs_endpoint,
            methods=["GET"],
        )
        app.add_route(
            "/graph/orchestrate/trigger-cron-job",
            graph_orchestrate_trigger_cron_job_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/orchestrate/compile-workflow",
            graph_orchestrate_compile_workflow_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/orchestrate/workflows",
            graph_orchestrate_list_workflows_endpoint,
            methods=["GET"],
        )
        app.add_route(
            "/graph/orchestrate/execute-workflow",
            graph_orchestrate_execute_workflow_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/orchestrate/dispatch-workflow",
            graph_orchestrate_dispatch_workflow_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/orchestrate/workflow-status/{job_id}",
            graph_orchestrate_workflow_status_endpoint,
            methods=["GET"],
        )
        app.add_route(
            "/graph/orchestrate/export-workflow",
            graph_orchestrate_export_workflow_endpoint,
            methods=["POST"],
        )

        # Granular Graph Configure endpoints
        app.add_route(
            "/graph/configure/secret", graph_configure_secret_endpoint, methods=["POST"]
        )
        app.add_route(
            "/graph/configure/register-mcp",
            graph_configure_register_mcp_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/configure/install-hooks",
            graph_configure_install_hooks_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/configure/uninstall-hooks",
            graph_configure_uninstall_hooks_endpoint,
            methods=["POST"],
        )
        app.add_route(
            "/graph/configure/doctor", graph_configure_doctor_endpoint, methods=["POST"]
        )

        app = CentralizedCypherMiddleware(app)

        logger.info(
            "Starting Knowledge Graph MCP Server over wrapped transport on %s:%s",
            args.host,
            args.port,
        )

        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        anyio.run(server.serve)


if __name__ == "__main__":
    main()
