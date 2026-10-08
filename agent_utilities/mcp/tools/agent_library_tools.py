"""The ``agent_library`` MCP tool: list, get and save Agent Library records.

AU-CONTROL-R024. The graph-os intent surface reaches it through ``find`` (rank),
``ask`` (the read-only ``list``/``get`` actions) and ``manage`` (``save`` and
``seed_roles``, previewed then executed with a plan reference). Every action
reads or writes through :class:`agent_utilities.orchestration.agent_library.AgentLibrary`,
the same store the agent-webui ``/api/enhanced/agent-library`` routes use.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.security.error_surface import public_error_text

AgentLibraryAction = Literal["list", "get", "save", "seed_roles"]


def _field(data: dict[str, Any], *keys: str) -> str:
    """The first non-empty ``keys`` value of ``data``, stripped."""
    for key in keys:
        value = str(data.get(key) or "").strip()
        if value:
            return value
    return ""


def _record_from_json(agent_json: str) -> Any:
    from agent_utilities.orchestration.agent_library import AgentRecord

    data = json.loads(agent_json or "{}")
    if not isinstance(data, dict):
        raise ValueError("agent_json must decode to an object")
    return AgentRecord(
        name=_field(data, "name"),
        system_prompt=_field(data, "system_prompt", "instructions"),
        description=_field(data, "description"),
        role=_field(data, "role"),
        tools=tuple(str(t) for t in data.get("tools") or ()),
        skills=tuple(str(s) for s in data.get("skills") or ()),
        model_profile=_field(data, "model_profile"),
        mcp_server=_field(data, "mcp_server"),
        context_policy=dict(data.get("context_policy") or {}),
        source="agent_library tool",
    )


def _list(library: Any, *, kind: str, **_: Any) -> dict[str, Any]:
    return {"agents": [r.view() for r in library.list(kind or None)]}


def _get(library: Any, *, agent_id: str, **_: Any) -> dict[str, Any]:
    record = library.get(agent_id)
    if record is None:
        return {"error": "agent not found", "agent_id": agent_id}
    view = record.view()
    view["system_prompt"] = record.system_prompt
    view["tools"] = [dict(ref) for ref in record.tool_refs]
    view["graph"] = dict(record.graph) if record.graph else None
    return {"agent": view}


def _save(library: Any, *, agent_json: str, **_: Any) -> dict[str, Any]:
    return {"agent": library.save(_record_from_json(agent_json)).view()}


def _seed_roles(library: Any, **_: Any) -> dict[str, Any]:
    from agent_utilities.orchestration.agent_library import seed_role_agents

    return {"agents": [r.view() for r in seed_role_agents(library)]}


_ACTIONS = {"list": _list, "get": _get, "save": _save, "seed_roles": _seed_roles}


def run_agent_library_action(
    engine: Any,
    action: str,
    *,
    agent_id: str = "",
    kind: str = "",
    agent_json: str = "",
) -> dict[str, Any]:
    """One ``agent_library`` action against ``engine``'s Agent Library."""
    from agent_utilities.orchestration.agent_library import AgentLibrary

    handler = _ACTIONS.get(action)
    if handler is None:
        return {"error": f"unknown agent_library action {action!r}"}
    return handler(
        AgentLibrary(engine), agent_id=agent_id, kind=kind, agent_json=agent_json
    )


def register_agent_library_tool(mcp: Any) -> None:
    """Register the ``agent_library`` tool and its REST twin route."""

    @mcp.tool(
        name="agent_library",
        description=(
            "The Agent Library of built agents (system prompt, tools, skills, model "
            "profile, context policy, role), stored in the knowledge graph. Actions: "
            "'list' lists agents (optional kind: local | role | agent_graph); 'get' "
            "returns one agent by agent_id; 'save' stores agent_json "
            "{name, system_prompt, description, role, tools, skills, model_profile, "
            "mcp_server, context_policy}; 'seed_roles' saves a role agent for every "
            "packaged fleet prompt that names a role."
        ),
        tags=["graph-os", "agents", "agent-library"],
    )
    async def agent_library(
        action: AgentLibraryAction = Field(
            default="list", description="list | get | save | seed_roles"
        ),
        agent_id: str = Field(default="", description="Agent id (action=get)."),
        kind: str = Field(
            default="", description="list filter: local | role | agent_graph."
        ),
        agent_json: str = Field(
            default="{}", description="JSON agent record (action=save)."
        ),
    ) -> str:
        engine = kg_server._get_engine()
        if engine is None:
            return json.dumps({"error": "IntelligenceGraphEngine not active."})
        try:
            result = run_agent_library_action(
                engine, action, agent_id=agent_id, kind=kind, agent_json=agent_json
            )
        except PermissionError:
            raise
        except Exception as exc:
            return public_error_text(exc)
        return json.dumps(result, default=str)

    kg_server.REGISTERED_TOOLS["agent_library"] = agent_library
    kg_server.ACTION_TOOL_ROUTES["agent_library"] = "/graph/agent-library"


__all__ = ["register_agent_library_tool", "run_agent_library_action"]
