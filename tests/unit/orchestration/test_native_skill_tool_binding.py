"""Native GraphOS tool binding for delegated KG-backed skills."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Annotated, Any
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import Field
from pydantic_ai import Tool
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    current_session,
    use_session,
)
from agent_utilities.mcp import kg_server
from agent_utilities.models.company_brain import ActorType
from agent_utilities.orchestration.agent_runner import _bind_native_skill_toolset
from agent_utilities.security.brain_context import (
    ActorContext,
    current_actor,
    use_actor,
)
from agent_utilities.security.tool_guard import flag_mcp_tool_definitions


async def _lookup_tool(
    query: Annotated[
        str,
        Field(min_length=2, description="Search expression."),
    ],
    limit: Annotated[int, Field(ge=1, le=25)] = 5,
) -> dict[str, Any]:
    """Look up graph records without mutating them."""
    return {"query": query, "limit": limit}


async def _write_tool(node_id: str) -> dict[str, str]:
    """Write one graph record."""
    return {"node_id": node_id}


async def _identity_probe(subject: str, limit: int = 2) -> dict[str, Any]:
    """Return authority observed inside the registered tool implementation."""
    session = current_session()
    actor = current_actor()
    assert session is not None
    return {
        "subject": subject,
        "limit": limit,
        "actor_id": actor.actor_id,
        "tenant": session.tenant,
        "same_actor": actor == session.actor,
    }


@pytest.fixture
def isolated_tool_registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Keep these binding tests independent from GraphOS server registration."""
    registry: dict[str, Any] = {}
    monkeypatch.setattr(kg_server, "REGISTERED_TOOLS", registry)
    return registry


def test_skill_binds_exact_caller_allowed_and_declared_native_tools(
    isolated_tool_registry: dict[str, Any],
) -> None:
    isolated_tool_registry.update(
        {
            "graph_lookup": _lookup_tool,
            "graph_write_record": _write_tool,
        }
    )
    config: dict[str, Any] = {
        "invoker_allowed_tools": ["graph_write_record"],
        "mcp_toolsets": [],
    }
    agent_meta = {
        "type": "skill",
        "tools": [
            {"name": "graph_lookup"},
            {"name": "graph_write_record"},
        ],
    }

    _bind_native_skill_toolset(
        config=config,
        agent_meta=agent_meta,
        agent_name="modeling-skill",
    )

    assert len(config["mcp_toolsets"]) == 1
    native = config["mcp_toolsets"][0]
    assert native.id == "modeling-skill"
    assert native.metadata == {"graphos_native": True}
    assert list(native.tools) == ["graph_write_record"]

    kernel = Mock()
    kernel.authorize_tool.return_value = "allow"
    identity = object()
    guarded = flag_mcp_tool_definitions(
        [native],
        permissions_kernel=kernel,
        agent_identity=identity,
    )

    assert len(guarded) == 1
    assert isinstance(guarded[0], ApprovalRequiredToolset)
    assert guarded[0].wrapped is native
    assert (
        guarded[0].approval_required_func(
            SimpleNamespace(),
            SimpleNamespace(name="graph_write_record", metadata={}),
            {"node_id": "node-1"},
        )
        is False
    )
    kernel.authorize_tool.assert_called_once_with(
        identity,
        "graph_write_record",
        required_capability=None,
    )


def test_skill_rejects_caller_allowed_tool_not_declared_by_skill(
    isolated_tool_registry: dict[str, Any],
) -> None:
    isolated_tool_registry["graph_lookup"] = _lookup_tool
    config: dict[str, Any] = {
        "invoker_allowed_tools": ["graph_lookup"],
        "mcp_toolsets": [],
    }

    with pytest.raises(PermissionError, match="undeclared tool"):
        _bind_native_skill_toolset(
            config=config,
            agent_meta={"type": "skill", "tools": [{"name": "graph_other"}]},
            agent_name="query-skill",
        )

    assert config["mcp_toolsets"] == []


def test_platform_skill_binds_explicit_caller_allow_list_without_verb_ownership(
    isolated_tool_registry: dict[str, Any],
) -> None:
    """Platform skills own workflows, while caller/session policy grants tools."""

    isolated_tool_registry["graph_lookup"] = _lookup_tool
    config: dict[str, Any] = {
        "invoker_allowed_tools": ["graph_lookup"],
        "mcp_toolsets": [],
    }

    _bind_native_skill_toolset(
        config=config,
        agent_meta={"type": "skill", "tools": []},
        agent_name="platform-skill",
    )

    assert list(config["mcp_toolsets"][0].tools) == ["graph_lookup"]


def test_native_skill_binding_rejects_recursive_graphos_delegation(
    isolated_tool_registry: dict[str, Any],
) -> None:
    isolated_tool_registry["graph_orchestrate"] = _lookup_tool
    config: dict[str, Any] = {
        "invoker_allowed_tools": ["graph_orchestrate"],
        "mcp_toolsets": [],
    }

    with pytest.raises(PermissionError, match="recursive native GraphOS delegation"):
        _bind_native_skill_toolset(
            config=config,
            agent_meta={"type": "skill", "tools": []},
            agent_name="platform-skill",
        )

    assert config["mcp_toolsets"] == []


def test_skill_rejects_declared_tool_missing_from_graphos_registry(
    isolated_tool_registry: dict[str, Any],
) -> None:
    assert "graph_missing" not in isolated_tool_registry
    config: dict[str, Any] = {
        "invoker_allowed_tools": ["graph_missing"],
        "mcp_toolsets": [],
    }

    with pytest.raises(RuntimeError, match="native tool is unavailable"):
        _bind_native_skill_toolset(
            config=config,
            agent_meta={"type": "skill", "tools": [{"name": "graph_missing"}]},
            agent_name="query-skill",
        )

    assert config["mcp_toolsets"] == []


def test_native_toolset_preserves_registered_tool_schema(
    isolated_tool_registry: dict[str, Any],
) -> None:
    isolated_tool_registry["graph_lookup"] = _lookup_tool
    source = Tool(_lookup_tool, name="graph_lookup")

    native = kg_server.build_native_graphos_toolset(
        ["graph_lookup"],
        toolset_id="query-skill",
    )
    bound = native.tools["graph_lookup"]

    assert bound.name == source.name
    assert bound.description == source.description
    assert bound.function_schema.json_schema == source.function_schema.json_schema
    assert bound.function_schema.json_schema == {
        "additionalProperties": False,
        "properties": {
            "query": {
                "description": "Search expression.",
                "minLength": 2,
                "type": "string",
            },
            "limit": {
                "default": 5,
                "maximum": 25,
                "minimum": 1,
                "type": "integer",
            },
        },
        "required": ["query"],
        "type": "object",
    }


async def test_native_wrapper_dispatches_through_execute_tool_with_verified_session(
    isolated_tool_registry: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_tool_registry["graph_identity_probe"] = _identity_probe
    native = kg_server.build_native_graphos_toolset(
        ["graph_identity_probe"],
        toolset_id="runtime-skill",
    )
    real_execute_tool = kg_server._execute_tool
    execute_spy = AsyncMock(wraps=real_execute_tool)
    monkeypatch.setattr(kg_server, "_execute_tool", execute_spy)

    actor = ActorContext(
        actor_id="principal:skill-test",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("kg:read",),
        tenant_id="tenant-test",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant="tenant-test",
        scopes=frozenset({"kg:read"}),
        policy_version="policy-test",
        audience="agent-services",
    )

    with use_actor(actor), use_session(session):
        result = await native.tools["graph_identity_probe"].function(
            subject="delegated-skill"
        )

    execute_spy.assert_awaited_once_with(
        "graph_identity_probe",
        subject="delegated-skill",
    )
    assert result == {
        "subject": "delegated-skill",
        "limit": 2,
        "actor_id": "principal:skill-test",
        "tenant": "tenant-test",
        "same_actor": True,
    }
