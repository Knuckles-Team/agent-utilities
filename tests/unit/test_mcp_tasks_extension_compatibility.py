"""Wire-level compatibility guard for the 2026 MCP Tasks extension.

FastMCP 3.4.5 currently carries the 2025-11-25 experimental Tasks types, so
GraphOS must not advertise the newer extension until it can register all of its
handlers.  This exercises the real FastMCP initialization metadata, rather than
testing a local capability model.
"""

from __future__ import annotations

import pytest


def test_graphos_does_not_advertise_unimplemented_2026_tasks_extension() -> None:
    from agent_utilities.mcp.server_factory import create_mcp_server

    _args, mcp, _middlewares = create_mcp_server(
        name="graph-os-test",
        command_args=[],
    )

    capabilities = mcp._mcp_server.create_initialization_options().capabilities
    extensions = (capabilities.model_extra or {}).get("extensions", {})
    assert "io.modelcontextprotocol/tasks" not in extensions

    handlers = mcp._mcp_server.request_handlers
    assert not any(
        request_type.__name__ in {"TasksGetRequest", "TasksUpdateRequest"}
        for request_type in handlers
    )


@pytest.mark.asyncio
async def test_graph_jobs_cancel_uses_the_dispatched_work_item(monkeypatch) -> None:
    import json

    import agent_utilities.mcp.kg_server as kg
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
    from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
    from agent_utilities.mcp.tools.job_tools import register_job_tools
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext

    class _MCP:
        def tool(self, **_kwargs):
            return lambda function: function

    actor = ActorContext(
        actor_id="principal:mcp-task-test",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("kg:read", "kg:write"),
        tenant_id="tenant-mcp-task-test",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:read", "kg:write"}),
        policy_version="test",
        audience="agent-services",
    )
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    GraphComputeEngine(backend_type="rust")
    engine = IntelligenceGraphEngine(db_path=":memory:")
    register_job_tools(_MCP())
    monkeypatch.setattr(kg, "_get_engine", lambda: engine)

    with use_session(session):
        dispatched = json.loads(
            await kg._execute_tool("graph_jobs", action="dispatch", task="cancel me")
        )
        cancelled = json.loads(
            await kg._execute_tool(
                "graph_jobs", action="cancel", job_id=dispatched["job_id"]
            )
        )
        status = json.loads(
            await kg._execute_tool(
                "graph_jobs", action="status", job_id=dispatched["job_id"]
            )
        )

    assert cancelled == {"status": "cancelled", "job_id": dispatched["job_id"]}
    assert status["status"] == "cancelled"
