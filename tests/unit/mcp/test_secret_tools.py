"""Tests for the graph_secret MCP tool + REST twin (CONCEPT:AU-OS.identity.encrypted-secret-store).

Exercises the live two-surface path: the shared ``_execute_tool`` action core
that both the MCP tool and the auto-mounted ``/graph/secret`` REST route dispatch
into, plus the ActionPolicy governance on mutations.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.mcp import kg_server


@pytest.fixture
def registered():
    kg_server.ensure_tools_registered()
    return kg_server


@pytest.fixture
def isolated_secrets(monkeypatch):
    """Pin the tool's ``create_secrets_client`` to a unique throwaway-graph
    engine-backed client, so the CRUD test is isolated from the shared
    ``__secrets__`` graph (the tool imports the symbol at call time)."""
    import uuid

    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
    from agent_utilities.security import secrets_client as sc

    graph = GraphComputeEngine(graph_name=f"__secrets_test_{uuid.uuid4().hex[:12]}__")
    client = sc.SecretsClient(backend=sc.InEpistemicGraphBackend(graph=graph))
    monkeypatch.setattr(sc, "create_secrets_client", lambda config=None: client)
    return client


@pytest.mark.concept("CONCEPT:AU-OS.identity.encrypted-secret-store")
def test_graph_secret_registered_on_both_surfaces(registered):
    assert "graph_secret" in registered.REGISTERED_TOOLS
    assert registered.ACTION_TOOL_ROUTES.get("graph_secret") == "/graph/secret"


@pytest.mark.concept("CONCEPT:AU-OS.identity.encrypted-secret-store")
@pytest.mark.asyncio
async def test_set_get_list_delete_round_trip(
    registered, isolated_secrets, monkeypatch
):
    """Full CRUD through the shared action core (set is governed → allow it)."""
    # Allow secret.set/secret.delete for this test (default is approval_required).
    import agent_utilities.mcp.tools.secret_tools as st

    monkeypatch.setattr(
        st, "_gate", lambda kind, target, reason: (True, {"tier": "auto"})
    )

    set_res = json.loads(
        await kg_server._execute_tool(
            "graph_secret",
            action="set",
            key="svc/token",
            value="s3cr3t",
            reason="test",
        )
    )
    assert set_res["status"] == "stored"

    get_res = json.loads(
        await kg_server._execute_tool("graph_secret", action="get", key="svc/token")
    )
    assert get_res["value"] == "s3cr3t"

    list_res = json.loads(await kg_server._execute_tool("graph_secret", action="list"))
    assert "svc/token" in list_res["keys"]
    # list never leaks values.
    assert "s3cr3t" not in json.dumps(list_res)

    del_res = json.loads(
        await kg_server._execute_tool(
            "graph_secret", action="delete", key="svc/token", reason="test"
        )
    )
    assert del_res["deleted"] is True


@pytest.mark.concept("CONCEPT:AU-OS.identity.encrypted-secret-store")
@pytest.mark.asyncio
async def test_set_denied_by_policy_does_not_store(
    registered, isolated_secrets, monkeypatch
):
    """When ActionPolicy denies secret.set, nothing is written."""
    import agent_utilities.mcp.tools.secret_tools as st

    monkeypatch.setattr(
        st,
        "_gate",
        lambda kind, target, reason: (False, {"decision": "queue_approval"}),
    )
    res = json.loads(
        await kg_server._execute_tool(
            "graph_secret",
            action="set",
            key="blocked/key",
            value="nope",
        )
    )
    assert res["error"] == "policy_denied"
    got = json.loads(
        await kg_server._execute_tool("graph_secret", action="get", key="blocked/key")
    )
    assert got["value"] is None


@pytest.mark.concept("CONCEPT:AU-OS.identity.encrypted-secret-store")
@pytest.mark.asyncio
async def test_unknown_action(registered):
    res = json.loads(await kg_server._execute_tool("graph_secret", action="bogus"))
    assert "unknown action" in res["error"]
