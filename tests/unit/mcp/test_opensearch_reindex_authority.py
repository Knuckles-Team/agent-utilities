"""The public reindex action must use verified graph authority before side effects."""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    ScopeError,
    SessionRequiredError,
    suspend_session,
    use_session,
)
from agent_utilities.knowledge_graph.search import rebuild
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools.write_ingest_tools import (
    _ingest_action_opensearch_reindex,
    register_write_ingest_tools,
)
from agent_utilities.mcp.write_ingest_types import IngestRequest
from agent_utilities.security.brain_context import ActorContext, ActorType


def _session(*scopes: str) -> GraphSession:
    actor = ActorContext(
        actor_id="reindex-test-service",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=(),
        tenant_id="tenant-a",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset(scopes),
        graph="tenant-a",
        policy_version="test-policy",
        audience="test-audience",
    )


@pytest.fixture
def reindex_calls(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, str | None, dict]] = []

    def fake_reindex(tenant: str, object_type: str | None, **kwargs):
        calls.append((tenant, object_type, kwargs))
        return {"status": "ok"}

    monkeypatch.setattr(rebuild, "mcp_reindex", fake_reindex)
    return calls


@pytest.mark.parametrize(
    "kwargs",
    [
        {"corpus_name": "tenant-b"},
        {"graph": "tenant-b"},
        {"connection": "other-backend"},
    ],
)
async def test_reindex_denies_caller_retarget_before_worker(
    reindex_calls, kwargs: dict[str, str]
) -> None:
    with use_session(_session("kg:admin")), pytest.raises(PermissionError):
        await _ingest_action_opensearch_reindex(
            object(), IngestRequest(action="opensearch_reindex", **kwargs)
        )
    assert reindex_calls == []


async def test_reindex_requires_admin_scope_before_worker(reindex_calls) -> None:
    with use_session(_session("kg:write")), pytest.raises(ScopeError):
        await _ingest_action_opensearch_reindex(
            object(), IngestRequest(action="opensearch_reindex", corpus_name="tenant-a")
        )
    assert reindex_calls == []


async def test_reindex_requires_verified_session_before_worker(reindex_calls) -> None:
    with suspend_session(), pytest.raises(SessionRequiredError):
        await _ingest_action_opensearch_reindex(
            object(), IngestRequest(action="opensearch_reindex", corpus_name="tenant-a")
        )
    assert reindex_calls == []


async def test_reindex_uses_verified_tenant_and_preserves_incremental_index(
    reindex_calls,
) -> None:
    with use_session(_session("kg:admin")):
        result = await _ingest_action_opensearch_reindex(
            object(),
            IngestRequest(
                action="opensearch_reindex", target_path="Person", base_path="7"
            ),
        )
    assert json.loads(result) == {"status": "ok"}
    assert reindex_calls == [
        ("tenant-a", "Person", {"from_seq": 7, "drop_existing": False})
    ]


async def test_reindex_full_replay_drops_only_verified_tenant(
    reindex_calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastmcp import FastMCP

    previous_session = kg_server._PROCESS_SESSION
    register_write_ingest_tools(FastMCP("reindex-authority-test"))
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    kg_server._PROCESS_SESSION = _session("kg:admin")
    try:
        result = await kg_server._execute_tool(
            "graph_ingest", action="opensearch_reindex", corpus_name="tenant-a"
        )
    finally:
        kg_server._PROCESS_SESSION = previous_session
    assert json.loads(result) == {"status": "ok"}
    assert reindex_calls == [
        ("tenant-a", None, {"from_seq": 0, "drop_existing": True})
    ]
