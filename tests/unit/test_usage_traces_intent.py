"""Reviewed traces classification preserves the real intent and tenant guards."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.core.session import current_session, use_session
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import intent_tools
from agent_utilities.security.brain_context import use_actor
from tests.unit.cpd_generator_support import generated_action_items


@pytest.fixture
def traces_surface(monkeypatch):
    kg_server.ensure_tools_registered()
    cpds = copy.deepcopy(intent_tools._load_cpds_required())
    monkeypatch.setattr(intent_tools, "_load_cpds_required", lambda: cpds)
    for attr in ("_CANDIDATES_CACHE", "_ACTIONS_BY_TOOL_CACHE", "_OUTCOME_ROUTER"):
        monkeypatch.setattr(intent_tools, attr, None)
    monkeypatch.setattr(intent_tools, "_RESOLUTION_CACHE", {})
    monkeypatch.setattr(intent_tools, "_PREVIEW_PLAN_CACHE", {})
    from agent_utilities.observability import langfuse_exporter
    from agent_utilities.usage import service

    seen = []

    def sessions(**filters):
        seen.append(filters)
        return [SimpleNamespace(id="trace-example", project="example")]

    monkeypatch.setattr(
        service, "get_usage_service", lambda: SimpleNamespace(sessions=sessions)
    )
    monkeypatch.setattr(
        langfuse_exporter,
        "get_langfuse_exporter",
        lambda: SimpleNamespace(enabled=True),
    )
    return cpds, seen


async def ask_traces(**hints):
    return json.loads(
        await kg_server._execute_tool(
            "ask",
            intent="show usage runtime traces",
            hints_json=json.dumps({"tool": "usage_query", "action": "traces", **hints}),
        )
    )


@pytest.mark.asyncio
async def test_traces_ask_reaches_real_tenant_scoped_handler(traces_surface):
    _, seen = traces_surface
    session = current_session()
    result = await ask_traces(origin="ingested", limit=999)
    assert result["executed"] is True
    assert result["routing"]["chosen_tool"] == "usage_query"
    payload = json.loads(result["result"])
    assert payload["trace_count"] == 1
    assert payload["traces"][0]["trace_ref"].startswith("pref_")
    assert seen == [
        {"tenant_id": session.actor.tenant_id, "origin": "runtime", "limit": 500}
    ]
    assert current_session() is session


@pytest.mark.asyncio
async def test_traces_ask_cannot_select_another_tenant(traces_surface):
    _, seen = traces_surface
    session = current_session()
    actor = replace(session.actor, roles=("usage:read",))
    with use_actor(actor), use_session(replace(session, actor=actor)):
        result = await ask_traces(tenant_id="another-tenant")
    assert "cross-tenant usage access denied" in str(result)
    assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["unknown_action", "delete_traces"])
async def test_ask_unknown_usage_actions_never_dispatch(traces_surface, action):
    _, seen = traces_surface
    result = await ask_traces(action=action)
    assert result["executed"] is False
    assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", [None, "true"])
async def test_unknown_or_mutating_effect_fails_closed(traces_surface, mutation):
    cpds, seen = traces_surface
    operation = next(d for d in cpds["usage_query"]["does"] if d["action"] == "traces")
    if mutation is None:
        operation.pop("mutates", None)
    else:
        operation["mutates"] = mutation
    result = await ask_traces()
    assert result["executed"] is False
    assert seen == []


def test_classification_is_exactly_one_action():
    items = generated_action_items(
        "usage_query", ["traces", "summary", "delete_traces"]
    )
    assert items["traces"]["mutates"] == "false"
    assert items["traces"]["eg_method"] is None
    assert "mutates" not in items["summary"]
    assert "mutates" not in items["delete_traces"]
    assert "mutates" not in generated_action_items("other_tool", ["traces"])["traces"]


@pytest.mark.asyncio
async def test_authenticated_tenant_read_does_not_require_graph_scopes(traces_surface):
    _, seen = traces_surface
    session = current_session()
    actor = replace(session.actor, roles=())
    with (
        use_actor(actor),
        use_session(replace(session, actor=actor, scopes=frozenset())),
    ):
        result = await ask_traces()
    assert result["executed"] is True
    assert seen == [{"tenant_id": actor.tenant_id, "origin": "runtime", "limit": 50}]


@pytest.mark.asyncio
async def test_verified_identity_middleware_preserves_authenticated_tenant_read(
    traces_surface, monkeypatch
):
    import time

    from agent_utilities.core.config import config
    from agent_utilities.mcp.middlewares import ActorContextMiddleware

    _, seen = traces_surface
    monkeypatch.setattr(config, "auth_jwt_audience", "trace-test-audience")
    monkeypatch.setattr(config, "kg_policy_version", "trace-test-policy")
    context = SimpleNamespace(
        auth=SimpleNamespace(
            claims={
                "sub": "trace-reader",
                "tenant_id": "trace-test-tenant",
                "exp": int(time.time()) + 300,
            }
        )
    )

    async def call_next(_context):
        session = current_session()
        assert session.actor.authenticated
        assert session.scopes == frozenset()
        assert not session.actor.roles
        return await ask_traces()

    result = await ActorContextMiddleware(require_verified_session=True).on_call_tool(
        context, call_next
    )
    assert result["executed"] is True
    assert seen == [
        {"tenant_id": "trace-test-tenant", "origin": "runtime", "limit": 50}
    ]


def test_missing_actor_and_session_denied_at_identity_middleware(
    traces_surface, monkeypatch
):
    import asyncio
    import contextvars

    from agent_utilities.knowledge_graph.core.session import SessionRequiredError
    from agent_utilities.mcp.middlewares import ActorContextMiddleware

    _, seen = traces_surface
    monkeypatch.setattr(kg_server, "_PROCESS_SESSION", None)

    async def call_next(_context):
        return await ask_traces()

    def isolated():
        assert current_session() is None
        with pytest.raises(SessionRequiredError):
            asyncio.run(
                ActorContextMiddleware(require_verified_session=True).on_call_tool(
                    SimpleNamespace(auth=None),
                    call_next,
                )
            )

    contextvars.Context().run(isolated)
    assert seen == []


@pytest.mark.asyncio
async def test_intent_and_usage_api_return_same_tenant_trace_view(
    traces_surface, monkeypatch
):
    from agent_utilities.gateway import usage_api

    _, seen = traces_surface
    result = await ask_traces()
    from agent_utilities.usage import service

    monkeypatch.setattr(usage_api, "get_usage_service", service.get_usage_service)
    expected = usage_api.traces(session_id=None, limit=50, tenant_id=None)
    assert json.loads(result["result"]) == expected
    assert seen[0] == seen[1]


def test_packaged_trace_descriptor_matches_reviewed_source():
    operation = next(
        d
        for d in intent_tools._load_cpds_required()["usage_query"]["does"]
        if d["action"] == "traces"
    )
    assert operation == generated_action_items("usage_query", ["traces"])["traces"]
