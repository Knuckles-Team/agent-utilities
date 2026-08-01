"""CONCEPT:AU-ECO.messaging.native-backend-abstraction"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agent_utilities.server import build_agent_app


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.toolsets = []
    agent.to_a2a.return_value = MagicMock()
    return agent


@pytest.fixture
def client(mock_agent):
    # This file exercises route/handler behavior (chat listing, MCP config,
    # approvals, codemap generation, streaming, ...), not the auth boundary
    # itself — that is test_security_server.py's job (it asserts the 401/403
    # shape directly). The server now requires a verified Bearer identity on
    # every non-health route by default (CONCEPT:AU-OS.config.secrets-authentication), so give every
    # request here one, the same way test_security_server.py's ``secure_client``
    # does: patch the auth seam to accept one fixed token and default it onto
    # every request via the TestClient's own default headers.
    import agent_utilities.core.config as _config_module
    from agent_utilities.knowledge_graph.core.session import GraphSession
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext

    # ActorIdentityMiddleware 401s ANY request carrying a bearer token when
    # JWT auth isn't configured at all ("Token validation unavailable") — even
    # for otherwise-unauthenticated paths like /health — so this must be set
    # for every request in this file to reach its handler, not just the ones
    # asserting a specific status code.
    original_jwks = _config_module.config.auth_jwt_jwks_uri
    original_issuer = _config_module.config.auth_jwt_issuer
    original_audience = _config_module.config.auth_jwt_audience
    _config_module.config.auth_jwt_jwks_uri = "https://identity.example.test/jwks"
    _config_module.config.auth_jwt_issuer = "https://identity.example.test"
    _config_module.config.auth_jwt_audience = "agent-services"

    actor = ActorContext(
        actor_id="test-subject",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id="test-tenant",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant="test-tenant",
        scopes=frozenset({"kg:read", "kg:write", "kg:admin", "*"}),
        policy_version="test-policy",
        audience="agent-services",
    )

    async def authenticate(*, authorization):
        if authorization != [b"Bearer test-token"]:
            raise PermissionError("authentication required")
        # "api_key" (not "jwt") short-circuits the fine-grained per-route
        # capability check in agent_ui.py's _require_agent_invoke (and
        # similar dependencies elsewhere) — this file's synthetic identity is
        # a full-access system test credential, not a narrowly-scoped user
        # JWT, so it should behave like one instead of also needing every
        # route's specific identity_group_capability_map entry configured.
        return {
            "auth_type": "api_key",
            "sub": "test-subject",
            "tenant_id": "test-tenant",
            "scope": "*",
        }

    async def actor_from_bearer_token(token):
        if token != "test-token":
            raise PermissionError("invalid token")
        return actor

    # Mocking create_agent to return our mock_agent. The auth patches must stay
    # active for the LIFETIME of the yielded client, not just app construction —
    # authentication runs per-request, at request time, well after this fixture
    # function would otherwise have returned — so this is a generator fixture
    # that yields from inside the ``with`` block (mirroring secure_client).
    try:
        with (
            patch(
                "agent_utilities.server.app.create_agent",
                return_value=(mock_agent, []),
            ),
            patch(
                "agent_utilities.security.auth.authenticate_header_values",
                new=authenticate,
            ),
            patch(
                "agent_utilities.security.request_identity.actor_from_claims",
                return_value=actor,
            ),
            patch(
                "agent_utilities.security.request_identity.actor_from_bearer_token",
                new=actor_from_bearer_token,
            ),
            patch(
                "agent_utilities.security.request_identity.mint_graph_session",
                return_value=session,
            ),
        ):
            app = build_agent_app(
                provider="test-provider",
                model_id="test-model",
                enable_web_ui=False,
                enable_acp=False,
                enable_otel=False,
                graph_bundle=("graph", "config"),
            )
            yield TestClient(app, headers={"Authorization": "Bearer test-token"})
    finally:
        _config_module.config.auth_jwt_jwks_uri = original_jwks
        _config_module.config.auth_jwt_issuer = original_issuer
        _config_module.config.auth_jwt_audience = original_audience


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["cache-control"] == "no-store"


def test_list_chats(client):
    with patch(
        "agent_utilities.server.routers.core.list_chats_from_disk",
        return_value=["chat1", "chat2"],
    ):
        response = client.get("/chats")
        assert response.status_code == 200
        assert response.json() == ["chat1", "chat2"]


def test_get_chat_success(client):
    with patch(
        "agent_utilities.server.routers.core.get_chat_from_disk",
        return_value={"id": "chat1", "messages": []},
    ):
        response = client.get("/chats/chat1")
        assert response.status_code == 200
        assert response.json()["id"] == "chat1"


def test_get_chat_not_found(client):
    with patch(
        "agent_utilities.server.routers.core.get_chat_from_disk", return_value=None
    ):
        response = client.get("/chats/missing")
        assert response.status_code == 404
        assert "error" in response.json()


def test_mcp_config(client):
    with patch("agent_utilities.core.workspace.get_workspace_path") as mock_path:
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.read_text.return_value = json.dumps(
            {"mcpServers": {"test": {}}}
        )

        response = client.get("/mcp/config")
        assert response.status_code == 200
        assert "mcpServers" in response.json()


def test_resolve_approval_missing_rid(client):
    response = client.post("/api/approve", json={})
    assert response.status_code == 400
    assert "request_id is required" in response.json()["error"]


def test_resolve_approval_success(client):
    with patch("agent_utilities.server.routers.human._approval_manager") as mock_mgr:
        mock_mgr.resolve.return_value = True
        response = client.post(
            "/api/approve", json={"request_id": "req1", "decisions": {}}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "resolved"


def test_resolve_approval_not_found(client):
    with patch("agent_utilities.server.routers.human._approval_manager") as mock_mgr:
        mock_mgr.resolve.return_value = False
        response = client.post(
            "/api/approve", json={"request_id": "missing", "decisions": {}}
        )
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_reload_mcp_config(client):
    # reload_mcp_config() imports get_discovery_registry lazily, function-
    # local (`from ...core.config import get_discovery_registry`), so it is
    # never bound as a module-level attribute of
    # agent_utilities.server.routers.interop — patching it there raises
    # AttributeError. Patch it at its actual definition site
    # (agent_utilities.core.config); the lazy import re-resolves the patched
    # name on every call.
    with (
        patch(
            "agent_utilities.mcp.agent_manager.sync_mcp_agents", new_callable=AsyncMock
        ),
        patch("agent_utilities.core.config.get_discovery_registry") as mock_reg,
    ):
        mock_reg.return_value.agents = [1, 2]
        mock_reg.return_value.tools = [1, 2, 3]

        response = client.post("/mcp/reload")
        print(f"DEBUG: response={response.json()}")
        assert response.status_code == 200
        assert response.json()["status"] == "reloaded"
        assert response.json()["agents"] == 2


def test_generate_codemap_not_initialized(client):
    with patch(
        "agent_utilities.knowledge_graph.core.engine.IntelligenceGraphEngine.get_active",
        return_value=None,
    ):
        response = client.post("/api/codemap", json={"prompt": "test"})
        assert response.status_code == 503
        assert "Knowledge Graph not initialized" in response.json()["message"]


@pytest.mark.asyncio
async def test_generate_codemap_success(client):
    mock_kg = MagicMock()
    mock_artifact = MagicMock()
    mock_artifact.id = "map1"
    mock_artifact.model_dump.return_value = {"id": "map1", "nodes": []}

    with (
        patch(
            "agent_utilities.knowledge_graph.core.engine.IntelligenceGraphEngine.get_active",
            return_value=mock_kg,
        ),
        patch(
            "agent_utilities.knowledge_graph.core.codemaps.CodemapGenerator"
        ) as mock_gen_class,
    ):
        mock_gen = mock_gen_class.return_value
        mock_gen.create = AsyncMock(return_value=mock_artifact)

        response = client.post(
            "/api/codemap", json={"prompt": "analyze this", "mode": "smart"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["codemap_id"] == "map1"
        assert "artifact" in data


@pytest.mark.asyncio
async def test_ag_ui_stream(client):
    with patch("pydantic_ai.ui.ag_ui.AGUIAdapter") as mock_adapter_class:
        mock_adapter = mock_adapter_class.return_value
        mock_adapter.dispatch_request = AsyncMock()

        # Mocking StreamingResponse
        from fastapi.responses import StreamingResponse

        async def mock_iterator():
            yield b"data: test\n\n"

        mock_adapter.dispatch_request.return_value = StreamingResponse(mock_iterator())

        response = client.post("/ag-ui", json={"query": "hello"})
        assert response.status_code == 200
        # Check if we can read the stream
        content = b"".join(response.iter_bytes())
        assert b"data: test" in content


def test_stream_endpoint(client, mock_agent):
    # Mocking stream_graph from its source module
    with patch(
        "agent_utilities.orchestration.engine.AgentOrchestrationEngine.stream_graph"
    ) as mock_run:

        async def mock_stream_gen(*args, **kwargs):
            yield "event: message\ndata: hello\n\n"

        mock_run.return_value = mock_stream_gen()

        response = client.post("/stream", json={"query": "test"})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        content = b"".join(response.iter_bytes())
        assert b"data: hello" in content


def test_list_mcp_tools(client, mock_agent):
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "test desc"

    mock_ts = MagicMock()
    mock_ts.get_tools.return_value = [mock_tool]
    mock_ts.name = "test_server"

    mock_agent.toolsets = [mock_ts]

    response = client.get("/mcp/tools")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "test_tool"
    assert data[0]["tag"] == "test_server"
