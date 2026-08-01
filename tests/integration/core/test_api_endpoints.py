"""CONCEPT:AU-ECO.messaging.native-backend-abstraction"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.name = "TestAgent"
    # Mock to_a2a to return a FastAPI app
    from fastapi import FastAPI

    a2a_app = FastAPI()

    @a2a_app.get("/info")
    async def info():
        return {"name": "TestAgent"}

    agent.to_a2a.return_value = a2a_app
    return agent


@pytest.fixture
def client(mock_agent):
    """A TestClient carrying a verified test identity on every request.

    CONCEPT:AU-OS.identity.authenticated-identity-enforcement — ``build_agent_app``
    now mounts ``AuthenticationBoundaryMiddleware`` + ``ActorIdentityMiddleware``
    unconditionally: any non-exempt path (every route here except ``/health``)
    401s without a verified actor. Mirrors
    ``tests/integration/core/test_security_server.py``'s ``secure_client``
    fixture exactly: patch the whole token->claims->actor->session chain (both
    middlewares) so a fixed ``Bearer test-token`` on every request mints a
    verified identity, instead of weakening either guard.
    """
    import agent_utilities.core.config as config_module
    from agent_utilities.knowledge_graph.core.session import GraphSession
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext

    original_jwks = config_module.config.auth_jwt_jwks_uri
    original_issuer = config_module.config.auth_jwt_issuer
    original_audience = config_module.config.auth_jwt_audience
    config_module.config.auth_jwt_jwks_uri = "https://identity.example.test/jwks"
    config_module.config.auth_jwt_issuer = "https://identity.example.test"
    config_module.config.auth_jwt_audience = "agent-services"

    actor = ActorContext(
        actor_id="test-subject",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("kg:read", "kg:write"),
        tenant_id="test-tenant",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant="test-tenant",
        scopes=frozenset({"kg:read", "kg:write", "*"}),
        policy_version="test-policy",
        audience="agent-services",
    )

    async def authenticate(*, authorization):
        if authorization != [b"Bearer test-token"]:
            raise PermissionError("authentication required")
        return {"auth_type": "jwt", "sub": "test-subject", "tenant_id": "test-tenant"}

    async def actor_from_bearer_token(token):
        if token != "test-token":
            raise PermissionError("invalid token")
        return actor

    try:
        with (
            patch("agent_utilities.server.app.create_agent") as mock_create,
            patch("agent_utilities.core.workspace.initialize_workspace"),
            patch(
                "agent_utilities.server.app.load_identity",
                return_value={"name": "TestAgent"},
            ),
            patch("agent_utilities.server.app.get_skills_path", return_value=None),
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
            # ``build_agent_app`` no longer mounts ``agent.to_a2a()`` at
            # "/a2a" — it builds a full FastA2A protocol server via
            # ``agent_to_epistemic_a2a`` (needing a real broker/storage +
            # a running TaskManager lifespan + a process identity token,
            # none of which this test wants to stand up). Patch that
            # construction point directly so "/a2a" mounts the test's own
            # trivial stub app instead, matching this test's original intent.
            patch(
                "agent_utilities.protocols.a2a_epistemic.agent_to_epistemic_a2a",
                return_value=mock_agent.to_a2a.return_value,
            ),
        ):
            mock_create.return_value = (mock_agent, [])
            from agent_utilities.server import build_agent_app

            app = build_agent_app(
                provider="test-provider",
                model_id="test-model",
                enable_web_ui=False,
                enable_acp=False,
                enable_otel=False,
            )
            test_client = TestClient(app)
            test_client.headers["Authorization"] = "Bearer test-token"
            yield test_client
    finally:
        config_module.config.auth_jwt_jwks_uri = original_jwks
        config_module.config.auth_jwt_issuer = original_issuer
        config_module.config.auth_jwt_audience = original_audience


def test_health_endpoint(client):
    """LIVENESS: dependency-free, status-only, always HTTP 200 (CONCEPT:
    AU-OS.deployment.liveness-vs-readiness-split). ``server/routers/core.py``'s
    ``health_check`` docstring is explicit that this stays the unconditional
    ``{"status": "ok"}`` stub by design — the richer, component-level report
    lives behind the authenticated dashboard / ``graph_configure(action=
    "health")`` surfaces, and the bounded up/down signal is ``GET
    /health/ready`` (``{"status": "ready"|"not_ready"}``), not this route.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["cache-control"] == "no-store"


def test_a2a_info_endpoint(client):
    response = client.get("/a2a/info")
    assert response.status_code == 200
    assert response.json()["name"] == "TestAgent"


def test_mcp_config_endpoint(client, tmp_path):
    with patch("agent_utilities.core.workspace.get_workspace_path") as mock_ws_path:
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text('{"mcpServers": {"test": {}}}')
        mock_ws_path.return_value = config_file

        response = client.get("/mcp/config")
        assert response.status_code == 200
        assert "test" in response.json()["mcpServers"]


def test_list_chats_endpoint(client):
    with patch(
        "agent_utilities.server.routers.core.list_chats_from_disk",
        return_value=["chat1", "chat2"],
    ):
        response = client.get("/chats")
        assert response.status_code == 200
        assert len(response.json()) == 2
