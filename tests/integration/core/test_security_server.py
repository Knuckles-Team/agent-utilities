"""CONCEPT:AU-OS.config.secrets-authentication"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import agent_utilities.core.config as config
from agent_utilities.server import build_agent_app


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.toolsets = []
    agent.to_a2a.return_value = MagicMock()
    return agent


@pytest.fixture
def secure_client(mock_agent):
    from agent_utilities.knowledge_graph.core.session import GraphSession
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext

    original_jwks = config.config.auth_jwt_jwks_uri
    original_issuer = config.config.auth_jwt_issuer
    original_audience = config.config.auth_jwt_audience
    config.config.auth_jwt_jwks_uri = "https://identity.example.test/jwks"
    config.config.auth_jwt_issuer = "https://identity.example.test"
    config.config.auth_jwt_audience = "agent-services"

    async def authenticate(*, authorization):
        if authorization != [b"Bearer valid-token"]:
            raise PermissionError("authentication required")
        return {
            "auth_type": "jwt",
            "sub": "subject",
            "tenant_id": "tenant",
            "scope": "kg:read",
        }

    actor = ActorContext(
        actor_id="subject",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("kg:read",),
        tenant_id="tenant",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant="tenant",
        scopes=frozenset({"kg:read"}),
        policy_version="test-policy",
        audience="agent-services",
    )

    async def actor_from_bearer_token(token):
        if token != "valid-token":
            raise PermissionError("invalid token")
        return actor

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
            )
            yield TestClient(app)
    finally:
        config.config.auth_jwt_jwks_uri = original_jwks
        config.config.auth_jwt_issuer = original_issuer
        config.config.auth_jwt_audience = original_audience


def test_health_probe_no_bearer(secure_client):
    """``/health`` stays unauthenticated liveness even with auth enforced
    elsewhere — always 200, body is the real shared health report.
    """
    response = secure_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("healthy", "unhealthy")
    assert isinstance(body["checks"], list)


def test_secure_endpoint_no_bearer(secure_client):
    response = secure_client.get("/models")
    assert response.status_code == 401
    assert response.json() == {"error": "authentication required"}


def test_secure_endpoint_wrong_bearer(secure_client):
    response = secure_client.get(
        "/models", headers={"Authorization": "Bearer wrong-token"}
    )
    assert response.status_code == 401


def test_secure_endpoint_correct_bearer(secure_client):
    response = secure_client.get(
        "/models", headers={"Authorization": "Bearer valid-token"}
    )
    assert response.status_code == 200


def test_max_upload_size_enforced():
    import asyncio

    from agent_utilities.core.config import config as agent_config
    from agent_utilities.server.dependencies import process_parts

    original_size = agent_config.max_upload_size
    agent_config.max_upload_size = 100  # 100 bytes

    try:
        import base64

        # Large image
        large_data = base64.b64encode(b"a" * 200).decode()
        parts = [{"image": large_data, "media_type": "image/png"}]

        with pytest.raises(HTTPException) as error:
            asyncio.run(process_parts(parts))
        assert error.value.status_code == 413

        # Small image
        small_data = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"a" * 42).decode()
        parts = [{"image": small_data, "media_type": "image/png"}]
        processed = asyncio.run(process_parts(parts))
        assert len(processed) == 1  # Should be accepted
    finally:
        agent_config.max_upload_size = original_size
