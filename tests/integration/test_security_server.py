import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from agent_utilities.server import build_agent_app
import agent_utilities.config as config

@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.toolsets = []
    agent.to_a2a.return_value = MagicMock()
    return agent

@pytest.fixture
def secure_client(mock_agent):
    # Set a dummy API key and enable auth
    original_key = config.config.agent_api_key
    original_enable = config.config.enable_api_auth
    config.config.agent_api_key = "secret-key"
    config.config.enable_api_auth = True

    try:
        with patch("agent_utilities.server.create_agent", return_value=(mock_agent, [])):
            app = build_agent_app(
                provider="test-provider",
                model_id="test-model",
                enable_web_ui=False,
                enable_acp=False,
                enable_otel=False
            )
            yield TestClient(app)
    finally:
        config.config.agent_api_key = original_key
        config.config.enable_api_auth = original_enable

def test_secure_endpoint_no_key(secure_client):
    response = secure_client.get("/health")
    assert response.status_code == 403
    assert "Could not validate credentials" in response.json()["detail"]

def test_secure_endpoint_wrong_key(secure_client):
    response = secure_client.get("/health", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 403

def test_secure_endpoint_correct_key(secure_client):
    response = secure_client.get("/health", headers={"X-API-Key": "secret-key"})
    assert response.status_code == 200

def test_max_upload_size_enforced():
    from agent_utilities.server import process_parts
    from agent_utilities.config import config as agent_config
    import asyncio

    original_size = agent_config.max_upload_size
    agent_config.max_upload_size = 100 # 100 bytes

    try:
        import base64
        # Large image
        large_data = base64.b64encode(b"a" * 200).decode()
        parts = [{"image": large_data, "media_type": "image/png"}]

        processed = asyncio.run(process_parts(parts))
        assert len(processed) == 0 # Should be rejected

        # Small image
        small_data = base64.b64encode(b"a" * 50).decode()
        parts = [{"image": small_data, "media_type": "image/png"}]
        processed = asyncio.run(process_parts(parts))
        assert len(processed) == 1 # Should be accepted
    finally:
        agent_config.max_upload_size = original_size
