from types import SimpleNamespace

import pytest

from agent_utilities.core.config import AgentConfig
from agent_utilities.messaging.alert_intake import (
    _MAX_ALERT_CHARS,
    _extract_text,
    _handle,
    _loopback_bind,
)


def test_alert_intake_defaults_to_loopback_and_requires_secret_reference():
    cfg = AgentConfig()
    assert cfg.messaging_alert_intake_host == "127.0.0.1"
    assert cfg.messaging_alert_intake_allow_remote is False
    assert cfg.messaging_alert_intake_token_ref is None

    with pytest.raises(ValueError, match="runtime secret refs"):
        AgentConfig(MESSAGING_ALERT_INTAKE_TOKEN_REF="plaintext-token")

    configured = AgentConfig(
        MESSAGING_ALERT_INTAKE_PORT=9123,
        MESSAGING_ALERT_INTAKE_TOKEN_REF="env://ALERT_INTAKE_TOKEN",
    )
    assert configured.messaging_alert_intake_port == 9123


def test_alert_intake_bind_and_payload_bounds():
    assert _loopback_bind("127.0.0.1")
    assert _loopback_bind("::1")
    assert _loopback_bind("localhost")
    assert not _loopback_bind("0.0.0.0")
    assert not _loopback_bind("service.internal")

    assert len(_extract_text("x" * (_MAX_ALERT_CHARS + 100))) == _MAX_ALERT_CHARS
    assert _extract_text({"alerts": [None, {"labels": "invalid"}]})


@pytest.mark.asyncio
async def test_alert_intake_rejects_missing_or_wrong_bearer_before_processing():
    app = {"alert_intake_token": "expected"}
    missing = SimpleNamespace(headers={}, app=app)
    response = await _handle(missing)
    assert response.status == 401

    wrong = SimpleNamespace(headers={"Authorization": "Bearer wrong"}, app=app)
    response = await _handle(wrong)
    assert response.status == 401
