"""Exception details must not cross GraphOS REST or bootstrap boundaries."""

from __future__ import annotations

import json

import pytest

from agent_utilities.mcp import kg_server


@pytest.mark.asyncio
async def test_rest_handler_returns_only_stable_error_and_correlation_id(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sensitive = "https://identity:credential@private.invalid/local/path"

    async def _fail(_tool_name: str, **_kwargs: object) -> object:
        raise RuntimeError(sensitive)

    monkeypatch.setattr(kg_server, "_execute_tool", _fail)
    response = await kg_server.graph_query_endpoint(
        kg_server._build_dummy_request(json_body={})
    )
    payload = json.loads(response.body)

    assert response.status_code == 500
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "operation_failed"
    assert payload["error"]["correlation_id"].startswith("correlation:")
    assert sensitive not in response.body.decode()
    assert sensitive not in caplog.text
    assert "RuntimeError" in caplog.text


def test_invalid_request_payload_is_generic_and_never_echoes_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive = "query=secret&endpoint=https://private.invalid"

    payload = kg_server._external_failure_payload(
        ValueError(sensitive), code="invalid_request"
    )

    assert payload["error"]["code"] == "invalid_request"
    assert sensitive not in json.dumps(payload)
    assert sensitive not in caplog.text
    assert "ValueError" in caplog.text
