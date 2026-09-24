"""EH-380: a REST tool twin never restates a typed failed operation as success.

Tools return ``public_error_json`` (an ``OperationResult`` with
``status: "failed"``) instead of raising. The REST wrappers used to answer
``{"status": "success"}`` with HTTP 200, so an engine ``ACCESS_DENIED`` write
looked like a successful one.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.mcp.kg_server import _tool_success_response
from agent_utilities.security.error_surface import public_error_payload


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("operation_failed", 500),
        ("permission_denied", 403),
        ("invalid_request", 400),
        ("dependency_unavailable", 503),
    ],
)
def test_typed_failure_maps_to_failed_status(code, status) -> None:
    payload = public_error_payload(RuntimeError("ACCESS_DENIED: nope"), code=code)
    response = _tool_success_response(payload)
    body = json.loads(response.body)
    assert response.status_code == status
    assert body["status"] == "failed"
    assert body["result"]["error"]["code"] == code


@pytest.mark.parametrize(
    "result",
    [
        {"status": "failed"},  # not an OperationResult: no operation_id/error
        {"status": "ok", "nodes": 3},
        "Node n1 added.",
        [1, 2],
    ],
)
def test_ordinary_results_stay_success(result) -> None:
    response = _tool_success_response(result)
    assert response.status_code == 200
    assert json.loads(response.body) == {"status": "success", "result": result}


def test_unknown_failure_code_maps_to_500_through_the_shared_helper() -> None:
    """graph-os serves the same REST twins, so the mapping is one public helper."""
    from agent_utilities.security.error_surface import failed_operation_http_status

    payload = public_error_payload(RuntimeError("boom"))
    payload["error"]["code"] = "not_a_public_code"
    assert failed_operation_http_status(payload) == 500
    assert failed_operation_http_status({"status": "failed"}) is None
