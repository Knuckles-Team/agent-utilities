"""GatewayMetricsMiddleware tolerates `gateway_health` being absent.

``observability/gateway_health.py`` moved to
``graph_os.observability.gateway_health`` (GRAPHOS-HOST-R008,
AU-BOUNDARY-R014); it is not installed here. The middleware's own
request-duration sample feed into it was already written as a function-local,
``try``/``except Exception``-wrapped, explicitly "best-effort" call (comment:
"must never affect the response path") — this proves that documented
fallback now permanently exercises, rather than the middleware (or the
response it serves) breaking.

Replaces the agent runtime's prior
``test_middleware_feeds_the_real_duration_into_record_request_duration``
(deleted with ``gateway_health.py``, which it imported at module level),
which proved the opposite direction: that the middleware called INTO the
(then-present) module. That assertion now lives in graph-os's own
``tests/observability/test_gateway_health_evidence_live_path.py``, ported
alongside the module.
"""

from __future__ import annotations

import logging

import pytest

from agent_utilities.observability import gateway_metrics as gm


@pytest.mark.spec("AU-BOUNDARY-R014")
def test_gateway_health_module_is_gone() -> None:
    """Pin the precondition this test depends on."""
    with pytest.raises(ModuleNotFoundError):
        import agent_utilities.observability.gateway_health  # noqa: F401


@pytest.mark.spec("AU-BOUNDARY-R014")
@pytest.mark.asyncio
async def test_middleware_serves_the_response_without_gateway_health(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw = gm.GatewayMetricsMiddleware(app)
    scope = {"type": "http", "method": "GET", "path": "/api/graph/query", "headers": []}

    async def receive():
        return {"type": "http.request"}

    sent: list[dict] = []

    async def send(msg):
        sent.append(msg)

    with caplog.at_level(
        logging.DEBUG, logger="agent_utilities.observability.gateway_metrics"
    ):
        await mw(scope, receive, send)

    # The response the ASGI app produced is untouched by the missing module.
    assert sent[0]["status"] == 200
    assert sent[1]["body"] == b"ok"

    # The documented fallback branch ran (not an unrelated swallowed error).
    assert any(
        "gateway health sample failed" in record.message for record in caplog.records
    )
