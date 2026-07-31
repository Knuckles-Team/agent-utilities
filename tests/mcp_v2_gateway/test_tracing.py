"""OTel span + structured logging behavior for the isolated MCP v2 gateway.

CONCEPT:AU-ECO.mcp.v2-gateway-otel-tracing
"""

from __future__ import annotations

import json
import logging

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from mcp_v2_gateway import tracing
from mcp_v2_gateway.gateway import (
    MCP_V2_PROTOCOL_VERSION,
    TASKS_EXTENSION,
    GatewayRequestContext,
    GraphOSClient,
    GraphOSV2Gateway,
)


class _StubGraphOS(GraphOSClient):
    def __init__(self) -> None:
        self.status = "queued"
        self.tools: list[dict[str, object]] = [
            {
                "name": "graph_jobs",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "job_id": {"type": "string"},
                    },
                },
            }
        ]

    async def list_tools(self, context: GatewayRequestContext) -> dict[str, object]:
        return {"tools": self.tools}

    async def call_tool(
        self, name: str, arguments: dict[str, object], context: GatewayRequestContext
    ) -> dict[str, object]:
        if arguments.get("action") == "dispatch":
            return {"job_id": "job:opaque"}
        if arguments.get("action") == "status":
            return {"status": self.status, "created_at": "2026-07-30T00:00:00Z"}
        return {"content": [{"type": "text", "text": "ok"}], "isError": False}


def _meta(*, tasks: bool = False, **extra: object) -> dict[str, object]:
    capabilities: dict[str, object] = {}
    if tasks:
        capabilities["extensions"] = {TASKS_EXTENSION: {}}
    return {
        "io.modelcontextprotocol/protocolVersion": MCP_V2_PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientCapabilities": capabilities,
        **extra,
    }


@pytest.fixture
def recorded_spans(monkeypatch: pytest.MonkeyPatch):
    """Fresh provider per test, wired to an in-memory exporter (no network)."""
    # tests/conftest.py globally sets OTEL_SDK_DISABLED=true so unrelated tests
    # never pay for real span creation; this suite is specifically testing span
    # creation, so it opts back in -- same pattern as
    # tests/unit/knowledge_graph/test_graph_compute_rpc_span.py.
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    tracing.reset_for_tests()
    tracing.configure_tracing(service_name="test-gateway")
    exporter = InMemorySpanExporter()
    tracing._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter
    tracing.reset_for_tests()


@pytest.fixture
def captured_logs(monkeypatch: pytest.MonkeyPatch):
    # A real (non-NoOp) span is required for trace_id/span_id to land in the
    # log fields -- see the OTEL_SDK_DISABLED note on `recorded_spans` above.
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    tracing.configure_logging(level="INFO")
    handler = _Capture()
    handler.setFormatter(tracing._JSONLogFormatter())
    logger = logging.getLogger("mcp_v2_gateway")
    logger.addHandler(handler)
    yield records
    logger.removeHandler(handler)


class TestConfigureTracing:
    def test_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        tracing.reset_for_tests()
        first = tracing.configure_tracing()
        second = tracing.configure_tracing()
        assert first is second
        tracing.reset_for_tests()

    def test_protocol_version_is_a_resource_attribute(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        tracing.reset_for_tests()
        tracing.configure_tracing()
        resource = tracing._tracer_provider.resource
        assert resource.attributes["mcp.protocol_version"] == MCP_V2_PROTOCOL_VERSION
        tracing.reset_for_tests()

    def test_no_export_processor_without_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        tracing.reset_for_tests()
        tracing.configure_tracing()
        processor = tracing._tracer_provider._active_span_processor
        assert processor._span_processors == ()
        tracing.reset_for_tests()

    def test_export_processor_added_when_endpoint_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
        monkeypatch.setenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.example:4318"
        )
        tracing.reset_for_tests()
        tracing.configure_tracing()
        processor = tracing._tracer_provider._active_span_processor
        assert len(processor._span_processors) == 1
        tracing.reset_for_tests()


class TestTracedDispatch:
    def test_span_carries_protocol_version_and_method(self, recorded_spans) -> None:
        with tracing.traced_dispatch(
            method="tools/list",
            protocol_version=MCP_V2_PROTOCOL_VERSION,
            task_id=None,
            traceparent=None,
            tracestate=None,
            baggage=None,
        ) as span:
            tracing.finish_dispatch(
                span,
                method="tools/list",
                protocol_version=MCP_V2_PROTOCOL_VERSION,
                task_id=None,
                response={"jsonrpc": "2.0", "id": "1", "result": {}},
            )
        (finished,) = recorded_spans.get_finished_spans()
        assert finished.name == "mcp.dispatch"
        assert finished.kind == SpanKind.SERVER
        assert finished.attributes["mcp.protocol_version"] == MCP_V2_PROTOCOL_VERSION
        assert finished.attributes["mcp.method"] == "tools/list"
        assert finished.status.status_code == StatusCode.OK
        assert "mcp.task_id" not in finished.attributes

    def test_task_id_attribute_set_when_present(self, recorded_spans) -> None:
        with tracing.traced_dispatch(
            method="tasks/get",
            protocol_version=MCP_V2_PROTOCOL_VERSION,
            task_id="job:opaque",
            traceparent=None,
            tracestate=None,
            baggage=None,
        ) as span:
            tracing.finish_dispatch(
                span,
                method="tasks/get",
                protocol_version=MCP_V2_PROTOCOL_VERSION,
                task_id="job:opaque",
                response={"jsonrpc": "2.0", "id": "1", "result": {}},
            )
        (finished,) = recorded_spans.get_finished_spans()
        assert finished.attributes["mcp.task_id"] == "job:opaque"

    def test_error_response_sets_error_status_and_code(self, recorded_spans) -> None:
        with tracing.traced_dispatch(
            method="tools/call",
            protocol_version=MCP_V2_PROTOCOL_VERSION,
            task_id=None,
            traceparent=None,
            tracestate=None,
            baggage=None,
        ) as span:
            tracing.finish_dispatch(
                span,
                method="tools/call",
                protocol_version=MCP_V2_PROTOCOL_VERSION,
                task_id=None,
                response={
                    "jsonrpc": "2.0",
                    "id": "1",
                    "error": {"code": -32001, "message": "Unauthorized"},
                },
            )
        (finished,) = recorded_spans.get_finished_spans()
        assert finished.status.status_code == StatusCode.ERROR
        assert finished.attributes["mcp.error_code"] == -32001

    def test_child_span_inherits_incoming_trace_id(self, recorded_spans) -> None:
        incoming_trace_id = "0123456789abcdef0123456789abcdef"
        traceparent = f"00-{incoming_trace_id}-0123456789abcdef-01"
        with tracing.traced_dispatch(
            method="tasks/get",
            protocol_version=MCP_V2_PROTOCOL_VERSION,
            task_id="job:opaque",
            traceparent=traceparent,
            tracestate="vendor=value",
            baggage="tenant=tenant-a",
        ) as span:
            tracing.finish_dispatch(
                span,
                method="tasks/get",
                protocol_version=MCP_V2_PROTOCOL_VERSION,
                task_id="job:opaque",
                response={"jsonrpc": "2.0", "id": "1", "result": {}},
            )
        (finished,) = recorded_spans.get_finished_spans()
        assert format(finished.context.trace_id, "032x") == incoming_trace_id
        assert finished.parent is not None
        assert format(finished.parent.span_id, "016x") == "0123456789abcdef"

    def test_no_traceparent_starts_a_fresh_root_span(self, recorded_spans) -> None:
        with tracing.traced_dispatch(
            method="tools/list",
            protocol_version=MCP_V2_PROTOCOL_VERSION,
            task_id=None,
            traceparent=None,
            tracestate=None,
            baggage=None,
        ) as span:
            tracing.finish_dispatch(
                span,
                method="tools/list",
                protocol_version=MCP_V2_PROTOCOL_VERSION,
                task_id=None,
                response={"jsonrpc": "2.0", "id": "1", "result": {}},
            )
        (finished,) = recorded_spans.get_finished_spans()
        assert finished.parent is None


@pytest.fixture
def allowlisted_spans(monkeypatch: pytest.MonkeyPatch):
    """A fresh provider routed through the real `_AllowlistSpanExporter`
    sanitization wrapper -- unlike `recorded_spans`, which attaches a raw
    `InMemorySpanExporter` directly (so other tests can see everything a span
    was actually given), these tests must exercise the sanitizer itself.
    """
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    tracing.reset_for_tests()
    tracing.configure_tracing(service_name="test-gateway")
    exporter = InMemorySpanExporter()
    sanitizing = tracing._AllowlistSpanExporter(exporter, service_name="test-gateway")
    tracing._tracer_provider.add_span_processor(SimpleSpanProcessor(sanitizing))
    yield exporter
    tracing.reset_for_tests()


class TestAllowlistSpanExporter:
    def test_disallowed_attributes_are_stripped_before_export(
        self, allowlisted_spans
    ) -> None:
        tracer = tracing._tracer_provider.get_tracer("test")
        with tracer.start_as_current_span("mcp.dispatch") as span:
            span.set_attribute("mcp.protocol_version", MCP_V2_PROTOCOL_VERSION)
            span.set_attribute("authorization", "Bearer super-secret-token")
            span.set_attribute("tool.arguments", "{'job_id': 'do-not-leak'}")
        (finished,) = allowlisted_spans.get_finished_spans()
        assert (
            finished.attributes.get("mcp.protocol_version") == MCP_V2_PROTOCOL_VERSION
        )
        assert "authorization" not in finished.attributes
        assert "tool.arguments" not in finished.attributes

    def test_unlisted_span_name_is_relabeled(self, allowlisted_spans) -> None:
        tracer = tracing._tracer_provider.get_tracer("test")
        with tracer.start_as_current_span("leaks-a-bearer-in-the-name"):
            pass
        (finished,) = allowlisted_spans.get_finished_spans()
        assert finished.name == "mcp.dispatch"

    def test_events_and_tracestate_are_stripped(self, allowlisted_spans) -> None:
        tracer = tracing._tracer_provider.get_tracer("test")
        with tracer.start_as_current_span("mcp.dispatch") as span:
            span.add_event("downstream failure: secret detail")
        (finished,) = allowlisted_spans.get_finished_spans()
        assert finished.events == ()
        assert len(finished.context.trace_state) == 0


class TestStructuredLogging:
    def test_finish_dispatch_emits_one_json_log_line(self, captured_logs) -> None:
        with tracing.traced_dispatch(
            method="tools/list",
            protocol_version=MCP_V2_PROTOCOL_VERSION,
            task_id=None,
            traceparent=None,
            tracestate=None,
            baggage=None,
        ) as span:
            tracing.finish_dispatch(
                span,
                method="tools/list",
                protocol_version=MCP_V2_PROTOCOL_VERSION,
                task_id=None,
                response={"jsonrpc": "2.0", "id": "1", "result": {}},
            )
        assert len(captured_logs) == 1
        payload = json.loads(tracing._JSONLogFormatter().format(captured_logs[0]))
        assert payload["mcp.method"] == "tools/list"
        assert payload["mcp.protocol_version"] == MCP_V2_PROTOCOL_VERSION
        assert payload["outcome"] == "ok"
        assert "trace_id" in payload and "span_id" in payload
        assert "duration_ms" in payload

    def test_log_line_never_contains_bearer_or_arguments(self, captured_logs) -> None:
        secret_bearer = "Bearer super-secret-token-do-not-leak"
        with tracing.traced_dispatch(
            method="tools/call",
            protocol_version=MCP_V2_PROTOCOL_VERSION,
            task_id=None,
            traceparent=None,
            tracestate=None,
            baggage=None,
        ) as span:
            tracing.finish_dispatch(
                span,
                method="tools/call",
                protocol_version=MCP_V2_PROTOCOL_VERSION,
                task_id=None,
                response={
                    "jsonrpc": "2.0",
                    "id": "1",
                    "error": {"code": -32001, "message": "Unauthorized"},
                },
            )
        rendered = tracing._JSONLogFormatter().format(captured_logs[0])
        assert secret_bearer not in rendered
        assert "arguments" not in rendered


class TestGatewayIntegration:
    """End-to-end: `GraphOSV2Gateway.dispatch()` actually produces a span+log."""

    @pytest.mark.asyncio
    async def test_dispatch_produces_exactly_one_span_per_call(
        self, recorded_spans, captured_logs
    ) -> None:
        gateway = GraphOSV2Gateway(_StubGraphOS())
        context = GatewayRequestContext(authorization="Bearer tenant-token")
        request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tools/list",
            "params": {"_meta": _meta()},
        }
        response = await gateway.dispatch(request, context=context)
        assert "result" in response
        (finished,) = recorded_spans.get_finished_spans()
        assert finished.attributes["mcp.method"] == "tools/list"
        assert finished.status.status_code == StatusCode.OK
        assert len(captured_logs) == 1

    @pytest.mark.asyncio
    async def test_unauthorized_call_records_error_span(
        self, recorded_spans, captured_logs
    ) -> None:
        gateway = GraphOSV2Gateway(_StubGraphOS())
        context = GatewayRequestContext(authorization="")
        request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tools/list",
            "params": {"_meta": _meta()},
        }
        response = await gateway.dispatch(request, context=context)
        assert response["error"]["code"] == -32001
        (finished,) = recorded_spans.get_finished_spans()
        assert finished.status.status_code == StatusCode.ERROR
        assert finished.attributes["mcp.error_code"] == -32001

    @pytest.mark.asyncio
    async def test_tasks_get_span_carries_task_id(
        self, recorded_spans, captured_logs
    ) -> None:
        gateway = GraphOSV2Gateway(_StubGraphOS())
        context = GatewayRequestContext(authorization="Bearer tenant-token")
        request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tasks/get",
            "params": {"taskId": "job:opaque", "_meta": _meta(tasks=True)},
        }
        response = await gateway.dispatch(request, context=context)
        assert "result" in response
        (finished,) = recorded_spans.get_finished_spans()
        assert finished.attributes["mcp.task_id"] == "job:opaque"
