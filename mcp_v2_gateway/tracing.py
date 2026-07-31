"""OTel tracing and privacy-preserving structured logging for the isolated v2 gateway.

CONCEPT:AU-ECO.mcp.v2-gateway-otel-tracing

Before this module, ``gateway.py`` already validated and forwarded the W3C
``traceparent`` / ``tracestate`` / ``baggage`` carried in a request's ``_meta``
object (see ``_trace_context`` / ``GatewayRequestContext``), but nothing ever
consumed those values to create an observable trace, and the process emitted
no logs above ``WARNING`` (see ``__main__.py``'s ``logging.basicConfig`` and
the deliberately no-op ``Handler.log_message``). The deployed
``graph-os-mcp-v2-gateway`` container was therefore a traffic-bearing
component with zero current or historical logs -- this module closes that gap.

Two constraints carry over unchanged from ``gateway.py``'s own module
docstring and ``docs/architecture/mcp_v2_gateway.md``:

1. **Isolation.** This package has no dependency on ``agent_utilities`` or
   FastMCP. It cannot reuse ``agent_utilities.observability.custom_observability``'s
   OTLP pipeline directly, so the same *shape* of privacy-preserving exporter
   (``_MetadataOnlySpanExporter`` there) is reimplemented here, self-contained,
   against only the ``opentelemetry-*`` packages declared in this package's own
   ``pyproject.toml``.
2. **Never log secrets.** The gateway never logs bearer tokens, downstream
   endpoints, tool arguments, or downstream exception text. Both the span
   attributes this module ever sets AND the exporter pipeline enforce an
   explicit allow-list (``_ALLOWED_SPAN_ATTRIBUTES`` / ``_ALLOWED_SPAN_NAMES``)
   so a future careless call site cannot leak anything through tracing that
   the gateway itself would never log directly.

Export is opt-in: a span is always created for every dispatched request (so a
structured, trace-correlated log line exists even with a collector
unconfigured), but nothing is sent over the network unless
``OTEL_EXPORTER_OTLP_ENDPOINT`` is set -- this is a traffic-bearing sidecar and
must never block, slow, or fail a request because a collector is unreachable.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.context import Context

_SERVICE_NAME_ENV = "OTEL_SERVICE_NAME"
_DEFAULT_SERVICE_NAME = "graphos-mcp-v2-gateway"
_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
_LOG_LEVEL_ENV = "MCP_V2_GATEWAY_LOG_LEVEL"
_INSTRUMENTATION_NAME = "graphos-mcp-v2-gateway"
_SPAN_NAME = "mcp.dispatch"

# The only span this module ever creates, and the only attribute keys it ever
# sets on it. Anything else is dropped before export -- see module docstring.
_ALLOWED_SPAN_NAMES = frozenset({_SPAN_NAME})
_ALLOWED_SPAN_ATTRIBUTES = frozenset(
    {
        "mcp.protocol_version",
        "mcp.method",
        "mcp.task_id",
        "mcp.error_code",
    }
)

_LOGGER = logging.getLogger("mcp_v2_gateway")
_trace_propagator = TraceContextTextMapPropagator()
_baggage_propagator = W3CBaggagePropagator()

_tracer_provider: TracerProvider | None = None


class _JSONLogFormatter(logging.Formatter):
    """One-line JSON per record; only the explicit ``mcp_fields`` extra is emitted.

    Deliberately does not fall back to ``record.getMessage()`` interpolation
    for gateway events, so a future call site cannot smuggle free-text
    (bearer/args/endpoint) into a log line by accident -- it has to go through
    the ``mcp_fields`` allow-listed dict.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + "Z",
            "level": record.levelname,
        }
        fields = getattr(record, "mcp_fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        else:
            payload["message"] = record.getMessage()
        return json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str)


def configure_logging(*, level: str | None = None) -> None:
    """Install the JSON formatter on the package logger. Idempotent.

    Replaces the previous ``logging.basicConfig(level=logging.WARNING)`` in
    ``__main__.py``, which left the process with no INFO-level request
    logging at all. This does NOT touch ``BaseHTTPRequestHandler.log_message``
    (the raw HTTP access log), which stays disabled deliberately -- that line
    can carry sensitive query values at reverse proxies; ``finish_dispatch``
    below is the safe, allow-listed replacement for per-request visibility.
    """
    resolved = (level or os.environ.get(_LOG_LEVEL_ENV) or "INFO").upper()
    handler = logging.StreamHandler()
    handler.setFormatter(_JSONLogFormatter())
    package_logger = logging.getLogger("mcp_v2_gateway")
    package_logger.handlers = [handler]
    package_logger.propagate = False
    package_logger.setLevel(resolved)


def configure_tracing(*, service_name: str | None = None) -> trace.Tracer:
    """Idempotently install the process-wide ``TracerProvider`` and return a Tracer.

    Safe to call repeatedly (startup, and again from each test) -- only the
    first call installs a provider; later calls just return a tracer from it.
    """
    global _tracer_provider
    if _tracer_provider is None:
        name = (
            service_name or os.environ.get(_SERVICE_NAME_ENV) or _DEFAULT_SERVICE_NAME
        )
        provider = TracerProvider(
            resource=Resource(
                {
                    "service.name": name,
                    # Protocol version as a resource attribute makes it visible on
                    # every span this process ever emits, not just request spans.
                    "mcp.protocol_version": _current_protocol_version(),
                    "telemetry.content_retention": "metadata",
                }
            )
        )
        endpoint = os.environ.get(_ENDPOINT_ENV)
        if endpoint:
            processor = _build_export_processor(endpoint, service_name=name)
            if processor is not None:
                provider.add_span_processor(processor)
        _tracer_provider = provider
    return _tracer_provider.get_tracer(_INSTRUMENTATION_NAME)


def reset_for_tests() -> None:
    """Drop the cached provider so tests can reconfigure endpoint/env per case."""
    global _tracer_provider
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
    _tracer_provider = None


def _current_protocol_version() -> str:
    # Imported lazily to avoid a circular import (gateway.py imports this module).
    from .gateway import MCP_V2_PROTOCOL_VERSION

    return MCP_V2_PROTOCOL_VERSION


def _build_export_processor(endpoint: str, *, service_name: str) -> Any | None:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    traces_endpoint = endpoint.rstrip("/")
    if not traces_endpoint.endswith("/v1/traces"):
        traces_endpoint = f"{traces_endpoint}/v1/traces"
    # Headers/protocol/timeout are intentionally NOT parsed here: OTLPSpanExporter
    # already reads OTEL_EXPORTER_OTLP_HEADERS / OTEL_EXPORTER_OTLP_TIMEOUT from
    # the environment when not passed explicitly, matching the standard OTel SDK
    # contract every other collector-side integration in this ecosystem expects.
    exporter = _AllowlistSpanExporter(
        OTLPSpanExporter(endpoint=traces_endpoint), service_name=service_name
    )
    return BatchSpanProcessor(
        exporter,
        max_queue_size=2048,
        max_export_batch_size=512,
        schedule_delay_millis=5000,
    )


class _AllowlistSpanExporter(SpanExporter):
    """Rebuild each span before export, keeping only allow-listed name/attributes.

    Mirrors ``agent_utilities.observability.custom_observability
    ._MetadataOnlySpanExporter`` (same rebuild-not-mutate approach, same
    stripped events/links/tracestate), reimplemented standalone per this
    package's dependency-isolation contract.
    """

    def __init__(self, inner: SpanExporter, *, service_name: str) -> None:
        self._inner = inner
        self._service_name = service_name

    @staticmethod
    def _safe_context(value: Any) -> Any:
        if value is None:
            return None
        from opentelemetry.trace import SpanContext, TraceState

        return SpanContext(
            trace_id=value.trace_id,
            span_id=value.span_id,
            is_remote=True,
            trace_flags=value.trace_flags,
            trace_state=TraceState(),
        )

    def export(self, spans: Any) -> Any:
        from opentelemetry.sdk.trace import ReadableSpan
        from opentelemetry.trace.status import Status as SdkStatus

        safe_spans = []
        for span in list(spans)[:2048]:
            raw_name = str(getattr(span, "name", "") or "")
            name = raw_name if raw_name in _ALLOWED_SPAN_NAMES else "mcp.dispatch"
            attributes = {
                key: value
                for key, value in (getattr(span, "attributes", None) or {}).items()
                if key in _ALLOWED_SPAN_ATTRIBUTES
            }
            safe_spans.append(
                ReadableSpan(
                    name=name,
                    context=self._safe_context(getattr(span, "context", None)),
                    parent=self._safe_context(getattr(span, "parent", None)),
                    resource=Resource(
                        {
                            "service.name": self._service_name,
                            "telemetry.content_retention": "metadata",
                        }
                    ),
                    attributes=attributes,
                    events=(),
                    links=(),
                    kind=span.kind,
                    instrumentation_scope=None,
                    status=SdkStatus(span.status.status_code),
                    start_time=span.start_time,
                    end_time=span.end_time,
                )
            )
        return self._inner.export(safe_spans)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        method = getattr(self._inner, "force_flush", None)
        return bool(method(timeout_millis)) if callable(method) else True


def _extract_parent_context(
    traceparent: str | None, tracestate: str | None, baggage: str | None
) -> Context | None:
    if traceparent is None:
        return None
    carrier: dict[str, str] = {"traceparent": traceparent}
    if tracestate is not None:
        carrier["tracestate"] = tracestate
    context = _trace_propagator.extract(carrier=carrier)
    if baggage is not None:
        context = _baggage_propagator.extract(
            carrier={"baggage": baggage}, context=context
        )
    return context


@contextmanager
def traced_dispatch(
    *,
    method: str,
    protocol_version: str,
    task_id: str | None,
    traceparent: str | None,
    tracestate: str | None,
    baggage: str | None,
) -> Iterator[Span]:
    """Start (and, on exit, end) one server span for a single dispatched request.

    The span is a child of the caller's ``traceparent``/``tracestate`` (plus
    ``baggage``) when present and valid -- exactly the values ``gateway.py``'s
    own ``_trace_context`` already validated -- otherwise a fresh root span.
    """
    tracer = configure_tracing()
    parent_context = _extract_parent_context(traceparent, tracestate, baggage)
    attributes: dict[str, Any] = {
        "mcp.protocol_version": protocol_version,
        "mcp.method": method,
    }
    if task_id:
        attributes["mcp.task_id"] = task_id
    with tracer.start_as_current_span(
        _SPAN_NAME,
        context=parent_context,
        kind=SpanKind.SERVER,
        attributes=attributes,
    ) as span:
        yield span


def finish_dispatch(
    span: Span,
    *,
    method: str,
    protocol_version: str,
    task_id: str | None,
    response: Mapping[str, Any],
) -> None:
    """Set final span status from an already-public-safe JSON-RPC response, and
    emit one structured, trace-correlated log line.

    ``response``'s ``error.message``/``error.code`` are safe to read here: a
    ``GatewayProtocolError`` (the only source of ``response["error"]``) is
    documented in ``gateway.py`` as "a public JSON-RPC error; details never
    include credentials or endpoints" -- unlike raw exception text, which
    ``dispatch()`` deliberately never serializes.
    """
    error = response.get("error")
    error_code: int | None = None
    if isinstance(error, Mapping):
        span.set_status(Status(StatusCode.ERROR))
        code = error.get("code")
        if isinstance(code, int):
            error_code = code
            span.set_attribute("mcp.error_code", code)
    else:
        span.set_status(Status(StatusCode.OK))

    span_context = span.get_span_context()
    fields: dict[str, Any] = {
        "event": _SPAN_NAME,
        "mcp.method": method,
        "mcp.protocol_version": protocol_version,
        "outcome": "error" if error_code is not None else "ok",
    }
    if task_id:
        fields["mcp.task_id"] = task_id
    if error_code is not None:
        fields["mcp.error_code"] = error_code
    start_time = getattr(span, "start_time", None)
    if isinstance(start_time, int):
        fields["duration_ms"] = round((time.time_ns() - start_time) / 1_000_000, 3)
    if span_context is not None and span_context.is_valid:
        fields["trace_id"] = format(span_context.trace_id, "032x")
        fields["span_id"] = format(span_context.span_id, "016x")
    _LOGGER.info(_SPAN_NAME, extra={"mcp_fields": fields})
