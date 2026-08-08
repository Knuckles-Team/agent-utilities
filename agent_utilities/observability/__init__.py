"""Telemetry Engine — Synthesized Observability Facade.

CONCEPT:AU-OS.observability.telemetry-observability — Telemetry Engine

Provides a single entry point for all observability concerns:
- Token usage tracking (OS-5.5 via ``TokenTracker``)
- Audit logging (OS-5.6 via ``AuditLogger``)
- Deterministic replay (OS-5.6 via ``DistributedReplayEngine``)
- Real OpenTelemetry TracerProvider/MeterProvider export (OBS-P1-2 — see
  :meth:`TelemetryEngine._setup_otel`)

This facade wires the previously unwired AuditLogger and TokenTracker
into the main graph execution pipeline via ``on_graph_start()``,
``on_graph_end()``, and ``on_response()`` hooks.

OBS-P1-1 wired the self-ingest LOG pipeline (:mod:`.self_ingest`) into the
engine's own OTLP collector. OBS-P1-2 (this module) closes the remaining
gap the OS-5.8 comment used to flag: ``TelemetryEngine`` now configures a
REAL ``opentelemetry.sdk.trace.TracerProvider`` + ``opentelemetry.sdk.
metrics.MeterProvider``, each wired with a real OTLP/HTTP exporter (the same
``OTLPSpanExporter`` construction :func:`.custom_observability.
_create_otlp_span_processor` already uses for the Langfuse pipeline, plus
its metric counterpart) pointed at the ENGINE's own collector — reusing
OBS-P1-1's ``EPISTEMIC_GRAPH_OBS_ADDR`` endpoint config, falling back to the
generic ``OTEL_EXPORTER_OTLP_ENDPOINT``/``_HEADERS``/``_PROTOCOL`` settings
used elsewhere in this package. Opt-in and non-fatal: with no endpoint
configured (or the OTel SDK missing), setup is a clean no-op — but once
BOTH are present, ``on_graph_start``/``on_response``/``on_graph_end`` drive
REAL spans and metric instruments, never a placeholder/no-op facade.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from agent_utilities.core.config import setting
from agent_utilities.security.persistence_privacy import (
    PersistencePrivacyGuard,
    persistence_reference,
)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import SpanExporter

logger = logging.getLogger(__name__)

#: Env vars carrying the OTLP collector endpoint, checked in priority order.
#: ``EPISTEMIC_GRAPH_OBS_ADDR`` is OBS-P1-1's self-ingest endpoint config (the
#: engine's own collector — the primary target for this engine-native OTel
#: pipeline); ``OTEL_EXPORTER_OTLP_ENDPOINT`` is the generic OTel endpoint
#: :func:`.custom_observability.setup_otel` already uses for Langfuse, kept
#: as a fallback so a deployment with only the generic var set still works.
_OTEL_ENDPOINT_SETTINGS = ("EPISTEMIC_GRAPH_OBS_ADDR", "OTEL_EXPORTER_OTLP_ENDPOINT")
_STATUS_VALUES = frozenset(
    {
        "cancelled",
        "completed",
        "degraded",
        "error",
        "failed",
        "ok",
        "success",
        "unknown",
    }
)
#: The DERIVED epistemic-status vocabulary :func:`~agent_utilities.knowledge_graph.
#: core.epistemic_row.epistemic_status` returns — distinct from ``_STATUS_VALUES``
#: (run/graph-execution status). Sharing one validator/frozenset between the two
#: silently collapsed every real ``epistemic.status`` span attribute to
#: ``"unresolved"`` (the epistemic default), since none of "confirmed"/"contested"/
#: "low_confidence" are run-status values (X2 gap-fill).
_EPISTEMIC_STATUS_VALUES = frozenset(
    {
        "confirmed",
        "contested",
        "low_confidence",
        "unresolved",
    }
)
#: ``OTEL_TRACES_EXPORTER`` values that mean "export traces" — the OTel spec
#: default is ``"otlp"``; any other explicit value (most commonly ``"none"``)
#: disables trace export even when an endpoint resolves, matching the standard
#: SDK's own env-based auto-configuration.
_OTEL_TRACES_EXPORTERS_ENABLED = frozenset({"otlp"})


def _telemetry_ref(kind: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return persistence_reference(kind, text[:8192], namespace="telemetry")


def _status_label(value: Any) -> str:
    normalized = str(value or "unknown").strip().lower()
    return normalized if normalized in _STATUS_VALUES else "unknown"


def _epistemic_status_label(value: Any) -> str:
    normalized = str(value or "unresolved").strip().lower()
    return normalized if normalized in _EPISTEMIC_STATUS_VALUES else "unresolved"


def _traces_exporter_disabled() -> bool:
    """Whether the standard ``OTEL_TRACES_EXPORTER`` var explicitly disables export.

    Unset (the default) preserves current behavior — export is governed purely by
    endpoint presence. An explicit value that names no ``otlp`` exporter (e.g. the
    standard ``"none"``, or any other non-"otlp" exporter this engine does not
    implement) is a hard opt-out, even when a collector endpoint resolves.
    """

    raw = str(setting("OTEL_TRACES_EXPORTER", "") or "").strip().lower()
    if not raw:
        return False
    requested = {part.strip() for part in raw.split(",") if part.strip()}
    return not (requested & _OTEL_TRACES_EXPORTERS_ENABLED)


class _LoudFailureSpanExporter:
    """Wrap a span exporter so a broken trace pipeline is never silent.

    CONCEPT:AU-OS.observability.otlp-trace-fanout. Export MUST fail soft — an
    unreachable collector can never take down graph-os — but "soft" has been
    read as "silent" before, and a trace backend that quietly stopped receiving
    spans is indistinguishable from a system that is simply not busy. So the
    first failure (and every ~60 s thereafter) logs at ERROR with the endpoint,
    and the recovering export logs at INFO. The exporter itself never raises:
    an exception from the inner exporter is turned into a FAILURE result, which
    is exactly what ``BatchSpanProcessor`` expects.
    """

    _REPEAT_LOG_INTERVAL_S = 60.0

    def __init__(self, inner: Any, *, endpoint: str) -> None:
        self._inner = inner
        self._endpoint = endpoint
        self._failing = False
        self._last_log = 0.0

    def _note_failure(self, detail: str) -> None:
        import time

        now = time.monotonic()
        if not self._failing or (now - self._last_log) >= self._REPEAT_LOG_INTERVAL_S:
            logger.error(
                "OTLP span export to %s FAILED (%s). Traces are being dropped; "
                "graph-os itself is unaffected. Check the collector's "
                "reachability and TLS trust. "
                "(CONCEPT:AU-OS.observability.otlp-trace-fanout)",
                self._endpoint,
                detail,
            )
            self._last_log = now
        self._failing = True

    def _note_success(self) -> None:
        if self._failing:
            logger.info("OTLP span export to %s recovered.", self._endpoint)
            self._failing = False

    def export(self, spans: Any) -> Any:
        from opentelemetry.sdk.trace.export import SpanExportResult

        try:
            result = self._inner.export(spans)
        except Exception as exc:
            self._note_failure(f"exception_type={type(exc).__name__}: {exc}")
            return SpanExportResult.FAILURE
        if result is SpanExportResult.SUCCESS:
            self._note_success()
        else:
            self._note_failure(f"exporter returned {result!r}")
        return result

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        method = getattr(self._inner, "force_flush", None)
        return bool(method(timeout_millis)) if callable(method) else True


def _resolve_traces_endpoint(base: str) -> str:
    """Resolve the trace-signal endpoint, honouring the OTel-standard override.

    CONCEPT:AU-OS.observability.otlp-trace-fanout — the OpenTelemetry spec
    defines ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` as the signal-specific
    override of ``OTEL_EXPORTER_OTLP_ENDPOINT``, and (unlike the base var) it is
    a COMPLETE URL — the ``/v1/traces`` suffix is not appended. This engine used
    to ignore it and always derive ``{base}/v1/traces``, which made it
    impossible to send spans to a trace store (Tempo) while metrics/LLM
    telemetry went elsewhere. Honouring the standard var adds no invented knob.
    """
    override = str(setting("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "") or "").strip()
    return override or f"{base}/v1/traces"


def _resolve_metrics_endpoint() -> str:
    """Resolve the metrics-signal OTLP endpoint — explicit opt-in only (D-OG-3).

    CONCEPT:AU-OS.observability.otlp-metrics-exporter-gated. Unlike traces,
    there is no safe *default* derivation here: ``_setup_otel`` used to always
    build a ``MeterProvider`` pointed at ``{base}/v1/metrics`` even though
    neither of this deployment's actual OTLP destinations (Langfuse — traces
    only; Tempo — traces only) accepts OTLP metrics, so the
    ``PeriodicExportingMetricReader`` retried a structurally-404 endpoint
    forever, pure noise with no destination that could ever succeed.
    Prometheus's ``GET /metrics`` (:mod:`.gateway_metrics`) is this project's
    real metrics path, so the OTLP MeterProvider is built ONLY when an
    operator explicitly points it at a metrics-capable collector via the
    OTel-standard ``OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`` — a COMPLETE URL, no
    ``/v1/metrics`` suffix appended, mirroring :func:`_resolve_traces_endpoint`.
    Traces are unaffected either way.
    """
    return str(setting("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "") or "").strip()


def _resolve_otel_endpoint() -> str:
    """Resolve the canonical OTLP endpoint, preferring the engine collector.

    Purely standard-env-var driven: ``OTEL_TRACES_EXPORTER`` set to anything
    other than ``"otlp"`` (e.g. ``"none"``) is a hard kill-switch, checked
    before either endpoint setting resolves.
    """

    if _traces_exporter_disabled():
        return ""

    from agent_utilities.observability.custom_observability import (
        _resolve_otel_endpoint as resolve_runtime_otel_endpoint,
    )

    for key in _OTEL_ENDPOINT_SETTINGS:
        value = str(setting(key, "") or "").strip()
        if value:
            try:
                return resolve_runtime_otel_endpoint(value)
            except ValueError:
                return ""
    try:
        return resolve_runtime_otel_endpoint(None)
    except ValueError:
        return ""


class TelemetryEngine:
    """Synthesized observability engine.

    CONCEPT:AU-OS.observability.telemetry-observability — Telemetry Engine

    Usage::

        telemetry = TelemetryEngine()

        # At graph start
        telemetry.on_graph_start(run_id="run-1", agent_id="agent-1", query="...")

        # After each LLM response
        telemetry.on_response(run_id="run-1", usage={"prompt": 100, "response": 50})

        # At graph end
        telemetry.on_graph_end(run_id="run-1", status="success")

        # On process shutdown (flush the OTel exporters, if configured)
        telemetry.shutdown()
    """

    def __init__(
        self,
        enable_audit: bool = True,
        enable_tokens: bool = True,
        enable_otel: bool = True,
    ) -> None:
        self._audit_logger: Any = None
        self._token_tracker: Any = None
        self._enable_audit = enable_audit
        self._enable_tokens = enable_tokens
        self._enable_otel = enable_otel
        self._initialized = False

        # Real OTel state (populated by :meth:`_setup_otel` — ``None`` until
        # ``_lazy_init`` runs, and stays ``None`` forever if opted out or no
        # collector endpoint resolves; never a placeholder object).
        self._tracer_provider: Any = None
        self._meter_provider: Any = None
        self._tracer: Any = None
        self._meter: Any = None
        self._token_counter: Any = None
        self._graph_run_counter: Any = None
        self._otel_configured = False
        self._otel_transport_security: Any = None
        self._active_spans: dict[str, Any] = {}
        # ``opentelemetry.context.attach()``'s return ``Token`` for this run's
        # span (see :meth:`on_graph_start`) — a bare, finalizer-free object,
        # deliberately NOT ``start_as_current_span``'s generator-based context
        # manager (whose eventual garbage collection re-triggers its own
        # ``context.detach()`` if left un-exited — the exact D-CDX-21 hazard,
        # just deferred to an unpredictable later point instead of avoided).
        self._span_tokens: dict[str, Any] = {}
        # D-CDX-21: the OTel ``Context`` object that was ambient immediately
        # after ``on_graph_start``'s ``context.attach()`` call attached this
        # run's span. ``context.attach()``/``context.detach()`` must run in
        # the SAME Context (the same asyncio Task/thread's contextvars
        # lineage) — a multi-stage delegation run routinely crosses task
        # boundaries between ``on_graph_start`` and ``on_graph_end`` (e.g. MCP
        # child calls run in their own shielded task, ``mcp/child_resilience.
        # py::_call_once``), so equality cannot be assumed. See
        # :meth:`on_graph_end` for how this is used to avoid a cross-context
        # detach instead of merely swallowing the ``ValueError`` it raises.
        self._span_attach_context: dict[str, Any] = {}

    def _lazy_init(self) -> None:
        """Lazily initialize sub-engines to avoid import-time overhead."""
        if self._initialized:
            return
        self._initialized = True

        if self._enable_audit:
            try:
                from .audit_logger import AuditLogger

                self._audit_logger = AuditLogger()
            except Exception:
                logger.debug("AuditLogger not available, skipping audit logging")

        if self._enable_tokens:
            try:
                from .token_tracker import TokenUsageTracker

                self._token_tracker = TokenUsageTracker()
            except Exception:
                logger.debug("TokenTracker not available, skipping token tracking")

        if self._enable_otel:
            self._setup_otel()

    def _setup_otel(self) -> bool:
        """Configure a REAL OTel ``TracerProvider``/``MeterProvider`` exporting via OTLP.

        CONCEPT:AU-OS.observability.telemetry-observability — replaces the old
        OS-5.8 placeholder. Opt-in: returns ``False`` (clean no-op, no
        provider objects created) unless a collector endpoint resolves via
        :func:`_resolve_otel_endpoint` (OBS-P1-1's ``EPISTEMIC_GRAPH_OBS_ADDR``,
        falling back to the generic ``OTEL_EXPORTER_OTLP_ENDPOINT``) AND the
        ``opentelemetry`` SDK is importable. When both hold, this method
        builds real ``opentelemetry.sdk.trace.TracerProvider`` /
        ``opentelemetry.sdk.metrics.MeterProvider`` instances — each wired
        with a real OTLP/HTTP exporter — never a stub/no-op object.
        """
        endpoint = _resolve_otel_endpoint()
        if not endpoint:
            logger.debug(
                "TelemetryEngine: no OTLP collector endpoint configured "
                "(EPISTEMIC_GRAPH_OBS_ADDR / OTEL_EXPORTER_OTLP_ENDPOINT) — "
                "OTel export left disabled."
            )
            return False

        try:
            from opentelemetry import metrics as otel_metrics
            from opentelemetry import trace as otel_trace
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import (
                PeriodicExportingMetricReader,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:
            logger.warning(
                "TelemetryEngine: OpenTelemetry SDK unavailable — OTel export "
                "disabled (exception_type=%s)",
                type(exc).__name__,
            )
            return False

        from agent_utilities.base_utilities import retrieve_package_name
        from agent_utilities.observability.custom_observability import (
            _MetadataOnlySpanExporter,
            _resolve_otel_headers,
            _resolve_otel_transport,
            parse_otlp_headers,
        )

        try:
            raw_headers = str(setting("OTEL_EXPORTER_OTLP_HEADERS", "") or "")
            resolved_headers, _ = _resolve_otel_headers(
                endpoint=endpoint,
                headers=raw_headers or None,
                public_key=None,
                secret_key=None,
            )
            headers = parse_otlp_headers(resolved_headers)
        except Exception as exc:
            logger.warning(
                "TelemetryEngine: OTLP authentication invalid; export disabled "
                "(exception_type=%s)",
                type(exc).__name__,
            )
            return False
        service_name = str(
            setting("OTEL_SERVICE_NAME", "")
            or retrieve_package_name()
            or "agent-utilities"
        )
        service_ref = _telemetry_ref("service", service_name)
        base = endpoint.rstrip("/")
        for signal_path in ("/v1/traces", "/v1/metrics"):
            if base.endswith(signal_path):
                base = base.removesuffix(signal_path)
                break
        traces_endpoint = _resolve_traces_endpoint(base)
        # D-OG-3: no default derivation — see :func:`_resolve_metrics_endpoint`.
        # Empty string means "no metrics-capable collector configured", not
        # "use the base collector"; the MeterProvider is skipped entirely below.
        metrics_endpoint = _resolve_metrics_endpoint()

        # A signal-specific traces endpoint pointing somewhere OTHER than the
        # base collector must not carry the base collector's credentials — that
        # would hand (for example) Langfuse basic-auth to a trace store that
        # never asked for it. Re-resolve auth against the actual trace
        # destination; ``_resolve_otel_headers`` only auto-reuses Langfuse
        # credentials for a same-origin endpoint, so a different host correctly
        # gets none.
        trace_headers = headers
        if traces_endpoint != f"{base}/v1/traces":
            try:
                resolved_trace_headers, _ = _resolve_otel_headers(
                    endpoint=traces_endpoint,
                    headers=str(setting("OTEL_EXPORTER_OTLP_HEADERS", "") or "")
                    or None,
                    public_key=None,
                    secret_key=None,
                )
                trace_headers = parse_otlp_headers(resolved_trace_headers)
            except Exception as exc:
                logger.warning(
                    "TelemetryEngine: could not resolve auth for the "
                    "trace-signal endpoint %s (exception_type=%s: %s); "
                    "exporting spans without credentials.",
                    traces_endpoint,
                    type(exc).__name__,
                    exc,
                )
                trace_headers = {}

        # Same non-reuse-across-origins rule as traces above, applied to an
        # explicitly configured metrics endpoint (D-OG-3): a metrics collector
        # named via OTEL_EXPORTER_OTLP_METRICS_ENDPOINT never inherits the base
        # collector's credentials unless it resolves to the same origin.
        metric_headers = headers
        if metrics_endpoint and metrics_endpoint != f"{base}/v1/metrics":
            try:
                resolved_metric_headers, _ = _resolve_otel_headers(
                    endpoint=metrics_endpoint,
                    headers=str(setting("OTEL_EXPORTER_OTLP_HEADERS", "") or "")
                    or None,
                    public_key=None,
                    secret_key=None,
                )
                metric_headers = parse_otlp_headers(resolved_metric_headers)
            except Exception as exc:
                logger.warning(
                    "TelemetryEngine: could not resolve auth for the "
                    "metrics-signal endpoint %s (%s); "
                    "exporting metrics without credentials.",
                    metrics_endpoint,
                    exc,
                )
                metric_headers = {}

        trust = None
        try:
            from agent_utilities.core.http_client import create_requests_session

            trust = _resolve_otel_transport(endpoint)
            trace_session = create_requests_session(transport_security=trust)
            resource = Resource.create({"service.name": service_ref})

            tracer_provider = TracerProvider(resource=resource)
            tracer_provider.add_span_processor(
                BatchSpanProcessor(
                    cast(
                        "SpanExporter",
                        _LoudFailureSpanExporter(
                            _MetadataOnlySpanExporter(
                                OTLPSpanExporter(
                                    endpoint=traces_endpoint,
                                    headers=trace_headers,
                                    session=trace_session,
                                ),
                                service_ref=service_ref,
                            ),
                            endpoint=traces_endpoint,
                        ),
                    ),
                )
            )

            meter_provider: Any = None
            if metrics_endpoint:
                metric_session = create_requests_session(transport_security=trust)
                metric_reader = PeriodicExportingMetricReader(
                    OTLPMetricExporter(
                        endpoint=metrics_endpoint,
                        headers=metric_headers,
                        session=metric_session,
                    )
                )
                meter_provider = MeterProvider(
                    resource=resource, metric_readers=[metric_reader]
                )
            else:
                logger.info(
                    "TelemetryEngine: OTLP metrics export left disabled — no "
                    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT configured (D-OG-3). "
                    "Prometheus GET /metrics (gateway_metrics.py) is this "
                    "deployment's metrics path; traces still export normally."
                )
        except Exception as exc:  # noqa: BLE001 — OTel setup must never crash the caller
            if trust is not None:
                try:
                    trust.cleanup()
                except Exception:
                    pass
            logger.warning(
                "TelemetryEngine: OTel provider setup failed (exception_type=%s)",
                type(exc).__name__,
            )
            return False

        self._tracer_provider = tracer_provider
        self._meter_provider = meter_provider
        self._otel_transport_security = trust
        self._tracer = tracer_provider.get_tracer(service_ref)
        if meter_provider is not None:
            self._meter = meter_provider.get_meter(service_ref)
            self._token_counter = self._meter.create_counter(
                "agent_utilities.llm.tokens",
                unit="token",
                description="LLM tokens observed per TelemetryEngine.on_response call.",
            )
            self._graph_run_counter = self._meter.create_counter(
                "agent_utilities.graph.runs",
                unit="run",
                description="Graph executions observed per TelemetryEngine.on_graph_end call.",
            )

        # Register globally too (best-effort) so library instrumentation that
        # reads the ambient global provider (e.g. auto-instrumented HTTP
        # clients) picks this up — never load-bearing: this engine always
        # uses its OWN local provider/tracer/meter references above, so a
        # losing race against another global registrant (e.g. Logfire's own
        # ``configure()``) never breaks this engine's own export.
        try:
            otel_trace.set_tracer_provider(tracer_provider)
            if meter_provider is not None:
                otel_metrics.set_meter_provider(meter_provider)
        except Exception as exc:  # noqa: BLE001 — best-effort global registration
            logger.debug(
                "TelemetryEngine: global OTel provider registration skipped "
                "(exception_type=%s)",
                type(exc).__name__,
            )

        self._otel_configured = True
        logger.info(
            "TelemetryEngine: real OTel pipeline configured (service=%s)", service_ref
        )
        return True

    def on_graph_start(
        self,
        run_id: str,
        agent_id: str = "",
        query: str = "",
        execution_mode: str = "pending",
        **metadata: Any,
    ) -> None:
        """Record the start of an agent execution.

        The root span represents the public ``run_agent`` entrypoint.  It is
        deliberately not named ``graph.run`` because routing may select a
        direct single-server agent or another non-graph execution mode.
        """
        self._lazy_init()
        run_ref = _telemetry_ref("run", run_id)
        agent_ref = _telemetry_ref("agent", agent_id or "system")
        clean_metadata, _privacy = PersistencePrivacyGuard().sanitize(metadata)
        if not isinstance(clean_metadata, dict):
            clean_metadata = {}
        if self._audit_logger:
            self._audit_logger.log(
                actor=agent_ref,
                action="graph.start",
                resource_type="graph",
                resource_id=run_ref,
                details={"query_length": len(query), **clean_metadata},
            )
        if self._tracer is not None:
            try:
                # D-CDX-21: build the span via ``start_span`` + an explicit
                # ``context.attach`` we hold as a bare ``Token``, NOT via
                # ``start_as_current_span``'s generator-based context manager
                # (``span_cm.__enter__()``/``__exit__()`` manually split
                # across this method and ``on_graph_end``). A live, suspended
                # generator CM left un-exited (the natural result of
                # detecting a cross-context mismatch in ``on_graph_end`` and
                # skipping ``__exit__()``) still gets ``GeneratorExit``-closed
                # by the garbage collector at some LATER, unpredictable point
                # -- reproduced while building this fix: skipping the exit
                # call did not prevent the corrupting cross-context
                # ``context.detach()``, it only delayed it to GC time. A bare
                # ``contextvars.Token`` has no such finalizer, so simply
                # dropping the reference on a mismatch (see on_graph_end) is
                # inert and safe.
                span = self._tracer.start_span(
                    "agent.run",
                    attributes={
                        "run_ref": run_ref,
                        "agent_ref": agent_ref,
                        "query_length": len(query),
                        "agent_utilities.execution.mode": execution_mode,
                        # gen_ai semantic conventions (X2): this span covers one
                        # ``run_agent`` execution, mediated end-to-end by pydantic-ai —
                        # the invariant "which framework" fact, set once at span-open;
                        # per-call facts (model/tokens/tool-call count) land at
                        # :meth:`on_response`/:meth:`on_graph_end` as they become known.
                        "gen_ai.system": "pydantic_ai",
                    },
                )
                from opentelemetry import context as otel_context
                from opentelemetry import trace as otel_trace

                token = otel_context.attach(otel_trace.set_span_in_context(span))
                self._active_spans[run_id] = span
                self._span_tokens[run_id] = token
                # Snapshot the Context this attach landed in, so on_graph_end
                # can detect a cross-task/cross-context detach BEFORE
                # attempting it (see there).
                self._span_attach_context[run_id] = otel_context.get_current()
            except Exception as exc:  # noqa: BLE001 — tracing must never break the caller
                logger.debug(
                    "TelemetryEngine: span start failed (exception_type=%s)",
                    type(exc).__name__,
                )

    def on_response(
        self,
        run_id: str,
        usage: dict[str, int] | None = None,
        model: str = "",
        **metadata: Any,
    ) -> None:
        """Record token usage from an LLM response.

        In addition to the token-tracker record and the ``token_counter``
        metric (both keyed by opaque refs, unchanged), this stamps the
        gen_ai semantic-convention token/model attributes onto the run's own
        active span (opened by :meth:`on_graph_start`) when one is tracked —
        ``gen_ai.request.model`` (the plain model identifier — a name, not a
        secret) and ``gen_ai.usage.input_tokens``/``gen_ai.usage.
        output_tokens`` (counts). Best-effort: a run with no tracked span
        (OTel unconfigured, or called outside ``on_graph_start``/``on_graph_end``)
        skips span stamping and still records the metric/tracker as before.
        """
        self._lazy_init()
        span = self._active_spans.get(run_id)
        if span is not None and usage:
            try:
                if model:
                    span.set_attribute("gen_ai.request.model", model)
                input_tokens = usage.get("prompt", 0)
                if input_tokens:
                    span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
                output_tokens = usage.get("response", 0)
                if output_tokens:
                    span.set_attribute("gen_ai.usage.output_tokens", int(output_tokens))
            except Exception as exc:  # noqa: BLE001 — tracing must never break the caller
                logger.debug(
                    "TelemetryEngine: gen_ai span attribute set failed "
                    "(exception_type=%s)",
                    type(exc).__name__,
                )
        if self._token_tracker and usage:
            try:
                from .token_tracker import TokenUsageRecord

                record = TokenUsageRecord(
                    session_id=_telemetry_ref("run", run_id),
                    model_name=_telemetry_ref("model", model),
                    prompt_tokens=usage.get("prompt", 0),
                    response_tokens=usage.get("response", 0),
                    thoughts_tokens=usage.get("thoughts", 0),
                    tool_use_tokens=usage.get("tool_use", 0),
                )
                self._token_tracker.record(record)
            except Exception as exc:  # noqa: BLE001 — cost/usage telemetry mirror; a failed record only degrades the token-usage dashboard for this one call, it does not affect the actual LLM call/response already completed above
                logger.debug("Token recording failed: %s", exc)
        if self._token_counter is not None and usage:
            try:
                attrs = {
                    "run_ref": _telemetry_ref("run", run_id),
                    "model_ref": _telemetry_ref("model", model),
                }
                for kind in ("prompt", "response", "thoughts", "tool_use"):
                    count = usage.get(kind, 0)
                    if count:
                        self._token_counter.add(count, {**attrs, "kind": kind})
            except Exception as exc:  # noqa: BLE001 — metric export must never break the caller
                logger.debug(
                    "TelemetryEngine: token metric recording failed (exception_type=%s)",
                    type(exc).__name__,
                )

    def on_graph_end(
        self,
        run_id: str,
        status: str = "success",
        duration_ms: float = 0.0,
        *,
        model: str = "",
        tool_call_count: int | None = None,
        execution_mode: str = "",
        graph_execution_evidence: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> None:
        """Record the end of a graph execution.

        ``model`` and ``tool_call_count`` are optional gen_ai attrs stamped
        onto the run's span BEFORE it closes (an ended span rejects further
        attributes) — the caller passes whatever it has: a run that never
        resolved a model, or whose tool calls weren't tallied, simply omits
        them (X2 "tokens/tool-call count if available").  Validated graph
        execution evidence is projected onto the same root span; its checkpoint
        identifiers remain observational and explicitly carry
        ``resume_supported=false``.
        """
        self._lazy_init()
        run_ref = _telemetry_ref("run", run_id)
        status_label = _status_label(status)
        clean_metadata, _privacy = PersistencePrivacyGuard().sanitize(metadata)
        if not isinstance(clean_metadata, dict):
            clean_metadata = {}
        if self._audit_logger:
            self._audit_logger.log(
                actor="system",
                action="graph.end",
                resource_type="graph",
                resource_id=run_ref,
                details={
                    "status": status_label,
                    "duration_ms": duration_ms,
                    **clean_metadata,
                },
            )
        if self._graph_run_counter is not None:
            try:
                self._graph_run_counter.add(1, {"status": status_label})
            except Exception as exc:  # noqa: BLE001 — metric export must never break the caller
                logger.debug(
                    "TelemetryEngine: graph-run metric recording failed "
                    "(exception_type=%s)",
                    type(exc).__name__,
                )
        span = self._active_spans.pop(run_id, None)
        attach_token = self._span_tokens.pop(run_id, None)
        attach_ctx = self._span_attach_context.pop(run_id, None)
        if span is not None:
            try:
                span.set_attribute("status", status_label)
                span.set_attribute("duration_ms", duration_ms)
                if model:
                    span.set_attribute("gen_ai.request.model", model)
                if tool_call_count is not None:
                    span.set_attribute(
                        "gen_ai.response.tool_call_count", int(tool_call_count)
                    )
                if execution_mode:
                    span.set_attribute("agent_utilities.execution.mode", execution_mode)
                if graph_execution_evidence:
                    from agent_utilities.models import GraphExecutionEvidence

                    evidence = GraphExecutionEvidence.model_validate(
                        graph_execution_evidence
                    )
                    if evidence.topology_digest:
                        span.set_attribute(
                            "agent_utilities.graph.topology_digest",
                            evidence.topology_digest,
                        )
                    if evidence.version_digest:
                        span.set_attribute(
                            "agent_utilities.graph.version_digest",
                            evidence.version_digest,
                        )
                    if evidence.runtime_version:
                        span.set_attribute(
                            "agent_utilities.graph.runtime_version",
                            evidence.runtime_version,
                        )
                    if evidence.node_sequence:
                        span.set_attribute(
                            "agent_utilities.graph.node_sequence",
                            tuple(evidence.node_sequence),
                        )
                    span.set_attribute(
                        "agent_utilities.graph.transition_count",
                        len(evidence.transitions),
                    )
                    if evidence.checkpoint_ids:
                        span.set_attribute(
                            "agent_utilities.graph.checkpoint_ids",
                            tuple(evidence.checkpoint_ids),
                        )
                    span.set_attribute(
                        "agent_utilities.graph.resume_supported",
                        evidence.resume_supported,
                    )
                    for transition in evidence.transitions:
                        span.add_event(
                            "pydantic_graph.transition",
                            attributes={
                                "sequence": transition.sequence,
                                "node_ids": tuple(
                                    task.node_id for task in transition.scheduled_tasks
                                ),
                                "task_ids": tuple(
                                    task.task_id for task in transition.scheduled_tasks
                                ),
                            },
                        )
                    for checkpoint_id in evidence.checkpoint_ids:
                        span.add_event(
                            "pydantic_graph.checkpoint",
                            attributes={"checkpoint_id": checkpoint_id},
                        )
            except Exception as exc:  # noqa: BLE001 — tracing must never break the caller
                logger.debug(
                    "TelemetryEngine: span attribute set failed (exception_type=%s)",
                    type(exc).__name__,
                )
        if span is not None:
            try:
                # D-CDX-21: opentelemetry.context.attach()/detach() must run
                # in the SAME Context or detach() corrupts the ambient
                # "current span" state for whatever runs next in that
                # context, instead of restoring it — the "ServiceNow MCP
                # teardown detaches...token in the wrong context" defect.
                # OTel's own context.detach() already swallows the resulting
                # ValueError internally (logger.exception), so nothing
                # crashes; the damage — a corrupted span tree — is silent.
                # Verify same-context BEFORE calling detach(); on a mismatch
                # (this run's lifecycle crossed a task boundary between
                # on_graph_start's attach and here — e.g. an MCP child
                # call's own shielded task,
                # mcp/child_resilience.py::_call_once), skip the detach
                # entirely and just drop the ``Token`` — a bare Token has no
                # finalizer/GC side effect (unlike the generator-based
                # ``start_as_current_span`` context manager this replaced,
                # whose eventual garbage-collection re-triggered the exact
                # same cross-context detach at an unpredictable later point —
                # reproduced while building this fix). ``span.end()`` always
                # runs regardless, so the span itself still finalizes/
                # exports correctly either way.
                from opentelemetry import context as otel_context

                if attach_token is not None:
                    same_context = (
                        attach_ctx is None or otel_context.get_current() is attach_ctx
                    )
                    if same_context:
                        try:
                            otel_context.detach(attach_token)
                        except Exception as exc:  # noqa: BLE001 — the ambient-context restore is best-effort; the span itself still finalizes below regardless
                            logger.debug(
                                "TelemetryEngine: context detach failed for run "
                                "%r (exception_type=%s)",
                                run_id,
                                type(exc).__name__,
                            )
                    else:
                        logger.debug(
                            "TelemetryEngine: on_graph_end for run %r is running "
                            "in a different OTel Context than on_graph_start "
                            "attached (the run crossed a task/thread boundary) "
                            "-- skipping the ambient-context detach instead of "
                            "risking a cross-context Token.reset() (D-CDX-21).",
                            run_id,
                        )
            except Exception as exc:  # noqa: BLE001 — tracing must never break the caller
                logger.debug(
                    "TelemetryEngine: context handling failed for run %r "
                    "(exception_type=%s)",
                    run_id,
                    type(exc).__name__,
                )
            finally:
                try:
                    span.end()
                except Exception as exc:  # noqa: BLE001 — tracing must never break the caller
                    logger.debug(
                        "TelemetryEngine: span close failed (exception_type=%s)",
                        type(exc).__name__,
                    )

    def annotate_epistemic(
        self,
        *,
        confidence: float | None = None,
        status: str | None = None,
        contradiction_count: int | None = None,
        policy_labels: list[str] | tuple[str, ...] | None = None,
        source_count: int | None = None,
        model: str | None = None,
    ) -> None:
        """Stamp epistemic-vocabulary attributes onto the CURRENT active OTel
        span (CONCEPT:AU-KB-CURRENCY — OTel projection of the light epistemic
        layer, `04-five-intersections.md` item 4 "MISSING: no OTEL semantic-
        convention span attributes for epistemic decisions").

        This is the read/answer-path counterpart of :meth:`on_graph_start`/
        :meth:`on_response`: rather than opening a new span (a KG read
        already runs inside SOME span when tracing is on — the caller's
        ``@trace``d function, or a pydantic-ai tool-call span), this method
        just widens whichever span is currently recording with the
        ``epistemic.*`` vocabulary (confidence/status/contradiction_count/
        policy_labels — CONCEPT:EPI-P3-1) plus ``gen_ai.*`` where applicable
        (the model that produced/consumed the read, source count as a rough
        analogue of ``gen_ai.response.*``).

        ``status`` is the DERIVED epistemic-status vocabulary (``"confirmed"``/
        ``"contested"``/``"low_confidence"``/``"unresolved"`` — see
        :func:`~agent_utilities.knowledge_graph.core.epistemic_row.
        epistemic_status`), validated against ``_EPISTEMIC_STATUS_VALUES`` — a
        DIFFERENT vocabulary from :meth:`on_graph_end`'s run/execution status.
        ``policy_labels`` and ``model`` are stamped as their plain values
        (controlled-vocabulary tags and a model identifier respectively — names,
        not secrets or row content), following ``gen_ai.request.model`` semconv
        literally so a generic gen_ai dashboard/APM can key on it.

        Default-on wherever ANY OTel pipeline is already active — this
        engine's OWN provider (:meth:`_setup_otel`), the separate Logfire/
        ``custom_observability.setup_otel()`` pipeline this package also
        ships, or an externally-configured global provider — because it
        reads the AMBIENT current span via the OTel API rather than
        requiring `self`'s own provider to be the one that started it. A
        clean no-op otherwise: does nothing (no span created, no exporter
        touched) when the ``opentelemetry`` API is unavailable or the
        current span isn't recording (no pipeline configured anywhere) —
        never raises, never adds overhead to an untraced read.
        """
        try:
            from opentelemetry import trace as otel_trace

            span = otel_trace.get_current_span()
            if span is None or not span.is_recording():
                return
            if confidence is not None:
                span.set_attribute("epistemic.confidence", float(confidence))
            if status is not None:
                span.set_attribute("epistemic.status", _epistemic_status_label(status))
            if contradiction_count is not None:
                span.set_attribute(
                    "epistemic.contradiction_count", int(contradiction_count)
                )
            if policy_labels is not None:
                span.set_attribute(
                    "epistemic.policy_labels",
                    [str(label) for label in policy_labels[:32]],
                )
            if source_count is not None:
                span.set_attribute("gen_ai.response.source_count", int(source_count))
            if model:
                span.set_attribute("gen_ai.request.model", str(model))
        except Exception as exc:  # noqa: BLE001 — tracing must never break a read
            logger.debug(
                "TelemetryEngine: epistemic span annotation failed (exception_type=%s)",
                type(exc).__name__,
            )

    def annotate_context_compiler(
        self,
        *,
        items_selected: int | None = None,
        tokens_in: int | None = None,
        tokens_selected: int | None = None,
        token_budget: int | None = None,
        dropped_policy: int | None = None,
        dropped_redundant: int | None = None,
        dropped_budget: int | None = None,
        kv_cache_hit: bool | None = None,
    ) -> None:
        """Stamp ``ContextCompiler.compile()`` efficiency onto the CURRENT OTel span.

        CONCEPT:AU-KG.retrieval.context-compiler / CONCEPT:AU-KG.retrieval.context-compiler-kv-seam
        (WS-4) — the answer-path counterpart of :meth:`annotate_epistemic`: same
        "widen the ambient current span, never open one of our own" shape, same
        default-on-wherever-tracing-is-on / clean-no-op-otherwise posture. This is
        the OTEL-span half of the WS-4 instrumentation; the Prometheus counters/
        histograms (``observability.gateway_metrics.CONTEXT_COMPILER_*``) are the
        other, so a single compile() call is visible in both a trace waterfall
        (this) and a dashboard (those) without maintaining two separate stats.

        Args:
            items_selected: Final ``len(bundle.items)``.
            tokens_in: Tokens in the MMR-selected pool offered to the token-budget
                fit (before truncation).
            tokens_selected: ``bundle.tokens_used`` — tokens actually kept.
            token_budget: The caller's token budget for this call.
            dropped_policy: ``bundle.dropped_policy``.
            dropped_redundant: ``bundle.dropped_redundant``.
            dropped_budget: ``bundle.dropped_budget``.
            kv_cache_hit: ``bundle.kv_cache_hit`` when ``compile(kv_backend=...)``
                was used, ``None`` when the Seam-6 cache wasn't in play.
        """
        try:
            from opentelemetry import trace as otel_trace

            span = otel_trace.get_current_span()
            if span is None or not span.is_recording():
                return
            if items_selected is not None:
                span.set_attribute(
                    "context_compiler.items_selected", int(items_selected)
                )
            if tokens_in is not None:
                span.set_attribute("context_compiler.tokens_in", int(tokens_in))
            if tokens_selected is not None:
                span.set_attribute(
                    "context_compiler.tokens_selected", int(tokens_selected)
                )
            if token_budget is not None:
                span.set_attribute("context_compiler.token_budget", int(token_budget))
            if dropped_policy is not None:
                span.set_attribute(
                    "context_compiler.dropped_policy", int(dropped_policy)
                )
            if dropped_redundant is not None:
                span.set_attribute(
                    "context_compiler.dropped_redundant", int(dropped_redundant)
                )
            if dropped_budget is not None:
                span.set_attribute(
                    "context_compiler.dropped_budget", int(dropped_budget)
                )
            if kv_cache_hit is not None:
                span.set_attribute("context_compiler.kv_cache_hit", bool(kv_cache_hit))
        except Exception as e:  # noqa: BLE001 — tracing must never break a compile
            logger.debug(
                "TelemetryEngine: context-compiler span annotation failed: %s", e
            )

    def annotate_grounding(self, *, status: str, reason: str = "") -> None:
        """Stamp the mandatory evidence-compilation grounding outcome onto the
        CURRENT OTel span.

        CONCEPT:AU-KG.retrieval.fail-closed-grounding-contract — the answer-path
        counterpart of :meth:`annotate_epistemic`/:meth:`annotate_context_compiler`:
        same "widen the ambient current span, never open one of our own" shape,
        same default-on-wherever-tracing-is-on / clean-no-op-otherwise posture.
        Called from ``core.contextual_model``'s model-transport wrapper on every
        model call so a compile timeout, a compile error, an open circuit breaker,
        or a retrieval-quality-gate failure is queryable on the run's trace waterfall
        after the fact — not just visible as a transient log line.

        Args:
            status: ``"compiled"`` (genuine governed evidence reached the model),
                ``"bound_tool"`` (trusted bound-tool-result grounding, no retrieval),
                or ``"degraded"``/``"none"`` (no usable evidence; the caller's
                grounding policy allowed the request to proceed anyway).
            reason: Present only for a degraded/none status — e.g. ``"timeout"``,
                ``"error:<ExceptionType>"``, ``"circuit_breaker_open"``, or
                ``"quality_gate:<failure_mode>"``.
        """
        try:
            from opentelemetry import trace as otel_trace

            span = otel_trace.get_current_span()
            if span is None or not span.is_recording():
                return
            span.set_attribute("grounding.status", str(status))
            if reason:
                span.set_attribute("grounding.reason", str(reason))
        except Exception as e:  # noqa: BLE001 — tracing must never break a model call
            logger.debug("TelemetryEngine: grounding span annotation failed: %s", e)

    def is_otel_configured(self) -> bool:
        """Whether :meth:`_setup_otel` configured a REAL TracerProvider/MeterProvider.

        Triggers lazy init first, so this reflects the effective state even
        before any ``on_graph_*``/``on_response`` call.
        """
        self._lazy_init()
        return self._otel_configured

    @property
    def tracer_provider(self) -> Any:
        """The REAL ``opentelemetry.sdk.trace.TracerProvider`` :meth:`_setup_otel`
        built (wired with a ``BatchSpanProcessor`` + ``OTLPSpanExporter`` pointed at
        the live collector), or ``None`` when no endpoint is configured.

        Triggers lazy init first, mirroring :meth:`is_otel_configured`. This is the
        seam external instrumentation (e.g. pydantic-ai's
        ``pydantic_ai.capabilities.Instrumentation``, whose own
        ``InstrumentationSettings.tracer_provider`` otherwise defaults to the
        ambient global provider — typically configured via ``logfire.configure()``,
        which this codebase does not run) uses to land its spans on THIS engine's
        pipeline instead of a second, unconfigured one. See
        ``agent_utilities.capabilities.telemetry_instrumentation``.
        """
        self._lazy_init()
        return self._tracer_provider

    @property
    def meter_provider(self) -> Any:
        """The REAL ``opentelemetry.sdk.metrics.MeterProvider`` :meth:`_setup_otel`
        built, or ``None`` when no metrics-capable collector is configured. See
        :attr:`tracer_provider`.
        """
        self._lazy_init()
        return self._meter_provider

    def shutdown(self) -> None:
        """Flush and shut down the OTel providers, if configured. Never raises."""
        for provider in (self._tracer_provider, self._meter_provider):
            if provider is None:
                continue
            try:
                provider.shutdown()
            except Exception as exc:  # noqa: BLE001 — shutdown must never raise
                logger.debug(
                    "TelemetryEngine: OTel provider shutdown failed (exception_type=%s)",
                    type(exc).__name__,
                )
        if self._otel_transport_security is not None:
            try:
                self._otel_transport_security.cleanup()
            except Exception as exc:  # noqa: BLE001 - cleanup must never raise
                logger.debug(
                    "TelemetryEngine: OTel transport cleanup failed (exception_type=%s)",
                    type(exc).__name__,
                )
            self._otel_transport_security = None

    def get_token_summary(self, run_id: str | None = None) -> dict[str, Any]:
        """Get token usage summary, optionally filtered by run_id."""
        self._lazy_init()
        if self._token_tracker:
            if run_id:
                return self._token_tracker.get_session_totals(
                    _telemetry_ref("run", run_id)
                ).model_dump()
            return self._token_tracker.export_summary()
        return {}

    def get_audit_trail(
        self, limit: int = 100, action_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """Get recent audit entries."""
        self._lazy_init()
        if self._audit_logger:
            records = self._audit_logger.query(action=action_filter, limit=limit)
            return [r.model_dump() for r in records]
        return []


_TELEMETRY_ENGINE: TelemetryEngine | None = None


def get_telemetry_engine() -> TelemetryEngine:
    """Process-wide :class:`TelemetryEngine` singleton (CONCEPT:AU-OS.observability.telemetry-observability).

    Built once, lazily; :meth:`TelemetryEngine._lazy_init` (triggered by the
    first ``on_*``/``annotate_epistemic``/``is_otel_configured`` call) still
    gates the actual OTel provider setup, so constructing this singleton
    early (e.g. at import time of a caller) costs nothing until it is first
    used — mirrors :func:`.langfuse_exporter.get_langfuse_exporter`'s
    process-wide-singleton convention.
    """
    global _TELEMETRY_ENGINE
    if _TELEMETRY_ENGINE is None:
        _TELEMETRY_ENGINE = TelemetryEngine()
    return _TELEMETRY_ENGINE


# Replay Engine (OS-5.6) — Deterministic execution trace recording & replay
# HITL Escalation Matrix (OS-5.12) — formal value/risk → approver policy
from .escalation_matrix import (  # noqa: E402
    EscalationDecision,
    EscalationGate,
    EscalationMatrix,
    EscalationOutcome,
    EscalationRule,
    Fallback,
    RiskTier,
    ValueTier,
    classify_risk_tier,
    classify_value_tier,
    make_decision_provider,
)

# Langfuse exporter (ECO-4.24) — optional auto span/token/trace export
from .langfuse_exporter import (  # noqa: E402
    LangfuseExporter,
    get_langfuse_exporter,
)
from .replay_engine import (  # noqa: E402
    DistributedReplayEngine,
    InteractionRecord,
    ReplayManifest,
)

# Self-ingest telemetry (KG-2.304) — ship our OWN logs/RunTrace/ToolCall into the
# epistemic-graph engine obs store (dogfooding). Opt-in, default-off.
from .self_ingest import (  # noqa: E402
    SelfIngestConfig,
    SelfIngestLogHandler,
    SelfIngestSink,
    SpillBuffer,
    emit_run_trace,
    emit_tool_call,
    get_self_ingest_sink,
    install_self_ingest_logging,
    reset_self_ingest_sink,
    set_self_ingest_sink,
)

__all__ = [
    "TelemetryEngine",
    "get_telemetry_engine",
    # Replay Engine (OS-5.6)
    "DistributedReplayEngine",
    "ReplayManifest",
    "InteractionRecord",
    # HITL Escalation Matrix (OS-5.12)
    "EscalationMatrix",
    "EscalationGate",
    "EscalationRule",
    "EscalationDecision",
    "EscalationOutcome",
    "RiskTier",
    "ValueTier",
    "Fallback",
    "classify_risk_tier",
    "classify_value_tier",
    "make_decision_provider",
    # Langfuse exporter (ECO-4.24)
    "LangfuseExporter",
    "get_langfuse_exporter",
    # Self-ingest telemetry (KG-2.304)
    "SelfIngestSink",
    "SelfIngestConfig",
    "SelfIngestLogHandler",
    "SpillBuffer",
    "get_self_ingest_sink",
    "set_self_ingest_sink",
    "reset_self_ingest_sink",
    "install_self_ingest_logging",
    "emit_run_trace",
    "emit_tool_call",
]
