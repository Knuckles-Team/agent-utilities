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
from typing import Any

from agent_utilities.core.config import setting
from agent_utilities.security.persistence_privacy import (
    PersistencePrivacyGuard,
    persistence_reference,
)

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
        self._span_tokens: dict[str, Any] = {}

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
        traces_endpoint = f"{base}/v1/traces"
        metrics_endpoint = f"{base}/v1/metrics"

        trust = None
        try:
            from agent_utilities.core.http_client import create_requests_session

            trust = _resolve_otel_transport(endpoint)
            trace_session = create_requests_session(transport_security=trust)
            metric_session = create_requests_session(transport_security=trust)
            resource = Resource.create({"service.name": service_ref})

            tracer_provider = TracerProvider(resource=resource)
            tracer_provider.add_span_processor(
                BatchSpanProcessor(
                    _MetadataOnlySpanExporter(
                        OTLPSpanExporter(
                            endpoint=traces_endpoint,
                            headers=headers,
                            session=trace_session,
                        ),
                        service_ref=service_ref,
                    )
                )
            )

            metric_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=metrics_endpoint,
                    headers=headers,
                    session=metric_session,
                )
            )
            meter_provider = MeterProvider(
                resource=resource, metric_readers=[metric_reader]
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
        **metadata: Any,
    ) -> None:
        """Record the start of a graph execution."""
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
                span_cm = self._tracer.start_as_current_span(
                    "graph.run",
                    attributes={
                        "run_ref": run_ref,
                        "agent_ref": agent_ref,
                        "query_length": len(query),
                        # gen_ai semantic conventions (X2): this span covers one
                        # ``run_agent`` execution, mediated end-to-end by pydantic-ai —
                        # the invariant "which framework" fact, set once at span-open;
                        # per-call facts (model/tokens/tool-call count) land at
                        # :meth:`on_response`/:meth:`on_graph_end` as they become known.
                        "gen_ai.system": "pydantic_ai",
                    },
                )
                span = span_cm.__enter__()
                self._active_spans[run_id] = span
                self._span_tokens[run_id] = span_cm
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
            except Exception as exc:
                logger.debug(
                    "Token recording failed (exception_type=%s)", type(exc).__name__
                )
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
        **metadata: Any,
    ) -> None:
        """Record the end of a graph execution.

        ``model`` and ``tool_call_count`` are optional gen_ai attrs stamped
        onto the run's span BEFORE it closes (an ended span rejects further
        attributes) — the caller passes whatever it has: a run that never
        resolved a model, or whose tool calls weren't tallied, simply omits
        them (X2 "tokens/tool-call count if available").
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
        span_cm = self._span_tokens.pop(run_id, None)
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
            except Exception as exc:  # noqa: BLE001 — tracing must never break the caller
                logger.debug(
                    "TelemetryEngine: span attribute set failed (exception_type=%s)",
                    type(exc).__name__,
                )
        if span_cm is not None:
            try:
                span_cm.__exit__(None, None, None)
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

    def is_otel_configured(self) -> bool:
        """Whether :meth:`_setup_otel` configured a REAL TracerProvider/MeterProvider.

        Triggers lazy init first, so this reflects the effective state even
        before any ``on_graph_*``/``on_response`` call.
        """
        self._lazy_init()
        return self._otel_configured

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
