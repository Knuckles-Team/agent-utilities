#!/usr/bin/python
from __future__ import annotations

"""Custom Observability Module.

CONCEPT:OS-5.1 — Telemetry & Observability

This module provides instrumentation for OpenTelemetry (OTel) tracing
via Logfire, routing all telemetry to Langfuse's OTLP ingestion endpoint.

Architecture::

    ┌──────────────┐   instrument_all()   ┌──────────────────┐
    │  Pydantic-AI  │ ──────────────────► │   Logfire SDK     │
    │   Agents      │                     │ (send_to_logfire  │
    └──────────────┘                     │    = False)       │
                                          └────────┬─────────┘
                                                   │ OTel Spans
                                          ┌────────▼─────────┐
                                          │  BatchSpanProc    │
                                          │  + OTLPSpanExport │
                                          └────────┬─────────┘
                                                   │ http/protobuf
                                          ┌────────▼─────────┐
                                          │  Langfuse OTLP    │
                                          │  /api/public/otel │
                                          └──────────────────┘

The pipeline handles:
    - OTLP header generation from Langfuse public/secret key pairs
    - Logfire SDK configuration with ``send_to_logfire=False``
    - ``BatchSpanProcessor`` with ``OTLPSpanExporter`` pointed at Langfuse
    - ``pydantic_ai.Agent.instrument_all()`` for automatic agent tracing
    - Distributed tracing context propagation across agent calls
    - Health verification against the Langfuse OTLP endpoint

Environment Variables (auto-set from ``config.json``):
    - ``OTEL_EXPORTER_OTLP_ENDPOINT``: Langfuse OTLP endpoint
    - ``OTEL_EXPORTER_OTLP_HEADERS``: ``Authorization=Basic <b64>``
    - ``OTEL_EXPORTER_OTLP_PROTOCOL``: ``http/protobuf`` (default)
    - ``OTEL_SERVICE_NAME``: Service name for trace attribution

References:
    - Langfuse OTLP: https://langfuse.com/docs/integrations/opentelemetry
    - Logfire + Langfuse: https://logfire.pydantic.dev/docs/integrations/langfuse/
    - OTel Python SDK: https://opentelemetry-python.readthedocs.io/
"""


import base64
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


from pydantic_ai import Agent

from agent_utilities.core.config import (
    DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT,
    DEFAULT_OTEL_EXPORTER_OTLP_HEADERS,
    DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
    DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY,
    DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY,
    HAS_LOGFIRE,
)

logfire: Any
try:
    import logfire
except ImportError:
    logfire = None


from agent_utilities.base_utilities import (
    retrieve_package_name,
)

logger = logging.getLogger(__name__)

_otel_initialized = False


def _generate_otlp_auth_header(public_key: str, secret_key: str) -> str:
    """Generate a Basic Auth header value from Langfuse public/secret key pair.

    CONCEPT:OS-5.1 — OTel Authentication

    Langfuse's OTLP endpoint expects HTTP Basic Auth where:
        username = public_key
        password = secret_key

    Args:
        public_key: Langfuse public key (e.g. ``lf_pk_...``).
        secret_key: Langfuse secret key (e.g. ``lf_sk_...``).

    Returns:
        Header string in format ``Authorization=Basic <base64>``.
    """
    auth_string = f"{public_key}:{secret_key}"
    auth_encoded = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
    return f"Authorization=Basic {auth_encoded}"


def _create_otlp_span_processor(
    endpoint: str,
    headers: str,
    protocol: str = "http/protobuf",
) -> Any | None:
    """Create an OTel BatchSpanProcessor with OTLPSpanExporter for Langfuse.

    CONCEPT:OS-5.1 — OTLP Span Export Pipeline

    This creates the actual export pipeline that sends spans from Logfire
    to Langfuse's OTLP ingestion endpoint. The ``BatchSpanProcessor``
    buffers spans and sends them in batches for efficiency.

    Args:
        endpoint: Full OTLP endpoint URL (e.g. ``http://langfuse.arpa/api/public/otel``).
        headers: OTLP headers string (``key=value`` format).
        protocol: OTLP protocol (``http/protobuf`` or ``grpc``).

    Returns:
        Configured ``BatchSpanProcessor`` or ``None`` if OTel SDK is unavailable.
    """
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Parse headers from "key=value,key2=value2" format to dict
        header_dict: dict[str, str] = {}
        if headers:
            for part in headers.split(","):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    header_dict[k.strip()] = v.strip()

        # Ensure endpoint includes /v1/traces for the trace exporter
        traces_endpoint = endpoint.rstrip("/")
        if not traces_endpoint.endswith("/v1/traces"):
            traces_endpoint = f"{traces_endpoint}/v1/traces"

        exporter = OTLPSpanExporter(
            endpoint=traces_endpoint,
            headers=header_dict,
        )

        processor = BatchSpanProcessor(
            exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            schedule_delay_millis=5000,
        )

        logger.info(
            "OTLP span processor created: endpoint=%s, headers=%s",
            traces_endpoint,
            list(header_dict.keys()),
        )
        return processor

    except ImportError as e:
        logger.warning(
            "Failed to create OTLP span processor — missing OTel SDK packages: %s. "
            "Install with: pip install opentelemetry-exporter-otlp-proto-http",
            e,
        )
        return None
    except Exception as e:
        logger.error("Failed to create OTLP span processor: %s", e)
        return None


def setup_otel(
    service_name: str | None = None,
    endpoint: str | None = DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT,
    headers: str | None = DEFAULT_OTEL_EXPORTER_OTLP_HEADERS,
    public_key: str | None = DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY,
    secret_key: str | None = DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY,
    protocol: str | None = DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
    service_version: str | None = None,
    environment: str | None = None,
):
    """Setup OpenTelemetry tracing via Logfire, exporting to Langfuse OTLP.

    CONCEPT:OS-5.1 — Full OTel Pipeline Setup

    This is the primary entry point for initializing the observability pipeline.
    It configures:

    1. **OTLP Auth Headers**: Generated from Langfuse public/secret keys
    2. **Logfire SDK**: Configured with ``send_to_logfire=False`` for self-hosted routing
    3. **BatchSpanProcessor**: With ``OTLPSpanExporter`` pointed at Langfuse
    4. **Agent Instrumentation**: ``pydantic_ai.Agent.instrument_all()`` auto-traces all agents
    5. **Environment Variables**: Set for any downstream OTel-aware libraries

    The pipeline ensures that every pydantic-ai agent call, tool invocation,
    and LLM request is captured as an OTel span and routed to Langfuse
    for centralized observability.

    Args:
        service_name: Service name for OTel resource attribution.
            Defaults to the package name (e.g. ``agent-utilities``).
        endpoint: OTLP exporter endpoint URL.
            Default: ``http://langfuse.arpa/api/public/otel``
        headers: Pre-built OTLP headers string. If ``None``, generated
            from ``public_key`` and ``secret_key``.
        public_key: Langfuse public key for auth header generation.
        secret_key: Langfuse secret key for auth header generation.
        protocol: OTLP protocol. Default: ``http/protobuf``.
        service_version: Optional service version for trace metadata.
        environment: Optional environment tag (e.g. ``production``, ``development``).

    Example::

        from agent_utilities.observability.custom_observability import setup_otel

        # Minimal — uses config.json defaults
        setup_otel(service_name="my-agent")

        # Explicit configuration
        setup_otel(
            service_name="my-agent",
            endpoint="http://langfuse.arpa/api/public/otel",
            public_key="lf_pk_...",
            secret_key="lf_sk_...",
        )
    """
    global _otel_initialized

    if not HAS_LOGFIRE:
        logger.warning(
            "OpenTelemetry is enabled but logfire is not installed. "
            "Trace logging is disabled. Install with: pip install pydantic-ai-slim[logfire]"
        )
        return

    # Step 1: Resolve OTLP auth headers
    resolved_headers = headers or os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    if not resolved_headers:
        pk = public_key or os.getenv("OTEL_EXPORTER_OTLP_PUBLIC_KEY")
        sk = secret_key or os.getenv("OTEL_EXPORTER_OTLP_SECRET_KEY")
        if pk and sk:
            resolved_headers = _generate_otlp_auth_header(pk, sk)
            logger.debug("Generated OTLP Basic Auth headers from public/secret keys")

    if not resolved_headers:
        logger.warning(
            "No OTLP headers or Langfuse keys configured — traces will not authenticate. "
            "Set langfuse_public_key + langfuse_secret_key in config.json."
        )

    # Step 2: Resolve endpoint and protocol
    target_endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    target_protocol = protocol or os.getenv(
        "OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"
    )
    target_service_name = service_name or retrieve_package_name()

    # Step 3: Set environment variables for downstream OTel SDK consumers
    if target_endpoint:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = target_endpoint
    if resolved_headers:
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = resolved_headers
    if target_protocol:
        os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = target_protocol
    if target_service_name:
        os.environ["OTEL_SERVICE_NAME"] = target_service_name

    logger.debug(
        "OTel Config: endpoint=%s, protocol=%s, headers=%s, service=%s",
        target_endpoint,
        target_protocol,
        "CONFIGURED" if resolved_headers else "NONE",
        target_service_name,
    )

    if _otel_initialized:
        logger.debug("Re-configuring OTel for service: %s", target_service_name)

    # Step 4: Create the OTLP span processor for Langfuse
    span_processors = []
    if target_endpoint and resolved_headers:
        processor = _create_otlp_span_processor(
            endpoint=target_endpoint,
            headers=resolved_headers,
            protocol=target_protocol or "http/protobuf",
        )
        if processor:
            span_processors.append(processor)
            logger.info(
                "OTLP exporter configured: %s → %s",
                target_service_name,
                target_endpoint,
            )
    else:
        logger.warning(
            "OTLP export disabled — missing endpoint (%s) or headers (%s). "
            "Traces will be collected locally only.",
            "set" if target_endpoint else "missing",
            "set" if resolved_headers else "missing",
        )

    # Step 5: Configure Logfire with the OTLP span processor
    configure_kwargs: dict[str, Any] = {
        "send_to_logfire": False,
        "service_name": target_service_name,
        "distributed_tracing": True,
    }

    if service_version:
        configure_kwargs["service_version"] = service_version
    if environment:
        configure_kwargs["environment"] = environment

    if span_processors:
        configure_kwargs["additional_span_processors"] = span_processors

    logfire.configure(**configure_kwargs)

    # Step 6: Instrument pydantic-ai agents and FastMCP for automatic tracing
    logfire.instrument_pydantic_ai()
    Agent.instrument_all()

    # FastMCP 3.3 native OpenTelemetry instrumentation via fastapi/starlette
    try:
        logfire.instrument_fastapi()
        logfire.instrument_starlette()
    except Exception as e:
        logger.debug(f"Could not instrument fastapi/starlette for FastMCP: {e}")

    _otel_initialized = True
    logger.info(
        "OpenTelemetry pipeline initialized: service=%s, endpoint=%s, "
        "processors=%d, distributed_tracing=True",
        target_service_name,
        target_endpoint or "none",
        len(span_processors),
    )


def verify_otel_pipeline() -> dict[str, Any]:
    """Verify the OTel → Langfuse pipeline is operational.

    CONCEPT:OS-5.1 — Pipeline Health Check

    Tests connectivity to the Langfuse OTLP endpoint and returns
    a diagnostic report.

    Returns:
        Dict with keys: ``initialized``, ``endpoint``, ``headers_set``,
        ``logfire_available``, ``exporter_ok``, ``agent_instrumented``.
    """
    report: dict[str, Any] = {
        "initialized": _otel_initialized,
        "logfire_available": HAS_LOGFIRE,
        "endpoint": os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
        "headers_set": bool(os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")),
        "protocol": os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", ""),
        "service_name": os.environ.get("OTEL_SERVICE_NAME", ""),
        "exporter_ok": False,
        "agent_instrumented": False,
    }

    # Check if agents are instrumented
    try:
        report["agent_instrumented"] = Agent._instrument_all  # type: ignore[attr-defined]
    except AttributeError:
        report["agent_instrumented"] = _otel_initialized

    # Ping the OTLP endpoint
    if report["endpoint"]:
        try:
            import httpx

            resp = httpx.get(
                report["endpoint"].rstrip("/").replace("/v1/traces", ""),
                timeout=5.0,
                follow_redirects=True,
            )
            report["exporter_ok"] = resp.status_code < 500
            report["endpoint_status"] = resp.status_code
        except Exception as e:
            report["exporter_ok"] = False
            report["endpoint_error"] = str(e)

    return report


def get_otel_status_summary() -> str:
    """Get a human-readable summary of the OTel pipeline status.

    CONCEPT:OS-5.1 — Diagnostics

    Returns:
        Multi-line string summarizing the pipeline health.
    """
    report = verify_otel_pipeline()
    lines = [
        "=== OTel Pipeline Status ===",
        f"  Initialized:     {report['initialized']}",
        f"  Logfire:         {report['logfire_available']}",
        f"  Endpoint:        {report['endpoint'] or 'NOT SET'}",
        f"  Headers:         {'✅ Set' if report['headers_set'] else '❌ Missing'}",
        f"  Exporter:        {'✅ OK' if report['exporter_ok'] else '❌ Not connected'}",
        f"  Instrumented:    {report['agent_instrumented']}",
        f"  Service:         {report['service_name'] or 'NOT SET'}",
    ]
    if "endpoint_error" in report:
        lines.append(f"  Endpoint Error:  {report['endpoint_error']}")
    return "\n".join(lines)
