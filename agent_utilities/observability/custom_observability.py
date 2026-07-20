#!/usr/bin/python
from __future__ import annotations

"""Custom Observability Module.

CONCEPT:AU-OS.config.secrets-authentication — Telemetry & Observability

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
import contextlib
import logging
import math
import os
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from agent_utilities.core.config import AgentConfig, resolve_langfuse_host, setting
from agent_utilities.security.persistence_privacy import persistence_reference

if TYPE_CHECKING:
    pass


from agent_utilities.core.config import (
    HAS_LOGFIRE,
)
from agent_utilities.core.contextual_model import (
    disable_context_agent_instrumentation,
    instrument_context_agents,
)

__all__ = [
    "setup_otel",
    "verify_otel_pipeline",
    "get_otel_status_summary",
    "parse_otlp_headers",
]

logfire: Any
try:
    import logfire  # type: ignore[no-redef]  # annotated above as Any for the ImportError fallback
except ImportError:
    logfire = None


from agent_utilities.base_utilities import (
    retrieve_package_name,
)

logger = logging.getLogger(__name__)

_otel_initialized = False
_agent_instrumented_metadata_only = False
_web_instrumentation_enabled = False
_HEADER_NAME_RE = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,256}\Z")
_SAFE_ATTRIBUTE_STRINGS = frozenset(
    {
        "cancelled",
        "client",
        "completed",
        "error",
        "failed",
        "internal",
        "ok",
        "server",
        "success",
        "unknown",
    }
)
_SAFE_SPAN_NAMES = frozenset(
    {
        "agent run",
        "execute_tool",
        "graph.run",
        "invoke_agent",
        "model request",
        "running output function",
        "running tool",
    }
)
_SENSITIVE_ATTRIBUTE_TERMS = frozenset(
    {
        "argument",
        "arguments",
        "authorization",
        "body",
        "completion",
        "content",
        "cookie",
        "credential",
        "email",
        "header",
        "input",
        "output",
        "password",
        "path",
        "prompt",
        "query",
        "request",
        "response",
        "result",
        "secret",
        "token",
        "url",
        "user",
    }
)


def _generate_otlp_auth_header(public_key: str, secret_key: str) -> str:
    """Generate a Basic Auth header value from Langfuse public/secret key pair.

    CONCEPT:AU-OS.config.secrets-authentication — OTel Authentication

    Langfuse's OTLP endpoint expects HTTP Basic Auth where:
        username = public_key
        password = secret_key

    Args:
        public_key: Langfuse public key (e.g. ``lf_pk_...``).
        secret_key: Langfuse secret key (e.g. ``lf_sk_...``).

    Returns:
        Header string in format ``Authorization=Basic <base64>``.
    """
    if (
        not public_key
        or not secret_key
        or len(public_key) > 16_384
        or len(secret_key) > 16_384
        or any(character in f"{public_key}{secret_key}" for character in "\r\n\x00")
    ):
        raise ValueError("invalid OTLP credentials")
    auth_string = f"{public_key}:{secret_key}"
    auth_encoded = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
    return f"Authorization=Basic {auth_encoded}"


def _same_origin(left: str, right: str) -> bool:
    """Compare origins after applying the canonical secure-transport policy."""

    try:
        left_url = urlparse(resolve_langfuse_host("", environ={"LANGFUSE_HOST": left}))
        right_url = urlparse(
            resolve_langfuse_host("", environ={"LANGFUSE_HOST": right})
        )
        left_port = left_url.port or (443 if left_url.scheme == "https" else 80)
        right_port = right_url.port or (443 if right_url.scheme == "https" else 80)
        return bool(
            left_url.scheme == right_url.scheme
            and left_url.hostname
            and right_url.hostname
            and left_url.hostname.casefold() == right_url.hostname.casefold()
            and left_port == right_port
        )
    except ValueError:
        return False


def _resolve_otel_endpoint(
    endpoint: str | None,
    *,
    public_key: str | None = None,
    secret_key: str | None = None,
) -> str:
    """Resolve one secure OTLP endpoint without exposing it.

    An explicit endpoint or ``OTEL_EXPORTER_OTLP_ENDPOINT`` wins. When neither
    exists, a canonical Langfuse credential-reference pair—or an already-resolved
    in-memory pair—selects that deployment's OTLP trace endpoint. Cleartext is
    accepted only for the exact loopback hosts allowed by ``LANGFUSE_HOST``.
    """

    from agent_utilities.observability.langfuse_trust import (
        langfuse_credentials_configured,
    )

    if bool(public_key) != bool(secret_key):
        raise ValueError("OTLP credential pair is incomplete")
    target = str(endpoint or setting("OTEL_EXPORTER_OTLP_ENDPOINT", "") or "").strip()
    if not target and (
        (public_key and secret_key) or langfuse_credentials_configured()
    ):
        target = f"{resolve_langfuse_host().rstrip('/')}/api/public/otel"
    if not target:
        return ""
    return resolve_langfuse_host("", environ={"LANGFUSE_HOST": target})


def _is_langfuse_otel_endpoint(endpoint: str, host: str) -> bool:
    """Return whether ``endpoint`` is the canonical Langfuse trace collector."""

    if not _same_origin(endpoint, host):
        return False
    normalized = endpoint.rstrip("/")
    if normalized.endswith("/v1/traces"):
        normalized = normalized.removesuffix("/v1/traces")
    return normalized == f"{host.rstrip('/')}/api/public/otel"


def _resolve_otel_headers(
    *,
    endpoint: str,
    headers: str | None,
    public_key: str | None,
    secret_key: str | None,
) -> tuple[str, bool]:
    """Resolve one runtime-only OTLP auth source.

    Durable configuration accepts only secret references.  Raw values may reach this
    boundary only as already-resolved in-memory arguments (for example from the
    reference-only CLI).  When the endpoint shares the configured Langfuse origin,
    its canonical credential-reference pair is reused automatically.
    """

    from agent_utilities.observability.langfuse_trust import (
        langfuse_credentials_configured,
        resolve_langfuse_credentials,
        resolve_langfuse_host,
    )
    from agent_utilities.security.cli_secrets import (
        resolve_runtime_secret_reference,
    )

    header_ref = str(setting("OTEL_EXPORTER_OTLP_HEADERS_REF", "") or "").strip()
    public_ref = str(setting("OTEL_EXPORTER_OTLP_PUBLIC_KEY_REF", "") or "").strip()
    secret_ref = str(setting("OTEL_EXPORTER_OTLP_SECRET_KEY_REF", "") or "").strip()
    explicit_pair = bool(public_key) or bool(secret_key)
    configured_pair = bool(public_ref) or bool(secret_ref)
    if bool(public_key) != bool(secret_key) or bool(public_ref) != bool(secret_ref):
        raise ValueError("OTLP credential pair is incomplete")
    if header_ref and (explicit_pair or configured_pair):
        raise ValueError("OTLP authentication source is ambiguous")
    if headers and explicit_pair:
        raise ValueError("OTLP authentication source is ambiguous")

    resolved = str(headers or "")
    if not resolved and header_ref:
        resolved = resolve_runtime_secret_reference(header_ref)
    if not resolved and explicit_pair:
        resolved = _generate_otlp_auth_header(str(public_key), str(secret_key))
    if not resolved and configured_pair:
        resolved = _generate_otlp_auth_header(
            resolve_runtime_secret_reference(public_ref),
            resolve_runtime_secret_reference(secret_ref),
        )

    reused_langfuse = False
    if not resolved and langfuse_credentials_configured():
        langfuse_host = resolve_langfuse_host()
        if _same_origin(endpoint, langfuse_host):
            langfuse_public, langfuse_secret = resolve_langfuse_credentials()
            resolved = _generate_otlp_auth_header(langfuse_public, langfuse_secret)
            reused_langfuse = True
    if resolved:
        parse_otlp_headers(resolved)
    return resolved, reused_langfuse


def _resolve_otel_transport(endpoint: str) -> Any:
    """Resolve the purpose-specific TLS profile for the selected collector."""

    from agent_utilities.core.transport_security import (
        resolve_configured_tls_profile,
    )

    cfg = AgentConfig()
    same_langfuse_origin = _same_origin(endpoint, resolve_langfuse_host())
    profile_name = cfg.otel_tls_profile
    profile_ref = cfg.otel_tls_profile_ref
    if same_langfuse_origin and not (profile_name or profile_ref):
        profile_name = cfg.langfuse_tls_profile
        profile_ref = cfg.langfuse_tls_profile_ref
    return resolve_configured_tls_profile(
        "OTEL",
        profile_name=profile_name,
        profile_ref=profile_ref,
        config=cfg,
    )


def parse_otlp_headers(headers: str) -> dict[str, str]:
    """Parse an OTLP headers string (``"key=value,key2=value2"``) into a dict.

    CONCEPT:AU-OS.config.secrets-authentication — shared OTLP header parsing

    Extracted so every OTLP exporter constructed in this package (the Langfuse
    span processor below, and :class:`agent_utilities.observability.
    TelemetryEngine`'s real TracerProvider/MeterProvider setup) parses the
    ``OTEL_EXPORTER_OTLP_HEADERS``-style string identically instead of each
    reimplementing the same split/strip logic.
    """
    if len(headers) > 65_536 or any(character in headers for character in "\r\n\x00"):
        raise ValueError("invalid OTLP headers")
    header_dict: dict[str, str] = {}
    if headers:
        parts = headers.split(",")
        if len(parts) > 32:
            raise ValueError("too many OTLP headers")
        for part in parts:
            part = part.strip()
            if "=" not in part:
                raise ValueError("invalid OTLP header")
            raw_key, raw_value = part.split("=", 1)
            key = raw_key.strip()
            value = raw_value.strip()
            normalized = key.casefold()
            if (
                not _HEADER_NAME_RE.fullmatch(key)
                or normalized in {existing.casefold() for existing in header_dict}
                or len(value) > 16_384
            ):
                raise ValueError("invalid OTLP header")
            header_dict[key] = value
    return header_dict


def _opaque_label(kind: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return persistence_reference(kind, text[:8192], namespace="otel")


def _safe_attribute_value(key: str, value: Any) -> Any | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return max(-(2**63), min(2**63 - 1, value))
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _SAFE_ATTRIBUTE_STRINGS:
            return normalized
        return _opaque_label(f"attribute_{key[:64]}", value)
    if isinstance(value, (list, tuple)):
        cleaned = [_safe_attribute_value(key, item) for item in value[:64]]
        return [item for item in cleaned if item is not None]
    return None


def _metadata_only_attributes(attributes: Any) -> dict[str, Any]:
    if not attributes:
        return {}
    output: dict[str, Any] = {}
    for index, (raw_key, value) in enumerate(attributes.items()):
        if index >= 128:
            break
        key = str(raw_key)
        if not key or len(key) > 256 or any(ord(character) < 32 for character in key):
            key = f"attribute.{_opaque_label('attribute_key', key)}"
        key_terms = {term for term in re.split(r"[^a-z0-9]+", key.casefold()) if term}
        if key_terms.intersection(_SENSITIVE_ATTRIBUTE_TERMS):
            continue
        safe = _safe_attribute_value(key, value)
        if safe is not None:
            output[key] = safe
    return output


class _MetadataOnlySpanExporter:
    """Strip content, identities, resource detectors, events, links, and trace-state."""

    def __init__(self, inner: Any, *, service_ref: str) -> None:
        self._inner = inner
        self._service_ref = service_ref or "agent-utilities"

    @staticmethod
    def _context(value: Any) -> Any:
        if value is None:
            return None
        from opentelemetry.trace import SpanContext, TraceState

        return SpanContext(
            trace_id=value.trace_id,
            span_id=value.span_id,
            is_remote=bool(value.is_remote),
            trace_flags=value.trace_flags,
            trace_state=TraceState(),
        )

    def export(self, spans: Any) -> Any:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import ReadableSpan
        from opentelemetry.trace.status import Status

        safe_spans = []
        for span in list(spans)[:2048]:
            raw_name = str(getattr(span, "name", "") or "")
            name = (
                raw_name
                if raw_name in _SAFE_SPAN_NAMES
                else f"operation:{_opaque_label('span_name', raw_name)}"
            )
            safe_spans.append(
                ReadableSpan(
                    name=name,
                    context=self._context(getattr(span, "context", None)),
                    parent=self._context(getattr(span, "parent", None)),
                    resource=Resource(
                        {
                            "service.name": self._service_ref,
                            "telemetry.content_retention": "metadata",
                        }
                    ),
                    attributes=_metadata_only_attributes(
                        getattr(span, "attributes", None)
                    ),
                    events=(),
                    links=(),
                    kind=span.kind,
                    instrumentation_info=None,
                    status=Status(span.status.status_code),
                    start_time=span.start_time,
                    end_time=span.end_time,
                    instrumentation_scope=None,
                )
            )
        return self._inner.export(safe_spans)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        method = getattr(self._inner, "force_flush", None)
        return bool(method(timeout_millis)) if callable(method) else True


def _create_otlp_span_processor(
    endpoint: str,
    headers: str,
    protocol: str = "http/protobuf",
    transport_security: Any | None = None,
    service_ref: str = "agent-utilities",
) -> Any | None:
    """Create an OTel BatchSpanProcessor with OTLPSpanExporter for Langfuse.

    CONCEPT:AU-OS.config.secrets-authentication — OTLP Span Export Pipeline

    This creates the actual export pipeline that sends spans from Logfire
    to Langfuse's OTLP ingestion endpoint. The ``BatchSpanProcessor``
    buffers spans and sends them in batches for efficiency.

    Args:
        endpoint: Full HTTPS OTLP endpoint supplied by runtime configuration.
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

        if protocol != "http/protobuf":
            raise ValueError("unsupported OTLP protocol")
        header_dict = parse_otlp_headers(headers)

        # Ensure endpoint includes /v1/traces for the trace exporter
        traces_endpoint = endpoint.rstrip("/")
        if not traces_endpoint.endswith("/v1/traces"):
            traces_endpoint = f"{traces_endpoint}/v1/traces"

        exporter_kwargs: dict[str, Any] = {
            "endpoint": traces_endpoint,
            "headers": header_dict,
        }
        if transport_security is not None:
            from agent_utilities.core.http_client import create_requests_session

            exporter_kwargs["session"] = create_requests_session(
                transport_security=transport_security
            )
        exporter = _MetadataOnlySpanExporter(
            OTLPSpanExporter(**exporter_kwargs), service_ref=service_ref
        )

        processor = BatchSpanProcessor(
            exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            schedule_delay_millis=5000,
        )

        logger.info("Metadata-only OTLP span processor created")
        return processor

    except ImportError as exc:
        logger.warning(
            "Failed to create OTLP span processor (exception_type=%s)",
            type(exc).__name__,
        )
        return None
    except Exception as exc:
        logger.error(
            "Failed to create OTLP span processor (exception_type=%s)",
            type(exc).__name__,
        )
        return None


def setup_otel(
    service_name: str | None = None,
    endpoint: str | None = None,
    headers: str | None = None,
    public_key: str | None = None,
    secret_key: str | None = None,
    protocol: str | None = None,
    service_version: str | None = None,
    environment: str | None = None,
):
    """Setup OpenTelemetry tracing via Logfire, exporting to Langfuse OTLP.

    CONCEPT:AU-OS.config.secrets-authentication — Full OTel Pipeline Setup

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
        endpoint: Secure OTLP exporter endpoint URL. HTTPS is required except for
            canonical loopback HTTP. When omitted it is read from runtime
            configuration or derived from the configured Langfuse host.
        headers: Already-resolved, in-memory OTLP headers. Durable configuration
            uses ``OTEL_EXPORTER_OTLP_HEADERS_REF`` instead.
        public_key: Already-resolved, in-memory public key.
        secret_key: Already-resolved, in-memory secret key. Durable configuration
            uses the OTLP key-reference pair or the canonical Langfuse references.
        protocol: OTLP protocol. Default: ``http/protobuf``.
        service_version: Optional service version for trace metadata.
        environment: Optional environment tag (e.g. ``production``, ``development``).

    Example::

        from agent_utilities.observability.custom_observability import setup_otel

        # Minimal — uses config.json defaults
        setup_otel(service_name="my-agent")

        # Endpoint/auth/TLS are resolved from AgentConfig references.
    """
    global _agent_instrumented_metadata_only, _otel_initialized

    try:
        target_endpoint = _resolve_otel_endpoint(
            endpoint,
            public_key=public_key,
            secret_key=secret_key,
        )
    except Exception as exc:
        logger.warning(
            "OTLP setup skipped: endpoint configuration is invalid (%s)",
            type(exc).__name__,
        )
        return
    if not target_endpoint:
        logger.debug("OTLP setup skipped: no runtime endpoint configured")
        return

    if not HAS_LOGFIRE:
        logger.warning(
            "OpenTelemetry is enabled but logfire is not installed. "
            "Trace logging is disabled. Install with: pip install pydantic-ai-slim[logfire]"
        )
        return

    # Step 1: Resolve OTLP auth headers
    try:
        resolved_headers, _ = _resolve_otel_headers(
            endpoint=target_endpoint,
            headers=headers,
            public_key=public_key,
            secret_key=secret_key,
        )
    except Exception as exc:
        logger.warning(
            "OTLP setup skipped: credential references are invalid (%s)",
            type(exc).__name__,
        )
        return

    if not resolved_headers:
        logger.warning(
            "No OTLP headers or Langfuse keys configured — traces will not authenticate. "
            "Set langfuse_public_key_ref + langfuse_secret_key_ref in config.json."
        )

    # Step 2: Resolve endpoint and protocol
    target_protocol = str(
        protocol or setting("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    ).strip()
    if target_protocol != "http/protobuf":
        logger.warning("OTLP setup skipped: unsupported protocol")
        return
    target_service_name = _opaque_label(
        "service", service_name or retrieve_package_name() or "agent-utilities"
    )

    # Step 3: Set environment variables for downstream OTel SDK consumers
    if target_endpoint:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = target_endpoint
    if resolved_headers:
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = resolved_headers
    if target_protocol:
        os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = target_protocol
    if target_service_name:
        os.environ["OTEL_SERVICE_NAME"] = target_service_name

    logger.debug("OTel metadata-only configuration resolved")

    if _otel_initialized:
        logger.debug("Re-configuring metadata-only OTel")

    # Step 4: Resolve runtime-only trust and create the OTLP span processor.
    try:
        transport_security = _resolve_otel_transport(
            target_endpoint,
        )
        os.environ.update(transport_security.child_env())
    except Exception as exc:
        logger.warning(
            "OTLP setup skipped: transport security profile is invalid (%s)",
            type(exc).__name__,
        )
        return
    span_processors = []
    if target_endpoint and resolved_headers:
        processor = _create_otlp_span_processor(
            endpoint=target_endpoint,
            headers=resolved_headers,
            protocol=target_protocol or "http/protobuf",
            transport_security=transport_security,
            service_ref=target_service_name,
        )
        if processor:
            span_processors.append(processor)
            logger.info("Metadata-only OTLP exporter configured")
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
        configure_kwargs["service_version"] = _opaque_label(
            "service_version", service_version
        )
    if environment:
        configure_kwargs["environment"] = _opaque_label("environment", environment)

    if span_processors:
        configure_kwargs["additional_span_processors"] = span_processors

    logfire.configure(**configure_kwargs)

    # Step 6: PydanticAI defaults to including prompts, outputs, tool arguments,
    # tool results, and binary content. Pass explicit false settings through
    # both integration entry points and verify the installed API accepted them.
    try:
        from pydantic_ai import InstrumentationSettings

        settings = InstrumentationSettings(
            include_content=False,
            include_binary_content=False,
            version=5,
        )
        logfire.instrument_pydantic_ai(
            include_content=False,
            include_binary_content=False,
            version=5,
        )
        installed = instrument_context_agents(settings)
        if (
            getattr(installed, "include_content", None) is not False
            or getattr(installed, "include_binary_content", None) is not False
        ):
            raise RuntimeError("instrumentation content policy was not installed")
        _agent_instrumented_metadata_only = True
    except Exception as exc:
        with contextlib.suppress(Exception):
            disable_context_agent_instrumentation()
        _agent_instrumented_metadata_only = False
        logger.warning(
            "PydanticAI tracing disabled: metadata-only policy unavailable "
            "(exception_type=%s)",
            type(exc).__name__,
        )

    # FastAPI/Starlette instrumentation requires a concrete application and
    # otherwise records URL/route/user-agent/request attributes. setup_otel has
    # no app-bound redaction hook, so it is intentionally skipped here instead
    # of enabling an ungoverned global auto-instrumentor.

    _otel_initialized = True
    logger.info(
        "Metadata-only OpenTelemetry pipeline initialized (processors=%d)",
        len(span_processors),
    )


def verify_otel_pipeline() -> dict[str, Any]:
    """Verify the OTel → Langfuse pipeline is operational.

    CONCEPT:AU-OS.config.secrets-authentication — Pipeline Health Check

    Tests connectivity to the Langfuse OTLP endpoint and returns
    a diagnostic report.

    Returns:
        Dict with keys: ``initialized``, ``endpoint_configured``, ``headers_set``,
        ``logfire_available``, ``exporter_ok``, ``agent_instrumented``.
    """
    endpoint = ""
    resolved_headers = ""
    resolution_error: str | None = None
    try:
        endpoint = _resolve_otel_endpoint(None)
        if endpoint:
            runtime_headers = str(
                setting("OTEL_EXPORTER_OTLP_HEADERS", "") or ""
            ).strip()
            resolved_headers, _ = _resolve_otel_headers(
                endpoint=endpoint,
                headers=runtime_headers or None,
                public_key=None,
                secret_key=None,
            )
    except Exception as exc:
        resolution_error = type(exc).__name__

    report: dict[str, Any] = {
        "initialized": _otel_initialized,
        "logfire_available": HAS_LOGFIRE,
        "endpoint_configured": bool(endpoint),
        "headers_set": bool(resolved_headers),
        "protocol": (
            "http/protobuf"
            if setting("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
            == "http/protobuf"
            else "unsupported"
        ),
        "service_configured": bool(setting("OTEL_SERVICE_NAME", "")),
        "exporter_ok": False,
        "agent_instrumented": _agent_instrumented_metadata_only,
        "content_capture": False,
        "web_instrumented": _web_instrumentation_enabled,
    }
    if resolution_error:
        report["endpoint_error"] = resolution_error
        return report

    # A Langfuse API read is the authenticated health contract for its OTLP sink.
    # Generic collectors do not expose one portable read endpoint, so they remain
    # unproven instead of treating an arbitrary 4xx response as success.
    if endpoint:
        try:
            from agent_utilities.core.http_client import create_http_client

            langfuse_host = resolve_langfuse_host()
            if not _is_langfuse_otel_endpoint(endpoint, langfuse_host):
                report["endpoint_error"] = "authenticated_health_unsupported"
                return report
            if not resolved_headers:
                report["endpoint_error"] = "authentication_missing"
                return report
            trust = _resolve_otel_transport(endpoint)
            probe_endpoint = f"{langfuse_host.rstrip('/')}/api/public/traces"
            try:
                with create_http_client(
                    timeout=5.0,
                    follow_redirects=False,
                    **trust.httpx_kwargs(),
                ) as client:
                    resp = client.get(
                        probe_endpoint,
                        params={"limit": 1},
                        headers=parse_otlp_headers(resolved_headers),
                    )
            finally:
                trust.cleanup()
            report["endpoint_status"] = resp.status_code
            if not 200 <= resp.status_code < 300:
                report["endpoint_error"] = (
                    "authentication_failed"
                    if resp.status_code in {401, 403}
                    else "api_probe_failed"
                )
                return report
            payload = resp.json()
            report["exporter_ok"] = bool(
                isinstance(payload, dict) and isinstance(payload.get("data"), list)
            )
            if not report["exporter_ok"]:
                report["endpoint_error"] = "api_response_invalid"
        except Exception as exc:
            report["exporter_ok"] = False
            report["endpoint_error"] = type(exc).__name__

    return report


def get_otel_status_summary() -> str:
    """Get a human-readable summary of the OTel pipeline status.

    CONCEPT:AU-OS.config.secrets-authentication — Diagnostics

    Returns:
        Multi-line string summarizing the pipeline health.
    """
    report = verify_otel_pipeline()
    lines = [
        "=== OTel Pipeline Status ===",
        f"  Initialized:     {report['initialized']}",
        f"  Logfire:         {report['logfire_available']}",
        f"  Endpoint:        {'CONFIGURED' if report['endpoint_configured'] else 'NOT SET'}",
        f"  Headers:         {'✅ Set' if report['headers_set'] else '❌ Missing'}",
        f"  Exporter:        {'✅ OK' if report['exporter_ok'] else '❌ Not connected'}",
        f"  Instrumented:    {report['agent_instrumented']}",
        f"  Service:         {'CONFIGURED' if report['service_configured'] else 'NOT SET'}",
        "  Content:         METADATA ONLY",
    ]
    if "endpoint_error" in report:
        lines.append(f"  Endpoint Error:  {report['endpoint_error']}")
    return "\n".join(lines)
