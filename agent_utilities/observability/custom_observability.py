#!/usr/bin/python
"""Custom Observability Module.

This module provides instrumentation for OpenTelemetry (OTel) using Logfire.
It handles the generation of OTLP headers from credentials, service-level
configuration, and automatic instrumentation of pydantic-ai agents for
distributed tracing.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import TYPE_CHECKING

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

try:
    import logfire
except ImportError:
    logfire = None


from agent_utilities.base_utilities import (
    retrieve_package_name,
)

logger = logging.getLogger(__name__)

_otel_initialized = False


def setup_otel(
    service_name: str | None = None,
    endpoint: str | None = DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT,
    headers: str | None = DEFAULT_OTEL_EXPORTER_OTLP_HEADERS,
    public_key: str | None = DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY,
    secret_key: str | None = DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY,
    protocol: str | None = DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
):
    """Setup OpenTelemetry tracing using Logfire and instrument pydantic_ai."""
    global _otel_initialized

    if not HAS_LOGFIRE:
        logger.warning(
            "OpenTelemetry is enabled but logfire is not installed. Trace logging is disabled."
        )
        return

    if not (headers or os.getenv("OTEL_EXPORTER_OTLP_HEADERS")) and (
        (public_key or os.getenv("OTEL_EXPORTER_OTLP_PUBLIC_KEY"))
        and (secret_key or os.getenv("OTEL_EXPORTER_OTLP_SECRET_KEY"))
    ):
        pk = public_key or os.getenv("OTEL_EXPORTER_OTLP_PUBLIC_KEY")
        sk = secret_key or os.getenv("OTEL_EXPORTER_OTLP_SECRET_KEY")
        auth_string = f"{pk}:{sk}"
        auth_encoded = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth_encoded}"
        logger.debug("Generated OTLP Basic Auth headers from public/secret keys")

    target_service_name = service_name or retrieve_package_name()

    if _otel_initialized:
        logger.debug(f"Re-configuring OTel for service: {target_service_name}")

    target_endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    target_headers = headers or os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    logger.debug(
        f"OTel Config: endpoint={target_endpoint}, protocol={protocol}, headers={'REDACTED' if target_headers else 'None'}"
    )

    logfire.configure(
        send_to_logfire=False,
        service_name=target_service_name,
        distributed_tracing=True,
    )

    logfire.instrument_pydantic_ai()
    Agent.instrument_all()

    _otel_initialized = True
    logger.info(
        f"OpenTelemetry logging enabled via logfire for service: {target_service_name}"
    )
