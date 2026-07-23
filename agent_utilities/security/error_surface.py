"""Stable, privacy-safe failures for externally reachable surfaces.

Exception strings are not safe response or log material: HTTP clients, database
drivers, parsers, and subprocess wrappers commonly include credentials,
endpoints, local paths, queries, or payload fragments in them.  This module is
the single public-failure boundary for REST, MCP, and streaming adapters.
"""

from __future__ import annotations

import json
import logging
import uuid
from types import MappingProxyType
from typing import Any

from agent_utilities.protocols.epistemic_operations import (
    OperationError,
    OperationResult,
)

_DEFAULT_LOGGER = logging.getLogger(__name__)

PUBLIC_ERROR_MESSAGES = MappingProxyType(
    {
        "operation_failed": "The requested operation could not be completed.",
        "invalid_request": "The request could not be processed.",
        "dependency_unavailable": "A required service is unavailable.",
        "permission_denied": "The operation is not authorized.",
    }
)


def public_error_payload(
    exc: BaseException,
    *,
    logger: logging.Logger | None = None,
    code: str = "operation_failed",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a strict ``OperationResult`` failure and a correlation log that
    preserves the real cause.

    Unknown codes fail closed to ``operation_failed``.  Callers cannot supply a
    log message or public message, which prevents an untrusted value from being
    reintroduced accidentally at a call site. ``context`` is an optional
    diagnostic mapping (e.g. ``{"action": ..., "tool": ...}``): only its **keys**
    are logged (never values, and never the public payload) so it can aid
    correlation without reintroducing an untrusted value.

    The exception itself IS logged (not just its class name): this is the
    single public-failure boundary for REST/MCP/streaming (~200+ call sites),
    so collapsing it to ``type(exc).__name__`` here made every failure across
    the whole external surface equally undiagnosable server-side (a denied
    scope, a missing driver, an unreadable durable record all looked like the
    same bare "RuntimeError") even though the PUBLIC payload already stays
    minimal (``detail_ref=None``, a fixed generic message) by design. The
    process-wide log-privacy boundary (``core/log_privacy.py``) still
    sanitizes the exception's message text (redacts endpoints/paths/emails)
    before it reaches any handler, so this does not reintroduce a leak.
    """

    safe_code = code if code in PUBLIC_ERROR_MESSAGES else "operation_failed"
    correlation_id = f"correlation:{uuid.uuid4().hex}"
    sink = logger or _DEFAULT_LOGGER
    sink.warning(
        "External operation failed (code=%s correlation_id=%s exception=%s "
        "context_keys=%s)",
        safe_code,
        correlation_id,
        exc,
        sorted(context) if context else [],
    )
    return OperationResult(
        schema_version="1",
        operation_id=f"operation:{uuid.uuid4().hex}",
        status="failed",
        result_kind=None,
        result_ref=None,
        error=OperationError(
            code=safe_code,
            retryable=safe_code == "dependency_unavailable",
            correlation_id=correlation_id,
            detail_ref=None,
        ),
        redirect=None,
    ).model_dump(mode="json")


def public_error_json(
    exc: BaseException,
    *,
    logger: logging.Logger | None = None,
    code: str = "operation_failed",
    context: dict[str, Any] | None = None,
) -> str:
    """Serialize :func:`public_error_payload` for string-returning MCP tools."""

    payload = public_error_payload(exc, logger=logger, code=code, context=context)
    return json.dumps(payload)


def public_error_text(
    exc: BaseException,
    *,
    logger: logging.Logger | None = None,
    code: str = "operation_failed",
    context: dict[str, Any] | None = None,
) -> str:
    """Return the same structured operation failure as compact JSON text."""

    return public_error_json(exc, logger=logger, code=code, context=context)
