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
) -> dict[str, Any]:
    """Return a strict ``OperationResult`` failure and a type-only correlation log.

    Unknown codes fail closed to ``operation_failed``.  Callers cannot supply a
    log message or public message, which prevents an untrusted value from being
    reintroduced accidentally at a call site.
    """

    safe_code = code if code in PUBLIC_ERROR_MESSAGES else "operation_failed"
    correlation_id = f"correlation:{uuid.uuid4().hex}"
    sink = logger or _DEFAULT_LOGGER
    sink.warning(
        "External operation failed (code=%s correlation_id=%s exception_type=%s)",
        safe_code,
        correlation_id,
        type(exc).__name__,
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
) -> str:
    """Serialize :func:`public_error_payload` for string-returning MCP tools."""

    payload = public_error_payload(exc, logger=logger, code=code)
    return json.dumps(payload)


def public_error_text(
    exc: BaseException,
    *,
    logger: logging.Logger | None = None,
    code: str = "operation_failed",
) -> str:
    """Return the same structured operation failure as compact JSON text."""

    return public_error_json(exc, logger=logger, code=code)
