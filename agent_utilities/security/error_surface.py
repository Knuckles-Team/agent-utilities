"""Stable, privacy-safe failures for externally reachable surfaces.

Exception strings are not safe response or log material: HTTP clients, database
drivers, parsers, and subprocess wrappers commonly include credentials,
endpoints, local paths, queries, or payload fragments in them.  This module is
the single public-failure boundary for REST, MCP, and streaming adapters.
"""

from __future__ import annotations

import json
import logging
import threading
import traceback
import uuid
from collections import OrderedDict
from types import MappingProxyType
from typing import Any

from agent_utilities.protocols.epistemic_operations import (
    OperationError,
    OperationResult,
)
from agent_utilities.security.persistence_privacy import sanitize_for_persistence

_DEFAULT_LOGGER = logging.getLogger(__name__)

PUBLIC_ERROR_MESSAGES = MappingProxyType(
    {
        "operation_failed": "The requested operation could not be completed.",
        "invalid_request": "The request could not be processed.",
        "dependency_unavailable": "A required service is unavailable.",
        "permission_denied": "The operation is not authorized.",
    }
)

#: Coarse, deterministic module-prefix -> layer-label mapping used to derive
#: ``failing_layer`` from the exception's own innermost traceback frame — never
#: guessed, always read from where the exception actually originated. Ordered
#: most-specific first so a sub-package (e.g. ``agent_utilities.mcp``) is
#: matched before the generic ``agent_utilities`` fallback.
_FAILING_LAYER_PREFIXES: tuple[tuple[str, str], ...] = (
    ("agent_utilities.mcp", "mcp"),
    ("agent_utilities.gateway", "gateway"),
    ("agent_utilities.orchestration", "orchestration"),
    ("agent_utilities.knowledge_graph", "knowledge_graph"),
    ("agent_utilities.security", "security"),
    ("agent_utilities.protocols", "protocols"),
    ("agent_utilities.server", "server"),
    ("agent_utilities.usage", "usage"),
    ("agent_utilities.observability", "observability"),
    ("agent_utilities.core", "core"),
    ("agent_utilities", "agent_utilities"),
    ("epistemic_graph", "engine"),
)

#: Bounded, process-local, in-memory store of sanitized failure detail keyed by
#: the SAME ``correlation_id`` already logged and returned to the caller as
#: ``error.detail_ref`` — this is the "detail store" behind that reference.
#: Bounded (never durable, never a leak vector: every entry is already
#: sanitized before it is stored) so a busy process cannot grow it unbounded.
_DETAIL_STORE_MAX_ENTRIES = 512
_detail_store: OrderedDict[str, dict[str, Any]] = OrderedDict()
_detail_store_lock = threading.Lock()


def _infer_failing_layer(exc: BaseException) -> str:
    """Best-effort layer label from the exception's own innermost frame.

    Walks the real traceback (never guessed from the exception's class or
    message) to the last frame and maps its module to a coarse, stable layer
    name. Falls back to ``"unknown"`` when the traceback is absent (e.g. an
    exception constructed but never raised) rather than fabricating a layer.
    """

    module_name = ""
    tb = exc.__traceback__
    while tb is not None:
        module_name = tb.tb_frame.f_globals.get("__name__", "") or module_name
        tb = tb.tb_next
    for prefix, layer in _FAILING_LAYER_PREFIXES:
        if module_name == prefix or module_name.startswith(f"{prefix}."):
            return layer
    return "unknown"


def _record_error_detail(
    correlation_id: str, *, error_class: str, failing_layer: str, exc: BaseException
) -> None:
    """Store a sanitized diagnostic bundle resolvable via :func:`resolve_error_detail`.

    The full exception chain (``__cause__``/``__context__``) is rendered via
    :func:`traceback.format_exception` — so a wrapped/re-raised failure keeps
    its real origin — then sanitized through the SAME persistence privacy gate
    (:mod:`agent_utilities.security.persistence_privacy`) used for any other
    payload crossing a persistence/telemetry boundary, before it ever enters
    this process-local store.
    """

    rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    sanitized_traceback, _report = sanitize_for_persistence(rendered)
    with _detail_store_lock:
        _detail_store[correlation_id] = {
            "error_class": error_class,
            "failing_layer": failing_layer,
            "traceback": sanitized_traceback,
        }
        _detail_store.move_to_end(correlation_id)
        while len(_detail_store) > _DETAIL_STORE_MAX_ENTRIES:
            _detail_store.popitem(last=False)


def resolve_error_detail(detail_ref: str) -> dict[str, Any] | None:
    """Resolve a previously recorded, sanitized failure detail by its ``detail_ref``.

    Returns ``None`` for an unknown, expired (evicted), or empty reference —
    never raises, so a stale/foreign ref degrades to "no detail available"
    rather than a second failure.
    """

    if not detail_ref:
        return None
    with _detail_store_lock:
        record = _detail_store.get(detail_ref)
        return dict(record) if record is not None else None


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
    same bare "RuntimeError") even though the PUBLIC payload was diagnostically
    opaque by design (every failure returned the same ``operation_failed``
    code, ``retryable=false``, and a ``detail_ref`` that was ALWAYS ``null`` —
    no error class, no failing layer, no way to resolve more detail). The
    process-wide log-privacy boundary (``core/log_privacy.py``) still
    sanitizes the exception's message text (redacts endpoints/paths/emails)
    before it reaches any handler, so none of this reintroduces a leak.

    The returned dict now also carries, as siblings of the strict
    ``OperationResult`` fields (the shared cross-repo Epistemic Operations
    Protocol schema stays untouched — no engine-side regeneration required):

    * ``error_class`` — ``type(exc).__name__``. Never the exception's message —
      just the class name, exactly as already accepted into logs/caplog by
      the existing exception-surface contract (``test_exception_surface_
      hardening.py`` et al. assert the class name IS logged while the message
      never crosses the boundary); this repeats that same, already-reviewed
      trade-off in the payload instead of inventing a new one.
    * ``failing_layer`` — inferred from the exception's own innermost
      traceback frame (:func:`_infer_failing_layer`), never guessed.

    The exception's MESSAGE itself is never added to this payload — the
    existing static/behavioral gates (``test_exception_surface_hardening.py``,
    ``test_graph_query_sql.py``, ``test_graph_tool_targeting.py``, the
    ``test_served_packages_do_not_expose_raw_exception_text`` static gate)
    already enforce that the public envelope never echoes it, and query
    strings, target names, or validation detail in a message are not reliably
    caught by the persistence-privacy pattern list (it targets known secret/
    PII/location shapes, not arbitrary application text). ``error.detail_ref``
    is the SAME value as ``error.correlation_id`` rather than ``None``:
    :func:`_record_error_detail` stores the sanitized full exception chain
    (``__cause__``/``__context__`` included, via
    :func:`traceback.format_exception`, sanitized through the persistence
    privacy gate) under that key, resolvable with :func:`resolve_error_detail`
    — for an internal/authorized caller, never echoed automatically into an
    external response. A null ``detail_ref`` used to imply "no further detail
    exists"; now it is a real, dereferenceable reference.
    """

    safe_code = code if code in PUBLIC_ERROR_MESSAGES else "operation_failed"
    correlation_id = f"correlation:{uuid.uuid4().hex}"
    error_class = type(exc).__name__
    failing_layer = _infer_failing_layer(exc)
    sink = logger or _DEFAULT_LOGGER
    sink.warning(
        "External operation failed (code=%s correlation_id=%s exception=%s "
        "error_class=%s failing_layer=%s context_keys=%s)",
        safe_code,
        correlation_id,
        exc,
        error_class,
        failing_layer,
        sorted(context) if context else [],
    )
    _record_error_detail(
        correlation_id, error_class=error_class, failing_layer=failing_layer, exc=exc
    )
    payload = OperationResult(
        schema_version="1",
        operation_id=f"operation:{uuid.uuid4().hex}",
        status="failed",
        result_kind=None,
        result_ref=None,
        error=OperationError(
            code=safe_code,
            retryable=safe_code == "dependency_unavailable",
            correlation_id=correlation_id,
            detail_ref=correlation_id,
        ),
        redirect=None,
    ).model_dump(mode="json")
    payload["error_class"] = error_class
    payload["failing_layer"] = failing_layer
    return payload


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
