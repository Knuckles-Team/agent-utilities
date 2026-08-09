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
from agent_utilities.security.brain_context import ActorContext, current_actor
from agent_utilities.security.persistence_privacy import sanitize_for_persistence

_DEFAULT_LOGGER = logging.getLogger(__name__)

PUBLIC_ERROR_MESSAGES = MappingProxyType(
    {
        "operation_failed": "The requested operation could not be completed.",
        "invalid_request": "The request could not be processed.",
        "dependency_unavailable": "A required service is unavailable.",
        "permission_denied": "The operation is not authorized.",
        # BUG-048: a breaker-open / transport-down failure is transient and
        # retryable — distinct from "operation_failed" (which reads as a
        # rejected/malformed request) and from "dependency_unavailable"
        # (which today means "could not even obtain an engine handle").
        "engine_degraded": (
            "The knowledge graph engine is temporarily overloaded; retry shortly."
        ),
    }
)

#: Exception CLASS NAMES (never instances or messages) that mean "the engine
#: circuit breaker tripped or the transport is down" — a transient condition,
#: not a rejected/malformed request (BUG-048). Mirrors
#: ``agent_utilities.knowledge_graph.core.engine_breaker``'s own trip/retry
#: vocabulary (``_TRIP_EXCEPTIONS`` / ``_RETRY_EXCEPTIONS`` plus its typed
#: ``EngineCircuitOpenError``, itself a ``ConnectionError`` subclass),
#: duplicated here BY NAME rather than imported, so this module (the public
#: sanitization boundary) does not take on a security -> knowledge_graph
#: layering dependency.
_ENGINE_DEGRADED_TYPE_NAMES = frozenset(
    {"EngineCircuitOpenError", "ConnectionError", "TimeoutError", "OSError", "EOFError"}
)


def _is_engine_degraded(exc: BaseException) -> bool:
    """True when ``exc`` is a breaker-open / transport-down condition.

    Two shapes reach this boundary:

    1. The breaker's own :class:`EngineCircuitOpenError` (or a bare
       transport ``OSError``/``EOFError`` — ``ConnectionError`` and
       ``TimeoutError`` are both ``OSError`` SIBLINGS, not one a subclass of
       the other, so both must be checked explicitly) raised directly. This
       is exactly the breaker's own ``_TRIP_EXCEPTIONS = (OSError, EOFError)``
       vocabulary (:mod:`agent_utilities.knowledge_graph.core.engine_breaker`)
       — anything that trips the breaker is, by definition, this condition.
    2. The native-Cypher sanitization boundary's ``CypherEngineError``
       (:mod:`agent_utilities.knowledge_graph.backends.epistemic_graph_backend`),
       which wraps ``from None`` — deliberately discarding
       ``__cause__``/``__context__`` so the original query/connection detail
       never crosses the boundary — but preserves the ORIGINAL exception's
       class NAME on its own ``error_type`` attribute for exactly this
       purpose: telling a transient breaker/connection failure apart from a
       genuine query rejection without resurrecting the sanitized detail.
       Checked by class name only (never message text), so this stays a
       class-level distinction — sanitizing the detail, not the class.
    """
    if isinstance(exc, (OSError, EOFError)):
        return True
    error_type = getattr(exc, "error_type", None)
    return isinstance(error_type, str) and error_type in _ENGINE_DEGRADED_TYPE_NAMES


def _engine_retry_after_hint() -> float:
    """Best-effort ``retry_after_s`` hint for an ``engine_degraded`` failure.

    Sourced from the SAME breaker cooldown configuration
    :class:`~agent_utilities.knowledge_graph.core.engine_breaker.CircuitBreaker`
    itself uses, so the hint is never fabricated — falls back to the
    breaker's own default cooldown only when the config isn't reachable
    here, and never raises (a hint must not break the error path it decorates).
    """
    try:
        from agent_utilities.core.config import config

        return float(config.engine_breaker_cooldown)
    except Exception:  # noqa: BLE001 — a retry hint must never break error reporting
        return 15.0  # CircuitBreaker's own default cooldown


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
#: This remains the primary, fastest-path source of truth for
#: :func:`resolve_error_detail` (checked first); a registered
#: :data:`_detail_persistence_sink` with a ``resolve`` method (D-24) is
#: consulted only on a miss here — e.g. after a restart, or a resolve call
#: served by a different replica than the one that recorded the failure. See
#: :func:`register_detail_persistence_sink` for the durable-audit seam and
#: ``agent_utilities.observability.error_detail_sink.GraphErrorDetailSink``
#: for the concrete KG-provenance backend wired through it.
_DETAIL_STORE_MAX_ENTRIES = 512
_detail_store: OrderedDict[str, dict[str, Any]] = OrderedDict()
_detail_store_lock = threading.Lock()

#: De-opinionated durability seam (CONCEPT:AU-KG.audit.error-detail-resolution):
#: error_surface takes no position on the durable backend — a KG-provenance
#: writer, a database, or nothing at all (the default, ``None``). When set,
#: every recorded detail is ALSO handed to the sink, best-effort: a failing or
#: unset sink never affects the in-process store or the caller's exception
#: path (see :func:`_record_error_detail`). A sink that additionally exposes a
#: ``resolve(correlation_id) -> dict | None`` method (duck-typed, not part of
#: the required write contract) is also used as a read-back fallback by
#: :func:`resolve_error_detail` (CONCEPT:AU-KG.audit.durable-error-detail).
DetailPersistenceSink = Any  # Callable[[str, dict[str, Any]], None]
_detail_persistence_sink: Any | None = None


def register_detail_persistence_sink(sink: Any | None) -> None:
    """Register (or clear, with ``None``) a durable sink for recorded error detail.

    The sink is called as ``sink(correlation_id, record)`` with the SAME
    sanitized bundle stored in-process. It is invoked best-effort: an
    exception raised by the sink is logged (with its real cause, never
    swallowed silently) and never propagated, so a broken durable backend can
    never break the error-reporting path it is meant to observe. There is no
    default implementation baked into this module — wiring a concrete backend
    (``agent_utilities.observability.error_detail_sink.GraphErrorDetailSink``,
    a KG-provenance writer linking the detail to the failing operation's
    ``:RunTrace``) is left to the process that wants durability past this
    process's lifetime; see ``agent_utilities.gateway.daemon.start_host_daemon``
    for where the host daemon installs it.

    If ``sink`` also defines ``resolve(correlation_id) -> dict | None``,
    :func:`resolve_error_detail` uses it as a fallback when the in-process
    store misses (D-24) — this is optional and duck-typed, so a plain write
    callable (as every existing test/consumer already passes) remains valid
    with no read-back capability.
    """
    global _detail_persistence_sink
    _detail_persistence_sink = sink


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


def _recording_actor_tenant() -> str:
    """Best-effort tenant scope for the failure being recorded.

    Most ``public_error_payload`` call sites run inside a verified MCP tool
    dispatch, where :func:`current_actor` already resolves (the
    ``ActorContextMiddleware`` / ``verified_tool_session_scope`` boundary binds
    it for the whole call). Some do not (a unit test, a widget fetch outside
    any served request) — those failures are recorded with an empty tenant
    scope rather than raising, since ``public_error_payload`` itself must
    never fail because diagnostics couldn't determine who was calling.
    """

    try:
        return str(current_actor().tenant_id or "").strip()
    except Exception as exc:  # noqa: BLE001 — diagnostics must never fail the error path
        # Logged with its REAL cause (not just the type name) so an unexpected
        # identity-layer failure here is still diagnosable, at debug level because
        # the ordinary "no actor bound outside a served request" case is expected
        # and would otherwise be pure noise.
        _DEFAULT_LOGGER.debug(
            "error-detail tenant scope unresolved; recording with empty scope: %s", exc
        )
        return ""


def _record_error_detail(
    correlation_id: str, *, error_class: str, failing_layer: str, exc: BaseException
) -> None:
    """Store a sanitized diagnostic bundle resolvable via :func:`resolve_error_detail`.

    The full exception chain (``__cause__``/``__context__``) is rendered via
    :func:`traceback.format_exception` — so a wrapped/re-raised failure keeps
    its real origin — then sanitized through the SAME persistence privacy gate
    (:mod:`agent_utilities.security.persistence_privacy`) used for any other
    payload crossing a persistence/telemetry boundary, before it ever enters
    this process-local store. The recording actor's ``tenant_id`` is captured
    alongside it so :func:`resolve_error_detail_for_actor` can enforce that
    only a caller in the SAME authority scope may resolve it.
    """

    rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    sanitized_traceback, _report = sanitize_for_persistence(rendered)
    record = {
        "error_class": error_class,
        "failing_layer": failing_layer,
        "traceback": sanitized_traceback,
        "tenant_id": _recording_actor_tenant(),
    }
    with _detail_store_lock:
        _detail_store[correlation_id] = record
        _detail_store.move_to_end(correlation_id)
        while len(_detail_store) > _DETAIL_STORE_MAX_ENTRIES:
            _detail_store.popitem(last=False)

    sink = _detail_persistence_sink
    if sink is not None:
        try:
            sink(correlation_id, dict(record))
        except Exception:
            # A durable-audit sink is an OPTIONAL hardening seam (D-24, see
            # register_detail_persistence_sink) — it must never turn a
            # diagnostics side-channel into a second failure for the caller
            # whose original exception is already being reported. Logged with
            # the real cause (never swallowed silently), just not re-raised.
            _DEFAULT_LOGGER.warning(
                "detail persistence sink failed for correlation_id=%s",
                correlation_id,
                exc_info=True,
            )


def resolve_error_detail(detail_ref: str) -> dict[str, Any] | None:
    """Resolve a previously recorded, sanitized failure detail by its ``detail_ref``.

    Returns ``None`` for an unknown, expired (evicted), or empty reference —
    never raises, so a stale/foreign ref degrades to "no detail available"
    rather than a second failure.

    Checks the bounded in-process store first (the common, fastest case);
    on a miss, falls back to the registered durable sink's ``resolve`` method
    when it defines one (D-24) — so a ``detail_ref`` recorded before a restart,
    or by a different replica, stays resolvable. The fallback is best-effort:
    a broken/unavailable durable backend degrades to the same "no detail
    available" result as an unknown ref, never a second failure.

    This is the unscoped, in-process/durable lookup used internally (and by
    tests). A served surface reachable by an external caller MUST use
    :func:`resolve_error_detail_for_actor` instead, which enforces the tenant
    boundary below.
    """

    if not detail_ref:
        return None
    with _detail_store_lock:
        record = _detail_store.get(detail_ref)
        if record is not None:
            return dict(record)

    sink = _detail_persistence_sink
    resolver = getattr(sink, "resolve", None)
    if not callable(resolver):
        return None
    try:
        durable_record = resolver(detail_ref)
    except Exception:  # noqa: BLE001 — a durable-backend failure must not break resolution
        _DEFAULT_LOGGER.warning(
            "durable detail persistence sink resolve failed for correlation_id=%s",
            detail_ref,
            exc_info=True,
        )
        return None
    return dict(durable_record) if isinstance(durable_record, dict) else None


def resolve_error_detail_for_actor(
    detail_ref: str, *, actor: ActorContext | None = None
) -> dict[str, Any] | None:
    """Resolve ``detail_ref`` scoped to the caller's own verified tenant authority.

    Fails closed exactly like the existing read-path guard
    (:func:`agent_utilities.knowledge_graph.core.secured_reads._verified_actor`):
    an unauthenticated caller, or one with no ``tenant_id``, raises
    ``PermissionError`` rather than resolving anything — this reuses the SAME
    tenancy model every other served read already enforces, not a new scheme.

    A ref that does not exist and a ref that exists but belongs to a
    DIFFERENT tenant are deliberately indistinguishable from the caller's
    point of view (both return ``None``): confirming the latter would leak
    that a foreign-tenant failure occurred.
    """

    resolved_actor = actor if actor is not None else current_actor()
    resolved_actor.ensure_credential_current()
    caller_tenant = str(getattr(resolved_actor, "tenant_id", "") or "").strip()
    if not getattr(resolved_actor, "authenticated", False) or not caller_tenant:
        raise PermissionError(
            "Error-detail resolution requires verified tenant authority"
        )

    record = resolve_error_detail(detail_ref)
    if record is None:
        return None
    if str(record.get("tenant_id", "") or "").strip() != caller_tenant:
        return None
    return record


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
    privacy gate) under that key. An external caller resolves it through the
    ``graph_observe`` MCP tool's ``error_detail`` action (D-23), which calls
    :func:`resolve_error_detail_for_actor` — never echoed automatically into
    this response, and only resolvable by a caller in the SAME tenant scope
    that recorded it. A null ``detail_ref`` used to imply "no further detail
    exists"; now it is a real, dereferenceable reference.
    """

    safe_code = code if code in PUBLIC_ERROR_MESSAGES else "operation_failed"
    # BUG-048: a caller that didn't explicitly classify its failure (the
    # ubiquitous bare ``except Exception: return public_error_json(e)`` at
    # ~200+ call sites) previously collapsed a breaker-open/transport-down
    # condition into the SAME "operation_failed"/``retryable=False`` shape as
    # a rejected or malformed query — indistinguishable to the caller, who
    # then chases a query-correctness hypothesis instead of retrying. Only
    # the ambiguous DEFAULT is upgraded here; an explicit
    # ``permission_denied``/``invalid_request``/``dependency_unavailable``
    # the caller deliberately chose is never overridden.
    if safe_code == "operation_failed" and _is_engine_degraded(exc):
        safe_code = "engine_degraded"
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
            # BUG-048: a breaker trip IS the definition of retryable — the
            # breaker exists to fail fast now and recover on its own after
            # ``cooldown``, not to reject the request outright.
            retryable=safe_code in ("dependency_unavailable", "engine_degraded"),
            correlation_id=correlation_id,
            detail_ref=correlation_id,
        ),
        redirect=None,
    ).model_dump(mode="json")
    payload["error_class"] = error_class
    payload["failing_layer"] = failing_layer
    if safe_code == "engine_degraded":
        payload["retry_after_s"] = _engine_retry_after_hint()
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
