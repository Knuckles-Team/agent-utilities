"""Cross-agent trace correlation (CONCEPT:OS-5.11).

The ``@trace`` decorators in :mod:`agent_utilities.harness.tracing` already nest
spans within a single process via ``contextvars``. This module closes the
*cross-agent* gap so a multi-agent run is one correlated story end to end:

* a stable **correlation id** that ties every agent, span, and side-effect in a
  run together — the key that answers "which agents touched record X?";
* **W3C trace-context** (``traceparent``) (de)serialization so a spawned agent
  in another process/worker links its spans under the parent's trace;
* **header injection** so outbound side-effects (Kafka events, ServiceNow /
  connector calls) carry ``traceparent`` + ``x-correlation-id``, making the
  external write joinable back to the agent that caused it.

It builds on the existing trace/span contextvars rather than duplicating them.
"""

from __future__ import annotations

import contextvars
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from agent_utilities.harness import tracing

# Stable, run-wide correlation id (survives nested agent spawns within a run).
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_correlation_id", default=None
)

TRACEPARENT_HEADER = "traceparent"
CORRELATION_HEADER = "x-correlation-id"
SESSION_HEADER = "x-session-id"


def _hex(n: int) -> str:
    return uuid.uuid4().hex[:n].ljust(n, "0")


def get_correlation_id() -> str | None:
    """Return the current run-wide correlation id, if any."""
    return _correlation_id.get()


def ensure_correlation_id() -> str:
    """Return the current correlation id, creating one if absent.

    Defaults to the active root trace id (so the correlation id and the Langfuse
    trace share a value when possible), else a fresh id.
    """
    cid = _correlation_id.get()
    if cid:
        return cid
    cid = tracing.get_trace_id() or uuid.uuid4().hex
    _correlation_id.set(cid)
    return cid


def _format_traceparent(trace_id: str | None, span_id: str | None) -> str | None:
    """Render a W3C ``traceparent`` (version-00) from the active trace/span ids."""
    if not trace_id:
        return None
    # W3C requires 32-hex trace-id and 16-hex parent-id; derive deterministically.
    t = uuid.uuid5(uuid.NAMESPACE_OID, trace_id).hex  # 32 hex
    s = uuid.uuid5(uuid.NAMESPACE_OID, span_id or trace_id).hex[:16]
    return f"00-{t}-{s}-01"


def current_carrier() -> dict[str, str]:
    """Serialize the active trace context for handing to a spawned agent.

    Returns a flat string->string carrier ({traceparent, x-correlation-id,
    x-session-id}) safe to pass over A2A / MCP / HTTP to a child agent, which
    restores it via :func:`bind_carrier`.
    """
    carrier: dict[str, str] = {CORRELATION_HEADER: ensure_correlation_id()}
    tp = _format_traceparent(tracing.get_trace_id(), tracing._current_span_id.get())
    if tp:
        carrier[TRACEPARENT_HEADER] = tp
    sid = tracing.get_session_id()
    if sid:
        carrier[SESSION_HEADER] = sid
    return carrier


@contextmanager
def bind_carrier(carrier: dict[str, str] | None) -> Iterator[str]:
    """Restore a parent's trace context in a child agent for the block's duration.

    Sets the correlation id (and session id) so the child's traces and
    side-effects join the parent's story, then restores the prior context on
    exit. Yields the effective correlation id.
    """
    carrier = carrier or {}
    tokens: list[tuple[contextvars.ContextVar, Any]] = []
    cid = carrier.get(CORRELATION_HEADER) or ensure_correlation_id()
    tokens.append((_correlation_id, _correlation_id.set(cid)))

    sid = carrier.get(SESSION_HEADER)
    if sid:
        tokens.append((tracing._current_session_id, tracing._current_session_id.set(sid)))

    try:
        yield cid
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def inject(headers: dict[str, str] | None = None) -> dict[str, str]:
    """Add correlation headers to an outbound side-effect's header/metadata dict.

    Use for Kafka record headers, ServiceNow / connector HTTP calls, etc. so the
    external write is joinable back to the originating agent run.
    """
    headers = dict(headers or {})
    headers[CORRELATION_HEADER] = ensure_correlation_id()
    tp = _format_traceparent(tracing.get_trace_id(), tracing._current_span_id.get())
    if tp:
        headers[TRACEPARENT_HEADER] = tp
    sid = tracing.get_session_id()
    if sid:
        headers[SESSION_HEADER] = sid
    return headers


def extract(headers: dict[str, str] | None) -> dict[str, str]:
    """Pull the correlation carrier out of inbound headers (inverse of inject)."""
    headers = headers or {}
    # Case-insensitive lookup for HTTP-style headers.
    lower = {k.lower(): v for k, v in headers.items()}
    carrier: dict[str, str] = {}
    for key in (CORRELATION_HEADER, TRACEPARENT_HEADER, SESSION_HEADER):
        if key in lower:
            carrier[key] = lower[key]
    return carrier
