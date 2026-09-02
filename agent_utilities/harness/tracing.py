"""Native Langfuse Tracing Decorators.

CONCEPT:AU-OS.config.secrets-authentication — Instrumentation Decorators

Provides ``@trace`` and ``@generation`` decorators that emit structured
traces and spans to Langfuse via the batch ingestion API. These decorators
complement the OTel/Logfire auto-instrumentation by adding explicit
application-level tracing with:

- **Parent-child nesting**: Traces contain spans, spans contain generations
- **Session grouping**: Related traces are grouped into Langfuse sessions
- **Structured metadata**: Model names, token counts, tags, environment
- **Error tracking**: Exception details with stack traces

Architecture::

    @trace("my_workflow")           → Creates a top-level Langfuse Trace
    └── @trace("step_1")           → Creates a Span under the parent Trace
        └── @generation("llm")     → Creates a Generation under the Span

Context Propagation::

    The module uses ``contextvars`` to propagate trace/span IDs through
    async call chains, ensuring proper nesting without explicit passing.

Usage::

    from agent_utilities.harness.tracing import trace, generation, get_session_id

    @trace(name="research_pipeline", tags=["research"])
    async def run_research(query: str):
        results = await search(query)
        return await synthesize(results)

    @generation(name="llm_call", model="operator/model-id")
    async def call_llm(prompt: str):
        return await model.generate(prompt)
"""

import contextvars
import functools
import inspect
import logging
import time
import traceback
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from agent_utilities.core.config import config
from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

logger = logging.getLogger(__name__)
_TRACE_PRIVACY = PersistencePrivacyGuard()

# Context variables for trace propagation
_current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_trace_id", default=None
)
_current_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_span_id", default=None
)
_current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_session_id", default=None
)

# Always-on KG-native trace sink (CONCEPT:AU-OS.config.model-factory-passthrough). The daemon/orchestrator injects a
# facade-backed ``KGTraceBackend`` ONCE at startup via ``set_kg_trace_sink``; when set,
# every traced call also persists a Trace/Span/Generation node so traces are
# graph-queryable — independent of any Langfuse key. Left ``None`` in a bare process
# (e.g. a unit import), so there is NO behavior change until a sink is wired.
_kg_trace_sink: Any = None


def set_kg_trace_sink(sink: Any) -> None:
    """Install the always-on KG-native trace sink (a ``KGTraceBackend``). Called once
    by the daemon/orchestrator at startup with a facade-backed backend (OS-5.68)."""
    global _kg_trace_sink
    _kg_trace_sink = sink


def get_kg_trace_sink() -> Any:
    """The installed KG-native trace sink, or ``None`` if none is wired."""
    return _kg_trace_sink


def _tracing_active() -> bool:
    """Trace for an installed KG sink or explicitly authorized Langfuse export.

    Credentials alone never authorize external emission. A bare process with no
    local sink and no authorized export short-circuits with zero overhead.
    """
    return _kg_trace_sink is not None or bool(
        config.trace_export_enabled and config.langfuse_secret_key_ref
    )


_TRACING_MODEL_CLS: Any = None


def _tracing_model_cls() -> Any:
    """Lazily build (once) the WrapperModel subclass that emits a GenerationNode per LLM
    request. WrapperModel is a delegating ``Model`` subclass, so ``isinstance(.., Model)``
    still holds — the safe way to instrument the model without breaking pydantic-ai."""
    global _TRACING_MODEL_CLS
    if _TRACING_MODEL_CLS is not None:
        return _TRACING_MODEL_CLS
    try:
        from pydantic_ai.models.wrapper import WrapperModel
    except Exception:  # pragma: no cover - pydantic-ai optional
        return None

    class _TracingModel(WrapperModel):  # type: ignore[misc, valid-type]
        async def request(self, messages: Any, model_settings: Any, mrp: Any) -> Any:
            t0 = time.time()
            resp = await super().request(messages, model_settings, mrp)
            sink = _kg_trace_sink
            if sink is not None and hasattr(sink, "record_event"):
                try:
                    u = getattr(resp, "usage", None)
                    # CONCEPT:AU-ORCH.optimization.provider-prompt-cache — surface provider prompt-cache
                    # usage on the SAME always-on per-call trace event so cache savings are measurable
                    # without a second capture path.
                    from agent_utilities.caching.prompt_cache import (
                        record_prompt_cache_usage,
                    )

                    cache_read, cache_write = record_prompt_cache_usage(
                        provider=getattr(self, "system", None), usage=u
                    )
                    sink.record_event(
                        trace_id=_current_trace_id.get() or f"trace:{uuid.uuid4()}",
                        span_id=f"gen:{uuid.uuid4()}",
                        name="llm.request",
                        is_root=False,
                        kind="llm",
                        parent_span_id=_current_span_id.get(),
                        model=getattr(self, "model_name", None),
                        input_tokens=int(getattr(u, "input_tokens", 0) or 0),
                        output_tokens=int(getattr(u, "output_tokens", 0) or 0),
                        cache_read_tokens=cache_read,
                        cache_write_tokens=cache_write,
                        latency_ms=(time.time() - t0) * 1000,
                    )
                except Exception as exc:  # pragma: no cover - capture is best-effort
                    # D-SWG-2: loud, not debug — wrap_model_for_tracing's own
                    # docstring promises "EVERY LLM request persists a
                    # GenerationNode"; a silently dropped capture here is the
                    # exact D-DG-7 shape (a tracing write reported nowhere).
                    logger.warning("per-call generation capture failed: %s", exc)
            return resp

    _TRACING_MODEL_CLS = _TracingModel
    return _TRACING_MODEL_CLS


def wrap_model_for_tracing(model: Any) -> Any:
    """Wrap a pydantic-ai ``Model`` so EVERY LLM request persists a ``GenerationNode``
    (model/tokens/cost/latency) to the KG trace sink — the always-on per-call capture
    (CONCEPT:AU-OS.config.model-factory-passthrough). Returns the model UNCHANGED when no sink is installed (zero
    overhead) or WrapperModel is unavailable. Non-streaming requests only; streaming
    delegates untouched."""
    if _kg_trace_sink is None:
        return model
    cls = _tracing_model_cls()
    if cls is None:
        return model
    try:
        return cls(model)
    except Exception as exc:  # pragma: no cover - never break model construction
        # D-SWG-2: loud, not debug — falling back to the unwrapped model means
        # this model's calls are NEVER traced, silently breaking the
        # "EVERY LLM request" guarantee this function's docstring promises.
        logger.warning("model tracing wrap failed: %s", exc)
        return model


def set_session_id(session_id: str) -> None:
    """Set the current Langfuse session ID for trace grouping.

    CONCEPT:AU-OS.config.secrets-authentication — Session Management

    All traces emitted within this context will be grouped under
    the given session ID in Langfuse.

    Args:
        session_id: Unique session identifier for grouping related traces.
    """
    _current_session_id.set(session_id)


def get_session_id() -> str | None:
    """Get the current Langfuse session ID.

    Returns:
        Current session ID or None if not set.
    """
    return _current_session_id.get()


def get_trace_id() -> str | None:
    """Get the current Langfuse trace ID from context.

    Returns:
        Current trace ID or None if not in a traced context.
    """
    return _current_trace_id.get()


@contextmanager
def _trace_context(
    func: Callable,
    *,
    name: str | None,
    trace_type: str,
    tags: list[str] | None,
    metadata: dict[str, Any] | None,
    session_id: str | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
):
    """Set trace context and emit the completed call for either wrapper shape."""
    parent_trace_id = _current_trace_id.get()
    parent_span_id = _current_span_id.get()
    current_session = session_id or _current_session_id.get()

    trace_id = parent_trace_id or str(uuid.uuid4())
    span_id = str(uuid.uuid4())
    span_name = name or func.__name__

    token_trace = _current_trace_id.set(trace_id)
    token_span = _current_span_id.set(span_id)

    start_time = time.time()
    start_iso = _iso_timestamp(start_time)
    result: Any = None

    def capture_result(value: Any) -> Any:
        nonlocal result
        result = value
        return value

    output_data: Any = None
    level = "DEFAULT"
    status_message: str | None = None
    end_iso = start_iso

    def emit() -> None:
        _emit_trace(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=span_name,
            trace_type=trace_type if parent_trace_id else "trace-create",
            start_time=start_iso,
            end_time=end_iso,
            input_data=_safe_serialize({"args": args, "kwargs": kwargs}),
            output_data=output_data,
            level=level,
            status_message=status_message,
            tags=tags,
            metadata=metadata,
            session_id=current_session,
            is_root=not parent_trace_id,
        )

    try:
        yield capture_result
        end_iso = _iso_timestamp(time.time())
        output_data = _safe_serialize(result)
        emit()
    except Exception as e:
        end_iso = _iso_timestamp(time.time())
        level = "ERROR"
        status_message = str(e)
        output_data = {"error": str(e), "traceback": traceback.format_exc()}
        emit()
        raise
    finally:
        _current_trace_id.reset(token_trace)
        _current_span_id.reset(token_span)


def trace(
    name: str | None = None,
    trace_type: str = "SPAN",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
):
    """Decorator for native Langfuse tracing with proper nesting.

    CONCEPT:AU-OS.config.secrets-authentication — Trace Instrumentation

    If a parent trace exists in context, this creates a child span.
    Otherwise, it creates a new top-level trace. This enables automatic
    nesting of traces across async call chains.

    Args:
        name: Name of the trace/span. Defaults to the function name.
        trace_type: Langfuse event type (``SPAN``, ``GENERATION``, ``EVENT``).
        tags: Optional tags for filtering in Langfuse UI.
        metadata: Optional key-value metadata attached to the trace.
        session_id: Optional session ID override for this trace.

    Example::

        @trace(name="agent_execution", tags=["live", "orchestration"])
        async def run_agent(task: str):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _tracing_active():
                return func(*args, **kwargs)
            with _trace_context(
                func,
                name=name,
                trace_type=trace_type,
                tags=tags,
                metadata=metadata,
                session_id=session_id,
                args=args,
                kwargs=kwargs,
            ) as capture_result:
                return capture_result(func(*args, **kwargs))

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _tracing_active():
                return await func(*args, **kwargs)
            with _trace_context(
                func,
                name=name,
                trace_type=trace_type,
                tags=tags,
                metadata=metadata,
                session_id=session_id,
                args=args,
                kwargs=kwargs,
            ) as capture_result:
                return capture_result(await func(*args, **kwargs))

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def generation(
    name: str | None = None,
    model: str | None = None,
    tags: list[str] | None = None,
):
    """Decorator for LLM generation tracing.

    CONCEPT:AU-OS.config.secrets-authentication — Generation Tracing

    Creates a ``generation-create`` event in Langfuse that tracks
    model name, token usage, and latency for LLM calls.

    Args:
        name: Name of the generation. Defaults to function name.
        model: LLM model identifier (e.g. ``qwen3.6-27b``).
        tags: Optional tags for Langfuse filtering.
    """
    return trace(
        name=name,
        trace_type="generation-create",
        tags=tags,
        metadata={"model": model} if model else None,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iso_timestamp(ts: float) -> str:
    """Convert a Unix timestamp to ISO 8601 format for Langfuse."""
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(ts))


def _safe_serialize(data: Any, max_length: int = 10000) -> Any:
    """Safely serialize data for Langfuse, truncating large payloads.

    CONCEPT:AU-OS.config.secrets-authentication — Safe Serialization

    Prevents trace ingestion failures from non-serializable or
    excessively large payloads.
    """
    clean, report = _TRACE_PRIVACY.sanitize(data)
    if clean is None:
        return None
    try:
        import json

        serialized = json.dumps(clean, default=str)
        if len(serialized) > max_length:
            return {
                "_truncated": True,
                "preview": serialized[:max_length],
                "privacy": report.as_dict(),
            }
        if report.changed and isinstance(clean, dict):
            clean = dict(clean)
            clean.setdefault("_privacy", report.as_dict())
        return clean
    except (TypeError, ValueError):
        return {
            "_type": type(data).__name__,
            "_privacy": report.as_dict(),
            "value": "[REDACTED_UNSERIALIZABLE]",
        }


def _identifiers_are_safe(
    trace_id: str, span_id: str, parent_span_id: str | None
) -> bool:
    """Identifiers are join keys and cannot be safely rewritten without breaking
    the trace graph. Refuse persistence when the privacy guard detects personal
    or machine-specific content in one of them."""
    for identifier in (trace_id, span_id, parent_span_id):
        if identifier is None:
            continue
        _, identifier_report = _TRACE_PRIVACY.sanitize_text(identifier)
        if identifier_report.changed:
            logger.debug("Trace emission skipped: unsafe identifier")
            return False
    return True


def _sanitize_trace_fields(
    name: Any,
    input_data: Any,
    output_data: Any,
    metadata: dict[str, Any] | None,
    status_message: str | None,
    tags: list[str] | None,
    session_id: str | None,
) -> tuple[str, Any, Any, dict[str, Any], str | None, list[str], str | None]:
    name = str(_safe_serialize(name))
    input_data = _safe_serialize(input_data)
    output_data = _safe_serialize(output_data)
    metadata = _safe_serialize(metadata or {})
    if not isinstance(metadata, dict):
        metadata = {"value": metadata}
    status_message = (
        str(_safe_serialize(status_message)) if status_message is not None else None
    )
    tags = [str(_safe_serialize(tag)) for tag in (tags or [])]
    session_id = str(_safe_serialize(session_id)) if session_id is not None else None
    return name, input_data, output_data, metadata, status_message, tags, session_id


def _export_metadata_and_status(
    metadata: dict[str, Any],
    status_message: str | None,
    level: str,
    capture_content: bool,
) -> tuple[dict[str, Any], str | None]:
    export_metadata = metadata
    if not capture_content:
        export_metadata = {
            key: metadata[key]
            for key in (
                "model",
                "provider",
                "input_tokens",
                "output_tokens",
            )
            if key in metadata
        }
        export_metadata["content_retention"] = "metadata"
    exported_status_message = status_message if capture_content else None
    if level == "ERROR" and status_message and not capture_content:
        exported_status_message = "error"
    return export_metadata, exported_status_message


def _kg_sink_event_kwargs(event: dict[str, Any]) -> dict[str, Any]:
    md = event["metadata"]
    kind = "llm" if "generation" in (event["trace_type"] or "").lower() else "general"
    return {
        "trace_id": event["trace_id"],
        "span_id": event["span_id"],
        "name": event["name"],
        "is_root": event["is_root"],
        "kind": kind,
        "parent_span_id": event["parent_span_id"],
        "session_id": event["session_id"],
        "error": (
            event["exported_status_message"] if event["level"] == "ERROR" else None
        ),
        "model": md.get("model"),
        "provider": md.get("provider"),
        "input_tokens": int(md.get("input_tokens", 0) or 0),
        "output_tokens": int(md.get("output_tokens", 0) or 0),
        "tags": event["tags"],
        # Content retention is a single explicit opt-in across both sinks.
        # Metadata-only mode must not persist arguments or results locally
        # while the optional external fan-out remains disabled by default.
        "input_text": (
            str(event["input_data"])[:4000]
            if event["is_root"] and event["capture_content"]
            else ""
        ),
        "output_text": (
            str(event["output_data"])[:4000]
            if event["is_root"] and event["capture_content"]
            else ""
        ),
    }


def _emit_kg_sink_event(event: dict[str, Any]) -> None:
    """Always-on KG-native sink (CONCEPT:AU-OS.config.model-factory-passthrough):
    persist a Trace/Span/Generation node independent of Langfuse, so every traced
    call is graph-queryable. Best-effort — a sink failure never breaks the traced
    function."""
    sink = _kg_trace_sink
    if sink is None or not hasattr(sink, "record_event"):
        return
    try:
        sink.record_event(**_kg_sink_event_kwargs(event))
    except Exception as e:  # pragma: no cover - tracing must never break callers
        logger.debug("KG trace emit failed (%s)", type(e).__name__)


def _build_langfuse_batch(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the Langfuse ingestion batch: a root call emits a ``trace-create``
    event first; every call (except a root ``trace-create``, which needs no
    extra span) also emits its span/generation event."""
    batch: list[dict[str, Any]] = []

    if event["is_root"]:
        # Create the parent trace first
        trace_body: dict[str, Any] = {
            "id": event["trace_id"],
            "name": event["name"],
            "metadata": event["export_metadata"],
            "tags": event["tags"],
        }
        if event["capture_content"]:
            trace_body.update(
                {"input": event["input_data"], "output": event["output_data"]}
            )
        trace_event: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "type": "trace-create",
            "timestamp": event["start_time"],
            "body": trace_body,
        }
        if event["session_id"]:
            trace_event["body"]["sessionId"] = event["session_id"]
        batch.append(trace_event)

    # Create the span/generation under the trace
    actual_type = event["trace_type"] if not event["is_root"] else "span-create"
    if event["is_root"] and event["trace_type"] == "trace-create":
        # Root traces don't need an additional span
        pass
    else:
        span_body: dict[str, Any] = {
            "id": event["span_id"],
            "traceId": event["trace_id"],
            "name": event["name"],
            "startTime": event["start_time"],
            "endTime": event["end_time"],
            "level": event["level"],
            "metadata": event["export_metadata"],
        }
        if event["capture_content"]:
            span_body.update(
                {"input": event["input_data"], "output": event["output_data"]}
            )
        span_event: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "type": actual_type,
            "timestamp": event["start_time"],
            "body": span_body,
        }
        if event["parent_span_id"]:
            span_event["body"]["parentObservationId"] = event["parent_span_id"]
        if event["exported_status_message"]:
            span_event["body"]["statusMessage"] = event["exported_status_message"]
        batch.append(span_event)

    return batch


def _emit_langfuse_trace(event: dict[str, Any]) -> None:
    # Credentials identify the destination; TRACE_EXPORT_ENABLED separately
    # authorizes external emission. Neither one may implicitly enable the other.
    if not config.trace_export_enabled or not config.langfuse_secret_key_ref:
        return
    try:
        from agent_utilities.harness.trace_backend import (
            LangfuseTraceBackend,
            create_trace_backend,
        )

        backend = create_trace_backend(backend_type="langfuse")
        if not isinstance(backend, LangfuseTraceBackend):
            return

        api = backend._get_api()
        batch = _build_langfuse_batch(event)

        if batch:
            api.ingestion_batch(batch=batch)

    except Exception as e:
        logger.debug("Failed to emit Langfuse trace (%s).", type(e).__name__)


def _emit_trace(
    trace_id: str,
    span_id: str,
    parent_span_id: str | None,
    name: str,
    trace_type: str,
    start_time: str,
    end_time: str,
    input_data: Any,
    output_data: Any,
    level: str = "DEFAULT",
    status_message: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    is_root: bool = False,
) -> None:
    """Emit a trace event to Langfuse via the batch ingestion API.

    CONCEPT:AU-OS.config.secrets-authentication — Trace Emission

    Creates properly nested trace→span hierarchy in Langfuse:
    - Root calls create ``trace-create`` events
    - Child calls create ``span-create`` events linked to the parent

    Args:
        trace_id: The top-level trace identifier.
        span_id: This span's unique identifier.
        parent_span_id: Parent span ID for nesting (None for root).
        name: Display name in Langfuse UI.
        trace_type: Event type (``trace-create``, ``span-create``, ``generation-create``).
        start_time: ISO 8601 start timestamp.
        end_time: ISO 8601 end timestamp.
        input_data: Input payload (serializable).
        output_data: Output payload (serializable).
        level: Log level (``DEFAULT``, ``WARNING``, ``ERROR``).
        status_message: Optional status/error message.
        tags: Optional tags for Langfuse filtering.
        metadata: Optional structured metadata.
        session_id: Optional session ID for grouping.
        is_root: Whether this is a root trace (vs child span).
    """
    if not _identifiers_are_safe(trace_id, span_id, parent_span_id):
        return

    name, input_data, output_data, metadata, status_message, tags, session_id = (
        _sanitize_trace_fields(
            name, input_data, output_data, metadata, status_message, tags, session_id
        )
    )

    capture_content = bool(config.langfuse_capture_content)
    export_metadata, exported_status_message = _export_metadata_and_status(
        metadata, status_message, level, capture_content
    )

    event = {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "is_root": is_root,
        "trace_type": trace_type,
        "start_time": start_time,
        "end_time": end_time,
        "level": level,
        "metadata": metadata,
        "export_metadata": export_metadata,
        "exported_status_message": exported_status_message,
        "tags": tags,
        "session_id": session_id,
        "input_data": input_data,
        "output_data": output_data,
        "capture_content": capture_content,
    }
    _emit_kg_sink_event(event)
    _emit_langfuse_trace(event)
