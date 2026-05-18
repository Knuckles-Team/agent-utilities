import functools
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from agent_utilities.core.config import config

logger = logging.getLogger(__name__)


def trace(name: str | None = None, trace_type: str = "SPAN"):
    """
    Decorator for native Langfuse tracing.

    If Langfuse credentials are set, this will batch and emit a trace
    using the LangfuseTraceBackend via the LangfuseApi.

    Args:
        name: Name of the trace/span. Defaults to the function name.
        trace_type: The type of trace ('SPAN', 'GENERATION', 'EVENT', etc.).
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not config.langfuse_secret_key:
                # No-op if Langfuse is not configured
                return func(*args, **kwargs)

            trace_id = str(uuid.uuid4())
            start_time = time.time()
            start_time_iso = time.strftime(
                "%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(start_time)
            )
            span_name = name or func.__name__

            try:
                result = func(*args, **kwargs)
                _emit_trace(
                    trace_id=trace_id,
                    name=span_name,
                    trace_type=trace_type,
                    start_time=start_time_iso,
                    end_time=time.strftime(
                        "%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(time.time())
                    ),
                    input_data={"args": args, "kwargs": kwargs},
                    output_data=result,
                    level="DEFAULT",
                )
                return result
            except Exception as e:
                _emit_trace(
                    trace_id=trace_id,
                    name=span_name,
                    trace_type=trace_type,
                    start_time=start_time_iso,
                    end_time=time.strftime(
                        "%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(time.time())
                    ),
                    input_data={"args": args, "kwargs": kwargs},
                    output_data={"error": str(e)},
                    level="ERROR",
                    status_message=str(e),
                )
                raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not config.langfuse_secret_key:
                # No-op if Langfuse is not configured
                return await func(*args, **kwargs)

            trace_id = str(uuid.uuid4())
            start_time = time.time()
            start_time_iso = time.strftime(
                "%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(start_time)
            )
            span_name = name or func.__name__

            try:
                result = await func(*args, **kwargs)
                _emit_trace(
                    trace_id=trace_id,
                    name=span_name,
                    trace_type=trace_type,
                    start_time=start_time_iso,
                    end_time=time.strftime(
                        "%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(time.time())
                    ),
                    input_data={"args": args, "kwargs": kwargs},
                    output_data=result,
                    level="DEFAULT",
                )
                return result
            except Exception as e:
                _emit_trace(
                    trace_id=trace_id,
                    name=span_name,
                    trace_type=trace_type,
                    start_time=start_time_iso,
                    end_time=time.strftime(
                        "%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(time.time())
                    ),
                    input_data={"args": args, "kwargs": kwargs},
                    output_data={"error": str(e)},
                    level="ERROR",
                    status_message=str(e),
                )
                raise

        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _emit_trace(
    trace_id: str,
    name: str,
    trace_type: str,
    start_time: str,
    end_time: str,
    input_data: Any,
    output_data: Any,
    level: str = "DEFAULT",
    status_message: str | None = None,
) -> None:
    """Emits a trace using the LangfuseApi batch endpoint."""
    try:
        from agent_utilities.harness.trace_backend import (
            LangfuseTraceBackend,
            create_trace_backend,
        )

        backend = create_trace_backend(backend_type="langfuse")
        if isinstance(backend, LangfuseTraceBackend):
            api = backend._get_api()
            event: dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "type": trace_type,
                "timestamp": start_time,
                "body": {
                    "id": trace_id,
                    "name": name,
                    "startTime": start_time,
                    "endTime": end_time,
                    "input": input_data,
                    "output": output_data,
                    "level": level,
                },
            }
            if status_message:
                event["body"]["statusMessage"] = status_message

            # Fire and forget batch ingestion
            api.ingestion_batch(batch=[event])
    except Exception as e:
        logger.debug(f"Failed to emit native Langfuse trace: {e}")
