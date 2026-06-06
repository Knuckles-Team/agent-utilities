"""CONCEPT:ORCH-1.33 — Stream-format dispatch: parse adapter stdout into canonical events.

Each :class:`~agent_utilities.core.execution.adapters.base.StreamFormat` maps to a handler that turns
a backend's native stdout into a uniform stream of
:class:`~agent_utilities.core.execution.adapters.base.ExecEvent`. Mirrors open-design's
``streamFormat`` → handler-factory dispatch (server.ts), decoupling the spawn machinery from parsing.

Adding a new format = a new handler + one registry entry; the engine is untouched.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Iterator

from .adapters.base import ExecEvent, ExecEventType, StreamFormat

logger = logging.getLogger(__name__)

# A handler maps an iterable of raw stdout lines → an iterator of canonical events.
StreamHandler = Callable[[Iterable[str]], Iterator[ExecEvent]]


def _plain_handler(lines: Iterable[str]) -> Iterator[ExecEvent]:
    """``plain``: accumulate raw text; emit start, the joined text, then end."""
    yield ExecEvent(ExecEventType.START)
    buf: list[str] = []
    for line in lines:
        buf.append(line)
        yield ExecEvent(ExecEventType.TEXT_DELTA, text=line)
    yield ExecEvent(ExecEventType.END, data={"text": "".join(buf)})


def _jsonl_handler(lines: Iterable[str]) -> Iterator[ExecEvent]:
    """``jsonl``: each line is a JSON object with a ``type`` understood across CLIs.

    Recognised shapes (superset across the CLIs we target):
      - ``{"type":"text"|"delta"|"content", "text"|"content": "..."}`` → TEXT_DELTA
      - ``{"type":"tool_use", ...}``                                   → TOOL_USE
      - ``{"type":"error", "error"|"message": "..."}``                 → ERROR
    Unparseable lines are surfaced as TEXT_DELTA (never dropped silently).
    """
    yield ExecEvent(ExecEventType.START)
    for line in lines:
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            yield ExecEvent(ExecEventType.TEXT_DELTA, text=line)
            continue
        kind = str(obj.get("type", "")).lower()
        if kind in {"text", "delta", "content", "text_delta", "assistant"}:
            yield ExecEvent(
                ExecEventType.TEXT_DELTA,
                text=str(obj.get("text") or obj.get("content") or obj.get("delta") or ""),
            )
        elif kind in {"tool_use", "tool", "tool_call"}:
            yield ExecEvent(ExecEventType.TOOL_USE, data=obj)
        elif kind in {"error", "err"}:
            yield ExecEvent(ExecEventType.ERROR, text=str(obj.get("error") or obj.get("message") or s))
        elif kind in {"end", "done", "result", "turn_end"}:
            yield ExecEvent(ExecEventType.END, data=obj)
        else:
            yield ExecEvent(ExecEventType.TEXT_DELTA, text=str(obj.get("text") or s))
    yield ExecEvent(ExecEventType.END)


_HANDLERS: dict[StreamFormat, StreamHandler] = {
    StreamFormat.PLAIN: _plain_handler,
    StreamFormat.JSONL: _jsonl_handler,
}


def get_stream_handler(fmt: StreamFormat) -> StreamHandler:
    """Return the handler for ``fmt`` (defaults to the plain handler if unregistered)."""
    return _HANDLERS.get(fmt, _plain_handler)


def register_stream_handler(fmt: StreamFormat, handler: StreamHandler) -> None:
    """Register/override a stream handler (extension point for new adapter protocols)."""
    _HANDLERS[fmt] = handler


def collect_text(events: Iterable[ExecEvent]) -> str:
    """Convenience: fold an event stream into the final assistant text."""
    parts: list[str] = []
    for ev in events:
        if ev.type is ExecEventType.TEXT_DELTA:
            parts.append(ev.text)
        elif ev.type is ExecEventType.END and ev.data.get("text"):
            return str(ev.data["text"])
    return "".join(parts)
