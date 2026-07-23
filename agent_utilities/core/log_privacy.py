"""Process-wide privacy boundary for agent-utilities log records.

Application logs retain event names, status codes, counts, and exception types.
They do not retain runtime endpoints, filesystem locations, email-shaped caller
identifiers, or traceback/stack data (which embeds host paths).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_ENDPOINT = re.compile(
    r"(?i)\b(?:https?|wss?|tcp|tls|bolt|neo4j|postgres(?:ql)?|mongodb|redis(?:s)?)://\S+"
)
_WINDOWS_PATH = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/][^\s,;]+")
_POSIX_PATH = re.compile(r"(?<![A-Za-z0-9.])/(?:[^\s,;:/]+/)*[^\s,;:]+")
_HOST_PORT = re.compile(
    r"(?i)\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63}:\d{1,5}\b|"
    r"\b(?:\d{1,3}\.){3}\d{1,3}:\d{1,5}\b"
)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,63}\b")
_FACTORY_MARKER = "_agent_utilities_privacy_factory"


def sanitize_log_text(value: str) -> str:
    """Replace transport/location/caller-shaped material with neutral tokens."""
    rendered = _ENDPOINT.sub("<endpoint>", value)
    rendered = _WINDOWS_PATH.sub("<path>", rendered)
    rendered = _POSIX_PATH.sub("<path>", rendered)
    rendered = _HOST_PORT.sub("<endpoint>", rendered)
    return _EMAIL.sub("<caller>", rendered)


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, BaseException):
        # Sanitize the exception's MESSAGE; do not discard it. Collapsing an
        # exception to its class name made every `logger.error("...(%s)", exc)`
        # emit "RuntimeError: RuntimeError" — the privacy boundary silently
        # destroying the one piece of information the log existed to carry.
        # That turned real faults (a denied scope, a missing driver, an
        # unreadable durable record) into indistinguishable noise and cost real
        # debugging time. The message still goes through the SAME text
        # sanitizer as any other string, so endpoints, paths, host:port pairs
        # and emails are redacted exactly as before — the class name alone was
        # never what made this safe.
        rendered = sanitize_log_text(str(value))
        return f"{type(value).__name__}: {rendered}" if rendered else type(value).__name__
    if isinstance(value, Path):
        return "<path>"
    if isinstance(value, str):
        return sanitize_log_text(value)
    if isinstance(value, tuple):
        return tuple(_sanitize_value(item) for item in value)
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _sanitize_value(item) for key, item in value.items()}
    return value


def install_log_privacy_boundary() -> None:
    """Install an idempotent LogRecord factory for this package's records."""
    previous = logging.getLogRecordFactory()
    if getattr(previous, _FACTORY_MARKER, False):
        return

    def privacy_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        if record.name == "agent_utilities" or record.name.startswith(
            "agent_utilities."
        ):
            record.msg = _sanitize_value(record.msg)
            record.args = _sanitize_value(record.args)
            # Traceback and stack renderings contain host filesystem paths.
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return record

    setattr(privacy_factory, _FACTORY_MARKER, True)
    logging.setLogRecordFactory(privacy_factory)


__all__ = ["install_log_privacy_boundary", "sanitize_log_text"]
