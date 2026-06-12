"""Usage / cost / session analytics store (CONCEPT:ECO-4.39).

Backend-abstracted: SQLite+FTS5 is the zero-dependency native default; Postgres
and DuckDB are enterprise-scale options. One store, two data planes — ingested
external agent logs (plane A) and our own runtime telemetry (plane B) — sharing
the same row shape so a single API and the three frontends serve both.
"""

from .backend import UsageBackend
from .backends import get_usage_backend, make_backend, reset_usage_backend_for_tests
from .models import (
    ORIGIN_INGESTED,
    ORIGIN_RUNTIME,
    ParsedSessionBundle,
    UsageEvent,
    UsageMessage,
    UsageSession,
    UsageToolCall,
)
from .recorder import UsageRecorder, get_usage_recorder
from .service import UsageService, get_usage_service

__all__ = [
    "ORIGIN_INGESTED",
    "ORIGIN_RUNTIME",
    "ParsedSessionBundle",
    "UsageBackend",
    "UsageEvent",
    "UsageMessage",
    "UsageRecorder",
    "UsageService",
    "UsageSession",
    "UsageToolCall",
    "get_usage_backend",
    "get_usage_recorder",
    "get_usage_service",
    "make_backend",
    "reset_usage_backend_for_tests",
]
