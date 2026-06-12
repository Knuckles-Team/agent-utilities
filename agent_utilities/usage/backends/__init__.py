"""Usage store backends (CONCEPT:ECO-4.39).

``sqlite`` is the zero-dependency native default; ``postgres`` and ``duckdb``
are enterprise-scale options. Selected by ``USAGE_DB_BACKEND``.
"""

from __future__ import annotations

import threading

from ..backend import UsageBackend

_lock = threading.Lock()
_singleton: UsageBackend | None = None
_singleton_key: str | None = None


def make_backend(name: str | None = None, uri: str | None = None) -> UsageBackend:
    """Construct a usage backend by name (no caching)."""
    name = (name or "sqlite").lower()
    if name == "sqlite":
        from .sqlite_fts import SqliteUsageBackend

        return SqliteUsageBackend(uri or None)
    if name == "postgres":
        from .postgres import PostgresUsageBackend

        return PostgresUsageBackend()
    if name == "duckdb":
        from .duckdb_mirror import DuckDBUsageBackend

        return DuckDBUsageBackend(uri or None)
    raise ValueError(f"unknown USAGE_DB_BACKEND: {name!r}")


def get_usage_backend() -> UsageBackend:
    """Process-wide usage backend, selected from config and schema-ensured once."""
    global _singleton, _singleton_key
    try:
        from agent_utilities.core.config import config

        name = getattr(config, "usage_db_backend", "sqlite")
        uri = getattr(config, "usage_db_uri", None)
    except Exception:  # noqa: BLE001 — zero-config default
        name, uri = "sqlite", None
    key = f"{name}:{uri or ''}"
    with _lock:
        if _singleton is not None and _singleton_key == key:
            return _singleton
        backend = make_backend(name, uri)
        backend.ensure_schema()
        _singleton = backend
        _singleton_key = key
        return backend


def reset_usage_backend_for_tests() -> None:
    global _singleton, _singleton_key
    with _lock:
        if _singleton is not None:
            try:
                _singleton.close()
            except Exception:  # noqa: BLE001
                pass
        _singleton = None
        _singleton_key = None
