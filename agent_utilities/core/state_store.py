"""Unified durable-state externalization layer.

CONCEPT:AU-OS.state.unified-durable-state-externalization — Unified durable-state externalization — one STATE_DB_URI flag selects a shared Postgres state store over the per-host SQLite default

One config flag — ``state_db_uri`` (``STATE_DB_URI``) — selects where the
platform's durable state lives:

* **Unset (default)** — the existing zero-infra per-host SQLite files; every
  store keeps its current behavior and file layout.
* **``postgresql://`` URI** — durable-execution checkpoints, sessions/turns/
  goals (and their fleet registry), and the KG task queue all move onto ONE
  shared Postgres, so a second host can safely participate and the gateway
  becomes stateless.

This module is the single seam the per-store backends share:

* :func:`state_pool` — one process-wide ``psycopg_pool.ConnectionPool`` (the
  same driver the KG :class:`PostgreSQLBackend` uses), sized by
  ``state_db_pool_size``.
* :func:`ensure_state_schema` — lightweight idempotent ``CREATE TABLE IF NOT
  EXISTS`` migrations, run once per process per store (the convention the
  existing Postgres checkpoint backend follows).
* :func:`open_state_connection` — a DB-API-ish connection that adapts SQLite
  ``?`` placeholders to psycopg ``%s`` and yields rows addressable both by
  index and by column name, so callers keep their existing SQL.
* :func:`state_claim_guard` — cross-host claim serialization via Postgres
  advisory locks (no-op under SQLite, where the per-host flock/thread locks
  already suffice).

Tests never require a live Postgres: with ``state_db_uri`` unset nothing here
ever touches the network.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_pool_lock = threading.Lock()
_pool: Any = None
_pool_dsn: str | None = None

_migrations_lock = threading.Lock()
_migrated_stores: set[str] = set()


def state_db_uri() -> str | None:
    """The configured external state DSN, or ``None`` for the SQLite default."""
    try:
        from agent_utilities.core.config import config

        uri = getattr(config, "state_db_uri", None)
    except Exception:  # noqa: BLE001 — config unavailable → zero-infra default
        return None
    if uri and str(uri).startswith(("postgresql://", "postgres://")):
        return str(uri)
    return None


def postgres_state_enabled() -> bool:
    """True when durable state is externalized to Postgres (CONCEPT:AU-OS.state.unified-durable-state-externalization)."""
    return state_db_uri() is not None


def state_pool() -> Any:
    """Lazily open the ONE shared psycopg connection pool for all state stores.

    Raises if ``state_db_uri`` is unset — callers must check
    :func:`postgres_state_enabled` first.
    """
    global _pool, _pool_dsn
    dsn = state_db_uri()
    if dsn is None:
        raise RuntimeError("state_db_uri is not configured (SQLite default in effect)")
    with _pool_lock:
        if _pool is not None and _pool_dsn == dsn:
            return _pool
        from psycopg_pool import ConnectionPool

        try:
            from agent_utilities.core.config import config

            max_size = max(1, int(getattr(config, "state_db_pool_size", 8)))
        except Exception:  # noqa: BLE001
            max_size = 8
        _pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=max_size,
            open=True,
            kwargs={"autocommit": False},
        )
        _pool_dsn = dsn
        logger.info(
            "state-store pool opened (max=%d) — durable state on Postgres", max_size
        )
        return _pool


def reset_state_store_for_tests() -> None:
    """Drop cached pool/migration state (test isolation helper)."""
    global _pool, _pool_dsn
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.close()
            except Exception:  # noqa: BLE001
                pass
        _pool = None
        _pool_dsn = None
    with _migrations_lock:
        _migrated_stores.clear()


def ensure_state_schema(store: str, ddl: str, pool: Any | None = None) -> None:
    """Run a store's idempotent DDL once per process (Postgres path only).

    Follows the existing Postgres checkpoint backend's convention: schema is a
    set of ``CREATE TABLE IF NOT EXISTS`` statements applied on first connect —
    no migration framework, safe to run concurrently from many hosts.
    """
    if store in _migrated_stores:
        return
    with _migrations_lock:
        if store in _migrated_stores:
            return
        p = pool if pool is not None else state_pool()
        with p.connection() as conn:
            conn.execute(ddl)
        _migrated_stores.add(store)
        logger.debug("state-store schema ensured for %r", store)


# ── Tenant scoping (AU-P0-5) ───────────────────────────────────────────────
#
# The state-store pool is a SEPARATE Postgres connection pool from the KG
# backend's own (:class:`~agent_utilities.knowledge_graph.backends.postgresql_backend.PostgreSQLBackend`),
# so it needs its own RLS-GUC seam even though the convention is identical:
# a per-session/transaction GUC (``app.tenant_id``) that ``deploy/postgres/tenant_rls.sql``
# (or an equivalent per-store RLS policy) checks — empty/unset means
# unrestricted (the historical/system path), a non-empty tenant means "this
# tenant's rows + commons only". Every ``sessions``/``turns``/``usage`` etc.
# table sharing this pool gets tenant isolation the moment it adds a
# ``tenant_id`` column + RLS policy, because the connection ALREADY carries
# the GUC by the time the caller's SQL runs.
STATE_RLS_GUC = "app.tenant_id"


def state_tenant_guc_sql(tenant_id: str | None) -> str:
    """The ``SET LOCAL app.tenant_id`` statement for a state connection's tenant."""
    safe = (tenant_id or "").replace("'", "''")
    return f"SET LOCAL {STATE_RLS_GUC} = '{safe}'"


def ambient_state_tenant() -> str:
    """Resolve the tenant a freshly-checked-out state connection should carry.

    Prefers the ambient :class:`~agent_utilities.knowledge_graph.core.session.GraphSession`
    (AU-P0-1's one currency), falling back to the ambient actor's
    ``tenant_id``; ``""`` (unrestricted/commons) when neither is scoped — the
    same fallback the KG Postgres backend uses, so a bare/system caller sees
    identical (unrestricted) behaviour to before this GUC existed.
    """
    try:
        from agent_utilities.knowledge_graph.core.session import current_session

        session = current_session()
        if session is not None and session.tenant:
            return session.tenant
    except Exception:  # noqa: BLE001 — tenant resolution must never break a checkout
        pass
    try:
        from agent_utilities.security.brain_context import current_actor

        return current_actor().tenant_id or ""
    except Exception:  # noqa: BLE001
        return ""


def set_state_tenant(conn: Any, tenant_id: str | None) -> None:
    """Scope a pooled state connection to ``tenant_id`` via the RLS GUC (AU-P0-5).

    Fail-CLOSED for a real tenant, same rationale as
    :meth:`PostgreSQLBackend.set_request_tenant`: the state pool is reused, so a
    swallowed ``SET app.tenant_id`` for a NON-EMPTY tenant would leave the
    connection carrying the previous checkout's tenant — a silent cross-tenant
    leak. A failed SET for a non-empty tenant RAISES so
    :func:`open_state_connection` aborts the checkout rather than proceeding
    unscoped. Fail-open only for the empty/``None`` baseline (unrestricted /
    system path), preserving historical best-effort behaviour.
    """
    try:
        conn.execute(state_tenant_guc_sql(tenant_id))
    except Exception as e:
        if tenant_id:
            logger.warning(
                "set_state_tenant failed for tenant %r — aborting checkout to "
                "avoid serving a stale tenant context: %s",
                tenant_id,
                e,
            )
            raise
        logger.debug("set_state_tenant (unscoped) failed: %s", e)


def advisory_key(name: str) -> int:
    """Stable signed 64-bit advisory-lock key for ``name`` (same on every host)."""
    digest = hashlib.sha256(f"agent-utilities:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


@contextmanager
def state_claim_guard(name: str) -> Iterator[None]:
    """Cross-host critical section for claim operations (CONCEPT:AU-OS.state.unified-durable-state-externalization).

    Under Postgres state, holds a session advisory lock (``pg_advisory_lock``)
    for the duration so claims are atomic across N hosts. Under the SQLite
    default this is a no-op — the existing per-process thread locks plus the
    per-host singleton flock already serialize claims on a single host.
    """
    if not postgres_state_enabled():
        with nullcontext():
            yield
        return
    key = advisory_key(name)
    pool = state_pool()
    with pool.connection() as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (key,))
        try:
            yield
        finally:
            try:
                conn.execute("SELECT pg_advisory_unlock(%s)", (key,))
            except Exception:  # noqa: BLE001 — conn teardown releases the lock anyway
                pass


# ── DB-API adaptation ─────────────────────────────────────────────────────


def _make_row_type(fields: tuple[str, ...]) -> type:
    """Build a tuple subclass bound to ``fields`` — rows addressable by position
    AND column name (and ``dict(row)`` works, mirroring ``sqlite3.Row``)."""

    class _Row(tuple):
        __slots__ = ()

        def keys(self) -> list[str]:
            return list(fields)

        def __getitem__(self, item: Any) -> Any:
            if isinstance(item, str):
                try:
                    return tuple.__getitem__(self, fields.index(item))
                except ValueError:
                    raise KeyError(item) from None
            return tuple.__getitem__(self, item)

        def get(self, item: str, default: Any = None) -> Any:
            try:
                return self[item]
            except KeyError:
                return default

    return _Row


class StateCursor:
    """Cursor adapter: ``?`` placeholders in, name-or-index addressable rows out."""

    def __init__(self, raw_cursor: Any, dialect: str):
        self._cur = raw_cursor
        self.dialect = dialect

    def execute(self, sql: str, params: Any = ()) -> StateCursor:
        if self.dialect == "postgres":
            sql = sql.replace("?", "%s")
        self._cur.execute(sql, params)
        return self

    def _row_type(self) -> type | None:
        desc = self._cur.description
        if desc is None:
            return None
        return _make_row_type(tuple(col[0] for col in desc))

    def fetchone(self) -> Any:
        row = self._cur.fetchone()
        if row is None or self.dialect != "postgres":
            return row
        rt = self._row_type()
        return rt(row) if rt else row

    def fetchall(self) -> list[Any]:
        rows = self._cur.fetchall()
        if self.dialect != "postgres":
            return rows
        rt = self._row_type()
        return [rt(r) for r in rows] if rt else rows

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount


class StateConnection:
    """Connection adapter over sqlite3 or a pooled psycopg connection.

    Exposes the small DB-API surface the sessions/fleet/durable-exec code
    already uses (``cursor``/``execute``/``commit``/``close``) so the SQLite
    code paths stay intact and Postgres slots in behind the same SQL.
    """

    def __init__(
        self, raw: Any, dialect: str, on_close: Callable[[], None] | None = None
    ):
        self._raw = raw
        self.dialect = dialect
        self._on_close = on_close

    def cursor(self) -> StateCursor:
        return StateCursor(self._raw.cursor(), self.dialect)

    def execute(self, sql: str, params: Any = ()) -> StateCursor:
        return self.cursor().execute(sql, params)

    def commit(self) -> None:
        self._raw.commit()

    def rollback(self) -> None:
        try:
            self._raw.rollback()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        if self._on_close is not None:
            try:
                self.rollback()
            finally:
                self._on_close()
        else:
            self._raw.close()

    def __enter__(self) -> StateConnection:
        return self

    def __exit__(self, exc_type: Any, _exc: Any, _tb: Any) -> None:
        if exc_type is None:
            try:
                self.commit()
            except Exception:  # noqa: BLE001
                pass
        else:
            self.rollback()
        self.close()


def open_state_connection(
    store: str,
    sqlite_path: Callable[[], Path | str] | Path | str,
    postgres_ddl: str | None = None,
) -> StateConnection:
    """Open a connection to ``store`` on the selected backend.

    * SQLite default: connects to ``sqlite_path`` (callable resolved late so
      tests can monkeypatch it), ``sqlite3.Row`` rows.
    * Postgres (``state_db_uri`` set): borrows from the shared pool after
      ensuring ``postgres_ddl`` once per process; ``close()`` returns the
      connection to the pool. Every checkout is scoped to the ambient tenant
      via ``SET LOCAL app.tenant_id`` (AU-P0-5, :func:`set_state_tenant`)
      BEFORE the caller's SQL runs, so any store sharing this pool gets tenant
      isolation for free once it adds a ``tenant_id`` column + RLS policy.
    """
    if postgres_state_enabled():
        pool = state_pool()
        if postgres_ddl:
            ensure_state_schema(store, postgres_ddl, pool=pool)
        raw = pool.getconn()
        try:
            # Fail-closed tenant scoping (AU-P0-5): a non-empty tenant that
            # can't be set RAISES — return the borrowed connection to the pool
            # and abort the checkout rather than proceed unscoped.
            set_state_tenant(raw, ambient_state_tenant())
        except Exception:
            pool.putconn(raw)
            raise
        return StateConnection(raw, "postgres", on_close=lambda: pool.putconn(raw))
    path = sqlite_path() if callable(sqlite_path) else sqlite_path
    raw = sqlite3.connect(str(path))
    raw.row_factory = sqlite3.Row
    return StateConnection(raw, "sqlite")
