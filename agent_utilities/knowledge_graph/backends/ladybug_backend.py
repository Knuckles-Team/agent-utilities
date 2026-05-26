#!/usr/bin/python
"""LadybugDB Graph Backend.

CONCEPT:KG-2.0

This module provides the LadybugDB implementation of the GraphBackend interface,
supporting strict schema-bound Cypher queries.
"""

import atexit
import logging
import os
import sys
import time as _time
import typing
import weakref
from typing import Any

import httpx

from .base import GraphBackend

try:
    import ladybug

    LADYBUG_AVAILABLE = True
except ImportError:
    LADYBUG_AVAILABLE = False

from agent_utilities.models.schema_definition import SCHEMA

logger = logging.getLogger(__name__)

import threading

_ACTIVE_DATABASES: dict[str, Any] = {}
_ACTIVE_DATABASES_LOCK = threading.Lock()

_ACTIVE_LOCKS: dict[str, threading.Lock] = {}
_ACTIVE_LOCKS_LOCK = threading.Lock()

_ACTIVE_BACKENDS: weakref.WeakSet[Any] = weakref.WeakSet()


def _cleanup_all_backends() -> None:
    """atexit handler to cleanly close all active Ladybug backends in order.

    This avoids C++ Kuzu abort/segfaults due to random GC ordering at exit.
    """
    for backend in list(_ACTIVE_BACKENDS):
        try:
            backend.close()
        except Exception:
            pass
    with _ACTIVE_DATABASES_LOCK:
        _ACTIVE_DATABASES.clear()


atexit.register(_cleanup_all_backends)

# ── G1: TTL-cached gateway health state ──────────────────────────────────
# Avoids per-query TCP+HTTP health check overhead. At 1000 agents × 100 qps,
# this reduces health checks from ~100K/sec to 1 every _HEALTH_TTL seconds.
_HEALTH_CACHE: dict[str, tuple[bool, float]] = {}
_HEALTH_TTL = 5.0  # seconds


def _is_gateway_healthy(host: str, port: int) -> bool:
    """TTL-cached health check for the centralized KG gateway."""
    key = f"{host}:{port}"
    cached = _HEALTH_CACHE.get(key)
    now = _time.monotonic()
    if cached and (now - cached[1]) < _HEALTH_TTL:
        return cached[0]
    try:
        from agent_utilities.mcp.kg_coordinator import KGCoordinator

        healthy = KGCoordinator.is_server_healthy(host=host, port=port)
    except Exception:
        healthy = False
    _HEALTH_CACHE[key] = (healthy, now)
    return healthy


# ── G2: Persistent httpx connection pool ─────────────────────────────────
# Reuses TCP connections across queries instead of creating a new connection
# per execute() call. HTTP keepalive yields 10-50x faster routing at scale.
_HTTP_CLIENT: httpx.Client | None = None


def _get_http_client(timeout: float = 30.0) -> httpx.Client:
    """Get or create a persistent httpx client with connection pooling."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
        )
    return _HTTP_CLIENT


class CombinedLock:
    """A pessimistic lock that combines thread lock and cross-process file lock."""

    def __init__(self, thread_lock: typing.Any, file_lock: typing.Any):
        self.thread_lock = thread_lock
        self.file_lock = file_lock

    def __enter__(self) -> "CombinedLock":
        self.thread_lock.acquire()
        try:
            self.file_lock.acquire()
        except Exception:
            self.thread_lock.release()
            raise
        return self

    def __exit__(
        self, exc_type: typing.Any, exc_val: typing.Any, exc_tb: typing.Any
    ) -> None:
        try:
            self.file_lock.release()
        finally:
            self.thread_lock.release()


class LadybugBackend(GraphBackend):
    """LadybugDB backend implementation."""

    def _get_lock(self):
        """Get a cross-process pessimistic lock for the database."""
        if self.db_path == ":memory:":
            from contextlib import nullcontext

            return nullcontext()

        from filelock import FileLock

        return CombinedLock(
            self._thread_lock,
            FileLock(f"{self.db_path}.lock", timeout=30.0),
        )

    def __init__(self, db_path: str = "knowledge_graph.db", max_retries: int = 15):
        if not LADYBUG_AVAILABLE:
            raise ImportError(
                "ladybug package is not installed. Install with 'pip install ladybug'"
            )
        self.db_path = db_path
        self.read_only = os.environ.get("LADYBUG_DB_READ_ONLY", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self.max_retries = max_retries
        self.db: typing.Any = None
        self.conn: typing.Any = None
        self._schema_created = False
        abs_db_path = (
            os.path.abspath(self.db_path) if self.db_path != ":memory:" else ":memory:"
        )
        with _ACTIVE_LOCKS_LOCK:
            if abs_db_path not in _ACTIVE_LOCKS:
                _ACTIVE_LOCKS[abs_db_path] = threading.Lock()
            self._thread_lock = _ACTIVE_LOCKS[abs_db_path]

        # Transient connection mode closes the database connection after every query
        # to allow multiple concurrent writer processes to work without file locks.
        transient_env = os.environ.get("LADYBUG_TRANSIENT_CONNECTIONS")
        if transient_env is not None:
            self.transient = transient_env.lower() in ("1", "true", "yes")
        else:
            # Default to transient if we are in testing mode to avoid hung pytests
            self.transient = os.environ.get(
                "AGENT_UTILITIES_TESTING", "false"
            ).lower() in ("1", "true", "yes")

        if self.db_path == ":memory:":
            self.transient = False

        _ACTIVE_BACKENDS.add(self)

    def _recover_connection(self) -> None:
        """Perform a deep connection cleanup and garbage collection to recover from locking/timeout deadlocks."""
        logger.warning(
            f"LadybugBackend: self-healing recovery started for {self.db_path}..."
        )
        abs_db_path = (
            os.path.abspath(self.db_path) if self.db_path != ":memory:" else ":memory:"
        )
        with _ACTIVE_DATABASES_LOCK:
            _ACTIVE_DATABASES.pop(abs_db_path, None)
        try:
            self.close()
        except Exception as e:
            logger.debug(f"Error during self-healing close: {e}")

        # Run Python garbage collection to clean up C++ object wrappers
        import gc

        gc.collect()

        # Re-open connection
        try:
            self._ensure_connection()
            logger.info("LadybugBackend: self-healing recovery completed successfully.")
        except Exception as e:
            logger.error(
                f"LadybugBackend: failed to restore connection in self-healing: {e}"
            )

    def _ensure_connection(self, max_retries: int | None = None) -> None:
        """Lazily ensure the Database and Connection are open with robust retry-backoff."""
        if self.conn is not None:
            return

        import time

        retries = max_retries if max_retries is not None else self.max_retries
        last_error: Exception = RuntimeError("Max retries exceeded")

        for attempt in range(retries):
            try:
                buffer_size = os.getenv("LADYBUG_MAX_DB_SIZE") or os.getenv(
                    "LADYBUG_BUFFER_SIZE"
                )
                from typing import Any

                db_params: dict[str, Any] = {}
                if self.read_only:
                    db_params["read_only"] = True
                if buffer_size:
                    try:
                        db_params["max_db_size"] = int(buffer_size)
                    except ValueError:
                        logger.warning(f"Invalid LADYBUG buffer/db size: {buffer_size}")

                # Safely open database
                abs_db_path = (
                    os.path.abspath(self.db_path)
                    if self.db_path != ":memory:"
                    else ":memory:"
                )
                with _ACTIVE_DATABASES_LOCK:
                    if abs_db_path in _ACTIVE_DATABASES:
                        self.db = _ACTIVE_DATABASES[abs_db_path]
                    else:
                        self.db = ladybug.Database(
                            self.db_path if self.db_path != ":memory:" else None,
                            **db_params,  # type: ignore[arg-type]
                        )
                        _ACTIVE_DATABASES[abs_db_path] = self.db
                self.conn = ladybug.Connection(self.db)

                # Apply WAL pragmas if supported
                try:
                    self.conn.execute("PRAGMA journal_mode=WAL;")
                    self.conn.execute("PRAGMA synchronous=NORMAL;")
                    self.conn.execute("PRAGMA busy_timeout=10000;")
                except Exception as e:
                    logger.debug(f"WAL pragma not supported or ignored: {e}")

                # Load VECTOR extension
                try:
                    self.conn.execute("INSTALL VECTOR;")
                    self.conn.execute("LOAD EXTENSION VECTOR;")
                    logger.debug("LadybugDB VECTOR extension loaded successfully")
                except Exception as ve:
                    logger.debug(f"Could not load VECTOR extension: {ve}")

                # Auto-initialize schema if not read-only
                if not self.read_only:
                    try:
                        self._create_schema_unlocked()
                    except Exception as schema_err:
                        logger.warning(f"Auto-initializing schema failed: {schema_err}")

                # Backup only if we successfully recovered after retries
                if attempt > 0:
                    self._backup_db()
                return
            except Exception as e:
                # Always clean up partial state on failure
                self.close()
                last_error = e
                msg = str(e).lower()
                if (
                    "corrupted" in msg
                    or "invalid wal record" in msg
                    or "read out invalid" in msg
                    or "unreachable_code" in msg
                    or "shadow" in msg
                    or "database id" in msg
                    or "cannot open file" in msg
                    or "cannot read from file" in msg
                    or "no such file or directory" in msg
                ):
                    logger.warning(
                        f"Detected database corruption or WAL/shadow error in {self.db_path} "
                        f"(attempt {attempt + 1}/{retries}). Self-healing by cleaning up WAL/shadow files."
                    )
                    self._backup_db()
                    self._cleanup_corrupted()
                    if attempt >= 2 and self.db_path != ":memory:":
                        logger.error(
                            f"Persistent database corruption detected in {self.db_path} after WAL cleanup. "
                            f"Moving main database file aside to allow complete self-healing."
                        )
                        try:
                            from pathlib import Path

                            p = Path(self.db_path)
                            if p.exists():
                                p.rename(p.with_suffix(".corrupted"))
                        except Exception as rename_err:
                            logger.error(
                                f"Failed to move corrupted database: {rename_err}"
                            )
                    continue
                elif (
                    "lock" in msg
                    or "busy" in msg
                    or "catalog exception" in msg
                    or "already exists" in msg
                    or "bad_alloc" in msg
                    or "io exception" in msg
                    or "no such file" in msg
                ):
                    if attempt == retries - 1:
                        raise e
                    import secrets

                    wait_time = (
                        (2**attempt) * 0.1
                    ) + secrets.SystemRandom().random() * 0.2
                    logger.warning(
                        f"Graph DB locked or catalog race, retrying connection in {wait_time:.2f}s "
                        f"(attempt {attempt + 1}/{retries})..."
                    )
                    time.sleep(wait_time)
                else:
                    raise e
        raise last_error

    def close(self) -> None:
        """Close the database connection and database object."""
        import gc

        conn = getattr(self, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # nosec B110
                pass
            self.conn = None
            gc.collect()
        db = getattr(self, "db", None)
        if db is not None:
            self.db = None
            gc.collect()

    def __del__(self) -> None:
        """Ensure connection is destroyed before database to avoid C++ Kuzu abort."""
        try:
            conn = getattr(self, "conn", None)
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
                self.conn = None
            db = getattr(self, "db", None)
            if db is not None:
                self.db = None
        except Exception:  # nosec B110
            pass

    def _cleanup_corrupted(self):
        """Removes corrupted WAL/journal files to allow a clean restart.

        Note: Does NOT delete the main DB file — only transient WAL/journal
        artifacts that can cause UNREACHABLE_CODE assertions in ladybug.
        """
        from pathlib import Path

        base_path = Path(self.db_path)
        if base_path.name == ":memory:":
            return

        # Only remove transient WAL/journal files, NOT the main DB
        wal_exts = [
            ".wal",
            "-wal",
            ".shm",
            "-shm",
            ".lock",
            ".shadow",
            ".wal.checkpoint",
        ]
        for ext in wal_exts:
            p = base_path.parent / (base_path.name + ext)
            if p.exists():
                try:
                    p.unlink()
                    logger.info(f"Cleaned up corrupted file: {p}")
                except Exception as e:
                    logger.error(f"Failed to cleanup {p}: {e}")

    def _backup_db(self):
        """Maintains up to N most recent backups of the database."""
        import datetime
        import shutil
        from pathlib import Path

        from agent_utilities.core.config import DEFAULT_KG_BACKUPS

        if DEFAULT_KG_BACKUPS <= 0 or self.db_path == ":memory:":
            return

        base_path = Path(self.db_path)
        if not base_path.exists():
            return

        try:
            # Check disk space before backup (skip if < 1GB free)
            db_size = base_path.stat().st_size
            statvfs = os.statvfs(base_path.parent)
            free_bytes = statvfs.f_bavail * statvfs.f_frsize
            if free_bytes < max(db_size * 2, 1_073_741_824):  # Need 2x DB size or 1GB
                logger.info(
                    f"Skipping DB backup: only {free_bytes / 1e9:.1f}GB free "
                    f"(need {max(db_size * 2, 1_073_741_824) / 1e9:.1f}GB)"
                )
                return

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = base_path.with_name(f"{base_path.name}.{timestamp}.bak")

            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")

            # Prune old backups (keep N most recent)
            backups = sorted(
                base_path.parent.glob(f"{base_path.name}.*.bak"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            for old_backup in backups[DEFAULT_KG_BACKUPS:]:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup}")
        except Exception as e:
            # Don't crash the server if backup fails, just log it
            logger.warning(f"Database backup failed: {e}")

    # ── G3: Extracted gateway routing helper ──────────────────────────────
    def _route_to_gateway(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        batch: list[dict[str, Any]] | None = None,
        chunk_size: int = 500,
    ) -> list[dict[str, Any]] | None:
        """Attempt to route a query to the centralized KG gateway.

        Returns results on success, or None if the query should fall back
        to local SQLite execution. Uses TTL-cached health checks (G1) and
        a persistent connection pool (G2) for minimal overhead.
        """
        is_testing = (
            os.environ.get("AGENT_UTILITIES_TESTING") == "true"
            or "pytest" in sys.modules
            or os.getenv("PYTEST_CURRENT_TEST") is not None
        )
        is_validation = os.environ.get("DEFAULT_VALIDATION_MODE") == "true"
        is_server = (
            os.environ.get("IS_KG_SERVER") == "true"
            or "agent_utilities.mcp.kg_server" in sys.modules
        )

        if is_testing or is_validation or is_server:
            return None

        try:
            kg_host = os.getenv("KG_SERVER_HOST", "127.0.0.1")
            kg_port = int(os.getenv("KG_SERVER_PORT", "8100"))
            if not _is_gateway_healthy(kg_host, kg_port):
                return None

            url = f"http://{kg_host}:{kg_port}/cypher"
            headers = {
                "X-Agent-ID": os.getenv("AGENT_NAME")
                or os.getenv("AGENT_ID")
                or "anonymous_agent",
                "X-Session-ID": os.getenv("SESSION_ID")
                or os.getenv("CHAT_SESSION_ID")
                or "default_session",
            }

            # Build payload — batch mode vs single query
            payload: dict[str, Any] = {"query": query}
            if batch is not None:
                payload["batch"] = batch
                payload["chunk_size"] = chunk_size
            else:
                payload["params"] = params or {}

            timeout = 60.0 if batch is not None else 30.0
            client = _get_http_client(timeout=timeout)
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "error":
                raise RuntimeError(data.get("message"))
            return data.get("results") or []
        except Exception as e:
            logger.debug(
                f"Routing query to centralized KG server failed: {e}. "
                f"Falling back to local SQLite execution."
            )
            # Invalidate health cache on failure to trigger re-check next time
            kg_host = os.getenv("KG_SERVER_HOST", "127.0.0.1")
            kg_port = int(os.getenv("KG_SERVER_PORT", "8100"))
            _HEALTH_CACHE.pop(f"{kg_host}:{kg_port}", None)
            return None

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query on LadybugDB."""
        # G1+G2+G3: Attempt centralized gateway routing with cached health + pooled client
        routed = self._route_to_gateway(query, params=params)
        if routed is not None:
            return routed

        import time

        max_retries = self.max_retries
        last_error = None

        for attempt in range(max_retries):
            try:
                with self._get_lock():
                    self._ensure_connection()
                    if self.conn is None:
                        logger.warning(
                            "LadybugBackend.execute: connection could not be opened."
                        )
                        return []
                    res = self.conn.execute(query, params or {})
                    if isinstance(res, list):
                        if not res:
                            ret_rows = []
                        else:
                            res = res[0]
                            from typing import cast

                            ret_rows = cast(
                                list[dict[str, Any]], res.rows_as_dict().get_all()
                            )
                    else:
                        from typing import cast

                        ret_rows = cast(
                            list[dict[str, Any]], res.rows_as_dict().get_all()
                        )

                    # If transient mode is enabled, immediately close connection inside the lock
                    if self.transient:
                        self.close()

                return ret_rows
            except Exception as e:
                # On error, make sure we close the connection inside the lock
                if self.transient:
                    try:
                        with self._get_lock():
                            self.close()
                    except Exception:
                        pass

                msg = str(e).lower()
                from filelock import Timeout as FileLockTimeout

                if (
                    isinstance(e, FileLockTimeout)
                    or "lock" in msg
                    or "busy" in msg
                    or "database is locked" in msg
                    or "catalog exception" in msg
                ):
                    import secrets

                    # Trigger self-healing recovery before retrying
                    self._recover_connection()

                    wait_time = (
                        2**attempt
                    ) * 0.1 + secrets.SystemRandom().random() * 0.1
                    logger.warning(
                        f"Database locked or timeout, healed connection. Retrying execute in {wait_time:.2f}s... (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    last_error = e
                    continue
                elif (
                    "already has property" in msg
                    or "duplicate" in msg
                    or "already exists" in msg
                ):
                    logger.debug(f"LadybugDB expected migration error: {e}")
                elif "table" in msg and "does not exist" in msg:
                    logger.warning(f"LadybugDB table not found (check schema): {e}")
                elif "binder exception" in msg:
                    if (
                        "doesn't have an index with name" in msg
                        or "cannot find property" in msg
                    ):
                        logger.debug(
                            f"LadybugDB vector index or property missing (expected): {e}"
                        )
                    else:
                        logger.error(f"LadybugDB binder issue (invalid property?): {e}")
                else:
                    logger.error(
                        f"LadybugDB Cypher execution failed: {e}\nQuery: {query}"
                    )
                return []

        if last_error:
            logger.error(
                f"Failed to execute query after {max_retries} retries due to locking: {last_error}"
            )
        return []

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]], chunk_size: int = 500
    ) -> list[dict[str, Any]]:
        """Execute a batch query in chunks to avoid blocking the DB for too long."""
        # G1+G2+G3: Attempt centralized gateway routing with cached health + pooled client
        routed = self._route_to_gateway(query, batch=batch, chunk_size=chunk_size)
        if routed is not None:
            return routed

        import secrets
        import time

        results: list[dict[str, Any]] = []
        max_retries = self.max_retries
        for i in range(0, len(batch), chunk_size):
            chunk = batch[i : i + chunk_size]
            attempt = 0
            while attempt < max_retries:
                try:
                    with self._get_lock():
                        self._ensure_connection()
                        if self.conn is None:
                            logger.warning(
                                "LadybugBackend.execute_batch: connection could not be opened."
                            )
                            break
                        for params in chunk:
                            res = self.conn.execute(query, params or {})
                            # ladybug return format: list of QueryResult objects
                            if res and hasattr(res, "get_as_df"):
                                df = res.get_as_df()
                                results.extend(
                                    typing.cast(
                                        list[dict[str, Any]], df.to_dict("records")
                                    )
                                )
                        # Close connection inside the lock in transient mode
                        if self.transient:
                            self.close()
                    break  # Success, move to next chunk
                except Exception as e:
                    if self.transient:
                        try:
                            with self._get_lock():
                                self.close()
                        except Exception:
                            pass
                    msg = str(e).lower()
                    from filelock import Timeout as FileLockTimeout

                    if (
                        isinstance(e, FileLockTimeout)
                        or "lock" in msg
                        or "busy" in msg
                        or "database is locked" in msg
                        or "catalog exception" in msg
                    ):
                        attempt += 1

                        # Trigger self-healing recovery before retrying
                        self._recover_connection()

                        wait_time = (
                            2**attempt
                        ) * 0.05 + secrets.SystemRandom().random() * 0.1
                        logger.warning(
                            f"Database locked or timeout during batch, healed connection. Retrying chunk in {wait_time:.2f}s... (attempt {attempt}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    logger.warning(f"Batch execution chunk failed: {e}")
                    break
        return results

    def wal_checkpoint(self) -> bool:
        """Perform a WAL checkpoint if the underlying engine supports it."""
        try:
            with self._get_lock():
                self._ensure_connection()
                if self.conn is None:
                    return False
                self.conn.execute("CHECKPOINT;")
                if self.transient:
                    self.close()
            return True
        except Exception as e:
            if self.transient:
                try:
                    with self._get_lock():
                        self.close()
                except Exception:
                    pass
            logger.debug(f"WAL checkpoint not supported or failed: {e}")
            return False

    def _create_schema_unlocked(self) -> None:
        """Internal method to synchronize schema without acquiring the connection lock."""
        if self.conn is None:
            return

        # 1. Create Node Tables
        for node in SCHEMA.nodes:
            cols = ", ".join(
                [f"`{name}` {dtype}" for name, dtype in node.columns.items()]
            )
            stmt = f"CREATE NODE TABLE IF NOT EXISTS {node.name} ({cols});"
            try:
                self.conn.execute(stmt)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Node table creation issue ({node.name}): {e}")

        # 2. Create Rel Tables
        for rel in SCHEMA.edges:
            conns = ", ".join(
                [f"FROM {c['from']} TO {c['to']}" for c in rel.connections]
            )
            stmt = f"CREATE REL TABLE IF NOT EXISTS {rel.type} ({conns});"
            try:
                self.conn.execute(stmt)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Rel table creation issue ({rel.type}): {e}")

    def create_schema(self) -> None:
        """Create LadybugDB schema from the unified schema definition.
        Ladybug requires strict DDL for Node and Rel tables.
        """
        logger.info(
            f"Synchronizing Knowledge Graph Schema ({len(SCHEMA.nodes)} node tables, {len(SCHEMA.edges)} edge tables)..."
        )
        with self._get_lock():
            self._ensure_connection()
            if self.conn is None:
                logger.warning(
                    "LadybugBackend.create_schema: connection could not be opened."
                )
                return
            self._create_schema_unlocked()
            if self.transient:
                self.close()

    def build_vector_indices(self, tables: list[str] | None = None) -> None:
        """Create Vector Indices for any FLOAT column named 'embedding'.

        Note: LadybugDB (Kuzu) currently does not support updating properties
        (via SET) that are part of a vector index. Therefore, vector indices
        should only be built AFTER all initial ingestion is complete.

        Args:
            tables: Optional list of specific table names to build indexes for.
                When None, builds for all tables with embedding columns.
        """
        embedding_tables = [
            node.name
            for node in SCHEMA.nodes
            if "embedding" in node.columns
            and "FLOAT" in node.columns["embedding"].upper()
        ]
        if tables:
            embedding_tables = [t for t in embedding_tables if t in tables]
        if embedding_tables:
            with self._get_lock():
                self._ensure_connection()
                if self.conn is None:
                    logger.warning(
                        "LadybugBackend.build_vector_indices: connection could not be opened."
                    )
                    return
                try:
                    self.conn.execute("INSTALL VECTOR;")
                    self.conn.execute("LOAD EXTENSION VECTOR;")
                    vector_extension_loaded = True
                except Exception as e:
                    logger.info(
                        "LadybugDB VECTOR extension unavailable; skipping vector "
                        "index DDL for %d embedding table(s): %s",
                        len(embedding_tables),
                        e,
                    )
                    vector_extension_loaded = False

                if vector_extension_loaded:
                    skip_reason: str | None = None
                    for table in embedding_tables:
                        idx_name = f"idx_{table.lower()}_embedding"
                        stmt = (
                            f"CALL CREATE_VECTOR_INDEX('{table}', "
                            f"'{idx_name}', 'embedding');"
                        )
                        try:
                            self.conn.execute(stmt)
                        except Exception as e:
                            msg = str(e)
                            if "already exists" in msg.lower():
                                continue
                            if "FLOAT/DOUBLE ARRAY" in msg:
                                skip_reason = msg
                                break
                            logger.warning(
                                f"Vector index creation issue ({idx_name}): {e}"
                            )
                    if skip_reason is not None:
                        logger.info(
                            "LadybugDB vector indexes skipped for %d table(s): %s. "
                            "Define embedding columns as FLOAT[N] (fixed size) to "
                            "enable HNSW indexing.",
                            len(embedding_tables),
                            skip_reason,
                        )
                if self.transient:
                    self.close()

    def drop_vector_indices(self, tables: list[str] | None = None) -> None:
        """Drop HNSW vector indexes so that embedding SET operations succeed.

        Must be called before ingestion if indexes were previously built,
        since LadybugDB (Kuzu) does not support SET on indexed columns.

        Args:
            tables: Optional list of specific table names to drop indexes for.
                When None, drops all embedding indexes.
        """
        embedding_tables = [
            node.name
            for node in SCHEMA.nodes
            if "embedding" in node.columns
            and "FLOAT" in node.columns["embedding"].upper()
        ]
        if tables:
            embedding_tables = [t for t in embedding_tables if t in tables]
        dropped = 0
        with self._get_lock():
            self._ensure_connection()
            if self.conn is None:
                logger.warning(
                    "LadybugBackend.drop_vector_indices: connection could not be opened."
                )
                return
            for table in embedding_tables:
                idx_name = f"idx_{table.lower()}_embedding"
                try:
                    self.conn.execute(
                        f"CALL DROP_VECTOR_INDEX('{table}', '{idx_name}');"
                    )
                    dropped += 1
                except Exception as e:
                    if (
                        "not found" not in str(e).lower()
                        and "does not exist" not in str(e).lower()
                    ):
                        logger.debug(f"Drop vector index issue ({idx_name}): {e}")
            if self.transient:
                self.close()
        if dropped:
            logger.info("Dropped %d HNSW vector indexes for re-ingestion.", dropped)

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Add embedding to an existing node."""
        query = "MATCH (n {id: $id}) SET n.embedding = $emb"
        # The _get_lock is inside self.execute()
        self.execute(query, {"id": node_id, "emb": embedding})

    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Perform a semantic vector search returning top matching nodes."""
        query = """
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        WITH n, array_cosine_similarity(n.embedding, $query_embedding) AS similarity
        ORDER BY similarity DESC
        LIMIT $n_results
        RETURN n
        """
        return self.execute(
            query, {"query_embedding": query_embedding, "n_results": n_results}
        )

    def prune(self, criteria: dict[str, Any]) -> None:
        """Prune nodes based on criteria.

        Args:
            criteria: A dictionary defining pruning rules:
                - node_type: (str) Optional filter for specific node labels.
                - age_days: (int) Delete nodes older than this number of days.
                - min_importance: (float) Delete nodes with importance_score below this.
        """
        node_type = criteria.get("node_type", "")
        label = f":{node_type}" if node_type else ""

        where_clauses = []
        params = {}

        if "age_days" in criteria:
            import datetime

            cutoff = (
                datetime.datetime.now() - datetime.timedelta(days=criteria["age_days"])
            ).isoformat()
            where_clauses.append("n.timestamp < $cutoff")
            params["cutoff"] = cutoff

        if "min_importance" in criteria:
            where_clauses.append("n.importance_score < $min_imp")
            params["min_imp"] = criteria["min_importance"]

        if not where_clauses:
            logger.warning("Prune called without any meaningful criteria.")
            return

        where_str = " AND ".join(where_clauses)
        query = f"MATCH (n{label}) WHERE {where_str} DETACH DELETE n"

        logger.info(f"Pruning nodes: {query} with params {params}")
        self.execute(query, params)

        # Reclaim WAL space after bulk deletes
        self.checkpoint_wal()

    def checkpoint_wal(self) -> None:
        """Force a WAL checkpoint to prevent unbounded WAL growth under multi-writer load.

        Should be called periodically during maintenance or after bulk operations
        to reclaim disk space and ensure readers see the latest committed state.
        """
        if self.db_path == ":memory:":
            return
        self.wal_checkpoint()
