#!/usr/bin/python
"""LadybugDB Graph Backend.

CONCEPT:AU-KG.query.object-graph-mapper

This module provides the LadybugDB implementation of the GraphBackend interface,
supporting strict schema-bound Cypher queries.
"""

import atexit
import logging
import os
import re
import time as _time
import typing
import weakref
from typing import Any

# Parse the engine's node/edge MERGE shapes so an unknown label/rel auto-creates
# its Kuzu table. Node refs carry an inline ``{id...}`` map: ``(n:Label {``,
# ``(s:Src {``; the rel type is in ``-[r:REL]->``.
_NODE_LABEL_RE = re.compile(r"\(\s*\w+\s*:\s*`?(\w+)`?\s*\{")
_REL_TYPE_RE = re.compile(r"-\s*\[\s*\w*\s*:\s*`?(\w+)`?\s*\]\s*->")

# Per-retry backoff cap for file-lock contention. Bounded so a transiently-blocked
# drainer recovers promptly once the lock frees, instead of over-sleeping into a
# multi-minute stall (an uncapped 2**n*0.1s reached 400s+ after ~13 attempts).
_LOCK_BACKOFF_MAX_S = 30.0

from agent_utilities.core.config import setting
from agent_utilities.security.identifiers import (
    InvalidIdentifierError,
    validate_identifier,
)

from ..base import GraphBackend, embedding_values_match

try:
    import ladybug

    LADYBUG_AVAILABLE = True
except ImportError:
    LADYBUG_AVAILABLE = False

from agent_utilities.models.schema_definition import SCHEMA

logger = logging.getLogger(__name__)

# Cross-cutting governance properties every write-path stamps onto node
# properties unconditionally (tenant scoping: CONCEPT:AU-KG.backend.company-brain-write-guard
# TenancyManager.scope_cypher_query / tenant_sharing.stamp_ownership; owner/scope
# visibility: tenant_sharing.stamp_ownership / apply_visibility; write-time ACL
# classification: tenant_sharing.stamp_classification) but that no individual
# ``TableDefinition`` in schema_definition.py declares as a column. Kuzu/Ladybug
# tables are strict-schema: a write/read referencing an undeclared property fails
# with "Binder exception: Cannot find property ... for <var>" instead of simply
# matching/storing nothing, so every governance-stamped write or tenant-scoped
# read against this backend was broken for every node type. Declared generically
# here (not duplicated across ~40 individual TableDefinitions) so any node table
# gains these columns the same way the KG-2.9g code-symbol columns are migrated
# onto pre-existing tables below. Kept in lockstep with
# ``materialization._GOVERNANCE_COLUMNS`` (the SET-clause/valid-keys side of the
# same contract).
_GOVERNANCE_COLUMNS: dict[str, str] = {
    "tenant_id": "STRING",
    "_owner_id": "STRING",
    "_shared_scope": "STRING",
    "classification": "STRING",
    # D-ACL-6: secured_reads._durable_access_rows unconditionally SELECTs
    # ``n.external_access`` for ACL hydration (a JSON-encoded connector
    # access descriptor). Same undeclared-column story as the other four —
    # every non-connector node type (e.g. CallableResource) had no such
    # column, so the ACL-hydration read itself failed with a Kuzu Binder
    # exception ("Cannot find property external_access for n") instead of
    # simply finding none, for EVERY node write on this backend.
    "external_access": "STRING",
}

import threading

_ACTIVE_DATABASES: dict[str, Any] = {}
_ACTIVE_DATABASES_LOCK = threading.Lock()
# Reference count per abs_db_path (CONCEPT:AU-KG.backend.mirror-health-repair):
# multiple LadybugBackend instances may share one cached ladybug.Database (see
# `_ensure_connection`'s cache-hit branch below). Each `ladybug.Database` reserves
# an ~8TB mmap virtual-address-space region (ladybug's own documented workaround
# for its buffer manager's default mmap ceiling), so leaving a "closed" backend's
# entry in `_ACTIVE_DATABASES` forever — as plain `close()` used to — leaked one
# such reservation per unique db_path for the life of the process. Enough unique
# paths (e.g. one per test, each via `tempfile.mkdtemp()`) exhausts the process's
# addressable virtual memory and every subsequent `ladybug.Database(...)` mmap
# fails with "Buffer manager exception: Mmap for size 8796093022208 failed."
# Only release (pop the cache entry + call the native `Database.close()`) once the
# last referencing backend has closed.
_ACTIVE_DATABASE_REFCOUNTS: dict[str, int] = {}

_ACTIVE_LOCKS: dict[str, threading.Lock] = {}
_ACTIVE_LOCKS_LOCK = threading.Lock()

_ACTIVE_BACKENDS: weakref.WeakSet[Any] = weakref.WeakSet()

_SYNCHRONIZED_DB_PATHS: set[str] = set()
_SYNCHRONIZED_DB_PATHS_LOCK = threading.Lock()


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
        _ACTIVE_DATABASE_REFCOUNTS.clear()


atexit.register(_cleanup_all_backends)

_LAST_GC_TIME = 0.0
_GC_LOCK = threading.Lock()


def _throttled_gc() -> None:
    """Run garbage collection, but throttled to at most once per second.

    This avoids massive GC thrashing when transient connections are closed
    after every query.
    """
    global _LAST_GC_TIME
    with _GC_LOCK:
        now = _time.monotonic()
        if now - _LAST_GC_TIME > 1.0:
            import gc

            gc.collect()
            _LAST_GC_TIME = now


# Substring markers used to classify a failed connection attempt's exception
# message (CONCEPT:AU-KG.backend.mirror-health-repair): "corrupted" self-heals
# by cleaning WAL/shadow artifacts (and, if persistent, quarantining the main
# file); "lock" retries with backoff; anything else is a hard failure.
_CORRUPTION_ERROR_MARKERS: tuple[str, ...] = (
    "corrupted",
    "invalid wal record",
    "read out invalid",
    "unreachable_code",
    "shadow",
    "database id",
    "cannot open file",
    "cannot read from file",
    "no such file or directory",
    "not a valid lbug",
    "unable to open database",
)
_LOCK_ERROR_MARKERS: tuple[str, ...] = (
    "lock",
    "busy",
    "already exists",
    "bad_alloc",
    "io exception",
    "no such file",
)


def _classify_connection_error(msg: str) -> str:
    """Classify a lower-cased connection-error message: corrupted/lock/other."""
    if any(marker in msg for marker in _CORRUPTION_ERROR_MARKERS):
        return "corrupted"
    if any(marker in msg for marker in _LOCK_ERROR_MARKERS):
        return "lock"
    return "other"


# Substring markers for classifying a failed execute() Cypher call, mirroring
# the connection-error markers above; lock-contention is checked separately
# (it also matches on exception type, not just message).
_MIGRATION_ERROR_MARKERS: tuple[str, ...] = (
    "already has property",
    "duplicate",
    "already exists",
)
_BINDER_EXPECTED_MARKERS: tuple[str, ...] = (
    "doesn't have an index with name",
    "cannot find property",
)


def _classify_execute_error(msg: str) -> str:
    """Classify a lower-cased execute() error message (lock handled separately)."""
    if any(marker in msg for marker in _MIGRATION_ERROR_MARKERS):
        return "migration"
    if "table" in msg and "does not exist" in msg:
        return "missing_table"
    if "binder exception" in msg:
        if any(marker in msg for marker in _BINDER_EXPECTED_MARKERS):
            return "binder_expected"
        return "binder_issue"
    return "other"


class LadybugLockContentionError(ConnectionError):
    """LadybugDB file/lock contention — retryable after connection self-heal."""


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
        self, _exc_type: typing.Any, _exc_val: typing.Any, _exc_tb: typing.Any
    ) -> None:
        try:
            self.file_lock.release()
        finally:
            self.thread_lock.release()


class LadybugBackend(GraphBackend):
    """LadybugDB backend implementation."""

    embedding_is_node_property = True

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

    @property
    def conn(self) -> typing.Any:
        local_store = getattr(self, "_local", None)
        if local_store is None:
            return None
        return getattr(local_store, "conn", None)

    @conn.setter
    def conn(self, value: typing.Any) -> None:
        local_store = getattr(self, "_local", None)
        if local_store is None:
            self._local = threading.local()
            local_store = self._local
        local_store.conn = value

    def __init__(self, db_path: str = "knowledge_graph.db", max_retries: int = 15):
        if not LADYBUG_AVAILABLE:
            raise ImportError(
                "ladybug package is not installed. Install with 'pip install ladybug'"
            )
        self._local = threading.local()
        # Never anchor the DB (and its sibling .lock/.wal/.corrupted files) to the
        # CWD: a relative default scattered them into whatever directory the process
        # happened to start in — the workspace-root `knowledge_graph.db.corrupted`
        # incident. Resolve a relative path under the agent-utilities data dir so it
        # is deterministic regardless of cwd.
        if db_path != ":memory:" and not os.path.isabs(db_path):
            from agent_utilities.core.paths import data_dir

            resolved = data_dir() / db_path
            resolved.parent.mkdir(parents=True, exist_ok=True)
            logger.warning(
                "LadybugBackend: relative db_path %r resolved to %s (avoiding "
                "cwd-relative DB files).",
                db_path,
                resolved,
            )
            db_path = str(resolved)
        self.db_path = db_path
        self.read_only = setting("LADYBUG_DB_READ_ONLY", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self.max_retries = max_retries
        self.db: typing.Any = None
        # abs_db_path this instance is counted against in `_ACTIVE_DATABASE_REFCOUNTS`
        # (set on successful `_ensure_connection`, cleared by `_release_cached_db`).
        self._db_cache_key: str | None = None
        self.conn = None
        self._schema_created = False
        # Auto-schema caches (CONCEPT:AU-KG.backend.mirror-health-repair): tables/rel-pairs known to exist, so
        # an arbitrary KG label/rel only triggers DDL on first sight. Seeded from
        # the declared SCHEMA on first write.
        self._known_node_tables: set[str] = set()
        self._known_rel_pairs: set[tuple[str, str, str]] = set()
        self._schema_cache_seeded = False
        # id → node-table label, learned on node writes (CONCEPT:AU-KG.backend.mirror-health-repair). Kuzu rel
        # creation must bind to specific node tables, but edge writes arrive
        # label-less (``MATCH (s {id:$x})…MERGE (s)-[:REL]->(t)``); this lets us
        # resolve each endpoint's table so the rel-pair table + bound MERGE work.
        self._node_labels: dict[str, str] = {}
        abs_db_path = (
            os.path.abspath(self.db_path) if self.db_path != ":memory:" else ":memory:"
        )
        with _ACTIVE_LOCKS_LOCK:
            if abs_db_path not in _ACTIVE_LOCKS:
                _ACTIVE_LOCKS[abs_db_path] = typing.cast(Any, threading.RLock())
            self._thread_lock = _ACTIVE_LOCKS[abs_db_path]

        # Transient connection mode closes the database connection after every query
        # to allow multiple concurrent writer processes to work without file locks.
        transient_env = setting("LADYBUG_TRANSIENT_CONNECTIONS")
        if transient_env is not None:
            self.transient = transient_env.lower() in ("1", "true", "yes")
        else:
            # Default to transient if we are in testing mode to avoid hung pytests
            self.transient = setting("AGENT_UTILITIES_TESTING", "false").lower() in (
                "1",
                "true",
                "yes",
            )

        if self.db_path == ":memory:":
            self.transient = False

        _ACTIVE_BACKENDS.add(self)

    def _recover_connection(self) -> None:
        """Perform a deep connection cleanup and garbage collection to recover from locking/timeout deadlocks."""
        with self._thread_lock:
            logger.warning(
                f"LadybugBackend: self-healing recovery started for {self.db_path}..."
            )
            abs_db_path = (
                os.path.abspath(self.db_path)
                if self.db_path != ":memory:"
                else ":memory:"
            )
            with _ACTIVE_DATABASES_LOCK:
                _ACTIVE_DATABASES.pop(abs_db_path, None)
                _ACTIVE_DATABASE_REFCOUNTS.pop(abs_db_path, None)
            try:
                self.close()
            except Exception as e:
                logger.debug("Error during self-healing close (%s)", type(e).__name__)

            # Run Python garbage collection to clean up C++ object wrappers
            import gc

            gc.collect()

            # Re-open connection
            try:
                self._ensure_connection()
                logger.info(
                    "LadybugBackend: self-healing recovery completed successfully."
                )
            except Exception as e:
                logger.error(
                    f"LadybugBackend: failed to restore connection in self-healing: {e}"
                )

    def _open_database_connection(self) -> None:
        """Resolve db_params, open (or reuse the cached) Database + Connection."""
        buffer_size = setting("LADYBUG_MAX_DB_SIZE")
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
            os.path.abspath(self.db_path) if self.db_path != ":memory:" else ":memory:"
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
            _ACTIVE_DATABASE_REFCOUNTS[abs_db_path] = (
                _ACTIVE_DATABASE_REFCOUNTS.get(abs_db_path, 0) + 1
            )
            self._db_cache_key = abs_db_path
        self.conn = ladybug.Connection(self.db)

    def _apply_connection_pragmas(self) -> None:
        """Apply WAL durability pragmas if supported by this LadybugDB build."""
        try:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA busy_timeout=10000;")
        except Exception as e:  # noqa: BLE001 — PRAGMA support varies by LadybugDB build; the connection is already open (self.conn = ladybug.Connection(self.db) above) — WAL/synchronous/busy_timeout are durability tuning, not required for correctness
            logger.debug(f"WAL pragma not supported or ignored: {e}")

    def _load_vector_extension(self) -> None:
        """Load the VECTOR extension; downstream paths fall back when absent."""
        try:
            self.conn.execute("INSTALL VECTOR;")
            self.conn.execute("LOAD EXTENSION VECTOR;")
            logger.debug("LadybugDB VECTOR extension loaded successfully")
        except Exception as ve:  # noqa: BLE001 — feature-detection: the VECTOR extension may not be present in this LadybugDB build; downstream vector-search paths already fall back when unavailable (not gated on any state advanced here)
            logger.debug(f"Could not load VECTOR extension: {ve}")

    def _auto_init_schema_if_needed(self) -> None:
        """Run schema auto-init once per abs_db_path (read-write connections only)."""
        abs_db_path = (
            os.path.abspath(self.db_path)
            if self.db_path != ":memory:"
            else f":memory:{id(self.db)}"
        )
        with _SYNCHRONIZED_DB_PATHS_LOCK:
            already_synced = abs_db_path in _SYNCHRONIZED_DB_PATHS
        if already_synced:
            return
        try:
            self._create_schema_unlocked()
            with _SYNCHRONIZED_DB_PATHS_LOCK:
                _SYNCHRONIZED_DB_PATHS.add(abs_db_path)
        except Exception as schema_err:
            logger.warning(f"Auto-initializing schema failed: {schema_err}")

    def _quarantine_corrupted_db(self) -> None:
        """Move a persistently-corrupted main DB file aside for full self-healing."""
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
            logger.error(f"Failed to move corrupted database: {rename_err}")

    def _handle_corruption_error(self, attempt: int, retries: int) -> None:
        logger.warning(
            f"Detected database corruption or WAL/shadow error in {self.db_path} "
            f"(attempt {attempt + 1}/{retries}). Self-healing by cleaning up WAL/shadow files."
        )
        self._backup_db()
        self._cleanup_corrupted()
        if attempt >= 2 and self.db_path != ":memory:":
            self._quarantine_corrupted_db()

    def _handle_lock_error(self, exc: Exception, attempt: int, retries: int) -> None:
        if attempt == retries - 1:
            raise exc
        import secrets
        import time

        wait_time = ((2**attempt) * 0.1) + secrets.SystemRandom().random() * 0.2
        logger.warning(
            f"Graph DB locked or catalog race (error: {exc}), retrying connection in "
            f"{wait_time:.2f}s (attempt {attempt + 1}/{retries})..."
        )
        time.sleep(wait_time)

    def _handle_connection_error(
        self, exc: Exception, attempt: int, retries: int
    ) -> None:
        """Classify one failed connection attempt and self-heal or re-raise."""
        kind = _classify_connection_error(str(exc).lower())
        if kind == "corrupted":
            self._handle_corruption_error(attempt, retries)
            return
        if kind == "lock":
            self._handle_lock_error(exc, attempt, retries)
            return
        raise exc

    def _ensure_connection(self, max_retries: int | None = None) -> None:
        """Lazily ensure the Database and Connection are open with robust retry-backoff."""
        if self.conn is not None:
            return

        retries = max_retries if max_retries is not None else self.max_retries
        last_error: Exception = RuntimeError("Max retries exceeded")

        for attempt in range(retries):
            try:
                self._open_database_connection()
                self._apply_connection_pragmas()
                self._load_vector_extension()
                if not self.read_only:
                    self._auto_init_schema_if_needed()
                # Backup only if we successfully recovered after retries
                if attempt > 0:
                    self._backup_db()
                return
            except Exception as e:
                # Always clean up partial state on failure
                self.close()
                last_error = e
                self._handle_connection_error(e, attempt, retries)
        raise last_error

    def _release_cached_db(self, db: typing.Any) -> None:
        """Drop this instance's reference to the shared cached `ladybug.Database`.

        Only the LAST backend referencing a given `abs_db_path` actually evicts
        the cache entry and calls the native `Database.close()` — releasing its
        ~8TB mmap virtual-address-space reservation. Without this, `close()`
        merely dropped the instance's own local reference while the module-level
        `_ACTIVE_DATABASES` cache kept a second, permanent reference alive, so the
        mmap reservation (and every one before it, one per unique db_path) was
        never freed for the life of the process — see `_ACTIVE_DATABASE_REFCOUNTS`.
        """
        cache_key = getattr(self, "_db_cache_key", None)
        self._db_cache_key = None
        if cache_key is None:
            return
        should_release = False
        with _ACTIVE_DATABASES_LOCK:
            remaining = _ACTIVE_DATABASE_REFCOUNTS.get(cache_key, 1) - 1
            if remaining <= 0:
                _ACTIVE_DATABASE_REFCOUNTS.pop(cache_key, None)
                _ACTIVE_DATABASES.pop(cache_key, None)
                should_release = True
            else:
                _ACTIVE_DATABASE_REFCOUNTS[cache_key] = remaining
        if should_release:
            try:
                db.close()
            except Exception as exc:  # noqa: BLE001 — best-effort release, see below
                # Best-effort by design: this runs on the teardown path after the
                # cache entry has ALREADY been dropped, so the mmap reservation is
                # released regardless and re-raising here would turn a clean close
                # into a failure. But the cause is logged rather than discarded —
                # a ladybug close() that keeps failing is exactly the signal that
                # this reservation leak is recurring, and swallowing it silently
                # is what let the original 8TiB-per-path leak run unnoticed until
                # the process ran out of addressable memory.
                logger.warning(
                    "ladybug: releasing cached Database for %s failed: %s",
                    cache_key,
                    exc,
                )

    def close(self) -> None:
        """Close the database connection and database object.

        KNOWN FOLLOW-ON COST (identified, not fixed here — CONCEPT:AU-KG.backend.mirror-health-repair):
        `transient` mode (the test-suite default) calls this after EVERY query,
        so `_release_cached_db` below runs, and drops into `_throttled_gc()`,
        that often. A test that issues MANY queries in a tight loop against a
        fresh `LadybugBackend` (e.g. iterating dozens of SCHEMA node tables) can
        make a single throttled `gc.collect()` call pathologically slow — 300s+
        pytest-timeout observed — apparently from the sheer volume of live
        pybind11-wrapped kuzu objects a full collection has to walk after many
        rapid Database create/destroy cycles. This was previously invisible
        because those same tests failed instantly at fixture SETUP with the
        mmap exhaustion this method's release logic fixes (see
        `_ACTIVE_DATABASE_REFCOUNTS`); removing that masking exposed this
        separate, narrower cost. A cache-warm-across-transient-recycling
        redesign (only release on true object death, not every `close()`) was
        tried and reverted: it eliminates this slowdown but reintroduces mmap
        exhaustion at full-suite scale (a single test's own instance can be
        collected promptly, but a run with many FAILING tests pins their
        `engine`/`backend` fixture objects alive via pytest's own traceback
        retention for the rest of the session, accumulating unreleased 8TB
        reservations faster than isolated micro-benchmarks showed). Fixing the
        `gc.collect()` cost itself (or its throttling policy) without
        reintroducing that regression needs its own investigation.
        """
        with self._thread_lock:
            conn = getattr(self, "conn", None)
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # nosec B110
                    pass
                self.conn = None
                _throttled_gc()
            db = getattr(self, "db", None)
            if db is not None:
                self.db = None
                self._release_cached_db(db)
                _throttled_gc()

    def _teardown_connection_and_db(self) -> None:
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
            self._release_cached_db(db)

    def __del__(self) -> None:
        """Ensure connection is destroyed before database to avoid C++ Kuzu abort."""
        try:
            lock = getattr(self, "_thread_lock", None)
            if lock is not None:
                with lock:
                    self._teardown_connection_and_db()
            else:
                self._teardown_connection_and_db()
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
                    logger.info("Cleaned up corrupted database artifact")
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
            logger.info("Database backup completed")

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

    def _rows_from_result(self, res: Any) -> list[dict[str, Any]]:
        from typing import cast

        if isinstance(res, list):
            if not res:
                return []
            res = res[0]
        return cast(list[dict[str, Any]], res.rows_as_dict().get_all())

    def _execute_attempt_body(
        self, query: str, params: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        with self._get_lock():
            self._ensure_connection()
            if self.conn is None:
                logger.warning(
                    "LadybugBackend.execute: connection could not be opened."
                )
                return []
            # Learn the node's table (for later edge binding), bind a
            # label-less edge write to typed Kuzu endpoints, and auto-create
            # any node table / rel pair the (bound) query needs. (KG-2.74)
            self._cache_node_label(query, params)
            q = self._bind_edge_query(query, params)
            self._ensure_schema_for_query(q)
            res = self.conn.execute(q, params or {})
            ret_rows = self._rows_from_result(res)
            # If transient mode is enabled, immediately close connection inside the lock
            if self.transient:
                self.close()
        return ret_rows

    def _close_after_error(self) -> None:
        # On error, make sure we close the connection inside the lock
        try:
            with self._get_lock():
                self.close()
        except Exception:
            pass

    def _is_lock_contention_error(self, exc: Exception, msg: str) -> bool:
        from filelock import Timeout as FileLockTimeout

        return (
            isinstance(exc, FileLockTimeout)
            or "lock" in msg
            or "busy" in msg
            or "database is locked" in msg
        )

    def _log_execute_error(self, exc: Exception, query: str, kind: str) -> None:
        if kind == "migration":
            logger.debug(f"LadybugDB expected migration error: {exc}")
        elif kind == "missing_table":
            logger.warning(f"LadybugDB table not found (check schema): {exc}")
        elif kind == "binder_expected":
            logger.debug(
                f"LadybugDB vector index or property missing (expected): {exc}"
            )
        elif kind == "binder_issue":
            logger.error(f"LadybugDB binder issue (invalid property?): {exc}")
        else:
            import traceback

            logger.error(
                f"LadybugDB Cypher execution failed: {exc}\n"
                f"Traceback: {traceback.format_exc()}\nQuery: {query}"
            )

    def _handle_execute_error(
        self, exc: Exception, query: str, max_retries: int
    ) -> list[dict[str, Any]]:
        msg = str(exc).lower()
        if self._is_lock_contention_error(exc, msg):
            # Trigger self-healing recovery before the policy retries
            # (exponential backoff with additive jitter, see below).
            self._recover_connection()
            logger.warning(
                f"Database locked or timeout (e={exc}), healed connection. "
                f"Retrying execute... (max {max_retries} attempts)"
            )
            raise LadybugLockContentionError(str(exc)) from exc
        self._log_execute_error(exc, query, _classify_execute_error(msg))
        return []

    def _execute_attempt(
        self, query: str, params: dict[str, Any] | None, max_retries: int
    ) -> list[dict[str, Any]]:
        try:
            return self._execute_attempt_body(query, params)
        except Exception as e:
            if self.transient:
                self._close_after_error()
            return self._handle_execute_error(e, query, max_retries)

    def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query on LadybugDB."""
        if include_epistemic:
            # CONCEPT:AU-KB-CURRENCY (Seam 1) — no id-seeded epistemic-envelope
            # primitive on this backend; degrade to ``[]`` per the ABC contract.
            logger.debug(
                "LadybugBackend.execute(include_epistemic=True): no epistemic "
                "envelope primitive; returning []"
            )
            return []

        import random

        from agent_utilities.orchestration.resilience import (
            ResiliencePolicy,
            run_with_resilience_sync,
        )

        max_retries = self.max_retries

        # Lock-contention backoff: (2**n)*0.1 + jitter, capped (CONCEPT:AU-ORCH.execution.retry-predicate-raised-treating).
        policy = ResiliencePolicy(
            max_attempts=max_retries,
            backoff_base_s=0.1,
            backoff_factor=2.0,
            max_backoff_s=_LOCK_BACKOFF_MAX_S,
            jitter=True,
            jitter_strategy="additive",
            retry_on=(LadybugLockContentionError,),
            name="ladybug-execute",
        )
        try:
            return run_with_resilience_sync(
                lambda: self._execute_attempt(query, params, max_retries),
                policy,
                rng=random.SystemRandom(),
            )
        except LadybugLockContentionError as exc:
            logger.error(
                f"Failed to execute query after {max_retries} retries due to "
                f"locking: {exc.__cause__ or exc}"
            )
            return []

    def execute_read(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute inside a Ladybug/Kuzu read-only transaction.

        The transient-connection close (test/ephemeral mode,
        ``AGENT_UTILITIES_TESTING``) must happen AFTER row extraction, not in
        the same ``finally`` as the transaction itself: ``result``/
        ``result_set`` are lazy handles onto the live connection —
        ``rows_as_dict()`` reads through them — so closing right after
        ``COMMIT`` (before that read) made every read in transient mode raise
        ``RuntimeError: Query result is closed`` instead of returning rows.
        This is load-bearing for ``secured_reads._durable_access_rows``, whose
        ACL-hydration read goes through this exact method — the ACL fallback
        this convergence lane wires up would itself have raised in every
        transient/test run without this fix.
        """
        if include_epistemic:
            return []
        try:
            with self._get_lock():
                self._ensure_connection()
                connection = self.conn
                if connection is None:
                    raise RuntimeError("LadybugDB read connection is unavailable")
                try:
                    connection.execute("BEGIN TRANSACTION READ ONLY")
                    result = connection.execute(query, params or {})
                    connection.execute("COMMIT")
                except Exception:
                    try:
                        connection.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise
            if isinstance(result, list):
                result_sets = result
            else:
                result_sets = [result]
            rows: list[dict[str, Any]] = []
            for result_set in result_sets:
                rows.extend(result_set.rows_as_dict().get_all())
            return rows
        finally:
            if self.transient:
                self.close()

    @staticmethod
    def _unwind_to_per_row(query: str) -> str:
        """Strip an ``UNWIND $batch AS row`` header and rewrite ``row.<k>`` /
        ``row.`<k>``` references to ``$<k>`` so each row runs as a normal
        parameterized statement (Kuzu has no UNWIND-over-param-list here). A
        non-UNWIND query is returned unchanged. (CONCEPT:AU-KG.backend.declared-columns-so-schema)"""
        import re

        q = (query or "").strip()
        m = re.match(r"(?is)^UNWIND\s+\$batch\s+AS\s+row\b(.*)$", q)
        if not m:
            return query
        body = m.group(1).strip()
        body = re.sub(r"row\.`([^`]+)`", r"$\1", body)
        body = re.sub(r"row\.(\w+)", r"$\1", body)
        return body

    def _execute_batch_row(
        self, query: str, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        # Per row: learn node labels, then bind a label-less
        # edge write to typed Kuzu endpoints (the (src→dst) pair
        # can differ per row within one rel-type batch). (KG-2.74)
        self._cache_node_label(query, params)
        q = self._bind_edge_query(query, params)
        if q is not query:
            self._ensure_schema_for_query(q)
        res = self.conn.execute(q, params or {})
        # ladybug return format: list of QueryResult objects
        if res and hasattr(res, "get_as_df"):
            df = res.get_as_df()
            return typing.cast(list[dict[str, Any]], df.to_dict("records"))
        return []

    def _execute_batch_chunk(
        self, query: str, chunk: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        """Run one chunk inside the lock; ``None`` means the connection wouldn't open."""
        results: list[dict[str, Any]] = []
        with self._get_lock():
            self._ensure_connection()
            if self.conn is None:
                logger.warning(
                    "LadybugBackend.execute_batch: connection could not be opened."
                )
                return None
            self._ensure_schema_for_query(query)
            for params in chunk:
                results.extend(self._execute_batch_row(query, params))
            # Close connection inside the lock in transient mode
            if self.transient:
                self.close()
        return results

    def _handle_batch_chunk_error(
        self, exc: Exception, attempt: int, max_retries: int
    ) -> tuple[int, bool]:
        """Classify one failed chunk attempt. Returns ``(new_attempt, should_retry)``."""
        if self.transient:
            self._close_after_error()
        msg = str(exc).lower()
        if not self._is_lock_contention_error(exc, msg):
            logger.warning(f"Batch execution chunk failed: {exc}")
            return attempt, False

        import secrets
        import time

        attempt += 1
        # Trigger self-healing recovery before retrying
        self._recover_connection()
        wait_time = (2**attempt) * 0.05 + secrets.SystemRandom().random() * 0.1
        logger.warning(
            f"Database locked or timeout during batch, healed connection. "
            f"Retrying chunk in {wait_time:.2f}s... (attempt {attempt}/{max_retries})"
        )
        time.sleep(wait_time)
        return attempt, True

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]], chunk_size: int = 500
    ) -> list[dict[str, Any]]:
        """Execute a batch query in chunks to avoid blocking the DB for too long."""
        # Bulk-writers emit ``UNWIND $batch AS row MERGE (n:L {id: row.id}) SET
        # n.`k` = row.`k` …``. Kuzu has no UNWIND-over-a-param-list here and the
        # per-row ``conn.execute`` below never bound ``$batch`` ("Parameter batch
        # not found"), so the whole batch no-op'd. Translate to the per-row
        # ``$param`` shape and run it per row — and ensure the node/rel tables for
        # the labels exist first (the raw ``conn.execute`` path skipped the schema
        # auto-create that ``execute`` does). (CONCEPT:AU-KG.backend.declared-columns-so-schema)
        query = self._unwind_to_per_row(query)

        results: list[dict[str, Any]] = []
        max_retries = self.max_retries
        for i in range(0, len(batch), chunk_size):
            chunk = batch[i : i + chunk_size]
            attempt = 0
            while attempt < max_retries:
                try:
                    chunk_results = self._execute_batch_chunk(query, chunk)
                    if chunk_results is not None:
                        results.extend(chunk_results)
                    break  # Success (or unopenable connection), move to next chunk
                except Exception as e:
                    attempt, should_retry = self._handle_batch_chunk_error(
                        e, attempt, max_retries
                    )
                    if should_retry:
                        continue
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
        except Exception as e:  # noqa: BLE001 — this except correctly returns False on failure (the `return True` above only executes on success), so callers cannot mistake a failed checkpoint for a completed one
            if self.transient:
                try:
                    with self._get_lock():
                        self.close()
                except Exception:
                    pass
            logger.debug(f"WAL checkpoint not supported or failed: {e}")  # noqa: BLE001 — this except correctly returns False on failure (see the return above only executes on success), so callers cannot mistake a failed checkpoint for a completed one
            return False

    def _seed_schema_cache(self) -> None:
        """Populate the known-tables caches from the declared SCHEMA so already-
        created tables/pairs never re-run DDL."""
        from agent_utilities.models.schema_definition import SCHEMA

        for node in SCHEMA.nodes:
            self._known_node_tables.add(node.name)
        for rel in SCHEMA.edges:
            for c in rel.connections:
                self._known_rel_pairs.add((rel.type, c["from"], c["to"]))
        self._schema_cache_seeded = True

    def _ensure_node_table_unlocked(self, label: str) -> None:
        """Create a generic node table for an unknown label (CONCEPT:AU-KG.backend.mirror-health-repair).

        Kuzu has fixed typed tables, so an undeclared label has no table and its
        MERGE fails. Create one with the canonical ``GENERIC_NODE_COLUMNS`` (the
        engine folds ad-hoc props into ``metadata`` for unknown labels, so these
        columns suffice). Must hold the connection lock; conn must be open."""
        if label in self._known_node_tables or self.conn is None:
            return
        label = validate_identifier(label, kind="label")
        from agent_utilities.models.schema_definition import GENERIC_NODE_COLUMNS

        all_columns = dict(GENERIC_NODE_COLUMNS)
        for gname, gtype in _GOVERNANCE_COLUMNS.items():
            all_columns.setdefault(gname, gtype)
        cols = ", ".join(f"`{n}` {t}" for n, t in all_columns.items())
        try:
            self.conn.execute(f"CREATE NODE TABLE IF NOT EXISTS {label} ({cols});")
        except Exception as e:  # noqa: BLE001
            if "exist" not in str(e).lower():
                logger.warning("auto-create node table %s failed: %s", label, e)
        self._known_node_tables.add(label)

    def _create_rel_table_fallback(self, rel: str, src: str, dst: str) -> None:
        # Re-validated here (not just trusted from the caller) so this
        # interpolation site is safe by construction on its own, regardless
        # of which call path reaches it.
        rel = validate_identifier(rel, kind="relationship type")
        src = validate_identifier(src, kind="label")
        dst = validate_identifier(dst, kind="label")
        try:
            self.conn.execute(
                f"CREATE REL TABLE IF NOT EXISTS {rel} "
                f"(FROM {src} TO {dst}, properties STRING);"
            )
        except Exception as e2:  # noqa: BLE001
            if "exist" not in str(e2).lower():
                logger.warning("auto-create rel %s failed: %s", rel, e2)

    def _handle_rel_pair_alter_failure(
        self, exc: Exception, rel: str, src: str, dst: str
    ) -> None:
        msg = str(exc).lower()
        if "does not exist" in msg or "not found" in msg:
            self._create_rel_table_fallback(rel, src, dst)
            return
        if not any(w in msg for w in ("already", "exist", "duplicate")):
            logger.warning("alter rel %s (%s->%s) failed: %s", rel, src, dst, exc)

    def _ensure_rel_pair_unlocked(self, rel: str, src: str, dst: str) -> None:
        """Ensure REL table ``rel`` carries the ``(src)->(dst)`` pair. Kuzu REL
        tables are typed by their FROM/TO node pairs, so an arbitrary edge needs the
        pair added (``ALTER TABLE .. ADD FROM .. TO ..``) or the table created. Try
        ALTER first; create the table if it does not exist yet."""
        key = (rel, src, dst)
        if key in self._known_rel_pairs or self.conn is None:
            return
        rel = validate_identifier(rel, kind="relationship type")
        src = validate_identifier(src, kind="label")
        dst = validate_identifier(dst, kind="label")
        try:
            self.conn.execute(f"ALTER TABLE {rel} ADD FROM {src} TO {dst};")
        except Exception as e:  # noqa: BLE001
            self._handle_rel_pair_alter_failure(e, rel, src, dst)
        self._known_rel_pairs.add(key)

    def _ensure_schema_for_query(self, query: str) -> None:
        """Before a MERGE/CREATE write, auto-create any node table / rel pair the
        query references that does not exist yet — so an arbitrary KG (labels/rels
        beyond the declared SCHEMA) mirrors into Kuzu losslessly."""
        up = query.upper()
        if "MERGE" not in up and "CREATE" not in up:
            return
        node_labels = _NODE_LABEL_RE.findall(query)
        if not node_labels:
            return
        if not self._schema_cache_seeded:
            self._seed_schema_cache()
        for lbl in node_labels:
            self._ensure_node_table_unlocked(lbl)
        m = _REL_TYPE_RE.search(query)
        if m and len(node_labels) >= 2:
            # The edge query lists (s:Src {..}), (t:Dst {..}) in order.
            self._ensure_rel_pair_unlocked(m.group(1), node_labels[0], node_labels[1])

    def _cache_node_label(self, query: str, params: dict[str, Any] | None) -> None:
        """Learn ``id → node-table`` from a node MERGE so a later label-less edge
        write can bind its endpoints. Cheap regex; no-op for non-node-MERGE."""
        m = re.search(
            r"MERGE\s*\(\s*\w+\s*:\s*`?(\w+)`?\s*\{\s*id\s*:\s*\$(\w+)\s*\}",
            query,
            re.I,
        )
        if not m:
            return
        nid = (params or {}).get(m.group(2))
        if nid is not None:
            self._node_labels[str(nid)] = m.group(1)

    def _resolve_node_label(self, node_id: Any) -> str | None:
        """Resolve a node's Kuzu table by id — cache first, else scan the known
        node tables (bounded; result cached). ``None`` if the node isn't present."""
        if node_id is None or self.conn is None:
            return None
        nid = str(node_id)
        cached = self._node_labels.get(nid)
        if cached:
            return cached
        if not self._schema_cache_seeded:
            self._seed_schema_cache()
        for label in list(self._known_node_tables):
            try:
                label = validate_identifier(label, kind="label")
                res = self.conn.execute(
                    f"MATCH (n:{label} {{id: $id}}) RETURN n.id LIMIT 1", {"id": nid}
                )
            except Exception:  # noqa: BLE001 — table may not exist / transient / invalid label
                continue
            rows = (
                res.get_as_df().to_dict("records")
                if (res is not None and hasattr(res, "get_as_df"))
                else []
            )
            if rows:
                self._node_labels[nid] = label
                return label
        return None

    def _parse_edge_query_endpoints(
        self, query: str
    ) -> tuple[str, str, str, str, str] | None:
        """Parse a label-less two-endpoint edge write; None when not that shape."""
        up = query.upper()
        if "->" not in query or ("MERGE" not in up and "CREATE" not in up):
            return None
        m = re.search(
            r"MATCH\s*\(\s*(\w+)\s*\{\s*id\s*:\s*\$(\w+)\s*\}\s*\)\s*"
            r"MATCH\s*\(\s*(\w+)\s*\{\s*id\s*:\s*\$(\w+)\s*\}\s*\)",
            query,
            re.I,
        )
        if not m:
            return None
        svar, sidp, tvar, tidp = m.groups()
        rel_m = re.search(r"-\s*\[\s*\w*\s*:\s*`?(\w+)`?[^\]]*\]\s*->", query, re.I)
        if not rel_m:
            return None
        return svar, sidp, tvar, tidp, rel_m.group(1)

    def _resolve_edge_endpoint_labels(
        self, params: dict[str, Any] | None, sidp: str, tidp: str
    ) -> tuple[str, str] | None:
        src_label = self._resolve_node_label((params or {}).get(sidp))
        dst_label = self._resolve_node_label((params or {}).get(tidp))
        if not src_label or not dst_label:
            return None
        return src_label, dst_label

    def _validate_edge_bind_identifiers(
        self, svar: str, tvar: str, src_label: str, dst_label: str
    ) -> tuple[str, str, str, str] | None:
        try:
            svar = validate_identifier(svar, kind="variable")
            tvar = validate_identifier(tvar, kind="variable")
            src_label = validate_identifier(src_label, kind="label")
            dst_label = validate_identifier(dst_label, kind="label")
        except InvalidIdentifierError:
            return None
        return svar, tvar, src_label, dst_label

    def _bind_edge_query(self, query: str, params: dict[str, Any] | None) -> str:
        """Bind a label-less edge write to typed Kuzu endpoints (CONCEPT:AU-KG.backend.mirror-health-repair).

        Ingest emits ``MATCH (s {id:$source}) MATCH (t {id:$target}) MERGE
        (s)-[r:REL]->(t)`` — but Kuzu cannot create a rel without knowing the
        endpoints' node tables ("Create rel bound by multiple node labels is not
        supported"). Resolve each endpoint's label by id, ensure the rel carries
        that ``(src→dst)`` pair, and inject the labels into the two MATCH clauses.
        Returns the query unchanged when it is not a label-less two-endpoint edge
        write, or when an endpoint can't be resolved (so the normal path surfaces
        a clear error instead of a silent mis-bind)."""
        parsed = self._parse_edge_query_endpoints(query)
        if parsed is None:
            return query
        svar, sidp, tvar, tidp, rel_type = parsed
        labels = self._resolve_edge_endpoint_labels(params, sidp, tidp)
        if labels is None:
            return query
        src_label, dst_label = labels
        validated = self._validate_edge_bind_identifiers(
            svar, tvar, src_label, dst_label
        )
        if validated is None:
            return query
        svar, tvar, src_label, dst_label = validated
        self._ensure_rel_pair_unlocked(rel_type, src_label, dst_label)
        bound = re.sub(
            rf"MATCH\s*\(\s*{svar}\s*\{{",
            f"MATCH ({svar}:{src_label} {{",
            query,
            count=1,
            flags=re.I,
        )
        bound = re.sub(
            rf"MATCH\s*\(\s*{tvar}\s*\{{",
            f"MATCH ({tvar}:{dst_label} {{",
            bound,
            count=1,
            flags=re.I,
        )
        return bound

    def _node_table_columns(self, node: Any) -> dict[str, str] | None:
        """Validated ``{column: dtype}`` for one schema node (+ governance/tenant_id).

        Every node table carries `tenant_id`, mirroring PostgreSQLBackend's
        `ensure_label_table` (backends/postgresql_backend.py, the RLS_GUC /
        "app.tenant_id" tenant isolation) — CONCEPT:AU-KG.query.object-graph-mapper.
        None of ladybug's schema_definition.py TableDefinitions declare it
        (it's engine-injected, not hand-authored per table), but the mandatory
        tenant-scoping chokepoint (company_brain.scope_cypher_query,
        KG-2.6 "the primary boundary") unconditionally injects a
        `<var>.tenant_id = '<tenant>'` predicate into EVERY Cypher read this
        backend serves, regardless of label — a table missing the column made
        that a hard `Binder exception: Cannot find property tenant_id`
        instead of the intended tenant filter. Returns None on an invalid name.
        """
        try:
            col_names = {
                validate_identifier(name, kind="column"): dtype
                for name, dtype in node.columns.items()
            }
            for gname, gtype in _GOVERNANCE_COLUMNS.items():
                col_names.setdefault(validate_identifier(gname, kind="column"), gtype)
        except InvalidIdentifierError:
            logger.warning("skipping node table with an invalid schema name")
            return None
        col_names.setdefault("tenant_id", "STRING")
        return col_names

    def _create_node_table(self, node_name: str, col_names: dict[str, str]) -> None:
        # Re-validated here so this interpolation site is safe by
        # construction on its own, independent of the caller.
        node_name = validate_identifier(node_name, kind="table")
        cols = ", ".join(f"`{name}` {dtype}" for name, dtype in col_names.items())
        stmt = f"CREATE NODE TABLE IF NOT EXISTS {node_name} ({cols});"
        try:
            self.conn.execute(stmt)
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"Node table creation issue ({node_name}): {e}")

    def _migrate_node_table_columns(
        self, node_name: str, col_names: dict[str, str]
    ) -> None:
        # Best-effort migration: add any newly-declared columns to a
        # pre-existing node table (CREATE..IF NOT EXISTS won't alter it). The
        # PK and embedding can't be added post-hoc; skip them. Mirrors the rel
        # ``properties`` ALTER below so an existing DB gains new columns (e.g.
        # the KG-2.9g code-symbol columns) instead of erroring on projection.
        # Re-validated here (not just trusted from the caller) so this
        # interpolation site is safe by construction on its own.
        node_name = validate_identifier(node_name, kind="table")
        for cname, ctype in col_names.items():
            if "PRIMARY KEY" in ctype.upper() or cname == "embedding":
                continue
            try:
                self.conn.execute(f"ALTER TABLE {node_name} ADD `{cname}` {ctype};")
            except Exception:  # noqa: BLE001 — already present / unsupported → ignore
                pass

    def _create_node_tables_unlocked(self) -> None:
        # 1. Create Node Tables
        for node in SCHEMA.nodes:
            try:
                node_name = validate_identifier(node.name, kind="table")
            except InvalidIdentifierError:
                logger.warning("skipping node table with an invalid schema name")
                continue
            col_names = self._node_table_columns(node)
            if col_names is None:
                continue
            self._create_node_table(node_name, col_names)
            self._migrate_node_table_columns(node_name, col_names)

    def _rel_table_connections(
        self, rel: Any
    ) -> tuple[str, list[tuple[str, str]]] | None:
        try:
            rel_type = validate_identifier(rel.type, kind="relationship type")
            connections = [
                (
                    validate_identifier(c["from"], kind="label"),
                    validate_identifier(c["to"], kind="label"),
                )
                for c in rel.connections
            ]
        except InvalidIdentifierError:
            logger.warning("skipping rel table with an invalid schema name")
            return None
        return rel_type, connections

    def _create_rel_table(
        self, rel_type: str, connections: list[tuple[str, str]]
    ) -> None:
        # Re-validated here so this interpolation site is safe by
        # construction on its own, independent of the caller.
        rel_type = validate_identifier(rel_type, kind="relationship type")
        connections = [
            (
                validate_identifier(frm, kind="label"),
                validate_identifier(to, kind="label"),
            )
            for frm, to in connections
        ]
        conns = ", ".join(f"FROM {frm} TO {to}" for frm, to in connections)
        stmt = (
            f"CREATE REL TABLE IF NOT EXISTS {rel_type} ({conns}, properties STRING);"
        )
        try:
            self.conn.execute(stmt)
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"Rel table creation issue ({rel_type}): {e}")
        # Best-effort migration: add the column to pre-existing rel tables.
        try:
            self.conn.execute(f"ALTER TABLE {rel_type} ADD properties STRING;")
        except Exception:  # noqa: BLE001 — already present / unsupported → ignore
            pass

    def _create_rel_tables_unlocked(self) -> None:
        # 2. Create Rel Tables. Every rel table carries a single JSON ``properties``
        # column so edges persist their properties (confidence/source/bitemporal
        # stamps/inferred flags) — Kuzu REL tables otherwise drop edge props, which
        # was a data-loss gap vs the schemaless backends (CONCEPT:AU-KG.query.vendor-agnostic-traversal parity).
        for rel in SCHEMA.edges:
            parsed = self._rel_table_connections(rel)
            if parsed is None:
                continue
            rel_type, connections = parsed
            self._create_rel_table(rel_type, connections)

    def _create_schema_unlocked(self) -> None:
        """Internal method to synchronize schema without acquiring the connection lock."""
        if self.conn is None:
            return
        self._create_node_tables_unlocked()
        self._create_rel_tables_unlocked()

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

    def _embedding_tables(self, tables: list[str] | None) -> list[str]:
        """Schema node names carrying a FLOAT ``embedding`` column, optionally filtered."""
        embedding_tables = [
            node.name
            for node in SCHEMA.nodes
            if "embedding" in node.columns
            and "FLOAT" in node.columns["embedding"].upper()
        ]
        if tables:
            embedding_tables = [t for t in embedding_tables if t in tables]
        return embedding_tables

    def _try_load_vector_extension_for_indices(self, table_count: int) -> bool:
        try:
            self.conn.execute("INSTALL VECTOR;")
            self.conn.execute("LOAD EXTENSION VECTOR;")
            return True
        except Exception as e:  # noqa: BLE001 — feature-detection, see build_vector_indices
            logger.info(
                "LadybugDB VECTOR extension unavailable; skipping vector "
                "index DDL for %d embedding table(s): %s",
                table_count,
                e,
            )
            return False

    def _create_vector_index_for_table(self, table: str) -> str | None:
        """Create one table's vector index. Returns a stop reason to end the sweep, else None."""
        try:
            table = validate_identifier(table, kind="table")
        except InvalidIdentifierError as e:
            logger.warning("skipping invalid embedding table: %s", e)
            return None
        idx_name = f"idx_{table.lower()}_embedding"
        stmt = f"CALL CREATE_VECTOR_INDEX('{table}', '{idx_name}', 'embedding');"
        try:
            self.conn.execute(stmt)
        except Exception as e:
            msg = str(e)
            if "already exists" in msg.lower():
                return None
            if "FLOAT/DOUBLE ARRAY" in msg:
                return msg
            logger.warning(f"Vector index creation issue ({idx_name}): {e}")
        return None

    def _create_vector_indices_for_tables(self, embedding_tables: list[str]) -> None:
        skip_reason: str | None = None
        for table in embedding_tables:
            stop_reason = self._create_vector_index_for_table(table)
            if stop_reason is not None:
                skip_reason = stop_reason
                break
        if skip_reason is not None:
            logger.info(
                "LadybugDB vector indexes skipped for %d table(s): %s. "
                "Define embedding columns as FLOAT[N] (fixed size) to "
                "enable HNSW indexing.",
                len(embedding_tables),
                skip_reason,
            )

    def build_vector_indices(self, tables: list[str] | None = None) -> None:
        """Create Vector Indices for any FLOAT column named 'embedding'.

        Note: LadybugDB (Kuzu) currently does not support updating properties
        (via SET) that are part of a vector index. Therefore, vector indices
        should only be built AFTER all initial ingestion is complete.

        Args:
            tables: Optional list of specific table names to build indexes for.
                When None, builds for all tables with embedding columns.
        """
        embedding_tables = self._embedding_tables(tables)
        if not embedding_tables:
            return
        with self._get_lock():
            self._ensure_connection()
            if self.conn is None:
                logger.warning(
                    "LadybugBackend.build_vector_indices: connection could not be opened."
                )
                return
            if self._try_load_vector_extension_for_indices(len(embedding_tables)):
                self._create_vector_indices_for_tables(embedding_tables)
            if self.transient:
                self.close()

    def _drop_vector_index_for_table(
        self, table: str, failed_tables: list[str]
    ) -> bool:
        """Drop one table's vector index; True when actually dropped."""
        try:
            table = validate_identifier(table, kind="table")
        except InvalidIdentifierError as e:  # noqa: BLE001 — narrow typed exception (InvalidIdentifierError, not a broad swallow): an unrecognized/invalid embedding table name is skipped and excluded from the drop attempt entirely, rather than being silently treated as dropped — the D-DSTK fix below (raising when a real DROP_VECTOR_INDEX call fails) only covers tables that reach that call
            logger.debug("skipping invalid embedding table: %s", e)
            return False
        idx_name = f"idx_{table.lower()}_embedding"
        try:
            self.conn.execute(f"CALL DROP_VECTOR_INDEX('{table}', '{idx_name}');")
            return True
        except Exception as e:  # noqa: BLE001 — per-table detail stays at debug; genuine failures are now aggregated into `failed_tables` and raised below, so the caller no longer mistakes this for success
            if (
                "not found" not in str(e).lower()
                and "does not exist" not in str(e).lower()
            ):
                logger.debug(f"Drop vector index issue ({idx_name}): {e}")
                failed_tables.append(table)
            return False

    def _drop_vector_indices_for_tables(
        self, embedding_tables: list[str]
    ) -> tuple[int, list[str]]:
        dropped = 0
        failed_tables: list[str] = []
        for table in embedding_tables:
            if self._drop_vector_index_for_table(table, failed_tables):
                dropped += 1
        return dropped, failed_tables

    def drop_vector_indices(self, tables: list[str] | None = None) -> None:
        """Drop HNSW vector indexes so that embedding SET operations succeed.

        Must be called before ingestion if indexes were previously built,
        since LadybugDB (Kuzu) does not support SET on indexed columns.

        Args:
            tables: Optional list of specific table names to drop indexes for.
                When None, drops all embedding indexes.
        """
        embedding_tables = self._embedding_tables(tables)
        # D-DSTK: tables whose DROP_VECTOR_INDEX call failed for a REAL reason (not the
        # benign "not found"/"does not exist" — already gone). engine_tasks.py's
        # submit_task (D-DST-3) only marks a table's index dropped AFTER this method
        # returns without raising, specifically so a failed drop is retried next
        # submission — but this method previously swallowed every per-table failure
        # internally and never raised, so it always "succeeded" from the caller's view
        # even when every single drop had genuinely failed. Raising here (after best-
        # effort attempting every table) closes that gap.
        with self._get_lock():
            self._ensure_connection()
            if self.conn is None:
                logger.warning(
                    "LadybugBackend.drop_vector_indices: connection could not be opened."
                )
                return
            dropped, failed_tables = self._drop_vector_indices_for_tables(
                embedding_tables
            )
            if self.transient:
                self.close()
        if dropped:
            logger.info("Dropped %d HNSW vector indexes for re-ingestion.", dropped)
        if failed_tables:
            raise RuntimeError(
                f"drop_vector_indices: {len(failed_tables)} table(s) could not be "
                f"confirmed dropped: {failed_tables}"
            )

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Add embedding to an existing node."""
        query = "MATCH (n {id: $id}) SET n.embedding = $emb"
        # The _get_lock is inside self.execute()
        self.execute(query, {"id": node_id, "emb": embedding})

    def verify_node_embedding(self, node_id: str, embedding: list[float]) -> bool:
        """Confirm the Kuzu vector through its fail-loud read transaction."""
        label = self._resolve_node_label(node_id)
        if not label:
            return False
        label = validate_identifier(label, kind="label")
        rows = self.execute_read(
            f"MATCH (n:{label} {{id: $id}}) RETURN n.embedding AS embedding",
            {"id": node_id},
        )
        return bool(rows) and embedding_values_match(
            rows[0].get("embedding"), embedding
        )

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
        label = f":{validate_identifier(node_type, kind='label')}" if node_type else ""

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
