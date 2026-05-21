#!/usr/bin/python
"""LadybugDB Graph Backend.

CONCEPT:KG-2.0

This module provides the LadybugDB implementation of the GraphBackend interface,
supporting strict schema-bound Cypher queries.
"""

import fcntl
import logging
import os
import typing
from contextlib import contextmanager
from typing import Any

from .base import GraphBackend

try:
    import ladybug

    LADYBUG_AVAILABLE = True
except ImportError:
    LADYBUG_AVAILABLE = False

from agent_utilities.models.schema_definition import SCHEMA

logger = logging.getLogger(__name__)


class LadybugBackend(GraphBackend):
    """LadybugDB backend implementation."""

    def _get_lock(self):
        """Get a cross-process pessimistic lock for the database."""
        if self.db_path == ":memory:":

            @contextmanager
            def no_op_lock():
                yield

            return no_op_lock()

        @contextmanager
        def file_lock():
            lock_path = f"{self.db_path}.lock"
            with open(lock_path, "w") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    yield
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

        return file_lock()

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
        # Use Database and Connection objects as required by newer ladybug versions
        # Add retry logic with jitter for multi-agent startup resilience
        import random
        import time

        last_error: Exception = RuntimeError("Max retries exceeded")
        for attempt in range(max_retries):
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

                self.db = ladybug.Database(
                    db_path if db_path != ":memory:" else None,
                    **db_params,  # type: ignore[arg-type]
                )
                self.conn = ladybug.Connection(self.db)

                # Apply WAL pragmas if supported (SQLite underlying)
                try:
                    self.conn.execute("PRAGMA journal_mode=WAL;")
                    self.conn.execute("PRAGMA synchronous=NORMAL;")
                    self.conn.execute("PRAGMA busy_timeout=10000;")
                except Exception as e:
                    logger.debug(f"WAL pragma not supported or ignored: {e}")

                # Load VECTOR extension for HNSW vector search
                try:
                    self.conn.execute("INSTALL VECTOR;")
                    self.conn.execute("LOAD EXTENSION VECTOR;")
                    logger.debug("LadybugDB VECTOR extension loaded successfully")
                except Exception as ve:
                    logger.debug(f"Could not load VECTOR extension: {ve}")

                # Successful connection — backup only if recovering
                if attempt > 0:
                    self._backup_db()
                return
            except Exception as e:
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
                    or "no such file or directory" in msg
                ):
                    logger.warning(
                        f"Detected database corruption, stale shadow, or WAL error in {db_path} "
                        f"(usually caused by a hard restart or process crash). "
                        f"Self-healing: cleaning up stale WAL/shadow/lock files and retrying."
                    )
                    self._backup_db()  # Safeguard before cleanup
                    self._cleanup_corrupted()
                    # retry immediately after cleanup
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
                    if attempt == max_retries - 1:
                        raise e

                    wait_time = (2**attempt) + random.random()  # nosec B311
                    logger.warning(
                        f"Graph DB locked or catalog race detected, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                else:
                    raise e
        raise last_error

    def close(self) -> None:
        """Close the database connection and database object."""
        if hasattr(self, "conn"):
            try:
                self.conn.close()
            except Exception:  # nosec B110
                pass
            del self.conn
        if hasattr(self, "db"):
            del self.db

    def __del__(self) -> None:
        """Ensure connection is destroyed before database to avoid C++ Kuzu abort."""
        try:
            self.close()
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

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query on LadybugDB."""
        import time

        max_retries = 5
        last_error = None

        for attempt in range(max_retries):
            try:
                if not hasattr(self, "conn"):
                    logger.warning(
                        "LadybugBackend.execute: connection closed, returning empty."
                    )
                    return []
                with self._get_lock():
                    res = self.conn.execute(query, params or {})
                    if isinstance(res, list):
                        if not res:
                            return []
                        res = res[0]
                # ladybug returns a QueryResult object. We want a list of dicts.
                from typing import cast

                return cast(list[dict[str, Any]], res.rows_as_dict().get_all())
            except Exception as e:
                msg = str(e).lower()
                if "lock" in msg or "busy" in msg or "database is locked" in msg:
                    import secrets

                    wait_time = (
                        2**attempt
                    ) * 0.1 + secrets.SystemRandom().random() * 0.1
                    logger.warning(
                        f"Database locked, retrying execute in {wait_time:.2f}s... (attempt {attempt + 1}/{max_retries})"
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
                    if "doesn't have an index with name" in msg:
                        logger.debug(f"LadybugDB vector index missing (expected): {e}")
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
        import secrets
        import time

        results = []
        max_retries = 10
        for i in range(0, len(batch), chunk_size):
            chunk = batch[i : i + chunk_size]
            attempt = 0
            while attempt < max_retries:
                try:
                    if not hasattr(self, "conn"):
                        logger.warning(
                            "LadybugBackend.execute_batch: connection closed, skipping chunk."
                        )
                        break
                    with self._get_lock():
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
                    break  # Success, move to next chunk
                except Exception as e:
                    msg = str(e).lower()
                    if "lock" in msg or "busy" in msg or "database is locked" in msg:
                        attempt += 1
                        wait_time = (
                            2**attempt
                        ) * 0.05 + secrets.SystemRandom().random() * 0.1
                        logger.warning(
                            f"Database locked during batch, retrying chunk in {wait_time:.2f}s... (attempt {attempt}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    logger.warning(f"Batch execution chunk failed: {e}")
                    break
        return results

    def wal_checkpoint(self) -> bool:
        """Perform a WAL checkpoint if the underlying engine supports it."""
        try:
            if not hasattr(self, "conn"):
                return False
            self.conn.execute("CHECKPOINT;")
            return True
        except Exception as e:
            logger.debug(f"WAL checkpoint not supported or failed: {e}")
            return False

    def create_schema(self) -> None:
        """Create LadybugDB schema from the unified schema definition.
        Ladybug requires strict DDL for Node and Rel tables.
        """
        logger.info(
            f"Synchronizing Knowledge Graph Schema ({len(SCHEMA.nodes)} node tables, {len(SCHEMA.edges)} edge tables)..."
        )
        # 1. Create Node Tables
        for node in SCHEMA.nodes:
            cols = ", ".join(
                [f"{name} {dtype}" for name, dtype in node.columns.items()]
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
                        logger.warning(f"Vector index creation issue ({idx_name}): {e}")
                if skip_reason is not None:
                    logger.info(
                        "LadybugDB vector indexes skipped for %d table(s): %s. "
                        "Define embedding columns as FLOAT[N] (fixed size) to "
                        "enable HNSW indexing.",
                        len(embedding_tables),
                        skip_reason,
                    )

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
        for table in embedding_tables:
            idx_name = f"idx_{table.lower()}_embedding"
            try:
                self.conn.execute(f"CALL DROP_VECTOR_INDEX('{table}', '{idx_name}');")
                dropped += 1
            except Exception as e:
                if (
                    "not found" not in str(e).lower()
                    and "does not exist" not in str(e).lower()
                ):
                    logger.debug(f"Drop vector index issue ({idx_name}): {e}")
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
        try:
            if not hasattr(self, "conn"):
                return
            self.conn.execute("CHECKPOINT;")
            logger.debug("WAL checkpoint completed for %s", self.db_path)
        except Exception as e:
            logger.warning("WAL checkpoint failed for %s: %s", self.db_path, e)
