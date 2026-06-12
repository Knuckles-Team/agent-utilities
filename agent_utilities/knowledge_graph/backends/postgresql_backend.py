"""PostgreSQL + pgGraph + pgvector + ParadeDB Backend (CONCEPT:OS-5.0).

Production-grade backend combining:
- PostgreSQL relational tables for node/edge storage
- pgGraph extension for graph traversal (BFS/DFS, shortest path)
- pgvector for embedding storage and cosine similarity search
- ParadeDB BM25 / native FTS for lexical search
- psycopg connection pooling for multi-agent concurrency

Requires: pip install agent-utilities[postgresql]
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from agent_utilities.core.config import config, setting

from .base import GraphBackend

logger = logging.getLogger(__name__)

# Embedding dimension from env (must match model output)
_EMBEDDING_DIM = int(config.kg_embedding_dim or "768")


class PostgresLockContentionError(ConnectionError):
    """PostgreSQL lock/deadlock contention — retryable with exponential backoff."""


class PostgreSQLBackend(GraphBackend):
    """PostgreSQL backend with pgGraph, pgvector, and ParadeDB.

    Args:
        dsn: PostgreSQL connection string.
        graph_name: Logical graph name (used for pgGraph schema prefix).
        pool_min: Minimum pool connections.
        pool_max: Maximum pool connections.
        pggraph_schema: Schema name for pgGraph registration.
    """

    @property
    def cypher_support(self) -> str:
        """Regex Cypher→SQL transpiler: only the bounded operational subset the
        engine emits runs here (CONCEPT:KG-2.63). The AGE subclass overrides this
        back to ``"full"`` for native openCypher."""
        return "subset"

    def __init__(
        self,
        dsn: str = "postgresql://localhost:5432/agent_utilities",
        graph_name: str = "agent_graph",
        pool_min: int = 2,
        pool_max: int = 10,
        pggraph_schema: str | None = None,
    ) -> None:
        self._dsn = dsn
        self._graph_name = graph_name
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._pggraph_schema = pggraph_schema or setting(
            "GRAPH_PGGRAPH_SCHEMA", "public"
        )
        self._pool: Any = None
        self._known_tables: set[str] = set()
        self._pggraph_available: bool | None = None  # lazy check
        self._pgvector_available: bool | None = None
        self._paradedb_available: bool | None = None
        logger.info("PostgreSQLBackend initialized (dsn=%s, graph=%s)", dsn, graph_name)

    # ── Connection Pool ─────────────────────────────────────────────

    def _ensure_pool(self) -> Any:
        """Lazy pool initialization with retry."""
        if self._pool is not None:
            return self._pool
        try:
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(
                self._dsn,
                min_size=self._pool_min,
                max_size=self._pool_max,
                open=True,
                kwargs={"autocommit": False},
            )
            logger.info(
                "PostgreSQL connection pool opened (min=%d, max=%d)",
                self._pool_min,
                self._pool_max,
            )
        except ImportError:
            # Fallback to single connection if pool not installed
            import psycopg

            self._pool = _SingleConnPool(psycopg.connect(self._dsn))
            logger.info("PostgreSQL single connection (psycopg_pool not installed)")
        return self._pool

    @contextmanager
    def _conn(self):
        """Get a connection from the pool."""
        pool = self._ensure_pool()
        if isinstance(pool, _SingleConnPool):
            yield pool.conn
            try:
                pool.conn.commit()
            except Exception:
                pool.conn.rollback()
        else:
            with pool.connection() as conn:
                yield conn

    # ── Extension Detection ──────────────────────────────────────────

    def _check_extension(self, name: str) -> bool:
        """Check if a PostgreSQL extension is available."""
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM pg_extension WHERE extname = %s", (name,)
                    )
                    return cur.fetchone() is not None
        except Exception:
            return False

    @property
    def pggraph_available(self) -> bool:
        if self._pggraph_available is None:
            # The graph extension on the pggraph image is Apache AGE (not an
            # extension literally named "pggraph"); fall back to the legacy name.
            self._pggraph_available = self._check_extension(
                "age"
            ) or self._check_extension("pggraph")
            if self._pggraph_available:
                logger.info("graph extension (AGE) detected")
            else:
                logger.info(
                    "graph extension (AGE) not available — graph traversal disabled"
                )
        return self._pggraph_available

    @property
    def pgvector_available(self) -> bool:
        if self._pgvector_available is None:
            self._pgvector_available = self._check_extension("vector")
        return self._pgvector_available

    @property
    def paradedb_available(self) -> bool:
        if self._paradedb_available is None:
            self._paradedb_available = self._check_extension("pg_search")
        return self._paradedb_available

    # ── Schema Management ────────────────────────────────────────────

    def create_schema(self) -> None:
        """Create PostgreSQL tables from the unified graph schema."""
        from agent_utilities.models.schema_definition import SCHEMA

        with self._conn() as conn:
            with conn.cursor() as cur:
                # Enable extensions
                for ext in ("vector", "pg_trgm"):
                    try:
                        cur.execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")
                    except Exception as e:
                        logger.debug("Extension %s not available: %s", ext, e)
                        conn.rollback()

                # Create node tables
                for table_def in SCHEMA.nodes:
                    cols = self._translate_columns(table_def.columns)
                    ddl = f'CREATE TABLE IF NOT EXISTS "{table_def.name}" ({cols})'
                    try:
                        cur.execute(ddl)
                    except Exception as e:
                        logger.debug("Table %s DDL error: %s", table_def.name, e)
                        conn.rollback()
                        continue  # don't cache a table whose CREATE failed
                    self._known_tables.add(table_def.name)

                # Create unified edge table
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kg_edges (
                        source_id TEXT NOT NULL,
                        target_id TEXT NOT NULL,
                        rel_type TEXT NOT NULL,
                        properties JSONB DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (source_id, target_id, rel_type)
                    )
                """
                )

                # Create indexes on edge table
                for idx_sql in (
                    "CREATE INDEX IF NOT EXISTS idx_edges_source ON kg_edges(source_id)",
                    "CREATE INDEX IF NOT EXISTS idx_edges_target ON kg_edges(target_id)",
                    "CREATE INDEX IF NOT EXISTS idx_edges_type ON kg_edges(rel_type)",
                ):
                    try:
                        cur.execute(idx_sql)
                    except Exception:
                        conn.rollback()

                conn.commit()

        # Register with pgGraph if available
        if self.pggraph_available:
            self._register_pggraph()

        logger.info(
            "PostgreSQL schema initialized (%d tables)", len(self._known_tables)
        )

    def ensure_label_table(self, label: str, force: bool = False) -> bool:
        """Auto-DDL: ensure a durable table exists for node ``label`` (self-healing).

        The durable tier ships tables only for types in the static schema; a node
        of a NEW type (e.g. a freshly-introduced ``SDD_Feature``) would otherwise
        silently fail to persist (``relation ... does not exist``). This creates a
        minimal universal node table on demand and registers it with the transpiler
        so the durable tier **self-extends to any type** instead of dropping it.
        Idempotent + cheap after the first call (guarded by ``_known_tables``).

        ``force`` bypasses the ``_known_tables`` cache: the self-heal path calls it
        with ``force=True`` because the database has just told us the relation does
        NOT exist, so the in-memory cache is authoritatively stale (it can claim a
        table whose CREATE silently failed). The ``CREATE TABLE IF NOT EXISTS`` is
        idempotent, so re-running it when the cache is wrong is safe + cheap.
        """
        import re as _re

        name = _re.sub(r"\W+", "_", str(label or "Node")).strip("_") or "Node"
        if not force and name in self._known_tables:
            return True
        ddl = (
            f'CREATE TABLE IF NOT EXISTS "{name}" ('
            "id TEXT PRIMARY KEY, name TEXT, type TEXT, "
            "properties JSONB DEFAULT '{}'::jsonb, "
            "created_at TIMESTAMPTZ DEFAULT NOW())"
        )
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(ddl)
                    conn.commit()
            self._known_tables.add(name)
            logger.info("auto-DDL: ensured durable table for label '%s'", name)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("ensure_label_table(%s) failed: %s", name, e)
            return False

    def ensure_column(self, table: str, column: str) -> bool:
        """Auto-DDL: ensure ``column`` exists on ``table`` (schema-drift self-heal).

        Schema tables ship with a fixed column set, but a write may reference a
        column the table lacks (``column ... does not exist``). This adds it as a
        nullable ``TEXT`` column so the write succeeds instead of being dropped —
        the column-level companion to :meth:`ensure_label_table`.
        """
        import re as _re

        t = _re.sub(r"\W+", "_", str(table or "")).strip("_")
        c = _re.sub(r"\W+", "_", str(column or "")).strip("_")
        if not t or not c:
            return False
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f'ALTER TABLE "{t}" ADD COLUMN IF NOT EXISTS "{c}" TEXT'
                    )
                    conn.commit()
            logger.info("auto-DDL: added column '%s' to durable table '%s'", c, t)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("ensure_column(%s,%s) failed: %s", t, c, e)
            return False

    def edge_count(self) -> int | None:
        """Durable edge count (``kg_edges``) — for exact drift metrics.

        Returns ``None`` if unavailable (the Cypher edge-count form does not
        transpile, so callers needing an exact figure use this direct count).
        """
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*) FROM kg_edges")
                    return int(cur.fetchone()[0])
        except Exception as e:  # noqa: BLE001
            logger.debug("edge_count failed: %s", e)
            return None

    # ── Row-Level Security: DB-level tenant isolation (CONCEPT:KG-2.61) ──
    #
    # Defense-in-depth BENEATH the KG-2.58 named-graph partition: even a
    # hand-crafted Cypher/SQL that forgets the tenant predicate cannot read
    # another org's rows, because Postgres itself filters them. The contract is
    # a per-session GUC ``app.tenant_id``:
    #
    #   * set to an org id  → that org's rows + commons (tenant_id '' / NULL)
    #   * unset or empty    → unrestricted (the platform-admin / system path)
    #
    # ``FORCE ROW LEVEL SECURITY`` is required because the app's DB role usually
    # owns the tables (owners otherwise bypass RLS).
    RLS_GUC = "app.tenant_id"

    @staticmethod
    def rls_statements(table: str) -> list[str]:
        """Idempotent DDL enabling tenant-isolation RLS on one table.

        Pure (no I/O) so it is unit-testable and reusable by the migration. The
        policy admits the row when it belongs to the session's tenant, is commons
        (empty/NULL tenant), or no tenant scope is set (privileged/legacy path).
        """
        guc = PostgreSQLBackend.RLS_GUC
        q = f'"{table}"'
        cond = (
            f"(tenant_id = current_setting('{guc}', true) "
            f"OR tenant_id IS NULL OR tenant_id = '' "
            f"OR current_setting('{guc}', true) IS NULL "
            f"OR current_setting('{guc}', true) = '')"
        )
        return [
            f"ALTER TABLE {q} ADD COLUMN IF NOT EXISTS tenant_id TEXT",
            f"ALTER TABLE {q} ENABLE ROW LEVEL SECURITY",
            f"ALTER TABLE {q} FORCE ROW LEVEL SECURITY",
            f"DROP POLICY IF EXISTS tenant_isolation ON {q}",
            f"CREATE POLICY tenant_isolation ON {q} USING {cond} WITH CHECK {cond}",
            f'CREATE INDEX IF NOT EXISTS "idx_{table}_tenant" ON {q}(tenant_id)',
        ]

    @staticmethod
    def tenant_guc_sql(tenant_id: str | None) -> str:
        """The ``SET LOCAL app.tenant_id`` statement for a request's tenant.

        Empty/None → ``''`` (no restriction, privileged path). The value is
        single-quote-escaped; callers pass a server-minted tenant id.
        """
        safe = (tenant_id or "").replace("'", "''")
        return f"SET LOCAL {PostgreSQLBackend.RLS_GUC} = '{safe}'"

    def enable_row_level_security(self) -> int:
        """Apply tenant-isolation RLS to every node table + ``kg_edges``.

        Idempotent and self-contained (run once after ``create_schema`` or via
        the ``deploy/postgres/tenant_rls.sql`` migration). Returns the number of
        tables secured. A per-table failure is logged and skipped so a single
        problem table never aborts the whole rollout.
        """
        tables = sorted(self._known_tables | {"kg_edges"})
        secured = 0
        with self._conn() as conn:
            for table in tables:
                try:
                    with conn.cursor() as cur:
                        for stmt in self.rls_statements(table):
                            cur.execute(stmt)
                    conn.commit()
                    secured += 1
                except Exception as e:  # noqa: BLE001 — skip the bad table, keep going
                    logger.warning("RLS enable failed for %s: %s", table, e)
                    conn.rollback()
        logger.info("Row-level security enabled on %d tables", secured)
        return secured

    def set_request_tenant(self, conn: Any, tenant_id: str | None) -> None:
        """Scope a connection/transaction to ``tenant_id`` via the RLS GUC."""
        try:
            with conn.cursor() as cur:
                cur.execute(self.tenant_guc_sql(tenant_id))
        except Exception as e:  # noqa: BLE001 — scoping must not crash the query
            logger.debug("set_request_tenant failed: %s", e)

    def _translate_columns(self, columns: dict[str, str]) -> str:
        """Translate schema column definitions to PostgreSQL DDL."""
        type_map = {
            "STRING": "TEXT",
            "STRING PRIMARY KEY": "TEXT PRIMARY KEY",
            "INT64": "BIGINT",
            "FLOAT": "DOUBLE PRECISION",
            "BOOLEAN": "BOOLEAN",
            "STRING[]": "TEXT[]",
        }
        parts = []
        for col_name, col_type in columns.items():
            if col_type.startswith("FLOAT["):
                # Embedding column → pgvector
                dim = col_type.split("[")[1].rstrip("]")
                pg_type = (
                    f"vector({dim})"
                    if self.pgvector_available
                    else "DOUBLE PRECISION[]"
                )
            else:
                pg_type = type_map.get(col_type, "TEXT")
            parts.append(f'"{col_name}" {pg_type}')
        return ", ".join(parts)

    def _register_pggraph(self) -> None:
        """Register tables and edges with pgGraph extension."""
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    for tbl in sorted(self._known_tables):
                        # Get searchable text columns
                        searchable = self._get_text_columns(cur, tbl)
                        cols_array = (
                            "ARRAY[" + ", ".join(f"'{c}'" for c in searchable) + "]"
                            if searchable
                            else "NULL"
                        )
                        try:
                            cur.execute(
                                f"""
                                SELECT graph.add_table(
                                    table_name := '{self._pggraph_schema}.{tbl}'::regclass,
                                    id_column := 'id',
                                    columns := {cols_array}
                                )
                            """
                            )
                        except Exception as e:
                            logger.debug("pgGraph add_table %s failed: %s", tbl, e)
                            conn.rollback()

                    # Register edge table as edges
                    try:
                        cur.execute(
                            """
                            SELECT graph.add_edge(
                                from_table := 'kg_edges'::regclass,
                                from_column := 'source_id',
                                to_table := 'kg_edges'::regclass,
                                to_column := 'target_id',
                                label := 'rel_type',
                                label_column := 'rel_type',
                                bidirectional := true
                            )
                        """
                        )
                    except Exception as e:
                        logger.debug("pgGraph edge registration failed: %s", e)
                        conn.rollback()

                    # Build the graph index
                    try:
                        cur.execute("SELECT * FROM graph.build()")
                        conn.commit()
                        logger.info("pgGraph index built successfully")
                    except Exception as e:
                        logger.debug("pgGraph build failed: %s", e)
                        conn.rollback()
        except Exception as e:
            logger.warning("pgGraph registration failed (non-fatal): %s", e)

    def _get_text_columns(self, cur: Any, table: str) -> list[str]:
        """Get TEXT columns for a table (for pgGraph search registration)."""
        target = ("name", "description", "content", "summary", "type", "status")
        try:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = %s AND data_type = 'text' "
                "AND column_name = ANY(%s)",
                (table, list(target)),
            )
            return [row[0] for row in cur.fetchall()]
        except Exception:
            return []

    # ── Cypher Execution (Transpiled to SQL) ─────────────────────────

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query by transpiling to SQL."""
        from agent_utilities.orchestration.resilience import (
            ResiliencePolicy,
            RetryableError,
            run_with_resilience_sync,
        )

        from .cypher_transpiler import QueryType, transpile

        params = params or {}
        tq = transpile(query, params, self._known_tables)

        if tq.query_type == QueryType.UNKNOWN:
            logger.debug("Skipping unknown Cypher pattern: %.120s", query)
            return []

        max_retries = 3
        attempts_used = 0

        def _attempt() -> list[dict[str, Any]]:
            nonlocal attempts_used
            attempts_used += 1
            try:
                with self._conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(tq.sql, tq.params)

                        if tq.query_type in (
                            QueryType.SELECT,
                            QueryType.LABEL_LOOKUP,
                            QueryType.COUNT,
                        ):
                            cols = (
                                [desc.name for desc in cur.description]
                                if cur.description
                                else []
                            )
                            rows = cur.fetchall()
                            results = []
                            for row in rows:
                                d = dict(zip(cols, row, strict=False))
                                # Wrap in node alias if engine expects {n: {...}}
                                if tq.node_alias and tq.query_type == QueryType.SELECT:
                                    results.append({tq.node_alias: d})
                                else:
                                    results.append(d)
                            return results

                        elif tq.query_type == QueryType.UPDATE:
                            if cur.description:
                                cols = [desc.name for desc in cur.description]
                                rows = cur.fetchall()
                                return [dict(zip(cols, r, strict=False)) for r in rows]
                            return [{"affected": cur.rowcount}] if cur.rowcount else []

                        elif tq.query_type in (
                            QueryType.INSERT,
                            QueryType.UPSERT_EDGE,
                            QueryType.DELETE,
                        ):
                            conn.commit()
                            return []

                        return []
            except Exception as e:
                msg = str(e).lower()
                if ("lock" in msg or "deadlock" in msg) and attempts_used < max_retries:
                    logger.warning("PG locked, retrying (attempt %d)", attempts_used)
                    raise PostgresLockContentionError(str(e)) from e
                # Auto-DDL self-heal: a write to a not-yet-created type table
                # ("relation X does not exist") creates the table and retries; a
                # write to a missing column ("column X of relation Y does not
                # exist") adds the column and retries — so a new node type or a
                # schema-drifted property persists durably instead of being dropped.
                import re as _re

                healed = False
                mc = _re.search(
                    r'column "([^"]+)" of relation "([^"]+)" does not exist', str(e)
                )
                if mc and attempts_used < max_retries:
                    healed = self.ensure_column(mc.group(2), mc.group(1))
                else:
                    mt = _re.search(r'relation "([^"]+)" does not exist', str(e))
                    if mt and attempts_used < max_retries:
                        # force=True: the DB just told us the table is missing, so
                        # the _known_tables cache is authoritatively stale here.
                        healed = self.ensure_label_table(mt.group(1), force=True)
                if healed:
                    # Schema just healed — retry immediately (backoff_s=0.0).
                    raise RetryableError(str(e), backoff_s=0.0) from e
                logger.error("PostgreSQL execute error: %s | SQL: %.200s", e, tq.sql)
                return []

        # Historical lock backoff (2**n)*0.1s, no jitter; healed-schema retries
        # carry a 0.0s delay hint so they stay immediate (CONCEPT:ORCH-1.36).
        policy = ResiliencePolicy(
            max_attempts=max_retries,
            backoff_base_s=0.1,
            backoff_factor=2.0,
            jitter=False,
            retry_on=(PostgresLockContentionError, RetryableError),
            name="postgresql-execute",
        )
        try:
            return run_with_resilience_sync(_attempt, policy)
        except (PostgresLockContentionError, RetryableError):
            return []  # All retries exhausted (guards make this unreachable)

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query over a batch of parameters."""
        from .cypher_transpiler import QueryType, transpile

        if not batch:
            return []

        results: list[dict[str, Any]] = []
        chunk_size = 500

        for i in range(0, len(batch), chunk_size):
            chunk = batch[i : i + chunk_size]
            try:
                with self._conn() as conn:
                    with conn.cursor() as cur:
                        for params in chunk:
                            tq = transpile(query, params, self._known_tables)
                            if tq.query_type == QueryType.UNKNOWN:
                                continue
                            cur.execute(tq.sql, tq.params)
                    conn.commit()
            except Exception as e:
                logger.error("Batch execute error at chunk %d: %s", i, e)

        return results

    # ── Vector Operations ────────────────────────────────────────────

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Store embedding in the node's vector column."""
        if not self.pgvector_available:
            logger.debug("pgvector not available — skipping embedding for %s", node_id)
            return

        # Find which table has this node
        table = self._find_node_table(node_id)
        if not table:
            return

        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f'UPDATE "{table}" SET embedding = %s::vector WHERE id = %s',
                        (str(embedding), node_id),
                    )
                    conn.commit()
        except Exception as e:
            logger.debug("add_embedding failed for %s: %s", node_id, e)

    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Cosine similarity search across all tables with embeddings."""
        if not self.pgvector_available:
            return []

        results: list[dict[str, Any]] = []
        embedding_str = str(query_embedding)

        # Tables known to have embedding columns
        embedding_tables = self._get_embedding_tables()

        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    for tbl in embedding_tables:
                        try:
                            cur.execute(
                                f"SELECT *, 1 - (embedding <=> %s::vector) AS _similarity "
                                f'FROM "{tbl}" WHERE embedding IS NOT NULL '
                                f"ORDER BY embedding <=> %s::vector LIMIT %s",  # nosec B608
                                (embedding_str, embedding_str, n_results),
                            )
                            cols = [d.name for d in cur.description]
                            for row in cur.fetchall():
                                d = dict(zip(cols, row, strict=False))
                                d["_table_label"] = tbl
                                results.append(d)
                        except Exception:
                            continue  # nosec B112
        except Exception as e:
            logger.error("semantic_search error: %s", e)

        results.sort(key=lambda x: x.get("_similarity", 0), reverse=True)
        return results[:n_results]

    def lexical_search(self, query: str, n_results: int = 10) -> list[dict[str, Any]]:
        """BM25/FTS search using ParadeDB or native PostgreSQL FTS."""
        results: list[dict[str, Any]] = []
        text_tables = [
            t
            for t in self._known_tables
            if t
            in (
                "Article",
                "Memory",
                "Code",
                "Agent",
                "Tool",
                "Skill",
                "KBConcept",
                "KBFact",
                "Message",
                "Concept",
            )
        ]

        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    for tbl in text_tables:
                        try:
                            if self.paradedb_available:
                                cur.execute(
                                    f"""SELECT *, paradedb.rank_bm25(id) AS _score
                                    FROM "{tbl}"
                                    WHERE "{tbl}" @@@ paradedb.parse(%s)
                                    ORDER BY _score DESC LIMIT %s""",  # nosec B608
                                    (query, n_results),
                                )
                            else:
                                # Native FTS fallback
                                cur.execute(
                                    f"""SELECT *, ts_rank(
                                        to_tsvector('english', COALESCE(name,'') || ' ' || COALESCE(description,'') || ' ' || COALESCE(content,'')),
                                        plainto_tsquery('english', %s)
                                    ) AS _score
                                    FROM "{tbl}"
                                    WHERE to_tsvector('english', COALESCE(name,'') || ' ' || COALESCE(description,'') || ' ' || COALESCE(content,''))
                                        @@ plainto_tsquery('english', %s)
                                    ORDER BY _score DESC LIMIT %s""",  # nosec B608
                                    (query, query, n_results),
                                )
                            cols = [d.name for d in cur.description]
                            for row in cur.fetchall():
                                d = dict(zip(cols, row, strict=False))
                                d["_table_label"] = tbl
                                results.append(d)
                        except Exception:
                            continue  # nosec B112
        except Exception as e:
            logger.debug("lexical_search error: %s", e)

        results.sort(key=lambda x: x.get("_score", 0), reverse=True)
        return results[:n_results]

    def build_vector_indices(self) -> None:
        """Create HNSW indexes on embedding columns."""
        if not self.pgvector_available:
            return
        for tbl in self._get_embedding_tables():
            try:
                with self._conn() as conn:
                    with conn.cursor() as cur:
                        idx_name = f"idx_{tbl.lower()}_embedding_hnsw"
                        cur.execute(
                            f"CREATE INDEX IF NOT EXISTS {idx_name} "
                            f'ON "{tbl}" USING hnsw (embedding vector_cosine_ops)'
                        )
                        conn.commit()
                        logger.info("HNSW index created on %s", tbl)
            except Exception as e:
                logger.debug("HNSW index on %s failed: %s", tbl, e)

    # ── pgGraph Operations ───────────────────────────────────────────

    def graph_traverse(
        self,
        seed_table: str,
        seed_id: str,
        max_depth: int = 2,
        direction: str = "any",
        hydrate: bool = True,
        max_rows: int = 100,
    ) -> list[dict[str, Any]]:
        """Traverse the graph using pgGraph's CSR index."""
        if not self.pggraph_available:
            return []
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT * FROM graph.traverse(
                            seed_table := %s::regclass,
                            seed_id := %s,
                            max_depth := %s,
                            direction := %s,
                            hydrate := %s,
                            max_rows := %s
                        )""",
                        (
                            f"{self._pggraph_schema}.{seed_table}",
                            seed_id,
                            max_depth,
                            direction,
                            hydrate,
                            max_rows,
                        ),
                    )
                    cols = [d.name for d in cur.description]
                    return [
                        dict(zip(cols, row, strict=False)) for row in cur.fetchall()
                    ]
        except Exception as e:
            logger.error("pgGraph traverse error: %s", e)
            return []

    def graph_shortest_path(
        self,
        source_table: str,
        source_id: str,
        target_table: str,
        target_id: str,
        max_depth: int = 20,
    ) -> list[dict[str, Any]]:
        """Find shortest path using pgGraph."""
        if not self.pggraph_available:
            return []
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT * FROM graph.shortest_path(
                            %s::regclass, %s,
                            %s::regclass, %s,
                            max_depth := %s, hydrate := true
                        )""",
                        (
                            f"{self._pggraph_schema}.{source_table}",
                            source_id,
                            f"{self._pggraph_schema}.{target_table}",
                            target_id,
                            max_depth,
                        ),
                    )
                    cols = [d.name for d in cur.description]
                    return [
                        dict(zip(cols, row, strict=False)) for row in cur.fetchall()
                    ]
        except Exception as e:
            logger.error("pgGraph shortest_path error: %s", e)
            return []

    def graph_search(
        self,
        property_key: str,
        property_value: str,
        table_filter: str | None = None,
        max_rows: int = 100,
    ) -> list[dict[str, Any]]:
        """Search graph nodes using pgGraph's search API."""
        if not self.pggraph_available:
            return []
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    tbl_clause = (
                        f"table_filter := '{self._pggraph_schema}.{table_filter}'::regclass,"
                        if table_filter
                        else ""
                    )
                    cur.execute(
                        f"""
                        SELECT * FROM graph.search(
                            property_key := %s,
                            property_value := %s,
                            {tbl_clause}
                            mode := 'contains',
                            case_sensitive := false,
                            max_rows := %s,
                            hydrate := true
                        )
                    """,  # nosec B608
                        (property_key, property_value, max_rows),
                    )
                    cols = [d.name for d in cur.description]
                    return [
                        dict(zip(cols, row, strict=False)) for row in cur.fetchall()
                    ]
        except Exception as e:
            logger.error("pgGraph search error: %s", e)
            return []

    def graph_build(self) -> dict[str, Any]:
        """Rebuild the pgGraph CSR index."""
        if not self.pggraph_available:
            return {"status": "skipped", "reason": "pgGraph not available"}
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM graph.build()")
                    cols = [d.name for d in cur.description]
                    row = cur.fetchone()
                    conn.commit()
                    return dict(zip(cols, row, strict=False)) if row else {}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def graph_status(self) -> dict[str, Any]:
        """Get pgGraph engine status."""
        if not self.pggraph_available:
            return {"available": False}
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM graph.status()")
                    cols = [d.name for d in cur.description]
                    row = cur.fetchone()
                    return dict(zip(cols, row, strict=False)) if row else {}
        except Exception as e:
            return {"available": True, "error": str(e)}

    # ── Pruning ──────────────────────────────────────────────────────

    def prune(self, criteria: dict[str, Any]) -> None:
        """Prune nodes by timestamp or importance_score threshold."""
        max_age = criteria.get("max_age")
        min_importance = criteria.get("min_importance")

        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    for tbl in self._known_tables:
                        conditions = []
                        values: list[Any] = []
                        if max_age:
                            conditions.append('"timestamp" < %s')
                            values.append(max_age)
                        if min_importance is not None:
                            conditions.append('"importance_score" < %s')
                            values.append(min_importance)
                        # Never prune permanent nodes
                        conditions.append('COALESCE("is_permanent", false) = false')

                        if conditions:
                            where = " AND ".join(conditions)
                            # Cascade: delete edges first
                            cur.execute(
                                f"DELETE FROM kg_edges WHERE source_id IN "
                                f'(SELECT id FROM "{tbl}" WHERE {where}) '
                                f"OR target_id IN "
                                f'(SELECT id FROM "{tbl}" WHERE {where})',
                                values + values,
                            )
                            cur.execute(f'DELETE FROM "{tbl}" WHERE {where}', values)
                    conn.commit()
                    logger.info("Pruning complete with criteria: %s", criteria)
        except Exception as e:
            logger.error("Prune error: %s", e)

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            if isinstance(self._pool, _SingleConnPool):
                self._pool.close()
            else:
                self._pool.close()
            self._pool = None
            logger.info("PostgreSQL connection pool closed")

    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    def get_stats(self) -> dict[str, Any]:
        """Return database statistics."""
        stats: dict[str, Any] = {
            "backend": "postgresql",
            "graph_name": self._graph_name,
            "tables": len(self._known_tables),
            "pgvector": self.pgvector_available,
            "pggraph": self.pggraph_available,
            "paradedb": self.paradedb_available,
        }
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    total_nodes = 0
                    for tbl in self._known_tables:
                        try:
                            cur.execute(f'SELECT COUNT(*) FROM "{tbl}"')
                            total_nodes += cur.fetchone()[0]
                        except Exception:
                            continue  # nosec B112
                    cur.execute("SELECT COUNT(*) FROM kg_edges")
                    total_edges = cur.fetchone()[0]
                    stats["total_nodes"] = total_nodes
                    stats["total_edges"] = total_edges
        except Exception:
            pass  # nosec B110

        if self.pggraph_available:
            stats["pggraph_status"] = self.graph_status()

        return stats

    # ── Helpers ───────────────────────────────────────────────────────

    def _find_node_table(self, node_id: str) -> str | None:
        """Find which table contains a node by ID."""
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    for tbl in self._known_tables:
                        cur.execute(
                            f'SELECT 1 FROM "{tbl}" WHERE id = %s LIMIT 1',
                            (node_id,),
                        )
                        if cur.fetchone():
                            return tbl
        except Exception:
            pass  # nosec B110
        return None

    def _get_embedding_tables(self) -> list[str]:
        """Get tables that have an embedding column."""
        tables = []
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT table_name FROM information_schema.columns "
                        "WHERE column_name = 'embedding' AND table_schema = %s",
                        (self._pggraph_schema,),
                    )
                    # When this instance hasn't loaded the schema (read/mirror
                    # backend), ``_known_tables`` is empty — don't filter it out,
                    # use the live column catalog (mirrors the transpiler's
                    # ``not known_tables`` behavior). Otherwise semantic_search
                    # silently finds 0 tables. (CONCEPT:KG-2.7)
                    tables = [
                        r[0]
                        for r in cur.fetchall()
                        if not self._known_tables or r[0] in self._known_tables
                    ]
        except Exception:
            pass  # nosec B110
        return tables


class _SingleConnPool:
    """Minimal shim when psycopg_pool is not installed."""

    def __init__(self, conn: Any):
        self.conn = conn

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass  # nosec B110
