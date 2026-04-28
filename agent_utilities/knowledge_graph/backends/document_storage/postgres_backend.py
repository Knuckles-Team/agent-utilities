"""
PostgreSQL Backend for Document Storage.

Implements document storage using PostgreSQL with unified ID support.
"""

from typing import Any

from .base import DocumentDB


class PostgreSQLBackend(DocumentDB):
    """
    PostgreSQL backend for document storage.

    Uses PostgreSQL with JSONB support for metadata.
    Supports unified ID system.

    Note: This requires psycopg2 to be installed.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "documents",
        user: str = "postgres",
        password: str = "",  # nosec B107
    ):
        """
        Initialize PostgreSQL connection.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
        """
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            self.psycopg2 = psycopg2
            self.RealDictCursor = RealDictCursor
        except ImportError as e:
            raise ImportError(
                "PostgreSQL backend requires psycopg2. "
                "Install with: pip install psycopg2-binary"
            ) from e

        self.conn = self.psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            cursor_factory=self.RealDictCursor,
        )

        self._initialize_schema()

    def _validate_identifier(self, identifier: str) -> None:
        """Validate SQL identifier to prevent injection."""
        if not identifier.isidentifier():
            raise ValueError(f"Invalid SQL identifier: {identifier}")

    def _initialize_schema(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata JSONB DEFAULT "'{}'"::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_deleted BOOLEAN DEFAULT FALSE,
                deleted_at TIMESTAMP
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_id ON documents(id);")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_created ON documents(created_at);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_deleted ON documents(is_deleted);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_metadata ON documents USING gin(metadata);"
        )
        self.conn.commit()

    def create_collection(self, collection_name: str) -> None:
        """Create a document collection."""
        self._validate_identifier(collection_name)
        cursor = self.conn.cursor()
        cursor.execute(  # nosec B608
            f"""
            CREATE TABLE IF NOT EXISTS {collection_name} (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata JSONB DEFAULT "'{{}}'"::jsonb,
                parent_doc_id TEXT,
                chunk_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
        )
        cursor.execute(  # nosec B608
            f"CREATE INDEX IF NOT EXISTS idx_{collection_name}_id ON {collection_name}(id);"
        )
        cursor.execute(  # nosec B608
            f"CREATE INDEX IF NOT EXISTS idx_{collection_name}_parent ON {collection_name}(parent_doc_id);"
        )
        self.conn.commit()

    def insert_document(self, document: dict[str, Any], collection_name: str) -> str:
        """Insert a document into a collection."""
        unified_id = document.get("id")
        if not unified_id:
            raise ValueError("Document must have 'id' field")

        if not unified_id.startswith("doc_"):
            raise ValueError(
                f"Invalid unified ID: {unified_id}. Must start with 'doc_'"
            )

        cursor = self.conn.cursor()

        if collection_name == "documents":
            cursor.execute(
                """
                INSERT INTO documents (id, content, metadata, created_at, updated_at, is_deleted, deleted_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at,
                    is_deleted = EXCLUDED.is_deleted,
                    deleted_at = EXCLUDED.deleted_at
                """,
                (
                    unified_id,
                    document.get("content", ""),
                    document.get("metadata", {}),
                    document.get("created_at"),
                    document.get("updated_at"),
                    document.get("is_deleted", False),
                    document.get("deleted_at"),
                ),
            )
        else:
            self._validate_identifier(collection_name)
            self.create_collection(collection_name)
            query = f"""
                INSERT INTO {collection_name} (id, content, metadata, parent_doc_id, chunk_index, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
                """  # nosec B608
            cursor.execute(
                query,
                (
                    unified_id,
                    document.get("content", ""),
                    document.get("metadata", {}),
                    document.get("parent_doc_id"),
                    document.get("chunk_index"),
                    document.get("created_at"),
                    document.get("updated_at"),
                ),
            )

        self.conn.commit()
        return unified_id

    def insert_many(
        self, documents: list[dict[str, Any]], collection_name: str
    ) -> list[str]:
        """Insert multiple documents into a collection."""
        doc_ids = []
        for doc in documents:
            doc_ids.append(self.insert_document(doc, collection_name))
        return doc_ids

    def find_document(self, doc_id: str, collection_name: str) -> dict[str, Any] | None:
        """Find a specific document by ID."""
        cursor = self.conn.cursor()

        if collection_name == "documents":
            cursor.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
        else:
            self._validate_identifier(collection_name)
            query = f"SELECT * FROM {collection_name} WHERE id = %s"  # nosec B608
            cursor.execute(query, (doc_id,))

        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def find_documents(
        self, filter_dict: dict[str, Any], collection_name: str
    ) -> list[dict[str, Any]]:
        """Find documents matching a filter."""
        cursor = self.conn.cursor()

        # Build WHERE clause from filter
        where_clauses = []
        params = []

        for key, value in filter_dict.items():
            if key == "metadata":
                # JSONB query for metadata
                for meta_key, meta_value in value.items():
                    where_clauses.append(f"metadata->>'{meta_key}' = %s")
                    params.append(meta_value)
            else:
                where_clauses.append(f"{key} = %s")
                params.append(value)

        where_clause = " AND ".join(where_clauses) if where_clauses else "TRUE"

        if collection_name == "documents":
            query = f"SELECT * FROM documents WHERE {where_clause}"  # nosec B608
            cursor.execute(query, tuple(params))
        else:
            self._validate_identifier(collection_name)
            query = f"SELECT * FROM {collection_name} WHERE {where_clause}"  # nosec B608
            cursor.execute(query, tuple(params))

        return [dict(row) for row in cursor.fetchall()]

    def update_document(
        self, doc_id: str, updates: dict[str, Any], collection_name: str
    ) -> bool:
        """Update a document."""
        cursor = self.conn.cursor()

        # Build SET clause
        set_clauses = []
        params = []

        for key, value in updates.items():
            set_clauses.append(f"{key} = %s")
            params.append(value)

        params.append(doc_id)

        if collection_name == "documents":
            query = f"UPDATE documents SET {', '.join(set_clauses)} WHERE id = %s"  # nosec B608
            cursor.execute(query, tuple(params))
        else:
            self._validate_identifier(collection_name)
            query = (
                f"UPDATE {collection_name} SET {', '.join(set_clauses)} WHERE id = %s"  # nosec B608
            )
            cursor.execute(query, tuple(params))

        self.conn.commit()
        return cursor.rowcount > 0

    def delete_document(self, doc_id: str, collection_name: str) -> bool:
        """Delete a document."""
        cursor = self.conn.cursor()

        if collection_name == "documents":
            cursor.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        else:
            self._validate_identifier(collection_name)
            query = f"DELETE FROM {collection_name} WHERE id = %s"  # nosec B608
            cursor.execute(query, (doc_id,))

        self.conn.commit()
        return cursor.rowcount > 0

    def count_documents(self, collection_name: str) -> int:
        """Count documents in a collection."""
        cursor = self.conn.cursor()

        if collection_name == "documents":
            cursor.execute("SELECT COUNT(*) FROM documents")
        else:
            self._validate_identifier(collection_name)
            cursor.execute(f"SELECT COUNT(*) FROM {collection_name}")  # nosec B608

        return cursor.fetchone()[0]

    def aggregate(
        self, pipeline: list[dict[str, Any]], collection_name: str
    ) -> list[dict[str, Any]]:
        """Execute an aggregation pipeline."""
        # PostgreSQL supports aggregation through various methods
        # This is a simplified implementation

        results = []

        if pipeline and pipeline[0].get("$group"):
            # Simple grouping
            group_field = pipeline[0]["$group"].replace("$", "")
            cursor = self.conn.cursor()

            if collection_name == "documents":
                self._validate_identifier(group_field)
                query = f"SELECT {group_field}, COUNT(*) as count FROM documents GROUP BY {group_field}"  # nosec B608
                cursor.execute(query)
            else:
                self._validate_identifier(collection_name)
                self._validate_identifier(group_field)
                query = f"SELECT {group_field}, COUNT(*) as count FROM {collection_name} GROUP BY {group_field}"  # nosec B608
                cursor.execute(query)

            for row in cursor.fetchall():
                results.append({"group": dict(row), "count": row["count"]})

        return results

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
