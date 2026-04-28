"""
SQLite Backend for Document Storage.

Implements document storage using SQLite with unified ID support.
"""

import json
import sqlite3
from datetime import datetime
from typing import Any

from .base import DocumentDB


class SQLiteMemoryBackend(DocumentDB):
    """
    In-memory SQLite backend for document storage.

    Uses SQLite in-memory database for development and testing.
    Supports unified ID system and JSON metadata storage.
    """

    def __init__(self):
        """Initialize in-memory SQLite database."""
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self._initialize_schema()

    def _validate_collection_name(self, name: str) -> None:
        """Validate collection name to prevent SQL injection."""
        if not name.isidentifier():
            raise ValueError(f"Invalid collection name: {name}")

    def _initialize_schema(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_deleted BOOLEAN DEFAULT FALSE,
                deleted_at TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_id ON documents(id)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_created ON documents(created_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_deleted ON documents(is_deleted)"
        )
        self.conn.commit()

    def create_collection(self, collection_name: str) -> None:
        """
        Create a document collection.

        SQLite doesn't require explicit collection creation.
        Collections are implemented as separate tables.

        Args:
            collection_name: Name of the collection
        """
        cursor = self.conn.cursor()

        if collection_name == "documents":
            # Documents collection has specific schema
            # Schema is already created in _initialize_schema
            pass
        else:
            # Other collections (chunks, etc.) have extended schema
            self._validate_collection_name(collection_name)
            cursor.execute(  # nosec B608
                f"""
                CREATE TABLE IF NOT EXISTS {collection_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    metadata TEXT,
                    parent_doc_id TEXT,
                    chunk_index INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            cursor.execute(  # nosec B608
                f"CREATE INDEX IF NOT EXISTS idx_{collection_name}_id ON {collection_name}(id)"
            )
            cursor.execute(  # nosec B608
                f"CREATE INDEX IF NOT EXISTS idx_{collection_name}_parent ON {collection_name}(parent_doc_id)"
            )
            self.conn.commit()

    def insert_document(self, document: dict[str, Any], collection_name: str) -> str:
        """
        Insert a document into a collection.

        Args:
            document: Document dictionary (must include 'id' field)
            collection_name: Name of the collection

        Returns:
            str: Document ID
        """
        unified_id = document.get("id")
        if not unified_id:
            raise ValueError("Document must have 'id' field")

        # Validate unified ID format
        if not unified_id.startswith("doc_"):
            raise ValueError(
                f"Invalid unified ID: {unified_id}. Must start with 'doc_'"
            )

        # Prepare document for insertion
        doc_data = {
            "id": unified_id,
            "content": document.get("content", ""),
            "metadata": json.dumps(document.get("metadata", {})),
            "created_at": document.get("created_at", datetime.now().isoformat()),
            "updated_at": document.get("updated_at", datetime.now().isoformat()),
            "is_deleted": document.get("is_deleted", False),
            "deleted_at": document.get("deleted_at"),
        }

        # Handle collection-specific fields
        if collection_name == "chunks":
            doc_data["parent_doc_id"] = document.get("parent_doc_id")
            doc_data["chunk_index"] = document.get("chunk_index")

        # Build INSERT query
        if collection_name == "documents":
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO documents (id, content, metadata, created_at, updated_at, is_deleted, deleted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_data["id"],
                    doc_data["content"],
                    doc_data["metadata"],
                    doc_data["created_at"],
                    doc_data["updated_at"],
                    doc_data["is_deleted"],
                    doc_data["deleted_at"],
                ),
            )
        else:
            # For other collections (chunks, etc.)
            self._validate_collection_name(collection_name)
            self.create_collection(collection_name)
            cursor = self.conn.cursor()
            query = f"""
                INSERT INTO {collection_name} (id, content, metadata, parent_doc_id, chunk_index, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """  # nosec B608
            cursor.execute(
                query,
                (
                    doc_data["id"],
                    doc_data["content"],
                    doc_data["metadata"],
                    doc_data.get("parent_doc_id"),
                    doc_data.get("chunk_index"),
                    doc_data["created_at"],
                    doc_data["updated_at"],
                ),
            )

        self.conn.commit()
        return unified_id

    def insert_many(
        self, documents: list[dict[str, Any]], collection_name: str
    ) -> list[str]:
        """
        Insert multiple documents into a collection.

        Args:
            documents: List of document dictionaries
            collection_name: Name of the collection

        Returns:
            List[str]: List of document IDs
        """
        doc_ids = []
        for doc in documents:
            doc_ids.append(self.insert_document(doc, collection_name))
        return doc_ids

    def find_document(self, doc_id: str, collection_name: str) -> dict[str, Any] | None:
        """
        Find a specific document by ID.

        Args:
            doc_id: Document ID
            collection_name: Name of the collection

        Returns:
            Optional[Dict]: Document dictionary, or None if not found
        """
        cursor = self.conn.cursor()

        if collection_name == "documents":
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        else:
            self._validate_collection_name(collection_name)
            cursor.execute(f"SELECT * FROM {collection_name} WHERE id = ?", (doc_id,))  # nosec B608

        row = cursor.fetchone()
        if row:
            doc = dict(row)
            # Parse JSON metadata
            if "metadata" in doc and doc["metadata"]:
                doc["metadata"] = json.loads(doc["metadata"])
            return doc
        return None

    def find_documents(
        self, filter_dict: dict[str, Any], collection_name: str
    ) -> list[dict[str, Any]]:
        """
        Find documents matching a filter.

        Args:
            filter_dict: Dictionary of filter criteria
            collection_name: Name of the collection

        Returns:
            List[Dict]: List of matching documents
        """
        cursor = self.conn.cursor()

        if collection_name == "documents":
            cursor.execute("SELECT * FROM documents")
        else:
            self._validate_collection_name(collection_name)
            cursor.execute(f"SELECT * FROM {collection_name}")  # nosec B608

        results = []
        for row in cursor.fetchall():
            doc = dict(row)
            # Parse JSON metadata
            if "metadata" in doc and doc["metadata"]:
                doc["metadata"] = json.loads(doc["metadata"])

            # Apply filter
            if self._matches_filter(doc, filter_dict):
                results.append(doc)

        return results

    def _matches_filter(self, doc: dict[str, Any], filter_dict: dict[str, Any]) -> bool:
        """
        Check if document matches filter criteria.

        Args:
            doc: Document dictionary
            filter_dict: Filter criteria

        Returns:
            bool: True if document matches filter
        """
        for key, value in filter_dict.items():
            if key not in doc:
                return False
            if doc[key] != value:
                return False
        return True

    def update_document(
        self, doc_id: str, updates: dict[str, Any], collection_name: str
    ) -> bool:
        """
        Update a document.

        Args:
            doc_id: Document ID
            updates: Dictionary of fields to update
            collection_name: Name of the collection

        Returns:
            bool: True if successful, False otherwise
        """
        cursor = self.conn.cursor()

        # Check if document exists
        existing = self.find_document(doc_id, collection_name)
        if not existing:
            return False

        # Prepare updates
        update_fields = []
        update_values = []

        for key, value in updates.items():
            if key == "metadata":
                update_fields.append("metadata = ?")
                update_values.append(json.dumps(value))
            elif key in ["content", "is_deleted", "deleted_at", "updated_at"]:
                update_fields.append(f"{key} = ?")
                update_values.append(value)
            elif (
                key in ["parent_doc_id", "chunk_index"]
                and collection_name != "documents"
            ):
                update_fields.append(f"{key} = ?")
                update_values.append(value)

        if not update_fields:
            return False

        update_values.append(doc_id)

        if collection_name == "documents":
            query = f"UPDATE documents SET {', '.join(update_fields)} WHERE id = ?"  # nosec B608
            cursor.execute(query, update_values)
        else:
            self._validate_collection_name(collection_name)
            query = (
                f"UPDATE {collection_name} SET {', '.join(update_fields)} WHERE id = ?"  # nosec B608
            )
            cursor.execute(query, update_values)

        self.conn.commit()
        return cursor.rowcount > 0

    def delete_document(self, doc_id: str, collection_name: str) -> bool:
        """
        Delete a document.

        Args:
            doc_id: Document ID
            collection_name: Name of the collection

        Returns:
            bool: True if successful, False otherwise
        """
        cursor = self.conn.cursor()

        if collection_name == "documents":
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        else:
            self._validate_collection_name(collection_name)
            cursor.execute(f"DELETE FROM {collection_name} WHERE id = ?", (doc_id,))  # nosec B608

        self.conn.commit()
        return cursor.rowcount > 0

    def count_documents(self, collection_name: str) -> int:
        """
        Count documents in a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            int: Number of documents
        """
        cursor = self.conn.cursor()

        if collection_name == "documents":
            cursor.execute("SELECT COUNT(*) FROM documents")
        else:
            self._validate_collection_name(collection_name)
            cursor.execute(f"SELECT COUNT(*) FROM {collection_name}")  # nosec B608

        return cursor.fetchone()[0]

    def aggregate(
        self, pipeline: list[dict[str, Any]], collection_name: str
    ) -> list[dict[str, Any]]:
        """
        Execute an aggregation pipeline.

        Args:
            pipeline: Aggregation pipeline (MongoDB-style)
            collection_name: Name of the collection

        Returns:
            List[Dict]: Aggregation results
        """
        # Simplified aggregation for SQLite
        # Full MongoDB-style aggregation would be complex to implement

        results = []

        if pipeline and pipeline[0].get("$group"):
            # Simple grouping
            group_field = pipeline[0]["$group"].replace("$", "")
            cursor = self.conn.cursor()

            if collection_name == "documents":
                self._validate_collection_name(group_field)
                query = f"SELECT {group_field}, COUNT(*) as count FROM documents GROUP BY {group_field}"  # nosec B608
                cursor.execute(query)
            else:
                self._validate_collection_name(collection_name)
                self._validate_collection_name(group_field)
                query = f"SELECT {group_field}, COUNT(*) as count FROM {collection_name} GROUP BY {group_field}"  # nosec B608
                cursor.execute(query)

            for row in cursor.fetchall():
                results.append({"group": dict(row), "count": row["count"]})

        return results


class SQLiteBackend(SQLiteMemoryBackend):
    """
    File-based SQLite backend for document storage.

    Uses SQLite file database for production use.
    Supports unified ID system and JSON metadata storage.
    """

    def __init__(self, db_path: str = "documents.db"):
        """
        Initialize file-based SQLite database.

        Args:
            db_path: Path to SQLite database file
        """
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._initialize_schema()
