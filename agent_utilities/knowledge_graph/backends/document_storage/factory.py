"""
Document Storage Factory.

Factory for creating document storage backends with unified ID support.
"""

import os
from typing import Any

from .base import DocumentDB


class DocumentStorageFactory:
    """
    Factory for creating document storage backends.

    Supports multiple backend types with unified ID validation:
    - sqlite_memory: In-memory SQLite (default)
    - sqlite: File-based SQLite
    - postgresql: PostgreSQL (requires psycopg2)
    - mongodb: MongoDB (requires pymongo)
    """

    SUPPORTED_BACKENDS = ["sqlite_memory", "sqlite", "postgresql", "mongodb"]

    @staticmethod
    def create_document_backend(db_type: str = "sqlite_memory", **kwargs) -> DocumentDB:
        """
        Create document storage backend.

        Args:
            db_type: Type of backend (sqlite_memory, sqlite, postgresql, mongodb)
            **kwargs: Backend-specific configuration

        Returns:
            DocumentDB: Document storage backend instance

        Raises:
            ValueError: If backend type is not supported
        """
        if db_type == "sqlite_memory":
            from .sqlite_backend import SQLiteMemoryBackend

            return SQLiteMemoryBackend(**kwargs)

        elif db_type == "sqlite":
            from .sqlite_backend import SQLiteBackend

            return SQLiteBackend(db_path=kwargs.get("db_path", "documents.db"))

        elif db_type == "postgresql":
            from .postgres_backend import PostgreSQLBackend

            return PostgreSQLBackend(
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", 5432),
                database=kwargs.get("database", "documents"),
                user=kwargs.get("user", "postgres"),
                password=kwargs.get("password", ""),
            )

        elif db_type == "mongodb":
            from .mongo_backend import MongoDBBackend

            return MongoDBBackend(uri=kwargs.get("uri", "mongodb://localhost:27017/"))

        else:
            raise ValueError(
                f"Unsupported backend: {db_type}. "
                f"Use one of {DocumentStorageFactory.SUPPORTED_BACKENDS}"
            )

    @staticmethod
    def create_from_environment() -> DocumentDB:
        """
        Create backend from environment variables.

        Environment variables:
            DOCUMENT_DB_TYPE: Backend type (default: sqlite_memory)
            DOCUMENT_DB_PATH: Database path (for SQLite)
            DOCUMENT_DB_HOST: Database host (for PostgreSQL)
            DOCUMENT_DB_PORT: Database port (for PostgreSQL)
            DOCUMENT_DB_DATABASE: Database name (for PostgreSQL)
            DOCUMENT_DB_USER: Database user (for PostgreSQL)
            DOCUMENT_DB_PASSWORD: Database password (for PostgreSQL)
            DOCUMENT_DB_URI: Database URI (for MongoDB)

        Returns:
            DocumentDB: Document storage backend instance
        """
        db_type = os.environ.get("DOCUMENT_DB_TYPE", "sqlite_memory")

        kwargs: dict[str, Any] = {}

        if db_type == "sqlite":
            kwargs["db_path"] = os.environ.get("DOCUMENT_DB_PATH", "documents.db")

        elif db_type == "postgresql":
            kwargs["host"] = os.environ.get("DOCUMENT_DB_HOST", "localhost")
            kwargs["port"] = int(os.environ.get("DOCUMENT_DB_PORT", 5432))
            kwargs["database"] = os.environ.get("DOCUMENT_DB_DATABASE", "documents")
            kwargs["user"] = os.environ.get("DOCUMENT_DB_USER", "postgres")
            kwargs["password"] = os.environ.get("DOCUMENT_DB_PASSWORD", "")

        elif db_type == "mongodb":
            kwargs["uri"] = os.environ.get(
                "DOCUMENT_DB_URI", "mongodb://localhost:27017/"
            )

        return DocumentStorageFactory.create_document_backend(db_type, **kwargs)

    @staticmethod
    def get_default_backend() -> DocumentDB:
        """
        Get the default backend (sqlite_memory).

        Returns:
            DocumentDB: Default document storage backend instance
        """
        return DocumentStorageFactory.create_document_backend("sqlite_memory")

    @staticmethod
    def is_backend_supported(db_type: str) -> bool:
        """
        Check if a backend type is supported.

        Args:
            db_type: Backend type to check

        Returns:
            bool: True if supported, False otherwise
        """
        return db_type in DocumentStorageFactory.SUPPORTED_BACKENDS

    @staticmethod
    def list_supported_backends() -> list[str]:
        """
        Get list of supported backend types.

        Returns:
            List[str]: List of supported backend types
        """
        return DocumentStorageFactory.SUPPORTED_BACKENDS.copy()
