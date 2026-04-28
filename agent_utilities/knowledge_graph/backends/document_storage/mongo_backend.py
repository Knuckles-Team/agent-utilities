"""
MongoDB Backend for Document Storage.

Implements document storage using MongoDB with unified ID support.
"""

from typing import Any

from .base import DocumentDB


class MongoDBBackend(DocumentDB):
    """
    MongoDB backend for document storage.

    Uses MongoDB for document storage.
    Supports unified ID system.

    Note: This requires pymongo to be installed.
    """

    def __init__(self, uri: str = "mongodb://localhost:27017/"):
        """
        Initialize MongoDB connection.

        Args:
            uri: MongoDB connection URI
        """
        try:
            import pymongo

            self.pymongo = pymongo
        except ImportError as e:
            raise ImportError(
                "MongoDB backend requires pymongo. Install with: pip install pymongo"
            ) from e

        self.client = self.pymongo.MongoClient(uri)
        self.db = self.client.get_database()

    def _get_collection(self, collection_name: str):
        """Get MongoDB collection."""
        return self.db[collection_name]

    def create_collection(self, collection_name: str) -> None:
        """Create a document collection."""
        # MongoDB creates collections automatically on first insert
        # We can create explicitly with options if needed
        if collection_name not in self.db.list_collection_names():
            self.db.create_collection(collection_name)

    def insert_document(self, document: dict[str, Any], collection_name: str) -> str:
        """Insert a document into a collection."""
        unified_id = document.get("id")
        if not unified_id:
            raise ValueError("Document must have 'id' field")

        if not unified_id.startswith("doc_"):
            raise ValueError(
                f"Invalid unified ID: {unified_id}. Must start with 'doc_'"
            )

        collection = self._get_collection(collection_name)
        collection.insert_one(document)
        return unified_id

    def insert_many(
        self, documents: list[dict[str, Any]], collection_name: str
    ) -> list[str]:
        """Insert multiple documents into a collection."""
        collection = self._get_collection(collection_name)
        collection.insert_many(documents)
        return [str(doc.get("id", "")) for doc in documents]

    def find_document(self, doc_id: str, collection_name: str) -> dict[str, Any] | None:
        """Find a specific document by ID."""
        collection = self._get_collection(collection_name)
        return collection.find_one({"id": doc_id})

    def find_documents(
        self, filter_dict: dict[str, Any], collection_name: str
    ) -> list[dict[str, Any]]:
        """Find documents matching a filter."""
        collection = self._get_collection(collection_name)
        return list(collection.find(filter_dict))

    def update_document(
        self, doc_id: str, updates: dict[str, Any], collection_name: str
    ) -> bool:
        """Update a document."""
        collection = self._get_collection(collection_name)
        result = collection.update_one({"id": doc_id}, {"$set": updates})
        return result.modified_count > 0

    def delete_document(self, doc_id: str, collection_name: str) -> bool:
        """Delete a document."""
        collection = self._get_collection(collection_name)
        result = collection.delete_one({"id": doc_id})
        return result.deleted_count > 0

    def count_documents(self, collection_name: str) -> int:
        """Count documents in a collection."""
        collection = self._get_collection(collection_name)
        return collection.count_documents({})

    def aggregate(
        self, pipeline: list[dict[str, Any]], collection_name: str
    ) -> list[dict[str, Any]]:
        """Execute an aggregation pipeline."""
        collection = self._get_collection(collection_name)
        return list(collection.aggregate(pipeline))

    def close(self):
        """Close database connection."""
        if self.client:
            self.client.close()
