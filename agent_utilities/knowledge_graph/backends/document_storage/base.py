"""
Document Storage Base Protocol.

Defines the abstract interface for document storage backends.
"""

from abc import ABC, abstractmethod
from typing import Any


class DocumentDB(ABC):
    """
    Abstract interface for document storage.

    This protocol defines the required methods for document storage backends
    including CRUD operations and aggregation capabilities.
    """

    @abstractmethod
    def create_collection(self, collection_name: str) -> None:
        """
        Create a document collection.

        Args:
            collection_name: Name of the collection to create
        """
        pass

    @abstractmethod
    def insert_document(self, document: dict[str, Any], collection_name: str) -> str:
        """
        Insert a document into a collection.

        Args:
            document: Document dictionary (must include 'id' field)
            collection_name: Name of the collection

        Returns:
            str: Document ID
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def find_document(self, doc_id: str, collection_name: str) -> dict[str, Any] | None:
        """
        Find a specific document by ID.

        Args:
            doc_id: Document ID
            collection_name: Name of the collection

        Returns:
            Optional[Dict]: Document dictionary, or None if not found
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def delete_document(self, doc_id: str, collection_name: str) -> bool:
        """
        Delete a document.

        Args:
            doc_id: Document ID
            collection_name: Name of the collection

        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    def count_documents(self, collection_name: str) -> int:
        """
        Count documents in a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            int: Number of documents
        """
        pass

    @abstractmethod
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
        pass
