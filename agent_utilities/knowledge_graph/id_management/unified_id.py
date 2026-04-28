"""
Unified ID Management System.

Provides unified ID generation and management across document database,
vector database, and knowledge graph storage layers.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


class UnifiedIDManager:
    """
    Manager for generating and validating unified IDs.

    Unified IDs follow the pattern:
    - Documents: doc_{uuid}
    - Chunks: doc_{uuid}_chunk_{index}
    - Entities: doc_{uuid}_entity_{type}_{index}

    This ensures consistent identification across all storage layers.
    """

    @staticmethod
    def generate_document_id() -> str:
        """
        Generate a unified document ID.

        Returns:
            str: Document ID in format 'doc_{uuid}'
        """
        return f"doc_{uuid.uuid4().hex}"

    @staticmethod
    def generate_chunk_id(parent_doc_id: str, chunk_index: int) -> str:
        """
        Generate a unified chunk ID.

        Args:
            parent_doc_id: Parent document ID
            chunk_index: Zero-based chunk index

        Returns:
            str: Chunk ID in format '{parent_doc_id}_chunk_{index}'
        """
        return f"{parent_doc_id}_chunk_{chunk_index:04d}"

    @staticmethod
    def generate_entity_id(
        parent_doc_id: str, entity_type: str, entity_index: int
    ) -> str:
        """
        Generate a unified entity ID.

        Args:
            parent_doc_id: Parent document ID
            entity_type: Type of entity (e.g., PERSON, ORGANIZATION)
            entity_index: Zero-based entity index

        Returns:
            str: Entity ID in format '{parent_doc_id}_entity_{type}_{index}'
        """
        return f"{parent_doc_id}_entity_{entity_type}_{entity_index:04d}"

    @staticmethod
    def is_unified_id(id_str: str) -> bool:
        """
        Check if an ID follows the unified ID pattern.

        Args:
            id_str: ID string to validate

        Returns:
            bool: True if ID follows unified ID pattern
        """
        return id_str.startswith("doc_")

    @staticmethod
    def extract_doc_id(unified_id: str) -> str | None:
        """
        Extract the parent document ID from a unified ID.

        Args:
            unified_id: Unified ID (document, chunk, or entity)

        Returns:
            Optional[str]: Parent document ID, or None if invalid
        """
        if not unified_id.startswith("doc_"):
            return None

        # Extract base document ID (remove _chunk_* or _entity_* suffixes)
        parts = unified_id.split("_")
        if len(parts) >= 2:
            # Return 'doc_{uuid}' part
            return f"{parts[0]}_{parts[1]}"

        return unified_id

    @staticmethod
    def get_id_type(unified_id: str) -> str | None:
        """
        Determine the type of unified ID.

        Args:
            unified_id: Unified ID string

        Returns:
            Optional[str]: One of 'document', 'chunk', 'entity', or None
        """
        if not unified_id.startswith("doc_"):
            return None

        if "_chunk_" in unified_id:
            return "chunk"
        elif "_entity_" in unified_id:
            return "entity"
        else:
            return "document"

    @staticmethod
    def parse_chunk_id(chunk_id: str) -> dict[str, Any] | None:
        """
        Parse a chunk ID to extract components.

        Args:
            chunk_id: Chunk ID in format '{doc_id}_chunk_{index}'

        Returns:
            Optional[Dict]: Dict with 'doc_id' and 'chunk_index', or None
        """
        if "_chunk_" not in chunk_id:
            return None

        parts = chunk_id.split("_chunk_")
        if len(parts) != 2:
            return None

        try:
            return {"doc_id": parts[0], "chunk_index": int(parts[1])}
        except ValueError:
            return None

    @staticmethod
    def parse_entity_id(entity_id: str) -> dict[str, Any] | None:
        """
        Parse an entity ID to extract components.

        Args:
            entity_id: Entity ID in format '{doc_id}_entity_{type}_{index}'

        Returns:
            Optional[Dict]: Dict with 'doc_id', 'entity_type', and 'entity_index', or None
        """
        if "_entity_" not in entity_id:
            return None

        parts = entity_id.split("_entity_")
        if len(parts) != 2:
            return None

        # Split the second part into type and index
        type_index_parts = parts[1].rsplit("_", 1)
        if len(type_index_parts) != 2:
            return None

        try:
            return {
                "doc_id": parts[0],
                "entity_type": type_index_parts[0],
                "entity_index": int(type_index_parts[1]),
            }
        except ValueError:
            return None


@dataclass
class UnifiedIDRegistry:
    """
    Registry to track unified IDs and their synchronization status across systems.

    Maintains a mapping of unified IDs to their metadata and synchronization
    status across document database, vector database, and knowledge graph.
    """

    document_ids: dict[str, dict[str, Any]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def register_document(
        self, doc_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Register a document in the unified ID registry.

        Args:
            doc_id: Unified document ID
            metadata: Optional metadata about the document
        """
        self.document_ids[doc_id] = {
            "metadata": metadata or {},
            "systems": {
                "document_db": False,
                "vector_db": False,
                "knowledge_graph": False,
            },
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "registered",
        }

    def mark_system_synced(self, doc_id: str, system: str) -> None:
        """
        Mark a document as synced to a specific system.

        Args:
            doc_id: Unified document ID
            system: One of 'document_db', 'vector_db', 'knowledge_graph'
        """
        if doc_id in self.document_ids:
            self.document_ids[doc_id]["systems"][system] = True
            self.document_ids[doc_id]["updated_at"] = datetime.now().isoformat()

            # Check if fully synced
            if self.is_fully_synced(doc_id):
                self.document_ids[doc_id]["status"] = "fully_synced"

    def is_fully_synced(self, doc_id: str) -> bool:
        """
        Check if a document is synced to all systems.

        Args:
            doc_id: Unified document ID

        Returns:
            bool: True if synced to all systems
        """
        if doc_id not in self.document_ids:
            return False
        return all(self.document_ids[doc_id]["systems"].values())

    def is_system_synced(self, doc_id: str, system: str) -> bool:
        """
        Check if a document is synced to a specific system.

        Args:
            doc_id: Unified document ID
            system: One of 'document_db', 'vector_db', 'knowledge_graph'

        Returns:
            bool: True if synced to the specified system
        """
        if doc_id not in self.document_ids:
            return False
        return self.document_ids[doc_id]["systems"].get(system, False)

    def get_document_info(self, doc_id: str) -> dict[str, Any] | None:
        """
        Get information about a registered document.

        Args:
            doc_id: Unified document ID

        Returns:
            Optional[Dict]: Document information, or None if not found
        """
        return self.document_ids.get(doc_id)

    def unregister_document(self, doc_id: str) -> bool:
        """
        Remove a document from the registry.

        Args:
            doc_id: Unified document ID

        Returns:
            bool: True if document was removed, False if not found
        """
        if doc_id in self.document_ids:
            del self.document_ids[doc_id]
            return True
        return False

    def get_all_documents(self) -> list[str]:
        """
        Get all registered document IDs.

        Returns:
            List[str]: List of all registered document IDs
        """
        return list(self.document_ids.keys())

    def get_documents_by_status(self, status: str) -> list[str]:
        """
        Get documents with a specific status.

        Args:
            status: Status to filter by ('registered', 'fully_synced', etc.)

        Returns:
            List[str]: List of document IDs with the specified status
        """
        return [
            doc_id
            for doc_id, info in self.document_ids.items()
            if info.get("status") == status
        ]

    def get_unsynced_documents(self, system: str) -> list[str]:
        """
        Get documents that are not synced to a specific system.

        Args:
            system: System to check ('document_db', 'vector_db', 'knowledge_graph')

        Returns:
            List[str]: List of document IDs not synced to the system
        """
        return [
            doc_id
            for doc_id, info in self.document_ids.items()
            if not info["systems"].get(system, False)
        ]

    def get_documents_older_than(self, days: int) -> list[str]:
        """
        Get documents older than a specified number of days.

        Args:
            days: Number of days threshold

        Returns:
            List[str]: List of document IDs older than the threshold
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            doc_id
            for doc_id, info in self.document_ids.items()
            if datetime.fromisoformat(info["created_at"]) < cutoff_date
        ]

    def update_document_metadata(
        self, doc_id: str, metadata_updates: dict[str, Any]
    ) -> bool:
        """
        Update document metadata.

        Args:
            doc_id: Unified document ID
            metadata_updates: Dictionary of metadata updates

        Returns:
            bool: True if metadata was updated, False if not found
        """
        if doc_id not in self.document_ids:
            return False

        self.document_ids[doc_id]["metadata"].update(metadata_updates)
        self.document_ids[doc_id]["updated_at"] = datetime.now().isoformat()
        return True

    def get_statistics(self) -> dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dict: Statistics about the registry
        """
        total = len(self.document_ids)
        fully_synced = len(self.get_documents_by_status("fully_synced"))

        system_stats = {"document_db": 0, "vector_db": 0, "knowledge_graph": 0}

        for info in self.document_ids.values():
            for system in system_stats:
                if info["systems"].get(system, False):
                    system_stats[system] += 1

        return {
            "total_documents": total,
            "fully_synced": fully_synced,
            "partially_synced": total - fully_synced,
            "system_sync_counts": system_stats,
            "registry_created_at": self.created_at.isoformat(),
        }
