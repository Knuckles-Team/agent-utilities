from __future__ import annotations

"""Memory management mixin for IntelligenceGraphEngine.

Extracted from engine.py. Contains CRUD operations for memory nodes.
"""
# CONCEPT:ORCH-1.2 — Memory Management


import typing

if typing.TYPE_CHECKING:
    from .._engine_protocol import _EngineProtocol

    _Base = _EngineProtocol
else:
    _Base = object


import logging
import time
import uuid
from typing import Any

from ...models.knowledge_graph import MemoryNode

logger = logging.getLogger(__name__)


class MemoryMixin(_Base):
    """Memory node CRUD capabilities for the KG engine."""

    def add_memory(
        self,
        content: str,
        name: str = "",
        category: str = "general",
        tags: list[str] | None = None,
    ) -> str:
        """Add a new memory to the unified graph."""
        memory_id = f"mem:{uuid.uuid4().hex[:8]}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = MemoryNode(
            id=memory_id,
            name=name or f"Memory {timestamp}",
            description=content,
            timestamp=timestamp,
            category=category,
            tags=tags or [],
        )

        # Generate embedding if model available
        if self.hybrid_retriever.embed_model:
            try:
                node.embedding = self.hybrid_retriever.embed_model.get_text_embedding(
                    node.description or node.name
                )
            except Exception as e:
                logger.warning(
                    f"Failed to generate embedding for memory {node.id}: {e}"
                )

        # Tiered write: backend is source of truth, NX is fallback
        if self.backend:
            data = self._serialize_node(node, label="Memory")
            self._upsert_node("Memory", node.id, data)
        else:
            self.graph.add_node(node.id, **node.model_dump())

        return memory_id

    def delete_memory(self, memory_id: str):
        """Delete a memory from the graph."""
        if memory_id in self.graph:
            self.graph.remove_node(memory_id)
        if self.backend:
            self.backend.execute(
                "MATCH (n {id: $id}) DETACH DELETE n", {"id": memory_id}
            )

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """Retrieve a specific memory by ID from the graph."""
        # Check NetworkX first (in-memory)
        if memory_id in self.graph:
            return {"id": memory_id, **self.graph.nodes[memory_id]}
        # Fallback to persistent backend
        if self.backend:
            results = self.backend.execute(
                "MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id}
            )
            if results:
                return results[0].get("m", results[0])
        return None

    def update_memory(self, memory_id: str, **kwargs):
        """Update properties of an existing memory."""
        if memory_id in self.graph:
            self.graph.nodes[memory_id].update(kwargs)
        if self.backend:
            set_clause = self._get_set_clause(kwargs, "n", label="Memory")
            self.backend.execute(
                f"MATCH (n {{id: $id}}){set_clause}",
                {"id": memory_id, **kwargs},
            )

    def link_nodes(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ):
        """Create a relationship between two nodes in the graph."""
        props = properties or {}
        if source_id in self.graph and target_id in self.graph:
            self.graph.add_edge(source_id, target_id, type=rel_type, **props)

        if self.backend:
            set_clause = self._get_set_clause(props, alias="r", label=rel_type)
            query = (
                f"MATCH (s {{id: $sid}}), (t {{id: $tid}}) "
                f"MERGE (s)-[r:{rel_type}]->(t){set_clause}"
            )
            params = {"sid": source_id, "tid": target_id}
            params.update(props)
            self.backend.execute(query, params)

    def add_memory_node(self, memory: MemoryNode):
        """Add a MemoryNode object to the graph."""
        if self.backend:
            data = self._serialize_node(memory, label="Memory")
            self._upsert_node("Memory", memory.id, data)
        else:
            self.graph.add_node(memory.id, **memory.model_dump())

    def get_memory_node(self, memory_id: str) -> MemoryNode | None:
        """Retrieve a MemoryNode object by ID."""
        data = self.get_memory(memory_id)
        if data:
            return MemoryNode(
                **{k: v for k, v in data.items() if not k.startswith("_")}
            )
        return None

    def update_memory_node(self, memory_id: str, memory: MemoryNode):
        """Update a memory using a MemoryNode object."""
        self.update_memory(memory_id, **memory.model_dump(exclude={"id"}))

    def delete_memory_node(self, memory_id: str):
        """Delete a memory node."""
        self.delete_memory(memory_id)

    # --- Enhanced Memory & Ingestion Tools ---
