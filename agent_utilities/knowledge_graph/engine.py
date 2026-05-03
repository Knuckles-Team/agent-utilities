#!/usr/bin/python
from __future__ import annotations

"""Unified Intelligence Graph Engine.

This module provides the high-level interface for querying the unified knowledge graph,
supporting structural Cypher queries, topological impact analysis, and hybrid search.

The engine is composed of focused mixins for maintainability:
- ``engine_query.py``: Query, search, and retrieval methods.
- ``engine_memory.py``: Memory CRUD operations.
- ``engine_ingestion.py``: Episode, MCP, A2A, and skill ingestion.
- ``engine_ahe.py``: AHE self-improvement cycle methods.
- ``engine_registry.py``: Identity, prompt, resource, and codemap management.
"""

import json
import logging
import math
from enum import Enum
from typing import Any

import networkx as nx

from .backends import create_backend, get_active_backend
from .backends.base import GraphBackend

# Import mixins
from .engine_ahe import AHEMixin
from .engine_federation import FederationMixin
from .engine_ingestion import IngestionMixin
from .engine_memory import MemoryMixin
from .engine_query import QueryMixin
from .engine_registry import FocusedSubgraph, RegistryMixin

logger = logging.getLogger(__name__)

__all__ = [
    "IntelligenceGraphEngine",
    "RegistryGraphEngine",
    "FocusedSubgraph",
    "cosine_similarity",
]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2, strict=False))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(a * a for a in v2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


class IntelligenceGraphEngine(
    QueryMixin, MemoryMixin, IngestionMixin, AHEMixin, RegistryMixin, FederationMixin
):
    """Engine for querying the unified intelligence graph (Agents, Tools, Code, Memory).

    Composed of focused mixins for maintainability. All 49+ existing importers
    continue to work since IntelligenceGraphEngine is still the single public class.
    """

    _ACTIVE_ENGINE: IntelligenceGraphEngine | None = None

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        backend: GraphBackend | None = None,
        db_path: str | None = None,
        external_ontologies: list[str] | None = None,
    ):
        self.graph = graph
        self.backend: GraphBackend | None = None

        # Use provided backend, or check for an active one, or create one from factory
        if backend is not None:
            self.backend = backend
        else:
            active_backend = get_active_backend()
            if active_backend is not None:
                self.backend = active_backend
            elif db_path:
                self.backend = create_backend(db_path=db_path)
            else:
                self.backend = None

        # Auto-register as active if none exists to support singleton pattern
        if IntelligenceGraphEngine._ACTIVE_ENGINE is None:
            IntelligenceGraphEngine._ACTIVE_ENGINE = self

        from .hybrid_retriever import HybridRetriever  # type: ignore
        from .inference_engine import InferenceEngine  # type: ignore

        self.hybrid_retriever = HybridRetriever(self)
        self.inference_engine = InferenceEngine(self)

        # Register external ontologies provided during initialization
        if external_ontologies:
            for ext in external_ontologies:
                # Basic parsing if formatted as URI|Endpoint
                parts = ext.split("|", 1)
                uri = parts[0].strip()
                endpoint = parts[1].strip() if len(parts) > 1 else None
                self.register_external_ontology(uri, endpoint)

    @classmethod
    def get_active(cls) -> IntelligenceGraphEngine | None:
        """Retrieve the currently active engine instance."""
        return cls._ACTIVE_ENGINE

    @classmethod
    def set_active(cls, engine: IntelligenceGraphEngine | None):
        """Explicitly set the active engine instance."""
        cls._ACTIVE_ENGINE = engine

    def _get_allowed_columns(self, label: str) -> list[str]:
        """Get the list of allowed columns for a given node label from the schema."""
        try:
            from ..models.schema_definition import SCHEMA

            for node_def in SCHEMA.nodes:
                if node_def.name == label:
                    return list(node_def.columns.keys())
        except ImportError:
            pass
        return []

    def _serialize_node(self, node: Any, label: str | None = None) -> dict[str, Any]:
        """Serialize a Pydantic node for backend storage, handling Enums and JSON fields."""
        data = node.model_dump() if hasattr(node, "model_dump") else dict(node)
        clean_data = {}

        # Define fields that Ladybug supports as native arrays
        ARRAY_FIELDS = [
            "capabilities",
            "tags",
            "tool_ids",
            "success_criteria_met",
            "embedding",
            "issues",
        ]

        # Filter by schema if label is provided
        allowed_cols = self._get_allowed_columns(label) if label else None

        for k, v in data.items():
            if v is None:
                continue
            if allowed_cols is not None and k not in allowed_cols:
                continue

            if isinstance(v, Enum):
                clean_data[k] = v.value
            elif isinstance(v, dict | list) and k not in ARRAY_FIELDS:
                clean_data[k] = json.dumps(v)
            else:
                clean_data[k] = v
        return clean_data

    def _get_set_clause(self, data: dict[str, Any], alias: str = "n") -> str:
        """Generate a SET clause for a Cypher query from a dictionary."""
        sets = []
        for k in data.keys():
            if k == "id":
                continue
            sets.append(f"{alias}.{k} = ${k}")
        return " SET " + ", ".join(sets) if sets else ""

    def _upsert_node(self, label: str, node_id: str, data: dict[str, Any]):
        """Perform an idempotent upsert of a node using MATCH/SET then CREATE."""
        if not self.backend:
            return

        # 1. Try to update existing
        set_clause = self._get_set_clause(data)
        update_query = f"MATCH (n:{label}) WHERE n.id = $id {set_clause} RETURN n.id"
        res = self.backend.execute(update_query, data)

        if not res:
            # 2. If not found, create
            cols = ", ".join([f"{k}: ${k}" for k in data.keys()])
            create_query = f"CREATE (n:{label} {{{cols}}})"
            self.backend.execute(create_query, data)

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
            query = (
                f"MATCH (s {{id: $sid}}), (t {{id: $tid}}) "
                f"MERGE (s)-[r:{rel_type}]->(t) SET r += $props"
            )
            self.backend.execute(
                query, {"sid": source_id, "tid": target_id, "props": props}
            )

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
    ):
        """Add a generic node to the graph.

        This is a convenience method for code that doesn't have a typed
        Pydantic model (e.g. council verdicts, ad-hoc decision nodes).
        """
        props = properties or {}
        props["type"] = node_type
        self.graph.add_node(node_id, **props)
        if self.backend:
            data = {"id": node_id, **props}
            self._upsert_node(node_type, node_id, data)


# Alias for backward compatibility
RegistryGraphEngine = IntelligenceGraphEngine
