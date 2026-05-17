#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:KG-2.0 — Knowledge Graph Object-Graph Mapper (OGM).

Provides declarative, bidirectional mapping between Pydantic
``RegistryNode`` subclasses and Knowledge Graph nodes/edges.

The OGM eliminates manual ``_upsert_node()`` / ``_serialize_node()`` calls
by deriving KG labels automatically from ``RegistryNodeType`` and using
``model_dump()`` / ``model_validate()`` for serialization round-trips.

Usage::

    from agent_utilities.knowledge_graph.core.ogm import KGMapper

    mapper = KGMapper(engine)
    mapper.upsert(my_node)              # Pydantic → KG
    node = mapper.load("id", NodeCls)   # KG → Pydantic
    mapper.delete("id")                 # Remove from both layers

See docs/pillars/architecture_c4.md §CONCEPT:KG-2.0
"""


import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from ...models.knowledge_graph import RegistryNode, RegistryNodeType

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=RegistryNode)

# ── Label Resolution ──────────────────────────────────────────────────

# Maps RegistryNodeType enum values to PascalCase KG labels.
# Built lazily on first access and cached.
_LABEL_CACHE: dict[str, str] | None = None


def _build_label_cache() -> dict[str, str]:
    """Build a mapping from RegistryNodeType values to PascalCase KG labels."""
    cache: dict[str, str] = {}
    for member in RegistryNodeType:
        # Convert snake_case enum value to PascalCase label
        # e.g. "memory_retriever" → "MemoryRetriever", "knowledge_base_topic" → "KnowledgeBaseTopic"
        parts = member.value.split("_")
        label = "".join(part.capitalize() for part in parts)
        cache[member.value] = label
    return cache


def resolve_label(node_type: RegistryNodeType | str) -> str:
    """Resolve a ``RegistryNodeType`` to its PascalCase KG label.

    Args:
        node_type: The node type enum member or its string value.

    Returns:
        PascalCase label suitable for use as a Cypher node label.

    Examples:
        >>> resolve_label(RegistryNodeType.SELF_MODEL)
        'MemoryRetriever'
        >>> resolve_label("knowledge_base_topic")
        'KnowledgeBaseTopic'
    """
    global _LABEL_CACHE
    if _LABEL_CACHE is None:
        _LABEL_CACHE = _build_label_cache()

    type_val = node_type.value if isinstance(node_type, RegistryNodeType) else node_type
    label = _LABEL_CACHE.get(type_val)
    if label is None:
        # Fallback: convert on-the-fly
        parts = type_val.split("_")
        label = "".join(part.capitalize() for part in parts)
        _LABEL_CACHE[type_val] = label
    return label


# ── Custom label decorator ────────────────────────────────────────────

_CUSTOM_LABELS: dict[type, str] = {}


def kg_label(label: str) -> Callable[[type[T]], type[T]]:
    """Class decorator to override the auto-derived KG label for a node type.

    Usage::

        @kg_label("MyCustomLabel")
        class MyNode(RegistryNode):
            type: RegistryNodeType = RegistryNodeType.AGENT
            ...
    """

    def decorator(cls: type[T]) -> type[T]:
        _CUSTOM_LABELS[cls] = label
        return cls

    return decorator


# ── Core Mapper ───────────────────────────────────────────────────────


class KGMapper:
    """Declarative Pydantic ↔ KG mapper.

    CONCEPT:KG-2.0 — KG Object-Graph Mapper

    Provides a thin, type-safe layer for persisting Pydantic models to the
    Knowledge Graph and hydrating them back. Replaces manual Cypher upsert
    patterns throughout the engine.

    Features:
        - Auto-derives KG label from ``RegistryNodeType``
        - ``@kg_label`` decorator for custom label overrides
        - Bidirectional: ``upsert()`` (Pydantic → KG), ``load()`` (KG → Pydantic)
        - Change watchers for event-driven invalidation
        - NetworkX + backend dual-write for consistency

    Args:
        engine: The ``IntelligenceGraphEngine`` instance to operate on.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._watchers: dict[str, list[Callable[..., Any]]] = {}

    # ── Serialization helpers ─────────────────────────────────────────

    def _get_label(self, node: RegistryNode) -> str:
        """Resolve the KG label for a node, checking custom overrides first."""
        custom = _CUSTOM_LABELS.get(type(node))
        if custom:
            return custom
        return resolve_label(node.type)

    def _serialize(self, node: RegistryNode, label: str) -> dict[str, Any]:
        """Serialize a Pydantic node to a flat dict suitable for KG storage.

        Handles nested structures by JSON-encoding complex types while
        preserving scalar fields as native KG properties.
        """
        import json

        data = node.model_dump()
        props: dict[str, Any] = {"_label": label}
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, dict | list):
                # KG backends generally don't support nested structures;
                # serialize to JSON string for storage, deserialize on load.
                if key == "embedding" and isinstance(value, list):
                    props[key] = value  # Keep embeddings as native arrays
                else:
                    props[key] = json.dumps(value)
            else:
                props[key] = value
        return props

    def _deserialize(self, data: dict[str, Any], model_cls: type[T]) -> T:
        """Hydrate a KG record back into a typed Pydantic model.

        Reverses the JSON-encoding applied during ``_serialize``.
        """
        import json

        clean: dict[str, Any] = {}
        # Get field types from the model for intelligent deserialization
        field_info = model_cls.model_fields

        for key, value in data.items():
            if key.startswith("_"):
                continue  # Skip internal KG properties
            if key in field_info and isinstance(value, str):
                annotation = field_info[key].annotation
                # Try to detect JSON-encoded fields
                origin = getattr(annotation, "__origin__", None)
                if origin in (dict, list) or (
                    isinstance(annotation, type) and issubclass(annotation, dict | list)
                ):
                    try:
                        clean[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        clean[key] = value
                else:
                    clean[key] = value
            else:
                clean[key] = value
        return model_cls.model_validate(clean)

    # ── CRUD Operations ───────────────────────────────────────────────

    def upsert(self, node: RegistryNode) -> str:
        """Persist a Pydantic node to both NetworkX and the KG backend.

        If the node already exists (by ``id``), it is updated in-place.

        Args:
            node: The Pydantic model instance to persist.

        Returns:
            The node's ``id``.
        """
        label = self._get_label(node)
        props = self._serialize(node, label)

        # 1. NetworkX layer
        self.engine.graph.add_node(node.id, **node.model_dump())

        # 2. Backend layer (if available)
        if self.engine.backend:
            self.engine._upsert_node(label, node.id, props)

        # 3. Fire watchers
        self._notify(label, "upsert", node)

        logger.debug("OGM upsert: %s (%s)", node.id, label)
        return node.id

    def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create or update a relationship between two nodes.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            edge_type: Relationship type string (from ``RegistryEdgeType``).
            properties: Optional additional properties for the edge.
        """
        edge_props = {"type": edge_type, **(properties or {})}

        # NetworkX
        self.engine.graph.add_edge(source_id, target_id, **edge_props)

        # Backend
        if self.engine.backend:
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            props_str = ", ".join(f"r.{k} = ${k}" for k in edge_props if k != "type")
            set_clause = "SET r.timestamp = $ts"
            if props_str:
                set_clause += f", {props_str}"

            query = (
                f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) "
                f"MERGE (a)-[r:{edge_type.upper()}]->(b) "
                f"{set_clause}"
            )
            params: dict[str, Any] = {
                "src": source_id,
                "tgt": target_id,
                "ts": ts,
                **{k: v for k, v in edge_props.items() if k != "type"},
            }
            try:
                self.engine.backend.execute(query, params)
            except Exception as e:
                logger.warning("OGM edge upsert failed: %s", e)

        logger.debug("OGM edge: %s -[%s]-> %s", source_id, edge_type, target_id)

    def load(self, node_id: str, model_cls: type[T]) -> T | None:
        """Load a node from the KG and hydrate it as a typed Pydantic model.

        Attempts backend first, falls back to NetworkX.

        Args:
            node_id: The unique ID of the node to load.
            model_cls: The Pydantic model class to hydrate into.

        Returns:
            The hydrated model instance, or ``None`` if not found.
        """
        # Try backend first
        if self.engine.backend:
            label = resolve_label(model_cls.model_fields["type"].default)
            results = self.engine.backend.execute(
                f"MATCH (n:{label} {{id: $id}}) RETURN n",
                {"id": node_id},
            )
            if results:
                data = results[0].get("n", results[0])
                return self._deserialize(data, model_cls)

        # Fallback to NetworkX
        if node_id in self.engine.graph:
            data = dict(self.engine.graph.nodes[node_id])
            return self._deserialize(data, model_cls)

        return None

    def load_by_label(
        self,
        model_cls: type[T],
        limit: int = 100,
        where: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[T]:
        """Load all nodes of a given type from the KG.

        Args:
            model_cls: The Pydantic model class (label derived from its type field).
            limit: Maximum number of nodes to return.
            where: Optional Cypher WHERE clause (without the WHERE keyword).
            params: Parameters for the WHERE clause.

        Returns:
            List of hydrated Pydantic model instances.
        """
        label = resolve_label(model_cls.model_fields["type"].default)
        where_clause = f"WHERE {where}" if where else ""
        query = f"MATCH (n:{label}) {where_clause} RETURN n LIMIT {limit}"

        results: list[T] = []
        if self.engine.backend:
            rows = self.engine.backend.execute(query, params or {})
            for row in rows:
                data = row.get("n", row)
                try:
                    results.append(self._deserialize(data, model_cls))
                except Exception as e:
                    logger.debug("OGM load_by_label skip: %s", e)
        else:
            # NetworkX fallback
            type_val = model_cls.model_fields["type"].default
            if isinstance(type_val, RegistryNodeType):
                type_val = type_val.value
            for nid, ndata in self.engine.graph.nodes(data=True):
                if ndata.get("type") == type_val:
                    try:
                        results.append(self._deserialize(dict(ndata), model_cls))
                    except Exception:
                        continue  # nosec B112
                    if len(results) >= limit:
                        break

        return results

    def delete(self, node_id: str) -> bool:
        """Remove a node from both NetworkX and the KG backend.

        Args:
            node_id: The ID of the node to remove.

        Returns:
            ``True`` if the node was found and deleted, ``False`` otherwise.
        """
        found = False

        # NetworkX
        if node_id in self.engine.graph:
            self.engine.graph.remove_node(node_id)
            found = True

        # Backend
        if self.engine.backend:
            try:
                self.engine.backend.execute(
                    "MATCH (n {id: $id}) SET n.status = 'ARCHIVED'",
                    {"id": node_id},
                )
                found = True
            except Exception as e:
                logger.warning("OGM delete failed: %s", e)

        if found:
            logger.debug("OGM delete: %s", node_id)

        return found

    # ── Change Watchers ───────────────────────────────────────────────

    def watch(self, label: str, callback: Callable[..., Any]) -> None:
        """Register a callback to fire on upsert/delete events for a label.

        Callback signature::

            def callback(event: str, node: RegistryNode | None) -> None

        where ``event`` is ``"upsert"`` or ``"delete"``.

        Args:
            label: The KG label to watch (e.g. ``"MemoryRetriever"``).
            callback: The function to invoke on changes.
        """
        self._watchers.setdefault(label, []).append(callback)

    def _notify(self, label: str, event: str, node: RegistryNode | None = None) -> None:
        """Fire all registered watchers for a label."""
        for cb in self._watchers.get(label, []):
            try:
                cb(event, node)
            except Exception as e:
                logger.warning("OGM watcher error (%s/%s): %s", label, event, e)
