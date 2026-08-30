"""Shared persistence seam for orchestration action-result nodes."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

_NodeProperties = dict[str, Any] | Callable[[str, str], dict[str, Any]]
_Link = tuple[str | None, str | None, str]


def _persist_action_result(
    engine: Any,
    id_prefix: str,
    label: str,
    node_type: Callable[..., Any],
    node_properties: _NodeProperties,
    *,
    backend_links: tuple[_Link, ...] = (),
    graph_edges: tuple[_Link, ...] = (),
) -> str:
    """Persist a domain action node while retaining the established write order.

    The action mixins provide domain-specific node types, properties, and link payloads.
    This seam owns ID/timestamp creation plus the shared compute-graph write,
    optional backend upsert, and backend-link/compute-edge sequencing.
    Exceptions intentionally propagate unchanged, matching the direct call
    sites it replaces.
    """
    node_id = f"{id_prefix}:{uuid.uuid4().hex}"
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    properties = (
        node_properties(node_id, timestamp)
        if callable(node_properties)
        else node_properties
    )
    node = node_type(id=node_id, **properties, timestamp=timestamp)
    engine.graph.add_node(node.id, **engine._serialize_node(node))

    if engine.backend:
        data = engine._serialize_node(node, label=label)
        engine._upsert_node(label, node.id, data)
        for source_id, target_id, relationship in backend_links:
            engine.link_nodes(
                node.id if source_id is None else source_id,
                node.id if target_id is None else target_id,
                relationship,
            )

    for source_id, target_id, relationship in graph_edges:
        source = node.id if source_id is None else source_id
        target = node.id if target_id is None else target_id
        if source in engine.graph:
            engine.graph.add_edge(source, target, relationship=relationship)

    return node.id
