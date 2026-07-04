"""KG-driven specialist designation (Plan 08 Synergy 1, CONCEPT:AU-KG.memory.tiered-memory-caching).

Wires the L2 ``CapabilityIndex`` into the live router. Instead of the O(n)
keyword scan, this builds (and caches on the engine) an ANN capability index
from the engine's callable nodes and uses ``designate()`` — O(log n) similarity
ranking with O(1) capability filtering and reward-boosted ordering.

Every entry point is fully guarded: if embeddings, an embedding model, or any
node data are unavailable, the functions return ``None`` so the router falls
back to its existing keyword scan. This wiring therefore never breaks routing —
it strictly augments it when the KG is rich enough.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_CALLABLE_TYPES = {
    "tool",
    "skill",
    "agent",
    "mcp_tool",
    "a2a_agent",
    "callable_resource",
    "internal_skill",
    "agent_skill",
}


def _callable_nodes_with_embeddings(engine: Any) -> list[dict[str, Any]]:
    """Best-effort enumeration of callable nodes that carry an embedding.

    Embeddings may live on the node properties (``embedding``) or in the
    backend's embedding store (``backend._embeddings``); both are checked.
    """
    graph = getattr(engine, "graph", None)
    if graph is None or not hasattr(graph, "node_ids"):
        return []
    backend = getattr(engine, "backend", None)
    backend_embeddings = getattr(backend, "_embeddings", {}) or {}

    # Release-channel gate (CONCEPT:AU-OS.scaling.resolve-active-channel-once): resolve the active channel once;
    # callable nodes tagged with a higher channel (e.g. ``edge``) are excluded
    # from the designation index unless the active channel admits them. On the
    # default ``stable`` channel only stable components are routable.
    from agent_utilities.core.release_channel import active_channel, channel_visible

    current_channel = active_channel()

    nodes: list[dict[str, Any]] = []
    try:
        node_ids = graph.node_ids()
    except Exception:
        return []

    for nid in node_ids:
        try:
            props = graph._get_node_properties(nid) or {}
        except Exception:  # nosec B112 — skip malformed/unreadable nodes during best-effort scan
            continue
        ntype = str(props.get("type", "")).lower()
        rtype = str(props.get("resource_type", "")).lower()
        if ntype not in _CALLABLE_TYPES and rtype not in _CALLABLE_TYPES:
            continue
        # Channel gate: hide components not released on the active channel.
        node_channel = props.get("release_channel") or props.get("channel")
        if node_channel and not channel_visible(node_channel, current_channel):
            continue
        emb = props.get("embedding") or backend_embeddings.get(nid)
        if not emb:
            continue
        caps = (
            props.get("capabilities")
            or props.get("providesCapability")
            or props.get("provides")
            or []
        )
        if isinstance(caps, str):
            caps = [caps]
        nodes.append(
            {"id": nid, "embedding": list(emb), "capabilities": [str(c) for c in caps]}
        )
    return nodes


def build_designation_index(engine: Any) -> Any | None:
    """Build a CapabilityIndex from the engine's callable nodes, or None."""
    from agent_utilities.knowledge_graph.retrieval.capability_index import (
        CapabilityIndex,
    )

    nodes = _callable_nodes_with_embeddings(engine)
    if not nodes:
        return None
    index = CapabilityIndex()
    for n in nodes:
        try:
            index.add(n["id"], n["embedding"], capabilities=set(n["capabilities"]))
        except Exception:  # nosec B112 — skip nodes that fail to index; index remains usable
            continue
    return index if len(index) else None


def get_designation_index(engine: Any, *, refresh: bool = False) -> Any | None:
    """Return a cached CapabilityIndex for ``engine`` (built on first use)."""
    cached = getattr(engine, "_designation_index", None)
    if cached is not None and not refresh:
        return cached
    index = build_designation_index(engine)
    try:
        engine._designation_index = index
    except Exception:
        pass  # engine may not allow attribute assignment; that's fine
    return index


def designate_specialists(
    engine: Any,
    query: str,
    *,
    k: int = 5,
    required_caps: list[str] | None = None,
    embed_fn: Any = None,
) -> list[str] | None:
    """Designate the top-``k`` callable resource ids for ``query``.

    Returns a list of ids, or ``None`` if KG-driven designation is unavailable
    (no embeddings, no model, empty index) — signalling the caller to fall back.
    """
    try:
        index = get_designation_index(engine)
        if index is None or len(index) == 0:
            return None

        if embed_fn is None:
            from agent_utilities.core.embedding_utilities import create_embedding_model

            model = create_embedding_model()
            if model is None:
                return None
            embed_fn = model.get_text_embedding

        embedding = embed_fn(query)
        if embedding is None:
            return None

        designations = index.designate(embedding, required_caps=required_caps, k=k)
        return [d.id for d in designations]
    except Exception as e:  # never break routing
        logger.debug("KG-driven designation unavailable, falling back: %s", e)
        return None
