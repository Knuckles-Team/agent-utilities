# agent_utilities/graph/client.py
import logging

from ..knowledge_graph.backends import create_backend, get_active_backend
from .models import GraphNode

logger = logging.getLogger(__name__)


async def get_graph_client():
    """Retrieve or create the active graph backend."""
    backend = get_active_backend()
    if backend is None:
        # Fallback to IntelligenceGraphEngine active instance if available
        from ..knowledge_graph.engine import IntelligenceGraphEngine

        engine = IntelligenceGraphEngine.get_active()
        if engine and engine.backend:
            return engine.backend
        backend = create_backend()
    return backend


async def create_or_merge_node(node: GraphNode):
    """Pydantic-validated insert that works with your existing Cypher layer."""
    from ..knowledge_graph.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()

    props = node.to_cypher_props()
    node_id = props.get("id")
    if not node_id:
        raise ValueError(f"Node {node} missing 'id' property")

    # 1. Update in-memory graph for immediate topological availability
    if engine:
        engine.graph.add_node(node_id, **props)
        logger.debug(f"Updated in-memory graph node: {node_id}")

    # 2. Persist to backend via Cypher
    client = await get_graph_client()
    if not client:
        raise RuntimeError("No graph backend available")
    labels_str = ":".join(node.labels)

    # Clean props for SET clause (exclude id as it's in the MERGE part)
    set_props = {k: v for k, v in props.items() if k != "id"}

    query = f"""
    MERGE (n:{labels_str} {{id: $id}})
    SET n += $props
    RETURN n {{.*}}
    """

    try:
        result = client.execute(query, {"id": node_id, "props": set_props})
        return result
    except Exception as e:
        logger.error(f"Failed to create/merge node {node_id}: {e}")
        raise e
