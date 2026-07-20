# agent_utilities/graph/client.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .models import GraphNode

if TYPE_CHECKING:
    from ..knowledge_graph.core.session import GraphSession

logger = logging.getLogger(__name__)


async def get_process_graph_backend(*, session: GraphSession | None = None):
    """Return the backend owned by the one process graph engine.

    The caller must inherit a verified ambient :class:`GraphSession`; this
    boundary never creates a transport or supplies identity.
    """
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine
    from ..knowledge_graph.core.session import resolve_session

    resolve_session(session, required_scope="kg:read")
    engine = IntelligenceGraphEngine.get_or_create()
    if engine.backend is None:
        raise RuntimeError("The process graph engine has no active backend")
    return engine.backend


async def create_or_merge_node(
    node: GraphNode,
    *,
    session: GraphSession | None = None,
):
    """Pydantic-validated insert through the sole engine write authority."""
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine
    from ..knowledge_graph.core.session import resolve_session

    session = resolve_session(session, required_scope="kg:write")
    engine = IntelligenceGraphEngine.get_or_create()

    props = node.to_cypher_props()
    node_id = props.get("id")
    if not node_id:
        raise ValueError(f"Node {node} missing 'id' property")

    try:
        # The first label is the canonical engine type; retain the complete
        # validated label set as data without issuing a second backend write.
        node_type = node.labels[0]
        props["labels"] = list(node.labels)
        return engine.add_node(
            node_id,
            node_type,
            {k: v for k, v in props.items() if k != "id"},
            session=session,
        )
    except Exception as e:
        logger.error("Graph node merge failed (%s)", type(e).__name__)
        raise
