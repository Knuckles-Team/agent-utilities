#!/usr/bin/python
from __future__ import annotations

import logging

from pydantic_ai import RunContext

from ..models import AgentDeps

logger = logging.getLogger(__name__)

# Try importing the engine
try:
    from ..knowledge_graph.engine import IntelligenceGraphEngine

    HAS_GRAPH_ENGINE = True
except ImportError:
    HAS_GRAPH_ENGINE = False
    IntelligenceGraphEngine = None  # type: ignore


async def export_subgraph(
    ctx: RunContext[AgentDeps], topic: str, depth: int = 2
) -> str:
    """Export a lightweight snapshot or subgraph for sharing with other agents.

    Export a subgraph centered around a topic for agent-to-agent sharing.

    Args:
        topic: The topic or entity ID to export a subgraph around.
        depth: How many hops to traverse for the export.
        ctx: The agent context.
    """
    if not HAS_GRAPH_ENGINE or IntelligenceGraphEngine is None:
        return "IntelligenceGraphEngine is not available."

    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        return "No active IntelligenceGraphEngine found."

    try:
        if not hasattr(engine, "hybrid_retriever"):
            return "HybridRetriever not initialized on the active engine."

        subgraph = engine.hybrid_retriever.retrieve_hybrid(
            topic, context_window=1, multi_hop_depth=depth
        )

        import json

        # Sanitize subgraph for export
        export_data = []
        for node in subgraph:
            clean_node = {k: v for k, v in node.items() if not k.startswith("_")}
            export_data.append(clean_node)

        return json.dumps(
            {"topic": topic, "depth": depth, "nodes": export_data}, indent=2
        )
    except Exception as e:
        logger.error(f"Failed to export subgraph: {e}")
        return f"Failed to export subgraph: {e}"


async def import_agent_card(
    ctx: RunContext[AgentDeps], agent_url: str, agent_card_json: str
) -> str:
    """Import an external agent card or subgraph snapshot into the local knowledge graph.

    Ingest an external agent's capabilities or knowledge subgraph.

    Args:
        agent_url: URL or endpoint of the external agent.
        agent_card_json: JSON string representation of the agent card or subgraph.
        ctx: The agent context.
    """
    if not HAS_GRAPH_ENGINE or IntelligenceGraphEngine is None:
        return "IntelligenceGraphEngine is not available."

    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        return "No active IntelligenceGraphEngine found."

    try:
        import json

        card = json.loads(agent_card_json)

        # We need an ingest_a2a_agent_card logic on the engine or handle it directly here
        if hasattr(engine, "ingest_a2a_agent_card"):
            engine.ingest_a2a_agent_card(agent_url, card)
        else:
            # Basic fallback ingestion
            card_id = f"agent_card:{agent_url.replace('://', '_').replace('/', '_')}"
            engine.graph.add_node(card_id, type="AgentCard", url=agent_url, data=card)
            if engine.backend:
                engine.backend.execute(
                    "MERGE (n:AgentCard {id: $id}) SET n.url = $url, n.data = $data",
                    {"id": card_id, "url": agent_url, "data": json.dumps(card)},
                )

        return f"Successfully imported agent card for {agent_url}."
    except Exception as e:
        logger.error(f"Failed to import agent card: {e}")
        return f"Failed to import agent card: {e}"


kg_share_tools = [export_subgraph, import_agent_card]
