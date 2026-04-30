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


async def extract_and_ingest_triples(
    ctx: RunContext[AgentDeps], episode_description: str, source: str
) -> str:
    """Extract knowledge triples from a recent episode and ingest them dynamically.

    Extract triples (Entity-Relation-Entity) using an LLM and ingest them into the graph.

    Args:
        episode_description: Description of the episode or interaction.
        source: Source of the episode (e.g., 'agent_episode_123').
        ctx: The agent context.
    """
    if not HAS_GRAPH_ENGINE or IntelligenceGraphEngine is None:
        return "IntelligenceGraphEngine is not available."

    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        return "No active IntelligenceGraphEngine found."

    try:
        # We use an LLM to extract triples
        from pydantic_ai import Agent

        from agent_utilities.core.model_factory import create_model

        model = create_model()
        agent = Agent(
            model=model,
            system_prompt="Extract key knowledge triples in the format: Entity1|Relation|Entity2. Return one triple per line.",
        )
        result = await agent.run(episode_description)
        triples_raw = result.data.split("\n")

        triples_added = 0

        for line in triples_raw:
            parts = line.split("|")
            if len(parts) == 3:
                e1, rel, e2 = [p.strip() for p in parts]
                e1_id = f"entity:{e1.lower().replace(' ', '_')}"
                e2_id = f"entity:{e2.lower().replace(' ', '_')}"

                # Ensure entities exist
                if engine.backend:
                    engine.backend.execute(
                        "MERGE (n:Entity {id: $id}) ON CREATE SET n.name = $name",
                        {"id": e1_id, "name": e1},
                    )
                    engine.backend.execute(
                        "MERGE (n:Entity {id: $id}) ON CREATE SET n.name = $name",
                        {"id": e2_id, "name": e2},
                    )
                else:
                    if e1_id not in engine.graph:
                        engine.graph.add_node(e1_id, name=e1, type="Entity")
                    if e2_id not in engine.graph:
                        engine.graph.add_node(e2_id, name=e2, type="Entity")

                # Create relation
                engine.link_nodes(
                    e1_id,
                    e2_id,
                    rel.upper().replace(" ", "_"),
                    properties={"source": source, "confidence": 0.9},
                )
                triples_added += 1

        return f"Successfully extracted and ingested {triples_added} triples."
    except Exception as e:
        logger.error(f"Failed to ingest triples: {e}")
        return f"Failed to ingest triples: {e}"


kg_evolution_tools = [extract_and_ingest_triples]
