#!/usr/bin/env python3
"""Experience Alignment for Few-Shot Adaptation.

Implements CONCEPT:KG-2.37 (Experience Alignment)
Leverages the native ExperienceNode to adapt agentic behavior through few-shot memory alignment.
"""

import logging

from ...models.knowledge_graph import ExperienceNode
from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class ExperienceAlignmentEngine:
    """Aligns agent behavior dynamically using Experience Nodes."""

    def __init__(self, engine: IntelligenceGraphEngine):
        self.engine = engine

    def ingest_experience(self, experience: ExperienceNode) -> str:
        """Ingest a new experience into the knowledge graph to enable few-shot adaptation.

        Args:
            experience: An ExperienceNode containing contextual and behavioral data.

        Returns:
            The ID of the ingested node.
        """
        logger.info(f"Ingesting Experience Node: {experience.id}")
        self.engine.add_node(
            node_id=experience.id,
            node_type="Experience",
            properties=experience.model_dump(),
        )
        return experience.id

    def retrieve_aligned_experiences(
        self, context_tags: list[str], limit: int = 3
    ) -> list[ExperienceNode]:
        """Retrieve the most relevant experiences aligned to the current context tags.

        Args:
            context_tags: Tags representing the current agent context.
            limit: Maximum number of experiences to retrieve.

        Returns:
            List of relevant ExperienceNode instances.
        """
        if not self.engine.backend:
            return []

        # Simplified retrieval using overlap of tags.
        # In a real system, this would use semantic similarity or topological RAG.
        logger.info(f"Retrieving aligned experiences for tags: {context_tags}")

        cypher = """
        MATCH (e:Experience)
        WHERE any(tag IN e.tags WHERE tag IN $tags)
        RETURN e.id AS id, e.name AS name, e.tags AS tags, e.importance_score AS score
        ORDER BY e.importance_score DESC
        LIMIT $limit
        """

        results = self.engine.backend.execute(
            cypher, {"tags": context_tags, "limit": limit}
        )
        experiences = []
        for row in results or []:
            try:
                # We instantiate minimal ExperienceNodes from backend rows
                exp = ExperienceNode(
                    id=row["id"],
                    name=row["name"],
                    condition="",
                    action="",
                    importance_score=row["score"] or 1.0,
                )
                experiences.append(exp)
            except Exception as e:
                logger.warning(f"Failed to parse experience {row['id']}: {e}")

        return experiences
