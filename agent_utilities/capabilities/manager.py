"""
Capabilities Manager — CONCEPT:ECO-4.1
Discovers, enriches, and manages agent capabilities within the Knowledge Graph.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CapabilityResult:
    def __init__(self, name: str, description: str, source: str, score: float):
        self.name = name
        self.description = description
        self.source = source
        self.relevance_score = score


class CapabilityManager:
    """
    Manages discovery and lifecycle of agent capabilities
    extracted from the Knowledge Graph.
    """

    def __init__(self, engine: Any):
        self.engine = engine

    def discover_capabilities(self, query: str) -> list[CapabilityResult]:
        """
        Discover capabilities matching a natural language query using
        the hybrid KG search.
        """
        raw_results = self.engine.search_hybrid(query=query, top_k=10)

        capabilities = []
        for res in raw_results:
            node = res.get("node", res)

            # Extract capabilities specifically from Tool, Skill, or Code nodes
            node_type = node.get("type", node.get("label", ""))

            name = node.get("name", node.get("id", "Unknown"))
            desc = node.get("description", node.get("content", ""))[:200]

            capabilities.append(
                CapabilityResult(
                    name=name,
                    description=desc,
                    source=node_type,
                    score=res.get("score", 0.0),
                )
            )

        return capabilities
