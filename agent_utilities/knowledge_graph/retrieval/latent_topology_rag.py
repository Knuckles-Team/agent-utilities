"""Verified-session adapter for epistemic-graph latent-topology retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from epistemic_graph.latent_topology import retrieve_latent_topology

if TYPE_CHECKING:
    from ..core.engine import IntelligenceGraphEngine
    from ..core.session import GraphSession


class LatentTopologicalRAG:
    """Run EG's bounded topology read through AU's governed query surface."""

    def __init__(self, engine: IntelligenceGraphEngine) -> None:
        self.engine = engine

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        routing_threshold: float = 0.7,
        *,
        session: GraphSession | None = None,
    ) -> list[dict[str, Any]]:
        """Return visible hierarchy nodes sorted by importance score."""
        if not self.engine.backend:
            return []

        def governed_read(statement: str) -> list[dict[str, Any]]:
            return self.engine.query_cypher(statement, session=session)

        return retrieve_latent_topology(
            query,
            read=governed_read,
            top_k=top_k,
            routing_threshold=routing_threshold,
        )
