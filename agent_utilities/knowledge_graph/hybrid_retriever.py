#!/usr/bin/python
from __future__ import annotations

"""Hybrid Retriever for Knowledge Graph.

Combines semantic vector similarity with topological graph traversal
and optional backlink-density retrieval weighting (CONCEPT:KG-2.2).
"""

import json
import logging
import math
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent

from agent_utilities.core.embedding_utilities import create_embedding_model
from agent_utilities.core.model_factory import create_model

from .engine import IntelligenceGraphEngine, cosine_similarity
from .hypergraph import PositionalInteractionEncoder

if TYPE_CHECKING:
    from agent_utilities.models.schema_pack import SchemaPack

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Retrieves relevant subgraph context using Hybrid GraphRAG.

    Combines semantic vector similarity, topological graph traversal,
    backlink-density retrieval weighting (CONCEPT:KG-2.2), and positional interaction
    encodings (CONCEPT:KG-2.4) for inductive hypergraph reasoning.

    When a pack is configured, its ``backlink_boost_strategy`` and
    ``backlink_boost_factor`` govern whether and how in-degree density
    influences scoring.

    Args:
        engine: The ``IntelligenceGraphEngine`` instance.
        schema_pack: Optional active schema pack for retrieval configuration.
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine,
        schema_pack: SchemaPack | None = None,
    ):
        self.engine = engine
        self._schema_pack = schema_pack

        # Backlink boost config from schema pack (CONCEPT:KG-2.2)
        if schema_pack:
            self._boost_strategy = schema_pack.backlink_boost_strategy
            self._boost_factor = schema_pack.backlink_boost_factor
        else:
            # Default: global boost with standard coefficient
            from agent_utilities.models.schema_pack import BacklinkBoostStrategy

            self._boost_strategy = BacklinkBoostStrategy.GLOBAL
            self._boost_factor = 0.1

        # CONCEPT:KG-2.4: Inductive Knowledge Hypergraphs
        self._enc_pi = PositionalInteractionEncoder()

        try:
            self.embed_model = create_embedding_model()
            logger.info("HybridRetriever initialized with LlamaIndex embedding model.")
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model: {e}")
            self.embed_model = None

    def _compute_positional_interactions(self, pos_a: int, pos_b: int) -> list[float]:
        """Computes the hyperedge interaction embedding for two positions.

        CONCEPT:KG-2.4: Provides the structural vector to match novel relation topologies
        by assessing their positional intersections.
        """
        return self._enc_pi.encode_interaction(pos_a, pos_b)

    def _backlink_boost(self, node_id: str) -> float:
        """Compute retrieval boost from inbound edge density (CONCEPT:KG-2.2).

        Uses logarithmic scaling to prevent hub nodes from dominating:
        ``boost = 1.0 + factor * log(1 + in_degree)``

        A node with 0 inbound edges gets boost 1.0 (neutral).
        A node with 10 inbound edges gets ~1.0 + 0.1 * 2.4 = ~1.24.
        A node with 100 inbound edges gets ~1.0 + 0.1 * 4.6 = ~1.46.

        Args:
            node_id: The node identifier to compute boost for.

        Returns:
            Multiplicative boost factor (>= 1.0).
        """
        if node_id not in self.engine.graph:
            return 1.0
        in_degree = self.engine.graph.in_degree(node_id)
        return 1.0 + self._boost_factor * math.log1p(in_degree)

    def _compute_query_weight(self, query: str) -> float:
        """Compute a context-aware weight based on query length and keyword density."""
        words = query.split()
        if not words:
            return 1.0
        # Longer queries or queries with high density might get a slight bump
        return 1.0 + (len(words) * 0.01)

    def retrieve_hybrid(
        self, query: str, context_window: int = 10, multi_hop_depth: int = 2
    ) -> list[dict[str, Any]]:
        """Perform a hybrid search using both vector similarity and graph topology.

        Args:
            query: The search string.
            context_window: The maximum number of base nodes to retrieve.
            multi_hop_depth: How many edges out to traverse for context assembly.

        Returns:
            A list of nodes with extended graph context.
        """
        # 1. Semantic Search (Vector)
        base_nodes = []
        if self.embed_model and self.engine.backend:
            # Generate query embedding
            try:
                query_emb = self.embed_model.get_text_embedding(query)
                # We fetch nodes that have embeddings and calculate similarity
                # In Ladybug DB (or Neo4j), we could use native vector indexes
                # Fallback to fetching all and computing cosine similarity locally
                res = self.engine.backend.execute(
                    "MATCH (n) WHERE n.embedding IS NOT NULL RETURN n.id as id, n.embedding as emb, n as data"
                )

                scored_nodes = []
                for row in res:
                    node_emb = row.get("emb")
                    if node_emb:
                        sim = cosine_similarity(query_emb, node_emb)
                        if sim > 0.6:  # Threshold
                            node_data = row.get("data", {})
                            node_data["id"] = row.get("id")
                            node_data["_score"] = sim * self._compute_query_weight(
                                query
                            )
                            scored_nodes.append(node_data)

                # 1b. Apply backlink-density boost (CONCEPT:KG-2.2)
                if self._boost_strategy == "global":
                    for node in scored_nodes:
                        node["_score"] *= self._backlink_boost(node["id"])

                scored_nodes.sort(key=lambda x: x["_score"], reverse=True)
                if scored_nodes:
                    base_nodes = scored_nodes[:context_window]
                else:
                    logger.debug("No semantic matches found, falling back to keyword")
                    base_nodes = self.engine._search_keyword(
                        query, top_k=context_window
                    )
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to keyword: {e}")
                base_nodes = self.engine._search_keyword(query, top_k=context_window)
        else:
            # Fallback to basic NetworkX/keyword search
            base_nodes = self.engine._search_keyword(query, top_k=context_window)

        # 2. Graph Traversal (Multi-hop context assembly)
        assembled_subgraph = []
        visited = set()

        for node in base_nodes:
            node_id = node["id"]
            if node_id in visited:
                continue

            # Fetch immediate neighborhood using backend Cypher
            if self.engine.backend:
                # Get 1 to multi_hop_depth neighbors
                query_str = (
                    f"MATCH (n {{id: $id}})-[*1..{multi_hop_depth}]-(m) RETURN m"
                )
                neighbors = self.engine.backend.execute(query_str, {"id": node_id})

                context_nodes = [node]
                for n_row in neighbors:
                    m = n_row.get("m")
                    if m and m.get("id") not in visited:
                        # Apply backlink boost during context assembly (CONCEPT:KG-2.2)
                        if self._boost_strategy == "context_only":
                            m_id = m.get("id", "")
                            boost = self._backlink_boost(m_id)
                            m["_context_boost"] = boost
                        visited.add(m["id"])
                        context_nodes.append(m)

                assembled_subgraph.extend(context_nodes)
            else:
                # NetworkX fallback
                try:
                    import networkx as nx

                    if node_id in self.engine.graph:
                        neighborhood = nx.ego_graph(
                            self.engine.graph, node_id, radius=multi_hop_depth
                        )
                        for n, data in neighborhood.nodes(data=True):
                            if n not in visited:
                                visited.add(n)
                                d = dict(data)
                                d["id"] = n
                                # Apply backlink boost during context assembly (CONCEPT:KG-2.2)
                                if self._boost_strategy == "context_only":
                                    d["_context_boost"] = self._backlink_boost(n)
                                assembled_subgraph.append(d)
                except Exception as e:
                    logger.debug(f"NX fallback traversal failed: {e}")
                    if node_id not in visited:
                        visited.add(node_id)
                        assembled_subgraph.append(node)

        return assembled_subgraph

    def retrieve_decomposed(
        self,
        query: str,
        context_window: int = 10,
        multi_hop_depth: int = 2,
        max_subtasks: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve using decomposed subqueries (CONCEPT:AHE-3.5)."""
        subqueries = self._decompose_query(query, max_subtasks=max_subtasks)

        # If it didn't decompose, just run normal retrieval
        if len(subqueries) <= 1:
            return self.retrieve_hybrid(query, context_window, multi_hop_depth)

        logger.info(f"Decomposed query into: {subqueries}")
        all_nodes = []
        seen_ids = set()

        # Divide context window among subqueries
        sub_context_window = max(2, context_window // len(subqueries))

        for subq in subqueries:
            nodes = self.retrieve_hybrid(
                subq, context_window=sub_context_window, multi_hop_depth=multi_hop_depth
            )
            for node in nodes:
                if node["id"] not in seen_ids:
                    seen_ids.add(node["id"])
                    all_nodes.append(node)

        # Sort final nodes by score if they have one
        all_nodes.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
        return all_nodes[:context_window]

    def _decompose_query(self, query: str, max_subtasks: int = 3) -> list[str]:
        """Decompose a complex query into sub-queries for targeted retrieval (CONCEPT:AHE-3.5)."""
        try:
            model = create_model()
            agent = Agent(
                model=model,
                system_prompt=(
                    f"Decompose the following complex task into up to {max_subtasks} abstract "
                    "technical sub-queries (e.g., 'dark image handling', 'geometric comparison'). "
                    "Return a JSON list of strings."
                ),
            )
            result = agent.run_sync(query)
            # Parse JSON list from result.data
            try:
                import re

                json_str = re.search(r"\[.*\]", result.data, re.DOTALL)
                if json_str:
                    queries = json.loads(json_str.group())
                    return queries[:max_subtasks]
                return [query]
            except json.JSONDecodeError:
                return [query]
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]
