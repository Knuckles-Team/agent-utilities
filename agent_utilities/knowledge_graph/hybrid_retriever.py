#!/usr/bin/python
from __future__ import annotations

"""Hybrid Retriever for Knowledge Graph.

Combines semantic vector similarity with topological graph traversal.
"""

import logging
from typing import Any

from agent_utilities.core.embedding_utilities import create_embedding_model

from .engine import IntelligenceGraphEngine, cosine_similarity

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Retrieves relevant subgraph context using Hybrid GraphRAG."""

    def __init__(self, engine: IntelligenceGraphEngine):
        self.engine = engine
        try:
            self.embed_model = create_embedding_model()
            logger.info("HybridRetriever initialized with LlamaIndex embedding model.")
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model: {e}")
            self.embed_model = None

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
                            node_data["_score"] = sim
                            scored_nodes.append(node_data)

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
                                assembled_subgraph.append(d)
                except Exception as e:
                    logger.debug(f"NX fallback traversal failed: {e}")
                    if node_id not in visited:
                        visited.add(node_id)
                        assembled_subgraph.append(node)

        return assembled_subgraph
