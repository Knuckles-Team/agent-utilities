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

from ..core.engine import IntelligenceGraphEngine, cosine_similarity
from ..core.hypergraph import PositionalInteractionEncoder

if TYPE_CHECKING:
    from agent_utilities.models.schema_pack import SchemaPack

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Retrieves relevant subgraph context using Hybrid GraphRAG.

    Combines semantic vector similarity, topological graph traversal,
    backlink-density retrieval weighting (CONCEPT:KG-2.2), positional interaction
    encodings (CONCEPT:KG-2.4) for inductive hypergraph reasoning, and
    retrieval quality gating (CONCEPT:KG-2.6) for faithfulness scoring.

    When a pack is configured, its ``backlink_boost_strategy``,
    ``backlink_boost_factor``, and ``min_relevance_threshold`` govern
    retrieval behaviour.

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
            self._relevance_threshold = schema_pack.min_relevance_threshold
        else:
            # Default: global boost with standard coefficient
            from agent_utilities.models.schema_pack import BacklinkBoostStrategy

            self._boost_strategy = BacklinkBoostStrategy.GLOBAL
            self._boost_factor = 0.1
            self._relevance_threshold = 0.6

        # CONCEPT:KG-2.4: Inductive Knowledge Hypergraphs
        self._enc_pi = PositionalInteractionEncoder()

        # CONCEPT:KG-2.6: Retrieval Quality Gate
        self._quality_gate: Any = None  # Lazy-initialized
        self._last_quality_report = None

        # CONCEPT:KG-2.3: Lazy embedding model — defer HTTP connection to first use
        self._embed_model = None
        self._embed_model_initialized = False

    @property
    def embed_model(self):
        """Lazy-initialized embedding model — only connects to LM Studio on first search."""
        if not self._embed_model_initialized:
            self._embed_model_initialized = True
            try:
                self._embed_model = create_embedding_model()
                logger.info("HybridRetriever: Embedding model initialized (lazy).")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding model: {e}")
                self._embed_model = None
        return self._embed_model

    @embed_model.setter
    def embed_model(self, value):
        self._embed_model = value
        self._embed_model_initialized = True

    def _vector_search_native(
        self, query_emb: list[float], top_k: int, target_paths: list[str] | None = None
    ) -> list[dict[str, Any]] | None:
        """Use LadybugDB's native HNSW vector index for O(log N) search."""
        if not self.engine.backend:
            return None

        # Dynamically derive tables from schema — matches build_vector_indices()
        from agent_utilities.models.schema_definition import SCHEMA

        embedding_tables = [
            node.name
            for node in SCHEMA.nodes
            if "embedding" in node.columns
            and "FLOAT" in node.columns["embedding"].upper()
        ]

        if not embedding_tables:
            return None

        try:
            results = []
            for table in embedding_tables:
                idx_name = f"idx_{table.lower()}_embedding"
                try:
                    res = self.engine.backend.execute(
                        f"CALL QUERY_VECTOR_INDEX('{table}', '{idx_name}', $emb, $k) "
                        f"YIELD node, score RETURN node as data, score",
                        {"emb": query_emb, "k": top_k * 3},
                    )

                    for row in res:
                        if not isinstance(row, dict):
                            continue
                        data = row.get("data", {})
                        if not data:
                            continue

                        if target_paths:
                            path = data.get("target_path", "")
                            if not path or not any(tp in path for tp in target_paths):
                                continue

                        data["id"] = data.get("id", "")
                        data["_score"] = float(row.get("score", 0.0))
                        results.append(data)
                except Exception as e:
                    logger.debug(
                        "Native vector search failed for table %s: %s", table, e
                    )

            if not results:
                return None

            results.sort(key=lambda x: x["_score"], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.debug(
                f"Native vector search failed, falling back to O(N) search: {e}"
            )
            return None

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
        self,
        query: str,
        context_window: int = 10,
        multi_hop_depth: int = 2,
        corpus_id: str | None = None,
        hard_negatives: set[str] | None = None,
        skip_quality_gate: bool = False,
        relevance_threshold: float | None = None,
        target_paths: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform a hybrid search using both vector similarity and graph topology.

        Args:
            query: The search string.
            context_window: The maximum number of base nodes to retrieve.
            multi_hop_depth: How many edges out to traverse for context assembly.
            corpus_id: Optional corpus ID to constrain search scope
                (CONCEPT:KG-2.3 — Fixed Corpus Evaluation Mode).
            hard_negatives: Optional set of document IDs to penalize
                (CONCEPT:KG-2.3 — Hard Negative Mining).

        Returns:
            A list of nodes with extended graph context.
        """
        # Resolve corpus constraint (CONCEPT:KG-2.3)
        corpus_doc_ids: set[str] | None = None
        if corpus_id:
            try:
                from .evaluation_corpus import CorpusManager

                mgr = CorpusManager(self.engine)
                corpus_doc_ids = mgr.get_document_ids(corpus_id)
                if not corpus_doc_ids:
                    logger.warning("Corpus %s is empty or not found", corpus_id)
            except Exception as e:
                logger.debug("Corpus resolution failed: %s", e)
        # 1. Semantic Search (Vector)
        base_nodes = []
        if self.embed_model and self.engine.backend:
            # Generate query embedding
            try:
                query_emb = self.embed_model.get_text_embedding(query)

                scored_nodes = []

                # Fast path: Native HNSW Vector Index (CONCEPT:KG-2.0 Optimization)
                # Skip if corpus_ids provided since HNSW doesn't support pre-filtering by ID natively yet
                native_results = None
                if not corpus_doc_ids:
                    native_results = self._vector_search_native(
                        query_emb, context_window, target_paths
                    )

                if native_results is not None:
                    scored_nodes = native_results
                    for node in scored_nodes:
                        node["_score"] *= self._compute_query_weight(query)
                else:
                    # Fallback: O(N) brute force search
                    if corpus_doc_ids:
                        res = self.engine.backend.execute(
                            "MATCH (n) WHERE n.embedding IS NOT NULL "
                            "AND n.id IN $corpus_ids "
                            "RETURN n.id as id, n.embedding as emb, n as data",
                            {"corpus_ids": list(corpus_doc_ids)},
                        )
                    elif target_paths:
                        # Push path filtering into Cypher — avoids deserializing 20K+ nodes
                        path_conditions = " OR ".join(
                            [
                                f"n.target_path CONTAINS $tp{i}"
                                for i in range(len(target_paths))
                            ]
                        )
                        params: dict[str, Any] = {
                            f"tp{i}": tp for i, tp in enumerate(target_paths)
                        }
                        res = self.engine.backend.execute(
                            f"MATCH (n) WHERE n.embedding IS NOT NULL "
                            f"AND ({path_conditions}) "
                            f"RETURN n.id as id, n.embedding as emb, n as data",
                            params,
                        )
                    else:
                        # Label-scoped fallback: query high-value tables first
                        # with limits to avoid scanning all 80K+ nodes
                        res = []
                        _SEARCH_TABLES = [
                            ("Article", 500),  # Research papers — highest value
                            ("Concept", 200),  # Concept nodes
                            ("KBConcept", 200),  # KB concepts
                            ("KBFact", 200),  # KB facts
                            ("Agent", 100),  # Agents
                            ("Tool", 200),  # Tools
                            ("Skill", 100),  # Skills
                            ("Code", 1000),  # Code — large, cap it
                        ]
                        for label, limit in _SEARCH_TABLES:
                            try:
                                rows = self.engine.backend.execute(
                                    f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL "
                                    f"RETURN n.id as id, n.embedding as emb, n as data "
                                    f"LIMIT {limit}"
                                )
                                res.extend(rows)
                            except Exception as e:
                                logger.debug("Failed to query label %s: %s", label, e)

                    for row in res:
                        if not isinstance(row, dict):
                            continue
                        node_emb = row.get("emb")
                        if node_emb:
                            sim = cosine_similarity(query_emb, node_emb)
                            threshold = (
                                relevance_threshold
                                if relevance_threshold is not None
                                else self._relevance_threshold
                            )
                            if sim > threshold:  # CONCEPT:KG-2.6 configurable threshold
                                node_data = row.get("data")
                                if not isinstance(node_data, dict):
                                    node_data = {}

                                node_data["id"] = row.get("id")
                                node_data["_score"] = sim * self._compute_query_weight(
                                    query
                                )
                                scored_nodes.append(node_data)

                # 1b. Apply backlink-density boost (CONCEPT:KG-2.2)
                if self._boost_strategy == "global":
                    for node in scored_nodes:
                        node["_score"] *= self._backlink_boost(node["id"])

                # Apply hard negative penalties (CONCEPT:KG-2.3)
                if hard_negatives:
                    for node in scored_nodes:
                        if node.get("id") in hard_negatives:
                            node["_score"] *= 0.5
                            node["_hard_negative"] = True

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
                    if not isinstance(n_row, dict):
                        continue
                    m = n_row.get("m")
                    if m and isinstance(m, dict) and m.get("id") not in visited:
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

        # CONCEPT:KG-2.6 — Assess retrieval quality
        if skip_quality_gate:
            return assembled_subgraph

        self._last_quality_report = None
        try:
            from .retrieval_quality import RetrievalQualityGate

            if self._quality_gate is None:
                self._quality_gate = RetrievalQualityGate(
                    self.engine,
                    min_relevance_threshold=self._relevance_threshold,
                )
            filtered, report = self._quality_gate.gate_results(
                assembled_subgraph, query
            )
            self._last_quality_report = report
            if not report.gate_passed:
                logger.warning(
                    "[CONCEPT:KG-2.6] Retrieval quality gate failed: %s",
                    [m.value for m in report.failure_modes_detected],
                )
            return filtered if report.gate_passed else assembled_subgraph
        except Exception as e:
            logger.debug("Quality gate assessment skipped: %s", e)
            return assembled_subgraph

    @property
    def last_quality_report(self):
        """The quality report from the most recent retrieval, if available."""
        return self._last_quality_report

    def retrieve_decomposed(
        self,
        query: str,
        context_window: int = 10,
        multi_hop_depth: int = 2,
        max_subtasks: int = 3,
        corpus_id: str | None = None,
        hard_negatives: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve using decomposed subqueries (CONCEPT:AHE-3.4)."""
        subqueries = self._decompose_query(query, max_subtasks=max_subtasks)

        # If it didn't decompose, just run normal retrieval
        if len(subqueries) <= 1:
            return self.retrieve_hybrid(
                query,
                context_window,
                multi_hop_depth,
                corpus_id=corpus_id,
                hard_negatives=hard_negatives,
            )

        logger.info(f"Decomposed query into: {subqueries}")
        all_nodes = []
        seen_ids = set()

        # Divide context window among subqueries
        sub_context_window = max(2, context_window // len(subqueries))

        for subq in subqueries:
            nodes = self.retrieve_hybrid(
                subq,
                context_window=sub_context_window,
                multi_hop_depth=multi_hop_depth,
                corpus_id=corpus_id,
                hard_negatives=hard_negatives,
            )
            for node in nodes:
                if node["id"] not in seen_ids:
                    seen_ids.add(node["id"])
                    all_nodes.append(node)

        # Sort final nodes by score if they have one
        all_nodes.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
        return all_nodes[:context_window]

    def _decompose_query(self, query: str, max_subtasks: int = 3) -> list[str]:
        """Decompose a complex query into sub-queries for targeted retrieval (CONCEPT:AHE-3.4)."""
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

                json_str = re.search(r"\[.*\]", str(result.data), re.DOTALL)
                if json_str:
                    queries = json.loads(json_str.group())
                    return queries[:max_subtasks]
                return [query]
            except json.JSONDecodeError:
                return [query]
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]
