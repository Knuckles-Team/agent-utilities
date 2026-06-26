#!/usr/bin/python
from __future__ import annotations

"""Hybrid Retriever for Knowledge Graph.

Combines semantic vector similarity with topological graph traversal
and optional backlink-density retrieval weighting (CONCEPT:KG-2.2).
"""

import json
import logging
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent

from agent_utilities.core.embedding_utilities import create_embedding_model
from agent_utilities.core.model_factory import create_model

from ..core.engine import IntelligenceGraphEngine, cosine_similarity
from ..core.hypergraph import PositionalInteractionEncoder

if TYPE_CHECKING:
    from agent_utilities.models.schema_pack import SchemaPack

logger = logging.getLogger(__name__)


def _parse_instant(value: Any) -> datetime | None:
    """Coerce a timestamp (ISO string or ``datetime``) to a tz-aware UTC instant.

    Returns ``None`` for missing/unparsable values so callers can treat unknown
    dates as neutral (CONCEPT:KG-2.22 recency, CONCEPT:KG-2.11 bi-temporal).
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return None


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
        enable_rerank: bool = True,
    ):
        self.engine = engine
        self._schema_pack = schema_pack

        # CONCEPT:KG-2.6 — Reasoning-aware reranking (default ON): reorder an
        # over-fetched candidate pool by query-relevance before capping to the
        # context window so the most relevant nodes drive context assembly.
        if enable_rerank:
            from .reasoning_reranker import ReasoningAwareReranker

            self._reranker: Any = ReasoningAwareReranker()
        else:
            self._reranker = None
        self._rerank_overfetch = 4

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
        # (typed Any to avoid importing the heavy llama_index BaseEmbedding onto the
        # retrieval path — dependency discipline).
        self._embed_model: Any = None
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

    def _engine_vector_search(
        self,
        query_emb: list[float],
        top_k: int,
        *,
        threshold: float,
        target_paths: list[str] | None = None,
        corpus_doc_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Vector candidates from the engine's native ANN (CONCEPT:KG-2.250).

        The vector neighbourhood is ALWAYS computed by the engine — never by an
        O(N) Python cosine scan. Preferred path: ONE cross-modal unified plan
        (``query.unified``, CONCEPT:KG-2.208) that the engine sequences over a
        single off-lock snapshot — an optional ``Scan`` seed + the vector ``Rank``
        leg, costed and executed in one round-trip. When the connected engine was
        built WITHOUT the ``query`` feature (the lean ``pi`` tier), the same vector
        index is reached through its native ``semantic_search`` ANN primitive
        instead — still the engine's vector index, still O(log N), still no Python
        scan. The returned ids are hydrated to full node dicts in one batched
        property fetch and tagged with ``_score`` (the engine similarity).

        ``corpus_doc_ids`` / ``target_paths`` further restrict the ranked
        candidate set: corpus membership is an id-set intersection and
        ``target_paths`` a substring match on the hydrated ``target_path`` — both
        applied to the BOUNDED ranked candidate pool the engine returned, never as
        a full-graph scan.
        """
        graph = getattr(self.engine, "graph", None)
        if graph is None:
            return []

        # Over-fetch so the post-rank corpus/path restriction still leaves a full
        # window of candidates.
        fetch_k = max(top_k * 3, 10)
        if corpus_doc_ids or target_paths:
            fetch_k = max(fetch_k, top_k * 8)

        ranked: list[tuple[str, float]] = []
        used_unified = False
        try:
            plan: list[dict[str, Any]] = [
                {"Rank": {"query": [float(x) for x in query_emb]}},
                {"Limit": {"k": fetch_k}},
            ]
            rows = graph.query_unified(plan)
            used_unified = True
            for row in rows or []:
                rid = row.get("id")
                if rid is None:
                    continue
                ranked.append((str(rid), float(row.get("score") or 0.0)))
        except Exception as e:  # noqa: BLE001 — engine lacks the `query` feature
            # No SQLite-style fallback: drop to the engine's native ANN primitive
            # (the SAME vector index), NOT an O(N) Python cosine scan.
            logger.debug(
                "unified plan unavailable (engine built without `query`?): %s — "
                "using native ANN primitive",
                e,
            )

        if not used_unified:
            ranked = [
                (str(nid), float(score))
                for nid, score in (graph.semantic_search(query_emb, fetch_k) or [])
                if nid
            ]

        if not ranked:
            return []

        # Threshold on the engine similarity (CONCEPT:KG-2.6), then hydrate the
        # surviving ids to full node dicts in ONE batched property fetch.
        ids = [nid for nid, score in ranked if score > threshold]
        if not ids:
            return []
        props = self._batch_node_properties(ids)

        results: list[dict[str, Any]] = []
        score_by_id = dict(ranked)
        for nid in ids:
            data = props.get(nid)
            data = dict(data) if isinstance(data, dict) else {}
            if corpus_doc_ids is not None and nid not in corpus_doc_ids:
                continue
            if target_paths:
                path = str(data.get("target_path", ""))
                if not path or not any(tp in path for tp in target_paths):
                    continue
            data["id"] = nid
            data["_score"] = score_by_id.get(nid, 0.0)
            results.append(data)
            if len(results) >= top_k:
                break
        return results

    def _batch_node_properties(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch properties for many node ids in ONE engine round-trip.

        Uses the engine's batched ``properties_batch`` (CONCEPT:KG-2.16) when the
        client exposes it, else the resident GraphComputeEngine projection — never
        N per-id round-trips.
        """
        graph = getattr(self.engine, "graph", None)
        out: dict[str, dict[str, Any]] = {}
        client = getattr(graph, "_client", None)
        nodes_ns = getattr(client, "nodes", None) if client is not None else None
        batch = getattr(nodes_ns, "properties_batch", None)
        if callable(batch):
            try:
                for nid, blob in (batch(ids) or {}).items():
                    if isinstance(blob, dict):
                        out[str(nid)] = blob
                return out
            except Exception as e:  # noqa: BLE001 — degrade to per-id projection
                logger.debug("properties_batch failed: %s", e)
        getter = getattr(graph, "_get_node_properties", None)
        if callable(getter):
            for nid in ids:
                try:
                    p = getter(nid)
                    if isinstance(p, dict):
                        out[nid] = p
                except Exception:  # noqa: BLE001,S112
                    continue
        return out

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
        if not self.engine.graph.has_node(node_id):
            return 1.0
        in_degree = len(self.engine.graph.get_predecessors(node_id))
        return 1.0 + self._boost_factor * math.log1p(in_degree)

    def _compute_query_weight(self, query: str) -> float:
        """Compute a context-aware weight based on query length and keyword density."""
        words = query.split()
        if not words:
            return 1.0
        # Longer queries or queries with high density might get a slight bump
        return 1.0 + (len(words) * 0.01)

    def _recency_boost(self, node: dict[str, Any], as_of: str | None = None) -> float:
        """Pack-driven temporal recency boost for a node (CONCEPT:KG-2.22).

        Reads the active pack's :class:`RecencyDecaySpec` for the node's type/label
        and decays against the node's bi-temporal ``event_time`` (CONCEPT:KG-2.11),
        falling back to ``timestamp``/``created_at``. The boost is always ``>= 1.0``;
        a missing/unparsable date yields a neutral ``1.0`` so unknown-date nodes are
        never penalised. The ``core`` pack (no ``recency_decay``) is a strict no-op.
        """
        if not self._schema_pack or not self._schema_pack.recency_decay:
            return 1.0
        spec = self._schema_pack.recency_spec_for(str(node.get("type", "")))
        if spec is None:
            return 1.0
        raw = (
            node.get("event_time")
            or node.get("timestamp")
            or node.get("created_at")
            or node.get("updated_at")
        )
        ts = _parse_instant(raw)
        if ts is None:
            return 1.0
        ref = _parse_instant(as_of) or datetime.now(UTC)
        age_days = max(0.0, (ref - ts).total_seconds() / 86400.0)
        if spec.mode == "hyperbolic":
            decay = spec.half_life_days / (spec.half_life_days + age_days)
        else:
            decay = 0.5 ** (age_days / spec.half_life_days)
        return 1.0 + spec.coefficient * decay

    def _source_trust_boost(self, node: dict[str, Any]) -> float:
        """Pack-driven source-trust/authority boost for a node (CONCEPT:KG-2.22).

        Looks up the node's source identifier (``source``/``source_id``/``domain``)
        in the active pack's ``source_trust`` table. Unknown source or ``core`` pack
        => neutral ``1.0``.
        """
        if not self._schema_pack or not self._schema_pack.source_trust:
            return 1.0
        src = node.get("source") or node.get("source_id") or node.get("domain")
        if not src:
            return 1.0
        return self._schema_pack.trust_for(str(src))

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
        active_task: str | None = None,
        as_of: str | None = None,
        query_analysis: bool = False,
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
            as_of: Optional ISO-8601 instant. Pack-driven recency decay is measured
                relative to this reference time, enabling knowledge-state-as-of-date-D
                retrieval over bi-temporal ``event_time``. Defaults to now
                (CONCEPT:KG-2.22, CONCEPT:KG-2.11).

        Returns:
            A list of nodes with extended graph context.
        """
        # Query analysis (CONCEPT:ECO-4.32) — opt-in. Derive a time window
        # (``as_of``) and source-type restriction from the natural-language query.
        # Default off so existing callers are unaffected; when on, a derived
        # ``as_of`` only fills an unset one (explicit caller value always wins),
        # and detected source types post-filter the result.
        qa_source_types: list[str] = []
        if query_analysis:
            try:
                from .query_analysis import analyze_query

                filters = analyze_query(query)
                qa_source_types = filters.source_types
                if as_of is None and filters.as_of:
                    as_of = filters.as_of
            except Exception as e:  # noqa: BLE001 — analysis must never break retrieval
                logger.debug("query analysis failed: %s", e)

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

        # 0. Relational-intent arm (CONCEPT:KG-2.34): deterministic, zero-LLM.
        # Parses "which papers support X" / "what contradicts Y" using the active
        # pack's verb vocabulary and walks typed edges. No-op (empty) for
        # non-relational queries or when the pack declares no relational verbs.
        relational_nodes: list[dict[str, Any]] = []
        if self._schema_pack and self._schema_pack.relational_verbs:
            try:
                from .relational_intent import parse_relational_intent, traverse

                rq = parse_relational_intent(query, self._schema_pack.relational_verbs)
                if rq is not None:
                    relational_nodes = traverse(self.engine, rq, context_window)
            except Exception as e:
                logger.debug("Relational-intent arm failed: %s", e)

        # 1. Semantic Search (Vector) — ONE engine unified plan (CONCEPT:KG-2.250).
        # The vector neighbourhood is computed by the engine's native ANN inside a
        # single costed cross-modal plan (filter + vector ``Rank``), NOT by an O(N)
        # Python cosine scan. There is no SQLite-style fallback: if the engine has
        # no embeddings the arm is empty and we degrade to keyword search.
        base_nodes = []
        if self.embed_model and self.engine.backend:
            # Generate query embedding
            try:
                query_emb = self.embed_model.get_text_embedding(query)

                threshold = (
                    relevance_threshold
                    if relevance_threshold is not None
                    else self._relevance_threshold
                )
                scored_nodes = self._engine_vector_search(
                    query_emb,
                    context_window,
                    threshold=threshold,
                    target_paths=target_paths,
                    corpus_doc_ids=corpus_doc_ids,
                )
                for node in scored_nodes:
                    node["_score"] *= self._compute_query_weight(query)

                # 1b. Apply backlink-density boost (CONCEPT:KG-2.2)
                if self._boost_strategy == "global":
                    for node in scored_nodes:
                        node["_score"] *= self._backlink_boost(node["id"])

                # 1b'. Pack-driven retrieval signals (CONCEPT:KG-2.22):
                # temporal recency decay + source-trust authority weighting. Both
                # are no-ops under the default ``core`` pack (empty config).
                if self._schema_pack and (
                    self._schema_pack.recency_decay or self._schema_pack.source_trust
                ):
                    for node in scored_nodes:
                        rb = self._recency_boost(node, as_of=as_of)
                        sb = self._source_trust_boost(node)
                        if rb != 1.0:
                            node["_recency_boost"] = rb
                        if sb != 1.0:
                            node["_source_trust_boost"] = sb
                        node["_score"] *= rb * sb

                # 1c. Apply Attention-Driven Context Filter (Retrieve query boost on active_task)
                if active_task:
                    try:
                        if self.embed_model:
                            active_task_emb = self.embed_model.get_text_embedding(
                                active_task
                            )
                            for node in scored_nodes:
                                node_emb = node.get("embedding")
                                if node_emb:
                                    task_sim = cosine_similarity(
                                        active_task_emb, node_emb
                                    )
                                    if task_sim > 0.0:
                                        node["_score"] *= 1.0 + 0.5 * task_sim
                                        node["_active_task_boost"] = task_sim
                                else:
                                    # Overlap-based attention boost fallback
                                    overlap = sum(
                                        1
                                        for w in active_task.lower().split()
                                        if w in str(node.get("name", "")).lower()
                                        or w in str(node.get("description", "")).lower()
                                    )
                                    if overlap > 0:
                                        node["_score"] *= 1.0 + 0.1 * overlap
                                        node["_active_task_boost_overlap"] = overlap
                        else:
                            # Overlap-based attention boost fallback if no embed model
                            for node in scored_nodes:
                                overlap = sum(
                                    1
                                    for w in active_task.lower().split()
                                    if w in str(node.get("name", "")).lower()
                                    or w in str(node.get("description", "")).lower()
                                )
                                if overlap > 0:
                                    node["_score"] *= 1.0 + 0.1 * overlap
                                    node["_active_task_boost_overlap"] = overlap
                    except Exception as e:
                        logger.debug("Active task boost computation failed: %s", e)

                # Apply hard negative penalties (CONCEPT:KG-2.3)
                if hard_negatives:
                    for node in scored_nodes:
                        if node.get("id") in hard_negatives:
                            node["_score"] *= 0.5
                            node["_hard_negative"] = True

                scored_nodes.sort(key=lambda x: x["_score"], reverse=True)
                if scored_nodes:
                    base_nodes = self._rerank_candidates(
                        query,
                        scored_nodes,
                        context_window,
                        instruction=active_task or "",
                    )
                else:
                    logger.debug("No semantic matches found, falling back to keyword")
                    base_nodes = self.engine._search_keyword(
                        query, top_k=context_window
                    )
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to keyword: {e}")
                base_nodes = self.engine._search_keyword(query, top_k=context_window)
        else:
            # Fallback to keyword search
            base_nodes = self.engine._search_keyword(query, top_k=context_window)

        # 1d. Merge the relational-intent arm (CONCEPT:KG-2.34) additively: typed-edge
        # hits are prepended (high priority) without displacing the vector arm, so
        # recall never regresses. De-duplicate by node id, keeping the first seen.
        if relational_nodes:
            seen_ids = {n.get("id") for n in relational_nodes}
            base_nodes = relational_nodes + [
                n for n in base_nodes if n.get("id") not in seen_ids
            ]

        # 1e. Autocut: trim the long tail at the largest relative score drop. Gated
        # by the active pack; recall-safe — never trims < min (CONCEPT:KG-2.22).
        if self._schema_pack and self._schema_pack.autocut_enabled and base_nodes:
            from .autocut import autocut

            base_nodes = autocut(
                base_nodes,
                threshold=self._schema_pack.autocut_threshold,
                min_results=self._schema_pack.autocut_min_results,
            )

        # 2. Graph Traversal (Multi-hop context assembly)
        assembled_subgraph = []
        visited = set()

        for node in base_nodes:
            node_id = node["id"]
            if node_id in visited:
                continue

            # Fetch immediate neighborhood using backend Cypher
            context_nodes: list[dict[str, Any]] = []
            if self.engine.backend:
                # Get 1 to multi_hop_depth neighbors
                query_str = (
                    f"MATCH (n {{id: $id}})-[*1..{multi_hop_depth}]-(m) RETURN m"
                )
                neighbors = self.engine.backend.execute(query_str, {"id": node_id})

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

            # The epistemic-graph backend cannot evaluate variable-length path
            # patterns, so yielding no usable neighbors is not authoritative —
            # fall through to the resident-graph BFS traversal in that case.
            if context_nodes:
                assembled_subgraph.append(node)
                assembled_subgraph.extend(context_nodes)
            else:
                # GraphComputeEngine BFS fallback
                try:
                    if self.engine.graph.has_node(node_id):
                        # BFS traversal up to multi_hop_depth
                        all_discovered = [node_id]
                        frontier = {node_id}
                        for _depth in range(multi_hop_depth):
                            next_frontier: set[str] = set()
                            for nid in frontier:
                                for s in self.engine.graph.get_successors(nid):
                                    if s not in visited and s not in all_discovered:
                                        next_frontier.add(s)
                                for p in self.engine.graph.get_predecessors(nid):
                                    if p not in visited and p not in all_discovered:
                                        next_frontier.add(p)
                            frontier = next_frontier
                            for f_node in sorted(frontier):
                                if f_node not in all_discovered:
                                    all_discovered.append(f_node)
                        # Collect all discovered nodes
                        for nid in all_discovered:
                            if nid not in visited:
                                visited.add(nid)
                                if nid == node_id:
                                    # Preserve the vector-scored base node — it carries
                                    # _score, the active-task attention boost and its
                                    # embedding. Enrich (don't overwrite) with any graph
                                    # properties (e.g. type) it lacks rather than
                                    # refetching a bare graph projection.
                                    d = dict(node)
                                    for k, v in self.engine.graph._get_node_properties(
                                        nid
                                    ).items():
                                        d.setdefault(k, v)
                                else:
                                    d = dict(
                                        self.engine.graph._get_node_properties(nid)
                                    )
                                d["id"] = nid
                                if self._boost_strategy == "context_only":
                                    d["_context_boost"] = self._backlink_boost(nid)
                                assembled_subgraph.append(d)
                except Exception as e:
                    logger.debug(f"Graph traversal fallback failed: {e}")
                    if node_id not in visited:
                        visited.add(node_id)
                        assembled_subgraph.append(node)

        # CONCEPT:ECO-4.32 — apply any query-analysis source-type restriction to
        # the assembled result (no-op when query_analysis is off / no types).
        def _qa(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if not qa_source_types:
                return nodes
            from .query_analysis import filter_nodes_by_source

            return filter_nodes_by_source(nodes, qa_source_types)

        # CONCEPT:KG-2.6 — Assess retrieval quality
        if skip_quality_gate:
            return _qa(assembled_subgraph)

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
            return _qa(filtered if report.gate_passed else assembled_subgraph)
        except Exception as e:
            logger.debug("Quality gate assessment skipped: %s", e)
            return _qa(assembled_subgraph)

    def _rerank_candidates(
        self,
        query: str,
        scored_nodes: list[dict[str, Any]],
        context_window: int,
        instruction: str = "",
    ) -> list[dict[str, Any]]:
        """Reasoning-aware rerank of the candidate pool, capped to the window.

        Over-fetches ``context_window * _rerank_overfetch`` vector-ranked
        candidates, reorders them by blended query-relevance (CONCEPT:KG-2.6),
        and returns the top ``context_window``. With reranking disabled this is
        the plain vector-order top-``context_window`` slice — identical to the
        prior behaviour.
        """
        if self._reranker is None:
            return scored_nodes[:context_window]
        pool = scored_nodes[
            : max(context_window * self._rerank_overfetch, context_window)
        ]
        if len(pool) <= 1:
            return pool[:context_window]
        reranked = self._reranker.rerank(
            query, pool, text_fn=self._node_text, instruction=instruction
        )
        return reranked[:context_window]

    @staticmethod
    def _node_text(node: dict[str, Any]) -> str:
        """Text whose tokens count against a retrieval budget."""
        parts = [str(node.get(k, "")) for k in ("content", "text", "summary", "name")]
        body = " ".join(p for p in parts if p)
        return body or str(node)

    def direct_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        labels: tuple[str, ...] = ("Article", "Code", "Concept", "KBFact", "KBConcept"),
        limit_per_label: int = 500,
    ) -> Any:
        """Direct Corpus Interaction — literal/term search over node text (CONCEPT:KG-2.12).

        A precise, auditable retrieval mode that greps document text directly
        instead of dense-vector similarity, returning ranked hits with line-level
        localization. Complements :meth:`retrieve_hybrid` for exact-term / code /
        identifier lookups where embeddings under-perform.

        Returns:
            A ``DciResult`` (see :mod:`direct_corpus`).
        """
        from .direct_corpus import searcher_from_nodes

        nodes: list[dict[str, Any]] = []
        if self.engine.backend:
            for label in labels:
                try:
                    rows = (
                        self.engine.backend.execute(
                            f"MATCH (n:{label}) RETURN n.id as id, n as data "
                            f"LIMIT {int(limit_per_label)}"
                        )
                        or []
                    )
                except Exception as e:
                    logger.debug("direct_search label %s failed: %s", label, e)
                    continue
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    _d = r.get("data")
                    data = dict(_d) if isinstance(_d, dict) else {}
                    data["id"] = r.get("id", data.get("id", ""))
                    nodes.append(data)

        return searcher_from_nodes(nodes).search(query, top_k=top_k)

    def retrieve_hybrid_budgeted(
        self,
        query: str,
        token_budget: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """``retrieve_hybrid`` capped to a context-token budget (CONCEPT:KG-2.1).

        Retrieves as usual, then keeps the highest-ranked nodes that fit within
        ``token_budget`` (no budget → unchanged). This is the retrieval-side fix
        for context-window bloat — the article's "memory ate 40% of context".
        """
        results = self.retrieve_hybrid(query, **kwargs)
        if not token_budget:
            return results
        from .budget import fit_within

        return fit_within(results, token_budget, text_of=self._node_text)

    def retrieve_executable(
        self,
        query: str,
        *,
        subqueries: list[str] | None = None,
        modes: tuple[str, ...] = ("vector", "grep"),
        top_k: int = 5,
        answer_fn: Any = None,
        use_planner: bool = False,
    ) -> Any:
        """Executable multi-hop RAG over retrieve()/answer() steps (CONCEPT:KG-2.12).

        Runs a deterministic program that dispatches each retrieve step to a mode
        — ``vector`` (``retrieve_hybrid``) or ``grep`` (``direct_search``) — with
        execution-driven adaptive retrieval + mode fallback (vector→grep), and an
        inspectable trace. Replaces ungrounded NL self-reflection with a grounded,
        self-repairing program. ``answer_fn(query, evidence)`` defaults to a
        deterministic extractive concatenation; inject an LLM answerer for prose.

        The plan is built deterministically by ``build_linear_plan`` (one retrieve
        per sub-query → answer). Set ``use_planner=True`` to instead synthesize the
        plan with the ORCH-1.27 ``planner`` role (richer / non-linear plans); a
        planner failure degrades to the deterministic plan, so the run never breaks.

        Returns:
            A ``RagResult`` (see :mod:`executable_rag`).
        """
        from .executable_rag import ExecutableRagProgram, build_linear_plan

        def retrieve_fn(q: str, mode: str, k: int) -> list[dict[str, Any]]:
            if mode == "grep":
                res = self.direct_search(q, top_k=k)
                return [
                    {"id": h.doc_id, "content": "", "score": h.score} for h in res.hits
                ]
            nodes = self.retrieve_hybrid(q, context_window=k, skip_quality_gate=True)
            return [
                {"id": n.get("id", ""), "content": self._node_text(n)} for n in nodes
            ]

        def _default_answer(q: str, evidence: list[dict[str, Any]]) -> str:
            snippets = [
                str(e.get("content", "")) for e in evidence[:3] if e.get("content")
            ]
            return " ".join(snippets) if snippets else "insufficient evidence"

        if use_planner:
            plan = self._synthesize_executable_plan(
                query, subqueries=subqueries, modes=modes, top_k=top_k
            )
        else:
            plan = build_linear_plan(
                subqueries or [query], question=query, mode=modes[0], top_k=top_k
            )
        program = ExecutableRagProgram(
            retrieve_fn,
            answer_fn or _default_answer,
            fallback_modes=list(modes[1:]),
        )
        return program.run(plan, question=query)

    def _synthesize_executable_plan(
        self,
        query: str,
        *,
        subqueries: list[str] | None = None,
        modes: tuple[str, ...] = ("vector", "grep"),
        top_k: int = 5,
    ) -> Any:
        """Synthesize an executable RAG plan via the ORCH-1.27 ``planner`` role.

        Returns a list of :class:`~.executable_rag.PlanStep`. Always succeeds — a
        planner/LLM failure degrades to :func:`~.executable_rag.build_linear_plan`
        through :func:`~.executable_rag.parse_executable_plan` (CONCEPT:KG-2.12).
        """
        from .executable_rag import parse_executable_plan

        allowed = " | ".join(modes)
        system_prompt = (
            "You are an executable-RAG plan synthesizer. Given a question, emit a "
            'JSON object {"steps": [...]} where each step is '
            '{"op": "retrieve"|"answer", "query": str, "mode": '
            f'"{allowed}", "top_k": int, "out_var": str}}. Use one or more '
            "retrieve steps (decompose multi-hop questions; pick the mode best "
            "suited to each sub-query) followed by a final answer step over the "
            "original question. Return ONLY the JSON object."
        )
        try:
            model = create_model(role="planner")
            agent = Agent(model=model, system_prompt=system_prompt)
            result: Any = agent.run_sync(query)
            raw = str(getattr(result, "output", None) or getattr(result, "data", ""))
        except Exception as e:  # pragma: no cover - planner is best-effort
            logger.debug("Executable-RAG planner failed, using linear plan: %s", e)
            raw = ""
        return parse_executable_plan(
            raw, question=query, subqueries=subqueries, mode=modes[0], top_k=top_k
        )

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
        active_task: str | None = None,
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
                active_task=active_task,
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
                active_task=active_task,
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
            result: Any = agent.run_sync(query)
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

    # ── CONCEPT:KG-2.12 — Memory-First Retrieval ──────────────────────────────────

    def _generate_hyde_plan(self, query: str, mode_hint: str | None = None):
        """Generate a structured HyDE retrieval plan via the ORCH-1.27 ``planner`` role.

        Returns a :class:`~.hyde_planner.HydePlan`. Always succeeds — a planner/LLM failure
        degrades to a single-query plan (see :func:`~.hyde_planner.parse_hyde_plan`).
        """
        from .hyde_planner import parse_hyde_plan

        system_prompt = (
            "You are a memory retrieval planner. Given a user question, emit a JSON object: "
            '{"vector_queries": [<baseline factual statement>, <entity/anchor focus>, '
            "<action/target focus>, <literal nouns, numbers & units>], "
            '"keywords": [<2-4 exact proper nouns>], "search_mode": "standard"|"deep"}. '
            'Use "deep" for aggregations, totals, counts, durations, or broad temporal spans; '
            '"standard" for a single point fact. Return ONLY the JSON object.'
        )
        try:
            model = create_model(role="planner")
            agent = Agent(model=model, system_prompt=system_prompt)
            result: Any = agent.run_sync(query)
            raw = str(getattr(result, "output", None) or getattr(result, "data", ""))
        except Exception as e:  # pragma: no cover - planner is best-effort
            logger.debug("HyDE planner failed, using fallback plan: %s", e)
            raw = ""
        return parse_hyde_plan(raw, original_query=query, mode_hint=mode_hint)

    def plan_and_retrieve(
        self,
        query: str,
        context_window: int = 10,
        mode: str = "hyde",
        self_correct: bool = False,
        corpus_id: str | None = None,
        hard_negatives: set[str] | None = None,
        active_task: str | None = None,
        with_ledger: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Memory-first retrieval: HyDE plan → dual-threshold multi-query → gated 2nd pass.

        CONCEPT:KG-2.12. Assimilates Quarq's retrieval policy (agent-oss/agent.py:1817-2825):

        - ``mode="standard"|"deep"``: a single ``retrieve_hybrid`` at the matching threshold
          (0.38 / 0.28) — Quarq's dual-threshold behavior.
        - ``mode="hyde"``: the planner role emits a multi-query plan; each query runs through the
          graph-native ``retrieve_hybrid`` at the plan's threshold; results are id-dedup/score
          merged (``merge_retrievals``).
        - ``self_correct=True``: if the quality gate reports ``gate_passed=False`` after the first
          pass, a second pass re-runs the plan at the **deep** threshold (0.28) and merges —
          an *evidence-based* trigger, not Quarq's model-self-report ``REQUIRED_DATA``.
        - ``with_ledger=True``: also returns a quantitative-fidelity ACCEPT/REJECT ledger.

        Returns the merged node list, or ``{"nodes": [...], "ledger": {...}, "plan": {...}}``
        when ``with_ledger`` is set. Keeps the 3-hop Wire-First ceiling (method, not a service).
        """
        from .hyde_planner import (
            HydePlan,
            build_evidence_ledger,
            is_trivial_query,
            merge_retrievals,
            threshold_for_mode,
        )

        # CONCEPT:KG-2.15 — social-closer gate: trivial turns skip the planner + retrieval entirely.
        if is_trivial_query(query):
            empty: list[dict[str, Any]] = []
            if with_ledger:
                return {
                    "nodes": empty,
                    "ledger": build_evidence_ledger(query, empty),
                    "plan": HydePlan(vector_queries=[query]).model_dump(),
                    "trivial": True,
                }
            return empty

        if mode in ("standard", "deep"):
            plan = HydePlan(vector_queries=[query], search_mode=mode)  # type: ignore[arg-type]
        else:
            plan = self._generate_hyde_plan(query)

        threshold = threshold_for_mode(plan.search_mode)
        queries = plan.effective_queries(query)
        sub_window = max(2, context_window)

        first_lists = [
            self.retrieve_hybrid(
                q,
                context_window=sub_window,
                corpus_id=corpus_id,
                hard_negatives=hard_negatives,
                relevance_threshold=threshold,
                active_task=active_task,
            )
            for q in queries
        ]
        nodes = merge_retrievals(first_lists, context_window)

        # Self-correcting second pass — fire only when the quality gate measured a failure.
        report = self.last_quality_report
        gate_failed = report is not None and not getattr(report, "gate_passed", True)
        if self_correct and gate_failed:
            deep_threshold = threshold_for_mode("deep")
            second_lists = [
                self.retrieve_hybrid(
                    q,
                    context_window=sub_window,
                    corpus_id=corpus_id,
                    hard_negatives=hard_negatives,
                    relevance_threshold=deep_threshold,
                    active_task=active_task,
                )
                for q in queries
            ]
            nodes = merge_retrievals([nodes, *second_lists], context_window)

        # CONCEPT:KG-2.15 — 4-level fallback cascade: hybrid (above) → dense-only is already
        # inside retrieve_hybrid → lexical keyword scan → backend scan. If the vector path yielded
        # nothing (e.g. embeddings/HNSW unavailable or offline), degrade to a lexical fallback so a
        # query always returns *something* instead of an empty result.
        if not nodes:
            nodes = self._lexical_fallback(query, context_window, corpus_id=corpus_id)

        # CONCEPT:KG-2.18 — record that these memories were RECALLED, on the live retrieval path.
        # The usage half (which recalled nodes actually informed the answer) is closed by the
        # generation step via record_answer_usage(); trust is then trained from the two counts.
        self.usage_telemetry.record_recall(
            [str(n.get("id")) for n in nodes if n.get("id")]
        )

        if with_ledger:
            return {
                "nodes": nodes,
                "ledger": build_evidence_ledger(query, nodes),
                "plan": plan.model_dump(),
            }
        return nodes

    @property
    def usage_telemetry(self):
        """Lazy KG-2.18 recall/usage telemetry, persistent across retrievals on this retriever."""
        tel = getattr(self, "_usage_telemetry", None)
        if tel is None:
            from .retrieval_quality import UsageTelemetry

            tel = UsageTelemetry()
            self._usage_telemetry = tel
        return tel

    def record_answer_usage(
        self, used_ids: list[str], query: str = ""
    ) -> dict[str, Any]:
        """Close the KG-2.18 loop: mark which recalled memories the answer actually used.

        Records usage, persists trained ``trust_score`` onto those nodes, and returns a generation
        lineage record (query → retrieved → used). Call this from the generation step.
        """
        from .retrieval_quality import build_lineage

        self.usage_telemetry.record_usage(used_ids)
        self.usage_telemetry.flush_to_engine(self.engine)
        return build_lineage(
            query, list(self.usage_telemetry._recalled), used_ids=used_ids
        ).model_dump()

    def _lexical_fallback(
        self, query: str, context_window: int, *, corpus_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Keyword/lexical fallback when the vector path returns nothing (CONCEPT:KG-2.15).

        Degradation tiers 3-4 of the cascade: a backend ``CONTAINS`` scan over node content/name
        for the query's distinctive tokens. Returns [] if no backend is available (tier-4 no-op).
        """
        backend = getattr(self.engine, "backend", None)
        if backend is None:
            return []
        import re as _re

        tokens = [t for t in _re.findall(r"[A-Za-z0-9_]{3,}", query)][:6]
        if not tokens:
            return []
        where = " OR ".join(
            f"toLower(n.content) CONTAINS $t{i} OR toLower(n.name) CONTAINS $t{i}"
            for i in range(len(tokens))
        )
        params = {f"t{i}": tok.lower() for i, tok in enumerate(tokens)}
        try:
            rows = backend.execute(
                f"MATCH (n) WHERE {where} "
                f"RETURN n.id as id, n as data LIMIT {max(1, context_window)}",
                params,
            )
        except Exception as e:  # pragma: no cover - backend dialect variance
            logger.debug("Lexical fallback query failed: %s", e)
            return []
        out: list[dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            _d = row.get("data")
            data = dict(_d) if isinstance(_d, dict) else {}
            data["id"] = row.get("id")
            data.setdefault("_score", 0.2)  # low confidence — it's a lexical fallback
            data["_fallback"] = "lexical"
            out.append(data)
        return out
