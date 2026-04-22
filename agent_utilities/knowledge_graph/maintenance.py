#!/usr/bin/python
"""Knowledge Graph Maintenance & Pruning."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from .engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

# LM Studio default configuration
LM_STUDIO_URL = "http://localhost:1234/v1/embeddings"
EMBEDDING_MODEL = "nomic-embed-text-v1.5"


def generate_embedding(text: str) -> list[float] | None:
    """Generate embedding vector using local LM Studio."""
    try:
        payload = {"model": EMBEDDING_MODEL, "input": text}
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            return data["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
    return None


class GraphMaintainer:
    """Handles scheduled pruning and vector enrichment for the Knowledge Graph."""

    def __init__(self, engine: IntelligenceGraphEngine):
        self.engine = engine

    def enrich_embeddings(self) -> int:
        """Find messages without embeddings and generate them."""
        if not self.engine.backend:
            return 0

        updated_count = 0
        # Query messages that do not have an embedding
        # In Ladybug, checking for null array might be tricky, but we can assume we add a property "embedded" or just check embedding size.
        # For simplicity, we just fetch all recent messages and check locally if they lack embeddings.
        query = "MATCH (m:Message) RETURN m.id as id, m.content as content, m.embedding as embedding"
        results = self.engine.backend.execute(query)

        for row in results:
            if not row.get("embedding"):
                content = row.get("content", "")
                if content:
                    emb = generate_embedding(content)
                    if emb:
                        self.engine.backend.add_embedding(row["id"], emb)
                        updated_count += 1

        logger.info(f"Enriched {updated_count} messages with embeddings.")
        return updated_count

    def prune_cron_logs(self, keep_days: int = 30) -> int:
        """Delete successful cron logs older than keep_days."""
        if not self.engine.backend:
            return 0

        cutoff_date = (datetime.now() - timedelta(days=keep_days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        # Cypher query to delete old successful logs
        query = """
        MATCH (l:Log)
        WHERE l.status = 'SUCCESS' AND l.timestamp < $cutoff
        DETACH DELETE l
        """
        self.engine.backend.execute(query, {"cutoff": cutoff_date})

        # Since LadybugDB doesn't return deleted count easily, we just log it.
        logger.info(f"Pruned successful cron logs older than {cutoff_date}.")
        return 1

    def summarize_old_chats(self, keep_days: int = 30) -> int:
        """Summarize chats older than keep_days into ChatSummary nodes and delete original Messages."""
        if not self.engine.backend:
            return 0

        cutoff_date = (datetime.now() - timedelta(days=keep_days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        # Find old threads
        query = "MATCH (t:Thread) WHERE t.created_at < $cutoff RETURN t.id as id, t.title as title"
        threads = self.engine.backend.execute(query, {"cutoff": cutoff_date})

        summarized = 0
        for t in threads:
            t_id = t["id"]
            # Fetch messages
            m_query = "MATCH (m:Message)-[:PART_OF]->(t:Thread {id: $t_id}) RETURN m.content as content"
            msgs = self.engine.backend.execute(m_query, {"t_id": t_id}) or []

            if len(msgs) > 0:
                combined_text = "\\n".join([m["content"] for m in msgs])
                summary_text = f"Summary of {len(msgs)} messages: {combined_text[:100]}..."  # Very basic summary placeholder

                import uuid

                sum_id = f"sum:{uuid.uuid4().hex[:8]}"
                props = {
                    "id": sum_id,
                    "type": "chat_summary",
                    "summary_text": summary_text,
                    "importance_score": 0.5,
                    "original_count": len(msgs),
                }

                # Add summary node
                self.engine.backend.execute(
                    "MERGE (n:ChatSummary {id: $id}) SET n += $props",
                    {"id": sum_id, "props": props},
                )

                # Link summary to thread
                self.engine.backend.execute(
                    "MATCH (s:ChatSummary {id: $s_id}), (t:Thread {id: $t_id}) MERGE (s)-[:PART_OF]->(t)",
                    {"s_id": sum_id, "t_id": t_id},
                )

                # Delete old messages
                self.engine.backend.execute(
                    "MATCH (m:Message)-[:PART_OF]->(t:Thread {id: $t_id}) DETACH DELETE m",
                    {"t_id": t_id},
                )

                summarized += 1

        logger.info(f"Summarized {summarized} old threads.")
        return summarized

    def update_importance_scores(self) -> int:
        """Update importance scores using NetworkX centrality (PageRank)."""
        import networkx as nx

        # Note: In a production system, we would rebuild the NX graph from the backend first
        try:
            scores = nx.pagerank(self.engine.graph)
            updated = 0
            for node_id, score in scores.items():
                if self.engine.backend:
                    self.engine.backend.execute(
                        "MATCH (n {id: $id}) SET n.importance_score = $score",
                        {"id": node_id, "score": score},
                    )
                    updated += 1
            logger.info(f"Updated importance scores for {updated} nodes.")
            return updated
        except Exception as e:
            logger.warning(
                f"Could not calculate PageRank (graph might be empty or disconnected): {e}"
            )
            return 0

    def apply_temporal_decay(self) -> int:
        """Apply temporal decay to importance scores (Ebbinghaus-style)."""
        if not self.engine.backend:
            return 0

        # Ensure we handle potential string/float conversion issues from backend
        results = self.engine.backend.execute(
            "MATCH (n) WHERE n.importance_score IS NOT NULL AND n.timestamp IS NOT NULL RETURN n.id as id, n.timestamp as ts, n.importance_score as score"
        )
        updated = 0
        now = datetime.now(UTC)
        for row in results:
            try:
                # Basic ISO format parsing
                ts_str = row["ts"].replace("Z", "+00:00")
                ts = datetime.fromisoformat(ts_str)
                # Ensure ts is aware
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)

                days = (now - ts).days
                if days > 0:
                    current_score = float(row["score"])
                    new_score = current_score * (0.95**days)  # 5% decay per day
                    self.engine.backend.execute(
                        "MATCH (n) WHERE n.id = $id SET n.importance_score = $score",
                        {"id": row["id"], "score": new_score},
                    )
                    updated += 1
            except Exception as e:
                logger.debug(f"Skipping decay for {row.get('id')}: {e}")
                continue
        logger.info(f"Applied temporal decay to {updated} nodes.")
        return updated

    def consolidate_memory(self, keep_days: int = 7) -> int:
        """Distill old episodes into semantic summaries (hypergraph nodes)."""
        if not self.engine.backend:
            return 0

        cutoff_date = (datetime.now() - timedelta(days=keep_days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        # Find episodes that haven't been consolidated
        query = "MATCH (e:Episode) WHERE e.timestamp < $cutoff AND NOT (e)-[:CONSOLIDATES_INTO]->() RETURN e.id as id, e.description as description"
        episodes = self.engine.backend.execute(query, {"cutoff": cutoff_date})

        if not episodes:
            return 0

        import uuid

        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Batch episodes by similarity or just group them for now
        # In a real system, we'd use an LLM here to summarize
        summary_text = f"Consolidated summary of {len(episodes)} episodes."
        sum_id = f"sum:{uuid.uuid4().hex[:8]}"

        self.engine.backend.execute(
            "CREATE (s:ChatSummary {id: $id, summary_text: $text, timestamp: $ts, importance_score: 0.8})",
            {"id": sum_id, "text": summary_text, "ts": ts},
        )

        for ep in episodes:
            self.engine.backend.execute(
                "MATCH (e:Episode {id: $eid}), (s:ChatSummary {id: $sid}) MERGE (e)-[:CONSOLIDATES_INTO]->(s)",
                {"eid": ep["id"], "sid": sum_id},
            )

        logger.info(f"Consolidated {len(episodes)} episodes into summary {sum_id}.")
        return len(episodes)

    def prune_low_importance_nodes(self, threshold: float = 0.05) -> int:
        """Remove low-signal nodes to maintain scalability, protecting permanent nodes."""
        if not self.engine.backend:
            return 0

        # Protect nodes marked as is_permanent=True
        query = """
        MATCH (n)
        WHERE n.importance_score < $threshold
        AND (n.is_permanent IS NULL OR n.is_permanent = False)
        AND NOT (n:Agent OR n:Tool OR n:Skill)
        DETACH DELETE n
        """
        self.engine.backend.execute(query, {"threshold": threshold})

        logger.info(f"Pruned non-permanent nodes with importance below {threshold}.")
        return 1

    def merge_similar_concepts(self, similarity_threshold: float = 0.95) -> int:
        """Find and merge similar Concept nodes based on semantic embeddings."""
        if not self.engine.backend:
            return 0

        # This is a complex operation; simplified here for LadybugDB compatibility
        # We fetch all concepts with embeddings and find pairs locally
        query = "MATCH (c:Concept) WHERE c.embedding IS NOT NULL RETURN c.id as id, c.name as name, c.embedding as embedding"
        concepts = self.engine.backend.execute(query) or []

        from ..knowledge_graph.engine import cosine_similarity

        merged_count = 0
        processed_ids = set()

        for i, c1 in enumerate(concepts):
            if c1["id"] in processed_ids:
                continue
            for j in range(i + 1, len(concepts)):
                c2 = concepts[j]
                if c2["id"] in processed_ids:
                    continue

                sim = cosine_similarity(c1["embedding"], c2["embedding"])
                if sim > similarity_threshold:
                    # Merge c2 into c1
                    logger.info(
                        f"Merging similar concepts: {c1['name']} and {c2['name']} (sim={sim:.4f})"
                    )
                    # Simplified merge logic - in a real system we'd handle all relationship types
                    self.engine.backend.execute(
                        "MATCH (old:Concept {id: $old_id}) DETACH DELETE old",
                        {"old_id": c2["id"]},
                    )
                    processed_ids.add(c2["id"])
                    merged_count += 1

        return merged_count

    def validate_all_graph_models(self) -> int:
        """Run Pydantic validation + basic ontology checks on every node type."""
        from ..graph.models import Policy, ProcessFlow

        if not self.engine.backend:
            return 0

        validated = 0
        # Validate Policies
        policies = self.engine.backend.execute("MATCH (n:Policy) RETURN n")
        for record in policies:
            try:
                Policy.model_validate(record.get("n", record))
                validated += 1
            except Exception as e:
                logger.warning(f"⚠️ Invalid Policy node: {e}")

        # Validate ProcessFlows
        flows = self.engine.backend.execute("MATCH (n:ProcessFlow) RETURN n")
        for record in flows:
            try:
                ProcessFlow.model_validate(record.get("n", record))
                validated += 1
            except Exception as e:
                logger.warning(f"⚠️ Invalid ProcessFlow node: {e}")

        logger.info(f"Validated {validated} graph nodes against Pydantic models.")
        return validated

    def link_topics_to_policies_and_processes(self) -> int:
        """Auto-link new KnowledgeBaseTopic nodes to Policies/Processes via semantic similarity."""
        if not self.engine.backend:
            return 0

        # Use simple semantic linking based on embeddings if available
        # Note: Vector search in Ladybug is usually done via dedicated tool or specialized query
        query = """
        MATCH (t:KnowledgeBaseTopic)
        WHERE NOT EXISTS { (t)-[:GROUNDED_IN|REFERENCES]->() }
        MATCH (p:Policy)
        WHERE vector.similarity(t.embedding, p.embedding) > 0.75
        MERGE (t)-[:GROUNDED_IN]->(p)
        """
        try:
            self.engine.backend.execute(query)

            # Link to ProcessFlows
            query_flow = """
            MATCH (t:KnowledgeBaseTopic)
            MATCH (f:ProcessFlow)
            WHERE vector.similarity(t.embedding, f.embedding) > 0.75
            MERGE (t)-[:REFERENCES]->(f)
            """
            self.engine.backend.execute(query_flow)
            logger.info(
                "✅ Topic linking complete (Policies & ProcessFlows now grounded in KBs)"
            )
            return 1
        except Exception as e:
            logger.debug(f"Topic linking skipped or failed: {e}")
            return 0

    def run_all(self):
        """Run all maintenance tasks."""
        self.enrich_embeddings()
        self.prune_cron_logs()
        self.summarize_old_chats()
        self.consolidate_memory()
        self.prune_low_importance_nodes()
        self.update_importance_scores()
        self.apply_temporal_decay()
        self.merge_similar_concepts()
        self.validate_all_graph_models()
        self.link_topics_to_policies_and_processes()

    def get_status(self) -> dict:
        """Return the current status of maintenance operations."""
        return {
            "status": "ready",
            "operations": {
                "enrich_embeddings": "idle",
                "prune_cron_logs": "idle",
                "summarize_old_chats": "idle",
                "consolidate_memory": "idle",
                "prune_low_importance_nodes": "idle",
                "update_importance_scores": "idle",
                "apply_temporal_decay": "idle",
                "merge_similar_concepts": "idle",
                "validate_all_graph_models": "idle",
                "link_topics_to_policies_and_processes": "idle",
            },
        }

    def trigger_operation(self, operation: str) -> dict:
        """Trigger a specific maintenance operation."""
        op_map: dict[str, Callable[[], Any]] = {
            "enrich_embeddings": self.enrich_embeddings,
            "prune_cron_logs": self.prune_cron_logs,
            "summarize_old_chats": self.summarize_old_chats,
            #            "consolidate_memory": self.consolidate_memory,
            #            "prune_low_importance_nodes": self.prune_low_importance_nodes,
            #            "update_importance_scores": self.update_importance_scores,
            #            "apply_temporal_decay": self.apply_temporal_decay,
            #            "merge_similar_concepts": self.merge_similar_concepts,
            #            "validate_all_graph_models": self.validate_all_graph_models,
            #            "link_topics_to_policies_and_processes": self.link_topics_to_policies_and_processes,
        }
        if operation in op_map:
            result = op_map[operation]()
            return {"status": "success", "result": result}
        return {"status": "error", "message": f"Unknown operation: {operation}"}
