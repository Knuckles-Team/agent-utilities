#!/usr/bin/python
# coding: utf-8
"""Knowledge Graph Maintenance & Pruning."""

import logging
import requests
from typing import List, Optional
from datetime import datetime, timedelta

from .engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

# LM Studio default configuration
LM_STUDIO_URL = "http://localhost:1234/v1/embeddings"
EMBEDDING_MODEL = "nomic-embed-text-v1.5"


def generate_embedding(text: str) -> Optional[List[float]]:
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
            msgs = self.engine.backend.execute(m_query, {"t_id": t_id})

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

        from datetime import timezone

        # Ensure we handle potential string/float conversion issues from backend
        results = self.engine.backend.execute(
            "MATCH (n) WHERE n.importance_score IS NOT NULL AND n.timestamp IS NOT NULL RETURN n.id as id, n.timestamp as ts, n.importance_score as score"
        )
        updated = 0
        now = datetime.now(timezone.utc)
        for row in results:
            try:
                # Basic ISO format parsing
                ts_str = row["ts"].replace("Z", "+00:00")
                ts = datetime.fromisoformat(ts_str)
                # Ensure ts is aware
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

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
        """Remove low-signal nodes to maintain scalability."""
        if not self.engine.backend:
            return 0

        # Protect certain labels from pruning
        query = "MATCH (n) WHERE n.importance_score < $threshold AND NOT n:Agent AND NOT n:Server DETACH DELETE n"
        self.engine.backend.execute(query, {"threshold": threshold})

        logger.info(f"Pruned nodes with importance below {threshold}.")
        return 1

    def run_all(self):
        """Run all maintenance tasks."""
        self.enrich_embeddings()
        self.prune_cron_logs()
        self.summarize_old_chats()
        self.consolidate_memory()
        self.prune_low_importance_nodes()
        self.update_importance_scores()
        self.apply_temporal_decay()
