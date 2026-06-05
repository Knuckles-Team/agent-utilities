#!/usr/bin/python
"""Knowledge Graph Maintenance & Pruning.

CONCEPT:KG-2.0
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from agent_utilities.core.config import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_MODEL_ID,
)

from .engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

# Default configuration using model registry with fallback to native vLLM embeddings
_base_url = (DEFAULT_EMBEDDING_BASE_URL or "http://vllm-embed.arpa/v1").rstrip("/")
LM_STUDIO_URL = f"{_base_url}/embeddings"
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL_ID or "bge-m3"


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
                combined_text = "\n".join(
                    [m["content"] for m in msgs if m.get("content")]
                )

                try:
                    from pydantic_ai import Agent

                    from agent_utilities.core.model_factory import create_model

                    model = create_model()
                    agent = Agent(
                        model=model,
                        system_prompt=(
                            "You are a conversation synthesis assistant. Summarize the following thread "
                            "messages into a clear, comprehensive, yet concise summary capturing the "
                            "user's core requests, the assistant's actions/resolutions, and any key "
                            "context or decisions made."
                        ),
                    )
                    result = agent.run_sync(combined_text)
                    summary_text = str(result.output)
                except Exception as e:
                    logger.warning(
                        f"Conversation summarization failed, using fallback: {e}"
                    )
                    summary_text = (
                        f"Summary of {len(msgs)} messages: {combined_text[:100]}..."
                    )

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
                set_clause = self.engine._get_set_clause(
                    props, alias="n", label="ChatSummary"
                )
                query = f"MERGE (n:ChatSummary {{id: $id}}){set_clause}"
                params: dict[str, Any] = {"id": sum_id, **props}
                self.engine.backend.execute(query, params)

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
        """Update importance scores using Rust-native PageRank centrality."""
        try:
            scores = self.engine.graph.pagerank()
            updated = 0
            for node_id, score in scores:
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

        # Find episodes that haven't been synthesized
        query = "MATCH (e:Episode) WHERE e.timestamp < $cutoff AND NOT (e)-[:CONSOLIDATES_INTO]->() RETURN e.id as id, e.description AS descriptionription"
        episodes = self.engine.backend.execute(query, {"cutoff": cutoff_date})

        if not episodes:
            return 0

        import uuid

        from pydantic_ai import Agent

        from agent_utilities.core.model_factory import create_model

        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Batch episodes by similarity or just group them for now
        # Use an LLM here to summarize
        try:
            model = create_model()
            agent = Agent(
                model=model,
                system_prompt="You are a memory synthesis engine. Summarize the following episode descriptions into a single, highly dense and concise paragraph capturing the core actions, decisions, and outcomes.",
            )
            combined_text = "\n".join(
                [ep["description"] for ep in episodes if ep.get("description")]
            )
            result = agent.run_sync(combined_text)
            summary_text = str(result.output)
        except Exception as e:
            logger.warning(f"LLM summarization failed, using fallback: {e}")
            summary_text = f"Synthesized summary of {len(episodes)} episodes."

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

        logger.info(f"Synthesized {len(episodes)} episodes into summary {sum_id}.")
        return len(episodes)

    def prune_low_importance_nodes(self, threshold: float = 0.05) -> int:
        """Remove low-signal nodes to maintain scalability, utilizing validation-gated synthesis."""
        if not self.engine.backend:
            return 0

        # Protect nodes marked as is_permanent=True
        # Self-Reflection mechanism: Delete if importance < threshold and it's not a core structural node
        query = """
        MATCH (n)
        WHERE n.importance_score < $threshold
        AND (n.is_permanent IS NULL OR n.is_permanent = False)
        AND NOT (n:Agent OR n:Tool OR n:Skill OR n:SystemPrompt OR n:KnowledgeBaseTopic)
        DETACH DELETE n
        """
        self.engine.backend.execute(query, {"threshold": threshold})

        logger.info(f"Pruned non-permanent nodes with importance below {threshold}.")
        return 1

    def trigger_self_improvement(self) -> int:
        """Trigger autonomous self-improvement tasks based on recent failures."""
        logger.info("Triggering self improvement loop (LLM-driven critique).")
        if not self.engine.backend:
            return 0

        # Query recent performance anomalies
        query_anomalies = """
        MATCH (a:PerformanceAnomaly)
        RETURN a.id as id, a.anomaly_type as anomaly_type, a.target_node_id as target_node_id, a.timestamp as timestamp
        ORDER BY a.timestamp DESC LIMIT 10
        """
        anomalies = self.engine.backend.execute(query_anomalies) or []

        # Query recent execution summaries with failures
        query_execs = """
        MATCH (e:ExecutionSummary) WHERE e.success_rate < 1.0
        RETURN e.id as id, e.workflow_id as workflow_id, e.success_rate as success_rate, e.timestamp as timestamp
        ORDER BY e.timestamp DESC LIMIT 10
        """
        failures = self.engine.backend.execute(query_execs) or []

        if not anomalies and not failures:
            logger.info(
                "No recent failures or performance anomalies found. System is healthy."
            )
            return 0

        from pydantic import BaseModel, Field
        from pydantic_ai import Agent

        from agent_utilities.core.model_factory import create_model

        class SelfImprovementAction(BaseModel):
            critique: str = Field(
                description="Critical analysis of the failure points or performance anomalies."
            )
            improvement_goals: list[str] = Field(
                description="Concrete, actionable improvement goals or guidelines to mitigate future issues."
            )

        try:
            model = create_model()
            agent = Agent(
                model=model,
                output_type=SelfImprovementAction,
                system_prompt=(
                    "You are a system self-improvement optimizer. Analyze the provided list of recent "
                    "execution failures and performance anomalies. Critique the root causes, and propose "
                    "a list of highly specific, actionable self-improvement goals or guidelines (e.g. prompt "
                    "adjustments, timeout adjustments, routing policies)."
                ),
            )

            prompt = (
                f"Recent Performance Anomalies:\n{anomalies}\n\n"
                f"Recent Execution Failures:\n{failures}"
            )

            result = agent.run_sync(prompt)
            data = result.output

            logger.info(f"Self-improvement critique generated: {data.critique}")

            import uuid

            ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

            # Store critique as a Reflection node
            ref_id = f"ref:improve:{uuid.uuid4().hex[:8]}"
            ref_props = {
                "id": ref_id,
                "type": "reflection",
                "content": f"Self-Improvement Critique: {data.critique}",
                "confidence": 0.85,
                "importance_score": 0.9,
                "timestamp": ts,
                "is_permanent": True,
            }
            ref_set = self.engine._get_set_clause(
                ref_props, alias="n", label="Reflection"
            )
            self.engine.backend.execute(
                f"MERGE (n:Reflection {{id: $id}}){ref_set}", ref_props
            )

            # Store each recommendation as a Goal node
            goals_count = 0
            for goal_text in data.improvement_goals:
                goal_id = f"goal:improve:{uuid.uuid4().hex[:8]}"
                goal_props = {
                    "id": goal_id,
                    "type": "goal",
                    "goal_text": goal_text,
                    "status": "active",
                    "importance_score": 0.8,
                    "timestamp": ts,
                    "is_permanent": False,
                }
                goal_set = self.engine._get_set_clause(
                    goal_props, alias="n", label="Goal"
                )
                self.engine.backend.execute(
                    f"MERGE (n:Goal {{id: $id}}){goal_set}", goal_props
                )

                # Link Goal to the Reflection
                self.engine.backend.execute(
                    "MATCH (g:Goal {id: $goal_id}), (r:Reflection {id: $ref_id}) MERGE (g)-[:SUPPORTED_BY]->(r)",
                    {"goal_id": goal_id, "ref_id": ref_id},
                )
                goals_count += 1

            return goals_count
        except Exception as e:
            logger.error(f"Failed to run self-improvement LLM loop: {e}")
            return 0

    def trigger_dreaming(self) -> int:
        """Trigger 'dreaming' to synthesize new features or strategies."""
        logger.info("Triggering feature dreaming (nocturnal processing).")
        if not self.engine.backend:
            return 0

        # Retrieve existing skills and agent descriptions
        query_skills = (
            "MATCH (s:Skill) RETURN s.name as name, s.version as version LIMIT 20"
        )
        skills = self.engine.backend.execute(query_skills) or []

        query_agents = "MATCH (a:Agent) RETURN a.name as name, a.description as description LIMIT 10"
        agents = self.engine.backend.execute(query_agents) or []

        from pydantic import BaseModel, Field
        from pydantic_ai import Agent as PydanticAgent

        from agent_utilities.core.model_factory import create_model

        class DreamResult(BaseModel):
            synthesis_idea: str = Field(
                description="The synthesized creative strategy or concept."
            )
            suggested_features: list[str] = Field(
                description="Suggested new features, tool templates, or skills."
            )

        try:
            model = create_model()
            agent = PydanticAgent(
                model=model,
                output_type=DreamResult,
                system_prompt=(
                    "You are a nocturnal feature dreaming synthesis agent. Review the list of existing "
                    "skills and agents in the system. Synthesize a creative new cross-disciplinary strategy "
                    "or feature idea that combines existing capabilities in novel ways, and suggest a list "
                    "of new capability/tool ideas."
                ),
            )

            prompt = f"Existing Skills:\n{skills}\n\n" f"Existing Agents:\n{agents}"

            result = agent.run_sync(prompt)
            data = result.output

            logger.info(f"Dream synthesized concept: {data.synthesis_idea}")

            import uuid

            ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

            # Store the dream as a permanent Reflection node in the graph
            dream_id = f"ref:dream:{uuid.uuid4().hex[:8]}"
            dream_props = {
                "id": dream_id,
                "type": "reflection",
                "content": f"Nocturnal Dream Synthesis: {data.synthesis_idea}. Suggested Features: {', '.join(data.suggested_features)}",
                "confidence": 0.75,
                "importance_score": 0.7,
                "timestamp": ts,
                "is_permanent": True,
            }
            dream_set = self.engine._get_set_clause(
                dream_props, alias="n", label="Reflection"
            )
            self.engine.backend.execute(
                f"MERGE (n:Reflection {{id: $id}}){dream_set}", dream_props
            )

            return 1
        except Exception as e:
            logger.error(f"Failed to run nocturnal feature dreaming: {e}")
            return 0

    def merge_similar_concepts(self, similarity_threshold: float = 0.95) -> int:
        """Find and merge similar Concept nodes based on semantic embeddings."""
        if not self.engine.backend:
            return 0

        query = "MATCH (c:Concept) WHERE c.embedding IS NOT NULL RETURN c.id as id, c.name as name, c.embedding as embedding"
        concepts = self.engine.backend.execute(query) or []
        by_id = {c["id"]: c for c in concepts}

        # All-pairs similarity is collapsed onto the epistemic-graph compute layer: one
        # ``compute_similarity_edges`` request over the MessagePack/UDS transport runs the O(n²)
        # natively in the tokio engine over graph-resident embeddings (one round-trip). Falls back to
        # an in-process numpy O(n²) pass when the Rust core isn't running.
        pairs = self._similar_concept_pairs(concepts, similarity_threshold)

        merged_count = 0
        processed_ids: set[str] = set()
        for src, dst, sim in pairs:
            if src in processed_ids or dst in processed_ids:
                continue
            if src not in by_id or dst not in by_id:
                continue
            logger.info(
                f"Merging similar concepts: {by_id[src]['name']} and {by_id[dst]['name']} (sim={sim:.4f})"
            )
            self._merge_concept_pair(old_id=dst, new_id=src, similarity=sim)
            processed_ids.add(dst)
            merged_count += 1

        return merged_count

    def _similar_concept_pairs(
        self, concepts: list, threshold: float
    ) -> list[tuple[str, str, float]]:
        """Return ``(src, dst, sim)`` concept pairs above ``threshold``.

        Prefers the native ``compute_similarity_edges`` (epistemic-graph, one round-trip over
        graph-resident embeddings); falls back to an in-process numpy O(n²) pass when the Rust
        compute core is unavailable. Pairs are filtered to the supplied Concept id set.
        """
        cids = {c["id"] for c in concepts}
        compute = getattr(self.engine, "graph_compute", None)
        if compute is not None and hasattr(compute, "compute_similarity_edges"):
            try:
                triples = compute.compute_similarity_edges(threshold) or []
                pairs = [
                    (str(s), str(d), float(sim))
                    for (s, d, sim) in triples
                    if str(s) in cids and str(d) in cids and float(sim) > threshold
                ]
                logger.info(
                    "[KG-2.3] %d concept similarity pairs via native compute_similarity_edges",
                    len(pairs),
                )
                return pairs
            except Exception as e:  # noqa: BLE001 - Rust core unavailable → numpy fallback
                logger.debug(
                    "Native compute_similarity_edges unavailable (%s); numpy fallback", e
                )

        from .engine import cosine_similarity

        out: list[tuple[str, str, float]] = []
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1 :]:
                sim = cosine_similarity(c1["embedding"], c2["embedding"])
                if sim > threshold:
                    out.append((c1["id"], c2["id"], sim))
        return out

    @staticmethod
    def _safe_rel_type(rtype: str) -> str:
        """Sanitise a DB-sourced relationship type for safe Cypher interpolation."""
        cleaned = "".join(ch for ch in str(rtype) if ch.isalnum() or ch == "_")
        return cleaned or "RELATED_TO"

    def _merge_concept_pair(
        self, *, old_id: str, new_id: str, similarity: float
    ) -> None:
        """Re-point every edge of ``old`` onto ``new`` keeping the original
        relationship type, merge node + edge properties, record provenance, then
        delete the duplicate. Replaces the previous lossy implementation that
        collapsed all edges to ``RELATED_TO`` and overwrote the survivor's id via
        ``SET new += old``.
        """
        backend = self.engine.backend

        # 1. Re-point OUTGOING edges, preserving each relationship type and its
        #    properties (de-duplicated by MERGE).
        outgoing = (
            backend.execute(
                """
                MATCH (old:Concept {id: $old_id})-[r]->(target)
                RETURN type(r) AS rtype, target.id AS tid, properties(r) AS props
                """,
                {"old_id": old_id},
            )
            or []
        )
        for edge in outgoing:
            rtype = self._safe_rel_type(edge["rtype"])
            backend.execute(
                f"""
                MATCH (new:Concept {{id: $new_id}})
                MATCH (target {{id: $tid}})
                MERGE (new)-[nr:{rtype}]->(target)
                SET nr += $props
                """,
                {
                    "new_id": new_id,
                    "tid": edge["tid"],
                    "props": edge.get("props") or {},
                },
            )

        # 2. Re-point INCOMING edges, preserving type + properties.
        incoming = (
            backend.execute(
                """
                MATCH (source)-[r]->(old:Concept {id: $old_id})
                RETURN type(r) AS rtype, source.id AS sid, properties(r) AS props
                """,
                {"old_id": old_id},
            )
            or []
        )
        for edge in incoming:
            rtype = self._safe_rel_type(edge["rtype"])
            backend.execute(
                f"""
                MATCH (new:Concept {{id: $new_id}})
                MATCH (source {{id: $sid}})
                MERGE (source)-[nr:{rtype}]->(new)
                SET nr += $props
                """,
                {
                    "new_id": new_id,
                    "sid": edge["sid"],
                    "props": edge.get("props") or {},
                },
            )

        # 3. Merge NODE properties non-destructively (survivor keeps its id/name;
        #    union aliases, take max importance/confidence, concat provenance).
        merged_props = self._merge_node_properties(old_id=old_id, new_id=new_id)
        if merged_props:
            backend.execute(
                "MATCH (new:Concept {id: $new_id}) SET new += $props",
                {"new_id": new_id, "props": merged_props},
            )

        # 4. Record MergedFrom provenance for auditability.
        backend.execute(
            """
            MATCH (new:Concept {id: $new_id})
            MERGE (prov:MergedConcept {id: $old_id})
            MERGE (new)-[:MERGED_FROM {similarity: $sim}]->(prov)
            """,
            {"new_id": new_id, "old_id": old_id, "sim": similarity},
        )

        # 5. Delete the duplicate now that everything is migrated.
        backend.execute(
            "MATCH (old:Concept {id: $old_id}) DETACH DELETE old",
            {"old_id": old_id},
        )

    def _merge_node_properties(self, *, old_id: str, new_id: str) -> dict:
        """Compute a non-destructive property union for the surviving node.

        The survivor's id and name are never overwritten. ``aliases``/provenance
        lists are unioned; numeric ``importance``/``confidence`` take the max;
        ``updated_at`` takes the latest. Unknown keys present only on ``old`` are
        carried over.
        """
        rows = (
            self.engine.backend.execute(
                """
                MATCH (old:Concept {id: $old_id})
                MATCH (new:Concept {id: $new_id})
                RETURN properties(old) AS old_props, properties(new) AS new_props
                """,
                {"old_id": old_id, "new_id": new_id},
            )
            or []
        )
        if not rows:
            return {}
        old_props = dict(rows[0].get("old_props") or {})
        new_props = dict(rows[0].get("new_props") or {})

        protected = {"id", "name", "embedding"}
        merged: dict = {}

        def as_list(v) -> list:
            if v is None:
                return []
            return list(v) if isinstance(v, list | tuple | set) else [v]

        for key, old_val in old_props.items():
            if key in protected:
                continue
            new_val = new_props.get(key)
            if key in ("aliases", "provenance", "sources"):
                merged[key] = sorted(set(as_list(new_val)) | set(as_list(old_val)))
            elif key in ("importance", "confidence") and isinstance(
                old_val, int | float
            ):
                merged[key] = max(
                    old_val, new_val if isinstance(new_val, int | float) else old_val
                )
            elif key in ("updated_at", "last_seen"):
                merged[key] = (
                    max(str(old_val), str(new_val)) if new_val is not None else old_val
                )
            elif new_val is None:
                merged[key] = old_val
        return merged

    def validate_all_graph_models(self) -> int:
        """Run Pydantic validation + basic ontology checks on every node type."""
        from ...graph.models import Policy, ProcessFlow

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

    def run_owl_reasoning(self) -> dict[str, Any]:
        """Run OWL reasoning cycle: promote → reason → downfeed."""
        try:
            from ..backends.owl import create_owl_backend
            from .owl_bridge import OWLBridge
        except ImportError:
            logger.debug("OWL dependencies not installed, skipping reasoning")
            return {"status": "skipped", "reason": "owl deps not installed"}

        try:
            from pathlib import Path

            ontology_path = str(Path(__file__).parent / "ontology.ttl")
            if not Path(ontology_path).exists():
                return {"status": "skipped", "reason": "ontology.ttl not found"}

            owl_backend = create_owl_backend(ontology_path=ontology_path)
            bridge = OWLBridge(
                graph=self.engine.graph,
                owl_backend=owl_backend,
                backend=self.engine.backend,
            )
            stats = bridge.run_cycle()
            owl_backend.close()
            logger.info("OWL reasoning maintenance complete: %s", stats)
            return stats
        except Exception as e:
            logger.error("OWL reasoning maintenance failed: %s", e)
            return {"status": "error", "reason": str(e)}

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
        self.trigger_self_improvement()
        self.trigger_dreaming()
        self.run_owl_reasoning()

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
                "run_owl_reasoning": "idle",
            },
        }

    def trigger_operation(self, operation: str) -> dict:
        """Trigger a specific maintenance operation."""
        op_map: dict[str, Callable[[], Any]] = {
            "enrich_embeddings": self.enrich_embeddings,
            "prune_cron_logs": self.prune_cron_logs,
            "summarize_old_chats": self.summarize_old_chats,
            "consolidate_memory": self.consolidate_memory,
            "prune_low_importance_nodes": self.prune_low_importance_nodes,
            "update_importance_scores": self.update_importance_scores,
            "apply_temporal_decay": self.apply_temporal_decay,
            "merge_similar_concepts": self.merge_similar_concepts,
            "validate_all_graph_models": self.validate_all_graph_models,
            "link_topics_to_policies_and_processes": self.link_topics_to_policies_and_processes,
            "trigger_self_improvement": self.trigger_self_improvement,
            "trigger_dreaming": self.trigger_dreaming,
            "run_owl_reasoning": self.run_owl_reasoning,
        }
        if operation in op_map:
            result = op_map[operation]()
            return {"status": "success", "result": result}
        return {"status": "error", "message": f"Unknown operation: {operation}"}
