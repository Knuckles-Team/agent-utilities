from __future__ import annotations

"""Ingestion mixin for IntelligenceGraphEngine.

Extracted from engine.py. Contains methods for ingesting episodes,
MCP servers, A2A agent cards, and agent skills into the KG.
"""


import typing

if typing.TYPE_CHECKING:
    from .._engine_protocol import _EngineProtocol

    _Base = _EngineProtocol
else:
    _Base = object


import logging
import time
import uuid
from typing import Any

from ...models.knowledge_graph import (
    CallableResourceNode,
    EpisodeNode,
    ToolMetadataNode,
)
from .context_builder import build_contextual_description

logger = logging.getLogger(__name__)


class IngestionMixin(_Base):
    """Ingestion capabilities for the KG engine."""

    def ingest_episode(
        self, content: str, source: str = "chat", timestamp: str | None = None
    ) -> str:
        """Ingest a new episode into the graph with automatic layer distribution."""
        ep_id = f"ep:{uuid.uuid4().hex[:8]}"
        ts = timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        node = EpisodeNode(
            id=ep_id,
            name=f"Episode {ts}",
            timestamp=ts,
            source=source,
            description=content,
            importance_score=0.5,
        )

        # Generate embedding if model available
        if self.hybrid_retriever.embed_model:
            try:
                ctx_desc = build_contextual_description(
                    node.id, self.graph, node.description or node.name
                )
                node.embedding = self.hybrid_retriever.embed_model.get_text_embedding(
                    ctx_desc
                )
            except Exception as e:
                logger.warning(
                    f"Failed to generate embedding for episode {node.id}: {e}"
                )
        # Tiered write: backend is source of truth, NX is fallback
        if self.backend:
            data = self._serialize_node(node, label="Episode")
            self._upsert_node("Episode", node.id, data)
        else:
            self.graph.add_node(node.id, **node.model_dump())
        return ep_id

    def ingest_mcp_server(
        self,
        name: str,
        url: str,
        tools: list[dict[str, Any]],
        resources: dict[str, Any] | None = None,
    ):
        """Ingest MCP server tools and metadata as CallableResource nodes."""
        server_id = f"srv:{name}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if self.backend:
            self.backend.execute(
                "MERGE (s:Server {id: $id}) SET s.name = $name, s.url = $url, s.timestamp = $ts",
                {"id": server_id, "name": name, "url": url, "ts": ts},
            )

            for tool in tools:
                meta_id = f"meta:{uuid.uuid4().hex[:8]}"
                res_id = f"res:{tool['name']}"

                # Create Metadata
                metadata = ToolMetadataNode(
                    id=meta_id,
                    name=f"Meta for {tool['name']}",
                    description=tool.get("description", ""),
                    source="MCP",
                    tags=tool.get("tags", []),
                    capabilities=tool.get("capabilities", []),
                    resources=resources or {},
                    timestamp=ts,
                )
                # Tiered write: backend is source of truth, NX is fallback
                if self.backend:
                    m_data = self._serialize_node(metadata, label="ToolMetadata")
                    self._upsert_node("ToolMetadata", meta_id, m_data)
                else:
                    self.graph.add_node(metadata.id, **metadata.model_dump())

                # Create Callable Resource
                resource = CallableResourceNode(
                    id=res_id,
                    name=tool["name"],
                    description=tool.get("description", ""),
                    resource_type="MCP_TOOL",
                    endpoint=url,
                    metadata_id=meta_id,
                    timestamp=ts,
                    importance_score=0.8,
                )
                # Generate embedding if model available
                if self.hybrid_retriever.embed_model:
                    try:
                        ctx_desc = build_contextual_description(
                            resource.id,
                            self.graph,
                            resource.description or resource.name,
                        )
                        resource.embedding = (
                            self.hybrid_retriever.embed_model.get_text_embedding(
                                ctx_desc
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate embedding for tool {resource.name}: {e}"
                        )

                # Tiered write: backend is source of truth, NX is fallback
                if self.backend:
                    r_data = self._serialize_node(resource, label="CallableResource")
                    self._upsert_node("CallableResource", res_id, r_data)
                else:
                    self.graph.add_node(resource.id, **resource.model_dump())

                # Linkage
                if self.backend:
                    self.backend.execute(
                        "MATCH (s:Server), (r:CallableResource) WHERE s.id = $sid AND r.id = $rid MERGE (s)-[:PROVIDES]->(r)",
                        {"sid": server_id, "rid": res_id},
                    )
                    self.backend.execute(
                        "MATCH (r:CallableResource), (m:ToolMetadata) WHERE r.id = $rid AND m.id = $mid MERGE (r)-[:HAS_METADATA]->(m)",
                        {"rid": res_id, "mid": meta_id},
                    )

    def ingest_a2a_agent_card(self, url: str, card: dict[str, Any]):
        """Ingest an A2A agent card as a CallableResource node."""
        agent_id = f"agent:{card.get('name', uuid.uuid4().hex[:8])}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta_id = f"meta:{uuid.uuid4().hex[:8]}"

        # Create Metadata
        metadata = ToolMetadataNode(
            id=meta_id,
            name=f"Meta for {agent_id}",
            description=card.get("description", ""),
            source="A2A",
            capabilities=card.get("capabilities", []),
            timestamp=ts,
        )
        # Tiered write: backend is source of truth, NX is fallback
        if self.backend:
            self._upsert_node(
                "ToolMetadata", meta_id, self._serialize_node(metadata, "ToolMetadata")
            )
        else:
            self.graph.add_node(metadata.id, **metadata.model_dump())

        # Create Callable Resource
        resource = CallableResourceNode(
            id=agent_id,
            name=card.get("name", "Unknown Agent"),
            description=card.get("description", ""),
            resource_type="A2A_AGENT",
            endpoint=url,
            agent_card=card,
            metadata_id=meta_id,
            timestamp=ts,
            importance_score=0.9,
        )
        # Generate embedding if model available
        if self.hybrid_retriever.embed_model:
            try:
                ctx_desc = build_contextual_description(
                    resource.id, self.graph, resource.description or resource.name
                )
                resource.embedding = (
                    self.hybrid_retriever.embed_model.get_text_embedding(ctx_desc)
                )
            except Exception as e:
                logger.warning(
                    f"Failed to generate embedding for A2A agent {resource.id}: {e}"
                )
        # Tiered write: backend is source of truth, NX is fallback
        if self.backend:
            self._upsert_node(
                "CallableResource",
                agent_id,
                self._serialize_node(resource, "CallableResource"),
            )
        else:
            self.graph.add_node(resource.id, **resource.model_dump())

        if self.backend:
            self.backend.execute(
                "MATCH (r:CallableResource), (m:ToolMetadata) WHERE r.id = $rid AND m.id = $mid MERGE (r)-[:HAS_METADATA]->(m)",
                {"rid": agent_id, "mid": meta_id},
            )

    def ingest_agent_skill(
        self, skill_file_path: str, frontmatter: dict[str, Any], content: str
    ):
        """Ingest a Claude-style agent skill with frontmatter metadata."""
        skill_id = f"skill:{frontmatter.get('name', uuid.uuid4().hex[:8])}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta_id = f"meta:{uuid.uuid4().hex[:8]}"

        metadata = ToolMetadataNode(
            id=meta_id,
            name=f"Meta for {skill_id}",
            description=frontmatter.get("description", ""),
            source="AGENT_SKILL",
            tags=frontmatter.get("tags", []),
            capabilities=frontmatter.get("capabilities", []),
            timestamp=ts,
        )
        # Tiered write: backend is source of truth, NX is fallback
        if self.backend:
            self._upsert_node(
                "ToolMetadata", meta_id, self._serialize_node(metadata, "ToolMetadata")
            )
        else:
            self.graph.add_node(metadata.id, **metadata.model_dump())

        resource = CallableResourceNode(
            id=skill_id,
            name=frontmatter.get("name", skill_id),
            description=frontmatter.get("description", ""),
            resource_type="AGENT_SKILL",
            skill_code_path=skill_file_path,
            metadata_id=meta_id,
            timestamp=ts,
            importance_score=0.7,
        )
        # Generate embedding if model available
        if self.hybrid_retriever.embed_model:
            try:
                ctx_desc = build_contextual_description(
                    resource.id, self.graph, resource.description or resource.name
                )
                resource.embedding = (
                    self.hybrid_retriever.embed_model.get_text_embedding(ctx_desc)
                )
            except Exception as e:
                logger.warning(
                    f"Failed to generate embedding for skill {resource.id}: {e}"
                )
        # Tiered write: backend is source of truth, NX is fallback
        if self.backend:
            self._upsert_node(
                "CallableResource",
                skill_id,
                self._serialize_node(resource, "CallableResource"),
            )
        else:
            self.graph.add_node(resource.id, **resource.model_dump())

        if self.backend:
            self.backend.execute(
                "MATCH (r:CallableResource), (m:ToolMetadata) WHERE r.id = $rid AND m.id = $mid MERGE (r)-[:HAS_METADATA]->(m)",
                {"rid": skill_id, "mid": meta_id},
            )

    def ingest_external_batch(
        self,
        domain: str,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Ingest a batch of standardized entities and relationships from a peripheral agent.

        This is the primary ingestion API for the hub-and-spoke model (e.g., AD, ServiceNow).
        Entities are expected to be pre-mapped to the BFO/PROV-O ontology.
        """
        if not self.backend:
            logger.warning(
                "Backend not available for batch ingestion. Falling back to slow NetworkX loop."
            )
            # NetworkX loop fallback
            for e in entities:
                eid = e.get("id")
                if eid:
                    self.add_node(str(eid), str(e.get("type", "Entity")), e)
            if relationships:
                for r in relationships:
                    src = r.get("source")
                    tgt = r.get("target")
                    rtype = r.get("type")
                    if src and tgt and rtype:
                        self.link_nodes(str(src), str(tgt), str(rtype), r)
            return {"status": "success", "nodes": len(entities), "backend": False}

        # Use backend high-throughput UNWIND batching
        q_nodes = """
        UNWIND $batch AS row
        MERGE (n:DomainEntity {id: row.id})
        SET n += row
        """
        self.backend.execute_batch(q_nodes, entities)

        if relationships:
            q_rels = """
            UNWIND $batch AS row
            MATCH (s {id: row.source})
            MATCH (t {id: row.target})
            MERGE (s)-[r:EXTERNAL_LINK {type: row.type}]->(t)
            SET r += row
            """
            self.backend.execute_batch(q_rels, relationships)

        return {"status": "success", "nodes": len(entities), "backend": True}

    def re_embed_node(self, node_id: str) -> bool:
        """Dynamically re-calculate and store the context-aware embedding for a node.

        Fetches the node from the graph, generates its topological context (OWL facts +
        hierarchical radius), embeds the context, and upserts the node.
        """
        if not self.hybrid_retriever.embed_model:
            return False

        if node_id not in self.graph:
            return False

        data = dict(self.graph.nodes[node_id])
        base_desc = (
            data.get("description") or data.get("name") or data.get("claim_text") or ""
        )

        try:
            ctx_desc = build_contextual_description(node_id, self.graph, base_desc)
            new_embedding = self.hybrid_retriever.embed_model.get_text_embedding(
                ctx_desc
            )

            # Update NX graph
            self.graph.nodes[node_id]["embedding"] = new_embedding

            # Update Backend
            if self.backend:
                node_type = data.get("type", "Entity")
                self.backend.execute(
                    f"MATCH (n:{node_type}) WHERE n.id = $id SET n.embedding = $emb",
                    {"id": node_id, "emb": new_embedding},
                )
            return True
        except Exception as e:
            logger.warning(f"Failed to re-embed node {node_id}: {e}")
            return False

    # --- Engineering Rules Ingestion (CONCEPT:KG-2.2) ---

    def ingest_engineering_rules(
        self,
        rules_books_path: str | None = None,
        tiers: list[str] | None = None,
        version: str = "1.0.0",
    ) -> dict[str, Any]:
        """Ingest engineering rules into the Knowledge Graph.

        CONCEPT:KG-2.2 — Engineering Rules Engine

        Parses structured markdown rule files and creates versioned KG
        nodes for context-sensitive retrieval, OWL reasoning, and AHE
        efficacy tracking.

        Uses the bundled ``data/engineering_rules/`` by default. Pass
        an external path to override with a custom rule set.

        Args:
            rules_books_path: Optional external path to agent-rules-books.
                Defaults to bundled data shipped with the package.
            tiers: Which tiers to ingest. Defaults to ``["mini"]``.
            version: Semantic version for this ingestion round.

        Returns:
            Statistics dict with counts of ingested books, rules, and edges.
        """
        from .rule_ingestor import RuleIngestor

        ingestor = RuleIngestor(self)  # type: ignore
        return ingestor.ingest_rules_books(
            rules_books_path=rules_books_path,
            tiers=tiers,
            version=version,
        )

    # --- Constitution & Policy Ingestion (CONCEPT:KG-2.2 Extension) ---

    def ingest_constitution(
        self,
        workspace_path: str,
        version: str = "1.0.0",
    ) -> dict[str, Any]:
        """Ingest a project constitution into the KG as PolicyNodes.

        CONCEPT:KG-2.2 — Engineering Rules Engine (Constitution Extension)

        Searches for constitution files in standard SDD locations
        (``.specify/memory/constitution.md``, ``CONSTITUTION.md``, etc.)
        and extracts governance rules, normative statements, quality gates,
        and tech stack constraints into the KG's policy layer.

        Args:
            workspace_path: Absolute path to the project workspace root.
            version: Semantic version for this ingestion round.

        Returns:
            Statistics dict with counts of ingested policies and edges.
        """
        from .policy_ingestor import PolicyIngestor

        ingestor = PolicyIngestor(self)  # type: ignore
        return ingestor.ingest_constitution(
            workspace_path=workspace_path,
            version=version,
        )

    def ingest_all_policies(
        self,
        workspace_path: str,
        rules_books_path: str | None = None,
        version: str = "1.0.0",
    ) -> dict[str, Any]:
        """Ingest all policy sources into the KG in one call.

        CONCEPT:KG-2.2 — Unified Policy Ingestion

        Combines:
            1. Constitution policies (from workspace SDD governance)
            2. Prompt rules (from agent_utilities/prompts/)
            3. Engineering rules (from agent-rules-books, if path given)

        Args:
            workspace_path: Project workspace root.
            rules_books_path: Optional path to agent-rules-books repo.
            version: Semantic version.

        Returns:
            Combined statistics dict.
        """
        from .policy_ingestor import PolicyIngestor

        ingestor = PolicyIngestor(self)  # type: ignore
        return ingestor.ingest_all(
            workspace_path=workspace_path,
            rules_books_path=rules_books_path,
            version=version,
        )

    # --- Discovery & Retrieval Tools ---

    # --- Knowledge Distillation (CONCEPT:KG-2.23) ---

    def ingest_ideablock(
        self,
        question: str,
        answer: str,
        name: str = "",
        tags: list[str] | None = None,
        keywords: list[str] | None = None,
        entities: list[dict[str, str]] | None = None,
        source: str = "manual",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Ingest a single structured IdeaBlock into the Knowledge Graph.

        CONCEPT:KG-2.23 — Knowledge Distillation Engine

        Creates an atomic knowledge unit as a question-answer pair with
        governance metadata, persists it to the KG, and returns its ID.

        Args:
            question: The critical question this block answers.
            answer: The trusted, validated answer.
            name: Optional human-readable title.
            tags: Classification tags (e.g., IMPORTANT, TECHNOLOGY).
            keywords: BM25-optimized retrieval keywords.
            entities: Named entity references (list of dicts with name/type).
            source: Provenance source identifier.
            metadata: Additional metadata dict.

        Returns:
            The created IdeaBlock node ID.
        """
        from typing import cast

        from ..distillation import DistillationEngine
        from .engine import IntelligenceGraphEngine

        engine = DistillationEngine(kg_engine=cast(IntelligenceGraphEngine, self))
        block = engine.ingest_ideablock(
            question=question,
            answer=answer,
            name=name,
            tags=tags,
            keywords=keywords,
            entities=entities,
            source=source,
            metadata=metadata,
        )
        return block["id"]

    def distill_knowledge(
        self,
        block_ids: list[str] | None = None,
        iterations: int = 3,
        threshold: float = 0.65,
    ) -> dict[str, Any]:
        """Run iterative knowledge distillation on IdeaBlocks in the KG.

        CONCEPT:KG-2.23 — Knowledge Distillation Engine

        Performs semantic deduplication via LSH + cosine similarity,
        community clustering, and LLM-powered merging of redundant
        knowledge blocks.

        Args:
            block_ids: Optional list of block IDs to distill.
                If None, distills all IDEA_BLOCK nodes in the graph.
            iterations: Number of deduplication rounds (default 3).
            threshold: Base similarity threshold (default 0.65).

        Returns:
            Dict with ``stats`` and ``rounds`` metrics.
        """
        from typing import cast

        from ..distillation import DistillationEngine, KnowledgeDeduplicator
        from .engine import IntelligenceGraphEngine

        dedup = KnowledgeDeduplicator(
            iterations=iterations,
            base_threshold=threshold,
        )
        engine = DistillationEngine(
            kg_engine=cast(IntelligenceGraphEngine, self),
            deduplicator=dedup,
        )

        # Collect IdeaBlock nodes from the graph
        if block_ids:
            ids = block_ids
        else:
            ids = [
                nid
                for nid, data in self.graph.nodes(data=True)
                if data.get("type") == "idea_block" or str(nid).startswith("ideablock:")
            ]

        if not ids:
            return {"stats": {"starting_count": 0, "final_count": 0}, "rounds": []}

        # Load blocks into the distillation engine
        for nid in ids:
            data = dict(self.graph.nodes.get(nid, {}))
            block = {
                "id": nid,
                "embedding": data.get("embedding"),
                "critical_question": data.get("critical_question", ""),
                "trusted_answer": data.get("trusted_answer", ""),
                "name": data.get("name", nid),
                "description": data.get("description", ""),
                "tags": data.get("tags", []),
                "keywords": data.get("keywords", []),
            }
            engine._blocks[nid] = block

        return engine.distill(iterations=iterations, base_threshold=threshold)
