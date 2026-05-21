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


import json
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
            env_str = json.dumps(resources.get("env", {})) if resources else "{}"
            self.backend.execute(
                "MERGE (s:Server {id: $id}) SET s.name = $name, s.url = $url, s.timestamp = $ts, s.env = $env",
                {"id": server_id, "name": name, "url": url, "ts": ts, "env": env_str},
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
        # Check backend capabilities
        is_ladybug = self.backend.__class__.__name__ == "LadybugBackend"

        if is_ladybug:
            # Iterative fallback since Ladybug doesn't support UNWIND
            for row in entities:
                node_type = row.get("type", "DomainEntity")
                set_clause = self._get_set_clause(row, "n", label=node_type)
                q_node = f"MERGE (n:{node_type} {{id: $id}}){set_clause}"
                self.backend.execute(q_node, row)

            if relationships:
                for row in relationships:
                    set_clause = self._get_set_clause(row, "r")
                    q_rel = f"""
                    MATCH (s {{id: $source}})
                    MATCH (t {{id: $target}})
                    MERGE (s)-[r:EXTERNAL_LINK {{type: $type}}]->(t){set_clause}
                    """
                    self.backend.execute(q_rel, row)
        else:
            # Use Neo4j/FalkorDB high-throughput UNWIND batching
            # dynamically generate SET keys to avoid SET n += row
            if entities:
                keys = [k for k in entities[0].keys() if k != "id"]
                set_clause = (
                    "SET " + ", ".join([f"n.{k} = row.{k}" for k in keys])
                    if keys
                    else ""
                )
                q_nodes = f"""
                UNWIND $batch AS row
                MERGE (n:DomainEntity {{id: row.id}})
                {set_clause}
                """
                self.backend.execute_batch(q_nodes, entities)

            if relationships:
                r_keys = [
                    k
                    for k in relationships[0].keys()
                    if k not in ("source", "target", "type")
                ]
                r_set_clause = (
                    "SET " + ", ".join([f"r.{k} = row.{k}" for k in r_keys])
                    if r_keys
                    else ""
                )
                q_rels = f"""
                UNWIND $batch AS row
                MATCH (s {{id: row.source}})
                MATCH (t {{id: row.target}})
                MERGE (s)-[r:EXTERNAL_LINK {{type: row.type}}]->(t)
                {r_set_clause}
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
        from ..security.rule_ingestor import RuleIngestor

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
        from ..security.policy_ingestor import PolicyIngestor

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
        from ..security.policy_ingestor import PolicyIngestor

        ingestor = PolicyIngestor(self)  # type: ignore
        return ingestor.ingest_all(
            workspace_path=workspace_path,
            rules_books_path=rules_books_path,
            version=version,
        )

    # --- Discovery & Retrieval Tools ---

    # --- Knowledge Distillation (CONCEPT:KG-2.2) ---

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

        CONCEPT:KG-2.2 — Knowledge Distillation Engine

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

        CONCEPT:KG-2.2 — Knowledge Distillation Engine

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

    # ------------------------------------------------------------------
    # CONCEPT:ECO-4.0 — Unified Agent Toolkit Ingestion
    # ------------------------------------------------------------------

    async def ingest_agent_toolkit(
        self,
        sources: list[str],
        agent_card_path: str = "/.well-known/agent.json",
    ) -> dict[str, Any]:
        """Ingest MCP server configs, agent skill directories, and A2A agent cards.

        CONCEPT:ECO-4.0 — Unified MCP/Skill/A2A ingestion pipeline with live
        tool discovery.

        Accepts a list of mixed sources and auto-detects the type of each:

        1. **URL** (``http://`` or ``https://``) → fetches the A2A agent card
           from ``agent_card_path`` (default ``/.well-known/agent.json``,
           override with ``/agent-card.json``).
        2. **JSON file** containing ``"mcpServers"`` key → parses as an MCP
           config and ingests each server entry with live tool discovery.
        3. **Directory** containing ``SKILL.md`` → parses frontmatter and
           ingests as an agent skill.
        4. **Remote JSON URL** (ending ``.json``) → fetches and checks for
           ``mcpServers`` key to determine if it is an MCP config.

        For MCP servers, the pipeline will:
        - Parse the config to extract server entries and tool-enable flags.
        - Attempt live connection (``list_tools()``) to discover real tools.
        - Fall back to tool-flag parsing if live connect fails.
        - Check KG freshness via config hash before re-ingesting.

        Args:
            sources: List of file paths, directory paths, or URLs.
            agent_card_path: Well-known path for A2A agent cards.
                Defaults to ``/.well-known/agent.json``. Override with
                ``/agent-card.json`` for non-standard agents.

        Returns:
            Summary dict with keys: ``mcp_servers``, ``tools_discovered``,
            ``skills``, ``a2a_agents``, ``errors``, ``skipped``.

        """

        summary: dict[str, Any] = {
            "mcp_servers": 0,
            "tools_discovered": 0,
            "skills": 0,
            "a2a_agents": 0,
            "errors": [],
            "skipped": 0,
        }

        for source in sources:
            source = source.strip()
            if not source:
                continue

            try:
                source_type = self._detect_toolkit_source_type(source)
                logger.info(
                    "[ECO-4.10] Processing source '%s' (detected: %s)",
                    source,
                    source_type,
                )

                if source_type == "a2a_url":
                    await self._ingest_a2a_from_url(source, agent_card_path, summary)
                elif source_type == "mcp_config":
                    await self._ingest_mcp_from_path(source, summary)
                elif source_type == "skill_directory":
                    self._ingest_skill_from_directory(source, summary)
                elif source_type == "remote_json":
                    await self._ingest_remote_json(source, agent_card_path, summary)
                else:
                    summary["errors"].append(f"Unknown source type for '{source}'")

            except Exception as e:
                logger.error("[ECO-4.10] Failed to ingest source '%s': %s", source, e)
                summary["errors"].append(f"{source}: {e}")

        logger.info(
            "[ECO-4.10] Toolkit ingestion complete: %d MCP servers, "
            "%d tools, %d skills, %d A2A agents, %d errors, %d skipped",
            summary["mcp_servers"],
            summary["tools_discovered"],
            summary["skills"],
            summary["a2a_agents"],
            len(summary["errors"]),
            summary["skipped"],
        )
        return summary

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_toolkit_source_type(source: str) -> str:
        """Auto-detect the type of a toolkit source.

        Returns one of: ``a2a_url``, ``mcp_config``, ``skill_directory``,
        ``remote_json``, ``unknown``.
        """
        from pathlib import Path

        # URLs
        if source.startswith(("http://", "https://")):
            if source.rstrip("/").endswith(".json"):
                return "remote_json"
            return "a2a_url"

        # Local paths
        p = Path(source)
        if p.is_file() and p.suffix.lower() == ".json":
            try:
                import json

                data = json.loads(p.read_text(encoding="utf-8"))
                if "mcpServers" in data:
                    return "mcp_config"
            except Exception:
                pass  # nosec B110
            return "unknown"

        if p.is_dir():
            skill_md = p / "SKILL.md"
            if skill_md.exists():
                return "skill_directory"
            # Check for mcp_config.json inside the directory
            mcp_config = p / "mcp_config.json"
            if mcp_config.exists():
                return "mcp_config"
            return "unknown"

        return "unknown"

    # ------------------------------------------------------------------
    # MCP config ingestion
    # ------------------------------------------------------------------

    async def _ingest_mcp_from_path(self, path: str, summary: dict[str, Any]) -> None:
        """Ingest MCP servers from a local config file path.

        Also handles a directory by looking for ``mcp_config.json`` inside it.
        """
        import json
        from pathlib import Path

        p = Path(path)
        if p.is_dir():
            p = p / "mcp_config.json"

        config_data = json.loads(p.read_text(encoding="utf-8"))
        await self._ingest_mcp_from_config(config_data, str(p), summary)

    async def _ingest_mcp_from_config(
        self,
        config_data: dict[str, Any],
        source_path: str,
        summary: dict[str, Any],
    ) -> None:
        """Ingest all servers from a parsed mcp_config.json payload."""
        import json
        from typing import cast

        engine_self = cast(Any, self)
        # Use MCPDiscoveryMixin methods (available via IntelligenceGraphEngine)
        server_entries = engine_self.parse_mcp_config(config_data)

        for entry in server_entries:
            server_name = entry["name"]

            # Freshness check — skip if KG cache is current
            if engine_self.check_server_freshness(server_name, entry["config_hash"]):
                logger.info(
                    "[ECO-4.10] Server '%s' is fresh in KG — skipping",
                    server_name,
                )
                summary["skipped"] += 1
                continue

            # Attempt live tool discovery
            live_tools = await engine_self.discover_mcp_tools(entry, timeout=30.0)

            if live_tools:
                # Live discovery succeeded — ingest real tools
                self.ingest_mcp_server(
                    name=server_name,
                    url=f"stdio://{entry['command']} {' '.join(entry['args'])}",
                    tools=live_tools,
                    resources={"source_config": source_path, "env": entry["env"]},
                )
                summary["tools_discovered"] += len(live_tools)
            else:
                # Fallback: synthesize tools from tool flags
                flag_tools = [
                    {
                        "name": f"{server_name}_{flag}",
                        "description": f"{flag} tools for {server_name}",
                        "tags": [flag],
                        "capabilities": [flag],
                    }
                    for flag in entry["tool_flags"]
                ]
                if flag_tools:
                    self.ingest_mcp_server(
                        name=server_name,
                        url=f"stdio://{entry['command']} {' '.join(entry['args'])}",
                        tools=flag_tools,
                        resources={"source_config": source_path, "env": entry["env"]},
                    )
                    summary["tools_discovered"] += len(flag_tools)
                else:
                    # No tools at all — still register the server node
                    self.ingest_mcp_server(
                        name=server_name,
                        url=f"stdio://{entry['command']} {' '.join(entry['args'])}",
                        tools=[],
                        resources={"source_config": source_path, "env": entry["env"]},
                    )

            # Update server node with config hash and disabled tools
            if self.backend:
                server_id = f"srv:{server_name}"
                ts = __import__("time").strftime(
                    "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()
                )
                self.backend.execute(
                    "MATCH (s:Server {id: $sid}) "
                    "SET s.config_hash = $hash, s.timestamp = $ts, "
                    "s.source_config = $src, s.tool_count = $tc, "
                    "s.command = $cmd, s.args = $args",
                    {
                        "sid": server_id,
                        "hash": entry["config_hash"],
                        "ts": ts,
                        "src": source_path,
                        "tc": len(live_tools)
                        if live_tools
                        else len(entry["tool_flags"]),
                        "cmd": entry["command"],
                        "args": json.dumps(entry["args"]),
                    },
                )

            summary["mcp_servers"] += 1

    # ------------------------------------------------------------------
    # Skill directory ingestion
    # ------------------------------------------------------------------

    def _ingest_skill_from_directory(
        self, dir_path: str, summary: dict[str, Any]
    ) -> None:
        """Ingest an agent skill from a directory containing SKILL.md."""
        from pathlib import Path

        skill_md = Path(dir_path) / "SKILL.md"
        if not skill_md.exists():
            summary["errors"].append(f"No SKILL.md found in {dir_path}")
            return

        content = skill_md.read_text(encoding="utf-8")
        frontmatter = self._parse_skill_frontmatter(content)

        if not frontmatter.get("name"):
            frontmatter["name"] = Path(dir_path).name

        self.ingest_agent_skill(
            skill_file_path=str(skill_md),
            frontmatter=frontmatter,
            content=content,
        )
        summary["skills"] += 1

    @staticmethod
    def _parse_skill_frontmatter(content: str) -> dict[str, Any]:
        """Parse YAML frontmatter from a SKILL.md file.

        Expects the standard format::

            ---
            name: my-skill
            description: Does things
            ---
            # Skill instructions...

        """
        import re

        frontmatter: dict[str, Any] = {}
        match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if not match:
            return frontmatter

        fm_text = match.group(1)
        for line in fm_text.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip().strip("\"'")
                if key:
                    frontmatter[key] = value

        return frontmatter

    # ------------------------------------------------------------------
    # A2A agent card ingestion
    # ------------------------------------------------------------------

    async def _ingest_a2a_from_url(
        self,
        base_url: str,
        agent_card_path: str,
        summary: dict[str, Any],
    ) -> None:
        """Fetch an A2A agent card from a URL and ingest it.

        Tries the primary ``agent_card_path`` first, then falls back to
        ``/agent-card.json`` if the primary path fails.
        """
        card = await self._fetch_a2a_card(base_url, agent_card_path)

        # Fallback to secondary standard path
        if card is None and agent_card_path != "/agent-card.json":
            logger.info("[ECO-4.10] Primary A2A path failed, trying /agent-card.json")
            card = await self._fetch_a2a_card(base_url, "/agent-card.json")

        if card is None:
            summary["errors"].append(f"Failed to fetch A2A agent card from {base_url}")
            return

        self.ingest_a2a_agent_card(url=base_url, card=card)
        summary["a2a_agents"] += 1

    @staticmethod
    async def _fetch_a2a_card(base_url: str, path: str) -> dict[str, Any] | None:
        """Fetch an A2A agent card JSON from a URL.

        Args:
            base_url: The base URL of the A2A agent (e.g., ``http://agent.local``).
            path: The well-known path (e.g., ``/.well-known/agent.json``).

        Returns:
            Parsed JSON dict, or None if the fetch fails.

        """
        import httpx

        url = base_url.rstrip("/") + path
        import os

        verify_ssl = os.environ.get("AGENTS_INSECURE_SSL", "0") != "1"
        try:
            async with httpx.AsyncClient(timeout=15.0, verify=verify_ssl) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.debug("A2A card fetch from '%s' failed: %s", url, e)
            return None

    # ------------------------------------------------------------------
    # Remote JSON handling
    # ------------------------------------------------------------------

    async def _ingest_remote_json(
        self,
        url: str,
        agent_card_path: str,
        summary: dict[str, Any],
    ) -> None:
        """Fetch a remote JSON file and determine if it's an MCP config or A2A card."""
        import os

        import httpx

        verify_ssl = os.environ.get("AGENTS_INSECURE_SSL", "0") != "1"
        try:
            async with httpx.AsyncClient(timeout=15.0, verify=verify_ssl) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            summary["errors"].append(f"Failed to fetch remote JSON from {url}: {e}")
            return

        if "mcpServers" in data:
            await self._ingest_mcp_from_config(data, url, summary)
        elif "name" in data and ("capabilities" in data or "skills" in data):
            # Looks like an A2A agent card
            self.ingest_a2a_agent_card(url=url, card=data)
            summary["a2a_agents"] += 1
        else:
            summary["errors"].append(
                f"Remote JSON from {url} does not match MCP config or A2A card format"
            )
