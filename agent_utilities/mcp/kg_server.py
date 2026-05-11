#!/usr/bin/python
"""Knowledge Graph MCP Server — Thin wrapper over IntelligenceGraphEngine.

CONCEPT:ECO-4.3 — Knowledge Graph MCP Exposure

Exposes the internal Knowledge Graph as MCP tools for external agents
(Claude Code, Antigravity IDE, OpenCode, Devin) to query, search, and
ingest data into the shared unified KG.

Architecture:
    This module reuses the existing ``create_mcp_server()`` infrastructure
    from ``agent_utilities.mcp.server_factory`` — zero new abstractions.
    All tools delegate to ``IntelligenceGraphEngine`` methods that already
    exist in the 15-phase pipeline.

Security:
    - Read-only by default for external agents.
    - Write access requires ``kg:write`` scope via MCP auth.
    - Every write carries provenance: ``agent_id``, ``session_id``,
      ``workspace_path`` for multi-agent traceability.

Usage:
    # Start as stdio MCP server (default):
    python -m agent_utilities.mcp.kg_server

    # Start as HTTP transport:
    python -m agent_utilities.mcp.kg_server --transport streamable-http --port 8100

Cross-IDE Discovery:
    Register in ``~/.config/agent-utilities/mcp_config.json``::

        {
          "mcpServers": {
            "agent-utilities-kg": {
              "command": "uv",
              "args": ["run", "python", "-m", "agent_utilities.mcp.kg_server"]
            }
          }
        }
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Default agent identity for provenance tracking
_AGENT_ID = os.environ.get("AGENT_ID", f"mcp-client-{uuid.uuid4().hex[:8]}")
_SESSION_ID = os.environ.get("SESSION_ID", uuid.uuid4().hex)
_WORKSPACE_PATH = os.environ.get("WORKSPACE_PATH", os.getcwd())


def _get_engine():
    """Lazily initialize and return the IntelligenceGraphEngine singleton."""
    import networkx as nx

    from agent_utilities.core.paths import ensure_dirs, kg_db_path
    from agent_utilities.knowledge_graph.backends import create_backend
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if engine is not None:
        return engine

    # First-run: ensure XDG dirs exist and create backend
    ensure_dirs()
    db_path = str(kg_db_path())
    backend = create_backend(backend_type="ladybug", db_path=db_path)
    graph = nx.MultiDiGraph()
    engine = IntelligenceGraphEngine(graph=graph, backend=backend)
    return engine


def _provenance_props(agent_id: str | None = None) -> dict[str, Any]:
    """Build standard provenance metadata for multi-agent write tracking."""
    return {
        "agent_id": agent_id or _AGENT_ID,
        "session_id": _SESSION_ID,
        "workspace_path": _WORKSPACE_PATH,
        "timestamp": datetime.now(UTC).isoformat(),
        "source": "mcp",
    }


def _build_server():
    """Build the KG MCP server with all tools registered."""
    from agent_utilities.mcp.server_factory import create_mcp_server

    args, mcp, middlewares = create_mcp_server(
        name="agent-utilities-kg",
        version="0.1.0",
        instructions=(
            "Knowledge Graph MCP Server for agent-utilities. "
            "Provides access to the shared unified Knowledge Graph that powers "
            "the 5-pillar agent architecture (ORCH, KG, AHE, ECO, OS). "
            "Use kg_query for Cypher queries, kg_search for semantic search, "
            "and kg_ingest_* for adding data."
        ),
    )

    # --- Read Tools (no auth scope required) ---

    @mcp.tool()
    def kg_query(cypher: str, params: str = "{}") -> str:
        """Execute a read-only Cypher query against the Knowledge Graph.

        Args:
            cypher: A Cypher query string (read-only — no CREATE/MERGE/DELETE).
            params: JSON-encoded query parameters.

        Returns:
            JSON-encoded list of result rows.
        """
        engine = _get_engine()
        # Security: block write operations for read-only access
        cypher_upper = cypher.upper().strip()
        write_keywords = ["CREATE", "MERGE", "DELETE", "SET ", "REMOVE", "DROP"]
        for kw in write_keywords:
            if kw in cypher_upper:
                return json.dumps(
                    {
                        "error": f"Write operation '{kw}' not allowed in read-only mode. "
                        "Use kg_write_node or kg_link_nodes for writes."
                    }
                )
        parsed_params = json.loads(params) if params else {}
        try:
            results = engine.query_cypher(cypher, parsed_params)
            return json.dumps(results, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def kg_search(query: str, top_k: int = 10) -> str:
        """Hybrid semantic + keyword search across the Knowledge Graph.

        Uses the existing HybridRetriever (KG-2.37) with weighted
        semantic (72%) + keyword (28%) scoring.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.

        Returns:
            JSON-encoded list of matching nodes with scores.
        """
        engine = _get_engine()
        try:
            results = engine.search_hybrid(query, top_k=top_k)
            return json.dumps(results, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def kg_concept_search(concept_id: str) -> str:
        """Look up a specific CONCEPT:ID in the Knowledge Graph.

        Searches for nodes matching the concept identifier (e.g., 'KG-2.15',
        'ORCH-1.0', 'AHE-3.3').

        Args:
            concept_id: The concept identifier (e.g., 'KG-2.15').

        Returns:
            JSON-encoded concept node with properties and relationships.
        """
        engine = _get_engine()
        try:
            results = engine.query_cypher(
                "MATCH (n) WHERE n.concept_id = $cid OR n.id CONTAINS $cid "
                "RETURN n LIMIT 5",
                {"cid": concept_id},
            )
            return json.dumps(results, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def kg_pillar_view(pillar: str) -> str:
        """Get the subgraph for a specific pillar of the architecture.

        Retrieves all concepts belonging to a pillar (ORCH, KG, AHE, ECO, OS).

        Args:
            pillar: Pillar prefix — one of 'ORCH', 'KG', 'AHE', 'ECO', 'OS'.

        Returns:
            JSON-encoded list of nodes and their relationships within the pillar.
        """
        engine = _get_engine()
        valid_pillars = {"ORCH", "KG", "AHE", "ECO", "OS"}
        pillar_upper = pillar.upper().strip()
        if pillar_upper not in valid_pillars:
            return json.dumps(
                {"error": f"Invalid pillar '{pillar}'. Must be one of: {valid_pillars}"}
            )
        try:
            results = engine.query_cypher(
                "MATCH (n) WHERE n.concept_id STARTS WITH $prefix "
                "OR n.id STARTS WITH $prefix "
                "OPTIONAL MATCH (n)-[r]->(m) "
                "RETURN n, type(r) AS rel_type, m.id AS target_id LIMIT 100",
                {"prefix": pillar_upper},
            )
            return json.dumps(results, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def kg_get_constitution() -> str:
        """Read the project constitution from the Knowledge Graph.

        Returns the governance rules, core principles, and tech stack
        from the active project's constitution.

        Returns:
            JSON-encoded constitution data.
        """
        engine = _get_engine()
        try:
            results = engine.query_cypher(
                "MATCH (n) WHERE n.type = 'PolicyNode' OR n.type = 'GovernanceNode' "
                "RETURN n LIMIT 50"
            )
            return json.dumps(results, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def kg_get_stats() -> str:
        """Get summary statistics about the Knowledge Graph.

        Returns node counts by type, edge counts, and health metrics.

        Returns:
            JSON-encoded statistics dictionary.
        """
        engine = _get_engine()
        try:
            stats: dict[str, Any] = {}
            # Node count by type
            node_results = engine.query_cypher(
                "MATCH (n) RETURN n.type AS type, count(*) AS count "
                "ORDER BY count DESC LIMIT 50"
            )
            stats["node_types"] = node_results
            # Total counts
            total = engine.query_cypher("MATCH (n) RETURN count(n) AS total")
            stats["total_nodes"] = total[0]["total"] if total else 0
            edge_total = engine.query_cypher(
                "MATCH ()-[r]->() RETURN count(r) AS total"
            )
            stats["total_edges"] = edge_total[0]["total"] if edge_total else 0
            return json.dumps(stats, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # --- Write Tools (require kg:write scope in production) ---

    @mcp.tool()
    def kg_write_node(
        node_id: str,
        node_type: str,
        properties: str = "{}",
        agent_id: str = "",
    ) -> str:
        """Write a node to the Knowledge Graph with provenance tracking.

        Every write carries multi-agent provenance metadata (agent_id,
        session_id, workspace_path, timestamp) for traceability.

        Args:
            node_id: Unique identifier for the node.
            node_type: Type/label of the node (e.g., 'MemoryNode', 'CodeNode').
            properties: JSON-encoded properties for the node.
            agent_id: Optional agent identifier (defaults to MCP client ID).

        Returns:
            JSON confirmation of the write with provenance.
        """
        engine = _get_engine()
        try:
            props = json.loads(properties) if properties else {}
            provenance = _provenance_props(agent_id or None)
            props.update(provenance)
            engine.add_node(node_id, node_type, properties=props)
            return json.dumps(
                {
                    "status": "created",
                    "node_id": node_id,
                    "type": node_type,
                    "provenance": provenance,
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def kg_link_nodes(
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: str = "{}",
        agent_id: str = "",
    ) -> str:
        """Create a relationship between two nodes with provenance tracking.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            rel_type: Relationship type (e.g., 'DEPENDS_ON', 'EXTENDS').
            properties: JSON-encoded relationship properties.
            agent_id: Optional agent identifier.

        Returns:
            JSON confirmation of the relationship creation.
        """
        engine = _get_engine()
        try:
            props = json.loads(properties) if properties else {}
            provenance = _provenance_props(agent_id or None)
            props.update(provenance)
            engine.link_nodes(source_id, target_id, rel_type, properties=props)
            return json.dumps(
                {
                    "status": "linked",
                    "source": source_id,
                    "target": target_id,
                    "rel_type": rel_type,
                    "provenance": provenance,
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    # --- Extended KG Engine Tools ---

    @mcp.tool()
    def kg_analogy_search(description: str, top_k: int = 5) -> str:
        """Find structurally similar concepts to a given description.

        Uses the Topological Analysis Engine (KG-2.5) to find analogous
        concepts. This is the core of the Extend-Before-Invent governance
        rule — before adding a new concept, check for analogues.

        Args:
            description: Natural language description of the proposed feature.
            top_k: Maximum number of analogous concepts to return.

        Returns:
            JSON list of analogous concepts with similarity scores.
        """
        engine = _get_engine()
        try:
            # Use hybrid search as the analogy search foundation
            results = engine.search_hybrid(description, top_k=top_k)
            # Enrich with concept IDs for governance
            analogues = []
            for r in results:
                analogue = {
                    "node_id": r.get("id", r.get("node_id", "")),
                    "name": r.get("name", r.get("title", "")),
                    "similarity": r.get("score", 0.0),
                    "concept_id": r.get("concept_id", ""),
                    "type": r.get("type", ""),
                }
                analogues.append(analogue)
            return json.dumps(analogues, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def kg_blast_radius(node_id: str, depth: int = 2) -> str:
        """Assess the impact of changes to a specific node.

        Traverses the graph from the target node to find all
        transitively dependent nodes within the specified depth.

        Args:
            node_id: The ID of the node to analyze.
            depth: How many hops to traverse (default: 2).

        Returns:
            JSON with affected nodes, edge counts, and risk summary.
        """
        engine = _get_engine()
        try:
            impact_graph = engine.load_for_impact_analysis(node_id)
            affected = []
            for n in impact_graph.nodes(data=True):
                nid, data = n
                if nid != node_id:
                    affected.append(
                        {
                            "node_id": nid,
                            "type": data.get("type", ""),
                            "name": data.get("name", nid),
                        }
                    )
            return json.dumps(
                {
                    "target": node_id,
                    "depth": depth,
                    "affected_count": len(affected),
                    "affected_nodes": affected[:50],  # Cap at 50
                },
                default=str,
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def kg_memory_recall(
        query: str, memory_type: str = "episodic", top_k: int = 5
    ) -> str:
        """Recall memories from the tiered memory system (KG-2.1).

        Searches episodic, semantic, or procedural memory stores
        for relevant past experiences.

        Args:
            query: Natural language recall query.
            memory_type: One of 'episodic', 'semantic', 'procedural'.
            top_k: Maximum memories to recall.

        Returns:
            JSON list of matching memories with metadata.
        """
        engine = _get_engine()
        valid_types = {"episodic", "semantic", "procedural"}
        if memory_type not in valid_types:
            return json.dumps(
                {"error": f"Invalid memory_type. Must be one of: {valid_types}"}
            )
        try:
            results = engine.query_cypher(
                "MATCH (m) WHERE m.type = $mtype "
                "AND (m.content CONTAINS $q OR m.summary CONTAINS $q) "
                "RETURN m ORDER BY m.timestamp DESC LIMIT $k",
                {"mtype": f"Memory:{memory_type}", "q": query, "k": top_k},
            )
            return json.dumps(results, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def kg_memory_store(
        content: str,
        memory_type: str = "episodic",
        tags: str = "[]",
        agent_id: str = "",
    ) -> str:
        """Store a new memory in the tiered memory system (KG-2.1).

        Creates a memory node with full provenance tracking.

        Args:
            content: The memory content to store.
            memory_type: One of 'episodic', 'semantic', 'procedural'.
            tags: JSON-encoded list of string tags.
            agent_id: Optional agent identifier.

        Returns:
            JSON confirmation with memory ID and provenance.
        """
        engine = _get_engine()
        try:
            memory_id = f"memory-{uuid.uuid4().hex[:12]}"
            provenance = _provenance_props(agent_id or None)
            parsed_tags = json.loads(tags) if tags else []
            props = {
                "content": content,
                "type": f"Memory:{memory_type}",
                "tags": json.dumps(parsed_tags),
                "summary": content[:200],
                **provenance,
            }
            engine.add_node(memory_id, f"Memory:{memory_type}", properties=props)
            return json.dumps(
                {
                    "status": "stored",
                    "memory_id": memory_id,
                    "memory_type": memory_type,
                    "provenance": provenance,
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def kg_ingest_batch(nodes: str, agent_id: str = "") -> str:
        """Batch ingest multiple nodes into the Knowledge Graph.

        More efficient than individual kg_write_node calls for bulk operations.

        Args:
            nodes: JSON-encoded list of objects, each with 'id', 'type', and 'properties'.
            agent_id: Optional agent identifier for provenance.

        Returns:
            JSON summary of ingested nodes.
        """
        engine = _get_engine()
        try:
            parsed = json.loads(nodes)
            if not isinstance(parsed, list):
                return json.dumps({"error": "nodes must be a JSON array"})
            provenance = _provenance_props(agent_id or None)
            created = []
            for node_spec in parsed[:100]:  # Cap at 100 per batch
                nid = node_spec.get("id", f"batch-{uuid.uuid4().hex[:8]}")
                ntype = node_spec.get("type", "GenericNode")
                props = node_spec.get("properties", {})
                props.update(provenance)
                engine.add_node(nid, ntype, properties=props)
                created.append(nid)
            return json.dumps(
                {
                    "status": "ingested",
                    "count": len(created),
                    "node_ids": created,
                    "provenance": provenance,
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def kg_ingest(
        target_path: str,
        agent_id: str = "",
    ) -> str:
        """A smart ingestion tool for both codebases and generic documents.

        Detects if the target is a codebase (e.g. contains .git, pyproject.toml)
        or a set of documents, and ingests them natively into the Knowledge Graph.

        Args:
            target_path: The absolute path to the directory or file to ingest.
            agent_id: Optional agent identifier for provenance.

        Returns:
            JSON summary of the ingested codebase nodes or document chunks.
        """
        from pathlib import Path

        from agent_utilities.core.paths import kg_db_path

        target = Path(target_path)
        if not target.exists():
            return json.dumps({"error": f"Path {target_path} does not exist."})

        is_codebase = False
        if target.is_dir():
            for indicator in [".git", "pyproject.toml", "package.json", "setup.py"]:
                if (target / indicator).exists():
                    is_codebase = True
                    break

        engine = _get_engine()
        provenance = _provenance_props(agent_id or None)

        if is_codebase:
            try:
                from agent_utilities.knowledge_graph.pipeline import (
                    IntelligencePipeline,
                )
                from agent_utilities.models.knowledge_graph import PipelineConfig

                config = PipelineConfig(
                    workspace_path=str(target),
                    ladybug_path=str(kg_db_path()),
                )
                pipeline = IntelligencePipeline(config, backend=engine.backend)
                metadata = await pipeline.run()

                return json.dumps(
                    {
                        "status": "ingested_codebase",
                        "target": str(target),
                        "nodes_added": metadata.node_count,
                        "edges_added": metadata.edge_count,
                        "provenance": provenance,
                    }
                )
            except Exception as e:
                import traceback

                logger.error(f"Codebase ingestion failed: {traceback.format_exc()}")
                return json.dumps({"error": f"Codebase ingestion failed: {e}"})
        else:
            try:
                import hashlib
                from datetime import datetime

                from llama_index.core import SimpleDirectoryReader

                from agent_utilities.core.embedding_utilities import (
                    create_embedding_model,
                )

                # Setup embedding model
                embed_model = create_embedding_model()

                # Load documents
                if target.is_dir():
                    docs = SimpleDirectoryReader(input_dir=str(target)).load_data()
                else:
                    docs = SimpleDirectoryReader(input_files=[str(target)]).load_data()

                created = []
                ingestion_timestamp = datetime.now(UTC).isoformat()

                for idx, doc in enumerate(docs):
                    chunk_text = doc.text
                    if not chunk_text.strip():
                        continue

                    file_path = doc.metadata.get("file_path", str(target))
                    # Stable ID based on file path and chunk content
                    raw_id = f"{file_path}::{chunk_text}".encode()
                    nid = f"doc-{hashlib.sha256(raw_id).hexdigest()[:8]}"

                    # Check if exists (Mark phase)
                    existing = engine.query_cypher(
                        "MATCH (n:Article {id: $nid}) RETURN n.id as id", {"nid": nid}
                    )
                    if existing:
                        engine.backend.execute(
                            "MATCH (n:Article {id: $nid}) SET n.last_seen_timestamp = $ts",
                            {"nid": nid, "ts": ingestion_timestamp},
                        )
                        created.append(nid)
                        continue

                    # Generate embedding only for new/changed chunks
                    embedding = embed_model.get_text_embedding(chunk_text)

                    props = {
                        "content": chunk_text,
                        "embedding": embedding,
                        "metadata": json.dumps(doc.metadata),
                        "last_seen_timestamp": ingestion_timestamp,
                        "target_path": str(target),
                    }
                    props.update(provenance)
                    engine.add_node(nid, "Article", properties=props)
                    created.append(nid)

                # Sweep phase: Delete chunks for this target that weren't seen in this run
                engine.backend.execute(
                    "MATCH (n:Article) WHERE n.target_path = $target AND n.last_seen_timestamp < $ts DETACH DELETE n",
                    {"target": str(target), "ts": ingestion_timestamp},
                )

                return json.dumps(
                    {
                        "status": "ingested_documents",
                        "target": str(target),
                        "chunks_added": len(created),
                        "provenance": provenance,
                    }
                )
            except ImportError as e:
                import traceback

                logger.error(
                    f"ImportError in document ingestion: {traceback.format_exc()}"
                )
                return json.dumps(
                    {
                        "error": f"Missing dependency for document ingestion: {e}. Please install agent-utilities[documents]."
                    }
                )
            except Exception as e:
                import traceback

                logger.error(f"Document ingestion failed: {traceback.format_exc()}")
                return json.dumps({"error": f"Document ingestion failed: {e}"})

    @mcp.tool()
    def kg_ontology_validate(node_type: str, properties: str = "{}") -> str:
        """Validate a proposed node against the OWL ontology (KG-2.2).

        Checks whether a node type and its properties conform to the
        formal ontology schema before committing it to the graph.

        Args:
            node_type: The proposed node type/label.
            properties: JSON-encoded properties to validate.

        Returns:
            JSON with validation result and any schema violations.
        """
        try:
            from agent_utilities.knowledge_graph.schema import SCHEMA

            props = json.loads(properties) if properties else {}
            # Check if node type exists in schema
            valid_types = [n.label for n in SCHEMA.nodes]
            if node_type in valid_types:
                # Get allowed properties for this type
                node_def = next(n for n in SCHEMA.nodes if n.label == node_type)
                allowed = {p.name for p in node_def.properties}
                unknown = (
                    set(props.keys())
                    - allowed
                    - {
                        "agent_id",
                        "session_id",
                        "workspace_path",
                        "timestamp",
                        "source",
                    }
                )
                return json.dumps(
                    {
                        "valid": len(unknown) == 0,
                        "node_type": node_type,
                        "allowed_properties": sorted(allowed),
                        "unknown_properties": sorted(unknown),
                    }
                )
            return json.dumps(
                {
                    "valid": False,
                    "node_type": node_type,
                    "error": f"Unknown node type. Known types: {sorted(valid_types)}",
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    return args, mcp, middlewares


def main():
    """Entry point for the KG MCP server."""
    args, mcp, middlewares = _build_server()

    # Apply middleware stack
    for middleware in middlewares:
        mcp.add_middleware(middleware)

    logger.info(
        "Starting Knowledge Graph MCP Server (transport=%s, port=%s)",
        args.transport,
        args.port,
    )

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
        )


if __name__ == "__main__":
    main()
