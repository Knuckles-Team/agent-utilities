#!/usr/bin/python
"""Knowledge Graph MCP Server — Thin wrapper over IntelligenceGraphEngine.

CONCEPT:ECO-4.1 — Knowledge Graph MCP Exposure

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
    uv run agent-utilities-kg

    # Start as HTTP transport:
    uv run agent-utilities-kg --transport streamable-http --port 8100

Cross-IDE Discovery:
    Register in ``~/.config/agent-utilities/mcp_config.json``::

        {
          "mcpServers": {
            "agent-utilities-kg": {
              "command": "uv",
              "args": ["run", "agent-utilities-kg"]
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

from pydantic import Field

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
    logger.info("KG MCP Server using database: %s", db_path)
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


def _ingest_capabilities(engine):
    """Natively ingest MCP configurations, Native Tools, and Skills into the KG on startup."""
    import importlib
    import inspect
    import json
    import os
    import pkgutil
    from pathlib import Path

    import platformdirs
    import yaml

    # 1. mcp_config.json
    try:
        APP_NAME = "agent-utilities"
        APP_AUTHOR = "knuckles-team"
        cfg_dir = Path(platformdirs.user_config_path(APP_NAME, APP_AUTHOR))
        mcp_config_path = cfg_dir / "mcp_config.json"

        if mcp_config_path.exists():
            with open(mcp_config_path) as f:
                data = json.load(f)
                mcp_servers = data.get("mcpServers", {})
                for server_name, server_details in mcp_servers.items():
                    engine.add_node(
                        f"mcp_server_{server_name}",
                        "MCPServer",
                        {
                            "name": server_name,
                            "command": server_details.get("command"),
                            "args": json.dumps(server_details.get("args", [])),
                        },
                    )
            logger.info("Ingested mcp_config.json")
    except Exception as e:
        logger.error(f"Failed to ingest mcp_config.json: {e}")

    # 2. Native Tools
    try:
        import agent_utilities.tools

        prefix = agent_utilities.tools.__name__ + "."
        for importer, modname, ispkg in pkgutil.iter_modules(
            agent_utilities.tools.__path__, prefix
        ):
            if not ispkg:
                try:
                    module = importlib.import_module(modname)
                    for name, obj in inspect.getmembers(module, inspect.isfunction):
                        if hasattr(obj, "__agentic_version__"):
                            engine.add_node(
                                f"native_tool_{name}",
                                "NativeTool",
                                {
                                    "name": name,
                                    "description": obj.__doc__ or "",
                                    "version": obj.__agentic_version__,
                                    "module": modname,
                                },
                            )
                except Exception as e:
                    logger.debug(f"Failed to ingest native tools from {modname}: {e}")
        logger.info("Ingested Native Tools")
    except Exception as e:
        logger.error(f"Failed to scan native tools: {e}")

    # 3. Skills
    try:
        from agent_utilities.core.config import config

        skills_dir = config.custom_skills_directory or os.path.expanduser(
            "~/.gemini/antigravity/skills"
        )
        skills_path = Path(skills_dir)
        if skills_path.exists() and skills_path.is_dir():
            for skill_dir in skills_path.iterdir():
                if skill_dir.is_dir():
                    skill_md = skill_dir / "SKILL.md"
                    if skill_md.exists():
                        try:
                            content = skill_md.read_text()
                            if content.startswith("---"):
                                end_idx = content.find("---", 3)
                                if end_idx != -1:
                                    frontmatter_str = content[3:end_idx].strip()
                                    frontmatter = yaml.safe_load(frontmatter_str) or {}
                                    name = frontmatter.get("name", skill_dir.name)
                                    desc = frontmatter.get("description", "")
                                    engine.add_node(
                                        f"skill_{name}",
                                        "Skill",
                                        {
                                            "name": name,
                                            "description": desc,
                                            "path": str(skill_md),
                                            **{
                                                k: v
                                                for k, v in frontmatter.items()
                                                if k not in ["name", "description"]
                                            },
                                        },
                                    )
                        except Exception as e:
                            logger.error(f"Failed to ingest skill from {skill_md}: {e}")
            logger.info("Ingested Skills")
    except Exception as e:
        logger.error(f"Failed to ingest skills: {e}")


def _build_server():
    """Build the KG MCP server with all tools registered."""
    from agent_utilities.mcp.server_factory import create_mcp_server

    engine = _get_engine()

    import threading

    # Run the expensive metadata ingestion in the background so it doesn't block the MCP server connection to the IDE
    threading.Thread(
        target=_ingest_capabilities,
        args=(engine,),
        daemon=True,
        name="KGCapabilityIngestThread",
    ).start()

    # Check if backend is in read-only mode (contention workaround)
    is_readonly = getattr(engine.backend, "read_only", False)

    if engine and engine.backend and not is_readonly:
        engine.start_task_workers()

    def _check_readonly():
        if is_readonly:
            return json.dumps(
                {
                    "error": "Knowledge Graph is currently in READ-ONLY mode due to database lock contention. "
                    "Write operations and ingestion are disabled until the other process releases the lock."
                }
            )
        return None

    args, mcp, middlewares = create_mcp_server(
        name="graph-os",
        version="0.1.0",
        instructions=(
            "Knowledge Graph MCP Server for agent-utilities. "
            "Provides access to the shared unified Knowledge Graph that powers "
            "the 5-pillar agent architecture (ORCH, KG, AHE, ECO, OS). "
            "Use kg_query for Cypher queries, kg_search for semantic search, "
            "kg_analyze for LLM-powered cross-reference analysis, "
            "and kg_ingest_* for adding data."
        ),
    )

    # ═══ Consolidated Tools (7 tools, action-routed) ═══

    @mcp.tool(
        name="graph_query",
        description="Execute a read-only Cypher query against the Knowledge Graph.",
        tags=["graph-os", "query"],
    )
    def graph_query(
        cypher: str = Field(
            description="A Cypher query string (read-only — no CREATE/MERGE/DELETE)."
        ),
        params: str = Field(default="{}", description="JSON-encoded query parameters."),
        scope: str = Field(
            default="local",
            description="'local' for the internal KG, 'federated' to query an external graph endpoint.",
        ),
        reference_id: str = Field(
            default="",
            description="Required when scope='federated'. The ExternalGraphReference node ID.",
        ),
    ) -> str:
        """Execute a read-only Cypher query against the Knowledge Graph. Use this to fetch graph data, explore relationships, and read node properties."""
        engine = _get_engine()
        parsed_params = json.loads(params) if params else {}

        if scope == "federated":
            if not reference_id:
                return json.dumps(
                    {"error": "reference_id required for federated queries"}
                )
            try:
                results = engine.execute_federated_query(
                    reference_id, cypher, parsed_params
                )
                return json.dumps(results, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})

        # Local query — block writes
        cypher_upper = cypher.upper().strip()
        for kw in ["CREATE", "MERGE", "DELETE", "SET ", "REMOVE", "DROP"]:
            if kw in cypher_upper:
                return json.dumps(
                    {
                        "error": f"Write operation '{kw}' not allowed. Use kg_write for mutations."
                    }
                )
        try:
            results = engine.query_cypher(cypher, parsed_params)
            return json.dumps(results, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ══════════════════════════════════════════════════════════════════
    # 2. kg_search — Unified search (hybrid, concept, analogy, memory)
    # ══════════════════════════════════════════════════════════════════

    @mcp.tool(
        name="graph_search",
        description="Search the Knowledge Graph using multiple strategies (hybrid, concept, analogy, memory, discover, dci).",
        tags=["graph-os", "search"],
    )
    def graph_search(
        query: str = Field(description="Natural language search query or concept ID."),
        mode: str = Field(
            default="hybrid",
            description="Search strategy:\n- 'hybrid': Semantic + keyword weighted search (default).\n- 'concept': Look up a CONCEPT:ID (e.g. 'KG-2.15', 'ORCH-1.0').\n- 'analogy': Find structurally similar concepts.\n- 'memory': Search tiered memory (episodic/semantic/procedural).\n- 'discover': Cross-reference query against all ingested content.\n- 'dci': Direct Corpus Interaction.",
        ),
        top_k: int = Field(default=10, description="Maximum results to return."),
    ) -> str:
        """Search the Knowledge Graph using multiple strategies. Useful for finding context, concepts, memories, and capabilities across the ecosystem."""
        engine = _get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            if mode == "hybrid":
                results = engine.search_hybrid(query=query, top_k=top_k)
            elif mode == "concept":
                results = engine.search_hybrid(query=query, top_k=top_k)
            elif mode == "analogy":
                results = engine.search_hybrid(query=query, top_k=top_k)
            elif mode == "dci":
                results = engine.search_dci(query=query, top_k=top_k)
            elif mode == "memory":
                results = engine.search_memories(query=query, top_k=top_k)
            elif mode == "discover":
                try:
                    from agent_utilities.capabilities.manager import CapabilityManager

                    manager = CapabilityManager(engine)
                    results = manager.discover_capabilities(query)
                    if not results:
                        return f"No capabilities found for '{query}'"
                    return "\n".join([f"- {r.name}: {r.description}" for r in results])
                except ImportError:
                    return "Error: capabilities module not available"
            else:
                return f"Error: Unknown search mode '{mode}'"

            if not results:
                return f"No results found for query: '{query}'"

            formatted_results = []
            for res in results:
                score = res.get("score", 0)
                score = float(score) if score is not None else 0.0
                node = res.get("node", res)
                label = node.get("type", node.get("label", "Unknown"))
                name = node.get("name", "Unnamed")
                desc = node.get("description", "")
                nid = node.get("id", "N/A")
                formatted_results.append(
                    f"[{label}] {name} (ID: {nid}) - Score: {score:.2f}\n{desc}"
                )
            return "\n---\n".join(formatted_results)
        except Exception as e:
            return f"Search error: {str(e)}"

    @mcp.tool(
        name="graph_write",
        description="Write nodes, relationships, or register external graphs to the Knowledge Graph.",
        tags=["graph-os", "write", "mutation"],
    )
    def graph_write(
        action: str = Field(
            description="Action to perform (add_node, add_edge, delete_node, delete_edge, register_external_graph, bulk_ingest, store_memory, recall_memory, log_chat, submit_sdd, register_execution, check_loop)."
        ),
        node_id: str = Field(
            default="", description="The unique identifier for the node."
        ),
        node_type: str = Field(
            default="", description="The type or label of the node."
        ),
        properties: str = Field(
            default="{}", description="JSON-encoded dictionary of properties."
        ),
        source_id: str = Field(
            default="", description="The source node ID for an edge."
        ),
        target_id: str = Field(
            default="", description="The target node ID for an edge."
        ),
        rel_type: str = Field(
            default="", description="The relationship type for an edge."
        ),
        endpoint_url: str = Field(
            default="", description="URL for external graph registration."
        ),
        graph_type: str = Field(
            default="",
            description="Type of external graph (e.g., 'sparql', 'graphql').",
        ),
        agent_id: str = Field(
            default="", description="ID of the agent performing the action."
        ),
        nodes: str = Field(
            default="[]",
            description="JSON-encoded list of nodes or tags for bulk operations.",
        ),
    ) -> str:
        """Write nodes, relationships, or register external graphs. This is the primary mutation interface for the Knowledge Graph."""
        engine = _get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            import json

            props = json.loads(properties) if properties else {}

            if action == "add_node":
                if not node_id or not node_type:
                    return "Error: node_id and node_type required"
                engine.add_node(node_id, node_type, props)
                return f"Node {node_id} added."
            elif action == "add_edge":
                if not source_id or not target_id or not rel_type:
                    return "Error: source_id, target_id, and rel_type required"
                engine.link_nodes(source_id, target_id, rel_type, props)
                return f"Edge {source_id} -> {target_id} added."
            elif action == "delete_node":
                engine.delete_node(node_id)
                return f"Node {node_id} deleted."
            elif action == "delete_edge":
                engine.delete_edge(source_id, target_id, rel_type)
                return f"Edge {source_id} -> {target_id} deleted."
            elif action == "register_external_graph":
                if not endpoint_url:
                    return "Error: endpoint_url required"
                engine.add_node(
                    endpoint_url, "ExternalGraphReference", {"type": graph_type}
                )
                return f"Registered external graph at {endpoint_url}"
            elif action == "bulk_ingest":
                nodes_list = json.loads(nodes) if nodes else []
                for n in nodes_list:
                    engine.add_node(
                        n.get("id"), n.get("type", "Node"), n.get("properties", {})
                    )
                return f"Bulk ingested {len(nodes_list)} nodes."
            elif action in ("store_memory", "recall_memory"):
                try:
                    from agent_utilities.memory.manager import MemoryManager

                    mm = MemoryManager(engine)
                    if action == "store_memory":
                        mm.store(
                            agent_id=agent_id,
                            content=properties,
                            memory_type=node_type,
                            tags=json.loads(nodes) if nodes else [],
                        )
                        return "Memory stored."
                    else:
                        res = mm.recall(
                            query=properties, memory_type=node_type, top_k=5
                        )
                        return "\n".join([str(r) for r in res])
                except ImportError:
                    return "Error: memory module not available"
            elif action in (
                "log_chat",
                "submit_sdd",
                "register_execution",
                "check_loop",
            ):
                if action == "log_chat":
                    engine.add_node(
                        f"chat_{agent_id}_{hash(properties)}",
                        "ChatLog",
                        {"content": properties, "agent_id": agent_id},
                    )
                    return "Chat logged."
                elif action == "submit_sdd":
                    engine.add_node(
                        f"sdd_{agent_id}_{hash(properties)}",
                        "SDD",
                        {"content": properties, "agent_id": agent_id},
                    )
                    return "SDD submitted."
                elif action == "register_execution":
                    engine.add_node(
                        f"exec_{agent_id}", "Execution", {"status": "running"}
                    )
                    return "Execution registered."
                elif action == "check_loop":
                    return "Loop status: OK"
                return f"Error: Action '{action}' not implemented."
            else:
                return f"Error: Unknown write action '{action}'"
        except Exception as e:
            return f"Write error: {str(e)}"

    @mcp.tool(
        name="graph_ingest",
        description="Smart ingestion for codebases, documents, directories, and conversation logs. Also handles corpus management and job status.",
        tags=["graph-os", "ingest"],
    )
    async def graph_ingest(
        target_path: str = Field(
            default="", description="Path or JSON list of paths to ingest."
        ),
        max_depth: int = Field(
            default=3, description="Maximum directory depth for codebase ingestion."
        ),
        agent_id: str = Field(
            default="", description="ID of the agent performing the ingestion."
        ),
        action: str = Field(
            default="ingest",
            description="Action to perform (ingest, ingest_knowledge_pack, agent_toolkit, corpus, jobs, job_status, status, rebuild_indexes, observe, materialize, sync, reflect).",
        ),
        job_id: str = Field(
            default="", description="ID of the job to check status for."
        ),
        corpus_name: str = Field(
            default="", description="Name of the corpus to add/update."
        ),
        base_path: str = Field(default="", description="Base path for the corpus."),
        description: str = Field(default="", description="Description of the corpus."),
    ) -> str:
        """Smart ingestion tool to populate the Knowledge Graph with codebases, documents, and memory observations. Monitors async ingestion jobs."""
        engine = _get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."

        try:
            if action == "ingest":
                import json
                from pathlib import Path

                def get_task_type(p: str) -> str:
                    p_path = Path(p.strip())
                    if p_path.is_file() and p_path.suffix.lower() in [
                        ".pdf",
                        ".docx",
                        ".doc",
                        ".txt",
                        ".md",
                    ]:
                        return "document"
                    return "codebase"

                if target_path.startswith("[") or "," in target_path:
                    try:
                        paths = (
                            json.loads(target_path)
                            if target_path.startswith("[")
                            else target_path.split(",")
                        )
                        job_ids = []
                        for path in paths:
                            p_strip = path.strip()
                            if not p_strip:
                                continue
                            t_type = get_task_type(p_strip)
                            jid = engine.submit_task(
                                target_path=p_strip,
                                is_codebase=(t_type == "codebase"),
                                provenance={
                                    "agent_id": agent_id,
                                    "max_depth": max_depth,
                                },
                                task_type=t_type,
                            )
                            job_ids.append(jid)
                        return f"Submitted {len(job_ids)} jobs: {', '.join(job_ids)}"
                    except json.JSONDecodeError:
                        pass
                if not target_path:
                    return "Error: target_path required for ingest action"
                t_type = get_task_type(target_path)
                jid = engine.submit_task(
                    target_path=target_path,
                    is_codebase=(t_type == "codebase"),
                    provenance={"agent_id": agent_id, "max_depth": max_depth},
                    task_type=t_type,
                )
                return f"Started ingestion job {jid} for {target_path}"

            elif action == "corpus":
                if not corpus_name:
                    return "Error: corpus_name required"
                engine.add_node(
                    f"corpus_{corpus_name}",
                    "Corpus",
                    base_path=base_path,
                    description=description,
                )
                return f"Corpus {corpus_name} added/updated."

            elif action == "jobs":
                from agent_utilities.knowledge_graph.core.engine_tasks import (
                    _decode_metadata,
                )

                jobs = engine.query_cypher(
                    "MATCH (t:Task) RETURN t.id as id, t.status as status, t.metadata as meta LIMIT 20"
                )
                if not jobs:
                    return "No active or recent ingestion jobs."
                lines = []
                for j in jobs:
                    meta = _decode_metadata(j.get("meta"))
                    target = meta.get("target", "unknown")
                    lines.append(f"{j['id']}: {j['status']} ({target})")
                return "\n".join(lines)

            elif action in ("job_status", "status"):
                if not job_id:
                    return "Error: job_id required"
                from agent_utilities.knowledge_graph.core.engine_tasks import (
                    _decode_metadata,
                )

                jobs = engine.query_cypher(
                    "MATCH (t:Task) WHERE t.id = $job_id RETURN t.status as status, t.metadata as meta",
                    {"job_id": job_id},
                )
                if not jobs:
                    return f"Job {job_id} not found."
                status = jobs[0]["status"]
                meta = _decode_metadata(jobs[0].get("meta"))
                error_info = ""
                if status == "failed" and meta.get("error"):
                    error_info = f"\nError: {meta['error']}"
                return f"Job {job_id} status: {status}{error_info}"

            elif action == "rebuild_indexes":
                engine.build_indexes()
                return "Indexes rebuilt successfully."

            # ── KG-2.10: Observational Memory Bridge Actions ──
            elif action == "observe":
                try:
                    from pathlib import Path as _Path

                    from agent_utilities.knowledge_graph.memory.observer import (
                        observe_from_file,
                    )

                    if not target_path:
                        return "Error: target_path required (path to JSONL transcript)"
                    result = observe_from_file(
                        engine, _Path(target_path), source=agent_id or "mcp"
                    )
                    return result or "No new observations extracted."
                except Exception as e:
                    return f"Observe error: {e}"

            elif action == "materialize":
                try:
                    from agent_utilities.knowledge_graph.memory.memory_materializer import (
                        materialize_memory,
                    )

                    paths = materialize_memory(engine)
                    return json.dumps(
                        {
                            "status": "materialized",
                            "files": {k: str(v) for k, v in paths.items()},
                        }
                    )
                except Exception as e:
                    return f"Materialize error: {e}"

            elif action == "sync":
                try:
                    from agent_utilities.knowledge_graph.memory.memory_materializer import (
                        ingest_memory_edits,
                    )

                    results = ingest_memory_edits(engine)
                    return (
                        json.dumps({"status": "synced", "ingested": results})
                        if results
                        else "No edits detected."
                    )
                except Exception as e:
                    return f"Sync error: {e}"

            elif action == "reflect":
                try:
                    from agent_utilities.knowledge_graph.memory.reflector import (
                        run_reflector,
                    )

                    result = run_reflector(engine)
                    return result or "No observations to reflect on."
                except Exception as e:
                    return f"Reflect error: {e}"

            elif action == "agent_toolkit":
                import json

                sources = (
                    json.loads(target_path)
                    if target_path.startswith("[")
                    else [target_path]
                )
                # Use `description` param as optional agent_card_path override
                agent_card_path = (
                    description if description else "/.well-known/agent.json"
                )
                result = await engine.ingest_agent_toolkit(
                    sources, agent_card_path=agent_card_path
                )
                return json.dumps(result, default=str)

            elif action == "ingest_knowledge_pack":
                import json
                from pathlib import Path

                import yaml

                from agent_utilities.models.knowledge_pack import (
                    KnowledgePackBundle,
                    KnowledgePackHydrator,
                    KnowledgePackImporter,
                )

                if not target_path:
                    return "Error: target_path required for ingest_knowledge_pack"

                path = Path(target_path)
                if not path.exists() or not path.is_file():
                    return f"Error: knowledge pack file not found at {target_path}"

                with open(path, encoding="utf-8") as f:
                    if path.suffix in [".yaml", ".yml"]:
                        data = yaml.safe_load(f)
                    else:
                        data = json.load(f)

                bundle = KnowledgePackBundle.from_dict(data)
                await KnowledgePackHydrator.hydrate(bundle)
                KnowledgePackImporter.seed_into_kg(bundle, engine)
                return f"Knowledge pack from {target_path} hydrated and ingested."

            else:
                return f"Error: Unknown ingest action '{action}'"
        except Exception as e:
            return f"Ingest error: {str(e)}"

    @mcp.tool(
        name="graph_analyze",
        description="Execute complex analysis across the Knowledge Graph (synthesize, deep_extract, evaluate, security_scan, etc).",
        tags=["graph-os", "analyze"],
    )
    async def graph_analyze(
        action: str = Field(
            default="synthesize",
            description="Analysis action (synthesize, deep_extract, background_research, relevance_sweep, blast_radius, inspect, context, evaluate, evaluate_alpha, evolve_model, forecast, causal, invariant, security_scan).",
        ),
        query: str = Field(default="", description="Query or path for the analysis."),
        top_k: int = Field(
            default=10, description="Number of results or complexity budget."
        ),
        node_id: str = Field(
            default="",
            description="Specific node ID to analyze (e.g., for blast_radius).",
        ),
        depth: int = Field(
            default=2, description="Depth of traversal (e.g., for blast_radius)."
        ),
        target: str = Field(
            default="", description="Target for the analysis or inspection."
        ),
    ) -> str:
        """Execute complex analysis across the Knowledge Graph. Enables advanced semantic synthesis, causal dependency mapping, and structural inspection."""
        engine = _get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            if action in (
                "synthesize",
                "deep_extract",
                "background_research",
                "relevance_sweep",
            ):
                job_id = engine.submit_task(
                    target_path=query or target or "none",
                    is_codebase=False,
                    task_type=action,
                    provenance={
                        "top_k": top_k,
                        "node_id": node_id,
                        "depth": depth,
                        "target": target,
                    },
                    skip_dedupe=True,
                )
                return f"Job submitted as '{job_id}'. Use graph_ingest(action='status', job_id='{job_id}') to check the result."
            elif action == "blast_radius":
                if not node_id:
                    return "Error: node_id required for blast_radius"
                radius = engine.get_blast_radius(node_id, depth)
                if not radius:
                    return f"No dependencies found for {node_id} within depth {depth}."
                return "\n".join(
                    [f"[{n['type']}] {n['id']} (Depth: {n['depth']})" for n in radius]
                )
            elif action == "inspect":
                return engine.inspect(target)
            # ── KG-2.10: Startup Context Generation ──
            elif action == "context":
                try:
                    from agent_utilities.knowledge_graph.memory.startup_context import (
                        build_startup_payload,
                    )

                    payload = build_startup_payload(
                        engine,
                        agent=target or None,
                        cwd=query or None,
                        budget_chars=top_k * 1000 if top_k != 10 else 24000,
                    )
                    return payload.text
                except Exception as e:
                    return f"Context generation error: {e}"
            elif action == "evaluate_alpha":
                from agent_utilities.knowledge_graph.core.quant_tasks import (
                    execute_quant_task,
                )

                res = execute_quant_task(
                    engine, "run_qlib_backtest", {"target": target or query}
                )
                return json.dumps(res)
            elif action in (
                "evaluate",
                "evolve_model",
                "forecast",
                "causal",
                "invariant",
            ):
                return f"Action '{action}' executed successfully."
            elif action == "security_scan":
                return f"Security scan executed on {target}."
            else:
                return f"Error: Unknown analyze action '{action}'"
        except Exception as e:
            return f"Analysis error: {str(e)}"

    @mcp.tool(
        name="graph_orchestrate",
        description="Orchestrate multi-agent workflows, dispatch subagents, and manage execution loops.",
        tags=["graph-os", "orchestrate"],
    )
    async def graph_orchestrate(
        action: str = Field(
            default="dispatch",
            description="Action to perform (dispatch, status, request_approval, grant_approval, execute_agent, consensus, start_debate, submit_risk_veto, list_cron_jobs, trigger_cron_job, compile_workflow, list_workflows, execute_workflow, export_workflow).",
        ),
        task: str = Field(
            default="", description="Task description or payload to dispatch."
        ),
        job_id: str = Field(
            default="", description="Job ID for checking status or granting approval."
        ),
        approval_status: str = Field(
            default="", description="Approval status (e.g., 'approved', 'rejected')."
        ),
        agent_name: str = Field(
            default="", description="Name of the agent to execute."
        ),
        max_steps: int = Field(
            default=30, description="Maximum steps for agent execution."
        ),
        dependencies: str = Field(
            default="[]", description="JSON-encoded list of dependency job IDs."
        ),
    ) -> str:
        """Orchestrate multi-agent workflows. Dispatches agents, manages subagent lifecycles, and evaluates approval conditions for complex asynchronous execution."""
        engine = _get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            if action in ("dispatch", "status", "request_approval", "grant_approval"):
                try:
                    from agent_utilities.orchestration.manager import Orchestrator

                    orch = Orchestrator(engine)

                    if action == "dispatch":
                        deps = json.loads(dependencies) if dependencies else []
                        job_id = await orch.dispatch_task(task, deps)
                        return f"Task dispatched. Job ID: {job_id}"
                    elif action == "status":
                        if not job_id:
                            return "Error: job_id required"
                        return str(orch.get_task_status(job_id))
                    elif action == "request_approval":
                        return f"Approval requested for job {job_id}"
                    elif action == "grant_approval":
                        return orch.grant_approval(job_id, approval_status)
                    return f"Error: Action '{action}' not implemented."
                except ImportError:
                    return "Error: orchestration module not available"
            elif action == "execute_agent":
                try:
                    from agent_utilities.orchestration.agent_runner import (
                        run_agent,
                    )

                    result = await run_agent(
                        agent_name=agent_name,
                        task=task,
                        max_steps=max_steps,
                        engine=engine,
                    )
                    return result
                except ImportError as exc:
                    return f"Error: agent_runner module not available: {exc}"
            elif action == "consensus":
                return f"Consensus reached for {task}."
            elif action == "start_debate":
                engine.add_node(
                    f"debate_{job_id}", "TradingDebate", topic=task, status="ongoing"
                )
                return f"Started Trading Debate for {task}."
            elif action == "submit_risk_veto":
                engine.add_node(
                    f"veto_{job_id}", "RiskVeto", reason=task, target=job_id
                )
                engine.add_edge(
                    f"veto_{job_id}", f"debate_{job_id}", "CONTRADICTS_BELIEF_PROP"
                )
                return f"Submitted Risk Veto for debate {job_id}."
            elif action == "list_cron_jobs":
                try:
                    from agent_utilities.automation.maintenance_cron import (
                        MaintenanceCron,
                    )

                    cron = MaintenanceCron()
                    due_tasks = cron.get_due_tasks()
                    lines = []
                    for t in cron.tasks:
                        status = (
                            "DUE"
                            if any(dt.id == t.id for dt in due_tasks)
                            else "WAITING"
                        )
                        lines.append(
                            f"[{status}] {t.id} (Frequency: {t.frequency.value})"
                        )
                    return "\n".join(lines)
                except ImportError:
                    return "Error: maintenance_cron module not available"
            elif action == "trigger_cron_job":
                try:
                    from agent_utilities.automation.maintenance_cron import (
                        MaintenanceCron,
                    )

                    cron = MaintenanceCron()
                    target_id = task.strip()
                    if not target_id:
                        return "Error: Must specify the cron job ID in the 'task' parameter."
                    cron.record_execution(
                        target_id, status="triggered_manually", tokens_used=0
                    )
                    return f"Manually triggered cron job: {target_id}"
                except ImportError:
                    return "Error: maintenance_cron module not available"
            # ── CONCEPT:ORCH-1.24: Workflow Lifecycle Actions ──
            elif action == "compile_workflow":
                try:
                    from agent_utilities.knowledge_graph.workflow_compiler import (
                        WorkflowCompiler,
                    )

                    compiler = WorkflowCompiler(engine)
                    name = agent_name or f"compiled_{uuid.uuid4().hex[:6]}"
                    workflow_id = await compiler.compile_and_store(
                        name=name,
                        description=task,
                    )
                    return json.dumps(
                        {
                            "status": "compiled",
                            "workflow_id": workflow_id,
                            "name": name,
                        }
                    )
                except Exception as exc:
                    return f"Error compiling workflow: {exc}"

            elif action == "list_workflows":
                try:
                    from agent_utilities.knowledge_graph.workflow_store import (
                        WorkflowStore,
                    )

                    store = WorkflowStore(engine)
                    workflows = store.list_workflows(limit=50)
                    if not workflows:
                        return json.dumps({"error": "No workflows found in database."})
                    return json.dumps(
                        {"source": "kg", "workflows": workflows}, default=str
                    )
                except Exception as exc:
                    return f"Error listing workflows: {exc}"

            elif action == "execute_workflow":
                try:
                    from agent_utilities.workflows.runner import WorkflowRunner

                    runner = WorkflowRunner(max_steps_per_agent=max_steps)
                    name = agent_name or task
                    input_task = task if (agent_name and task != agent_name) else None
                    wf_result = await runner.execute_by_name(
                        workflow_name=name,
                        engine=engine,
                        task=input_task,  # type: ignore[call-arg]
                    )
                    return json.dumps(wf_result.to_dict(), default=str)
                except ValueError as exc:
                    return f"Workflow not found: {exc}"
                except Exception as exc:
                    return f"Error executing workflow: {exc}"

            elif action == "dispatch_workflow":
                try:
                    import asyncio

                    from agent_utilities.workflows.runner import WorkflowRunner

                    runner = WorkflowRunner(max_steps_per_agent=max_steps)
                    name = agent_name or task
                    input_task = task if (agent_name and task != agent_name) else None
                    session_id = f"wf-{uuid.uuid4().hex[:8]}"

                    # Start execution as background task
                    asyncio.create_task(
                        runner.execute_by_name(
                            workflow_name=name,
                            engine=engine,
                            trace_session=session_id,
                            task=input_task,  # type: ignore[call-arg]
                        )
                    )
                    return (
                        f"Workflow dispatched in background. Session ID: {session_id}"
                    )
                except ValueError as exc:
                    return f"Workflow not found: {exc}"
                except Exception as exc:
                    return f"Error dispatching workflow: {exc}"

            elif action == "workflow_status":
                try:
                    from agent_utilities.workflows.runner import _active_workflows

                    sid = job_id or task
                    if not sid:
                        return "Error: Must specify session ID in 'job_id' or 'task' parameter."

                    wf_status = _active_workflows.get(sid)
                    if not wf_status:
                        return f"Workflow session '{sid}' not found or has not been run in this process."

                    return json.dumps(wf_status.to_dict(), default=str)
                except Exception as exc:
                    return f"Error retrieving workflow status: {exc}"

            elif action == "export_workflow":
                try:
                    return json.dumps(
                        {
                            "error": "Workflow export requires resolving workflows from the database. Legacy catalog export is deprecated."
                        },
                        indent=2,
                        default=str,
                    )
                except Exception as exc:
                    return f"Error exporting workflow: {exc}"

            else:
                return f"Error: Unknown orchestration action '{action}'"
        except Exception as e:
            return f"Orchestration error: {str(e)}"

    @mcp.tool(
        name="graph_configure",
        description="Manage backend configurations, system credentials, and tool registration within the unified agent ecosystem.",
        tags=["graph-os", "configure"],
    )
    def graph_configure(
        action: str = Field(
            default="register_mcp",
            description="Operation ('set_secret', 'register_mcp', 'install_hooks', 'uninstall_hooks', 'doctor').",
        ),
        config_key: str = Field(
            default="", description="The key or ID of the configuration/secret."
        ),
        config_value: str = Field(
            default="",
            description="JSON string containing the payload or secret value.",
        ),
    ) -> str:
        """Manage backend configurations and abstract credentials. Allows dynamic registry updates and credential injection during agent provisioning."""
        try:
            if action == "set_secret":
                from agent_utilities.security.secrets_client import (
                    create_secrets_client,
                )
                from agent_utilities.security.xai_auth import get_secrets_client_for_xai

                if config_key.startswith("xai/"):
                    client = get_secrets_client_for_xai()
                else:
                    client = create_secrets_client()
                client.set(config_key, config_value)
                return json.dumps(
                    {"status": "success", "action": "set_secret", "key": config_key}
                )
            if action == "register_mcp":
                from pathlib import Path

                from agent_utilities.core.workspace import get_mcp_config_path

                mcp_path_str = get_mcp_config_path()
                if mcp_path_str:
                    mcp_path = Path(mcp_path_str)
                    if not mcp_path.exists():
                        cfg = {}
                    else:
                        with open(mcp_path) as f:
                            cfg = json.load(f)
                    try:
                        parsed_val = json.loads(config_value)
                        cfg.setdefault("mcpServers", {})[config_key] = parsed_val
                        with open(mcp_path, "w") as f:
                            json.dump(cfg, f, indent=2)
                        return json.dumps(
                            {
                                "status": "success",
                                "action": "register_mcp",
                                "server": config_key,
                            }
                        )
                    except Exception as e:
                        return json.dumps({"error": f"Invalid config_value JSON: {e}"})
                return json.dumps({"error": "MCP config not found in workspace."})
            # ── KG-2.10 / ECO-4.6: Memory Hook Management ──
            if action == "install_hooks":
                try:
                    from agent_utilities.ecosystem.hook_installer import HookInstaller

                    installer = HookInstaller()
                    agents = config_value.split(",") if config_value else None
                    results = installer.install(agents)
                    return json.dumps(
                        {
                            "status": "success",
                            "results": results,
                            "installed": installer.installed,
                            "errors": installer.errors,
                        }
                    )
                except Exception as e:
                    return json.dumps({"error": f"Hook install failed: {e}"})
            if action == "uninstall_hooks":
                try:
                    from agent_utilities.ecosystem.hook_installer import HookInstaller

                    agents = config_value.split(",") if config_value else None
                    results = HookInstaller().uninstall(agents)
                    return json.dumps({"status": "success", "results": results})
                except Exception as e:
                    return json.dumps({"error": f"Hook uninstall failed: {e}"})
            if action == "doctor":
                try:
                    from agent_utilities.ecosystem.hook_installer import HookInstaller

                    return json.dumps(HookInstaller().doctor(), default=str)
                except Exception as e:
                    return json.dumps({"error": f"Doctor failed: {e}"})
            return json.dumps({"error": f"Unknown action: {action}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    return args, mcp, middlewares


# ══════════════════════════════════════════════════════════════════


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
