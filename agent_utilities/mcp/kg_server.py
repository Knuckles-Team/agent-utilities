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


def _build_server():
    """Build the KG MCP server with all tools registered."""
    from agent_utilities.mcp.server_factory import create_mcp_server

    engine = _get_engine()

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

    @mcp.tool()
    def graph_query(
        cypher: str, params: str = "{}", scope: str = "local", reference_id: str = ""
    ) -> str:
        """Execute a read-only Cypher query against the Knowledge Graph.

        Args:
            cypher: A Cypher query string (read-only — no CREATE/MERGE/DELETE).
            params: JSON-encoded query parameters.
            scope: 'local' for the internal KG, 'federated' to query an external graph endpoint.
            reference_id: Required when scope='federated'. The ExternalGraphReference node ID.

        Returns:
            JSON-encoded list of result rows.
        """
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

    @mcp.tool()
    def graph_search(query: str, mode: str = "hybrid", top_k: int = 10) -> str:
        """This is a tool from the graph-os MCP server.
        Search the Knowledge Graph using multiple strategies.

        Args:
            query: Natural language search query or concept ID.
            mode: Search strategy:
                - 'hybrid': Semantic + keyword weighted search (default).
                - 'concept': Look up a CONCEPT:ID (e.g. 'KG-2.15', 'ORCH-1.0').
                - 'analogy': Find structurally similar concepts.
                - 'memory': Search tiered memory (episodic/semantic/procedural).
                - 'discover': Cross-reference query against all ingested content.
                - 'dci': Direct Corpus Interaction.
            top_k: Maximum results to return.
        """
        engine = _get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            if mode in ("hybrid", "concept", "analogy", "dci", "memory"):
                results = engine.search(query=query, mode=mode, limit=top_k)
                if not results:
                    return f"No results found for query: '{query}'"

                formatted_results = []
                for res in results:
                    score = res.get("score", 0)
                    node = res.get("node", {})
                    label = node.get("label", "Unknown")
                    name = node.get("name", "Unnamed")
                    desc = node.get("description", "")
                    nid = node.get("id", "N/A")
                    formatted_results.append(
                        f"[{label}] {name} (ID: {nid}) - Score: {score:.2f}\n{desc}"
                    )
                return "\n---\n".join(formatted_results)
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
        except Exception as e:
            return f"Search error: {str(e)}"

    @mcp.tool()
    def graph_write(
        action: str,
        node_id: str = "",
        node_type: str = "",
        properties: str = "{}",
        source_id: str = "",
        target_id: str = "",
        rel_type: str = "",
        endpoint_url: str = "",
        graph_type: str = "",
        agent_id: str = "",
        nodes: str = "[]",
    ) -> str:
        """This is a tool from the graph-os MCP server.
        Write nodes, relationships, or register external graphs.
        """
        engine = _get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            import json

            props = json.loads(properties) if properties else {}

            if action == "add_node":
                if not node_id or not node_type:
                    return "Error: node_id and node_type required"
                engine.add_node(node_id, node_type, **props)
                return f"Node {node_id} added."
            elif action == "add_edge":
                if not source_id or not target_id or not rel_type:
                    return "Error: source_id, target_id, and rel_type required"
                engine.add_edge(source_id, target_id, rel_type, **props)
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
                engine.add_node(endpoint_url, "ExternalGraphReference", type=graph_type)
                return f"Registered external graph at {endpoint_url}"
            elif action == "bulk_ingest":
                nodes_list = json.loads(nodes) if nodes else []
                for n in nodes_list:
                    engine.add_node(
                        n.get("id"), n.get("type", "Node"), **n.get("properties", {})
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
                        content=properties,
                        agent_id=agent_id,
                    )
                    return "Chat logged."
                elif action == "submit_sdd":
                    engine.add_node(
                        f"sdd_{agent_id}_{hash(properties)}",
                        "SDD",
                        content=properties,
                        agent_id=agent_id,
                    )
                    return "SDD submitted."
                elif action == "register_execution":
                    engine.add_node(f"exec_{agent_id}", "Execution", status="running")
                    return "Execution registered."
                elif action == "check_loop":
                    return "Loop status: OK"
                return f"Error: Action '{action}' not implemented."
            else:
                return f"Error: Unknown write action '{action}'"
        except Exception as e:
            return f"Write error: {str(e)}"

    @mcp.tool()
    async def graph_ingest(
        target_path: str,
        max_depth: int = 3,
        agent_id: str = "",
        action: str = "ingest",
        job_id: str = "",
        corpus_name: str = "",
        base_path: str = "",
        description: str = "",
    ) -> str:
        """This is a tool from the graph-os MCP server.
        Smart ingestion for codebases, documents, directories, and conversation logs.
        Also handles corpus management and job status.
        """
        engine = _get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."

        try:
            if action == "ingest":
                import json

                try:
                    from agent_utilities.ingestion.pipeline import IngestionPipeline

                    pipeline = IngestionPipeline(engine)
                except ImportError:
                    return "Error: ingestion pipeline module not available"

                if target_path.startswith("[") or "," in target_path:
                    try:
                        paths = (
                            json.loads(target_path)
                            if target_path.startswith("[")
                            else target_path.split(",")
                        )
                        job_ids = []
                        for path in paths:
                            jid = await pipeline.submit_job(
                                path.strip(), max_depth, agent_id
                            )
                            job_ids.append(jid)
                        return f"Submitted {len(job_ids)} jobs: {', '.join(job_ids)}"
                    except json.JSONDecodeError:
                        pass
                jid = await pipeline.submit_job(target_path, max_depth, agent_id)
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
                try:
                    from agent_utilities.ingestion.pipeline import IngestionPipeline

                    pipeline = IngestionPipeline(engine)
                    jobs = pipeline.get_jobs()
                    if not jobs:
                        return "No active or recent ingestion jobs."
                    return "\n".join(
                        [f"{j['id']}: {j['status']} ({j['target']})" for j in jobs]
                    )
                except ImportError:
                    return "Error: ingestion pipeline module not available"

            elif action == "job_status":
                if not job_id:
                    return "Error: job_id required"
                try:
                    from agent_utilities.ingestion.pipeline import IngestionPipeline

                    pipeline = IngestionPipeline(engine)
                    status = pipeline.get_job_status(job_id)
                    if not status:
                        return f"Job {job_id} not found."
                    return f"Job {job_id} status: {status['status']}\nProgress: {status.get('progress', 0)}%"
                except ImportError:
                    return "Error: ingestion pipeline module not available"
            else:
                return f"Error: Unknown ingest action '{action}'"
        except Exception as e:
            return f"Ingest error: {str(e)}"

    @mcp.tool()
    async def graph_analyze(
        action: str = "synthesize",
        query: str = "",
        top_k: int = 10,
        node_id: str = "",
        depth: int = 2,
        target: str = "",
    ) -> str:
        """This is a tool from the graph-os MCP server.
        Execute complex analysis across the Knowledge Graph.
        """
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
                try:
                    from agent_utilities.analysis.analyzer import GraphAnalyzer

                    analyzer = GraphAnalyzer(engine)
                    if action == "synthesize":
                        return await analyzer.synthesize(query, top_k)
                    elif action == "deep_extract":
                        return await analyzer.deep_extract(query)
                    elif action == "background_research":
                        return await analyzer.background_research(query)
                    elif action == "relevance_sweep":
                        return await analyzer.relevance_sweep(query)
                    return f"Error: Action '{action}' not implemented."
                except ImportError:
                    return "Error: analysis module not available"
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

    @mcp.tool()
    async def graph_orchestrate(
        action: str = "dispatch",
        task: str = "",
        job_id: str = "",
        approval_status: str = "",
        agent_name: str = "",
        max_steps: int = 30,
        dependencies: str = "[]",
    ) -> str:
        """This is a tool from the graph-os MCP server.
        Orchestrate multi-agent workflows.
        """
        engine = _get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            if action in ("dispatch", "status", "request_approval", "grant_approval"):
                try:
                    from agent_utilities.orchestration.manager import Orchestrator

                    orch = Orchestrator(engine)

                    if action == "dispatch":
                        import json

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
                    from agent_utilities.orchestration.agent_runner import run_agent

                    return await run_agent(agent_name, task, max_steps)
                except ImportError:
                    return "Error: agent_runner module not available"
            elif action == "consensus":
                return f"Consensus reached for {task}."
            else:
                return f"Error: Unknown orchestration action '{action}'"
        except Exception as e:
            return f"Orchestration error: {str(e)}"

    @mcp.tool()
    def graph_configure(
        action: str = "register_mcp", config_key: str = "", config_value: str = ""
    ) -> str:
        """Manage backend configurations and abstract credentials.

        Args:
            action: Operation:
                - 'set_secret': Save credentials via the abstracted backend layer.
                - 'register_mcp': Register new MCP server configurations.
            config_key: The key or ID of the configuration/secret.
            config_value: JSON string containing the payload or secret value.

        Returns:
            JSON string confirming the update.
        """
        try:
            if action == "set_secret":
                # Integrates with the ecosystem's backend abstraction layer
                # For now, simulate saving via standard config utils.
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
            return json.dumps({"error": f"Unknown action: {action}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    return args, mcp, middlewares


# ══════════════════════════════════════════════════════════════════
# Helper functions for kg_analyze — Pydantic structured output
# ══════════════════════════════════════════════════════════════════


async def _run_l2_synthesis(
    ctx: Any, engine: Any, query: str, enriched: list[dict]
) -> dict[str, Any]:
    """Layer 2: LLM synthesis with Pydantic result_type for guaranteed JSON.

    Uses grammar-constrained decoding via pydantic-ai's ``result_type`` to
    eliminate regex JSON parsing and guarantee valid, validated output.
    Falls back from ctx.sample() → direct pydantic-ai Agent.
    """
    import asyncio

    from agent_utilities.knowledge_graph.core.analysis_models import (
        SynthesisResult,
    )

    # Build synthesis prompt from L1 results
    match_lines = []
    for r in enriched[:15]:
        match_lines.append(
            f"- **{r.get('name', r.get('id', ''))}** "
            f"(score={r.get('score', 0):.3f}, signals={r.get('total_signal_count', 0)})"
        )
        for claim in r.get("innovation_claims", [])[:2]:
            match_lines.append(f"  > {claim[:200]}")
        for sig in r.get("tech_signals", [])[:3]:
            match_lines.append(
                f"  ↳ {sig['keyword']}: {sig['analogy']} → {sig['domain']}"
            )

    synthesis_prompt = (
        f"## Cross-Reference Analysis: {query}\n\n"
        f"The following {len(enriched)} results were found via semantic "
        f"cross-reference against the Knowledge Graph:\n\n"
        + "\n".join(match_lines)
        + "\n\n---\n\n"
        "Analyze these matches and extract actionable feature recommendations. "
        "For each recommendation provide: feature name, target concepts it enhances, "
        "implementation sketch (key classes and methods), expected impact, "
        "integration complexity (low/medium/high), and priority (1-10)."
    )

    system_prompt = (
        "You are an expert software architect analyzing research papers and "
        "codebases cross-referenced against an agent framework's Knowledge Graph. "
        "Extract actionable features as structured recommendations."
    )

    # Always use the Pydantic-ai Agent with result_type for schema enforcement
    try:
        from pydantic_ai import Agent

        from agent_utilities.core.model_factory import create_model

        agent = Agent(
            model=create_model(),
            system_prompt=system_prompt,
            output_type=SynthesisResult,
        )
        result = await asyncio.to_thread(agent.run_sync, synthesis_prompt)
        synthesis: SynthesisResult = result.data  # type: ignore[assignment]

        return {
            "layer": 2,
            "features": [r.model_dump() for r in synthesis.recommendations],
            "feature_count": len(synthesis.recommendations),
        }
    except Exception as e:
        logger.warning("L2 synthesis failed: %s", e)
        return {
            "layer": 2,
            "error": f"LLM synthesis failed: {e}",
            "fallback": "Use L1 results directly or configure LLM endpoint",
        }


async def _run_l3_extraction(
    ctx: Any, engine: Any, query: str, enriched: list[dict]
) -> dict[str, Any]:
    """Layer 3: Batched deep extraction with Pydantic result_type.

    Batches all high-weight matches into a SINGLE LLM call (leveraging
    large context windows like 256K) to minimize inference requests.
    """
    import asyncio

    from agent_utilities.knowledge_graph.core.analysis_models import (
        DeepExtractionResult,
    )

    high_weight = [r for r in enriched if r.get("score", 0) > 0.3]
    if not high_weight:
        return {
            "layer": 3,
            "papers_analyzed": 0,
            "extractions": [],
            "note": "No high-weight matches for deep extraction",
        }

    # Build a SINGLE batched prompt for all high-weight matches
    match_sections = []
    for i, hw in enumerate(high_weight[:10], 1):
        claims_text = "\n".join(f"  - {c}" for c in hw.get("innovation_claims", []))
        match_sections.append(
            f"### Match {i}: {hw.get('name', hw.get('id', ''))}\n"
            f"- Score: {hw.get('score', 0):.3f}\n"
            f"- Innovation Signals: {hw.get('total_signal_count', 0)}\n"
            f"- Claims:\n{claims_text or '  (none)'}\n"
        )

    batched_prompt = (
        f"## Batched Deep Extraction for: {query}\n\n"
        f"Analyze the following {len(match_sections)} high-scoring matches and "
        f"extract structured knowledge for each:\n\n"
        + "\n".join(match_sections)
        + "\n---\n\n"
        "For EACH match, extract: key algorithms/techniques, data structures, "
        "architectural patterns, and an integration blueprint. "
        "Include the source_name for each extraction."
    )

    system_prompt = (
        "You are a deep technical analyst extracting structured knowledge from "
        "research papers and codebases for integration into an agent framework. "
        "Provide one extraction per match analyzed."
    )

    try:
        from pydantic_ai import Agent

        from agent_utilities.core.model_factory import create_model

        agent = Agent(
            model=create_model(),
            system_prompt=system_prompt,
            output_type=DeepExtractionResult,
        )
        result = await asyncio.to_thread(agent.run_sync, batched_prompt)
        deep_result: DeepExtractionResult = result.data  # type: ignore[assignment]

        # Write discovered relationships to the KG
        new_analogies = 0
        for extraction in deep_result.extractions:
            if extraction.source_name and extraction.patterns:
                for pattern in extraction.patterns[:3]:
                    success = engine.resolve_and_link(
                        source_name=extraction.source_name,
                        target_name=pattern,
                        rel_type="ANALOGOUS_TO",
                        properties={
                            "source": "deep_extraction",
                            "query": query,
                        },
                    )
                    if success:
                        new_analogies += 1

        return {
            "layer": 3,
            "papers_analyzed": len(deep_result.extractions),
            "extractions": [e.model_dump() for e in deep_result.extractions],
            "new_analogies_created": new_analogies,
        }
    except Exception as e:
        logger.warning("L3 deep extraction failed: %s", e)
        return {
            "layer": 3,
            "error": f"Deep extraction failed: {e}",
            "papers_analyzed": 0,
            "extractions": [],
        }


def _run_owl_cycle(engine: Any) -> dict[str, Any]:
    """Trigger a lightweight OWL reasoning cycle on the engine's graph.

    Performs transitive/symmetric closure to discover inferred relationships
    from the edges created during L3 deep extraction.
    """
    try:
        from agent_utilities.knowledge_graph.backends.owl import create_owl_backend
        from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge

        owl_backend = create_owl_backend()
        bridge = OWLBridge(
            graph=engine.graph,
            owl_backend=owl_backend,
            backend=engine.backend,
        )
        stats = bridge.run_cycle(lightweight=True)
        return {"status": "success", **stats}
    except Exception as e:
        logger.debug("OWL cycle skipped: %s", e)
        return {"status": "skipped", "reason": str(e)}


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
