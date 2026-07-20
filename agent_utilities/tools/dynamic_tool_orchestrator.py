#!/usr/bin/python
from __future__ import annotations

"""Dynamic Tool Assignment Orchestration (CONCEPT:AU-ECO.messaging.native-backend-abstraction).

Matches tool ontology to agent tasks dynamically at runtime. Resolves the
exact tools needed for a dynamically spawned agent by vectorizing the task schema.
"""


import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class DynamicToolOrchestrator:
    """Dynamically assigns tools based on task context and KG embeddings.

    CONCEPT:AU-ECO.messaging.native-backend-abstraction
    """

    def __init__(self, engine: IntelligenceGraphEngine):
        self.engine = engine

    def assign_tools_for_task(
        self, task_description: str, agent_role: str
    ) -> list[dict[str, Any]]:
        """Dynamically find the most relevant tools for a given task.

        Leverages ECO-4.6 (Self-Describing Function Registry) and
        KG-2.7 (Topological Analogy Engine).
        """
        if not self.engine.backend:
            return []

        tools = []
        try:
            # Query the KG for tools that are relevant to this task domain
            # and are capable of being used by this agent role.
            results = self.engine.query_cypher(
                "MATCH (t:CallableResource)-[:BELONGS_TO]->(d:Domain) "
                "WHERE toLower($task) CONTAINS toLower(d.name) "
                "RETURN t.id AS id, t.name AS tool_name, "
                "t.description AS tool_desc, t.schema AS schema "
                "LIMIT 5",
                {"task": task_description},
            )

            for r in results:
                name = r.get("tool_name")
                if name:
                    tools.append(
                        {
                            "name": name,
                            "description": r.get("tool_desc", ""),
                            "schema": r.get("schema", "{}"),
                        }
                    )

            logger.info(
                "[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Assigned %d dynamic tools",
                len(tools),
            )
        except Exception as exc:
            logger.debug("Dynamic tool assignment failed (%s)", type(exc).__name__)

        return tools

    def resolve_mcp_tools(
        self, query: str, server_name: str | None = None
    ) -> list[str]:
        """Resolve a list of tool names that match the query keyword or fuzzy criteria.

        If server_name is provided, filters to that server's tools.
        No match or query failure returns no tools. A requested narrowing filter
        is a least-privilege boundary and never widens to the server's full set.
        """
        if not self.engine.backend:
            return []

        matched_tools: list[str] = []
        try:
            # Multi-vector match: matches name, description, or tags (if present)
            cypher_query = """
            MATCH (s:Server)-[:PROVIDES]->(c:CallableResource)
            WHERE ($server_name IS NULL OR s.name = $server_name)
              AND (toLower(c.name) CONTAINS toLower($query)
                   OR toLower(c.description) CONTAINS toLower($query)
                   OR (c.tags IS NOT NULL AND any(t in c.tags WHERE toLower(t) CONTAINS toLower($query))))
            RETURN c.id AS id, c.name AS name
            """
            rows = self.engine.query_cypher(
                cypher_query, {"query": query, "server_name": server_name}
            )
            matched_tools = [str(r.get("name")) for r in rows if r.get("name")]
        except Exception as exc:
            logger.debug("Error during resolve_mcp_tools (%s)", type(exc).__name__)

        # Lazy Freshness Sweep: Check if the cache is older than 24 hours and trigger lazy sync
        if server_name:
            try:
                cypher_ts = """
                MATCH (s:Server {name: $name})
                RETURN s.id AS id, s.timestamp AS ts
                """
                rows = self.engine.query_cypher(cypher_ts, {"name": server_name})
                if rows:
                    ts_str = rows[0].get("ts")
                    if ts_str:
                        import time

                        try:
                            cached_time = time.mktime(
                                time.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
                            )
                            age_hours = (time.time() - cached_time) / 3600
                            if age_hours > 24.0:
                                # Spawn non-blocking background refresh
                                import asyncio

                                asyncio.create_task(
                                    self.refresh_cached_tools(server_name)
                                )
                        except Exception:
                            pass
            except Exception:
                pass

        return matched_tools

    async def refresh_cached_tools(self, server_name: str) -> bool:
        """Force-refresh the cached tool metadata for the given MCP server.

        Runtime transport is resolved only from the active external MCP
        configuration. The KG stores capability metadata and opaque references;
        it is never a command, argument, environment, or endpoint secret store.
        """
        if not self.engine.backend:
            return False

        try:
            from agent_utilities.core.config import setting
            from agent_utilities.knowledge_graph.core.engine_ingestion import (
                _mcp_persistence_resources,
                _neutral_mcp_alias,
            )
            from agent_utilities.mcp.multiplexer import (
                MCPMultiplexer,
                _resolve_config_path,
            )

            config_path = _resolve_config_path(
                str(setting("MCP_CONFIG", "") or "") or None
            )
            catalog = MCPMultiplexer(config_path).load_catalog()
            server_config = None
            declaration = catalog.get(server_name)
            if isinstance(declaration, dict):
                entries = self.engine.parse_mcp_config(
                    {"mcpServers": {server_name: declaration}}
                )
                server_config = entries[0] if len(entries) == 1 else None
            else:
                # Persisted server names are always opaque. Resolve the runtime
                # catalog entry by recomputing each keyed identity in memory;
                # neither the catalog name nor its endpoint enters the graph.
                for candidate_name, candidate in catalog.items():
                    entries = self.engine.parse_mcp_config(
                        {"mcpServers": {candidate_name: candidate}}
                    )
                    if len(entries) != 1:
                        continue
                    candidate_config = entries[0]
                    candidate_alias = _neutral_mcp_alias(
                        config_hash=candidate_config["config_hash"]
                    )
                    if candidate_alias == server_name:
                        server_config = candidate_config
                        break
            if server_config is None:
                logger.warning("MCP server declaration is unavailable for refresh")
                return False
            config_hash = server_config["config_hash"]
            persisted_name = _neutral_mcp_alias(config_hash=config_hash)

            # Call discover_mcp_tools (using the mixin method on the engine)
            live_tools = await self.engine.discover_mcp_tools(
                server_config, timeout=30.0
            )

            # Ingest/update tools
            self.engine.ingest_mcp_server(
                name=persisted_name,
                url=f"mcp-ref://{config_hash}",
                tools=live_tools,
                resources=_mcp_persistence_resources(
                    config_path, server_config.get("env")
                ),
            )

            # Update Server metadata in DB
            ts = __import__("time").strftime(
                "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()
            )
            self.engine.add_node(
                f"srv:{persisted_name}",
                "Server",
                {
                    "config_hash": config_hash,
                    "timestamp": ts,
                    "tool_count": len(live_tools),
                },
            )
            logger.info("Successfully refreshed one MCP tools cache")
            return True

        except Exception as exc:
            logger.error("Failed to refresh MCP tools cache (%s)", type(exc).__name__)
            return False

    # ── OrchestratorProtocol conformance ──────────────────────────────────

    async def dispatch(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """Dispatch a tool assignment task."""
        import uuid

        job_id = f"dto:{uuid.uuid4().hex}"
        role = kwargs.get("agent_role", "general")
        try:
            tools = self.assign_tools_for_task(task, role)
            return {"job_id": job_id, "status": "completed", "output": tools}
        except Exception as e:
            return {"job_id": job_id, "status": "failed", "error": str(e)}

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Return status of a dispatched job (synchronous — always terminal)."""
        return {"job_id": job_id, "status": "completed"}
