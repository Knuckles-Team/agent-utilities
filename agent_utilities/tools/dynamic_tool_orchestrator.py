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
            results = self.engine.backend.execute(
                "MATCH (t:CallableResource)-[:BELONGS_TO]->(d:Domain) "
                "WHERE toLower($task) CONTAINS toLower(d.name) "
                "RETURN t.name AS tool_name, t.description AS tool_desc, t.schema AS schema "
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
                "[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Assigned %d dynamic tools for role '%s'",
                len(tools),
                agent_role,
            )
        except Exception as e:
            logger.debug("Failed to dynamically assign tools: %s", e)

        return tools

    def resolve_mcp_tools(
        self, query: str, server_name: str | None = None
    ) -> list[str]:
        """Resolve a list of tool names that match the query keyword or fuzzy criteria.

        If server_name is provided, filters to that server's tools.
        If zero matches are found, or an error occurs, defaults to returning all tools
        associated with the server (or all tools in general).
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
            RETURN c.name AS name
            """
            rows = self.engine.backend.execute(
                cypher_query, {"query": query, "server_name": server_name}
            )
            matched_tools = [str(r.get("name")) for r in rows if r.get("name")]
        except Exception as e:
            logger.debug("Error during resolve_mcp_tools: %s", e)

        # Fallback: if zero matches found, get all tools for this server
        if not matched_tools and server_name:
            try:
                cypher_all = """
                MATCH (s:Server {name: $server_name})-[:PROVIDES]->(c:CallableResource)
                RETURN c.name AS name
                """
                rows = self.engine.backend.execute(
                    cypher_all, {"server_name": server_name}
                )
                matched_tools = [str(r.get("name")) for r in rows if r.get("name")]
            except Exception as e:
                logger.debug("Error during resolve_mcp_tools fallback: %s", e)

        # Lazy Freshness Sweep: Check if the cache is older than 24 hours and trigger lazy sync
        if server_name:
            try:
                cypher_ts = """
                MATCH (s:Server {name: $name})
                RETURN s.timestamp AS ts
                """
                rows = self.engine.backend.execute(cypher_ts, {"name": server_name})
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

        Reads the command/args/env from the database's Server node, runs live
        list_tools() discovery, and updates the CallableResource nodes in the KG.
        """
        if not self.engine.backend:
            return False

        try:
            cypher = """
            MATCH (s:Server {name: $name})
            RETURN s.command AS command, s.args AS args, s.env AS env, s.source_config AS source_config
            """
            rows = self.engine.backend.execute(cypher, {"name": server_name})
            if not rows:
                logger.warning("Server '%s' not found in KG for refresh", server_name)
                return False

            row = rows[0]
            command = row.get("command")
            if not command:
                logger.warning(
                    "Server '%s' has no command configured in KG", server_name
                )
                return False

            import json

            raw_args = row.get("args")
            args = (
                json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or [])
            )

            raw_env = row.get("env")
            env = json.loads(raw_env) if isinstance(raw_env, str) else (raw_env or {})

            source_config = row.get("source_config") or "unknown"

            # Re-discover tools
            import hashlib

            payload = json.dumps(
                {
                    "name": server_name,
                    "command": command,
                    "args": args,
                    "env": sorted(env.items()),
                },
                sort_keys=True,
            )
            config_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]

            server_config = {
                "name": server_name,
                "command": command,
                "args": args,
                "env": env,
                "config_hash": config_hash,
                "tool_flags": [],
            }

            # Call discover_mcp_tools (using the mixin method on the engine)
            live_tools = await self.engine.discover_mcp_tools(
                server_config, timeout=30.0
            )

            # Ingest/update tools
            self.engine.ingest_mcp_server(
                name=server_name,
                url=f"stdio://{command} {' '.join(args)}",
                tools=live_tools,
                resources={"source_config": source_config, "env": env},
            )

            # Update Server metadata in DB
            ts = __import__("time").strftime(
                "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()
            )
            self.engine.backend.execute(
                "MATCH (s:Server {id: $sid}) "
                "SET s.config_hash = $hash, s.timestamp = $ts, "
                "s.tool_count = $tc",
                {
                    "sid": f"srv:{server_name}",
                    "hash": config_hash,
                    "ts": ts,
                    "tc": len(live_tools),
                },
            )
            logger.info(
                "Successfully refreshed tools cache for server '%s'", server_name
            )
            return True

        except Exception as e:
            logger.exception(
                "Failed to refresh cached tools for server '%s': %s", server_name, e
            )
            return False

    # ── OrchestratorProtocol conformance ──────────────────────────────────

    async def dispatch(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """Dispatch a tool assignment task."""
        import uuid

        job_id = f"dto:{uuid.uuid4().hex[:8]}"
        role = kwargs.get("agent_role", "general")
        try:
            tools = self.assign_tools_for_task(task, role)
            return {"job_id": job_id, "status": "completed", "output": tools}
        except Exception as e:
            return {"job_id": job_id, "status": "failed", "error": str(e)}

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Return status of a dispatched job (synchronous — always terminal)."""
        return {"job_id": job_id, "status": "completed"}
