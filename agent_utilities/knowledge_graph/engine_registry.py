"""Registry and CRUD mixin for IntelligenceGraphEngine.

Extracted from engine.py. Contains agent identity, prompt management,
skill/tool listing, resource toggling, codemap operations, and mermaid generation.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from ._engine_protocol import _EngineProtocol

    _Base = _EngineProtocol
else:
    _Base = object


import contextlib
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

import networkx as nx

from ..models.knowledge_graph import (
    PromptNode,
    RegistryEdgeType,
    SystemPromptNode,
)

logger = logging.getLogger(__name__)


@dataclass
class FocusedSubgraph:
    """A task-specific subgraph extraction result."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    summary: str
    query: str

    def to_mermaid(self) -> str:
        """Convert this subgraph to a Mermaid diagram."""
        from agent_utilities.observability.mermaid import FlowchartBuilder

        builder = FlowchartBuilder(title=f"Subgraph: {self.query}")

        for node in self.nodes:
            n_type = node.get("type", "symbol")
            shape = "box"
            if n_type == "file":
                shape = "cylinder"
            elif n_type == "class":
                shape = "round"

            builder.add_node(
                node["id"],
                label=f"{node['label']}\n({n_type})",
                shape=shape,
                css_class=n_type.lower(),
            )

        for edge in self.edges:
            builder.add_edge(edge["source"], edge["target"], label=edge["type"])

        return builder.render()


class RegistryMixin(_Base):
    """Registry and CRUD capabilities for the KG engine."""

    async def extract_focused_subgraph(
        self,
        query: str,
        max_nodes: int = 150,
        min_centrality: float = 0.01,
    ) -> FocusedSubgraph:
        """
        Task-specific subgraph extraction — the core engine behind Codemaps.
        Reuses existing search + topological analysis.
        """
        # 1. Hybrid search for initial candidates
        logger.info(f"Extracting subgraph for query: {query}")
        search_results = self.search_hybrid(query, top_k=max_nodes * 2)
        logger.info(f"Hybrid search found {len(search_results)} candidates")

        # 2. Build initial node set
        seed_ids = [r["id"] for r in search_results]
        subgraph = self.graph.subgraph(seed_ids).copy()
        logger.info(f"Initial subgraph has {len(subgraph)} nodes")

        # 3. Expand to include neighbors with high centrality
        # In MultiDiGraph, neighbors() returns successors. We might want both.
        for node_id in list(subgraph.nodes):
            logger.info(f"Expanding node: {node_id}")
            if self.graph.nodes[node_id].get("centrality", 0) >= min_centrality:
                # Add successors and predecessors
                for successor in self.graph.successors(node_id):
                    if successor not in subgraph:
                        subgraph.add_node(successor, **self.graph.nodes[successor])
                    subgraph.add_edge(
                        node_id,
                        successor,
                        **self.graph.get_edge_data(node_id, successor)[0],
                    )

                for predecessor in self.graph.predecessors(node_id):
                    if predecessor not in subgraph:
                        subgraph.add_node(predecessor, **self.graph.nodes[predecessor])
                    subgraph.add_edge(
                        predecessor,
                        node_id,
                        **self.graph.get_edge_data(predecessor, node_id)[0],
                    )

        # 4. Prune if still too large using PageRank
        if len(subgraph) > max_nodes:
            centrality = nx.pagerank(nx.DiGraph(subgraph), alpha=0.85)
            top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:max_nodes]
            subgraph = subgraph.subgraph(top_nodes).copy()

        # 5. Convert to clean list of dicts
        nodes = []
        edges = []
        for node_id, data in subgraph.nodes(data=True):
            nodes.append(
                {
                    "id": node_id,
                    "label": data.get("name") or str(node_id).split(":")[-1],
                    "type": data.get("type", "symbol"),
                    "file": data.get("file", data.get("skill_code_path", "")),
                    "line": data.get("line"),
                    "centrality": data.get("centrality", 0.0),
                }
            )

        for u, v, key, edata in subgraph.edges(data=True, keys=True):
            edges.append(
                {
                    "source": u,
                    "target": v,
                    "type": edata.get("type", "calls"),
                    "weight": edata.get("weight", 1.0),
                }
            )

        summary = f"Subgraph for '{query}' with {len(nodes)} nodes focused on relevant execution paths."

        return FocusedSubgraph(
            nodes=nodes,
            edges=edges,
            summary=summary,
            query=query,
        )

    async def get_codemap_by_id(self, codemap_id: str) -> Any | None:
        """Retrieve a codemap artifact by its ID."""
        # Check in-memory first if we store them there
        if f"codemap:{codemap_id}" in self.graph:
            data = self.graph.nodes[f"codemap:{codemap_id}"]
            from ..models.codemap import CodemapArtifact

            return CodemapArtifact.model_validate(data)

        if self.backend:
            res = self.backend.execute(
                "MATCH (c:Codemap {id: $id}) RETURN c", {"id": codemap_id}
            )
            if res:
                import json

                from ..models.codemap import CodemapArtifact

                c_data = res[0]["c"]
                # Handle JSON serialization of complex fields if stored as strings
                for k in ["hierarchy", "nodes", "edges", "metadata"]:
                    if k in c_data and isinstance(c_data[k], str):
                        with contextlib.suppress(Exception):
                            c_data[k] = json.loads(c_data[k])
                return CodemapArtifact.model_validate(c_data)
        return None

    async def get_codemap_by_slug(self, slug: str) -> Any | None:
        """Retrieve a codemap artifact by a fuzzy match on prompt/slug."""
        if self.backend:
            res = self.backend.execute(
                "MATCH (c:Codemap) WHERE c.prompt CONTAINS $slug OR c.id CONTAINS $slug RETURN c LIMIT 1",
                {"slug": slug},
            )
            if res:
                import json

                from ..models.codemap import CodemapArtifact

                c_data = res[0]["c"]
                for k in ["hierarchy", "nodes", "edges", "metadata"]:
                    if k in c_data and isinstance(c_data[k], str):
                        with contextlib.suppress(Exception):
                            c_data[k] = json.loads(c_data[k])
                return CodemapArtifact.model_validate(c_data)
        return None

    async def store_codemap(self, artifact: Any):
        """Persist a codemap artifact to the graph."""
        node_id = f"codemap:{artifact.id}"
        data = artifact.model_dump()

        # Add to in-memory graph
        self.graph.add_node(node_id, **data)

        # Persist to backend
        if self.backend:
            clean_data = self._serialize_node(artifact, label="Codemap")
            self._upsert_node("Codemap", artifact.id, clean_data)

    # ─────────────────────────────────────────────────────────────────────
    #  Identity Management (KG-first, no legacy fallbacks)
    # ─────────────────────────────────────────────────────────────────────

    def get_agent_identity(self) -> dict[str, Any]:
        """Load the agent identity from the KG SystemPrompt node.

        CONCEPT:KG-001 — Identity Management

        Returns a dict with at minimum: name, description, emoji, content.
        """
        if self.backend:
            results = self.backend.execute(
                "MATCH (s:SystemPrompt) RETURN s.id, s.name, s.description, "
                "s.content, s.version, s.tags, s.parameters, s.source "
                "ORDER BY s.timestamp DESC LIMIT 1",
                {},
            )
            if results:
                row = results[0]
                return {
                    "id": row.get("s.id", ""),
                    "name": row.get("s.name", "Agent"),
                    "description": row.get("s.description", ""),
                    "content": row.get("s.content", ""),
                    "version": row.get("s.version", "1.0"),
                    "tags": row.get("s.tags", []),
                    "parameters": row.get("s.parameters", {}),
                    "source": row.get("s.source", "KG"),
                }

        # In-memory fallback within graph
        for nid, data in self.graph.nodes(data=True):
            if str(data.get("type", "")).lower() == "system_prompt":
                return {"id": nid, **data}

        return {"id": "", "name": "Agent", "description": "", "content": ""}

    def add_agent_identity(self, identity: dict[str, Any]) -> dict[str, Any]:
        """Create a new agent identity node in the KG.

        CONCEPT:KG-001 — Identity Management

        Args:
            identity: Dict with name, description, content, and optional tags/source.

        Returns:
            The created identity dict with generated id.
        """
        node_id = f"identity:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        node = SystemPromptNode(
            id=node_id,
            name=identity.get("name", "Agent"),
            description=identity.get("description", ""),
            content=identity.get("content", ""),
            version=identity.get("version", "1.0"),
            tags=identity.get("tags", []),
            parameters=identity.get("parameters", {}),
            source=identity.get("source", "MANUAL"),
            timestamp=ts,
        )

        self.graph.add_node(node.id, **node.model_dump())
        if self.backend:
            data = self._serialize_node(node, label="SystemPrompt")
            self._upsert_node("SystemPrompt", node.id, data)

        return {"id": node_id, **node.model_dump()}

    def update_agent_identity(self, identity: dict[str, Any]) -> None:
        """Update the existing agent identity node in the KG.

        CONCEPT:KG-001 — Identity Management
        """
        current = self.get_agent_identity()
        node_id = current.get("id")
        if not node_id:
            self.add_agent_identity(identity)
            return

        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        updates = {k: v for k, v in identity.items() if k != "id" and v is not None}
        updates["timestamp"] = ts

        if node_id in self.graph:
            self.graph.nodes[node_id].update(updates)
        if self.backend:
            self.backend.execute(
                "MATCH (n:SystemPrompt {id: $id}) SET n += $props",
                {"id": node_id, "props": updates},
            )

    # ─────────────────────────────────────────────────────────────────────
    #  Prompt Management (with versioning and rollback)
    # ─────────────────────────────────────────────────────────────────────

    def get_all_prompts(self) -> list[dict[str, Any]]:
        """Return all Prompt nodes from the KG with metadata.

        CONCEPT:KG-002 — Prompt Management
        """
        results: list[dict[str, Any]] = []
        if self.backend:
            rows = self.backend.execute(
                "MATCH (p:Prompt) RETURN p.id, p.name, p.description, "
                "p.system_prompt, p.capabilities, p.timestamp "
                "ORDER BY p.name",
                {},
            )
            for row in rows:
                results.append(
                    {
                        "id": row.get("p.id", ""),
                        "name": row.get("p.name", ""),
                        "description": row.get("p.description", ""),
                        "content": row.get("p.system_prompt", ""),
                        "capabilities": row.get("p.capabilities", []),
                        "timestamp": row.get("p.timestamp", ""),
                        "type": "prompt",
                    }
                )
            return results

        # In-memory
        for nid, data in self.graph.nodes(data=True):
            if str(data.get("type", "")).lower() == "prompt":
                results.append({"id": nid, **data})
        return results

    def get_prompt(self, prompt_id: str) -> dict[str, Any] | None:
        """Return a single prompt with its full content.

        CONCEPT:KG-002 — Prompt Management
        """
        if self.backend:
            rows = self.backend.execute(
                "MATCH (p:Prompt {id: $id}) RETURN p.id, p.name, p.description, "
                "p.system_prompt, p.capabilities, p.timestamp, p.json_blueprint",
                {"id": prompt_id},
            )
            if rows:
                row = rows[0]
                return {
                    "id": row.get("p.id", ""),
                    "name": row.get("p.name", ""),
                    "description": row.get("p.description", ""),
                    "content": row.get("p.system_prompt", ""),
                    "capabilities": row.get("p.capabilities", []),
                    "timestamp": row.get("p.timestamp", ""),
                    "json_blueprint": row.get("p.json_blueprint", {}),
                }
        if prompt_id in self.graph:
            return {"id": prompt_id, **self.graph.nodes[prompt_id]}
        return None

    def add_prompt(
        self, content: str, name: str, author: str = "user", description: str = ""
    ) -> dict[str, Any]:
        """Create a new prompt node in the KG.

        CONCEPT:KG-002 — Prompt Management

        Returns the created prompt dict with generated id.
        """
        prompt_id = f"prompt:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        node = PromptNode(
            id=prompt_id,
            name=name,
            description=description,
            system_prompt=content,
            timestamp=ts,
        )

        self.graph.add_node(node.id, **node.model_dump())
        if self.backend:
            data = self._serialize_node(node, label="Prompt")
            data["author"] = author
            data["version_number"] = 1
            self._upsert_node("Prompt", node.id, data)

        return {
            "id": prompt_id,
            "name": name,
            "content": content,
            "description": description,
            "author": author,
            "version_number": 1,
            "timestamp": ts,
        }

    def update_prompt(
        self, prompt_id: str, content: str, author: str = "user"
    ) -> dict[str, Any]:
        """Create a new version of a prompt via SUPERSEDES relationship.

        CONCEPT:KG-002 — Prompt Management

        The old version is preserved; a new node is created and linked
        to the previous one via SUPERSEDES (forward-only, never destructive).
        """
        old = self.get_prompt(prompt_id)
        if not old:
            raise ValueError(f"Prompt {prompt_id} not found")

        # Determine next version number
        versions = self.get_prompt_versions(prompt_id, limit=1)
        next_version = len(versions) + 1

        new_id = f"prompt:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = PromptNode(
            id=new_id,
            name=old.get("name", ""),
            description=old.get("description", ""),
            system_prompt=content,
            timestamp=ts,
        )

        self.graph.add_node(node.id, **node.model_dump())
        self.graph.add_edge(new_id, prompt_id, type=RegistryEdgeType.SUPERSEDES)

        if self.backend:
            data = self._serialize_node(node, label="Prompt")
            data["author"] = author
            data["version_number"] = next_version
            data["parent_id"] = prompt_id
            self._upsert_node("Prompt", new_id, data)
            self.backend.execute(
                "MATCH (a:Prompt {id: $new_id}), (b:Prompt {id: $old_id}) "
                "MERGE (a)-[:SUPERSEDES]->(b)",
                {"new_id": new_id, "old_id": prompt_id},
            )

        return {
            "id": new_id,
            "name": old.get("name", ""),
            "content": content,
            "author": author,
            "version_number": next_version,
            "parent_id": prompt_id,
            "timestamp": ts,
        }

    def get_prompt_versions(
        self, prompt_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Walk the SUPERSEDES chain to return version history for a prompt.

        CONCEPT:KG-002 — Prompt Management

        Returns versions ordered newest-first.
        """
        versions: list[dict[str, Any]] = []

        if self.backend:
            # Find all prompts in the SUPERSEDES chain
            rows = self.backend.execute(
                "MATCH path = (latest:Prompt)-[:SUPERSEDES*0..]->(root:Prompt) "
                "WHERE root.id = $id OR latest.id = $id "
                "RETURN latest.id, latest.name, latest.system_prompt, "
                "latest.author, latest.version_number, latest.timestamp, latest.parent_id "
                "ORDER BY latest.version_number DESC "
                "LIMIT $limit",
                {"id": prompt_id, "limit": limit},
            )
            seen = set()
            for row in rows:
                vid = row.get("latest.id", "")
                if vid and vid not in seen:
                    seen.add(vid)
                    versions.append(
                        {
                            "id": vid,
                            "name": row.get("latest.name", ""),
                            "content": row.get("latest.system_prompt", ""),
                            "author": row.get("latest.author", ""),
                            "version_number": row.get("latest.version_number", 1),
                            "timestamp": row.get("latest.timestamp", ""),
                            "parent_id": row.get("latest.parent_id", ""),
                        }
                    )
            return versions

        # In-memory traversal
        current = prompt_id
        visited = set()
        while current and current not in visited and len(versions) < limit:
            visited.add(current)
            if current in self.graph:
                data = dict(self.graph.nodes[current])
                versions.append({"id": current, **data})
            # Follow SUPERSEDES edges
            for _, target, edata in self.graph.out_edges(current, data=True):
                if edata.get("type") == RegistryEdgeType.SUPERSEDES:
                    current = target
                    break
            else:
                break
        return versions

    def rollback_prompt(self, prompt_id: str, target_version_id: str) -> dict[str, Any]:
        """Rollback a prompt to a previous version.

        CONCEPT:KG-002 — Prompt Management (AHE Rollback)

        This follows the agentic harness principle: always forward, never
        destructive. A new version is created that copies the target's content,
        linked via SUPERSEDES to the current latest.
        """
        target = self.get_prompt(target_version_id)
        if not target:
            raise ValueError(f"Target version {target_version_id} not found")

        target_content = target.get("content", target.get("system_prompt", ""))
        return self.update_prompt(prompt_id, content=target_content, author="rollback")

    # ─────────────────────────────────────────────────────────────────────
    #  Granular Resource Queries (Skills, Tools, Prompts)
    # ─────────────────────────────────────────────────────────────────────

    def get_skills(self) -> list[dict[str, Any]]:
        """Return pre-baked skills (Skill nodes and agent skills).

        CONCEPT:KG-003 — Granular Resource Queries
        """
        results: list[dict[str, Any]] = []
        if self.backend:
            rows = self.backend.execute(
                "MATCH (s:CallableResource) WHERE s.resource_type = 'AGENT_SKILL' "
                "RETURN s.id, s.name, s.description, s.skill_code_path, s.timestamp",
                {},
            )
            for row in rows:
                results.append(
                    {
                        "id": row.get("s.id", ""),
                        "name": row.get("s.name", ""),
                        "description": row.get("s.description", ""),
                        "enabled": True,
                        "type": "skill",
                        "source": "universal-skills",
                        "path": row.get("s.skill_code_path", ""),
                    }
                )

        # Also check in-memory graph for Skill-type nodes
        for nid, data in self.graph.nodes(data=True):
            n_type = str(data.get("type", "")).lower()
            r_type = str(data.get("resource_type", "")).lower()
            if n_type == "skill" or r_type == "agent_skill":
                if not any(r["id"] == nid for r in results):
                    results.append(
                        {
                            "id": nid,
                            "name": data.get("name", nid),
                            "description": data.get("description", ""),
                            "enabled": True,
                            "type": "skill",
                            "source": data.get("package_name", "custom"),
                        }
                    )

        return sorted(results, key=lambda x: x.get("name", "").lower())

    def get_tools(self) -> list[dict[str, Any]]:
        """Return MCP tools from the KG.

        CONCEPT:KG-003 — Granular Resource Queries
        """
        results: list[dict[str, Any]] = []
        if self.backend:
            rows = self.backend.execute(
                "MATCH (r:CallableResource) WHERE r.resource_type = 'MCP_TOOL' "
                "RETURN r.id, r.name, r.description, r.endpoint, r.timestamp",
                {},
            )
            for row in rows:
                results.append(
                    {
                        "id": row.get("r.id", ""),
                        "name": row.get("r.name", ""),
                        "description": row.get("r.description", ""),
                        "enabled": True,
                        "type": "mcp_tool",
                        "source": row.get("r.endpoint", "mcp"),
                    }
                )

        # Also check Server->PROVIDES->CallableResource
        for nid, data in self.graph.nodes(data=True):
            r_type = str(data.get("resource_type", "")).lower()
            if r_type == "mcp_tool":
                if not any(r["id"] == nid for r in results):
                    results.append(
                        {
                            "id": nid,
                            "name": data.get("name", nid),
                            "description": data.get("description", ""),
                            "enabled": True,
                            "type": "mcp_tool",
                            "source": data.get("endpoint", "mcp"),
                        }
                    )

        return sorted(results, key=lambda x: x.get("name", "").lower())

    def get_prompts_list(self) -> list[dict[str, Any]]:
        """Return system prompts and custom prompts from KG.

        Alias that wraps get_all_prompts for the unified granular API.
        """
        return self.get_all_prompts()

    def toggle_resource(self, resource_id: str) -> dict[str, Any]:
        """Toggle the enabled/disabled flag on any Skill, Tool, or Prompt node.

        CONCEPT:KG-003 — Granular Resource Queries

        Uses a KG flag approach — does not disconnect MCP servers.
        """
        if self.backend:
            # Check current enabled state
            rows = self.backend.execute(
                "MATCH (n {id: $id}) RETURN n.id, n.enabled, n.name",
                {"id": resource_id},
            )
            if rows:
                current = rows[0].get("n.enabled", True)
                new_state = not current
                self.backend.execute(
                    "MATCH (n {id: $id}) SET n.enabled = $enabled",
                    {"id": resource_id, "enabled": new_state},
                )
                return {
                    "id": resource_id,
                    "name": rows[0].get("n.name", ""),
                    "enabled": new_state,
                }

        # In-memory
        if resource_id in self.graph:
            data = self.graph.nodes[resource_id]
            current = data.get("enabled", True)
            new_state = not current
            data["enabled"] = new_state
            return {
                "id": resource_id,
                "name": data.get("name", ""),
                "enabled": new_state,
            }

        raise ValueError(f"Resource {resource_id} not found")

    # ─────────────────────────────────────────────────────────────────────
    #  Workspace Reload (re-ingest files into KG)
    # ─────────────────────────────────────────────────────────────────────

    def reload_from_workspace(self) -> dict[str, Any]:
        """Re-read workspace files and update KG nodes.

        CONCEPT:KG-004 — Workspace Reload

        Returns a change summary dict.
        """

        changes: dict[str, Any] = {
            "identity_changed": False,
            "prompts_updated": 0,
            "tools_synced": 0,
            "cron_tasks_refreshed": 0,
        }

        # 1. Reload identity from main_agent.json
        try:
            from agent_utilities.prompting.builder import load_identity

            new_identity = load_identity()
            if new_identity:
                current = self.get_agent_identity()
                if new_identity.get("name") != current.get("name") or new_identity.get(
                    "description"
                ) != current.get("description"):
                    self.update_agent_identity(new_identity)
                    changes["identity_changed"] = True
        except Exception as e:
            logger.warning(f"Failed to reload identity: {e}")

        # 2. Count current tools for sync summary
        try:
            tools = self.get_tools()
            changes["tools_synced"] = len(tools)
        except Exception as e:
            logger.warning(f"Failed to count tools: {e}")

        # 3. Count prompts
        try:
            prompts = self.get_all_prompts()
            changes["prompts_updated"] = len(prompts)
        except Exception as e:
            logger.warning(f"Failed to count prompts: {e}")

        logger.info(f"Workspace reload summary: {changes}")
        return changes

    def generate_mermaid_graph(
        self,
        query: str | None = None,
        max_nodes: int = 50,
        title: str = "Knowledge Graph",
    ) -> str:
        """Generate a Mermaid visualization for a portion of the graph."""
        from agent_utilities.observability.mermaid import FlowchartBuilder

        if query:
            # Simple heuristic for subgraph if query provided
            results = self.search_hybrid(query, top_k=max_nodes)
            node_ids = [r["id"] for r in results]
            subgraph = self.graph.subgraph(node_ids).copy()
        else:
            # Just take the first N nodes
            node_ids = list(self.graph.nodes)[:max_nodes]
            subgraph = self.graph.subgraph(node_ids).copy()

        builder = FlowchartBuilder(title=title)

        for n, data in subgraph.nodes(data=True):
            n_type = data.get("type", "unknown")
            shape = "box"
            if n_type == "episode":
                shape = "round"
            elif n_type == "memory":
                shape = "cylinder"
            elif n_type == "agent":
                shape = "circle"

            builder.add_node(
                n,
                label=f"{data.get('name', n)}\n({n_type})",
                shape=shape,
                css_class=n_type.lower(),
            )

        for u, v, data in subgraph.edges(data=True):
            builder.add_edge(u, v, label=data.get("type", ""))

        # Add some default styling for KG types
        builder.lines.append(
            "  classDef episode fill:#2e7d32,stroke:#1b5e20,color:#fff"
        )
        builder.lines.append("  classDef memory fill:#1565c0,stroke:#0d47a1,color:#fff")
        builder.lines.append("  classDef agent fill:#f57c00,stroke:#e65100,color:#fff")

        return builder.render()

    # ─────────────────────────────────────────────────────────────────────
    #  TeamConfig: Proven Team Reuse (CONCEPT:AHE-3.3)
    # ─────────────────────────────────────────────────────────────────────

    def find_matching_team_config(
        self,
        query: str,
        top_k: int = 3,
    ) -> list:
        """Find TeamConfig nodes whose task_pattern semantically matches the query.

        CONCEPT:AHE-3.3 — Proven Team Reuse

        Uses hybrid search on TeamConfig nodes to find previously
        successful team compositions for similar tasks.

        Args:
            query: The user query to match against.
            top_k: Maximum number of matches to return.

        Returns:
            A list of ``TeamConfigNode`` instances, sorted by success_rate.
        """
        from ..models.knowledge_graph import TeamConfigNode

        results: list[TeamConfigNode] = []

        if self.backend:
            rows = self.backend.execute(
                "MATCH (tc:TeamConfig) RETURN tc",
                {},
            )
            for row in rows:
                data = row.get("tc", row)
                try:
                    node = TeamConfigNode.model_validate(data)
                    # Simple keyword matching (cosine similarity can be added later)
                    query_lower = query.lower()
                    pattern_lower = node.task_pattern.lower()
                    overlap = len(set(query_lower.split()) & set(pattern_lower.split()))
                    if overlap > 0:
                        results.append(node)
                except Exception as e:
                    logger.debug(f"Failed to parse TeamConfig: {e}")

        # Also check in-memory
        for nid, data in self.graph.nodes(data=True):
            if str(data.get("type", "")).lower() == "team_config":
                try:
                    node = TeamConfigNode.model_validate({"id": nid, **data})
                    if node not in results:
                        results.append(node)
                except Exception:  # nosec B110
                    pass  # Defensive: skip malformed in-memory nodes

        # Sort by success_rate descending
        results.sort(key=lambda t: t.success_rate, reverse=True)
        return results[:top_k]

    def promote_coalition_to_template(
        self,
        coalition_id: str,
        task_pattern: str,
    ) -> dict[str, Any]:
        """Promote a successful SwarmCoalition into a reusable TeamConfig.

        CONCEPT:AHE-3.3 — Proven Team Reuse

        Creates a ``TeamConfigNode`` from a successful coalition's metadata,
        linking it via ``REUSED_TEAM`` to the original coalition.  Also
        invalidates the hot cache so the router sees the new template.

        Args:
            coalition_id: The SwarmCoalition node ID to promote.
            task_pattern: Semantic description of the task type.

        Returns:
            A dict with the new TeamConfig's metadata.
        """
        from ..models.knowledge_graph import (
            RegistryEdgeType,
            TeamConfigNode,
        )

        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tc_id = f"team_config:{uuid.uuid4().hex[:8]}"

        # Extract coalition metadata
        coalition_data: dict[str, Any] = {}
        if coalition_id in self.graph:
            coalition_data = dict(self.graph.nodes[coalition_id])
        elif self.backend:
            rows = self.backend.execute(
                "MATCH (sc {id: $id}) RETURN sc",
                {"id": coalition_id},
            )
            if rows:
                coalition_data = rows[0].get("sc", rows[0])

        # Create TeamConfigNode
        node = TeamConfigNode(
            id=tc_id,
            name=f"TeamConfig: {task_pattern[:60]}",
            description=f"Proven team for: {task_pattern}",
            task_pattern=task_pattern,
            specialist_ids=coalition_data.get("specialist_ids", []),
            success_rate=1.0,  # Initial success (just promoted)
            usage_count=0,
            timestamp=ts,
            importance_score=0.8,
        )

        # Persist to graph
        self.graph.add_node(tc_id, **node.model_dump())
        if self.backend:
            clean_data = self._serialize_node(node, label="TeamConfig")
            self._upsert_node("TeamConfig", tc_id, clean_data)

        # Link to coalition
        self.graph.add_edge(
            tc_id,
            coalition_id,
            type=RegistryEdgeType.REUSED_TEAM,
        )
        if self.backend:
            self.backend.execute(
                "MATCH (tc {id: $tc_id}), (sc {id: $sc_id}) "
                "MERGE (tc)-[:REUSED_TEAM]->(sc)",
                {"tc_id": tc_id, "sc_id": coalition_id},
            )

        # CONCEPT:ORCH-1.2 — Invalidate cache after TeamConfig promotion
        from ..graph.config_helpers import invalidate_registry_cache

        invalidate_registry_cache()

        logger.info(
            "Promoted coalition %s to TeamConfig %s (pattern: %s)",
            coalition_id,
            tc_id,
            task_pattern,
        )
        return node.model_dump()

    def record_team_outcome(
        self,
        team_config_id: str,
        reward: float,
    ) -> None:
        """Update a TeamConfig's success_rate after a reuse outcome.

        CONCEPT:AHE-3.3 — Proven Team Reuse

        Uses an exponential moving average (alpha=0.3) to update the
        rolling success rate.  Also increments the usage counter.

        Args:
            team_config_id: The TeamConfig node ID.
            reward: The outcome reward (0.0 to 1.0).
        """
        alpha = 0.3

        if team_config_id in self.graph:
            data = self.graph.nodes[team_config_id]
            old_rate = data.get("success_rate", 0.5)
            new_rate = alpha * reward + (1 - alpha) * old_rate
            data["success_rate"] = new_rate
            data["usage_count"] = data.get("usage_count", 0) + 1

        if self.backend:
            self.backend.execute(
                "MATCH (tc:TeamConfig {id: $id}) "
                "SET tc.success_rate = $rate, "
                "    tc.usage_count = COALESCE(tc.usage_count, 0) + 1",
                {
                    "id": team_config_id,
                    "rate": alpha * reward
                    + (1 - alpha)
                    * (
                        self.graph.nodes.get(team_config_id, {}).get(
                            "success_rate", 0.5
                        )
                    ),
                },
            )

        logger.info(
            "Recorded team outcome for %s: reward=%.2f",
            team_config_id,
            reward,
        )

    def link_prompt_to_agent(
        self,
        agent_id: str,
        prompt_id: str,
    ) -> None:
        """Create a USES_PROMPT edge from an agent to a prompt node.

        CONCEPT:AHE-3.3 — Proven Team Reuse

        Args:
            agent_id: The agent node ID.
            prompt_id: The prompt node ID.
        """
        self.graph.add_edge(
            agent_id,
            prompt_id,
            type=RegistryEdgeType.USES_PROMPT,
        )
        if self.backend:
            self.backend.execute(
                "MATCH (a {id: $aid}), (p {id: $pid}) MERGE (a)-[:USES_PROMPT]->(p)",
                {"aid": agent_id, "pid": prompt_id},
            )
        logger.info("Linked agent %s → prompt %s (USES_PROMPT)", agent_id, prompt_id)

    # ─────────────────────────────────────────────────────────────────────
    #  Self-Describing Function Registry (CONCEPT:ECO-4.6)
    # ─────────────────────────────────────────────────────────────────────

    def register_function(
        self,
        function_id: str,
        name: str,
        resource_type: str = "MCP_TOOL",
        description: str = "",
        input_schema: dict | None = None,
        output_schema: dict | None = None,
        trigger_bindings: list[dict] | None = None,
        endpoint: str | None = None,
        metadata_id: str = "",
    ) -> dict[str, Any]:
        """Register a self-describing function in the KG at runtime.

        CONCEPT:ECO-4.6 — Self-Describing Function Registry

        Creates a ``CallableResourceNode`` with input/output schemas and
        optional trigger bindings. Enables AgentOS-style category collapse
        where every capability is discoverable through the same graph query.

        Args:
            function_id: Unique identifier for the function.
            name: Human-readable function name.
            resource_type: Type (MCP_TOOL, A2A_AGENT, INTERNAL_SKILL, AGENT_SKILL).
            description: What the function does.
            input_schema: JSON Schema for input parameters.
            output_schema: JSON Schema for return type.
            trigger_bindings: List of trigger binding dicts.
            endpoint: Optional endpoint URL.
            metadata_id: Associated metadata node ID.

        Returns:
            The created function registration dict.
        """
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        node_data = {
            "type": "callable_resource",
            "resource_type": resource_type,
            "name": name,
            "description": description,
            "endpoint": endpoint,
            "metadata_id": metadata_id or function_id,
            "input_schema": input_schema or {},
            "output_schema": output_schema or {},
            "trigger_bindings": trigger_bindings or [],
            "timestamp": ts,
            "importance_score": 0.5,
        }

        self.graph.add_node(function_id, **node_data)

        if self.backend:
            self.backend.execute(
                "MERGE (n:CallableResource {id: $id}) SET n += $props",
                {"id": function_id, "props": node_data},
            )

        logger.info(
            "[CONCEPT:ECO-4.6] Registered function '%s' (type=%s, triggers=%d)",
            name,
            resource_type,
            len(trigger_bindings or []),
        )
        return {"id": function_id, **node_data}

    def deregister_function(self, function_id: str) -> bool:
        """Remove a function registration from the KG.

        CONCEPT:ECO-4.6 — Self-Describing Function Registry

        Args:
            function_id: The function ID to remove.

        Returns:
            True if found and removed, False otherwise.
        """
        if function_id in self.graph:
            self.graph.remove_node(function_id)
            if self.backend:
                self.backend.execute(
                    "MATCH (n:CallableResource {id: $id}) DETACH DELETE n",
                    {"id": function_id},
                )
            logger.info("[CONCEPT:ECO-4.6] Deregistered function '%s'", function_id)
            return True
        return False

    def discover_functions(
        self,
        resource_type: str | None = None,
        trigger_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Discover self-describing functions with optional filtering.

        CONCEPT:ECO-4.6 — Self-Describing Function Registry

        Returns function metadata including input/output schemas and
        trigger bindings, enabling the caller to understand how to
        invoke each function without external documentation.

        Args:
            resource_type: Filter by type (e.g., 'MCP_TOOL', 'A2A_AGENT').
            trigger_type: Filter by trigger type (e.g., 'http', 'cron').

        Returns:
            List of function metadata dicts.
        """
        functions: list[dict[str, Any]] = []

        for nid, data in self.graph.nodes(data=True):
            if str(data.get("type", "")).lower() != "callable_resource":
                continue

            # Apply resource_type filter
            if (
                resource_type
                and data.get("resource_type", "").upper() != resource_type.upper()
            ):
                continue

            # Apply trigger_type filter
            if trigger_type:
                triggers = data.get("trigger_bindings", [])
                if not any(
                    t.get("trigger_type") == trigger_type
                    for t in (triggers if isinstance(triggers, list) else [])
                ):
                    continue

            functions.append({"id": nid, **data})

        return sorted(functions, key=lambda x: x.get("name", "").lower())
