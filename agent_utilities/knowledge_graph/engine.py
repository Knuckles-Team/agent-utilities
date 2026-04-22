#!/usr/bin/python
from __future__ import annotations

"""Unified Intelligence Graph Engine.

This module provides the high-level interface for querying the unified knowledge graph,
supporting structural Cypher queries, topological impact analysis, and hybrid search.
"""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

import networkx as nx  # type: ignore

from ..models.knowledge_graph import (
    CallableResourceNode,
    CritiqueNode,
    EpisodeNode,
    ExperimentNode,
    MemoryNode,
    OutcomeEvaluationNode,
    ProposedSkillNode,
    RegistryEdgeType,
    RegistryNodeType,
    SelfEvaluationNode,
    SpawnedAgentNode,
    SystemPromptNode,
    ToolMetadataNode,
)
from .backends import create_backend, get_active_backend
from .backends.base import GraphBackend

logger = logging.getLogger(__name__)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    import math

    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2, strict=False))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(a * a for a in v2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


class IntelligenceGraphEngine:
    """Engine for querying the unified intelligence graph (Agents, Tools, Code, Memory)."""

    _ACTIVE_ENGINE: IntelligenceGraphEngine | None = None

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        backend: GraphBackend | None = None,
        db_path: str | None = None,
    ):
        self.graph = graph
        self.backend: GraphBackend | None = None

        # Use provided backend, or check for an active one, or create one from factory
        if backend is not None:
            self.backend = backend
        else:
            active_backend = get_active_backend()
            if active_backend is not None:
                self.backend = active_backend
            elif db_path:
                self.backend = create_backend(db_path=db_path)
            else:
                self.backend = None

        # Auto-register as active if none exists to support singleton pattern
        if IntelligenceGraphEngine._ACTIVE_ENGINE is None:
            IntelligenceGraphEngine._ACTIVE_ENGINE = self

    @classmethod
    def get_active(cls) -> IntelligenceGraphEngine | None:
        """Retrieve the currently active engine instance."""
        return cls._ACTIVE_ENGINE

    @classmethod
    def set_active(cls, engine: IntelligenceGraphEngine):
        """Explicitly set the active engine instance."""
        cls._ACTIVE_ENGINE = engine

    def _get_allowed_columns(self, label: str) -> list[str]:
        """Get the list of allowed columns for a given node label from the schema."""
        try:
            from ..models.schema_definition import SCHEMA

            for node_def in SCHEMA.nodes:
                if node_def.name == label:
                    return list(node_def.columns.keys())
        except ImportError:
            pass
        return []

    def _serialize_node(self, node: Any, label: str | None = None) -> dict[str, Any]:
        """Serialize a Pydantic node for backend storage, handling Enums and JSON fields."""
        import json
        from enum import Enum

        data = node.model_dump() if hasattr(node, "model_dump") else dict(node)
        clean_data = {}

        # Define fields that Ladybug supports as native arrays
        ARRAY_FIELDS = [
            "capabilities",
            "tags",
            "tool_ids",
            "success_criteria_met",
            "embedding",
            "issues",
        ]

        # Filter by schema if label is provided
        allowed_cols = self._get_allowed_columns(label) if label else None

        for k, v in data.items():
            if v is None:
                continue
            if allowed_cols is not None and k not in allowed_cols:
                continue

            if isinstance(v, Enum):
                clean_data[k] = v.value
            elif isinstance(v, (dict, list)) and k not in ARRAY_FIELDS:
                clean_data[k] = json.dumps(v)
            else:
                clean_data[k] = v
        return clean_data

    def _get_set_clause(self, data: dict[str, Any], alias: str = "n") -> str:
        """Generate a SET clause for a Cypher query from a dictionary."""
        sets = []
        for k in data.keys():
            if k == "id":
                continue
            sets.append(f"{alias}.{k} = ${k}")
        return " SET " + ", ".join(sets) if sets else ""

    def _upsert_node(self, label: str, node_id: str, data: dict[str, Any]):
        """Perform an idempotent upsert of a node using MATCH/SET then CREATE."""
        if not self.backend:
            return

        # 1. Try to update existing
        set_clause = self._get_set_clause(data)
        update_query = f"MATCH (n:{label}) WHERE n.id = $id {set_clause} RETURN n.id"
        res = self.backend.execute(update_query, data)

        if not res:
            # 2. If not found, create
            cols = ", ".join([f"{k}: ${k}" for k in data.keys()])
            create_query = f"CREATE (n:{label} {{{cols}}})"
            self.backend.execute(create_query, data)

    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query against the persistent Graph store."""
        if not self.backend:
            logger.warning(
                "GraphBackend not initialized; using basic NetworkX fallback for Cypher query."
            )
            return self._query_nx_fallback(query, params)
        return self.backend.execute(query, params or {})

    def _query_nx_fallback(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Basic fallback to NetworkX for simple Cypher queries (MATCH ... RETURN)."""
        query_lower = query.lower()
        results = []

        # Case 1: Pull recent failures
        if "o:outcomeevaluation" in query_lower and "reward < 0.5" in query_lower:
            # MATCH (e:Episode)-[:PRODUCED_OUTCOME]->(o:OutcomeEvaluation) WHERE o.reward < 0.5
            for u, v, data in self.graph.edges(data=True):
                if data.get("type") == "PRODUCED_OUTCOME":
                    o_data = self.graph.nodes.get(v, {})
                    if (
                        o_data.get("type") == RegistryNodeType.OUTCOME_EVALUATION
                        and o_data.get("reward", 1.0) < 0.5
                    ):
                        e_data = self.graph.nodes.get(u, {})
                        results.append(
                            {"id": u, "description": e_data.get("description", "")}
                        )
            return results

        # Case 2: Frequent tool sequences (handled in propose_new_skill_from_experience)
        if (
            "e:episode" in query_lower
            and "o:outcomeevaluation" in query_lower
            and "reward >= 0.8" in query_lower
        ):
            logger.info("NX Fallback: Searching for successful episodes...")
            # MATCH (e:Episode)-[:PRODUCED_OUTCOME]->(o:OutcomeEvaluation) WHERE o.reward >= 0.8
            # MATCH (e)-[:USED_TOOL]->(t:ToolCall)
            for e_id, e_data in self.graph.nodes(data=True):
                # logger.info(f"Checking node {e_id} type={e_data.get('type')}")
                n_type = str(e_data.get("type", "")).lower()
                if n_type == "episode" or "episode" in n_type:
                    # Check reward
                    has_reward = False
                    for _, v, d in self.graph.out_edges(e_id, data=True):
                        logger.info(f"  Checking edge {e_id}->{v} type={d.get('type')}")
                        if d.get("type") == "PRODUCED_OUTCOME":
                            o_data = self.graph.nodes.get(v, {})
                            logger.info(
                                f"    Found outcome node {v} reward={o_data.get('reward')}"
                            )
                            if o_data.get("reward", 0.0) >= 0.8:
                                has_reward = True
                                break
                    if has_reward:
                        logger.info(f"NX Fallback: Found successful episode {e_id}")
                        for _, v, d in self.graph.out_edges(e_id, data=True):
                            if d.get("type") == "USED_TOOL":
                                t_data = self.graph.nodes.get(v, {})
                                results.append(
                                    {
                                        "ep_id": e_id,
                                        "tool": t_data.get("tool_name", ""),
                                        "ts": t_data.get("timestamp", ""),
                                    }
                                )
            logger.info(f"NX Fallback: Found {len(results)} tool calls.")
            return results

        return []

    def search_hybrid(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Perform a multi-faceted search across code, agents, and memory."""
        results = []
        query_lower = query.lower()

        # 1. Search NetworkX for name/ID matches (keyword-based)
        keywords = [k.strip() for k in query_lower.split() if len(k.strip()) > 1]
        if not keywords:
            keywords = [query_lower]

        for node_id, data in self.graph.nodes(data=True):
            name = str(data.get("name", "")).lower()
            desc = str(data.get("description", "")).lower()
            nid = str(node_id).lower()

            # Match if any keyword is in name, desc, or ID
            if any(k in nid or k in name or k in desc for k in keywords):
                result = dict(data)
                result["id"] = node_id
                results.append(result)

        logger.debug(f"Search hybrid for '{query}' found {len(results)} nodes.")
        # Sort by relevance (heuristic: name match first)
        results.sort(
            key=lambda x: any(k in str(x.get("name", "")).lower() for k in keywords),
            reverse=True,
        )

        return results[:top_k]

    def search_memories(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search specifically for memory nodes."""
        results = self.search_hybrid(query, top_k=50)
        return [r for r in results if r.get("type") == RegistryNodeType.MEMORY][:top_k]

    def query_impact(self, symbol_or_file: str) -> list[dict[str, Any]]:
        """Calculate the topological impact set for a code entity."""
        target_id = symbol_or_file
        if target_id not in self.graph:
            # Try fuzzy match by name
            for node, data in self.graph.nodes(data=True):
                if data.get("name") == symbol_or_file:
                    target_id = node
                    break

        if target_id not in self.graph:
            return []

        # Ancestors in our graph (where edges mean 'depends on' or 'contains') are the impact set
        impact_nodes = nx.ancestors(self.graph, target_id)
        return [{"id": n, **self.graph.nodes[n]} for n in impact_nodes]

    def find_path(self, source: str, target: str) -> list[str]:
        """Find the shortest logical path between two nodes."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []

    def get_shortest_path(self, source: str, target: str) -> list[str]:
        """Alias for find_path."""
        return self.find_path(source, target)

    def get_agent_tools(self, agent_name: str) -> list[str]:
        """Get all tools provided by a specific agent."""
        tools = []
        if agent_name in self.graph:
            for u, v, data in self.graph.out_edges(agent_name, data=True):
                if data.get("type") == RegistryEdgeType.PROVIDES:
                    tools.append(v.replace("tool:", ""))
        return tools

    def find_agent_for_tool(self, tool_name: str) -> list[str]:
        """Find all agents that provide a specific tool."""
        agents = []
        tool_id = f"tool:{tool_name}"
        if tool_id not in self.graph:
            return []
        for u, v, data in self.graph.in_edges(tool_id, data=True):
            if data.get("type") == RegistryEdgeType.PROVIDES:
                agents.append(u)
        return agents

    def add_memory(
        self,
        content: str,
        name: str = "",
        category: str = "general",
        tags: list[str] | None = None,
    ) -> str:
        """Add a new memory to the unified graph."""
        memory_id = f"mem:{uuid.uuid4().hex[:8]}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = MemoryNode(
            id=memory_id,
            name=name or f"Memory {timestamp}",
            description=content,
            timestamp=timestamp,
            category=category,
            tags=tags or [],
        )

        self.graph.add_node(node.id, **node.model_dump())

        if self.backend:
            data = self._serialize_node(node, label="Memory")
            self._upsert_node("Memory", node.id, data)

        return memory_id

    def delete_memory(self, memory_id: str):
        """Delete a memory from the graph."""
        if memory_id in self.graph:
            self.graph.remove_node(memory_id)
        if self.backend:
            self.backend.execute(
                "MATCH (n {id: $id}) DETACH DELETE n", {"id": memory_id}
            )

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """Retrieve a specific memory by ID from the graph."""
        # Check NetworkX first (in-memory)
        if memory_id in self.graph:
            return {"id": memory_id, **self.graph.nodes[memory_id]}
        # Fallback to persistent backend
        if self.backend:
            results = self.backend.execute(
                "MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id}
            )
            if results:
                return results[0].get("m", results[0])
        return None

    def update_memory(self, memory_id: str, **kwargs):
        """Update properties of an existing memory."""
        if memory_id in self.graph:
            self.graph.nodes[memory_id].update(kwargs)
        if self.backend:
            self.backend.execute(
                "MATCH (n {id: $id}) SET n += $props",
                {"id": memory_id, "props": kwargs},
            )

    def link_nodes(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ):
        """Create a relationship between two nodes in the graph."""
        props = properties or {}
        if source_id in self.graph and target_id in self.graph:
            self.graph.add_edge(source_id, target_id, type=rel_type, **props)

        if self.backend:
            query = (
                f"MATCH (s {{id: $sid}}), (t {{id: $tid}}) "
                f"MERGE (s)-[r:{rel_type}]->(t) SET r += $props"
            )
            self.backend.execute(
                query, {"sid": source_id, "tid": target_id, "props": props}
            )

    def add_memory_node(self, memory: MemoryNode):
        """Add a MemoryNode object to the graph."""
        self.graph.add_node(memory.id, **memory.model_dump())
        if self.backend:
            data = self._serialize_node(memory, label="Memory")
            self._upsert_node("Memory", memory.id, data)

    def get_memory_node(self, memory_id: str) -> MemoryNode | None:
        """Retrieve a MemoryNode object by ID."""
        data = self.get_memory(memory_id)
        if data:
            return MemoryNode(
                **{k: v for k, v in data.items() if not k.startswith("_")}
            )
        return None

    def update_memory_node(self, memory_id: str, memory: MemoryNode):
        """Update a memory using a MemoryNode object."""
        self.update_memory(memory_id, **memory.model_dump(exclude={"id"}))

    def delete_memory_node(self, memory_id: str):
        """Delete a memory node."""
        self.delete_memory(memory_id)

    # --- Enhanced Memory & Ingestion Tools ---

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
        self.graph.add_node(node.id, **node.model_dump())
        if self.backend:
            data = self._serialize_node(node, label="Episode")
            self._upsert_node("Episode", node.id, data)
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
                self.graph.add_node(metadata.id, **metadata.model_dump())
                if self.backend:
                    m_data = self._serialize_node(metadata, label="ToolMetadata")
                    self._upsert_node("ToolMetadata", meta_id, m_data)

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
                self.graph.add_node(resource.id, **resource.model_dump())
                if self.backend:
                    r_data = self._serialize_node(resource, label="CallableResource")
                    self._upsert_node("CallableResource", res_id, r_data)

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
        self.graph.add_node(resource.id, **resource.model_dump())

        if self.backend:
            self._upsert_node(
                "ToolMetadata", meta_id, self._serialize_node(metadata, "ToolMetadata")
            )
            self._upsert_node(
                "CallableResource",
                agent_id,
                self._serialize_node(resource, "CallableResource"),
            )
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
        self.graph.add_node(resource.id, **resource.model_dump())

        if self.backend:
            self._upsert_node(
                "ToolMetadata", meta_id, self._serialize_node(metadata, "ToolMetadata")
            )
            self._upsert_node(
                "CallableResource",
                skill_id,
                self._serialize_node(resource, "CallableResource"),
            )
            self.backend.execute(
                "MATCH (r:CallableResource), (m:ToolMetadata) WHERE r.id = $rid AND m.id = $mid MERGE (r)-[:HAS_METADATA]->(m)",
                {"rid": skill_id, "mid": meta_id},
            )

    # --- Discovery & Retrieval Tools ---

    def find_relevant_callable_resources(
        self, task_description: str, required_caps: list[str] | None = None
    ) -> list[dict[str, Any]]:
        if required_caps is None:
            required_caps = []
        """Find the most relevant Tools, Agents, or Skills for a given task."""
        # Hybrid search: semantic similarity + capability filtering
        candidates = self.search_hybrid(task_description, top_k=20)
        filtered = []
        for c in candidates:
            c_type = str(c.get("type", "")).lower()
            if "callable_resource" in c_type:
                # Check caps in linked metadata if backend available
                if self.backend:
                    res = self.query_cypher(
                        "MATCH (r:CallableResource {id: $id})-[:HAS_METADATA]->(m) RETURN m.capabilities as caps",
                        {"id": c["id"]},
                    )
                    if res and res[0].get("caps"):
                        caps_raw = res[0]["caps"]
                        # Robust parsing for list or string representation
                        if isinstance(caps_raw, str):
                            import ast

                            try:
                                caps = ast.literal_eval(caps_raw)
                            except Exception:
                                caps = [caps_raw]
                        else:
                            caps = caps_raw

                        if not required_caps or all(
                            cap in caps for cap in required_caps
                        ):
                            filtered.append(c)
                    else:
                        # Fallback to model capabilities if available or if none required
                        if not required_caps:
                            filtered.append(c)
                else:
                    filtered.append(c)
        return filtered

    def list_callable_resources(self) -> list[dict[str, Any]]:
        """List all callable resources (MCP tools, A2A agents, skills)."""
        resources = []
        for n, data in self.graph.nodes(data=True):
            if data.get("type") == RegistryNodeType.CALLABLE_RESOURCE:
                resources.append({"id": n, **data})
        return resources

    def retrieve_orthogonal_context(
        self,
        query: str,
        views: list[str] | None = None,
    ) -> dict[str, Any]:
        if views is None:
            views = ["semantic", "temporal", "causal", "entity"]
        """Perform policy-guided retrieval across orthogonal MAGMA views."""
        context: dict[str, Any] = {"query": query, "views": {}}
        if "semantic" in views:
            context["views"]["semantic"] = self.search_hybrid(query, top_k=5)
        if "temporal" in views:
            context["views"]["temporal"] = self.query_cypher(
                "MATCH (e:Episode) RETURN e ORDER BY e.timestamp DESC LIMIT 5"
            )
        if "causal" in views:
            context["views"]["causal"] = self.query_cypher(
                "MATCH (r:ReasoningTrace)-[:CAUSED_BY]->(p) RETURN r, p LIMIT 5"
            )
        if "entity" in views:
            context["views"]["entity"] = self.query_cypher(
                "MATCH (e:Entity) WHERE e.id CONTAINS $q OR e.entity_type CONTAINS $q RETURN e",
                {"q": query},
            )
        return context

    def find_relevant_policies(self, query: str) -> list[dict[str, Any]]:
        """Search for policies that apply to the current query context."""
        # Hybrid search for policies
        results = self.query_cypher(
            "MATCH (p:Policy) WHERE p.name CONTAINS $q OR p.description CONTAINS $q RETURN p",
            {"q": query},
        )
        # Note: If LadybugDB supports vector search, we would use it here as well
        return [r.get("p", r) for r in results]

    def find_relevant_processes(self, query: str) -> list[dict[str, Any]]:
        """Search for process flows that match the current query goal."""
        results = self.query_cypher(
            "MATCH (f:ProcessFlow) WHERE f.goal CONTAINS $q OR f.name CONTAINS $q RETURN f",
            {"q": query},
        )
        return [r.get("f", r) for r in results]

    # --- Agent Spawning Tools ---

    def spawn_specialized_agent(
        self,
        task_description: str,
        tool_ids: list[str],
        parent_task_id: str | None = None,
    ) -> str:
        """Spawn a specialized sub-agent with a curated toolset and composed prompt."""
        agent_id = f"spawn:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Intelligent prompt composition: Find relevant base prompts in the graph
        base_prompts = self.query_cypher(
            "MATCH (p:SystemPrompt) WHERE p.tags CONTAINS $tag RETURN p.content as content LIMIT 1",
            {"tag": "agent-base"},
        )
        base_text = (
            base_prompts[0]["content"]
            if base_prompts
            else "You are a specialized agent."
        )

        prompt = f"{base_text}\nTask: {task_description}\nAvailable tools: {', '.join(tool_ids)}"

        node = SpawnedAgentNode(
            id=agent_id,
            name=f"Agent {ts}",
            system_prompt=prompt,
            tool_ids=tool_ids,
            created_at=ts,
            parent_task_id=parent_task_id,
            importance_score=0.9,
        )
        self.graph.add_node(node.id, **node.model_dump())
        if self.backend:
            data = self._serialize_node(node, label="SpawnedAgent")
            self._upsert_node("SpawnedAgent", agent_id, data)
            for tid in tool_ids:
                # Use explicit node match for resources
                self.backend.execute(
                    "MATCH (a:SpawnedAgent {id: $aid}), (t:CallableResource {id: $tid}) MERGE (a)-[:USES]->(t)",
                    {"aid": agent_id, "tid": tid},
                )
        return agent_id

    # --- Self-Improvement Tools (Lightning style) ---

    def record_outcome(
        self,
        episode_id: str,
        reward: float,
        feedback: str,
        success_criteria_met: list[str] | None = None,
    ):
        """Record the outcome and reward for an episode (Lightning step 1)."""
        eval_id = f"eval:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        node = OutcomeEvaluationNode(
            id=eval_id,
            name=f"Eval {episode_id}",
            reward=reward,
            success_criteria_met=success_criteria_met or [],
            feedback_text=feedback,
            timestamp=ts,
        )
        # Always add to in-memory graph
        self.graph.add_node(node.id, **node.model_dump())

        if self.backend:
            data = self._serialize_node(node, label="OutcomeEvaluation")
            self._upsert_node("OutcomeEvaluation", eval_id, data)
            # Ladybug requires labels for relationship creation
            label = (
                "Episode"
                if episode_id.startswith("ep:") or episode_id.startswith("run:")
                else "ReasoningTrace"
            )
            self.backend.execute(
                f"MATCH (e:{label}), (o:OutcomeEvaluation) WHERE e.id = $eid AND o.id = $oid MERGE (e)-[:PRODUCED_OUTCOME]->(o)",
                {"eid": episode_id, "oid": eval_id},
            )

        # Link in NetworkX as well
        # Note: we don't know the label in NX nodes reliably without checking 'type' property
        if episode_id in self.graph:
            self.graph.add_edge(episode_id, eval_id, type="PRODUCED_OUTCOME")

        return eval_id

    def record_self_evaluation(
        self, episode_id: str, confidence: float, difficulty: float
    ):
        """Record the agent's internal self-evaluation (confidence calibration)."""
        eval_id = f"self_eval:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        node = SelfEvaluationNode(
            id=eval_id,
            name=f"Self-Eval {episode_id}",
            confidence_calibration=confidence,
            task_difficulty=difficulty,
            timestamp=ts,
        )
        # Always add to in-memory graph
        self.graph.add_node(node.id, **node.model_dump())

        if self.backend:
            data = self._serialize_node(node, label="SelfEvaluation")
            self._upsert_node("SelfEvaluation", eval_id, data)
            self.backend.execute(
                "MATCH (e:Episode), (s:SelfEvaluation) WHERE e.id = $eid AND s.id = $sid MERGE (e)-[:SELF_REFLECTS_ON]->(s)",
                {"eid": episode_id, "sid": eval_id},
            )

        if episode_id in self.graph:
            self.graph.add_edge(episode_id, eval_id, type="SELF_REFLECTS_ON")

        return eval_id

    def record_experiment(
        self, name: str, variants: list[str], status: str = "running"
    ):
        """Record a new A/B experiment for prompt or tool variants."""
        exp_id = f"exp:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        node = ExperimentNode(
            id=exp_id,
            name=name,
            status=status,
            timestamp=ts,
        )
        # Always add to in-memory graph
        self.graph.add_node(node.id, **node.model_dump())
        self.graph.nodes[node.id]["variants"] = variants

        if self.backend:
            data = self._serialize_node(node, label="Experiment")
            data["variants"] = variants
            self._upsert_node("Experiment", exp_id, data)
        return exp_id

    def generate_critique(self, reasoning_trace_id: str, textual_gradient: str) -> str:
        """Generate a critique (textual gradient) for a reasoning trace (Lightning step 2)."""
        crit_id = f"crit:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        node = CritiqueNode(
            id=crit_id,
            name=f"Critique {ts}",
            textual_gradient=textual_gradient,
            timestamp=ts,
        )
        # Always add to in-memory graph
        self.graph.add_node(node.id, **node.model_dump())

        if self.backend:
            data = self._serialize_node(node, label="Critique")
            self._upsert_node("Critique", crit_id, data)
            self.backend.execute(
                "MATCH (r:ReasoningTrace), (c:Critique) WHERE r.id = $rid AND c.id = $cid MERGE (r)-[:GENERATED_CRITIQUE]->(c)",
                {"rid": reasoning_trace_id, "cid": crit_id},
            )

        if reasoning_trace_id in self.graph:
            self.graph.add_edge(reasoning_trace_id, crit_id, type="GENERATED_CRITIQUE")

        return crit_id

    def optimize_prompt(self, prompt_id: str, critique_id: str) -> str:
        """Create a new optimized version of a system prompt based on a critique (Lightning step 3)."""
        new_id = f"prompt:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if self.backend:
            old_prompt = self.query_cypher(
                "MATCH (p:SystemPrompt {id: $id}) RETURN p.content as content",
                {"id": prompt_id},
            )
            critique = self.query_cypher(
                "MATCH (c:Critique {id: $id}) RETURN c.textual_gradient as grad",
                {"id": critique_id},
            )

            content = old_prompt[0]["content"] if old_prompt else "Default prompt"
            grad = critique[0]["grad"] if critique else "Improve clarity"

            new_content = f"{content}\n# Optimized based on: {grad}"

            node = SystemPromptNode(
                id=new_id,
                name=f"Optimized {ts}",
                content=new_content,
                version="v-next",
                source="REFINED",
                timestamp=ts,
            )
            data = self._serialize_node(node, label="SystemPrompt")
            self._upsert_node("SystemPrompt", new_id, data)
            self.backend.execute(
                "MATCH (old:SystemPrompt), (new:SystemPrompt) WHERE old.id = $oid AND new.id = $nid MERGE (new)-[:EVOLVED_FROM]->(old)",
                {"oid": prompt_id, "nid": new_id},
            )
            self.backend.execute(
                "MATCH (c:Critique), (p:SystemPrompt) WHERE c.id = $cid AND p.id = $pid MERGE (c)-[:LED_TO]->(p)",
                {"cid": critique_id, "pid": new_id},
            )
        return new_id

    def run_self_improvement_cycle(self):
        """Autonomous loop for background optimization (Lightning trainer)."""
        # 1. Pull recent failures (low reward)
        failures = self.query_cypher(
            "MATCH (e:Episode)-[:PRODUCED_OUTCOME]->(o:OutcomeEvaluation) WHERE o.reward < 0.5 RETURN e.id as id, e.description as description LIMIT 5"
        )
        logger.info(f"Self-improvement cycle: found {len(failures)} failures.")
        for fail in failures:
            logger.info(f"Processing failure: {fail['id']}")
            # 2. Generate pseudo-critique if missing
            crit_id = self.generate_critique(
                fail["id"],
                f"Improve the following based on failure: {fail.get('description', '')}",
            )

            # 3. Optimize linked prompt
            # Step-by-step traversal for robustness
            agent_res = self.query_cypher(
                "MATCH (e {id: $eid})-[:EXECUTED_BY]->(a) RETURN a.id as id LIMIT 1",
                {"eid": fail["id"]},
            )
            prompt = None
            if agent_res:
                aid = agent_res[0]["id"]
                prompt_res = self.query_cypher(
                    "MATCH (a {id: $aid})-[:USES]->(p:SystemPrompt) RETURN p.id as id LIMIT 1",
                    {"aid": aid},
                )
                if prompt_res:
                    prompt = prompt_res

            if prompt:
                logger.info(
                    f"Optimizing prompt {prompt[0]['id']} for failure {fail['id']}"
                )
                self.optimize_prompt(prompt[0]["id"], crit_id)
            else:
                logger.warning(
                    f"No prompt linked to failure {fail['id']} via Episode->Agent->Prompt path."
                )

        # 4. Propose new skills
        new_skill_id = self.propose_new_skill_from_experience()
        if new_skill_id:
            logger.info(f"Proposed new skill: {new_skill_id}")

        logger.info("Self-improvement cycle completed.")

    def propose_new_skill_from_experience(self) -> str | None:
        """Analyze successful trajectories and propose a new skill node."""
        # Removed if not self.backend return to allow NX fallback

        # Strategy A: Frequent Tool Sequences
        # Fetch successful episodes and their tool calls in order
        query = """
        MATCH (e:Episode)-[:PRODUCED_OUTCOME]->(o:OutcomeEvaluation)
        WHERE o.reward >= 0.8
        MATCH (e)-[:USED_TOOL]->(t:ToolCall)
        RETURN e.id as ep_id, t.tool_name as tool, t.timestamp as ts
        ORDER BY ep_id, ts
        """
        results = self.query_cypher(query)

        episodes: dict[str, list[str]] = {}
        for row in results:
            ep_id = row["ep_id"]
            if ep_id not in episodes:
                episodes[ep_id] = []
            episodes[ep_id].append(row["tool"])

        # Count sequence frequency
        sequences: dict[tuple[str, ...], int] = {}
        for tools in episodes.values():
            if len(tools) >= 2:
                # Use window of 2-3 tools
                for i in range(len(tools) - 1):
                    seq = tuple(tools[i : i + 2])
                    sequences[seq] = sequences.get(seq, 0) + 1

        # Find most frequent sequence
        if not sequences:
            return None

        best_seq = max(sequences, key=lambda k: sequences[k])
        freq = sequences[best_seq]

        if freq < 3:  # Threshold for "repeated a lot"
            return None

        # Create ProposedSkillNode
        skill_id = f"skill_prop:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        name = f"Sequence: {' -> '.join(best_seq)}"
        node = ProposedSkillNode(
            id=skill_id,
            name=name,
            code_content=f"# Auto-generated skill for repeated sequence: {' -> '.join(best_seq)}\n"
            f"def frequent_sequence_skill(ctx, **kwargs):\n"
            f"    # This skill automates the frequent sequence detected in the KG\n"
            f"    pass",
            frontmatter={
                "name": name.lower().replace(" ", "_").replace(":", ""),
                "description": f"Automated skill for the frequent tool sequence: {', '.join(best_seq)}",
                "tools": list(best_seq),
                "frequency": freq,
            },
            timestamp=ts,
        )

        # Always add to in-memory graph
        self.graph.add_node(node.id, **node.model_dump())
        return node.id

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
        search_results = self.search_hybrid(query, top_k=max_nodes * 2)

        # 2. Build initial node set
        seed_ids = [r["id"] for r in search_results]
        subgraph = self.graph.subgraph(seed_ids).copy()

        # 3. Expand to include neighbors with high centrality
        # In MultiDiGraph, neighbors() returns successors. We might want both.
        for node_id in list(subgraph.nodes):
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
                        try:
                            c_data[k] = json.loads(c_data[k])
                        except Exception:
                            pass
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
                        try:
                            c_data[k] = json.loads(c_data[k])
                        except Exception:
                            pass
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

    def generate_mermaid_graph(
        self,
        query: str | None = None,
        max_nodes: int = 50,
        title: str = "Knowledge Graph",
    ) -> str:
        """Generate a Mermaid visualization for a portion of the graph."""
        from ..mermaid import FlowchartBuilder

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
        builder.lines.append("  classDef episode fill:#2e7d32,stroke:#1b5e20,color:#fff")
        builder.lines.append("  classDef memory fill:#1565c0,stroke:#0d47a1,color:#fff")
        builder.lines.append("  classDef agent fill:#f57c00,stroke:#e65100,color:#fff")

        return builder.render()


@dataclass
class FocusedSubgraph:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    summary: str
    query: str

    def to_mermaid(self) -> str:
        """Convert this subgraph to a Mermaid diagram."""
        from ..mermaid import FlowchartBuilder

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


# Alias for backward compatibility
RegistryGraphEngine = IntelligenceGraphEngine
