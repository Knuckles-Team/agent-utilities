#!/usr/bin/python
# coding: utf-8
"""Unified Intelligence Graph Engine.

This module provides the high-level interface for querying the unified knowledge graph,
supporting structural Cypher queries, topological impact analysis, and hybrid search.
"""

import logging
import time
import uuid
from typing import List, Dict, Any, Optional
import networkx as nx

from .backends.base import GraphBackend
from .backends import create_backend, get_active_backend
from ..models.knowledge_graph import (
    RegistryEdgeType,
    RegistryNodeType,
    MemoryNode,
    EpisodeNode,
    CallableResourceNode,
    ToolMetadataNode,
    SpawnedAgentNode,
    SystemPromptNode,
    OutcomeEvaluationNode,
    CritiqueNode,
)

logger = logging.getLogger(__name__)


class IntelligenceGraphEngine:
    """Engine for querying the unified intelligence graph (Agents, Tools, Code, Memory)."""

    _ACTIVE_ENGINE: Optional["IntelligenceGraphEngine"] = None

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        backend: Optional[GraphBackend] = None,
        db_path: Optional[str] = None,
    ):
        self.graph = graph

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
    def get_active(cls) -> Optional["IntelligenceGraphEngine"]:
        """Retrieve the currently active engine instance."""
        return cls._ACTIVE_ENGINE

    @classmethod
    def set_active(cls, engine: "IntelligenceGraphEngine"):
        """Explicitly set the active engine instance."""
        cls._ACTIVE_ENGINE = engine

    def _get_allowed_columns(self, label: str) -> List[str]:
        """Get the list of allowed columns for a given node label from the schema."""
        try:
            from ..models.schema_definition import SCHEMA

            for node_def in SCHEMA.nodes:
                if node_def.name == label:
                    return list(node_def.columns.keys())
        except ImportError:
            pass
        return []

    def _serialize_node(self, node: Any, label: Optional[str] = None) -> Dict[str, Any]:
        """Serialize a Pydantic node for backend storage, handling Enums and JSON fields."""
        from enum import Enum
        import json

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

    def _get_set_clause(self, data: Dict[str, Any], alias: str = "n") -> str:
        """Generate a SET clause for a Cypher query from a dictionary."""
        sets = []
        for k in data.keys():
            if k == "id":
                continue
            sets.append(f"{alias}.{k} = ${k}")
        return " SET " + ", ".join(sets) if sets else ""

    def _upsert_node(self, label: str, node_id: str, data: Dict[str, Any]):
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
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query against the persistent Graph store."""
        if not self.backend:
            logger.warning("GraphBackend not initialized; Cypher queries unavailable.")
            return []
        return self.backend.execute(query, params or {})

    def search_hybrid(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
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

    def search_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search specifically for memory nodes."""
        results = self.search_hybrid(query, top_k=50)
        return [r for r in results if r.get("type") == RegistryNodeType.MEMORY][:top_k]

    def query_impact(self, symbol_or_file: str) -> List[Dict[str, Any]]:
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

    def find_path(self, source: str, target: str) -> List[str]:
        """Find the shortest logical path between two nodes."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []

    def get_shortest_path(self, source: str, target: str) -> List[str]:
        """Alias for find_path."""
        return self.find_path(source, target)

    def get_agent_tools(self, agent_name: str) -> List[str]:
        """Get all tools provided by a specific agent."""
        tools = []
        if agent_name in self.graph:
            for u, v, data in self.graph.out_edges(agent_name, data=True):
                if data.get("type") == RegistryEdgeType.PROVIDES:
                    tools.append(v.replace("tool:", ""))
        return tools

    def find_agent_for_tool(self, tool_name: str) -> List[str]:
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
        tags: List[str] = None,
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

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
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

    # --- Enhanced Memory & Ingestion Tools ---

    def ingest_episode(
        self, content: str, source: str = "chat", timestamp: Optional[str] = None
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
        tools: List[Dict[str, Any]],
        resources: Optional[Dict[str, Any]] = None,
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

    def ingest_a2a_agent_card(self, url: str, card: Dict[str, Any]):
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
        self, skill_file_path: str, frontmatter: Dict[str, Any], content: str
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
        self, task_description: str, required_caps: List[str] = []
    ) -> List[Dict[str, Any]]:
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

    def retrieve_orthogonal_context(
        self,
        query: str,
        views: List[str] = ["semantic", "temporal", "causal", "entity"],
    ) -> Dict[str, Any]:
        """Perform policy-guided retrieval across orthogonal MAGMA views."""
        context = {"query": query, "views": {}}
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

    # --- Agent Spawning Tools ---

    def spawn_specialized_agent(
        self,
        task_description: str,
        tool_ids: List[str],
        parent_task_id: Optional[str] = None,
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

    def record_outcome(self, episode_id: str, reward: float, feedback: str):
        """Record the outcome and reward for an episode (Lightning step 1)."""
        eval_id = f"eval:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        node = OutcomeEvaluationNode(
            id=eval_id,
            name=f"Eval {episode_id}",
            reward=reward,
            feedback_text=feedback,
            timestamp=ts,
        )
        if self.backend:
            data = self._serialize_node(node, label="OutcomeEvaluation")
            self._upsert_node("OutcomeEvaluation", eval_id, data)
            # Ladybug requires labels for relationship creation
            label = "Episode" if episode_id.startswith("ep:") else "ReasoningTrace"
            self.backend.execute(
                f"MATCH (e:{label}), (o:OutcomeEvaluation) WHERE e.id = $eid AND o.id = $oid MERGE (e)-[:PRODUCED_OUTCOME]->(o)",
                {"eid": episode_id, "oid": eval_id},
            )

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
        if self.backend:
            data = self._serialize_node(node, label="Critique")
            self._upsert_node("Critique", crit_id, data)
            self.backend.execute(
                "MATCH (r:ReasoningTrace), (c:Critique) WHERE r.id = $rid AND c.id = $cid MERGE (r)-[:GENERATED_CRITIQUE]->(c)",
                {"rid": reasoning_trace_id, "cid": crit_id},
            )
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
            "MATCH (e:Episode)-[:PRODUCED_OUTCOME]->(o:OutcomeEvaluation) WHERE o.reward < 0.0 RETURN e.id as id, e.description as description LIMIT 5"
        )
        for fail in failures:
            # 2. Generate pseudo-critique if missing (placeholder for LLM)
            crit_id = self.generate_critique(
                fail["id"], f"Failure in episode: {fail['description']}"
            )
            # 3. Optimize linked prompt
            prompt = self.query_cypher(
                "MATCH (a:SpawnedAgent)-[:USES]->(p:SystemPrompt) RETURN p.id as id LIMIT 1"
            )
            if prompt:
                self.optimize_prompt(prompt[0]["id"], crit_id)
        logger.info("Self-improvement cycle completed.")


# Alias for backward compatibility
RegistryGraphEngine = IntelligenceGraphEngine
