from __future__ import annotations

"""Query and search mixin for IntelligenceGraphEngine.

Extracted from engine.py for maintainability. Contains all read-only
query, search, and retrieval methods.
"""


import typing

if typing.TYPE_CHECKING:
    from .._engine_protocol import _EngineProtocol

    _Base = _EngineProtocol
else:
    _Base = object


import logging
from typing import Any

import networkx as nx

from ...models.knowledge_graph import RegistryEdgeType, RegistryNodeType

logger = logging.getLogger(__name__)


class QueryMixin(_Base):
    """Query and search capabilities for the KG engine."""

    def query_cypher(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        clearance_level: int = 999,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query against the persistent Graph store.

        Enterprise ABAC/RBAC Middleware Interceptor:
        Queries are automatically scoped by the user's/agent's security clearance.
        Nodes with a requiresClassification > clearance_level are omitted.
        """
        if params is None:
            params = {}

        # Inject clearance level for the backend parser to utilize
        params["_clearance_level"] = clearance_level
        logger.debug(
            "RBAC Interceptor: Executing query with clearance level %d", clearance_level
        )

        if not self.backend:
            logger.warning(
                "GraphBackend not initialized; using basic NetworkX fallback for Cypher query."
            )
            return self._query_nx_fallback(query, params, clearance_level)
        return self.backend.execute(query, params)

    def _query_nx_fallback(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        clearance_level: int = 999,
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

    def _search_keyword(
        self, query: str, top_k: int = 10, clearance_level: int = 999
    ) -> list[dict[str, Any]]:
        """Perform a multi-faceted search across code, agents, and memory using keywords."""
        results = []
        query_lower = query.lower()

        # 1. Prepare keywords
        keywords = [k.strip() for k in query_lower.split() if len(k.strip()) > 1]
        if not keywords:
            keywords = [query_lower]

        # 2. Search Backend if available
        if self.backend:
            # Simple keyword search across all nodes in backend
            q = []
            params = {}
            for i, k in enumerate(keywords):
                q.append(
                    f"(toLower(n.name) CONTAINS $k{i} OR toLower(n.description) CONTAINS $k{i} OR toLower(n.id) CONTAINS $k{i})"
                )
                params[f"k{i}"] = k

            where_clause = " OR ".join(q)
            if where_clause:
                query_str = f"MATCH (n) WHERE {where_clause} RETURN n"
                try:
                    res = self.backend.execute(query_str, params)
                    for row in res:
                        node = row.get("n", {})
                        if isinstance(node, dict):
                            req_class = node.get("requiresClassification", 0)
                            if (
                                isinstance(req_class, int)
                                and req_class > clearance_level
                            ):
                                continue
                            results.append(node)
                except Exception as e:
                    logger.debug(f"Backend keyword search failed: {e}")

        # 2. Search NetworkX for name/ID matches
        if not results:
            for node_id, data in self.graph.nodes(data=True):
                # RBAC Enforcement Fallback
                req_class = data.get("requiresClassification", 0)
                if isinstance(req_class, int) and req_class > clearance_level:
                    continue

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

    def search_hybrid(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Perform a multi-faceted search using Hybrid GraphRAG."""
        return self.hybrid_retriever.retrieve_hybrid(query, context_window=top_k)

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

    def run_inference(self) -> int:
        """Run standard inference rules over the graph to derive new facts."""
        return self.inference_engine.run_inference()

    def find_relevant_callable_resources(
        self, task_description: str, required_caps: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Find the most relevant Tools, Agents, or Skills for a given task.

        CONCEPT:ECO-4.6 enhanced: searches ALL resource types (tools,
        agents, skills, memory) in a single hybrid query for unified
        capability discovery (AgentOS-style category collapse).
        """
        if required_caps is None:
            required_caps = []
        # Hybrid search: semantic similarity + capability filtering
        # Search across all node types, not just callable_resource
        candidates = self.search_hybrid(task_description, top_k=30)
        filtered = []

        # Resource types that constitute "callable" capabilities
        _callable_types = {
            "callable_resource",
            "tool",
            "skill",
            "agent",
            "mcp_tool",
            "a2a_agent",
            "internal_skill",
            "agent_skill",
        }

        for c in candidates:
            c_type = str(c.get("type", "")).lower()
            c_resource_type = str(c.get("resource_type", "")).lower()

            # Match callable resources AND their component types
            if c_type not in _callable_types and c_resource_type not in _callable_types:
                continue

            # Check caps in linked metadata if backend available
            if self.backend and required_caps:
                res = self.query_cypher(
                    "MATCH (r {id: $id})-[:HAS_METADATA]->(m) RETURN m.capabilities as caps",
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

                    if all(cap in caps for cap in required_caps):
                        filtered.append(c)
                else:
                    # Fallback if no metadata
                    if not required_caps:
                        filtered.append(c)
            else:
                filtered.append(c)
        return filtered

    def discover_all_capabilities(
        self,
        query: str | None = None,
        resource_types: list[str] | None = None,
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """Discover all capabilities across all resource types in a unified view.

        CONCEPT:ECO-4.6 — Self-Describing Function Registry

        Returns a unified function-level view of all discoverable
        capabilities, inspired by AgentOS's category collapse pattern.
        Each entry includes self-describing metadata (input/output
        schemas, trigger bindings) when available.

        Args:
            query: Optional semantic query to filter by relevance.
            resource_types: Optional filter by resource types.
            top_k: Maximum results to return.

        Returns:
            List of unified capability dicts with ``fn_namespace`` field
            matching AgentOS's ``fn namespace::action(args)`` pattern.
        """
        capabilities: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        # If query provided, use hybrid search for relevance ordering
        if query:
            results = self.search_hybrid(query, top_k=top_k * 2)
        else:
            results = [{"id": nid, **data} for nid, data in self.graph.nodes(data=True)]

        for r in results:
            rid = r.get("id", "")
            if rid in seen_ids:
                continue

            r_type = str(r.get("type", "")).lower()
            r_resource = str(r.get("resource_type", "")).lower()

            # Filter to capability-bearing node types
            _cap_types = {
                "callable_resource",
                "tool",
                "skill",
                "agent",
                "tool_metadata",
                "spawned_agent",
            }
            if r_type not in _cap_types and r_resource not in _cap_types:
                continue

            # Apply resource_type filter
            if resource_types:
                if r_resource not in [rt.lower() for rt in resource_types]:
                    continue

            seen_ids.add(rid)

            # Build unified fn namespace (AgentOS pattern)
            namespace = r_resource or r_type
            action = r.get("name", rid).replace(" ", "_").lower()
            fn_namespace = f"fn {namespace}::{action}"

            capabilities.append(
                {
                    "id": rid,
                    "fn_namespace": fn_namespace,
                    "name": r.get("name", rid),
                    "description": r.get("description", ""),
                    "resource_type": r_resource or r_type,
                    "input_schema": r.get("input_schema", {}),
                    "output_schema": r.get("output_schema", {}),
                    "trigger_bindings": r.get("trigger_bindings", []),
                    "endpoint": r.get("endpoint"),
                    "_score": r.get("_score", 0.0),
                }
            )

            if len(capabilities) >= top_k:
                break

        return capabilities

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
        """Perform policy-guided retrieval across orthogonal MAGMA views.

        V1 views: ``semantic``, ``temporal``, ``causal``, ``entity``.
        V2 views (stubs): ``place``, ``epistemic`` — see
        ``docs/KG_V2_DESIGN.md`` §5. These currently return empty lists;
        the full Cypher-backed implementations land in a follow-up.
        """
        if views is None:
            # V1 default preserved for backward compatibility. V2 callers
            # should pass views=["place","epistemic",...] explicitly.
            views = ["semantic", "temporal", "causal", "entity"]
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
        if "place" in views:
            context["views"]["place"] = self.retrieve_place_view(query, top_k=5)
        if "epistemic" in views:
            context["views"]["epistemic"] = self.retrieve_epistemic_view(query, top_k=5)
        return context

    def retrieve_place_view(
        self,
        query: str,
        place_ids: list[str] | None = None,
        phase_ids: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """MAGMA Place view — retrieve co-located entities (EcphoryRAG).

        Stub implementation: returns an empty list. Full Cypher query per
        ``docs/KG_V2_DESIGN.md`` §5.1 lands in a follow-up PR.
        """
        _ = (query, place_ids, phase_ids, top_k)  # unused in stub
        logger.debug(
            "retrieve_place_view stub called (query=%r); see "
            "docs/KG_V2_DESIGN.md §5.1 for full impl.",
            query,
        )
        return []

    def retrieve_epistemic_view(
        self,
        query: str,
        include_contradictions: bool = True,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """MAGMA Epistemic view — beliefs + supporting / contradicting evidence.

        CONCEPT:KG-2.2 — Entity-Claim Extraction / MAGMA Completion

        Queries the KG for ClaimNodes matching the query, then traverses
        BUILDS_ON, EXEMPLIFIES, CITES edges for supporting evidence and
        CONTRADICTS edges for contradicting evidence.

        Args:
            query: Search query for relevant claims.
            include_contradictions: Whether to include contradicting claims.
            top_k: Maximum number of beliefs to return.

        Returns:
            Dict with ``beliefs``, ``supporting``, and ``contradicting`` lists.
        """
        beliefs: list[dict[str, Any]] = []
        supporting: list[dict[str, Any]] = []
        contradicting: list[dict[str, Any]] = []

        if self.backend:
            # 1. Find claims matching the query (beliefs)
            belief_results = self.backend.execute(
                "MATCH (c:Claim) WHERE c.claim_text CONTAINS $q "
                "OR c.name CONTAINS $q "
                "RETURN c ORDER BY c.confidence DESC LIMIT $limit",
                {"q": query, "limit": top_k},
            )
            for row in belief_results:
                claim = row.get("c", row)
                beliefs.append(claim)
                claim_id = claim.get("id", "")

                # 2. Find supporting evidence (BUILDS_ON, EXEMPLIFIES, CITES)
                support_results = self.backend.execute(
                    "MATCH (s)-[r]->(c {id: $cid}) "
                    "WHERE type(r) IN ['BUILDS_ON', 'EXEMPLIFIES', 'CITES', "
                    "'builds_on', 'exemplifies', 'cites'] "
                    "RETURN s, type(r) as rel_type",
                    {"cid": claim_id},
                )
                for s_row in support_results:
                    support_node = s_row.get("s", s_row)
                    support_node["_relationship"] = s_row.get("rel_type", "supports")
                    support_node["_target_claim"] = claim_id
                    supporting.append(support_node)

                # 3. Find contradicting evidence (CONTRADICTS)
                if include_contradictions:
                    contradict_results = self.backend.execute(
                        "MATCH (s)-[r]->(c {id: $cid}) "
                        "WHERE type(r) IN ['CONTRADICTS', 'contradicts', "
                        "'CONTRADICTS_BELIEF', 'contradicts_belief'] "
                        "RETURN s, type(r) as rel_type",
                        {"cid": claim_id},
                    )
                    for c_row in contradict_results:
                        contra_node = c_row.get("s", c_row)
                        contra_node["_relationship"] = c_row.get(
                            "rel_type", "contradicts"
                        )
                        contra_node["_target_claim"] = claim_id
                        contradicting.append(contra_node)
        else:
            # NetworkX fallback
            query_lower = query.lower()
            for node_id, data in self.graph.nodes(data=True):
                node_type = str(data.get("type", "")).lower()
                if node_type in ("claim", "evidence"):
                    claim_text = str(
                        data.get("claim_text", data.get("claim", ""))
                    ).lower()
                    name = str(data.get("name", "")).lower()
                    if query_lower in claim_text or query_lower in name:
                        belief = dict(data)
                        belief["id"] = node_id
                        beliefs.append(belief)

                        if len(beliefs) >= top_k:
                            break

            # Find supporting/contradicting edges for found beliefs
            for belief in beliefs:
                belief_id = belief.get("id", "")
                for u, v, edata in self.graph.in_edges(belief_id, data=True):
                    edge_type = str(edata.get("type", "")).lower()
                    source_data = dict(self.graph.nodes.get(u, {}))
                    source_data["id"] = u
                    source_data["_target_claim"] = belief_id

                    if edge_type in ("builds_on", "exemplifies", "cites"):
                        source_data["_relationship"] = edge_type
                        supporting.append(source_data)
                    elif edge_type in (
                        "contradicts",
                        "contradicts_belief",
                        "contradicts_kb",
                    ):
                        if include_contradictions:
                            source_data["_relationship"] = edge_type
                            contradicting.append(source_data)

        logger.debug(
            "Epistemic view for %r: %d beliefs, %d supporting, %d contradicting",
            query,
            len(beliefs),
            len(supporting),
            len(contradicting),
        )

        return {
            "beliefs": beliefs,
            "supporting": supporting,
            "contradicting": contradicting,
        }

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
