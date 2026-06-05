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

from ...models.knowledge_graph import RegistryNodeType

logger = logging.getLogger(__name__)


class QueryMixin(_Base):
    """Query and search capabilities for the KG engine."""

    def query_cypher(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        clearance_level: int = 999,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query against the persistent Graph store.

        Enterprise ABAC/RBAC Middleware Interceptor:
        Queries are automatically scoped by the user's/agent's security clearance.
        Nodes with a requiresClassification > clearance_level are omitted.

        CONCEPT:KG-2.11 — when ``as_of`` (an ISO-8601 instant) is supplied, result rows are
        post-filtered to those whose bi-temporal validity interval
        (``valid_from <= as_of < valid_to``) contains that instant. Rows without temporal
        metadata pass through unchanged. This answers "what was true as of date T" — something
        Quarq's flat storage-ordered files cannot do.
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
                "GraphBackend not initialized; using basic graph compute fallback for Cypher query."
            )
            rows = self._query_nx_fallback(query, params, clearance_level)
        else:
            rows = self.backend.execute(query, params)

        if as_of:
            from agent_utilities.knowledge_graph.core.bitemporal import filter_as_of

            rows = filter_as_of(rows, as_of)
        return rows

    def resolve_temporal_contradiction(
        self,
        fact_a_id: str,
        fact_b_id: str,
        fact_a_props: dict[str, Any],
        fact_b_props: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve two contradicting facts by event-time precedence (CONCEPT:KG-2.11).

        The later ``event_time`` wins; a ``SUPERSEDES`` edge is written winner→loser and the
        loser's ``valid_to`` is closed at the winner's ``event_time`` (the fact is never
        deleted, preserving as-of history). Implements Quarq's prompt-only "newer supersedes
        older" rule (agent-oss/agent.py:2462-2468) as a structural graph mutation.

        Returns a summary dict ``{"winner": id, "loser": id, "valid_to": boundary}``.
        """
        from agent_utilities.knowledge_graph.core.bitemporal import (
            resolve_precedence,
            supersede,
        )

        a = {**fact_a_props, "id": fact_a_id}
        b = {**fact_b_props, "id": fact_b_id}
        winner, loser = resolve_precedence(a, b)
        supersede(winner, loser)
        # Persist: SUPERSEDES edge + close the loser's validity interval.
        try:
            self.link_nodes(
                winner["id"],
                loser["id"],
                "SUPERSEDES",
                properties={"event_time": winner.get("event_time")},
            )
            if self.backend:
                self.backend.execute(
                    "MATCH (n) WHERE n.id = $id SET n.valid_to = $vt",
                    {"id": loser["id"], "vt": loser.get("valid_to")},
                )
        except Exception as e:  # pragma: no cover - persistence is best-effort
            logger.warning("Temporal contradiction persistence failed: %s", e)
        return {
            "winner": winner["id"],
            "loser": loser["id"],
            "valid_to": loser.get("valid_to"),
        }

    def _query_nx_fallback(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        clearance_level: int = 999,
    ) -> list[dict[str, Any]]:
        """Basic fallback to graph compute for simple Cypher queries (MATCH ... RETURN)."""
        query_lower = query.lower()
        results = []

        # Case 1: Pull recent failures
        if "o:outcomeevaluation" in query_lower and "reward < 0.5" in query_lower:
            # MATCH (e:Episode)-[:PRODUCED_OUTCOME]->(o:OutcomeEvaluation) WHERE o.reward < 0.5
            for node_id in self.graph.node_ids():
                data = self.graph._get_node_properties(node_id)
                for succ in self.graph.get_successors(node_id):
                    if (
                        data.get("type") == RegistryNodeType.OUTCOME_EVALUATION
                        and data.get("reward", 1.0) < 0.5
                    ):
                        e_data = self.graph._get_node_properties(node_id)
                        results.append(
                            {
                                "id": node_id,
                                "description": e_data.get("description", ""),
                            }
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
            for e_id in self.graph.node_ids():
                e_data = self.graph._get_node_properties(e_id)
                n_type = str(e_data.get("type", "")).lower()
                if n_type == "episode" or "episode" in n_type:
                    # Check reward
                    has_reward = False
                    for v in self.graph.get_successors(e_id):
                        o_data = self.graph._get_node_properties(v)
                        logger.info(
                            f"  Checking edge {e_id}->{v} reward={o_data.get('reward')}"
                        )
                        if o_data.get("reward", 0.0) >= 0.8:
                            has_reward = True
                            break
                    if has_reward:
                        logger.info(f"GCE Fallback: Found successful episode {e_id}")
                        for v2 in self.graph.get_successors(e_id):
                            t_data = self.graph._get_node_properties(v2)
                            if str(t_data.get("type", "")).lower() == "tool_call":
                                results.append(
                                    {
                                        "ep_id": e_id,
                                        "tool": t_data.get("tool_name", ""),
                                        "ts": t_data.get("timestamp", ""),
                                    }
                                )
            logger.info(f"GCE Fallback: Found {len(results)} tool calls.")
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
                query_str = f"MATCH (n) WHERE ({where_clause}) AND coalesce(n.status, '') <> 'ARCHIVED' RETURN n"
                try:
                    res = self.backend.execute(query_str, params)
                    for row in res:
                        if not isinstance(row, dict):
                            continue
                        # Handle both the Cypher `RETURN n` wrapping ({"n": {...}})
                        # and flat-dict backends (e.g. EpistemicGraphBackend).
                        node = row.get("n", row)
                        if not isinstance(node, dict) or not node:
                            continue
                        # Filter client-side: some backends (in-memory
                        # EpistemicGraph) do not evaluate the WHERE clause and
                        # return every node, so re-apply the keyword match here.
                        name = str(node.get("name", "")).lower()
                        desc = str(node.get("description", "")).lower()
                        nid = str(node.get("id", "")).lower()
                        if not any(
                            k in nid or k in name or k in desc for k in keywords
                        ):
                            continue
                        # Soft-delete enforcement (backend-agnostic): in-memory /
                        # service backends may not evaluate the WHERE clause, so
                        # re-apply the ARCHIVED exclusion here too (matches the
                        # GCE fallback path and the Cypher intent).
                        if str(node.get("status", "")).upper() == "ARCHIVED":
                            continue
                        req_class = node.get("requiresClassification", 0)
                        if isinstance(req_class, int) and req_class > clearance_level:
                            continue
                        results.append(node)
                except Exception as e:
                    logger.debug(f"Backend keyword search failed: {e}")

        # 2. Search GCE for name/ID matches
        if not results:
            for node_id in self.graph.node_ids():
                data = self.graph._get_node_properties(node_id)
                # RBAC Enforcement Fallback
                req_class = data.get("requiresClassification", 0)
                if isinstance(req_class, int) and req_class > clearance_level:
                    continue

                # Soft-delete enforcement
                if str(data.get("status", "")).upper() == "ARCHIVED":
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

        # Sort by graded relevance, not a coarse boolean. A bare "any keyword in
        # name" predicate cannot distinguish "Agent A" from "Agent B" (both
        # contain "agent"), so the result order would fall back to the backend's
        # arbitrary iteration order. Rank by: (1) whole-query substring match in
        # name, then in id; (2) count of distinct keywords matched in name; then
        # in id/description — so the closest-matching node wins deterministically.
        def _relevance(x: dict[str, Any]) -> tuple:
            name = str(x.get("name", "")).lower()
            desc = str(x.get("description", "")).lower()
            nid = str(x.get("id", "")).lower()
            name_kw = sum(1 for k in keywords if k in name)
            id_desc_kw = sum(1 for k in keywords if k in nid or k in desc)
            return (
                query_lower in name,
                query_lower in nid,
                name_kw,
                id_desc_kw,
            )

        results.sort(key=_relevance, reverse=True)

        return results[:top_k]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        include_archived: bool = False,
        skip_quality_gate: bool = False,
        relevance_threshold: float | None = None,
        target_paths: list[str] | None = None,
        mode: str = "standard",
        self_correct: bool = False,
        corpus_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform a multi-faceted search using Hybrid GraphRAG.

        CONCEPT:KG-2.12 — when ``mode`` is ``"hyde"``/``"deep"`` (or ``self_correct`` is set),
        delegates to the memory-first ``plan_and_retrieve`` policy (HyDE multi-query plan, dual
        thresholds, and an evidence-gated second pass). ``mode="standard"`` preserves the prior
        single-query behavior.
        """
        if mode in ("hyde", "deep") or self_correct:
            results = self.hybrid_retriever.plan_and_retrieve(
                query,
                context_window=top_k * 2,
                mode=mode if mode in ("hyde", "standard", "deep") else "hyde",
                self_correct=self_correct,
                corpus_id=corpus_id,
            )
        else:
            results = self.hybrid_retriever.retrieve_hybrid(
                query,
                context_window=top_k * 2,
                skip_quality_gate=skip_quality_gate,
                relevance_threshold=relevance_threshold,
                target_paths=target_paths,
                corpus_id=corpus_id,
            )
        if not include_archived:
            results = [
                r for r in results if str(r.get("status", "")).upper() != "ARCHIVED"
            ]
        return results[:top_k]

    def search_dci(
        self,
        query: str,
        max_hops: int = 2,
        top_k: int = 10,
        evidence_chain: bool = True,
    ) -> list[dict[str, Any]]:
        """Direct Corpus Interaction — multi-hop graph traversal retrieval.

        CONCEPT:KG-2.3 — Research: 2605.05242v1

        Unlike vector search (finds similar chunks by embedding distance),
        DCI traverses the graph structure to find connected evidence chains.
        This enables multi-hop reasoning: "What papers cite the same methods?"
        or "Which concepts are connected through shared implementations?"

        Operates in 3 layers (matching our existing L1/L2/L3 pipeline):
            L1: Initial vector seed (fast, same as search_hybrid).
            L2: Graph neighbor expansion — retrieve 1-hop neighbors of L1 results.
            L3: Multi-hop traversal — follow edges up to ``max_hops`` deep,
                building an evidence chain that tracks the traversal path.

        Args:
            query: Search query for the initial vector seed.
            max_hops: Maximum graph traversal depth (default 2).
            top_k: Maximum results to return.
            evidence_chain: If True, include the traversal path in results.

        Returns:
            List of result dicts, each with:
                - Standard node properties (id, name, type, score, etc.)
                - ``hop_depth``: How many hops from the seed this result is.
                - ``evidence_path``: List of (node_id, edge_type) tuples
                  showing how this result was reached (if evidence_chain=True).
        """
        # L1: Vector seed — fast retrieval
        seeds = self.search_hybrid(query, top_k=min(top_k, 5))
        if not seeds:
            return []

        # Track all discovered nodes to avoid duplicates
        seen: set[str] = set()
        results: list[dict[str, Any]] = []

        # Add seeds as hop-0 results
        for seed in seeds:
            sid = str(seed.get("id", ""))
            if not sid or sid in seen:
                continue
            seen.add(sid)
            seed["hop_depth"] = 0
            seed["evidence_path"] = [(sid, "seed")]
            results.append(seed)

        # L2+L3: Graph traversal — expand outward from seeds
        frontier = [
            (str(s.get("id", "")), s.get("evidence_path", []))
            for s in seeds
            if s.get("id")
        ]

        for hop in range(1, max_hops + 1):
            next_frontier: list[tuple[str, list]] = []

            for node_id, path in frontier:
                if not self.graph.has_node(node_id):
                    continue

                # Expand neighbors (both directions)
                neighbors = list(self.graph.get_successors(node_id)) + list(
                    self.graph.get_predecessors(node_id)
                )

                for neighbor_id in neighbors:
                    if neighbor_id in seen:
                        continue
                    seen.add(neighbor_id)

                    # Get node data
                    node_data = dict(self.graph._get_node_properties(neighbor_id))
                    node_data["id"] = neighbor_id
                    node_data["hop_depth"] = hop

                    if evidence_chain:
                        node_data["evidence_path"] = path + [(neighbor_id, "RELATED")]

                    # Score decay: further hops get lower scores
                    base_score = float(node_data.get("importance_score", 0.5))
                    node_data["_score"] = round(base_score * (0.8**hop), 4)

                    results.append(node_data)
                    next_frontier.append(
                        (neighbor_id, node_data.get("evidence_path", []))
                    )

            frontier = next_frontier
            if not frontier:
                break

        # Sort by score (seeds first, then by decayed importance)
        results.sort(key=lambda x: (-x.get("_score", 0), x.get("hop_depth", 99)))

        return results[:top_k]

    def search_memories(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search specifically for memory nodes."""
        results = self.search_hybrid(query, top_k=50)
        return [r for r in results if r.get("type") == RegistryNodeType.MEMORY][:top_k]

    def query_impact(self, symbol_or_file: str) -> list[dict[str, Any]]:
        """Calculate the topological impact set for a code entity."""
        target_id = symbol_or_file
        if not self.graph.has_node(target_id):
            # Try fuzzy match by name
            for node in self.graph.node_ids():
                data = self.graph._get_node_properties(node)
                if data.get("name") == symbol_or_file:
                    target_id = node
                    break

        if not self.graph.has_node(target_id):
            return []

        # Ancestors via BFS on predecessors
        ancestors: set[str] = set()
        frontier = set(self.graph.get_predecessors(target_id))
        while frontier:
            next_frontier: set[str] = set()
            for n in frontier:
                if n not in ancestors:
                    ancestors.add(n)
                    next_frontier.update(self.graph.get_predecessors(n))
            frontier = next_frontier - ancestors
        return [{"id": n, **self.graph._get_node_properties(n)} for n in ancestors]

    def find_path(self, source: str, target: str) -> list[str]:
        """Find the shortest logical path between two nodes.

        CONCEPT:KG-2.16 (Plan 08 Synergy 4): offload the traversal to the Rust
        L0 compute tier in a single round-trip via ``GraphComputeEngine`` rather
        than a Python BFS that issues one L0 call per edge. Falls back to a
        local BFS only if the compiled shortest-path is unavailable.
        """
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return []

        # Primary: single-call Rust L0 shortest path.
        try:
            path = self.graph.get_shortest_path(source, target)
            if path is not None:
                return list(path)
        except Exception as e:
            logger.debug("L0 shortest_path unavailable, falling back to BFS: %s", e)

        # Fallback: backend-agnostic BFS over the in-memory graph.
        from collections import deque

        visited: set[str] = {source}
        queue: deque[list[str]] = deque([[source]])
        while queue:
            path_so_far = queue.popleft()
            current = path_so_far[-1]
            if current == target:
                return path_so_far
            for neighbor in self.graph.get_successors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path_so_far + [neighbor])
        return []

    def get_shortest_path(self, source: str, target: str) -> list[str]:
        """Alias for find_path."""
        return self.find_path(source, target)

    def get_agent_tools(self, agent_name: str) -> list[str]:
        """Get all tools provided by a specific agent."""
        tools = []
        if self.graph.has_node(agent_name):
            for v in self.graph.get_successors(agent_name):
                # Edge type info not directly available in GCE
                tools.append(v.replace("tool:", ""))
        return tools

    def find_agent_for_tool(self, tool_name: str) -> list[str]:
        """Find all agents that provide a specific tool."""
        agents = []
        tool_id = f"tool:{tool_name}"
        if not self.graph.has_node(tool_id):
            return []
        for u in self.graph.get_predecessors(tool_id):
            agents.append(u)
        return agents

    def run_inference(self) -> int:
        """Run standard inference rules over the graph to derive new facts."""
        return self.inference_engine.run_inference()

    def find_relevant_callable_resources(
        self, task_description: str, required_caps: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Find the most relevant Tools, Agents, or Skills for a given task.

        CONCEPT:ECO-4.0 enhanced: searches ALL resource types (tools,
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

        CONCEPT:ECO-4.0 — Self-Describing Function Registry

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
            results = [
                {"id": nid, **self.graph._get_node_properties(nid)}
                for nid in self.graph.node_ids()
            ]

        for r in results:
            rid = r.get("id", "")
            if rid in seen_ids:
                continue

            if str(r.get("status", "")).upper() == "ARCHIVED":
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
        for n in self.graph.node_ids():
            data = self.graph._get_node_properties(n)
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

        Retrieves co-located entities or places using a Cypher query from the backend.
        """
        if not self.backend:
            logger.debug("No backend configured for retrieve_place_view.")
            return []

        results = []
        if place_ids:
            # Match entities co-located at specified places
            cypher = (
                "MATCH (e:Entity)-[r:CO_LOCATED_AT|co_located_at]->(p:Place) "
                "WHERE p.id IN $place_ids "
                "RETURN e, p LIMIT $limit"
            )
            rows = self.backend.execute(
                cypher, {"place_ids": place_ids, "limit": top_k}
            )
            for row in rows:
                entity = row.get("e", row)
                place = row.get("p", {})
                if isinstance(entity, dict):
                    entity["_place"] = (
                        place.get("id") if isinstance(place, dict) else str(place)
                    )
                    results.append(entity)
        elif phase_ids:
            # Match entities associated with specific phases
            cypher = (
                "MATCH (e:Entity)-[r:ASSOCIATED_WITH|associated_with]->(p:Phase) "
                "WHERE p.id IN $phase_ids "
                "RETURN e, p LIMIT $limit"
            )
            rows = self.backend.execute(
                cypher, {"phase_ids": phase_ids, "limit": top_k}
            )
            for row in rows:
                entity = row.get("e", row)
                phase = row.get("p", {})
                if isinstance(entity, dict):
                    entity["_phase"] = (
                        phase.get("id") if isinstance(phase, dict) else str(phase)
                    )
                    results.append(entity)
        else:
            # Query based search for co-located entities matching query
            cypher = (
                "MATCH (e:Entity)-[r:CO_LOCATED_AT|co_located_at]->(p:Place) "
                "WHERE e.id CONTAINS $q OR p.id CONTAINS $q "
                "RETURN e, p LIMIT $limit"
            )
            rows = self.backend.execute(cypher, {"q": query, "limit": top_k})
            for row in rows:
                entity = row.get("e", row)
                place = row.get("p", {})
                if isinstance(entity, dict):
                    entity["_place"] = (
                        place.get("id") if isinstance(place, dict) else str(place)
                    )
                    results.append(entity)

        return results

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
            # GCE fallback
            query_lower = query.lower()
            for node_id in self.graph.node_ids():
                data = self.graph._get_node_properties(node_id)
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
                for u in self.graph.get_predecessors(belief_id):
                    edge_type = "related"  # Default when edge props unavailable
                    source_data = dict(self.graph._get_node_properties(u))
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

    # ── Native Innovation Discovery (CONCEPT:KG-2.0) ──────────────

    # Signal dictionaries for innovation extraction — embedded in the engine
    # so that all search modes can reuse them without external scripts.
    _BIOMIMICRY_KEYWORDS: dict[str, dict[str, str]] = {
        "swarm": {"analogy": "multi-agent coordination", "domain": "orchestration"},
        "colony": {"analogy": "distributed task allocation", "domain": "scheduling"},
        "pheromone": {
            "analogy": "stigmergic communication channels",
            "domain": "signal_boards",
        },
        "neural": {"analogy": "adaptive model routing", "domain": "model_selection"},
        "immune": {
            "analogy": "anomaly detection and self-healing",
            "domain": "circuit_breakers",
        },
        "evolution": {
            "analogy": "genetic algorithm optimization",
            "domain": "prompt_evolution",
        },
        "mutation": {"analogy": "parametric exploration", "domain": "variant_pool"},
        "symbiosis": {
            "analogy": "plugin ecosystem composition",
            "domain": "mcp_composition",
        },
        "quorum": {
            "analogy": "distributed consensus thresholds",
            "domain": "bft_voting",
        },
        "mycelium": {"analogy": "knowledge graph topology", "domain": "kg_routing"},
        "plasticity": {
            "analogy": "continual learning without forgetting",
            "domain": "ewc",
        },
        "homeostasis": {
            "analogy": "resource equilibrium maintenance",
            "domain": "cost_governors",
        },
        "emergent": {
            "analogy": "complex behavior from simple rules",
            "domain": "swarm_presets",
        },
        "stigmergy": {
            "analogy": "indirect coordination via environment",
            "domain": "signal_boards",
        },
    }

    _TECH_KEYWORDS: dict[str, dict[str, str]] = {
        "attention": {
            "analogy": "context-aware retrieval weighting",
            "domain": "hybrid_retriever",
        },
        "transformer": {
            "analogy": "parallel sequence processing",
            "domain": "batch_processing",
        },
        "embedding": {
            "analogy": "semantic vector representation",
            "domain": "knowledge_graph",
        },
        "reinforcement": {
            "analogy": "reward-driven routing optimization",
            "domain": "confidence_gating",
        },
        "chain-of-thought": {
            "analogy": "rationale persistence",
            "domain": "quiet_star",
        },
        "tree search": {"analogy": "MCTS planning fallback", "domain": "lats"},
        "consensus": {"analogy": "multi-agent agreement", "domain": "bft"},
        "knowledge graph": {"analogy": "structured relational memory", "domain": "kg"},
        "ontology": {"analogy": "formal domain modeling", "domain": "owl_reasoning"},
        "retrieval": {
            "analogy": "context-aware document fetching",
            "domain": "hybrid_retriever",
        },
        "rag": {
            "analogy": "retrieval-augmented generation",
            "domain": "hybrid_retriever",
        },
        "multi-agent": {
            "analogy": "coordinated specialist teams",
            "domain": "orchestration",
        },
        "orchestration": {
            "analogy": "workflow coordination engine",
            "domain": "orchestration",
        },
        "planning": {
            "analogy": "task decomposition and scheduling",
            "domain": "htn_planning",
        },
        "tool use": {
            "analogy": "dynamic capability invocation",
            "domain": "tool_interface",
        },
        "mcp": {
            "analogy": "model context protocol integration",
            "domain": "mcp_factory",
        },
        "safety": {
            "analogy": "guardrails and constraint enforcement",
            "domain": "guardrails",
        },
        "evaluation": {
            "analogy": "quality assessment framework",
            "domain": "evaluation_engine",
        },
        "telemetry": {
            "analogy": "runtime observability signals",
            "domain": "telemetry",
        },
        "scheduling": {
            "analogy": "resource allocation optimization",
            "domain": "cognitive_scheduler",
        },
        "memory": {
            "analogy": "persistent experience storage",
            "domain": "tiered_memory",
        },
        "federated": {
            "analogy": "distributed graph querying",
            "domain": "external_federation",
        },
        "hypergraph": {
            "analogy": "n-ary relationship modeling",
            "domain": "hyperedges",
        },
        "topology": {"analogy": "structural graph analysis", "domain": "partitioning"},
        "composability": {
            "analogy": "modular building-block assembly",
            "domain": "mcp_composition",
        },
        "middleware": {
            "analogy": "cross-cutting concern interception",
            "domain": "middleware",
        },
        "inference": {
            "analogy": "derived knowledge from assertions",
            "domain": "owl_reasoning",
        },
        "security": {"analogy": "threat defense mechanisms", "domain": "security"},
        "context window": {
            "analogy": "adaptive context management",
            "domain": "context_management",
        },
    }

    _INNOVATION_SIGNALS = [
        "novel",
        "new",
        "propose",
        "introduce",
        "demonstrate",
        "improve",
        "outperform",
        "achieve",
        "enable",
        "first",
        "state-of-the-art",
        "sota",
        "surpass",
        "advance",
    ]

    def _extract_signals(self, content: str) -> dict[str, Any]:
        """Extract biomimicry and tech innovation signals from text content."""
        import re

        content_lower = content.lower()
        bio_hits = []
        tech_hits = []

        for kw, info in self._BIOMIMICRY_KEYWORDS.items():
            count = content_lower.count(kw)
            if count > 0:
                bio_hits.append(
                    {
                        "keyword": kw,
                        "count": count,
                        "analogy": info["analogy"],
                        "domain": info["domain"],
                    }
                )

        for kw, info in self._TECH_KEYWORDS.items():
            count = content_lower.count(kw)
            if count > 0:
                tech_hits.append(
                    {
                        "keyword": kw,
                        "count": count,
                        "analogy": info["analogy"],
                        "domain": info["domain"],
                    }
                )

        # Extract key innovation claims
        claims = []
        for sent in re.split(r"[.!?]\s+", content):
            if any(sig in sent.lower() for sig in self._INNOVATION_SIGNALS):
                words = sent.split()
                if 5 < len(words) < 50:
                    claims.append(sent.strip())
        claims = claims[:5]

        return {
            "biomimicry_signals": sorted(bio_hits, key=lambda x: -x["count"]),
            "tech_signals": sorted(tech_hits, key=lambda x: -x["count"]),
            "innovation_claims": claims,
            "total_signal_count": len(bio_hits) + len(tech_hits),
        }

    def discover_innovations(
        self,
        query: str,
        top_k: int = 20,
        relevance_threshold: float = 0.2,
        exclude_assimilated: bool = False,
        target_codebase: str = "",
    ) -> dict[str, Any]:
        """Native Layer 1 cross-reference: vector search + innovation signal extraction.

        CONCEPT:KG-2.0 — Active Knowledge Graph Discovery

        Performs hybrid search, then enriches each result with biomimicry/tech
        innovation signals extracted from the node content. Results include
        signal counts, concept mappings, and innovation claims.

        This is the backend-native equivalent of the Layer 1 cross-reference
        pipeline, computed entirely on the KG without LLM calls.

        Args:
            query: Search query (concept name, research topic, etc.)
            top_k: Maximum enriched results to return.
            relevance_threshold: Minimum similarity score.
            exclude_assimilated: If True, filter out Article nodes that have
                ASSIMILATED_INTO edges with status='implemented' for the
                given target_codebase. CONCEPT:KG-2.6
            target_codebase: Codebase path to check assimilation against.
                Only used when exclude_assimilated=True.

        Returns:
            Dict with enriched results, signal summary, and top recommendations.
        """
        # 0. Build set of assimilated target_paths to exclude
        assimilated_paths: set[str] = set()
        if exclude_assimilated and self.backend:
            try:
                if target_codebase:
                    rows = self.query_cypher(
                        "MATCH (a:Article)-[r:ASSIMILATED_INTO]->(c) "
                        "WHERE r.status = 'implemented' AND r.codebase = $cb "
                        "RETURN DISTINCT a.target_path AS path",
                        {"cb": target_codebase},
                    )
                else:
                    rows = self.query_cypher(
                        "MATCH (a:Article)-[r:ASSIMILATED_INTO]->(c) "
                        "WHERE r.status = 'implemented' "
                        "RETURN DISTINCT a.target_path AS path",
                    )
                for row in rows:
                    p = row.get("path", "")
                    if p:
                        assimilated_paths.add(p)
                if assimilated_paths:
                    logger.info(
                        "Excluding %d assimilated paper paths from discovery",
                        len(assimilated_paths),
                    )
            except Exception as exc:
                logger.warning("Failed to load assimilated paths: %s", exc)

        # 1. Run hybrid vector search with low threshold to cast wide net
        raw_results = self.search_hybrid(
            query,
            top_k=top_k * 3,
            skip_quality_gate=True,
            relevance_threshold=relevance_threshold,
        )

        # 2. Enrich each result with innovation signals
        enriched = []
        domain_accumulator: dict[str, list[dict[str, Any]]] = {}

        for r in raw_results:
            # CONCEPT:KG-2.6 — Skip assimilated papers
            if assimilated_paths:
                r_path = r.get("target_path", "")
                if r_path and r_path in assimilated_paths:
                    continue

            # Build content from available fields
            content_parts = []
            for field in (
                "content",
                "description",
                "name",
                "summary",
                "text",
                "claim_text",
            ):
                val = r.get(field)
                if val and isinstance(val, str):
                    content_parts.append(val)
            content = " ".join(content_parts)

            if not content or len(content) < 20:
                continue

            signals = self._extract_signals(content)

            entry = {
                "id": r.get("id", ""),
                "name": r.get("name", r.get("title", "")),
                "type": r.get("type", ""),
                "target_path": r.get("target_path", ""),
                "score": round(r.get("_score", 0.0), 4),
                "concept_id": r.get("concept_id", ""),
                **signals,
            }

            # Accumulate by domain for recommendations
            for sig in signals["tech_signals"] + signals["biomimicry_signals"]:
                domain = sig["domain"]
                if domain not in domain_accumulator:
                    domain_accumulator[domain] = []
                domain_accumulator[domain].append(
                    {
                        "source": entry["name"] or entry["id"],
                        "keyword": sig["keyword"],
                        "analogy": sig["analogy"],
                        "score": entry["score"],
                    }
                )

            if signals["total_signal_count"] > 0:
                enriched.append(entry)

        # Sort by (score * signal_count) for emergent value ranking
        enriched.sort(
            key=lambda x: x["score"] * (1 + x["total_signal_count"]),
            reverse=True,
        )
        enriched = enriched[:top_k]

        # 3. Build domain-level recommendations
        recommendations = []
        for domain, sources in sorted(
            domain_accumulator.items(),
            key=lambda x: -len(x[1]),
        ):
            best = max(sources, key=lambda s: s["score"])
            recommendations.append(
                {
                    "domain": domain,
                    "analogy": best["analogy"],
                    "source_count": len(sources),
                    "top_source": best["source"],
                    "top_score": best["score"],
                    "priority": "high"
                    if len(sources) >= 3
                    else "medium"
                    if len(sources) >= 2
                    else "low",
                }
            )

        return {
            "query": query,
            "total_matches": len(enriched),
            "results": enriched,
            "domain_recommendations": recommendations[:20],
            "summary": {
                "total_signals": sum((e["total_signal_count"] for e in enriched), 0),
                "unique_domains": len(domain_accumulator),
                "high_priority_domains": len(
                    [r for r in recommendations if r["priority"] == "high"]
                ),
            },
        }
