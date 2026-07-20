"""graph_engineering — Graph Engineering / GraphRAG MCP tool.

CONCEPT:AU-KG.retrieval.graph-engineering-canonical-prompts

One coherent, action-routed surface over
:mod:`agent_utilities.knowledge_graph.retrieval.graph_engineering` — GraphRAG
local search (entity + relationship-path neighborhood), global search (bounded
map-reduce over ``:CommunityReport`` nodes), and on-demand community-report
(re)building. Gap-fill, not a new engine: community DETECTION already runs
natively in the Rust engine (``eg-compute::mining::community``); this surface
reuses it plus the existing ``ContextCompiler`` for grounded assembly.

Mirrors the ``graph_ops_causal`` shape exactly (single ``@mcp.tool``, an
``action`` enum, registered into ``kg_server.REGISTERED_TOOLS`` +
``kg_server.ACTION_TOOL_ROUTES`` — the generic REST-twin factory in
``kg_server._build_server`` auto-mounts ``POST /graph/engineering``, no
bespoke endpoint needed) rather than inventing a new tool convention.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.security.error_surface import public_error_json


def register_graph_engineering_tools(mcp: Any) -> None:
    """Register the ``graph_engineering`` group on the given FastMCP server."""

    @mcp.tool(
        name="graph_engineering",
        description=(
            "GraphRAG-style Graph Engineering surface: entity-neighborhood local "
            "search, community-report map-reduce global search, and on-demand "
            "community-report (re)building. Gap-fill over existing capability -- "
            "community DETECTION already runs natively in the Rust engine "
            "(eg-compute::mining::community, Louvain/label-propagation) and "
            "report summarization reuses the exact prompt already used at "
            "ingest time; this tool adds the on-demand path plus the two "
            "GraphRAG query modes. Actions: "
            "'local_search' (entity + its relationship-path neighborhood -- "
            "seed resolution tries a bounded, parameterized native-Cypher exact "
            "match guided by the graph-query canonical prompt (never raw "
            "LLM-authored Cypher), falling back to semantic search; the "
            "neighborhood is assembled via the existing ContextCompiler -- "
            "policy enforcement, MMR diversity, budget-fit, citations, proof "
            "graph, all reused -- and answered with the grounded-answer "
            "canonical prompt). "
            "'global_search' (bounded map-reduce over :CommunityReport nodes -- "
            "MAP ranks reports by embedding similarity to the query then scores "
            "a bounded number of per-report partial answers; REDUCE synthesizes "
            "the top-scored ones via the SAME ContextCompiler + grounded-answer "
            "prompt; auto-builds reports on first use if none exist yet). "
            "'build_community_reports' (explicitly (re)build :CommunityReport "
            "nodes over the LIVE graph -- theme+summary per community via the "
            "reused ingest-time summarizer, each one embedded so global_search "
            "can rank it; also writes the single level-1 global rollup report). "
            "REST twin: POST /graph/engineering with the same fields."
        ),
        tags=[
            "graph-os",
            "graphrag",
            "retrieval",
            "community",
            "local-search",
            "global-search",
        ],
    )
    def graph_engineering(
        action: str = Field(
            default="local_search",
            description="local_search | global_search | build_community_reports",
        ),
        query: str = Field(
            default="",
            description="Natural-language question (local_search, global_search).",
        ),
        node_id: str = Field(
            default="",
            description="Explicit seed node id for local_search, skipping seed "
            "resolution (semantic/graph-query lookup) entirely.",
        ),
        depth: int = Field(
            default=1, description="local_search neighborhood hops (bounded)."
        ),
        top_k: int = Field(default=8, description="local_search result cap."),
        level: int = Field(
            default=0,
            description="global_search community-report level: 0 = per-community "
            "reports, 1 = the single global rollup report.",
        ),
        max_communities: int = Field(
            default=8,
            description="global_search MAP-step fan-out bound, and "
            "build_community_reports' max communities to summarize (there "
            "default 50 when left at 8 has no special meaning -- pass "
            "explicitly for build_community_reports if you want a different cap).",
        ),
        token_budget: int = Field(
            default=0,
            description="ContextCompiler token budget. 0 (default) uses each "
            "action's own default (2000 for local_search, 4000 for global_search).",
        ),
        resolution: float = Field(
            default=1.0,
            description="build_community_reports: Louvain resolution parameter.",
        ),
        min_size: int = Field(
            default=8,
            description="build_community_reports: minimum community size to summarize.",
        ),
        embed: bool = Field(
            default=True,
            description="build_community_reports: embed each report's "
            "theme+summary so global_search can rank it.",
        ),
        synthesize_answer: bool = Field(
            default=True,
            description="local_search: render the grounded-answer canonical "
            "prompt over the assembled bundle (best-effort).",
        ),
    ) -> str:
        """Thin action-router over knowledge_graph.retrieval.graph_engineering."""
        from agent_utilities.knowledge_graph.retrieval.graph_engineering import (
            build_community_reports,
            global_search,
            local_search,
        )

        action_norm = (action or "local_search").strip().lower()
        engine = kg_server._get_engine()
        if engine is None:
            return json.dumps(
                {
                    "surface": "graph_engineering",
                    "action": action_norm,
                    "error": "no active graph engine",
                }
            )

        try:
            if action_norm == "local_search":
                kwargs: dict[str, Any] = (
                    {"token_budget": token_budget} if token_budget > 0 else {}
                )
                result = local_search(
                    engine,
                    query,
                    node_id=node_id,
                    depth=depth,
                    top_k=top_k,
                    synthesize_answer=synthesize_answer,
                    **kwargs,
                )
            elif action_norm == "global_search":
                kwargs = {"token_budget": token_budget} if token_budget > 0 else {}
                result = global_search(
                    engine,
                    query,
                    level=level,
                    max_communities=max_communities,
                    **kwargs,
                )
            elif action_norm == "build_community_reports":
                result = build_community_reports(
                    engine,
                    resolution=resolution,
                    min_size=min_size,
                    max_communities=max_communities,
                    embed=embed,
                )
            else:
                return json.dumps(
                    {
                        "surface": "graph_engineering",
                        "action": action_norm,
                        "error": f"unknown action {action_norm!r}",
                    }
                )
        except (ValueError, TypeError) as exc:
            return public_error_json(
                exc,
                code="invalid_request",
                context={"surface": "graph_engineering", "action": action_norm},
            )

        return json.dumps(
            {"surface": "graph_engineering", "action": action_norm, "result": result},
            default=str,
        )

    kg_server.REGISTERED_TOOLS["graph_engineering"] = graph_engineering
    # No bespoke endpoint needed — the generic REST-twin factory in
    # kg_server._build_server (mirrors graph_ops_causal) mounts
    # POST /graph/engineering for every ACTION_TOOL_ROUTES entry without a
    # bespoke handler, dispatching through the SAME _execute_tool core.
    kg_server.ACTION_TOOL_ROUTES["graph_engineering"] = "/graph/engineering"
