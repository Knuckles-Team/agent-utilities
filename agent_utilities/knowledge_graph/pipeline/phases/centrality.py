from typing import Any, Dict
import networkx as nx
from ..types import (
    PipelinePhase,
    PipelineContext,
    PhaseResult,
)


async def execute_centrality(
    ctx: PipelineContext, deps: Dict[str, PhaseResult]
) -> Dict[str, Any]:
    """Calculate PageRank centrality for all nodes."""
    graph = ctx.nx_graph
    if graph.number_of_nodes() == 0:
        return {"centrality_calculated": False}

    try:
        # PageRank works on Directed graphs
        pagerank = nx.pagerank(graph)

        for node, score in pagerank.items():
            graph.nodes[node]["centrality"] = score

        return {
            "centrality_calculated": True,
            "top_node": max(pagerank, key=pagerank.get) if pagerank else None,
        }
    except Exception:
        return {"centrality_calculated": False}


centrality_phase = PipelinePhase(
    name="centrality", deps=["communities"], execute_fn=execute_centrality
)
