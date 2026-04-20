from typing import Any, Dict
import networkx as nx
from ..types import (
    PipelinePhase,
    PipelineContext,
    PhaseResult,
)


async def execute_communities(
    ctx: PipelineContext, deps: Dict[str, PhaseResult]
) -> Dict[str, Any]:
    """Detect communities in the graph using the Louvain algorithm."""
    graph = ctx.nx_graph
    if graph.number_of_nodes() == 0:
        return {"communities": 0}

    # Convert MultiDiGraph to Graph for community detection
    undirected = graph.to_undirected()

    try:
        # Use Louvain (nx 3.0 has community.louvain_communities)
        communities = nx.community.louvain_communities(undirected)

        for i, community in enumerate(communities):
            for node in community:
                graph.nodes[node]["community"] = i

        return {"communities": len(communities)}
    except Exception:
        # Fallback if louvain fails
        return {"communities": 0}


communities_phase = PipelinePhase(
    name="communities",
    deps=["resolve", "mro", "reference"],
    execute_fn=execute_communities,
)
