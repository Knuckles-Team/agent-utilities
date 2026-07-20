from typing import Any

from ..types import (
    PhaseResult,
    PipelineContext,
    PipelinePhase,
)


async def execute_centrality(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Calculate PageRank centrality for all nodes via Rust-native engine."""
    graph = ctx.graph
    if graph.node_count() == 0:
        return {"centrality_calculated": False}

    try:
        pagerank_results = graph.pagerank()

        # Update node properties with centrality scores
        best_node = None
        best_score = -1.0
        for node_id, score in pagerank_results:
            props = graph._get_node_properties(node_id)
            props["centrality"] = score
            graph.add_node(node_id, props)
            if score > best_score:
                best_score = score
                best_node = node_id

        return {
            "centrality_calculated": True,
            "top_node": best_node,
        }
    except Exception as e:
        print(f"DEBUG: pagerank failed: {e}")
        return {"centrality_calculated": False}


centrality_phase = PipelinePhase(
    name="centrality", deps=["communities"], execute_fn=execute_centrality
)
