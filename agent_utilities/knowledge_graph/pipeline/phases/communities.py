from typing import Any

from ..types import (
    PhaseResult,
    PipelineContext,
    PipelinePhase,
)


async def execute_communities(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Detect communities in the graph using the Rust-native engine."""
    graph = ctx.graph
    if graph.node_count() == 0:
        return {"communities": 0}

    try:
        communities = graph.community_detection()

        for i, community in enumerate(communities):
            for node_id in community:
                props = graph._get_node_properties(node_id)
                props["community"] = i
                graph.add_node(node_id, props)

        return {"communities": len(communities)}
    except Exception:
        # Fallback if community detection fails
        return {"communities": 0}


communities_phase = PipelinePhase(
    name="communities",
    deps=["resolve", "mro", "reference"],
    execute_fn=execute_communities,
)
