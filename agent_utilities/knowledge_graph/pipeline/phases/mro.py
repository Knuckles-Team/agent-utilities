from typing import Any

from ..types import (
    PhaseResult,
    PipelineContext,
    PipelinePhase,
)


async def execute_mro(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Resolve class inheritance hierarchies."""
    graph = ctx.nx_graph

    # Map of class name to node ID
    class_map = {}
    for node, data in graph.nodes(data=True):
        if data.get("type") == "symbol" and data.get("subtype") == "Class":
            class_map[data.get("name")] = node
        # Handle cases where subtype is not set but type is Class from metadata
        elif data.get("type") == "Class":
            class_map[data.get("name")] = node

    mro_count = 0
    for node, data in list(graph.nodes(data=True)):
        if data.get("type") == "symbol" and data.get("subtype") == "Class":
            bases = data.get("args", [])
            for base in bases:
                if base in class_map:
                    graph.add_edge(node, class_map[base], type="inherits_from")
                    mro_count += 1
        elif data.get("type") == "Class":
            bases = data.get("args", [])
            for base in bases:
                if base in class_map:
                    graph.add_edge(node, class_map[base], type="inherits_from")
                    mro_count += 1

    return {"resolved_mro": mro_count}


mro_phase = PipelinePhase(name="mro", deps=["parse"], execute_fn=execute_mro)
