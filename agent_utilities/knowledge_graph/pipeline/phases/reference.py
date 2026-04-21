from typing import Any

from ..types import (
    PhaseResult,
    PipelineContext,
    PipelinePhase,
)


async def execute_reference(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Resolve call graph references."""
    graph = ctx.nx_graph

    # Map of symbol name to node ID
    symbol_map = {}
    for node, data in graph.nodes(data=True):
        if data.get("type") == "symbol":
            symbol_map[data.get("name")] = node
        elif data.get("type") in ("Function", "Method", "Class"):
            symbol_map[data.get("name")] = node

    ref_count = 0
    edges_to_fix = []

    for u, v, data in graph.edges(data=True):
        if data.get("type") == "calls_raw" and "raw" in data:
            raw_target = data["raw"]
            # Handle method calls like self.my_method -> my_method
            name = raw_target.split(".")[-1]

            if name in symbol_map:
                edges_to_fix.append((u, v, symbol_map[name], data))

    for u, old_v, new_v, data in edges_to_fix:
        if graph.has_edge(u, old_v):
            graph.remove_edge(u, old_v)
        graph.add_edge(u, new_v, type="calls")
        ref_count += 1

    return {"resolved_references": ref_count}


reference_phase = PipelinePhase(
    name="reference", deps=["parse"], execute_fn=execute_reference
)
