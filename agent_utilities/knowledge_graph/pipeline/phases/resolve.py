import os
from pathlib import Path
from typing import Any

from ..types import (
    PhaseResult,
    PipelineContext,
    PipelinePhase,
)


def resolve_relative_import(current_file: str, raw_import: str) -> str | None:
    """Resolve a relative import like '.models' or '..utils' to a potential file path."""
    if not raw_import.startswith("."):
        return None

    dots = len(raw_import) - len(raw_import.lstrip("."))
    module_path = raw_import.lstrip(".")

    current_dir = Path(current_file).parent
    for _ in range(dots - 1):
        current_dir = current_dir.parent

    if module_path:
        target = current_dir / module_path.replace(".", "/")
    else:
        target = current_dir / "__init__.py"

    return str(target)


async def execute_resolve(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Resolve cross-file dependencies by matching imports to file nodes."""
    graph = ctx.nx_graph

    # Map of file path to node ID for exact matching
    path_map = {}
    # Map of file base name to node ID for fuzzy matching
    name_map = {}

    for node, data in graph.nodes(data=True):
        if data.get("type") == "file":
            path_map[data.get("file_path")] = node
            name_map[data.get("name")] = node

    resolved_count = 0
    edges_to_fix = []

    for u, v, data in graph.edges(data=True):
        if data.get("type") == "depends_on_raw" and "raw" in data:
            raw_target = data["raw"]
            source_file = graph.nodes[u].get("file_path")

            target_node = None

            # 1. Try relative resolution
            if source_file and raw_target.startswith("."):
                resolved_path = resolve_relative_import(source_file, raw_target)
                if resolved_path:
                    # Check exact path or with suffixes
                    for suffix in ["", ".py", ".ts", ".js", "/__init__.py"]:
                        p = resolved_path + suffix
                        if p in path_map:
                            target_node = path_map[p]
                            break

            # 2. Try absolute name matching
            if not target_node:
                potential_names = [
                    f"{raw_target}.py",
                    f"{raw_target}.ts",
                    f"{raw_target}.js",
                    os.path.basename(raw_target) + ".py",
                ]
                for name in potential_names:
                    if name in name_map:
                        target_node = name_map[name]
                        break

            if target_node:
                edges_to_fix.append((u, v, target_node, data))

    for u, old_v, new_v, data in edges_to_fix:
        if graph.has_edge(u, old_v):
            graph.remove_edge(u, old_v)
        # Convert to USES relationship
        graph.add_edge(u, new_v, type="depends_on", weight=data.get("weight", 1.0))
        resolved_count += 1

    return {"resolved_dependencies": resolved_count}


resolve_phase = PipelinePhase(
    name="resolve", deps=["parse"], execute_fn=execute_resolve
)
