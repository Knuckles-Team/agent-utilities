"""Cross-repository import and dependency resolution (Phase 5).

Extends the base resolve phase to handle workspace-wide imports across
multiple repositories.  Builds a ``package_map`` from file nodes to
support absolute imports like ``from agent_utilities.server import X``.

Concept: cross-repo-symbols
"""

import os
import sys
from pathlib import Path
from typing import Any

from ..types import (
    PhaseResult,
    PipelineContext,
    PipelinePhase,
)

# Standard library top-level module names to ignore during resolution.
_STDLIB_MODULES: frozenset[str] = (
    frozenset(sys.stdlib_module_names)
    if hasattr(sys, "stdlib_module_names")
    else frozenset(
        {
            "os",
            "sys",
            "re",
            "json",
            "pathlib",
            "typing",
            "collections",
            "abc",
            "asyncio",
            "logging",
            "time",
            "datetime",
            "hashlib",
            "uuid",
            "io",
            "functools",
            "itertools",
            "dataclasses",
            "contextlib",
            "copy",
            "unittest",
            "math",
            "string",
            "textwrap",
            "traceback",
            "importlib",
            "subprocess",
            "shutil",
            "tempfile",
            "threading",
            "multiprocessing",
        }
    )
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


def _build_package_map(graph: Any) -> dict[str, str]:
    """Build a map of ``package.module`` → node_id from file nodes.

    For a file at ``/workspace/agent-utilities/agent_utilities/server.py``,
    this creates entries like:
    - ``agent_utilities.server`` → node_id
    - ``agent_utilities`` → node_id (if ``__init__.py``)
    """
    package_map: dict[str, str] = {}

    for node, data in graph.nodes(data=True):
        if data.get("type") != "file":
            continue
        file_path = data.get("file_path", "")
        if not file_path or not file_path.endswith(".py"):
            continue

        parts = Path(file_path).parts
        # Register all valid dotted-path suffixes so that cross-repo
        # imports like "agent_utilities.server" can be matched even when
        # the file lives under a deep workspace path.
        for i in range(len(parts) - 1):
            candidate = ".".join(parts[i:]).replace(".py", "").replace("/__init__", "")
            if candidate.endswith(".__init__"):
                candidate = candidate[:-9]
            candidate = candidate.replace("/", ".")
            if candidate and not candidate.startswith("."):
                # Prefer more-specific (shorter path) entries — don't
                # overwrite an existing entry from a shorter suffix
                if candidate not in package_map:
                    package_map[candidate] = node

    return package_map


def _is_stdlib(raw_target: str) -> bool:
    """Return True if the import target is a standard library module."""
    top_level = raw_target.split(".")[0]
    return top_level in _STDLIB_MODULES


async def execute_resolve(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Resolve cross-file dependencies by matching imports to file nodes.

    This version supports cross-repository resolution: when a file in
    repo A imports from repo B, the edge is created with a ``cross_repo=True``
    attribute for provenance tracking.
    """
    graph = ctx.nx_graph

    # Map of file path to node ID for exact matching
    path_map: dict[str, str] = {}
    # Map of file base name to node ID for fuzzy matching
    name_map: dict[str, str] = {}
    # Map of dotted package path to node ID
    package_map = _build_package_map(graph)

    for node, data in graph.nodes(data=True):
        if data.get("type") == "file":
            fp = data.get("file_path", "")
            path_map[fp] = node
            name_map[data.get("name", "")] = node

    resolved_count = 0
    cross_repo_count = 0
    edges_to_fix: list[tuple[str, str, str, dict, bool]] = []

    for u, v, data in graph.edges(data=True):
        if data.get("type") != "depends_on_raw" or "raw" not in data:
            continue

        raw_target = data["raw"]

        # Skip stdlib imports
        if _is_stdlib(raw_target):
            continue

        source_file = graph.nodes[u].get("file_path", "")
        source_repo = graph.nodes[u].get("repo_origin", "")

        target_node = None
        is_cross_repo = False

        # 1. Try relative resolution
        if source_file and raw_target.startswith("."):
            resolved_path = resolve_relative_import(source_file, raw_target)
            if resolved_path:
                for suffix in ["", ".py", ".ts", ".js", "/__init__.py"]:
                    p = resolved_path + suffix
                    if p in path_map:
                        target_node = path_map[p]
                        break

        # 2. Try package-level resolution (cross-repo capable)
        if not target_node and not raw_target.startswith("."):
            # Direct package match: "agent_utilities.server"
            if raw_target in package_map:
                target_node = package_map[raw_target]
            else:
                # Try prefix matching: "agent_utilities.server.build_agent_app"
                # → look for "agent_utilities.server"
                parts = raw_target.split(".")
                for depth in range(len(parts), 0, -1):
                    candidate = ".".join(parts[:depth])
                    if candidate in package_map:
                        target_node = package_map[candidate]
                        break

        # 3. Fallback: absolute name matching
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
            # Determine if cross-repo
            target_repo = graph.nodes[target_node].get("repo_origin", "")
            if source_repo and target_repo and source_repo != target_repo:
                is_cross_repo = True
            edges_to_fix.append((u, v, target_node, data, is_cross_repo))

    for u, old_v, new_v, data, cross_repo in edges_to_fix:
        if graph.has_edge(u, old_v):
            graph.remove_edge(u, old_v)
        edge_attrs: dict[str, Any] = {
            "type": "depends_on",
            "weight": data.get("weight", 1.0),
        }
        if cross_repo:
            edge_attrs["cross_repo"] = True
            cross_repo_count += 1
        graph.add_edge(u, new_v, **edge_attrs)
        resolved_count += 1

    return {
        "resolved_dependencies": resolved_count,
        "cross_repo_dependencies": cross_repo_count,
    }


resolve_phase = PipelinePhase(
    name="resolve", deps=["parse"], execute_fn=execute_resolve
)
