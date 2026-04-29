"""Tests for cross-repository symbol resolution.

Concept: cross-repo-symbols
"""

from __future__ import annotations

import networkx as nx
import pytest

from agent_utilities.knowledge_graph.pipeline.phases.resolve import (
    _build_package_map,
    _is_stdlib,
    execute_resolve,
    resolve_relative_import,
)
from agent_utilities.knowledge_graph.pipeline.types import PipelineContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(graph: nx.MultiDiGraph) -> PipelineContext:
    """Create a minimal PipelineContext wrapping a graph."""
    from agent_utilities.models.knowledge_graph import PipelineConfig

    return PipelineContext(
        nx_graph=graph,
        config=PipelineConfig(workspace_path="/workspace"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.concept("cross-repo-symbols")
@pytest.mark.asyncio
async def test_resolve_cross_repo_import() -> None:
    """File in repo A imports from repo B → edge created with cross_repo=True."""
    g = nx.MultiDiGraph()
    # Repo A file
    g.add_node("file_a", type="file", name="client.py",
               file_path="/workspace/repo-a/client.py",
               repo_origin="repo-a")
    # Repo B file
    g.add_node("file_b", type="file", name="server.py",
               file_path="/workspace/repo-b/agent_utilities/server.py",
               repo_origin="repo-b")
    # Raw import edge
    g.add_edge("file_a", "placeholder_b", type="depends_on_raw", raw="agent_utilities.server")

    ctx = _make_ctx(g)
    result = await execute_resolve(ctx, {})

    assert result["resolved_dependencies"] >= 1
    assert result["cross_repo_dependencies"] >= 1

    # Verify edge was rewired
    edges = list(g.out_edges("file_a", data=True))
    resolved = [e for e in edges if e[2].get("type") == "depends_on"]
    assert len(resolved) >= 1
    assert resolved[0][2].get("cross_repo") is True


@pytest.mark.concept("cross-repo-symbols")
@pytest.mark.asyncio
async def test_resolve_package_level_import() -> None:
    """import agent_utilities.server → resolves to the correct file node."""
    g = nx.MultiDiGraph()
    g.add_node("src", type="file", name="main.py",
               file_path="/workspace/app/main.py", repo_origin="app")
    g.add_node("srv", type="file", name="server.py",
               file_path="/workspace/utils/agent_utilities/server.py",
               repo_origin="utils")
    g.add_edge("src", "placeholder", type="depends_on_raw", raw="agent_utilities.server")

    ctx = _make_ctx(g)
    result = await execute_resolve(ctx, {})

    assert result["resolved_dependencies"] >= 1
    edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("type") == "depends_on"]
    assert ("src", "srv") in edges


@pytest.mark.concept("cross-repo-symbols")
@pytest.mark.asyncio
async def test_cross_repo_edge_has_provenance() -> None:
    """Resolved cross-repo edges should have cross_repo=True attribute."""
    g = nx.MultiDiGraph()
    g.add_node("a", type="file", name="a.py", file_path="/w/r1/a.py", repo_origin="r1")
    g.add_node("b", type="file", name="utils.py", file_path="/w/r2/utils.py", repo_origin="r2")
    g.add_edge("a", "ph", type="depends_on_raw", raw="utils")

    ctx = _make_ctx(g)
    await execute_resolve(ctx, {})

    for _, _, data in g.out_edges("a", data=True):
        if data.get("type") == "depends_on":
            assert data.get("cross_repo") is True


@pytest.mark.concept("cross-repo-symbols")
def test_resolve_ignores_stdlib() -> None:
    """Standard library imports like 'os' should not create false edges."""
    assert _is_stdlib("os") is True
    assert _is_stdlib("sys") is True
    assert _is_stdlib("json") is True
    assert _is_stdlib("agent_utilities") is False
    assert _is_stdlib("pydantic") is False


@pytest.mark.concept("cross-repo-symbols")
def test_build_package_map() -> None:
    """_build_package_map should create dotted paths from file paths."""
    g = nx.MultiDiGraph()
    g.add_node("n1", type="file", name="server.py",
               file_path="/workspace/agent_utilities/server.py")
    g.add_node("n2", type="file", name="__init__.py",
               file_path="/workspace/agent_utilities/__init__.py")
    g.add_node("n3", type="file", name="readme.md",
               file_path="/workspace/README.md")  # non-py should be skipped

    pkg_map = _build_package_map(g)

    assert len(pkg_map) >= 1
    # At least one entry should be for agent_utilities.server or similar
    has_server = any("server" in k for k in pkg_map)
    assert has_server, f"Expected 'server' in package map keys: {list(pkg_map.keys())}"
