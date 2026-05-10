from __future__ import annotations
"""Coverage push for agent_utilities.knowledge_graph.pipeline.phases.*.

Targets each phase's ``execute_fn`` via a mocked PipelineContext with a
pre-seeded NetworkX graph.  Backend / external services are replaced with
MagicMock to avoid any I/O.
"""


from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import networkx as nx
import pytest

from agent_utilities.knowledge_graph.backends.base import GraphBackend
from agent_utilities.knowledge_graph.pipeline.types import (
    PipelineContext,
)
from agent_utilities.models.knowledge_graph import PipelineConfig


def _fake_backend() -> MagicMock:
    """Return a MagicMock that passes isinstance(GraphBackend) check."""
    backend = MagicMock(spec=GraphBackend)
    backend.execute.return_value = []
    return backend


def _make_ctx(
    workspace_path: str = "/tmp/ws",
    graph: nx.MultiDiGraph | None = None,
    backend: Any | None = None,
    **config_kwargs: Any,
) -> PipelineContext:
    """Build a PipelineContext with a given graph and backend."""
    cfg = PipelineConfig(workspace_path=workspace_path, **config_kwargs)
    g = graph or nx.MultiDiGraph()
    ctx = PipelineContext(config=cfg, nx_graph=g, backend=backend)
    return ctx


# ---------------------------------------------------------------------------
# centrality phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_centrality_empty_graph() -> None:
    """Empty graph -> centrality_calculated=False."""
    from agent_utilities.knowledge_graph.pipeline.phases.centrality import (
        execute_centrality,
    )

    ctx = _make_ctx()
    result = await execute_centrality(ctx, {})
    assert result == {"centrality_calculated": False}


@pytest.mark.asyncio
async def test_centrality_with_nodes() -> None:
    """Graph with nodes -> PageRank computed, top_node returned."""
    from agent_utilities.knowledge_graph.pipeline.phases.centrality import (
        execute_centrality,
    )

    g = nx.MultiDiGraph()
    g.add_node("a", type="file")
    g.add_node("b", type="file")
    g.add_edge("a", "b")
    ctx = _make_ctx(graph=g)
    result = await execute_centrality(ctx, {})
    assert result["centrality_calculated"] is True
    assert result["top_node"] in ("a", "b")


@pytest.mark.asyncio
async def test_centrality_exception_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """If PageRank raises, returns False."""
    from agent_utilities.knowledge_graph.pipeline.phases import centrality

    def raise_pagerank(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("pagerank failed")

    monkeypatch.setattr(centrality.nx, "pagerank", raise_pagerank)
    g = nx.MultiDiGraph()
    g.add_node("a")
    ctx = _make_ctx(graph=g)
    result = await centrality.execute_centrality(ctx, {})
    assert result == {"centrality_calculated": False}


# ---------------------------------------------------------------------------
# communities phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_communities_empty_graph() -> None:
    """Empty graph -> communities=0."""
    from agent_utilities.knowledge_graph.pipeline.phases.communities import (
        execute_communities,
    )

    ctx = _make_ctx()
    result = await execute_communities(ctx, {})
    assert result == {"communities": 0}


@pytest.mark.asyncio
async def test_communities_with_graph() -> None:
    """Populated graph -> louvain_communities count > 0."""
    from agent_utilities.knowledge_graph.pipeline.phases.communities import (
        execute_communities,
    )

    g = nx.MultiDiGraph()
    g.add_node("a")
    g.add_node("b")
    g.add_edge("a", "b")
    ctx = _make_ctx(graph=g)
    result = await execute_communities(ctx, {})
    assert result["communities"] >= 1


@pytest.mark.asyncio
async def test_communities_louvain_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """If louvain fails, returns communities=0."""
    from agent_utilities.knowledge_graph.pipeline.phases import communities

    def raise_louvain(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("louvain failed")

    monkeypatch.setattr(communities.nx.community, "louvain_communities", raise_louvain)
    g = nx.MultiDiGraph()
    g.add_node("a")
    ctx = _make_ctx(graph=g)
    result = await communities.execute_communities(ctx, {})
    assert result == {"communities": 0}


# ---------------------------------------------------------------------------
# mro phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mro_no_classes() -> None:
    """No class nodes -> resolved_mro=0."""
    from agent_utilities.knowledge_graph.pipeline.phases.mro import (
        execute_mro,
    )

    ctx = _make_ctx()
    result = await execute_mro(ctx, {})
    assert result == {"resolved_mro": 0}


@pytest.mark.asyncio
async def test_mro_symbol_class_relationship() -> None:
    """Class symbol with bases resolves to inherits_from edge."""
    from agent_utilities.knowledge_graph.pipeline.phases.mro import (
        execute_mro,
    )

    g = nx.MultiDiGraph()
    g.add_node(
        "Parent",
        type="symbol",
        subtype="Class",
        name="Parent",
        args=[],
    )
    g.add_node(
        "Child",
        type="symbol",
        subtype="Class",
        name="Child",
        args=["Parent"],
    )
    ctx = _make_ctx(graph=g)
    result = await execute_mro(ctx, {})
    assert result["resolved_mro"] == 1
    assert g.has_edge("Child", "Parent")


@pytest.mark.asyncio
async def test_mro_class_type_without_subtype() -> None:
    """Class with type='Class' (no subtype) still resolves."""
    from agent_utilities.knowledge_graph.pipeline.phases.mro import (
        execute_mro,
    )

    g = nx.MultiDiGraph()
    g.add_node("Parent", type="Class", name="Parent", args=[])
    g.add_node("Child", type="Class", name="Child", args=["Parent"])
    ctx = _make_ctx(graph=g)
    result = await execute_mro(ctx, {})
    assert result["resolved_mro"] == 1


@pytest.mark.asyncio
async def test_mro_unknown_base_is_skipped() -> None:
    """Unknown base class is skipped (not in class_map)."""
    from agent_utilities.knowledge_graph.pipeline.phases.mro import (
        execute_mro,
    )

    g = nx.MultiDiGraph()
    g.add_node(
        "Child",
        type="symbol",
        subtype="Class",
        name="Child",
        args=["UnknownBase"],
    )
    ctx = _make_ctx(graph=g)
    result = await execute_mro(ctx, {})
    assert result == {"resolved_mro": 0}


# ---------------------------------------------------------------------------
# reference phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reference_no_calls() -> None:
    """Empty graph -> resolved_references=0."""
    from agent_utilities.knowledge_graph.pipeline.phases.reference import (
        execute_reference,
    )

    ctx = _make_ctx()
    result = await execute_reference(ctx, {})
    assert result == {"resolved_references": 0}


@pytest.mark.asyncio
async def test_reference_resolves_calls() -> None:
    """calls_raw edges are rewritten to resolved calls edges."""
    from agent_utilities.knowledge_graph.pipeline.phases.reference import (
        execute_reference,
    )

    g = nx.MultiDiGraph()
    g.add_node("caller", type="symbol", name="caller")
    g.add_node("target", type="symbol", name="do_thing")
    g.add_edge(
        "caller",
        "unresolved",
        type="calls_raw",
        raw="do_thing",
    )
    ctx = _make_ctx(graph=g)
    result = await execute_reference(ctx, {})
    assert result["resolved_references"] == 1


@pytest.mark.asyncio
async def test_reference_method_call_dot_notation() -> None:
    """calls_raw with 'self.method' resolves by name suffix."""
    from agent_utilities.knowledge_graph.pipeline.phases.reference import (
        execute_reference,
    )

    g = nx.MultiDiGraph()
    g.add_node("caller", type="Function", name="caller")
    g.add_node("method_target", type="Method", name="my_method")
    g.add_edge(
        "caller",
        "unresolved",
        type="calls_raw",
        raw="self.my_method",
    )
    ctx = _make_ctx(graph=g)
    result = await execute_reference(ctx, {})
    assert result["resolved_references"] == 1


@pytest.mark.asyncio
async def test_reference_unresolvable_skipped() -> None:
    """calls_raw to unknown symbol is skipped."""
    from agent_utilities.knowledge_graph.pipeline.phases.reference import (
        execute_reference,
    )

    g = nx.MultiDiGraph()
    g.add_node("caller", type="symbol", name="caller")
    g.add_edge("caller", "nowhere", type="calls_raw", raw="does_not_exist")
    ctx = _make_ctx(graph=g)
    result = await execute_reference(ctx, {})
    assert result == {"resolved_references": 0}


# ---------------------------------------------------------------------------
# memory phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_persistence_disabled() -> None:
    """persist_to_ladybug=False -> skipped."""
    from agent_utilities.knowledge_graph.pipeline.phases.memory import (
        execute_memory,
    )

    ctx = _make_ctx(persist_to_ladybug=False)
    result = await execute_memory(ctx, {})
    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_memory_with_shared_backend() -> None:
    """Uses ctx.backend when provided."""
    from agent_utilities.knowledge_graph.pipeline.phases.memory import (
        execute_memory,
    )

    backend = _fake_backend()
    # First call: nodes; second: edges
    backend.execute.side_effect = [
        [{"n": {"id": "n1", "type": "file", "name": "a.py"}}],
        [{"u": "n1", "v": "n2", "t": "CALLS"}],
    ]
    ctx = _make_ctx(backend=backend)
    # Add n2 so the edge has both ends
    ctx.nx_graph.add_node("n2")
    result = await execute_memory(ctx, {})
    assert result["nodes_loaded"] >= 1
    assert "duration_ms" in result


@pytest.mark.asyncio
async def test_memory_create_backend_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ctx.backend is None, create_backend is called."""
    from agent_utilities.knowledge_graph.pipeline.phases import memory

    backend = MagicMock()
    backend.execute.side_effect = [[], []]
    monkeypatch.setattr(memory, "create_backend", lambda db_path: backend)

    ctx = _make_ctx()
    result = await memory.execute_memory(ctx, {})
    assert "nodes_loaded" in result


@pytest.mark.asyncio
async def test_memory_create_backend_none_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_backend returns None -> skipped."""
    from agent_utilities.knowledge_graph.pipeline.phases import memory

    monkeypatch.setattr(memory, "create_backend", lambda db_path: None)
    ctx = _make_ctx()
    result = await memory.execute_memory(ctx, {})
    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_memory_backend_execute_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend execute raising -> failed status."""
    from agent_utilities.knowledge_graph.pipeline.phases.memory import (
        execute_memory,
    )

    backend = _fake_backend()
    backend.execute.side_effect = RuntimeError("db error")
    ctx = _make_ctx(backend=backend)
    result = await execute_memory(ctx, {})
    assert result["status"] == "failed"
    assert "db error" in result["error"]


# ---------------------------------------------------------------------------
# scan phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_missing_root() -> None:
    """Nonexistent root path returns empty list."""
    from agent_utilities.knowledge_graph.pipeline.phases.scan import (
        execute_scan,
    )

    ctx = _make_ctx(workspace_path="/definitely/not/here")
    result = await execute_scan(ctx, {})
    assert result == []


@pytest.mark.asyncio
async def test_scan_finds_code_files(tmp_path: Path) -> None:
    """Scan walks and finds supported suffixes."""
    from agent_utilities.knowledge_graph.pipeline.phases.scan import (
        execute_scan,
    )

    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.md").write_text("# doc")
    (tmp_path / "c.txt").write_text("skip me")  # unsupported suffix
    ctx = _make_ctx(workspace_path=str(tmp_path))
    result = await execute_scan(ctx, {})
    assert len(result) == 2


@pytest.mark.asyncio
async def test_scan_respects_gitignore(tmp_path: Path) -> None:
    """Files matching .gitignore are excluded."""
    from agent_utilities.knowledge_graph.pipeline.phases.scan import (
        execute_scan,
    )

    (tmp_path / ".gitignore").write_text("ignored.py\n")
    (tmp_path / "kept.py").write_text("x = 1")
    (tmp_path / "ignored.py").write_text("x = 2")
    ctx = _make_ctx(workspace_path=str(tmp_path))
    result = await execute_scan(ctx, {})
    assert any("kept.py" in f for f in result)
    assert not any("ignored.py" in f for f in result)


@pytest.mark.asyncio
async def test_scan_respects_exclude_patterns(tmp_path: Path) -> None:
    """exclude_patterns from config are honored."""
    from agent_utilities.knowledge_graph.pipeline.phases.scan import (
        execute_scan,
    )

    (tmp_path / "good.py").write_text("x = 1")
    # Create a file in a specially-named directory that matches exclude pattern
    sub = tmp_path / "node_modules"
    sub.mkdir()
    (sub / "bad.py").write_text("x = 2")
    ctx = _make_ctx(workspace_path=str(tmp_path))
    result = await execute_scan(ctx, {})
    # node_modules is in default exclude_patterns
    assert any("good.py" in f for f in result)
    assert not any("bad.py" in f for f in result)


@pytest.mark.asyncio
async def test_scan_skips_hidden_dirs(tmp_path: Path) -> None:
    """Hidden dirs are skipped."""
    from agent_utilities.knowledge_graph.pipeline.phases.scan import (
        execute_scan,
    )

    (tmp_path / "good.py").write_text("x = 1")
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "bad.py").write_text("x = 2")
    ctx = _make_ctx(workspace_path=str(tmp_path))
    result = await execute_scan(ctx, {})
    assert any("good.py" in f for f in result)
