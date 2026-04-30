"""Coverage push for agent_utilities.knowledge_graph.pipeline.phases.*.

Targets each phase's ``execute_fn`` via a mocked PipelineContext with a
pre-seeded NetworkX graph.  Backend / external services are replaced with
MagicMock to avoid any I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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

    monkeypatch.setattr(
        communities.nx.community, "louvain_communities", raise_louvain
    )
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


# ---------------------------------------------------------------------------
# resolve phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_empty_graph() -> None:
    """Empty graph -> resolved_dependencies=0."""
    from agent_utilities.knowledge_graph.pipeline.phases.resolve import (
        execute_resolve,
    )

    ctx = _make_ctx()
    result = await execute_resolve(ctx, {})
    assert result == {"cross_repo_dependencies": 0, "resolved_dependencies": 0}


@pytest.mark.asyncio
async def test_resolve_absolute_import_match() -> None:
    """depends_on_raw gets rewritten to depends_on via name_map."""
    from agent_utilities.knowledge_graph.pipeline.phases.resolve import (
        execute_resolve,
    )

    g = nx.MultiDiGraph()
    g.add_node(
        "src",
        type="file",
        name="src.py",
        file_path="/a/src.py",
    )
    g.add_node(
        "tgt",
        type="file",
        name="target.py",
        file_path="/a/target.py",
    )
    g.add_edge(
        "src",
        "raw",
        type="depends_on_raw",
        raw="target",
    )
    ctx = _make_ctx(graph=g)
    result = await execute_resolve(ctx, {})
    assert result["resolved_dependencies"] == 1


@pytest.mark.asyncio
async def test_resolve_relative_import() -> None:
    """Relative import '.models' is resolved via resolve_relative_import."""
    from agent_utilities.knowledge_graph.pipeline.phases.resolve import (
        execute_resolve,
    )

    g = nx.MultiDiGraph()
    g.add_node(
        "src",
        type="file",
        name="src.py",
        file_path="/a/src.py",
    )
    g.add_node(
        "tgt",
        type="file",
        name="models.py",
        file_path="/a/models.py",
    )
    g.add_edge("src", "raw", type="depends_on_raw", raw=".models")
    ctx = _make_ctx(graph=g)
    result = await execute_resolve(ctx, {})
    assert result["resolved_dependencies"] == 1


def test_resolve_relative_import_helper() -> None:
    """resolve_relative_import with a single-dot import."""
    from agent_utilities.knowledge_graph.pipeline.phases.resolve import (
        resolve_relative_import,
    )

    result = resolve_relative_import("/a/b/src.py", ".models")
    assert result is not None
    assert "models" in result


def test_resolve_relative_import_double_dot() -> None:
    """resolve_relative_import with '..utils'."""
    from agent_utilities.knowledge_graph.pipeline.phases.resolve import (
        resolve_relative_import,
    )

    result = resolve_relative_import("/a/b/c/src.py", "..utils")
    assert result is not None
    assert "utils" in result


def test_resolve_relative_import_bare_dot() -> None:
    """resolve_relative_import with bare '.' returns __init__.py path."""
    from agent_utilities.knowledge_graph.pipeline.phases.resolve import (
        resolve_relative_import,
    )

    result = resolve_relative_import("/a/b/src.py", ".")
    assert result is not None
    assert "__init__.py" in result


def test_resolve_relative_import_absolute_returns_none() -> None:
    """Absolute import (no leading dot) returns None."""
    from agent_utilities.knowledge_graph.pipeline.phases.resolve import (
        resolve_relative_import,
    )

    assert resolve_relative_import("/a/src.py", "absolute.module") is None


# ---------------------------------------------------------------------------
# sync phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_persistence_disabled() -> None:
    """persist_to_ladybug=False -> skipped."""
    from agent_utilities.knowledge_graph.pipeline.phases.sync import (
        execute_sync,
    )

    ctx = _make_ctx(persist_to_ladybug=False)
    result = await execute_sync(ctx, {})
    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_sync_backend_none_factory_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No backend in ctx and factory returns None -> skipped."""
    from agent_utilities.knowledge_graph.pipeline.phases import sync

    monkeypatch.setattr(sync, "create_backend", lambda db_path: None)
    ctx = _make_ctx()
    result = await sync.execute_sync(ctx, {})
    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_sync_happy_path() -> None:
    """Nodes and edges both sync via backend.execute."""
    from agent_utilities.knowledge_graph.pipeline.phases.sync import (
        execute_sync,
    )

    backend = _fake_backend()
    g = nx.MultiDiGraph()
    g.add_node("n1", type="tool", name="t1")
    g.add_node("n2", type="agent", name="a1")
    g.add_edge("n1", "n2", type="uses")
    ctx = _make_ctx(graph=g, backend=backend)
    result = await execute_sync(ctx, {})
    assert result["nodes_synced"] == 2
    assert result["edges_synced"] == 1


@pytest.mark.asyncio
async def test_sync_unknown_type_fallback() -> None:
    """Unmapped node type falls back to Titled version."""
    from agent_utilities.knowledge_graph.pipeline.phases.sync import (
        execute_sync,
    )

    backend = _fake_backend()
    g = nx.MultiDiGraph()
    g.add_node("n1", type="some_custom_type", name="x")
    ctx = _make_ctx(graph=g, backend=backend)
    result = await execute_sync(ctx, {})
    assert result["nodes_synced"] == 1


@pytest.mark.asyncio
async def test_sync_node_without_type_skipped() -> None:
    """Node with no type falls through when label is empty."""
    from agent_utilities.knowledge_graph.pipeline.phases.sync import (
        execute_sync,
    )

    backend = _fake_backend()
    g = nx.MultiDiGraph()
    g.add_node("n1")  # No type attr
    ctx = _make_ctx(graph=g, backend=backend)
    result = await execute_sync(ctx, {})
    # With no type, label is empty, so skipped
    assert result["nodes_synced"] == 0


@pytest.mark.asyncio
async def test_sync_edge_type_filtered_to_alnum() -> None:
    """Edge type is filtered to alphanumeric only."""
    from agent_utilities.knowledge_graph.pipeline.phases.sync import (
        execute_sync,
    )

    backend = _fake_backend()
    g = nx.MultiDiGraph()
    g.add_node("n1", type="tool", name="t")
    g.add_node("n2", type="tool", name="t2")
    g.add_edge("n1", "n2", type="has-child!@#")
    ctx = _make_ctx(graph=g, backend=backend)
    result = await execute_sync(ctx, {})
    assert result["edges_synced"] == 1


@pytest.mark.asyncio
async def test_sync_edge_type_all_filtered_empty_skipped() -> None:
    """Edge type that becomes empty after filtering is skipped."""
    from agent_utilities.knowledge_graph.pipeline.phases.sync import (
        execute_sync,
    )

    backend = _fake_backend()
    g = nx.MultiDiGraph()
    g.add_node("n1", type="tool", name="t")
    g.add_node("n2", type="tool", name="t2")
    g.add_edge("n1", "n2", type="!@#$%")  # All non-alnum
    ctx = _make_ctx(graph=g, backend=backend)
    result = await execute_sync(ctx, {})
    assert result["edges_synced"] == 0


@pytest.mark.asyncio
async def test_sync_node_execute_raises() -> None:
    """Failed node execute logs but doesn't halt the pipeline."""
    from agent_utilities.knowledge_graph.pipeline.phases.sync import (
        execute_sync,
    )

    backend = _fake_backend()
    call_count = {"n": 0}

    def boom(q: str, p: Any = None) -> list:
        call_count["n"] += 1
        raise RuntimeError("db error")

    backend.execute.side_effect = boom
    g = nx.MultiDiGraph()
    g.add_node("n1", type="tool", name="t")
    ctx = _make_ctx(graph=g, backend=backend)
    result = await execute_sync(ctx, {})
    # Failed, but no exception raised
    assert result["nodes_synced"] == 0


# ---------------------------------------------------------------------------
# knowledge_base phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_knowledge_base_disabled() -> None:
    """enable_knowledge_base=False -> skipped."""
    from agent_utilities.knowledge_graph.pipeline.phases.knowledge_base import (
        execute_knowledge_base,
    )

    ctx = _make_ctx(enable_knowledge_base=False)
    result = await execute_knowledge_base(ctx, {})
    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_knowledge_base_enabled_no_auto_ingest() -> None:
    """With enable_knowledge_base=True but auto_ingest=False -> complete, 0 kbs."""
    from agent_utilities.knowledge_graph.pipeline.phases.knowledge_base import (
        execute_knowledge_base,
    )

    ctx = _make_ctx(
        enable_knowledge_base=True,
        kb_auto_ingest_skill_graphs=False,
    )
    result = await execute_knowledge_base(ctx, {})
    assert result["status"] == "complete"
    assert result["kbs_processed"] == 0


@pytest.mark.asyncio
async def test_knowledge_base_skill_graphs_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing skill_graphs package -> skipped with reason."""
    import sys

    # Remove skill_graphs if present
    orig = sys.modules.pop("skill_graphs", None)

    # Shim to raise on import
    class _BadLoader:
        def find_module(self, name: str, path: Any = None) -> Any:
            if name == "skill_graphs":
                return self

        def load_module(self, name: str) -> None:
            raise ImportError("skill_graphs unavailable")

    sys.meta_path.insert(0, _BadLoader())  # type: ignore[arg-type]
    try:
        from agent_utilities.knowledge_graph.pipeline.phases.knowledge_base import (
            execute_knowledge_base,
        )

        ctx = _make_ctx(
            enable_knowledge_base=True,
            kb_auto_ingest_skill_graphs=True,
        )
        result = await execute_knowledge_base(ctx, {})
        assert result["status"] == "skipped"
    finally:
        sys.meta_path = [m for m in sys.meta_path if not isinstance(m, _BadLoader)]
        if orig is not None:
            sys.modules["skill_graphs"] = orig


# ---------------------------------------------------------------------------
# registry phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_empty() -> None:
    """Empty registry -> zeroes."""
    from agent_utilities.knowledge_graph.pipeline.phases.registry import (
        execute_registry,
    )
    from agent_utilities.models import MCPAgentRegistryModel

    with patch(
        "agent_utilities.graph.config_helpers.get_discovery_registry",
        return_value=MCPAgentRegistryModel(),
    ):
        ctx = _make_ctx()
        result = await execute_registry(ctx, {})
    assert result == {"agents": 0, "tools": 0}


@pytest.mark.asyncio
async def test_registry_with_agents_and_tools() -> None:
    """Agents and tools get added as nodes."""
    from agent_utilities.knowledge_graph.pipeline.phases.registry import (
        execute_registry,
    )
    from agent_utilities.models import (
        MCPAgent,
        MCPAgentRegistryModel,
        MCPToolInfo,
    )

    registry = MCPAgentRegistryModel(
        agents=[
            MCPAgent(
                name="router",
                description="Routes queries",
                agent_type="prompt",
                system_prompt="You route",
                tool_count=3,
            ),
        ],
        tools=[
            MCPToolInfo(
                name="search",
                description="search stuff",
                mcp_server="router",
                relevance_score=80,
                requires_approval=False,
            ),
        ],
    )
    with patch(
        "agent_utilities.graph.config_helpers.get_discovery_registry",
        return_value=registry,
    ):
        ctx = _make_ctx()
        result = await execute_registry(ctx, {})
    assert result["agents"] == 1
    assert result["tools"] == 1


# ---------------------------------------------------------------------------
# embedding phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_disabled() -> None:
    """enable_embeddings=False -> skipped."""
    from agent_utilities.knowledge_graph.pipeline.phases.embedding import (
        execute_embedding,
    )

    ctx = _make_ctx(enable_embeddings=False)
    result = await execute_embedding(ctx, {})
    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_embedding_no_candidate_nodes() -> None:
    """Graph with no embeddable text -> 0 generated, reason 'no nodes to embed'."""
    from agent_utilities.knowledge_graph.pipeline.phases.embedding import (
        execute_embedding,
    )

    g = nx.MultiDiGraph()
    # Nodes without any text fields
    g.add_node("n1")
    g.add_node("n2", name="a")  # Too short
    ctx = _make_ctx(graph=g)
    result = await execute_embedding(ctx, {})
    assert result["status"] == "completed"
    assert result["embeddings_generated"] == 0


@pytest.mark.asyncio
async def test_embedding_already_embedded_nodes_skipped() -> None:
    """Nodes with existing embedding are skipped."""
    from agent_utilities.knowledge_graph.pipeline.phases.embedding import (
        execute_embedding,
    )

    g = nx.MultiDiGraph()
    g.add_node(
        "n1",
        name="already embedded",
        embedding=[1.0, 2.0, 3.0],
    )
    ctx = _make_ctx(graph=g)
    result = await execute_embedding(ctx, {})
    assert result["embeddings_generated"] == 0


@pytest.mark.asyncio
async def test_embedding_with_http_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP embedding succeeds -> embeddings populated."""
    from agent_utilities.knowledge_graph.pipeline.phases import embedding

    def fake_batch(texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(embedding, "_generate_embedding_batch", fake_batch)

    g = nx.MultiDiGraph()
    g.add_node(
        "n1",
        name="long enough name string for embedding",
        description="some description",
    )
    ctx = _make_ctx(graph=g)
    result = await embedding.execute_embedding(ctx, {})
    assert result["embeddings_generated"] == 1


@pytest.mark.asyncio
async def test_embedding_http_fails_fallback_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP fails, LlamaIndex fallback succeeds."""
    from agent_utilities.knowledge_graph.pipeline.phases import embedding

    monkeypatch.setattr(
        embedding, "_generate_embedding_batch", lambda t: None
    )

    def fake_llama(texts: list[str]) -> list[list[float]]:
        return [[0.4, 0.5] for _ in texts]

    monkeypatch.setattr(embedding, "_generate_embedding_llamaindex", fake_llama)
    g = nx.MultiDiGraph()
    g.add_node(
        "n1",
        name="long enough name string for embedding",
        description="some description",
    )
    ctx = _make_ctx(graph=g)
    result = await embedding.execute_embedding(ctx, {})
    assert result["embeddings_generated"] == 1


@pytest.mark.asyncio
async def test_embedding_both_fail_error_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both HTTP and LlamaIndex fail -> errors=1."""
    from agent_utilities.knowledge_graph.pipeline.phases import embedding

    monkeypatch.setattr(
        embedding, "_generate_embedding_batch", lambda t: None
    )
    monkeypatch.setattr(
        embedding, "_generate_embedding_llamaindex", lambda t: None
    )
    g = nx.MultiDiGraph()
    g.add_node(
        "n1",
        name="long enough name string for embedding",
        description="some description",
    )
    ctx = _make_ctx(graph=g)
    result = await embedding.execute_embedding(ctx, {})
    assert result["errors"] == 1


@pytest.mark.asyncio
async def test_embedding_with_content_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nodes with 'content' field contribute to text."""
    from agent_utilities.knowledge_graph.pipeline.phases import embedding

    def fake_batch(texts: list[str]) -> list[list[float]]:
        return [[0.1] for _ in texts]

    monkeypatch.setattr(embedding, "_generate_embedding_batch", fake_batch)
    g = nx.MultiDiGraph()
    g.add_node("n1", content="this is long enough content text")
    ctx = _make_ctx(graph=g)
    result = await embedding.execute_embedding(ctx, {})
    assert result["embeddings_generated"] == 1


def test_embedding_generate_batch_http_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_generate_embedding_batch happy path."""
    from agent_utilities.knowledge_graph.pipeline.phases import embedding

    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2]},
            {"index": 1, "embedding": [0.3, 0.4]},
        ]
    }
    with patch("requests.post", return_value=fake_response):
        result = embedding._generate_embedding_batch(["text1", "text2"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_embedding_generate_batch_http_sorts_by_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Data rows are sorted by 'index' before extracting embeddings."""
    from agent_utilities.knowledge_graph.pipeline.phases import embedding

    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "data": [
            {"index": 1, "embedding": [0.3, 0.4]},
            {"index": 0, "embedding": [0.1, 0.2]},
        ]
    }
    with patch("requests.post", return_value=fake_response):
        result = embedding._generate_embedding_batch(["text1", "text2"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_embedding_generate_batch_http_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP exception returns None."""
    from agent_utilities.knowledge_graph.pipeline.phases import embedding

    with patch("requests.post", side_effect=RuntimeError("network down")):
        result = embedding._generate_embedding_batch(["text"])
    assert result is None


def test_embedding_generate_batch_no_data_key() -> None:
    """Response without 'data' key returns None."""
    from agent_utilities.knowledge_graph.pipeline.phases import embedding

    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {}
    with patch("requests.post", return_value=fake_response):
        result = embedding._generate_embedding_batch(["text"])
    assert result is None
