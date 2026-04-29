"""Knowledge Base ingestion smoke test.

Pytest migration of the Phase 6 Scenario 4 KB ingestion smoke from
``validate_stack.py``. Exercises the core ingest → graph → search loop
end-to-end against a tempfile-backed LadybugDB.

No live LLM is required. The extractor may log ``Connection error`` when
it can't reach a real embedding/extraction endpoint — that is tolerated;
the ingest still produces at least one ``Article`` node from the raw
markdown, which is what these tests assert.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import networkx as nx
import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine_with_tmp_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[nx.MultiDiGraph, object]]:
    """Build an ``IntelligenceGraphEngine`` on a tempfile-backed LadybugDB.

    Yields a ``(graph, engine)`` tuple and resets the engine singleton on
    teardown so other tests aren't hijacked by our throwaway instance.
    """
    from agent_utilities.knowledge_graph.backends import create_backend
    from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine

    db_path = tmp_path / "kg.db"
    monkeypatch.setenv("GRAPH_DB_PATH", str(db_path))
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))

    backend = create_backend(backend_type="ladybug", db_path=str(db_path))
    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    engine = IntelligenceGraphEngine(graph=graph, backend=backend)

    yield graph, engine

    # Release the singleton so later tests get a clean slate.
    if hasattr(engine, "backend") and engine.backend:
        try:
            engine.backend.close()
        except Exception:
            pass
    if IntelligenceGraphEngine.get_active() is engine:
        IntelligenceGraphEngine._ACTIVE_ENGINE = None  # type: ignore[attr-defined]


@pytest.fixture
def kb_engine(
    engine_with_tmp_db: tuple[nx.MultiDiGraph, object],
) -> object:
    """Fresh ``KBIngestionEngine`` sharing the graph from ``engine_with_tmp_db``."""
    from agent_utilities.knowledge_graph.kb.ingestion import KBIngestionEngine

    graph, engine = engine_with_tmp_db
    return KBIngestionEngine(graph=graph, backend=engine.backend)  # type: ignore[attr-defined]


@pytest.fixture
def tmp_doc_dir(tmp_path: Path) -> Path:
    """A tmp directory with a single small markdown file for ingestion."""
    src = tmp_path / "docs"
    src.mkdir()
    (src / "sample.md").write_text(
        "# Sample Knowledge\n\n"
        "This short article is about agents, knowledge graphs, and the "
        "agent-utilities ecosystem. It is intentionally minimal so the "
        "smoke test stays fast and deterministic.\n",
        encoding="utf-8",
    )
    return src


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_ingest_small_directory(
    kb_engine: object,
    engine_with_tmp_db: tuple[nx.MultiDiGraph, object],
    tmp_doc_dir: Path,
) -> None:
    """``ingest_directory`` creates at least one Article node in the graph."""
    from agent_utilities.models.knowledge_graph import RegistryNodeType

    graph, _engine = engine_with_tmp_db

    meta = await kb_engine.ingest_directory(  # type: ignore[attr-defined]
        tmp_doc_dir, kb_name="smoke-kb", topic="smoke test topic"
    )
    assert meta.status == "ready", f"ingest status: {meta.status!r}"
    assert meta.article_count >= 1

    article_nodes = [
        n for n, d in graph.nodes(data=True)
        if d.get("type") == RegistryNodeType.ARTICLE
    ]
    assert len(article_nodes) >= 1, (
        f"Expected >=1 Article node, got {len(article_nodes)}"
    )


@pytest.mark.asyncio
async def test_kb_search_returns_article(
    kb_engine: object,
    tmp_doc_dir: Path,
) -> None:
    """After ingest, ``search_knowledge_base`` returns a non-empty result set."""
    # Ingest first so the graph has content to search.
    meta = await kb_engine.ingest_directory(  # type: ignore[attr-defined]
        tmp_doc_dir, kb_name="smoke-kb", topic="smoke test topic"
    )
    assert meta.status == "ready"

    # The sample doc mentions "agents" — query for that token.
    hits = kb_engine.search_knowledge_base(  # type: ignore[attr-defined]
        "agents", top_k=5
    )
    assert isinstance(hits, list)
    assert len(hits) >= 1, "Expected at least one hit for query 'agents'"

    # Every hit should be a dict carrying an article identifier.
    for hit in hits:
        assert isinstance(hit, dict)
        assert hit.get("article_id"), f"Hit missing article_id: {hit}"
