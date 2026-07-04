"""WorkflowCompiler agent matching is bounded and never triggers a full scan.

CONCEPT:AU-ORCH.execution.nl-compilation-pipeline — NL workflow compilation. Resolving a step to an agent first
tries a structural KG query; the semantic fallback ranks candidates via the
engine's bounded vector index (``backend.semantic_search``, HNSW top-k). It must
NOT use ``engine.search_hybrid``, whose retriever materializes the whole graph
(a per-label full-node scan, once per search table) — O(graph) work that
OOM-crashed the graph-os child on a large KG. The embed is wall-clock bounded, so
a slow/down embedder degrades to the generic executor in seconds.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.workflow_compiler import WorkflowCompiler


class _Backend:
    """KG backend stub: structural query misses; bounded vector search hits."""

    def __init__(self) -> None:
        self.semantic_calls = 0

    def execute(self, _cypher: str, _params: dict[str, Any]) -> list[dict[str, Any]]:
        return []  # structural match misses → forces the semantic fallback

    def semantic_search(self, _vec: list[float], _n: int = 5) -> list[dict[str, Any]]:
        self.semantic_calls += 1
        return [{"id": "matched-server", "type": "Server"}]


class _Engine:
    """Minimal engine stub. ``search_hybrid`` must NEVER be reached."""

    def __init__(self) -> None:
        self.backend = _Backend()

    def search_hybrid(self, *_a: Any, **_k: Any) -> list[dict[str, Any]]:
        raise AssertionError(
            "search_hybrid must not be called — it triggers a full-graph scan"
        )


def _patch_embed(monkeypatch: pytest.MonkeyPatch, *, available: bool) -> dict[str, int]:
    """Stub the embedder so no real endpoint is contacted; count fn builds."""
    counters = {"make_fn": 0}

    def _make(*_a: Any, **_k: Any):
        counters["make_fn"] += 1
        return lambda texts: [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.semantic.make_embed_fn", _make
    )

    def _bounded(_embed_fn: Any, _text: str, _timeout: float):
        return [0.1, 0.2, 0.3] if available else None

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.research.search.bounded_embed", _bounded
    )
    return counters


def test_match_agent_skips_semantic_search_when_embedder_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_embed(monkeypatch, available=False)
    engine = _Engine()
    compiler = WorkflowCompiler(engine)

    agent_id, tools = compiler._match_agent("summarize the findings", "general")

    assert agent_id == "executor"
    assert tools == []
    # A down embedder means even the bounded vector search is skipped.
    assert engine.backend.semantic_calls == 0


def test_match_agent_uses_bounded_semantic_search_when_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_embed(monkeypatch, available=True)
    engine = _Engine()
    compiler = WorkflowCompiler(engine)

    # If the full-scan path were used, _Engine.search_hybrid would raise.
    agent_id, _tools = compiler._match_agent("summarize the findings", "general")

    assert agent_id == "matched-server"
    assert engine.backend.semantic_calls == 1


def test_embed_fn_built_once_per_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counters = _patch_embed(monkeypatch, available=True)
    engine = _Engine()
    compiler = WorkflowCompiler(engine)

    for _ in range(5):
        compiler._match_agent("do a thing", "general")

    # Five steps, but the embedding fn is constructed exactly once and reused.
    assert counters["make_fn"] == 1
    assert engine.backend.semantic_calls == 5


@pytest.mark.asyncio
async def test_compile_completes_with_embedder_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compile() returns a plan (no hang, no full scan) when the embedder is down."""
    _patch_embed(monkeypatch, available=False)
    engine = _Engine()
    compiler = WorkflowCompiler(engine)

    plan = await compiler.compile(
        "search the knowledge graph for servers, then summarize the count",
        domain="general",
    )

    assert plan is not None
    assert plan.steps
    assert engine.backend.semantic_calls == 0
