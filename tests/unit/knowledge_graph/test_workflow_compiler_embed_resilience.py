"""WorkflowCompiler degrades gracefully when the embedding endpoint is down.

CONCEPT:ORCH-1.23 — NL workflow compilation. The semantic agent-matching
fallback (`_match_agent` → `engine.search_hybrid`) embeds the step text; when
the embedder is unreachable (e.g. the GB10/vLLM power fault) an unbounded embed
call stalls compilation. These tests pin the bounded-probe behavior: one cached
probe decides embedder health for the whole compile, and the hybrid fallback is
skipped entirely when it is down — so compilation never hangs on a dead endpoint.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.workflow_compiler import WorkflowCompiler


class _Backend:
    """Structural KG query layer that never matches (forces the fallback)."""

    def execute(self, _cypher: str, _params: dict[str, Any]) -> list[dict[str, Any]]:
        return []


class _Engine:
    """Minimal engine stub exposing the surface `_match_agent` touches."""

    def __init__(self) -> None:
        self.backend = _Backend()
        self.hybrid_calls = 0

    def search_hybrid(self, _text: str, top_k: int = 3) -> list[dict[str, Any]]:
        self.hybrid_calls += 1
        return [{"name": "matched-server", "resource_type": "Server"}]


def _patch_embed(monkeypatch: pytest.MonkeyPatch, *, available: bool) -> dict[str, int]:
    """Stub the embedder so no real endpoint is contacted; count probes."""
    counters = {"probes": 0}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.semantic.make_embed_fn",
        lambda *a, **k: lambda texts: [[0.0] for _ in texts],
    )

    def _bounded_embed(_embed_fn: Any, _text: str, _timeout: float):
        counters["probes"] += 1
        return [0.0] if available else None

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.research.search.bounded_embed",
        _bounded_embed,
    )
    return counters


def test_match_agent_skips_hybrid_when_embedder_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_embed(monkeypatch, available=False)
    engine = _Engine()
    compiler = WorkflowCompiler(engine)

    agent_id, tools = compiler._match_agent("summarize the findings", "general")

    assert agent_id == "executor"
    assert tools == []
    # The whole point: a dead embedder means the hybrid call is never made.
    assert engine.hybrid_calls == 0


def test_match_agent_uses_hybrid_when_embedder_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_embed(monkeypatch, available=True)
    engine = _Engine()
    compiler = WorkflowCompiler(engine)

    agent_id, _tools = compiler._match_agent("summarize the findings", "general")

    assert agent_id == "matched-server"
    assert engine.hybrid_calls == 1


def test_embed_probe_is_cached_once_per_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counters = _patch_embed(monkeypatch, available=False)
    engine = _Engine()
    compiler = WorkflowCompiler(engine)

    for _ in range(5):
        compiler._match_agent("do a thing", "general")

    # Five steps, but the bounded embedder probe runs exactly once.
    assert counters["probes"] == 1
    assert engine.hybrid_calls == 0


@pytest.mark.asyncio
async def test_compile_completes_with_embedder_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end compile() returns a plan (no hang) when the embedder is down."""
    _patch_embed(monkeypatch, available=False)
    engine = _Engine()
    compiler = WorkflowCompiler(engine)

    plan = await compiler.compile(
        "search the knowledge graph for servers, then summarize the count",
        domain="general",
    )

    assert plan is not None
    assert plan.steps
    assert engine.hybrid_calls == 0
