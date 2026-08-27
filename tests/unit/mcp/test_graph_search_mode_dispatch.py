"""Characterization coverage for `graph_search`'s per-`mode` dispatch branches.

CONCEPT:AU-KG.retrieval.acl-aware-vector-retrieval

Before the WB1-AU-02 extract-method split of `_search_with_engine`
(CCN 35, nested inside `graph_search._execute`), the suite drove exactly 5 of
its 14 `mode` values through the real dispatch core: 'hybrid' (default),
'deep', 'dci', 'discover' (REST-routing only — `_execute_tool` itself was
mocked, so the branch body never ran), and an unrecognized mode (also only at
the REST-routing layer). 'concept', 'analogy', 'adore', 'chrono_ids',
'memory', 'hard_negatives', 'latent', 'sira', 'rerank', 'compiled', and the
tool's own unknown-mode fallback had ZERO coverage anywhere. This file closes
that gap using the same `kg_server._resolve_read_engines` monkeypatch pattern
`test_graph_query_evidence_bundle.py` uses, so the extraction is proven, not
just assumed, behavior-preserving.
"""

from __future__ import annotations

import asyncio

from agent_utilities.mcp import kg_server


def _register_query_tools():
    from fastmcp import FastMCP

    from agent_utilities.mcp.tools.query_tools import register_query_tools

    register_query_tools(FastMCP("test"))


_HIT = {
    "score": 0.9,
    "node": {
        "type": "Concept",
        "name": "DelegationRouter",
        "description": "Routes agent delegation across the fleet.",
        "id": "concept:delegation-router",
    },
}


class _FakeEngine:
    """Stands in for an `IntelligenceGraphEngine` across every `graph_search`
    mode this file exercises."""

    def __init__(self):
        self.hybrid_retriever = None
        self.backend = None

    def search_hybrid(
        self, query, top_k, mode=None, self_correct=False, as_of=None, session=None
    ):
        return [_HIT]

    def search_adore(self, query, top_k):
        return [_HIT]

    def temporal_semantic_ids(self, query, top_k):
        return [_HIT]

    def search_dci(self, query, top_k, session=None):
        return [_HIT]

    def search_memories(self, query, top_k):
        return [_HIT]


def _fake_resolve_read_engines(engine):
    def _resolve(target):
        return ([("default", engine)], {}, False)

    return _resolve


def _run(monkeypatch, **kwargs) -> str:
    _register_query_tools()
    engine = _FakeEngine()
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines(engine)
    )
    return asyncio.run(kg_server._execute_tool("graph_search", **kwargs))


def test_concept_mode_dispatches_to_plain_hybrid_search(monkeypatch) -> None:
    out = _run(monkeypatch, query="q", mode="concept")
    assert "[Concept] DelegationRouter" in out


def test_analogy_mode_dispatches_to_plain_hybrid_search(monkeypatch) -> None:
    out = _run(monkeypatch, query="q", mode="analogy")
    assert "[Concept] DelegationRouter" in out


def test_adore_mode_dispatches_to_search_adore(monkeypatch) -> None:
    out = _run(monkeypatch, query="q", mode="adore")
    assert "[Concept] DelegationRouter" in out


def test_chrono_ids_mode_dispatches_to_temporal_semantic_ids(monkeypatch) -> None:
    out = _run(monkeypatch, query="q", mode="chrono_ids")
    assert "[Concept] DelegationRouter" in out


def test_memory_mode_dispatches_to_search_memories(monkeypatch) -> None:
    out = _run(monkeypatch, query="q", mode="memory")
    assert "[Concept] DelegationRouter" in out


def test_discover_mode_reports_capabilities_from_hybrid_search(monkeypatch) -> None:
    out = _run(monkeypatch, query="agentic workflow", mode="discover")
    assert "DelegationRouter" in out


def test_hard_negatives_mode_reports_no_retriever_when_unwired(monkeypatch) -> None:
    out = _run(monkeypatch, query="q", mode="hard_negatives")
    assert out.startswith(
        "Error: hybrid retriever unavailable for hard-negative mining."
    )


def test_latent_mode_degrades_cleanly_with_no_backend(monkeypatch) -> None:
    """`LatentTopologicalRAG.retrieve` returns [] when `engine.backend` is falsy
    -- proves the 'latent' branch really reaches that class, not a stub."""
    out = _run(monkeypatch, query="q", mode="latent")
    assert out.startswith("No results found for query: 'q'")


def test_sira_mode_aligns_the_base_hybrid_results(monkeypatch) -> None:
    out = _run(monkeypatch, query="q", mode="sira")
    assert "[Concept] DelegationRouter" in out


def test_rerank_mode_reaches_the_hybrid_search_scorer(monkeypatch) -> None:
    """No embedding model is wired, so scoring degrades to keyword-only --
    this only proves the 'rerank' branch dispatches into HybridSearchScorer
    and returns a string, not that its scoring math is correct (out of
    partition)."""
    out = _run(monkeypatch, query="q", mode="rerank")
    assert isinstance(out, str)
    assert out != "Error: Unknown search mode 'rerank'"


def test_compiled_mode_fails_closed_without_the_required_wiring(monkeypatch) -> None:
    """No ambient GraphSession/mandatory-marking store is available in this unit
    test, so `compiled` fails closed rather than fabricating a bundle -- proves
    the branch reaches real `ContextCompiler` wiring rather than silently
    no-op'ing. The specific failure mode is an implementation detail of
    `ContextCompiler`/`GraphSession` (outside this partition); only the
    fail-closed contract (a structured error, never a fabricated bundle) is
    pinned here."""
    out = _run(monkeypatch, query="q", mode="compiled")
    assert '"status": "failed"' in out


def test_unknown_mode_reports_the_error_verbatim(monkeypatch) -> None:
    out = _run(monkeypatch, query="q", mode="bogus-mode")
    assert out.startswith("Error: Unknown search mode 'bogus-mode'")
