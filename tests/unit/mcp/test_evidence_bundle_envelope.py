"""Live-path tests for the current EvidenceBundle LLM-facing MCP contract.

Covers:
  * `graph_ask` / `nl_query` return an EvidenceBundle directly.
  * `graph_analyze action=code_context` returns the same current contract.
  * `graph_analyze action=executable_rag` (analysis_tools.py) — brand-new MCP
    exposure of the executable-RAG interpreter; no legacy consumer, so it
    returns the EvidenceBundle directly (no wrapping toggle needed).

Every surface is driven through the REAL registered tool coroutine/function
(mirroring the live-path pattern used by test_memory_weights_distillation.py and
test_graph_query_sql.py), with the underlying engine/LLM calls monkeypatched out
so the tests run with no live engine or model.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_utilities.knowledge_graph.retrieval.executable_rag import RagResult
from agent_utilities.mcp import kg_server


# ---------------------------------------------------------------------------
# graph_ask / nl_query (query_tools.py)
#
# Driven through kg_server._execute_tool (not the raw registered function) —
# that's the one path (shared by the real MCP protocol layer, the REST twins,
# and _execute_tool itself) that resolves an omitted `Field(default=...)` param
# to its actual default; calling the raw function directly would instead bind
# the omitted param to the FieldInfo object itself (see _execute_tool's own
# docstring/comment), which is exactly the "unset" case this regression test
# needs to exercise correctly.
# ---------------------------------------------------------------------------
def _register_query_tools():
    from fastmcp import FastMCP

    from agent_utilities.mcp.tools.query_tools import register_query_tools

    register_query_tools(FastMCP("test"))


_CANNED_NL_PAYLOAD = {
    "question": "which agents call run_agent?",
    "dialect": "cypher",
    "generated_query": "MATCH (a:Agent)-[:CALLS]->(f {name:'run_agent'}) RETURN a",
    "results": [{"id": "agent:foo", "name": "foo"}],
    "row_count": 1,
    "citations": ["agent:foo"],
    "schema": {"node_labels": ["Agent"]},
}


def test_graph_ask_returns_typed_bundle(monkeypatch):
    _register_query_tools()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.nl_query.nl_to_query",
        lambda *a, **kw: dict(_CANNED_NL_PAYLOAD),
    )
    out = asyncio.run(
        kg_server._execute_tool("graph_ask", question="which agents call run_agent?")
    ).model_dump()
    assert out["claims"] == _CANNED_NL_PAYLOAD["results"]
    assert out["evidence_spans"] == [{"ref": "agent:foo"}]
    assert out["confidence"] is None
    assert out["reasoning_trace"][-1]["generated_query"] == (
        _CANNED_NL_PAYLOAD["generated_query"]
    )


def test_nl_query_returns_typed_bundle(monkeypatch):
    _register_query_tools()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.nl_planner.nl_query",
        lambda *a, **kw: dict(_CANNED_NL_PAYLOAD),
    )

    out = asyncio.run(
        kg_server._execute_tool("nl_query", text="which agents call run_agent?")
    ).model_dump()
    assert out["claims"] == _CANNED_NL_PAYLOAD["results"]
    assert out["confidence"] is None
    assert out["reasoning_trace"][-1]["generated_query"] == (
        _CANNED_NL_PAYLOAD["generated_query"]
    )


# ---------------------------------------------------------------------------
# graph_analyze action=code_context (analysis_tools.py)
# ---------------------------------------------------------------------------
class _FakeMCP:
    """Captures the tool coroutines ``register_analysis_tools`` registers."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, *, name: str, description: str = "", tags: Any = None):
        def _decorator(fn):
            self.tools[name] = fn
            return fn

        return _decorator


def _register_analysis_suite():
    from agent_utilities.mcp.tools import analysis_tools, analyze_suite

    fake = _FakeMCP()
    analysis_tools.register_analysis_tools(fake)
    analyze_suite.register_analyze_suite_tools(fake)
    assert kg_server.REGISTERED_TOOLS.get("graph_analyze") is fake.tools["graph_analyze"]
    return fake


_CANNED_CODE_CONTEXT = {
    "status": "ok",
    "query": "how does run_agent work",
    "intent": "how",
    "answer": "`run_agent` (function) is defined at a.py:10.",
    "citations": [
        {
            "id": "code:a.py::run_agent",
            "symbol": "run_agent",
            "file": "a.py",
            "line": 10,
            "kind": "function",
            "language": "python",
            "source_system": "agent-utilities",
        }
    ],
    "sections": {},
    "anchors": [{"symbol": "run_agent", "file": "a.py", "line": 10}],
    "capability_id": "code_context:how:run_agent",
    "used_primitives": [],
    "cross_repo": False,
    "coverage": {"anchors": 1},
}


def test_graph_code_context_returns_typed_bundle(monkeypatch):
    _register_analysis_suite()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.retrieval.code_context.build_code_context",
        lambda *a, **kw: dict(_CANNED_CODE_CONTEXT),
    )

    out = asyncio.run(
        kg_server._execute_tool(
            "graph_code", action="code_context", query="how does run_agent work"
        )
    ).model_dump()
    bundle = out
    assert bundle["answer_candidate"] == _CANNED_CODE_CONTEXT["answer"]
    assert bundle["evidence_spans"] == _CANNED_CODE_CONTEXT["citations"]
    assert bundle["confidence"] is None


# ---------------------------------------------------------------------------
# graph_analyze action=executable_rag (brand-new MCP exposure)
# ---------------------------------------------------------------------------
def test_graph_explain_executable_rag_returns_evidence_bundle(monkeypatch):
    from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
        HybridRetriever,
    )

    _register_analysis_suite()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())

    canned = RagResult(
        answer="run_agent orchestrates a single agent turn.",
        evidence_ids=["n1", "n2"],
        trace=[],
        success=True,
    )
    monkeypatch.setattr(
        HybridRetriever,
        "retrieve_executable",
        lambda self, query, **kw: canned,
    )

    out = asyncio.run(
        kg_server._execute_tool(
            "graph_explain",
            action="executable_rag",
            query="what does run_agent do?",
        )
    ).model_dump()
    # No wrapping toggle needed — the bundle IS the top-level shape.
    assert out["answer_candidate"] == canned.answer
    assert out["evidence_spans"] == [{"id": "n1"}, {"id": "n2"}]
    assert out["confidence"] is None
    assert out["reasoning_trace"][-1] == {"step": "final", "success": True}


def test_graph_explain_executable_rag_requires_query(monkeypatch):
    _register_analysis_suite()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    out = asyncio.run(
        kg_server._execute_tool("graph_explain", action="executable_rag", query="")
    )
    assert "needs a question" in out.answer_candidate
