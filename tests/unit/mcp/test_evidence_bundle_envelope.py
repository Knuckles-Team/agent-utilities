"""Live-path tests for the additive `envelope` param on the LLM-facing MCP tools
(Epistemic Substrate Program, workstream C1 — CONCEPT:evidence-bundle-envelope).

Covers:
  * `graph_ask` / `nl_query` (query_tools.py) — `envelope="raw"` (default/unset)
    stays BYTE-IDENTICAL to the pre-existing behavior; `envelope="bundle"`
    additively attaches an `evidence_bundle` key without touching anything else.
  * `graph_analyze action=code_context` (analysis_tools.py) — same raw/bundle
    contract over `build_code_context`'s output.
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
import json
from typing import Any

import pytest

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


def test_graph_ask_envelope_raw_is_byte_identical(monkeypatch):
    _register_query_tools()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.nl_query.nl_to_query",
        lambda *a, **kw: dict(_CANNED_NL_PAYLOAD),
    )
    expected = json.dumps(_CANNED_NL_PAYLOAD, default=str)

    # envelope entirely unset (the pre-existing call shape).
    default_out = asyncio.run(
        kg_server._execute_tool("graph_ask", question="which agents call run_agent?")
    )
    # envelope explicitly "raw".
    explicit_raw_out = asyncio.run(
        kg_server._execute_tool(
            "graph_ask", question="which agents call run_agent?", envelope="raw"
        )
    )

    assert default_out == expected
    assert explicit_raw_out == expected
    assert "evidence_bundle" not in json.loads(default_out)


def test_graph_ask_envelope_bundle_adds_evidence_bundle_additively(monkeypatch):
    _register_query_tools()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.nl_query.nl_to_query",
        lambda *a, **kw: dict(_CANNED_NL_PAYLOAD),
    )

    out = json.loads(
        asyncio.run(
            kg_server._execute_tool(
                "graph_ask",
                question="which agents call run_agent?",
                envelope="bundle",
            )
        )
    )
    # every raw field is still present, unmodified
    for key, val in _CANNED_NL_PAYLOAD.items():
        assert out[key] == val
    bundle = out["evidence_bundle"]
    assert bundle["claims"] == _CANNED_NL_PAYLOAD["results"]
    assert bundle["evidence_spans"] == [{"ref": "agent:foo"}]
    assert bundle["confidence"] is None


def test_nl_query_envelope_raw_is_byte_identical(monkeypatch):
    _register_query_tools()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.nl_planner.nl_query",
        lambda *a, **kw: dict(_CANNED_NL_PAYLOAD),
    )

    out = asyncio.run(
        kg_server._execute_tool("nl_query", text="which agents call run_agent?")
    )
    assert out == json.dumps(_CANNED_NL_PAYLOAD, default=str)


def test_nl_query_envelope_bundle_wraps_additively(monkeypatch):
    _register_query_tools()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.nl_planner.nl_query",
        lambda *a, **kw: dict(_CANNED_NL_PAYLOAD),
    )

    out = json.loads(
        asyncio.run(
            kg_server._execute_tool(
                "nl_query",
                text="which agents call run_agent?",
                envelope="bundle",
            )
        )
    )
    assert out["generated_query"] == _CANNED_NL_PAYLOAD["generated_query"]
    assert out["evidence_bundle"]["confidence"] is None


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


def _register_graph_analyze():
    from agent_utilities.mcp.tools import analysis_tools

    fake = _FakeMCP()
    analysis_tools.register_analysis_tools(fake)
    assert kg_server.REGISTERED_TOOLS.get("graph_analyze") is fake.tools["graph_analyze"]
    return fake.tools["graph_analyze"]


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


def test_graph_analyze_code_context_raw_is_byte_identical(monkeypatch):
    _register_graph_analyze()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.retrieval.code_context.build_code_context",
        lambda *a, **kw: dict(_CANNED_CODE_CONTEXT),
    )

    out_default = asyncio.run(
        kg_server._execute_tool(
            "graph_analyze", action="code_context", query="how does run_agent work"
        )
    )
    out_explicit_raw = asyncio.run(
        kg_server._execute_tool(
            "graph_analyze",
            action="code_context",
            query="how does run_agent work",
            envelope="raw",
        )
    )
    expected = json.dumps(_CANNED_CODE_CONTEXT, default=str)
    assert out_default == expected
    assert out_explicit_raw == expected


def test_graph_analyze_code_context_bundle_is_additive(monkeypatch):
    _register_graph_analyze()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.retrieval.code_context.build_code_context",
        lambda *a, **kw: dict(_CANNED_CODE_CONTEXT),
    )

    out = json.loads(
        asyncio.run(
            kg_server._execute_tool(
                "graph_analyze",
                action="code_context",
                query="how does run_agent work",
                envelope="bundle",
            )
        )
    )
    for key, val in _CANNED_CODE_CONTEXT.items():
        assert out[key] == val
    bundle = out["evidence_bundle"]
    assert bundle["answer_candidate"] == _CANNED_CODE_CONTEXT["answer"]
    assert bundle["evidence_spans"] == _CANNED_CODE_CONTEXT["citations"]
    assert bundle["confidence"] is None


# ---------------------------------------------------------------------------
# graph_analyze action=executable_rag (brand-new MCP exposure)
# ---------------------------------------------------------------------------
def test_graph_analyze_executable_rag_returns_evidence_bundle(monkeypatch):
    from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
        HybridRetriever,
    )

    _register_graph_analyze()
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

    out = json.loads(
        asyncio.run(
            kg_server._execute_tool(
                "graph_analyze",
                action="executable_rag",
                query="what does run_agent do?",
            )
        )
    )
    # No wrapping toggle needed — the bundle IS the top-level shape.
    assert out["answer_candidate"] == canned.answer
    assert out["evidence_spans"] == [{"id": "n1"}, {"id": "n2"}]
    assert out["confidence"] is None
    assert out["reasoning_trace"][-1] == {"step": "final", "success": True}


def test_graph_analyze_executable_rag_requires_query(monkeypatch):
    _register_graph_analyze()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    out = asyncio.run(
        kg_server._execute_tool("graph_analyze", action="executable_rag", query="")
    )
    assert "needs a question" in out


@pytest.mark.parametrize("envelope_value", ["raw", "", "RAW"])
def test_envelope_values_normalize_to_raw(monkeypatch, envelope_value):
    _register_query_tools()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.nl_query.nl_to_query",
        lambda *a, **kw: dict(_CANNED_NL_PAYLOAD),
    )
    out = asyncio.run(
        kg_server._execute_tool("graph_ask", question="q", envelope=envelope_value)
    )
    assert "evidence_bundle" not in json.loads(out)
