"""Characterization coverage for `graph_document_tree`'s non-spine actions.

CONCEPT:AU-KG.retrieval.section-tree / tree-navigation

`tests/unit/knowledge_graph/ingestion/test_evidence_spine_contract.py` already
covers the 'fragments'/'cite' actions in depth, but nothing in the suite drove
'build'/'structure'/'content'/'retrieve' (or the unknown-action fallback)
through the tool's dispatch core before the WB1-AU-02 extract-method split of
`graph_document_tree` (CCN 39). This file closes that gap so the split is
proven, not just assumed, behavior-preserving.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_utilities.mcp import kg_server

TOOL = "graph_document_tree"

DOC = "# Title\n\nHello world.\n\n## Sub\n\nMore detail about the sub-section.\n"


@pytest.fixture
def tool(monkeypatch: pytest.MonkeyPatch):
    """Invoke the action through `_execute_tool` -- the ONE dispatch core
    (same pattern as test_evidence_spine_contract.py's `tool` fixture)."""
    kg_server.ensure_tools_registered()
    monkeypatch.setattr(kg_server, "_get_engine", lambda *a, **k: None)

    def call(**kwargs: object) -> str:
        return asyncio.run(kg_server._execute_tool(TOOL, **kwargs))

    return call


def test_build_action_returns_structure_without_persisting_when_no_engine(
    tool,
) -> None:
    out = json.loads(tool(action="build", text=DOC))
    assert out["action"] == "build"
    assert out["section_count"] > 0
    assert out["persisted"] is False
    assert out["structure"]


def test_build_action_requires_text_or_document_id(tool) -> None:
    out = json.loads(tool(action="build"))
    assert "requires 'text' or 'document_id'" in out["error"]


def test_build_refuses_ungoverned_partial_writes(monkeypatch) -> None:
    from agent_utilities.mcp.tools.query_tools import register_query_tools

    class ToolRegistry:
        def __init__(self):
            self.tools = {}

        def tool(self, *, name, **_kwargs):
            def register(func):
                self.tools[name] = func
                return func

            return register

    class GuardEngine:
        def add_node(self, *_args, **_kwargs):
            pytest.fail("partial section node write reached")

        def add_edge(self, *_args, **_kwargs):
            pytest.fail("partial section edge write reached")

    registry = ToolRegistry()
    register_query_tools(registry)
    monkeypatch.setattr(kg_server, "_get_engine", GuardEngine)
    build = registry.tools["graph_document_tree"]
    denied = json.loads(
        build(action="build", text=DOC, document_id="doc:1", persist=True)
    )
    assert "verified source access" in denied["error"]

    preview = json.loads(
        build(
            action="build",
            text=DOC,
            document_id="doc:1",
            persist=False,
            thin=False,
            summarize=False,
        )
    )
    assert preview["section_count"] > 0
    assert preview["persisted"] is False


def test_legacy_pipeline_requires_native_document_writer() -> None:
    from agent_utilities.knowledge_graph.pipeline.document_ingestion import (
        DocumentIngestionPipeline,
    )

    class GuardEngine:
        def add_node(self, *_args, **_kwargs):
            pytest.fail("partial section node write reached")

        def add_edge(self, *_args, **_kwargs):
            pytest.fail("partial section edge write reached")

    pipeline = DocumentIngestionPipeline(GuardEngine())
    with pytest.raises(PermissionError, match="verified source access"):
        pipeline.build_section_tree_for("doc:1", DOC, persist=True)
    preview = pipeline.build_section_tree_for("doc:1", DOC, persist=False)
    assert preview["section_count"] > 0
    assert preview["persisted"] is False


def test_structure_action_requires_document_id(tool) -> None:
    out = json.loads(tool(action="structure"))
    assert "requires 'document_id'" in out["error"]


def test_structure_action_requires_an_active_engine(tool) -> None:
    out = json.loads(tool(action="structure", document_id="doc:1"))
    assert out["error"] == "IntelligenceGraphEngine not active"


def test_content_action_requires_document_id(tool) -> None:
    out = json.loads(tool(action="content"))
    assert "requires 'document_id'" in out["error"]


def test_content_action_requires_an_active_engine(tool) -> None:
    out = json.loads(tool(action="content", document_id="doc:1", ranges="0..5"))
    assert out["error"] == "IntelligenceGraphEngine not active"


def test_retrieve_action_requires_query(tool) -> None:
    out = json.loads(tool(action="retrieve", text=DOC))
    assert "requires 'query'" in out["error"]


def test_retrieve_action_requires_text_or_document_id(tool) -> None:
    out = json.loads(tool(action="retrieve", query="sub"))
    assert "requires 'text' or 'document_id'" in out["error"]


def test_retrieve_action_walks_an_inline_tree(tool) -> None:
    out = json.loads(tool(action="retrieve", text=DOC, query="Sub"))
    assert out["action"] == "retrieve"
    assert out["query"] == "Sub"
    assert isinstance(out["results"], list)


def test_stored_retrieve_uses_native_citations(monkeypatch) -> None:
    from agent_utilities.mcp.tools.query_tools import register_query_tools

    class ToolRegistry:
        def __init__(self):
            self.tools = {}

        def tool(self, *, name, **_kwargs):
            def register(func):
                self.tools[name] = func
                return func

            return register

    class NativeCompute:
        def retrieve_document_sections(self, document_id, query, *, top_k):
            assert (document_id, query, top_k) == ("doc:1", "heading", 2)
            return {
                "document_id": "doc:1",
                "citations": [
                    {"node_id": "section:1", "char_start": 4, "char_end": 12}
                ],
            }

    registry = ToolRegistry()
    register_query_tools(registry)
    monkeypatch.setattr(
        kg_server,
        "_get_engine",
        lambda: type("Engine", (), {"graph_compute": NativeCompute()})(),
    )
    result = json.loads(
        registry.tools[TOOL](
            action="retrieve", document_id="doc:1", text="", query="heading", top_k=2
        )
    )
    assert result["results"] == [
        {"node_id": "section:1", "char_start": 4, "char_end": 12}
    ]


def test_stored_retrieve_does_not_load_legacy_sections(monkeypatch) -> None:
    from agent_utilities.mcp.tools.query_tools import register_query_tools

    class ToolRegistry:
        def __init__(self):
            self.tools = {}

        def tool(self, *, name, **_kwargs):
            def register(func):
                self.tools[name] = func
                return func

            return register

    registry = ToolRegistry()
    register_query_tools(registry)
    monkeypatch.setattr(
        kg_server,
        "_get_engine",
        lambda: type("Engine", (), {"graph_compute": object()})(),
    )
    result = json.loads(
        registry.tools[TOOL](
            action="retrieve", document_id="doc:1", text="", query="heading"
        )
    )
    assert result["error"] == "native document section retrieval unavailable"


def test_stored_structure_and_content_do_not_load_legacy_sections(monkeypatch) -> None:
    from agent_utilities.mcp.tools.query_tools import register_query_tools

    class ToolRegistry:
        def __init__(self):
            self.tools = {}

        def tool(self, *, name, **_kwargs):
            def register(func):
                self.tools[name] = func
                return func

            return register

    registry = ToolRegistry()
    register_query_tools(registry)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    tool = registry.tools[TOOL]
    structure = json.loads(tool(action="structure", document_id="doc:1"))
    content = json.loads(tool(action="content", document_id="doc:1", ranges="0..4"))
    assert "native verified read method" in structure["error"]
    assert "native verified read method" in content["error"]


def test_unknown_action_reports_the_expected_action_set(tool) -> None:
    out = json.loads(tool(action="bogus"))
    assert "unknown action 'bogus'" in out["error"]
    assert "build|structure|content|retrieve|fragments|cite" in out["error"]
