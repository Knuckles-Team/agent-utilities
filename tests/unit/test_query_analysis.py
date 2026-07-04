"""Tests for query analysis: source/time filters + citations (CONCEPT:AU-ECO.connector.apply-any-query-analysis)."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.retrieval.query_analysis import (
    CitationProcessor,
    analyze_query,
    filter_nodes_by_source,
)


@pytest.mark.concept("AU-ECO.connector.apply-any-query-analysis")
def test_detects_source_types_and_time_window():
    f = analyze_query("show me recent arxiv papers about graphs")
    assert "paper" in f.source_types
    assert f.as_of is not None  # "recent" → a time window
    assert f.time_range == "recent"


@pytest.mark.concept("AU-ECO.connector.apply-any-query-analysis")
def test_detects_last_n_window():
    f = analyze_query("list tickets from the last 3 weeks")
    assert "ticket" in f.source_types
    assert f.as_of is not None
    assert "3 week" in f.time_range


@pytest.mark.concept("AU-ECO.connector.apply-any-query-analysis")
def test_no_filters_for_plain_query():
    f = analyze_query("what is the architecture of the system")
    assert f.source_types == []
    assert f.as_of is None


@pytest.mark.concept("AU-ECO.connector.apply-any-query-analysis")
def test_llm_path_augments_sources():
    def llm(prompt: str) -> str:
        return '{"source_types": ["email"], "since_days": 5}'

    f = analyze_query("anything", llm_fn=llm)
    assert "email" in f.source_types
    assert f.as_of is not None


@pytest.mark.concept("AU-ECO.connector.apply-any-query-analysis")
def test_filter_nodes_by_source_keeps_unclassified():
    nodes = [
        {"id": "1", "doc_type": "paper"},
        {"id": "2", "doc_type": "email"},
        {"id": "3"},  # no type → kept
    ]
    kept = {n["id"] for n in filter_nodes_by_source(nodes, ["paper"])}
    assert kept == {"1", "3"}
    assert filter_nodes_by_source(nodes, []) == nodes  # no filter → passthrough


@pytest.mark.concept("AU-ECO.connector.apply-any-query-analysis")
def test_citation_processor_builds_and_annotates():
    nodes = [
        {"id": "1", "name": "Doc A", "source_url": "http://a"},
        {"id": "2", "name": "Doc B", "source": "http://b"},
    ]
    cp = CitationProcessor()
    cites = cp.build_citations(nodes)
    assert cites[0]["n"] == 1 and cites[0]["source"] == "http://a"
    annotated = cp.annotate("See [1] and [2] and [9].", nodes)
    assert "[1](http://a)" in annotated
    assert "[2](http://b)" in annotated
    assert "[9]" in annotated  # out-of-range marker left intact


@pytest.mark.concept("AU-ECO.connector.apply-any-query-analysis")
def test_retriever_accepts_query_analysis_flag():
    import inspect

    from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
        HybridRetriever,
    )

    params = inspect.signature(HybridRetriever.retrieve_hybrid).parameters
    assert "query_analysis" in params
