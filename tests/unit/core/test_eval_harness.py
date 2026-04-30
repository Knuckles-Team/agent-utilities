"""Tests for the Evaluation Harness.

Concept: eval-tracing
"""

from __future__ import annotations

import json

import networkx as nx
import pytest

from agent_utilities.tools.eval_harness import (
    EvalHarness,
    JsonSchemaScorer,
    LengthScorer,
    RegexScorer,
)


@pytest.mark.concept("eval-tracing")
def test_length_scorer_pass() -> None:
    scorer = LengthScorer(min_length=5, max_length=100)
    result = scorer.score("input", "Hello, world!")
    assert result.passed is True
    assert result.score == 1.0
    assert result.metrics["length"] == 13


@pytest.mark.concept("eval-tracing")
def test_length_scorer_fail() -> None:
    scorer = LengthScorer(min_length=50)
    result = scorer.score("input", "short")
    assert result.passed is False
    assert result.score == 0.0
    assert "outside" in result.reason


@pytest.mark.concept("eval-tracing")
def test_regex_scorer_match() -> None:
    scorer = RegexScorer(pattern=r"\d{3}-\d{4}")
    result = scorer.score("input", "Call 555-1234 now")
    assert result.passed is True
    assert result.score == 1.0


@pytest.mark.concept("eval-tracing")
def test_regex_scorer_no_match() -> None:
    scorer = RegexScorer(pattern=r"^ERROR:")
    result = scorer.score("input", "All good here")
    assert result.passed is False
    assert "not found" in result.reason


@pytest.mark.concept("eval-tracing")
def test_json_schema_scorer_valid() -> None:
    scorer = JsonSchemaScorer(required_keys=["name", "age"])
    output = json.dumps({"name": "Alice", "age": 30, "extra": True})
    result = scorer.score("input", output)
    assert result.passed is True
    assert result.score == 1.0


@pytest.mark.concept("eval-tracing")
def test_json_schema_scorer_missing_keys() -> None:
    scorer = JsonSchemaScorer(required_keys=["name", "age", "email"])
    output = json.dumps({"name": "Alice"})
    result = scorer.score("input", output)
    assert result.passed is False
    assert "age" in result.reason
    assert "email" in result.reason


@pytest.mark.concept("eval-tracing")
def test_eval_harness_runs_all_scorers() -> None:
    harness = EvalHarness()
    harness.register(LengthScorer(min_length=1))
    harness.register(RegexScorer(pattern=r"hello"))

    result = harness.evaluate("q", "hello world")
    assert len(result.results) == 2
    assert result.all_passed is True
    assert result.overall_score == 1.0


@pytest.mark.concept("eval-tracing")
def test_eval_harness_mixed_results() -> None:
    harness = EvalHarness()
    harness.register(LengthScorer(min_length=1))
    harness.register(RegexScorer(pattern=r"^MAGIC$"))

    result = harness.evaluate("q", "hello world")
    assert len(result.results) == 2
    assert result.all_passed is False
    assert 0.0 < result.overall_score < 1.0


@pytest.mark.concept("eval-tracing")
def test_eval_harness_persists_to_graph() -> None:
    harness = EvalHarness()
    harness.register(LengthScorer(min_length=1))

    result = harness.evaluate("q", "hello")
    graph = nx.MultiDiGraph()
    node_id = harness.persist_to_graph(graph, result, run_id="test_run")

    assert node_id in graph
    data = graph.nodes[node_id]
    assert data["type"] == "eval"
    assert data["overall_score"] == 1.0
    assert data["run_id"] == "test_run"
