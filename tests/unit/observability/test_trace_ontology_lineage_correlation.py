"""Unit tests for ``trace_ontology.correlate_lineage_run_trace`` (CA-25, DEC-CA-05).

(CONCEPT:AU-KG.ingest.openlineage-consumer)

Kept as its own small module (rather than extending the larger
``test_trace_ontology.py``, which pulls in the full company-brain/session
fixture stack this function does not need) — a pure read-only correlation
lookup over a minimal ``query_cypher`` double.
"""

from __future__ import annotations

from agent_utilities.observability.trace_ontology import (
    TRACE_NODE_LABEL,
    correlate_lineage_run_trace,
    trace_id,
)


class _Engine:
    def __init__(self, known_run_ids: set[str] = frozenset()) -> None:
        self._known = {trace_id(r) for r in known_run_ids}
        self.last_query: str | None = None
        self.last_params: dict | None = None

    def query_cypher(self, query: str, params: dict) -> list[dict]:
        self.last_query = query
        self.last_params = params
        if params.get("run_id") in self._known:
            return [{"run_id": params["run_id"]}]
        return []


class _NoQueryCypherEngine:
    """No query_cypher at all — correlation must fail closed to None, not raise."""


def test_correlate_lineage_run_trace_finds_existing() -> None:
    eng = _Engine(known_run_ids={"tool-run-42"})
    result = correlate_lineage_run_trace(eng, "tool-run-42")
    assert result == trace_id("tool-run-42")
    assert TRACE_NODE_LABEL in eng.last_query


def test_correlate_lineage_run_trace_returns_none_when_absent() -> None:
    eng = _Engine(known_run_ids={"tool-run-42"})
    assert correlate_lineage_run_trace(eng, "spark-run-99") is None


def test_correlate_lineage_run_trace_engine_none() -> None:
    assert correlate_lineage_run_trace(None, "any-run") is None


def test_correlate_lineage_run_trace_empty_run_id() -> None:
    eng = _Engine(known_run_ids={"tool-run-42"})
    assert correlate_lineage_run_trace(eng, "") is None


def test_correlate_lineage_run_trace_missing_query_cypher_fails_closed() -> None:
    """Best-effort: an engine with no query_cypher (or any lookup failure)
    returns None rather than raising — a lineage-only Activity is still
    written by the caller."""
    assert correlate_lineage_run_trace(_NoQueryCypherEngine(), "any-run") is None
