"""Natural-language → query (CONCEPT:AU-KG.ingest.mirror-inbound).

The LLM call is mocked (fake pydantic-ai Agent + create_model) so we assert the layer
parses the model's dialect/query, routes execution to the right engine surface, blocks
mutations, and returns the generated query + results + citations (grounded).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core import nl_query

pytestmark = pytest.mark.concept("AU-KG.ingest.mirror-inbound")


class _FakeResult:
    def __init__(self, output):
        self.output = output


def _patch_llm(monkeypatch, payload: str):
    """Make Agent.run_sync return ``payload`` and create_model a no-op."""

    class _FakeAgent:
        def __init__(self, *a, **k):
            pass

        def run_sync(self, prompt):
            return _FakeResult(payload)

    monkeypatch.setattr("pydantic_ai.Agent", _FakeAgent)
    monkeypatch.setattr(
        "agent_utilities.core.model_factory.create_model", lambda **k: object()
    )


class _FakeEngine:
    def __init__(self):
        self.sql_seen = self.cypher_seen = self.sparql_seen = None

    def query_cypher(self, query, *a, **k):
        if query.startswith("MATCH (n) RETURN DISTINCT labels"):
            return [{"labels": ["Agent"]}, {"labels": ["Service"]}]
        self.cypher_seen = query
        return [{"id": "n1", "name": "alpha"}]

    def sql(self, query):
        self.sql_seen = query
        return [{"id": "row1"}]

    def sparql(self, query):
        self.sparql_seen = query
        return [{"s": "iri1"}]


def test_nl_to_cypher_executes_and_cites(monkeypatch):
    _patch_llm(
        monkeypatch, '{"dialect": "cypher", "query": "MATCH (a:Agent) RETURN a"}'
    )
    eng = _FakeEngine()
    out = nl_query.nl_to_query(eng, "list the agents")
    assert out["dialect"] == "cypher"
    assert out["generated_query"] == "MATCH (a:Agent) RETURN a"
    assert out["results"] == [{"id": "n1", "name": "alpha"}]
    assert out["citations"] == ["n1"]
    assert eng.cypher_seen == "MATCH (a:Agent) RETURN a"
    # schema was grounded from the live engine
    assert "Agent" in out["schema"]["node_labels"]


def test_nl_to_sql_routes_to_sql_surface(monkeypatch):
    _patch_llm(monkeypatch, '{"dialect": "sql", "query": "SELECT id FROM nodes"}')
    eng = _FakeEngine()
    out = nl_query.nl_to_query(eng, "how many nodes")
    assert out["dialect"] == "sql"
    assert eng.sql_seen == "SELECT id FROM nodes"
    assert out["row_count"] == 1


def test_nl_to_sparql_routes_to_sparql_surface(monkeypatch):
    _patch_llm(
        monkeypatch, '{"dialect": "sparql", "query": "SELECT ?s WHERE { ?s a ?o }"}'
    )
    eng = _FakeEngine()
    out = nl_query.nl_to_query(eng, "all subjects")
    assert out["dialect"] == "sparql"
    assert eng.sparql_seen.startswith("SELECT ?s")


def test_preview_does_not_execute(monkeypatch):
    _patch_llm(monkeypatch, '{"dialect": "cypher", "query": "MATCH (n) RETURN n"}')
    eng = _FakeEngine()
    out = nl_query.nl_to_query(eng, "everything", execute=False)
    assert "results" not in out
    assert out["generated_query"] == "MATCH (n) RETURN n"


def test_generated_mutation_is_refused(monkeypatch):
    _patch_llm(monkeypatch, '{"dialect": "cypher", "query": "MATCH (n) DELETE n"}')
    eng = _FakeEngine()
    out = nl_query.nl_to_query(eng, "delete everything")
    assert "mutation" in out["error"]
    assert "results" not in out


def test_forced_dialect_overrides_model(monkeypatch):
    _patch_llm(monkeypatch, '{"dialect": "cypher", "query": "SELECT id FROM nodes"}')
    eng = _FakeEngine()
    out = nl_query.nl_to_query(eng, "rows", dialect="sql")
    assert out["dialect"] == "sql"
    assert eng.sql_seen == "SELECT id FROM nodes"


def test_unparseable_model_output_errors(monkeypatch):
    _patch_llm(monkeypatch, "I cannot help with that")
    eng = _FakeEngine()
    out = nl_query.nl_to_query(eng, "nonsense")
    assert "generation failed" in out["error"]
