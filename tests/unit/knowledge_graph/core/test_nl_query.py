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
        if query.startswith("MATCH (n) RETURN n.type"):
            return [{"t": "Agent", "nt": None, "lb": None}, {"t": "Service"}]
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


def test_schema_context_grounds_label_column_and_real_edge_columns():
    """CONCEPT:AU-KG.query.ask-gateway-rest-twin grounding fix.

    The SQL surface has no distinct "label" column — a cypher label is really the
    node's ``type`` property (``label`` is a separate, usually-empty property). The
    schema hint the planner receives must say so explicitly, and must describe the
    edges table's REAL fixed columns (``src``/``dst``/``rel``/``props``) instead of
    letting the model invent a column like ``source_node_id``.
    """
    eng = _FakeEngine()
    schema = nl_query.build_schema_context(eng)

    assert "Agent" in schema["node_labels"]
    assert "Service" in schema["node_labels"]

    columns_note = schema["sql_columns"]
    # the label/kind column mapping.
    assert "`type`" in columns_note
    assert "Do NOT use a `label` column" in columns_note
    # the real edges columns, and an explicit rejection of invented ones.
    assert "`src`" in columns_note
    assert "`dst`" in columns_note
    assert "`rel`" in columns_note
    assert "source_node_id" in columns_note  # named as the wrong column to avoid


def test_schema_context_label_probe_uses_type_property_not_labels_function():
    """The engine's Cypher executor does not implement ``labels(n)`` as a callable
    expression (it evaluates to null on every row), so the probe must read the
    scalar ``type``/``node_type``/``label`` properties directly instead.
    """
    eng = _FakeEngine()
    nl_query.build_schema_context(eng)
    # never issued the broken labels(n) query
    assert eng.cypher_seen != "MATCH (n) RETURN DISTINCT labels(n) AS labels LIMIT 200"


def test_concept_count_question_grounds_on_type_column(monkeypatch):
    """Regression for the silent-wrong-data bug: 'how many Concept nodes' must be
    grounded on the `type` column so the model doesn't repeat the old
    `WHERE label = 'Concept'` mistake (which always returned 0 against the real
    engine schema, even though the label genuinely exists as `type='Concept'`).
    """
    captured_prompt: dict[str, str] = {}

    class _FakeAgent:
        def __init__(self, *a, **k):
            pass

        def run_sync(self, prompt):
            captured_prompt["text"] = prompt
            return _FakeResult(
                '{"dialect": "sql", "query": "SELECT COUNT(*) FROM nodes WHERE type = \'Concept\'"}'
            )

    monkeypatch.setattr("pydantic_ai.Agent", _FakeAgent)
    monkeypatch.setattr(
        "agent_utilities.core.model_factory.create_model", lambda **k: object()
    )

    class _ConceptEngine(_FakeEngine):
        def sql(self, query):
            self.sql_seen = query
            if "type = 'Concept'" in query:
                return [{"count(*)": 13792}]
            return [{"count(*)": 0}]

    eng = _ConceptEngine()
    out = nl_query.nl_to_query(eng, "how many Concept nodes")

    # the prompt grounds the model on the real column before it ever calls the LLM.
    assert "`type`" in captured_prompt["text"]
    assert out["generated_query"] == "SELECT COUNT(*) FROM nodes WHERE type = 'Concept'"
    assert out["results"] == [{"count(*)": 13792}]
