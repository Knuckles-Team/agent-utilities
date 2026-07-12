"""AU-as-engine NL→query planner (CONCEPT:AU-KG.query.ask-gateway-rest-twin).

The fleet LLM is substituted (an injected ``run`` / a monkeypatched ``create_model`` +
``Agent``) so we assert the KG-2.305 provider: the AU-configured model produces a query
STRING, the layer routes it to the right AU→engine surface (uql/cypher/sql/sparql),
returns the generated query + results + citations, blocks mutations, previews without
executing, and — the additive contract — falls back to a CLEAN error when no LLM is
configured. Also asserts the ``nl_query`` MCP tool is registered.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.core import nl_planner

pytestmark = pytest.mark.concept("AU-KG.query.ask-gateway-rest-twin")


class _FakeEngine:
    """Minimal engine exposing the AU→engine query surfaces the planner executes over."""

    def __init__(self):
        self.uql_seen = self.sql_seen = self.cypher_seen = self.sparql_seen = None

    def query_cypher(self, query, *a, **k):
        if query.startswith("MATCH (n) RETURN n.type"):
            return [{"t": "Agent", "nt": None, "lb": None}, {"t": "Service"}]
        self.cypher_seen = query
        return [{"id": "n1", "name": "alpha"}]

    def uql(self, query):
        self.uql_seen = query
        return [{"id": "doc7", "score": 0.91}, {"id": "doc3", "score": 0.42}]

    def sql(self, query):
        self.sql_seen = query
        return [{"id": "row1"}]

    def sparql(self, query):
        self.sparql_seen = query
        return [{"s": "iri1"}]


def _canned(payload: str) -> nl_planner.AuNlPlanner:
    """A KG-2.305 planner whose fleet-LLM call is a canned response (no network)."""
    return nl_planner.AuNlPlanner(run=lambda prompt, system: payload)


def test_kg_2_305_nl_to_uql_executes_and_cites():
    eng = _FakeEngine()
    out = nl_planner.nl_query(
        eng,
        "top agents by relevance",
        planner=_canned('{"dialect": "uql", "query": "MATCH (:Agent) |> LIMIT 5"}'),
    )
    assert out["dialect"] == "uql"
    assert out["planner"] == "agent-utilities-fleet-llm"
    assert out["generated_query"] == "MATCH (:Agent) |> LIMIT 5"
    assert eng.uql_seen == "MATCH (:Agent) |> LIMIT 5"
    assert out["results"] == [
        {"id": "doc7", "score": 0.91},
        {"id": "doc3", "score": 0.42},
    ]
    assert out["citations"] == ["doc7", "doc3"]
    # schema grounded from the live engine
    assert "Agent" in out["schema"]["node_labels"]


def test_kg_2_305_nl_to_cypher_routes_to_cypher_surface():
    eng = _FakeEngine()
    out = nl_planner.nl_query(
        eng,
        "list agents",
        planner=_canned('{"dialect": "cypher", "query": "MATCH (a:Agent) RETURN a"}'),
    )
    assert out["dialect"] == "cypher"
    assert eng.cypher_seen == "MATCH (a:Agent) RETURN a"
    assert out["citations"] == ["n1"]


def test_kg_2_305_nl_to_sql_routes_to_sql_surface():
    eng = _FakeEngine()
    out = nl_planner.nl_query(
        eng,
        "count nodes",
        planner=_canned('{"dialect": "sql", "query": "SELECT id FROM nodes"}'),
    )
    assert out["dialect"] == "sql"
    assert eng.sql_seen == "SELECT id FROM nodes"
    assert out["row_count"] == 1


def test_kg_2_305_forced_dialect_overrides_model():
    eng = _FakeEngine()
    out = nl_planner.nl_query(
        eng,
        "rows",
        dialect="sql",
        planner=_canned('{"dialect": "uql", "query": "SELECT id FROM nodes"}'),
    )
    assert out["dialect"] == "sql"
    assert eng.sql_seen == "SELECT id FROM nodes"


def test_kg_2_305_preview_does_not_execute():
    eng = _FakeEngine()
    out = nl_planner.nl_query(
        eng,
        "everything",
        execute=False,
        planner=_canned('{"dialect": "uql", "query": "MATCH (:Agent) |> LIMIT 3"}'),
    )
    assert "results" not in out
    assert out["generated_query"] == "MATCH (:Agent) |> LIMIT 3"
    assert eng.uql_seen is None  # never executed


def test_kg_2_305_generated_mutation_is_refused():
    eng = _FakeEngine()
    out = nl_planner.nl_query(
        eng,
        "delete it all",
        planner=_canned('{"dialect": "cypher", "query": "MATCH (n) DELETE n"}'),
    )
    assert "mutation" in out["error"]
    assert "results" not in out
    assert eng.cypher_seen is None


def test_kg_2_305_unparseable_model_output_errors():
    eng = _FakeEngine()
    out = nl_planner.nl_query(
        eng, "nonsense", planner=_canned("I cannot help with that")
    )
    assert "planning failed" in out["error"]


def test_kg_2_305_no_llm_configured_is_clean_error(monkeypatch):
    """No injected planner + nothing configured → a clean error, not a crash."""
    monkeypatch.setattr(nl_planner, "is_llm_configured", lambda: False)
    eng = _FakeEngine()
    out = nl_planner.nl_query(eng, "list agents")
    assert "no LLM configured" in out["error"]
    assert "results" not in out
    # the schema/engine were never touched because planning short-circuited
    assert eng.uql_seen is None


def test_kg_2_305_uses_au_fleet_llm_when_configured(monkeypatch):
    """The default planner path builds the AU fleet model via create_model (KG-2.305)."""
    monkeypatch.setattr(nl_planner, "is_llm_configured", lambda: True)

    class _FakeAgent:
        def __init__(self, *a, **k):
            pass

        def run_sync(self, prompt):
            class _R:
                output = '{"dialect": "uql", "query": "MATCH (:Agent) |> LIMIT 2"}'

            return _R()

    created = {"n": 0}

    def _fake_create_model(**kwargs):
        created["n"] += 1
        assert kwargs.get("role") == "planner"
        return object()

    monkeypatch.setattr("pydantic_ai.Agent", _FakeAgent)
    monkeypatch.setattr(
        "agent_utilities.core.model_factory.create_model", _fake_create_model
    )
    eng = _FakeEngine()
    out = nl_planner.nl_query(eng, "top agents")
    assert created["n"] == 1  # the fleet model WAS built (role=planner)
    assert out["dialect"] == "uql"
    assert eng.uql_seen == "MATCH (:Agent) |> LIMIT 2"


def test_kg_2_305_is_llm_configured_reads_config(monkeypatch):
    import agent_utilities.core.config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "openai_base_url", "http://vllm.arpa/v1")
    assert nl_planner.is_llm_configured() is True


def test_kg_2_305_render_schema_grounds_label_column_and_real_edge_columns():
    """CONCEPT:AU-KG.query.ask-gateway-rest-twin grounding fix — the schema hint the AU
    planner sends the fleet LLM must state the real label/kind SQL column (`type`, not
    a same-named-but-unrelated `label` property) and the real fixed edges columns
    (`src`/`dst`/`rel`), not an invented column like `source_node_id`.
    """
    eng = _FakeEngine()
    schema = nl_planner.build_schema_context(eng)
    rendered = nl_planner._render_schema(schema)

    assert "Agent" in schema["node_labels"]
    assert "`type`" in rendered
    assert "`src`" in rendered
    assert "`dst`" in rendered
    assert "`rel`" in rendered
    assert "source_node_id" in rendered  # named as the wrong column to avoid


def test_kg_2_305_nl_query_mcp_tool_registered():
    from agent_utilities.mcp import kg_server
    from agent_utilities.mcp.tools.query_tools import register_query_tools

    class _FakeMCP:
        def tool(self, *a, **k):
            return lambda fn: fn

    register_query_tools(_FakeMCP())
    assert "nl_query" in kg_server.REGISTERED_TOOLS

    # Tool routes to the KG-2.305 provider and returns the generated query as JSON.
    monkeypatch_engine = _FakeEngine()
    orig_get = kg_server._get_engine
    kg_server._get_engine = lambda *a, **k: monkeypatch_engine
    try:
        import agent_utilities.knowledge_graph.core.nl_planner as np

        orig_conf = np.is_llm_configured
        np.is_llm_configured = lambda: True
        # Inject the canned planner by monkeypatching AuNlPlanner default construction.
        # Capture the real class FIRST, then have the default ctor build a canned one
        # (routing through the saved class, not the override — no self-recursion).
        orig_planner = np.AuNlPlanner
        np.AuNlPlanner = lambda **k: orig_planner(
            run=lambda prompt, system: (
                '{"dialect": "uql", "query": "MATCH (:Agent) |> LIMIT 1"}'
            )
        )
        try:
            res = json.loads(
                kg_server.REGISTERED_TOOLS["nl_query"](
                    text="top agents",
                    dialect="auto",
                    schema_hint="",
                    execute=True,
                    limit=50,
                )
            )
            assert res["generated_query"] == "MATCH (:Agent) |> LIMIT 1"
            assert res["planner"] == "agent-utilities-fleet-llm"
        finally:
            np.is_llm_configured = orig_conf
            np.AuNlPlanner = orig_planner
    finally:
        kg_server._get_engine = orig_get
