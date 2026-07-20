"""DB-GPT-style text2sql data-analysis agent loop (CONCEPT:AU-KG.query.data-gateway-rest-twin).

Exercises the full bounded ReAct loop with a mock LLM (an injected KG-2.305 planner +
injected/absent synthesis) over a mock engine: happy-path schema-link → plan → execute →
synthesize; bounded self-correction (a failing query, then a repaired one that succeeds);
the additive clean-fallback when no LLM is configured; deterministic answer synthesis with
no synthesis LLM; schema-linking picking the relevant tables; mutation refusal; and the
``ask_data`` MCP tool registration.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.core import nl_planner
from agent_utilities.knowledge_graph.orchestration import data_analyst

pytestmark = pytest.mark.concept("AU-KG.query.data-gateway-rest-twin")


class _FakeEngine:
    """Mock engine exposing the AU→engine query surfaces the loop executes over.

    ``bad_markers`` are substrings that make the SQL/uql surface raise (to drive the
    self-correction loop); anything else returns canned rows.
    """

    def __init__(self, *, bad_markers: tuple[str, ...] = ()):
        self.bad_markers = bad_markers
        self.sql_calls: list[str] = []
        self.cypher_calls: list[str] = []

    def query_cypher(self, query, *a, **k):
        if query.startswith("MATCH (n) RETURN DISTINCT labels"):
            return [{"labels": ["Order"]}, {"labels": ["Customer"]}]
        self.cypher_calls.append(query)
        return [{"id": "c1", "name": "acme"}]

    def sql(self, query):
        self.sql_calls.append(query)
        for marker in self.bad_markers:
            if marker in query:
                raise RuntimeError(f"no such column referenced: {marker}")
        return [{"id": "acme", "orders": 12}, {"id": "globex", "orders": 5}]

    def uql(self, query):  # pragma: no cover - not used by these tests
        return [{"id": "doc1", "score": 0.9}]

    def sparql(self, query):  # pragma: no cover - not used by these tests
        return [{"s": "iri1"}]


def _seq_planner(*payloads: str) -> nl_planner.AuNlPlanner:
    """KG-2.305 planner whose fleet-LLM call returns each canned payload in turn."""
    box = {"i": 0}

    def _run(prompt: str, system: str) -> str:
        i = min(box["i"], len(payloads) - 1)
        box["i"] += 1
        return payloads[i]

    return nl_planner.AuNlPlanner(run=_run)


def _canned(payload: str) -> nl_planner.AuNlPlanner:
    return nl_planner.AuNlPlanner(run=lambda prompt, system: payload)


def test_kg_2_308_happy_path_plan_execute_synthesize(monkeypatch):
    """schema-link → plan (KG-2.305) → execute → synthesize a NL answer, one shot."""
    monkeypatch.setattr(nl_planner, "is_llm_configured", lambda: True)
    eng = _FakeEngine()
    agent = data_analyst.DataAnalystAgent(
        eng,
        planner=_canned(
            '{"dialect": "sql", "query": "SELECT customer, count(*) AS orders '
            'FROM nodes GROUP BY customer"}'
        ),
        synthesize=lambda q, rows: f"There are {len(rows)} customers.",
    )
    out = agent.analyze("how many orders per customer")
    assert "error" not in out
    assert out["dialect"] == "sql"
    assert out["query"].startswith("SELECT customer")
    assert out["row_count"] == 2
    assert out["answer"] == "There are 2 customers."
    assert out["citations"] == ["acme", "globex"]  # id-keyed provenance from the rows
    # exactly one attempt, and it succeeded
    assert len(out["attempts"]) == 1
    assert eng.sql_calls == [out["query"]]


def test_kg_2_308_self_correction_on_query_error_then_success(monkeypatch):
    """A failing query is fed back with its error; the repaired query then succeeds."""
    monkeypatch.setattr(nl_planner, "is_llm_configured", lambda: True)
    eng = _FakeEngine(bad_markers=("WRONGCOL",))
    agent = data_analyst.DataAnalystAgent(
        eng,
        planner=_seq_planner(
            '{"dialect": "sql", "query": "SELECT WRONGCOL FROM nodes"}',
            '{"dialect": "sql", "query": "SELECT customer, orders FROM nodes"}',
        ),
        synthesize=lambda q, rows: "ok",
        max_corrections=2,
    )
    out = agent.analyze("orders by customer")
    assert "error" not in out
    assert out["query"] == "SELECT customer, orders FROM nodes"
    assert out["row_count"] == 2
    # two attempts recorded: the first failed, the second succeeded
    assert len(out["attempts"]) == 2
    assert "execution failed" in out["attempts"][0]["error"]
    assert "error" not in out["attempts"][1]
    # both queries were actually issued to the engine
    assert eng.sql_calls == [
        "SELECT WRONGCOL FROM nodes",
        "SELECT customer, orders FROM nodes",
    ]


def test_kg_2_308_correction_exhausted_returns_clean_error(monkeypatch):
    """When every attempt (incl. corrections) fails, a clean error + full trace is returned."""
    monkeypatch.setattr(nl_planner, "is_llm_configured", lambda: True)
    eng = _FakeEngine(bad_markers=("BADCOL",))
    agent = data_analyst.DataAnalystAgent(
        eng,
        planner=_canned('{"dialect": "sql", "query": "SELECT BADCOL FROM nodes"}'),
        max_corrections=1,
    )
    out = agent.analyze("something impossible")
    assert "error" in out
    assert "failed after 2 attempt" in out["error"]
    assert len(out["attempts"]) == 2  # 1 initial + 1 correction, both failed


def test_kg_2_308_no_llm_configured_is_clean_fallback(monkeypatch):
    """No injected planner + nothing configured → a clean error, never a crash."""
    monkeypatch.setattr(nl_planner, "is_llm_configured", lambda: False)
    eng = _FakeEngine()
    out = data_analyst.ask_data(eng, "how many orders")
    assert "no LLM configured" in out["error"]
    assert "answer" not in out
    assert eng.sql_calls == []  # planning short-circuited; the engine was never queried


def test_kg_2_308_deterministic_answer_without_synthesis_llm(monkeypatch):
    """With rows but no synthesis LLM, the answer degrades to a deterministic summary."""
    monkeypatch.setattr(nl_planner, "is_llm_configured", lambda: False)
    eng = _FakeEngine()
    # planner injected → query generation works; synthesize NOT injected → fallback answer
    out = data_analyst.DataAnalystAgent(
        eng,
        planner=_canned('{"dialect": "sql", "query": "SELECT customer FROM nodes"}'),
    ).analyze("list customers")
    assert "error" not in out
    assert out["answer"].startswith("Found 2 result rows for: list customers")


def test_kg_2_308_schema_linking_picks_relevant_tables():
    """schema_link scores schema names by question-token overlap (needs no LLM)."""
    schema = {
        "tables": ["orders", "customers", "products", "inventory_snapshots"],
        "node_labels": ["Order", "Customer", "Warehouse"],
    }
    linked = data_analyst.schema_link("how many orders did each customer place", schema)
    assert "orders" in linked["tables"]
    assert "customers" in linked["tables"]
    assert "products" not in linked["tables"]
    assert "Order" in linked["node_labels"]
    assert "Customer" in linked["node_labels"]
    assert "Warehouse" not in linked["node_labels"]


def test_kg_2_308_schema_linking_falls_back_to_all_when_no_overlap():
    """A question sharing no vocabulary with the schema keeps all names (never starved)."""
    schema = {"tables": ["alpha", "beta"], "node_labels": []}
    linked = data_analyst.schema_link("zzz qqq", schema)
    assert linked["tables"] == ["alpha", "beta"]


def test_kg_2_308_generated_mutation_is_refused(monkeypatch):
    """A generated mutation is a hard refusal (read-only surface), not a correction."""
    monkeypatch.setattr(nl_planner, "is_llm_configured", lambda: True)
    eng = _FakeEngine()
    out = data_analyst.DataAnalystAgent(
        eng,
        planner=_canned('{"dialect": "cypher", "query": "MATCH (n) DELETE n"}'),
    ).analyze("delete everything")
    assert "mutation" in out["error"]
    assert "answer" not in out
    assert eng.cypher_calls == []  # never executed


def test_kg_2_308_uses_au_fleet_llm_for_synthesis_when_configured(monkeypatch):
    """The default synthesis path builds the AU fleet model via create_model (generator)."""
    monkeypatch.setattr(nl_planner, "is_llm_configured", lambda: True)

    class _FakeAgent:
        def __init__(self, *a, **k):
            pass

        def run_sync(self, prompt):
            class _R:
                output = "Synthesized: 2 customers."

            return _R()

    created = {"n": 0}

    def _fake_create_model(**kwargs):
        created["n"] += 1
        assert kwargs.get("role") == "generator"
        return object()

    monkeypatch.setattr("pydantic_ai.Agent", _FakeAgent)
    monkeypatch.setattr(
        "agent_utilities.core.model_factory.create_model", _fake_create_model
    )
    eng = _FakeEngine()
    out = data_analyst.DataAnalystAgent(
        eng,
        planner=_canned('{"dialect": "sql", "query": "SELECT customer FROM nodes"}'),
    ).analyze("summarize customers")
    assert created["n"] == 1
    assert out["answer"] == "Synthesized: 2 customers."


def test_kg_2_308_synthesis_survives_call_from_inside_a_running_event_loop(
    monkeypatch,
):
    """BUG-2 (kg-exhaustive-smoke.md): ``ask_data`` -> :meth:`analyze` ->
    :meth:`_default_synthesize`, invoked from inside the gateway's
    already-running event loop (every real MCP/REST tool dispatch runs inside
    FastMCP/Starlette's live loop). Before the fix, the synthesis step's
    ``Agent.run_sync`` -- which itself spins its own ``asyncio.run()`` --
    collided with that loop and raised "This event loop is already running",
    silently swallowed by ``_default_synthesize``'s own ``except Exception``
    into the deterministic fallback answer instead of the real synthesized
    one. The fake Agent's ``run_sync`` genuinely calls ``asyncio.run()``
    internally (mirroring the real pydantic-ai facade) so a regression
    reproduces the exact failure mode.
    """
    import asyncio

    monkeypatch.setattr(nl_planner, "is_llm_configured", lambda: True)

    class _FakeAgent:
        def __init__(self, *a, **k):
            pass

        def run_sync(self, prompt):
            async def _inner():
                class _R:
                    output = "Synthesized: 2 customers."

                return _R()

            return asyncio.run(_inner())

    monkeypatch.setattr("pydantic_ai.Agent", _FakeAgent)
    monkeypatch.setattr(
        "agent_utilities.core.model_factory.create_model", lambda **k: object()
    )
    eng = _FakeEngine()
    agent = data_analyst.DataAnalystAgent(
        eng,
        planner=_canned('{"dialect": "sql", "query": "SELECT customer FROM nodes"}'),
    )

    async def _call_from_running_loop():
        # analyze() is a plain sync method called directly (not awaited) from
        # inside a running loop -- exactly how the async ask_data MCP tool
        # handler reaches it in production.
        return agent.analyze("summarize customers")

    out = asyncio.run(_call_from_running_loop())

    # A real synthesized answer, NOT the deterministic fallback the
    # event-loop RuntimeError used to silently degrade to.
    assert out["answer"] == "Synthesized: 2 customers."


def test_kg_2_308_ask_data_mcp_tool_registered():
    from agent_utilities.mcp import kg_server
    from agent_utilities.mcp.tools.query_tools import register_query_tools

    class _FakeMCP:
        def tool(self, *a, **k):
            return lambda fn: fn

    register_query_tools(_FakeMCP())
    assert "ask_data" in kg_server.REGISTERED_TOOLS

    eng = _FakeEngine()
    orig_get = kg_server._get_engine
    kg_server._get_engine = lambda *a, **k: eng
    try:
        orig_conf = nl_planner.is_llm_configured
        nl_planner.is_llm_configured = lambda: True
        orig_planner = nl_planner.AuNlPlanner
        nl_planner.AuNlPlanner = lambda **k: orig_planner(
            run=lambda prompt, system: (
                '{"dialect": "sql", "query": "SELECT customer FROM nodes"}'
            )
        )
        try:
            res = json.loads(
                kg_server.REGISTERED_TOOLS["ask_data"](
                    question="list customers",
                    dialect="auto",
                    max_corrections=2,
                    limit=50,
                )
            )
            assert res["query"] == "SELECT customer FROM nodes"
            assert res["row_count"] == 2
            assert "answer" in res
        finally:
            nl_planner.is_llm_configured = orig_conf
            nl_planner.AuNlPlanner = orig_planner
    finally:
        kg_server._get_engine = orig_get
