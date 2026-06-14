"""CONCEPT:KG-2.65 — Code-intelligence tools resolve symbols via graph queries.

Exercises the real :class:`CodeIntelligence` core against a backend that records the cypher it
receives and returns canned rows — asserting both that results parse and that the queries carry
the right ontology patterns (CALLS / COVERS), which is what makes the grounding work on a live,
ingested repo.
"""

from __future__ import annotations

from agent_utilities.tools.code_intelligence_tools import CodeIntelligence


class _Backend:
    def __init__(self, rows):
        self.rows = rows
        self.seen: list[str] = []

    def execute(self, cypher, params=None):
        self.seen.append(cypher)
        return self.rows


class _Engine:
    def __init__(self, rows):
        self.backend = _Backend(rows)


def test_who_calls_uses_calls_edge_and_parses_rows():
    eng = _Engine([{"id": "Code:a.py::caller", "name": "caller"}])
    ci = CodeIntelligence(eng)
    rows = ci.who_calls("target")
    assert rows == [{"id": "Code:a.py::caller", "name": "caller"}]
    assert "CALLS" in eng.backend.seen[-1] and "calls" in eng.backend.seen[-1]


def test_impacted_tests_uses_covers_edge():
    eng = _Engine([{"id": "Test:t.py::test_x", "name": "test_x"}])
    ci = CodeIntelligence(eng)
    rows = ci.impacted_tests("target")
    assert rows[0]["name"] == "test_x"
    assert "COVERS" in eng.backend.seen[-1]
    assert "(t:Test)" in eng.backend.seen[-1]


def test_call_graph_depth_is_bounded():
    eng = _Engine([])
    ci = CodeIntelligence(eng)
    ci.call_graph("f", depth=99)
    # depth is clamped to <=5 to bound the traversal
    assert "*1..5" in eng.backend.seen[-1]


def test_find_definition_matches_name_or_id_suffix():
    eng = _Engine([{"id": "pkg/m.py::foo", "name": "foo", "file": "pkg/m.py"}])
    ci = CodeIntelligence(eng)
    rows = ci.find_definition("foo")
    assert rows[0]["file"] == "pkg/m.py"
    q = eng.backend.seen[-1]
    assert "ENDS WITH $suffix" in q and "c.name = $s" in q


def test_no_backend_degrades_to_empty():
    class _NoBackend:
        backend = None

    ci = CodeIntelligence(_NoBackend())
    assert ci.who_calls("x") == []
    assert ci.find_definition("x") == []
