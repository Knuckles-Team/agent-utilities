"""Tests for the synthesized, cited code_context Q&A (CONCEPT:KG-2.134 / KG-2.135).

Exercises the composition + synthesis offline via a fake engine that routes
Cypher reads to canned rows, plus path normalization and cross-repo grouping.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.retrieval.code_context import (
    build_code_context,
    cross_repo_usages,
    normalize_path,
    resolve_anchors,
)

_CANON = "/home/apps/workspace/agent-packages/agent-utilities/agent_utilities/orchestration/engine.py"
_AU = "/au/agent_utilities/orchestration/engine.py"


class FakeEngine:
    """Routes read-only Cypher to canned rows by inspecting the query text."""

    def __init__(self, *, with_similar=False, with_routes=False):
        self.with_similar = with_similar
        self.with_routes = with_routes
        self.calls: list[str] = []

    def query_cypher(self, cypher, params):  # noqa: D401
        self.calls.append(cypher)
        c = cypher
        if "c.id = $id" in c:
            return [self._def(params.get("id"))]
        if "c.name = $tok" in c or "CONTAINS $tok" in c:
            return [self._def("code:" + _CANON + "::run_agent", name="run_agent")]
        if "(caller:Code)-[:calls]->(def:Code)" in c:  # find_references (callers)
            return [
                self._row("execute_agent", _CANON, 71),
                # same file under the /au mount alias -> must dedupe to one cite
                self._row("execute_agent", _AU, 71),
                self._row(
                    "_run_stream_events",
                    "/home/apps/workspace/open-source-libraries/agent-frameworks/pydantic-ai/x.py",
                    1252,
                ),
            ]
        if "[:calls*" in c and "->(callee:Code)" in c:  # callees (how)
            return [self._row("create_model", _CANON, 200)]
        if "[:calls*" in c and "->(t:Code)" in c:  # impact (callers)
            return [self._row("loop_cycle", _CANON, 999)]
        if "similar_to" in c:
            return (
                [
                    self._row(
                        "run_agent_v2", _CANON, 1500, score=0.91, extra={"score": 0.91}
                    )
                ]
                if self.with_similar
                else []
            )
        if ":Route" in c:
            return (
                [
                    {
                        "method": "POST",
                        "path": "/agent/run",
                        "handler": "run_agent",
                        "service": "svc:graph-os",
                    }
                ]
                if self.with_routes
                else []
            )
        return []  # concepts / coupling / docs not enriched

    @staticmethod
    def _def(nid, name="run_agent"):
        return {
            "id": nid,
            "name": name,
            "file_path": _CANON,
            "line": 1378,
            "language": "python",
            "kind": "function",
            "instance": None,
            "source_system": None,
        }

    @staticmethod
    def _row(name, fp, line, score=None, extra=None):
        row = {
            "id": f"code:{fp}::{name}",
            "name": name,
            "file_path": fp,
            "line": line,
            "language": "python",
            "kind": "function",
            "instance": None,
            "source_system": None,
        }
        if extra:
            row.update(extra)
        return row


@pytest.mark.concept("KG-2.135")
def test_normalize_path_folds_au_mount():
    assert normalize_path(_AU) == _CANON
    assert normalize_path(_CANON) == _CANON
    assert normalize_path("") == ""


@pytest.mark.concept("KG-2.134")
def test_resolve_anchors_by_name():
    anchors = resolve_anchors(FakeEngine(), query="how does run_agent work")
    assert anchors and anchors[0]["symbol"] == "run_agent"
    assert anchors[0]["file"] == _CANON


@pytest.mark.concept("KG-2.134")
def test_how_intent_synthesizes_with_calls_and_citation():
    res = build_code_context(
        FakeEngine(), query="how does run_agent work", intent="how"
    )
    assert res["status"] == "ok"
    assert res["intent"] == "how"
    assert "call_graph" in res["used_primitives"]
    assert "`run_agent`" in res["answer"]
    assert "create_model" in res["answer"]  # a callee surfaced in prose
    assert res["capability_id"] == "code_context:how:run_agent"
    # every citation carries a real file path
    assert all(cite["file"] for cite in res["citations"])


@pytest.mark.concept("KG-2.135")
def test_usage_intent_dedupes_au_mount_and_groups_cross_repo():
    res = build_code_context(
        FakeEngine(), query="where is run_agent used", intent="usage"
    )
    files = [(c["file"], c["line"]) for c in res["citations"]]
    # the /au alias collapsed: execute_agent@engine.py:71 appears once, not twice
    assert files.count((_CANON, 71)) == 1
    assert all(not f.startswith("/au/") for f, _ in files)
    # cross_repo section present (usage implies cross-repo) and spans repos
    cr = res["sections"]["cross_repo"][0]
    assert cr["usage_count"] >= 2
    assert any("agent-utilities" in r for r in cr["repos"])
    assert any(r.startswith("oss/") for r in cr["repos"])


@pytest.mark.concept("KG-2.134")
def test_impact_intent_reports_blast_radius():
    res = build_code_context(
        FakeEngine(), query="impact of changing run_agent", intent="impact"
    )
    assert "impact_of_change" in res["used_primitives"]
    assert "loop_cycle" in str(res["sections"]["impacted_callers"])
    assert "impacts" in res["answer"]


@pytest.mark.concept("KG-2.134")
def test_no_anchor_degrades_gracefully():
    class Empty:
        def query_cypher(self, cypher, params):
            return []

    res = build_code_context(Empty(), query="nonexistent_symbol_xyz", intent="how")
    assert res["status"] == "ok"
    assert res["anchors"] == []
    assert "No resolved code symbol" in res["answer"]


@pytest.mark.concept("KG-2.135")
def test_cross_repo_usages_grouped_by_repo():
    info = cross_repo_usages(FakeEngine(), "run_agent")
    assert info["symbol"] == "run_agent"
    assert info["usage_count"] >= 2
    assert "agent-utilities" in info["usages_by_repo"]
    assert info["definitions"]


@pytest.mark.concept("KG-2.134")
def test_similar_and_routes_surface_when_present():
    eng = FakeEngine(with_similar=True, with_routes=True)
    res = build_code_context(eng, query="where is run_agent used", intent="usage")
    assert "similar_code" in res["used_primitives"]
    assert "routes" in res["used_primitives"]
    assert json.dumps(res)  # fully JSON-serializable
