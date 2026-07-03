"""Cross-layer error-diagnosis context provider.

CONCEPT:KG-2.297 (troubleshoot provider over :RunTrace/:ToolCall + the layered
app-trace→container→system→host→cross-cutting tool playbook).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.retrieval import context_plane
from agent_utilities.knowledge_graph.retrieval.troubleshoot_context import (
    diagnose_symptom,
)


class FakeTraceEngine:
    """Returns a failed run with a failing :ToolCall in its chain."""

    def query_cypher(self, cypher, params):
        if "MATCH (t:RunTrace {id: $tid}) RETURN" in cypher:
            return [
                {
                    "id": "trace:run-42",
                    "agent_name": "knowledge-graph",
                    "status": "failed",
                    "error": "engine breaker open",
                    "duration_ms": 1200.0,
                }
            ]
        if "[:MADE_TOOL_CALL]->(tc:ToolCall)" in cypher:
            return [
                {
                    "tool_name": "graph_search",
                    "status": "ok",
                    "error": "",
                    "sequence": 0,
                },
                {
                    "tool_name": "graph_query",
                    "status": "error",
                    "error": "connection refused",
                    "sequence": 1,
                },
            ]
        return []


class FakeFailedRunsEngine:
    def query_cypher(self, cypher, params):
        if "WHERE t.status IN ['failed','error']" in cypher:
            return [
                {"id": "trace:run-9", "agent_name": "arr-specialist", "error": "boom"}
            ]
        return []


@pytest.mark.concept("KG-2.297")
def test_diagnose_symptom_pulls_run_trace_and_failing_tool_call():
    res = diagnose_symptom(
        FakeTraceEngine(), query="why did my agent run fail", node_id="run-42"
    )
    assert res["status"] == "ok" and res["domain"] == "troubleshoot"
    # The run + its failing tool call surface in the synthesized answer.
    assert "trace:run-42" in res["answer"]
    assert "graph_query" in res["answer"] and "connection refused" in res["answer"]
    # The full layered ladder is always present in the playbook.
    layers = [p["layer"] for p in res["sections"]["playbook"]]
    assert set(layers) == {
        "app_trace",
        "container",
        "system",
        "host",
        "cross_cutting",
    }
    # Citations carry the run + tool-call provenance rows.
    types = {c["type"] for c in res["citations"]}
    assert "run_trace" in types and "tool_call" in types
    assert res["capability_id"].startswith("troubleshoot:")


@pytest.mark.concept("KG-2.297")
def test_diagnose_symptom_service_intent_leads_with_host_split():
    res = diagnose_symptom(
        FakeFailedRunsEngine(),
        query="the .arpa endpoint is unreachable, 502",
        intent="service",
    )
    assert res["intent"] == "service"
    # Reachability symptom → the host-vs-service decision leads.
    assert "host-vs-service" in res["answer"].lower()
    assert res["sections"]["playbook"][0]["layer"] == "host"


@pytest.mark.concept("KG-2.297")
def test_diagnose_symptom_no_trace_lists_recent_errored_runs():
    res = diagnose_symptom(FakeFailedRunsEngine(), query="something errored")
    assert "trace:run-9" in res["answer"]
    assert any(c["type"] == "failed_run" for c in res["citations"])


@pytest.mark.concept("KG-2.297")
def test_diagnose_symptom_degrades_on_empty_engine():
    class Dead:
        def query_cypher(self, cypher, params):
            raise RuntimeError("backend down")

    res = diagnose_symptom(Dead(), query="diagnose this")
    # Best-effort reads never raise; the playbook still comes back.
    assert res["status"] == "ok"
    assert len(res["sections"]["playbook"]) == 5


@pytest.mark.concept("KG-2.297")
def test_troubleshoot_registered_in_context_plane():
    # Reachable via graph_analyze action=explain target="troubleshoot:..."
    assert "troubleshoot" in context_plane._BUILTIN_PROVIDERS
    res = context_plane.synthesize_context(
        FakeTraceEngine(),
        domain="troubleshoot",
        query="x",
        intent="run",
        node_id="run-42",
    )
    assert res["domain"] == "troubleshoot" and res["status"] == "ok"
