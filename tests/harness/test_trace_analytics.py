"""Moat trace analytics (CONCEPT:KG-2.257) — queries Opik's opaque store can't do."""

from __future__ import annotations

from agent_utilities.harness import trace_analytics as ta


class _FakeBackend:
    """Returns canned rows keyed by the node type the cypher filters on."""

    def __init__(self, by_type):
        self._by_type = by_type

    def execute(self, cypher):
        for t, rows in self._by_type.items():
            if f"'{t}'" in cypher:
                # crude WHERE status='failed' / score<0.5 filters for the tests
                if t == "assertion_result" and "failed" in cypher:
                    return [r for r in rows if r.get("status") == "failed"] or [
                        {k: v for k, v in r.items() if k != "status"} for r in rows
                        if r.get("status") == "failed"
                    ]
                if t == "online_score" and "score < 0.5" in cypher:
                    return [r for r in rows if r.get("score", 1) < 0.5]
                return rows
        return []


def test_trace_rootcause_groups_failures_by_agent():
    be = _FakeBackend({
        "trace": [
            {"id": "t1", "agent": "planner", "status": "error", "name": "run"},
            {"id": "t2", "agent": "planner", "status": "ok", "name": "run"},
        ],
        "assertion_result": [
            {"trace_id": "t1", "assertion": "answer is 4", "reasoning": "said 5", "status": "failed"},
            {"trace_id": "t2", "assertion": "answer is 4", "reasoning": "said 5", "status": "failed"},
        ],
        "online_score": [],
    })
    out = ta.trace_rootcause(be)
    assert out["total"] == 2
    assert out["by_agent"]["planner"] == 2


def test_prompt_regression_ranks_versions_by_mean_score():
    be = _FakeBackend({
        "generation": [
            {"pv": "vA", "trace_id": "t1"},
            {"pv": "vB", "trace_id": "t2"},
        ],
        "online_score": [
            {"trace_id": "t1", "score": 0.9},
            {"trace_id": "t2", "score": 0.2},
        ],
    })
    out = ta.prompt_regression(be)
    assert out["worst"] == "vB"
    assert out["by_version"]["vB"]["mean_score"] == 0.2


def test_failure_cluster_groups_by_assertion():
    be = _FakeBackend({
        "trace": [
            {"id": "t1", "agent": "a1", "name": "run"},
            {"id": "t2", "agent": "a2", "name": "run"},
        ],
        "assertion_result": [
            {"trace_id": "t1", "assertion": "must cite source", "status": "failed"},
            {"trace_id": "t2", "assertion": "must cite source", "status": "failed"},
        ],
    })
    out = ta.failure_cluster(be)
    c = out["clusters"][0]
    assert c["assertion"] == "must cite source" and c["failures"] == 2
    assert set(c["agents"]) == {"a1", "a2"}
