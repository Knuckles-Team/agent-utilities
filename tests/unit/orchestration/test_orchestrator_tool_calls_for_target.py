"""Orchestrator.get_tool_calls_for_target (G23, audit-trail closure).

CONCEPT:AU-KG.audit.tool-call-acted-on-reverse-index

The entity-anchored reverse index: given a target id, reconstruct every
``:ToolCall`` that acted on it (in call order) plus a best-effort
``audit_verify()`` snapshot. Exercised against a minimal fake engine/backend —
no real KG required.
"""

from __future__ import annotations

from agent_utilities.orchestration.manager import Orchestrator


class _FakeGraphClient:
    def __init__(self, report=None, error=None):
        self._report = report
        self._error = error

    def audit_verify(self):
        if self._error is not None:
            raise self._error
        return dict(self._report)


class _FakeBackend:
    def __init__(self, rows=None, error=None):
        self._rows = rows or []
        self._error = error

    def execute(self, query, params=None):
        if self._error is not None:
            raise self._error
        return list(self._rows)


class _FakeEngine:
    def __init__(self, graph=None, backend=None):
        self.graph = graph
        self.backend = backend


_ROWS = [
    {
        "id": "toolcall:1:0",
        "run_id": "run:1",
        "agent_name": "agent-x",
        "server": "server-y",
        "tool_name": "engine_nodes",
        "args": '{"ticket_id": "ticket:T1"}',
        "result_preview": "read",
        "error": "",
        "status": "ok",
        "sequence": 0,
        "timestamp": "2026-01-01T00:00:00Z",
    },
    {
        "id": "toolcall:1:1",
        "run_id": "run:1",
        "agent_name": "agent-x",
        "server": "server-y",
        "tool_name": "spec_ticket",
        "args": '{"ticket_id": "ticket:T1"}',
        "result_preview": "created",
        "error": "",
        "status": "ok",
        "sequence": 1,
        "timestamp": "2026-01-01T00:00:01Z",
    },
]

_REPORT = {"graph": "g", "ok": True, "entries": 2, "first_broken_seq": None, "detail": "ok"}


def test_no_backend_active_reports_error():
    engine = _FakeEngine(graph=_FakeGraphClient(report=_REPORT), backend=None)
    out = Orchestrator(engine).get_tool_calls_for_target("ticket:T1")
    assert out["tool_calls"] == []
    assert "error" in out


def test_reconstructs_history_in_order_with_audit_snapshot():
    engine = _FakeEngine(
        graph=_FakeGraphClient(report=_REPORT), backend=_FakeBackend(rows=_ROWS)
    )
    out = Orchestrator(engine).get_tool_calls_for_target("ticket:T1")
    assert out["target_id"] == "ticket:T1"
    assert out["tool_call_count"] == 2
    assert [tc["id"] for tc in out["tool_calls"]] == [r["id"] for r in _ROWS]
    assert out["audit"] == _REPORT


def test_query_failure_degrades_cleanly():
    engine = _FakeEngine(
        graph=_FakeGraphClient(report=_REPORT),
        backend=_FakeBackend(error=RuntimeError("backend down")),
    )
    out = Orchestrator(engine).get_tool_calls_for_target("ticket:T1")
    assert out["tool_calls"] == []
    assert "error" in out


def test_no_results_still_returns_audit_snapshot():
    engine = _FakeEngine(
        graph=_FakeGraphClient(report=_REPORT), backend=_FakeBackend(rows=[])
    )
    out = Orchestrator(engine).get_tool_calls_for_target("ticket:NONE")
    assert out["tool_call_count"] == 0
    assert out["audit"] == _REPORT


def test_audit_verify_failure_is_best_effort_none():
    """When the engine build doesn't support AuditVerify, the reverse-index
    result still comes back — audit is None, not an exception."""
    engine = _FakeEngine(
        graph=_FakeGraphClient(error=RuntimeError("not supported")),
        backend=_FakeBackend(rows=_ROWS),
    )
    out = Orchestrator(engine).get_tool_calls_for_target("ticket:T1")
    assert out["tool_call_count"] == 2
    assert out["audit"] is None
