"""Tests for the holistic `agent-utilities doctor` health sweep.

The doctor is an aggregator of existing checks; tests assert the contract (report
shape, status precedence, fix routing, defensiveness) with the underlying checks
monkeypatched, plus a live-path through the graph_configure MCP action.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.deployment import doctor as D


def _ok(name):
    return lambda **kw: D._result(name, "ok", "fine")


def test_run_doctor_all_ok(monkeypatch):
    monkeypatch.setattr(D, "CHECKS", {n: _ok(n) for n in ("a", "b", "c")})
    rep = D.run_doctor()
    assert rep["status"] == "healthy"
    assert len(rep["checks"]) == 3
    assert rep["summary"] == "All checks passed."


def test_run_doctor_worst_status_wins(monkeypatch):
    monkeypatch.setattr(
        D,
        "CHECKS",
        {
            "a": _ok("a"),
            "b": lambda **kw: D._result("b", "warn", "meh"),
            "c": lambda **kw: D._result(
                "c", "fail", "broken", remediation="do x", skill="s"
            ),
        },
    )
    rep = D.run_doctor()
    assert rep["status"] == "unhealthy"  # a fail dominates
    assert rep["counts"]["fail"] == 1 and rep["counts"]["warn"] == 1


def test_run_doctor_skip_is_not_unhealthy(monkeypatch):
    monkeypatch.setattr(
        D, "CHECKS", {"a": _ok("a"), "b": lambda **kw: D._result("b", "skip", "n/a")}
    )
    assert D.run_doctor()["status"] == "healthy"


def test_run_doctor_check_exception_is_contained(monkeypatch):
    def _boom(**kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(D, "CHECKS", {"a": _ok("a"), "b": _boom})
    rep = D.run_doctor()
    # The crashing check becomes an 'error' result, doctor still completes.
    b = next(c for c in rep["checks"] if c["name"] == "b")
    assert b["status"] == "error" and "kaboom" in b["detail"]
    assert rep["status"] == "unhealthy"


def test_run_doctor_only_filter(monkeypatch):
    monkeypatch.setattr(D, "CHECKS", {n: _ok(n) for n in ("a", "b", "c")})
    rep = D.run_doctor(["b"])
    assert [c["name"] for c in rep["checks"]] == ["b"]


def test_run_doctor_fix_runs_autofix_then_reruns(monkeypatch):
    state = {"fixed": False}

    def flaky(**kw):
        return D._result(
            "hooks",
            "ok" if state["fixed"] else "warn",
            "hooks",
            auto_fixable=True,
        )

    monkeypatch.setattr(D, "CHECKS", {"hooks": flaky})

    def fake_fix(name):
        state["fixed"] = True
        return {"fixed": name, "result": "ok"}

    monkeypatch.setattr(D, "_auto_fix", fake_fix)
    rep = D.run_doctor(fix=True)
    assert rep["fixes"] and rep["fixes"][0]["fixed"] == "hooks"
    # After the fix the re-run flipped the check to ok → overall healthy.
    assert rep["checks"][0]["status"] == "ok"
    assert rep["status"] == "healthy"


def test_individual_checks_never_raise():
    # Every real check must return a dict with the contract keys, never raise,
    # even with nothing deployed.
    for name, fn in D.CHECKS.items():
        res = fn(live=False) if name == "mcp_fleet" else fn()
        assert set(res) >= {"name", "status", "detail"}
        assert res["status"] in ("ok", "warn", "fail", "skip", "error")


# ── live path: graph_configure system_doctor MCP action ────────────────────
class _MockMCP:
    def __init__(self):
        self.funcs = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.funcs[fn.__name__] = fn
            return fn

        return deco

    def custom_route(self, *a, **k):
        def deco(fn):
            self.funcs[fn.__name__] = fn
            return fn

        return deco


@pytest.fixture
def registered_tools():
    mock_mcp = _MockMCP()
    engine = MagicMock()
    engine.backend = MagicMock()
    engine.backend.read_only = False
    with patch(
        "agent_utilities.mcp.server_factory.create_mcp_server",
        return_value=(None, mock_mcp, []),
    ):
        with patch("agent_utilities.mcp.kg_server._get_engine", return_value=engine):
            from agent_utilities.mcp.kg_server import _build_server

            _build_server()
    return mock_mcp.funcs


@pytest.mark.asyncio
async def test_graph_configure_system_doctor_live_path(registered_tools, monkeypatch):
    from agent_utilities.mcp import kg_server

    monkeypatch.setattr(D, "CHECKS", {"a": _ok("a")})
    raw = await kg_server._execute_tool(
        "graph_configure",
        action="system_doctor",
        config_value=json.dumps({"only": ["a"]}),
    )
    rep = json.loads(raw)
    assert rep["status"] == "healthy"
    assert rep["checks"][0]["name"] == "a"
