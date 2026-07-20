"""Tests for the host dependency preflight.

Asserts the contract (report shape, profile/component scoping, wheel-first engine
messaging, defensiveness) and a live path through the graph_configure MCP action.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.deployment import preflight as P


def test_report_shape_and_python_check():
    rep = P.run_preflight("tiny")
    assert set(rep) >= {
        "status",
        "profile",
        "components",
        "counts",
        "checks",
        "summary",
    }
    assert rep["profile"] == "tiny"
    names = [c["name"] for c in rep["checks"]]
    # tiny always checks python/installer/engine; docker is a skip (not required).
    assert {"python", "installer", "engine_binary", "docker"} <= set(names)
    assert rep["status"] in ("ready", "warnings", "blocked")


def test_docker_required_above_tiny(monkeypatch):
    monkeypatch.setattr(P.shutil, "which", lambda name: None)  # nothing on PATH
    tiny = {c["name"]: c for c in P.run_preflight("tiny")["checks"]}
    ent = {c["name"]: c for c in P.run_preflight("enterprise")["checks"]}
    assert tiny["docker"]["status"] == "skip"
    assert ent["docker"]["status"] == "fail"


def test_engine_is_wheel_first_rust_only_fallback(monkeypatch):
    # Engine binary absent → warn, and the remediation must point at the wheel, not Rust.
    monkeypatch.setattr(P, "_engine_binary_path", lambda: None)
    monkeypatch.setattr(P.shutil, "which", lambda name: None)
    res = P._check_engine()
    assert res["status"] == "warn"
    assert "pip install agent-utilities" in res["remediation"]
    assert "only needed if no prebuilt wheel" in res["remediation"].lower()


def test_engine_present_means_no_rust(monkeypatch):
    monkeypatch.setattr(
        P, "_engine_binary_path", lambda: "/venv/bin/epistemic-graph-server"
    )
    res = P._check_engine()
    assert res["status"] == "ok" and "no Rust needed" in res["detail"]


def test_geniusbot_blocks_headless(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    rep = P.run_preflight("tiny", ["geniusbot"])
    gb = next(c for c in rep["checks"] if c["name"] == "geniusbot")
    assert gb["status"] == "fail"
    assert rep["status"] == "blocked"


def test_unknown_component_is_skip_not_crash():
    rep = P.run_preflight("tiny", ["nope"])
    c = next(c for c in rep["checks"] if c["name"] == "nope")
    assert c["status"] == "skip"


def test_component_checks_never_raise():
    for name, fn in P._COMPONENT_CHECKS.items():
        res = fn()
        assert set(res) >= {"name", "status", "detail"}
        assert res["status"] in ("ok", "warn", "fail", "skip", "error")


# ── live path: graph_configure preflight MCP action ────────────────────────
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
async def test_graph_configure_preflight_live_path(registered_tools):
    from agent_utilities.mcp import kg_server

    raw = await kg_server._execute_tool(
        "graph_configure",
        action="preflight",
        config_key="tiny",
        config_value=json.dumps({"components": ["agent-terminal-ui"]}),
    )
    rep = json.loads(raw)
    assert rep["profile"] == "tiny"
    assert any(c["name"] == "agent-terminal-ui" for c in rep["checks"])
