"""BUG-5 (kg-exhaustive-smoke.md): ``graph_analyze``/``graph_evaluate`` actions
``evaluate``/``evolve_model``/``forecast``/``causal``/``invariant`` used to
``return f"Action '{action}' executed successfully."`` and ``security_scan``
returned ``f"Security scan executed on {target}."`` — hardcoded canned-success
strings that did NOTHING regardless of input, silently reporting success for a
no-op (worse than an error: a caller gating on "success" is misled).

The fix replaces each with an honest ``{"status": "not_implemented", "error":
...}`` payload naming the real tool/service to use instead — no fake success,
ever, for these six actions.
"""

from __future__ import annotations

import asyncio
import json

from agent_utilities.mcp import kg_server


def _get_tool():
    kg_server.ensure_tools_registered()
    return kg_server.REGISTERED_TOOLS["graph_analyze"]


def _run(monkeypatch, action: str) -> dict:
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    tool = _get_tool()
    out = asyncio.run(tool(action=action, target="some-target"))
    return json.loads(out)


def test_evaluate_no_longer_fake_succeeds(monkeypatch):
    payload = _run(monkeypatch, "evaluate")
    assert payload["status"] == "not_implemented"
    assert "not implemented" in payload["error"].lower()
    assert "executed successfully" not in payload["error"].lower()


def test_evolve_model_no_longer_fake_succeeds(monkeypatch):
    payload = _run(monkeypatch, "evolve_model")
    assert payload["status"] == "not_implemented"
    assert "data-science-mcp" in payload["error"]


def test_forecast_no_longer_fake_succeeds(monkeypatch):
    payload = _run(monkeypatch, "forecast")
    assert payload["status"] == "not_implemented"
    assert "deep_forecast" in payload["error"] or "engine_timeseries" in payload["error"]


def test_causal_no_longer_fake_succeeds(monkeypatch):
    payload = _run(monkeypatch, "causal")
    assert payload["status"] == "not_implemented"
    assert "graph_ops_causal" in payload["error"]


def test_invariant_no_longer_fake_succeeds(monkeypatch):
    payload = _run(monkeypatch, "invariant")
    assert payload["status"] == "not_implemented"
    assert "formal_reasoning_core" in payload["error"]


def test_security_scan_no_longer_fake_succeeds(monkeypatch):
    payload = _run(monkeypatch, "security_scan")
    assert payload["status"] == "not_implemented"
    assert "executed on" not in payload["error"].lower()
    assert payload["target"] == "some-target"


def test_all_six_actions_are_json_not_plain_strings(monkeypatch):
    # Every one of these must be parseable JSON with an explicit status — not
    # a bare success sentence a caller could mistake for anything but data.
    for action in (
        "evaluate",
        "evolve_model",
        "forecast",
        "causal",
        "invariant",
        "security_scan",
    ):
        payload = _run(monkeypatch, action)
        assert payload["action"] == action
        assert payload["status"] == "not_implemented"
