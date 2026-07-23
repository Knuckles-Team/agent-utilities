"""BUG-5 (kg-exhaustive-smoke.md): ``graph_analyze``/``graph_evaluate`` actions
``evaluate``/``evolve_model``/``forecast``/``causal``/``invariant`` used to
``return f"Action '{action}' executed successfully."`` and ``security_scan``
returned ``f"Security scan executed on {target}."`` — hardcoded canned-success
strings that did NOTHING regardless of input, silently reporting success for a
no-op (worse than an error: a caller gating on "success" is misled).

The fix replaces each with an honest ``{"status": "not_implemented", "error":
...}`` payload naming the real tool/service to use instead — no fake success,
ever, for these six actions.

Both the tool surface and the response envelope have since moved on:
``evaluate``/``evolve_model``/``forecast``/``causal``/``invariant`` were split
off ``graph_analyze`` onto the dedicated ``graph_evaluate`` tool
(``security_scan`` stayed on ``graph_analyze``); and every action now returns
the sole typed :class:`~agent_utilities.models.evidence_bundle.EvidenceBundle`
envelope, not a raw JSON string — ``EvidenceBundle.from_payload`` losslessly
retains the original ``{"status": ..., "error": ..., "action": ...}`` payload
as its (only) claim.
"""

from __future__ import annotations

import asyncio

from agent_utilities.mcp import kg_server

# evaluate/evolve_model/forecast/causal/invariant now live on graph_evaluate;
# security_scan stayed on graph_analyze.
_TOOL_FOR_ACTION = {
    "evaluate": "graph_evaluate",
    "evolve_model": "graph_evaluate",
    "forecast": "graph_evaluate",
    "causal": "graph_evaluate",
    "invariant": "graph_evaluate",
    "security_scan": "graph_analyze",
}


def _get_tool(action: str):
    kg_server.ensure_tools_registered()
    return kg_server.REGISTERED_TOOLS[_TOOL_FOR_ACTION[action]]


def _run(monkeypatch, action: str) -> dict:
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    tool = _get_tool(action)
    bundle = asyncio.run(tool(action=action, target="some-target"))
    # The not-implemented payload is the bundle's sole (lossless) claim.
    return bundle.claims[0]


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
