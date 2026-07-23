"""Tests for the graph_epistemic MCP tool (CONCEPT:AU-KB-CURRENCY).

Mirrors the ``_CollectingMCP`` + monkeypatched-client pattern of
``test_audit_tools.py`` / ``test_engine_tools_scope_policy.py`` — no live
engine required. ``graph_epistemic`` dispatches through
``engine_tools._dispatch``, so the client is faked the same way
``test_engine_tools_scope_policy.py`` does (monkeypatch ``engine_tools.
_client_for``).
"""

from __future__ import annotations

import json

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import engine_tools
from agent_utilities.mcp.tools.epistemic_tools import register_epistemic_tools


class _CollectingMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}
        self.descriptions: dict[str, str] = {}

    def tool(self, *, name, description="", tags=None):  # noqa: ANN001
        def _deco(fn):
            self.tools[name] = fn
            self.descriptions[name] = description
            return fn

        return _deco


def _fake_query_client(responses: dict[str, object]):
    calls: list[tuple[str, dict]] = []

    class _Query:
        def __getattr__(self, name):
            def _call(**kwargs):
                calls.append((name, kwargs))
                if name in responses:
                    result = responses[name]
                    if isinstance(result, Exception):
                        raise result
                    return result
                raise AttributeError(name)

            return _call

    class _Client:
        query = _Query()

    return _Client(), calls


def _register(monkeypatch, responses):
    mcp = _CollectingMCP()
    register_epistemic_tools(mcp)
    client, calls = _fake_query_client(responses)
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)
    return mcp.tools["graph_epistemic"], calls


def test_registered_on_graphos_tool_table():
    mcp = _CollectingMCP()
    register_epistemic_tools(mcp)
    assert "graph_epistemic" in mcp.tools
    assert kg_server.REGISTERED_TOOLS.get("graph_epistemic") is not None
    assert kg_server.ACTION_TOOL_ROUTES.get("graph_epistemic") == "/epistemic"


def test_status_action_dispatches_epistemic_status(monkeypatch):
    tool, calls = _register(
        monkeypatch,
        {"epistemic_status": {"believed": True, "since": 100, "evidence": ["ev:1"]}},
    )
    out = json.loads(tool(action="status", node_id="claim:1"))
    assert out["surface"] == "epistemic"
    assert out["action"] == "status"
    assert out["engine_method"] == "epistemic_status"
    assert out["result"]["believed"] is True
    assert calls == [("epistemic_status", {"node_id": "claim:1"})]


def test_why_action_dispatches_explain_belief_with_disclosure_level(monkeypatch):
    tool, calls = _register(
        monkeypatch,
        {"explain_belief": {"root": {"claim": "claim:1", "rule": "Asserted"}}},
    )
    out = json.loads(tool(action="why", node_id="claim:1", disclosure_level="Skeleton"))
    assert out["result"]["root"]["rule"] == "Asserted"
    assert calls == [
        ("explain_belief", {"node_id": "claim:1", "disclosure_level": "Skeleton"})
    ]


def test_why_action_without_disclosure_level_omits_it(monkeypatch):
    tool, calls = _register(monkeypatch, {"explain_belief": {"root": {}}})
    tool(action="why", node_id="claim:1", disclosure_level="")
    assert calls == [("explain_belief", {"node_id": "claim:1"})]


def test_why_requires_node_id(monkeypatch):
    tool, calls = _register(monkeypatch, {})
    out = json.loads(tool(action="why", node_id=""))
    assert "error" in out
    assert calls == []


def test_what_changed_action(monkeypatch):
    tool, calls = _register(
        monkeypatch, {"what_changed": {"added": [], "removed": [], "modified": []}}
    )
    out = json.loads(tool(action="what_changed", tx_from=100, tx_to=200))
    assert out["engine_method"] == "what_changed"
    assert calls == [("what_changed", {"tx_from": 100, "tx_to": 200})]


def test_resolve_conflict_action(monkeypatch):
    tool, calls = _register(
        monkeypatch, {"resolve_conflict": {"accepted": ["claim:1"], "rejected": []}}
    )
    out = json.loads(
        tool(
            action="resolve_conflict",
            node_ids=json.dumps(["claim:1", "claim:2"]),
            semantics="grounded",
        )
    )
    assert out["result"]["accepted"] == ["claim:1"]
    assert calls == [
        (
            "resolve_conflict",
            {"node_ids": ["claim:1", "claim:2"], "semantics": "grounded"},
        )
    ]


def test_resolve_conflict_requires_non_empty_node_ids(monkeypatch):
    tool, calls = _register(monkeypatch, {})
    out = json.loads(tool(action="resolve_conflict", node_ids="[]"))
    assert "error" in out
    assert calls == []


def test_degrades_cleanly_when_engine_lacks_epistemic_tms(monkeypatch):
    tool, _calls = _register(
        monkeypatch, {"epistemic_status": RuntimeError("epistemic-tms not built")}
    )
    out = json.loads(tool(action="status", node_id="claim:1"))
    assert "error" in out["result"]


def test_unknown_action(monkeypatch):
    tool, calls = _register(monkeypatch, {})
    out = json.loads(tool(action="bogus"))
    assert "error" in out
    assert calls == []


# ── why_not / what_would_invalidate (W0.10) ─────────────────────────────────
# Both project a field out of epistemic_status's OWN response rather than
# calling a separate engine method (none exists on the wire — see the
# epistemic_tools module docstring). The mocked response below is the REAL
# wire shape (`EpistemicStatusResult` nests every field under a "status" key —
# confirmed against `crates/eg-types/src/protocol.rs` /
# `tests/advanced_crossmodal_roundtrip.rs` in the epistemic-graph engine repo),
# not the flattened stand-in the other `status`-action tests above use.


def test_why_not_and_what_would_invalidate_documented_in_description():
    mcp = _CollectingMCP()
    register_epistemic_tools(mcp)
    description = mcp.descriptions["graph_epistemic"]
    assert "'why_not'" in description
    assert "'what_would_invalidate'" in description


def test_why_not_action_dispatches_epistemic_status_and_projects_the_field(
    monkeypatch,
):
    tool, calls = _register(
        monkeypatch,
        {
            "epistemic_status": {
                "status": {
                    "claim": "claim:1",
                    "believed": False,
                    "confidence": 0.2,
                    "why_not": {
                        "claim": "claim:1",
                        "reason": "InsufficientConfidence",
                        "blockers": [],
                        "competing": [],
                        "confidence": 0.2,
                    },
                    "what_would_invalidate": None,
                }
            }
        },
    )
    out = json.loads(tool(action="why_not", node_id="claim:1"))
    assert out["surface"] == "epistemic"
    assert out["action"] == "why_not"
    assert out["engine_method"] == "epistemic_status"
    assert out["result"] == {
        "claim": "claim:1",
        "reason": "InsufficientConfidence",
        "blockers": [],
        "competing": [],
        "confidence": 0.2,
    }
    # dispatches the SAME engine call 'status' makes — no separate wire method.
    assert calls == [("epistemic_status", {"node_id": "claim:1"})]


def test_why_not_action_is_null_when_the_claim_is_believed(monkeypatch):
    tool, _calls = _register(
        monkeypatch,
        {
            "epistemic_status": {
                "status": {"claim": "claim:1", "believed": True, "why_not": None}
            }
        },
    )
    out = json.loads(tool(action="why_not", node_id="claim:1"))
    assert out["result"] is None


def test_why_not_requires_node_id(monkeypatch):
    tool, calls = _register(monkeypatch, {})
    out = json.loads(tool(action="why_not", node_id=""))
    assert "error" in out
    assert calls == []


def test_what_would_invalidate_action_projects_the_field(monkeypatch):
    tool, calls = _register(
        monkeypatch,
        {
            "epistemic_status": {
                "status": {
                    "claim": "claim:1",
                    "believed": True,
                    "what_would_invalidate": {
                        "claim": "claim:1",
                        "believed_now": True,
                        "evidence_ids": ["ev:1"],
                        "believed_after": False,
                    },
                }
            }
        },
    )
    out = json.loads(tool(action="what_would_invalidate", node_id="claim:1"))
    assert out["action"] == "what_would_invalidate"
    assert out["engine_method"] == "epistemic_status"
    assert out["result"] == {
        "claim": "claim:1",
        "believed_now": True,
        "evidence_ids": ["ev:1"],
        "believed_after": False,
    }
    assert calls == [("epistemic_status", {"node_id": "claim:1"})]


def test_what_would_invalidate_is_null_when_no_flip_exists(monkeypatch):
    tool, _calls = _register(
        monkeypatch,
        {
            "epistemic_status": {
                "status": {"claim": "solo", "what_would_invalidate": None}
            }
        },
    )
    out = json.loads(tool(action="what_would_invalidate", node_id="solo"))
    assert out["result"] is None


def test_what_would_invalidate_requires_node_id(monkeypatch):
    tool, calls = _register(monkeypatch, {})
    out = json.loads(tool(action="what_would_invalidate", node_id=""))
    assert "error" in out
    assert calls == []


def test_why_not_propagates_dispatch_error_without_projecting(monkeypatch):
    """An engine-level failure (e.g. epistemic-tms not built) must surface as
    the error dict, never be silently swallowed into a projected `None`."""
    tool, _calls = _register(
        monkeypatch, {"epistemic_status": RuntimeError("epistemic-tms not built")}
    )
    out = json.loads(tool(action="why_not", node_id="claim:1"))
    assert "error" in out["result"]


def test_what_would_invalidate_propagates_dispatch_error_without_projecting(
    monkeypatch,
):
    tool, _calls = _register(
        monkeypatch, {"epistemic_status": RuntimeError("epistemic-tms not built")}
    )
    out = json.loads(tool(action="what_would_invalidate", node_id="claim:1"))
    assert "error" in out["result"]
