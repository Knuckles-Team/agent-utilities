"""Seam 7 — close the wire-first gaps the catalog audit found: the X-8 Agent Digital
Twin (CONCEPT:AU-ORCH.twin.agent-digital-twin) and the X-4 routing-explainability
function (CONCEPT:AU-P1-3, `capability_routing.explain_routing_eligibility`) were
implemented + unit-tested but reachable from NO MCP tool / REST route.

Both are now new actions on existing tools (per the "Two surfaces by default" edict —
``ACTION_TOOL_ROUTES``/``_execute_tool`` parity means a new action= on an already-routed
tool gets its REST twin for free):

* ``graph_runvcs`` — ``twin_capture`` / ``twin_replay`` / ``twin_counterfactual`` /
  ``twin_incident`` (state_tools.py).
* ``ontology_interface`` — ``explain_routing_eligibility`` (ontology_tools.py).

These tests exercise the LIVE path through ``kg_server._execute_tool`` (the single
core both MCP and the REST gateway dispatch through) and assert on outcomes that only
the REAL ``agent_digital_twin``/``capability_routing`` functions could have produced
(hydrated ids from a fake KG, a genuine policy-recompute divergence, a genuine
ontology-subsumption path) — not a stub returning a canned shape.
"""

from __future__ import annotations

import json
import types
from typing import Any

import pytest

from agent_utilities.mcp import kg_server

pytestmark = [
    pytest.mark.concept("AU-ORCH.twin.agent-digital-twin"),
    pytest.mark.asyncio,
]


# ---------------------------------------------------------------------------
# X-8 — graph_runvcs twin_* actions
# ---------------------------------------------------------------------------


class _FakeTwinEngine:
    """Minimal engine double covering exactly what ``capture_twin_from_kg``/
    ``persist_twin`` read/write: ``query_cypher`` (2 read shapes), ``add_node``,
    ``link_nodes`` — mirrors ``tests/unit/orchestration/test_agent_digital_twin.py``'s
    identically-shaped fixture."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        props = dict(properties or {})
        props["label"] = node_type
        self.nodes[node_id] = props
        return props

    def link_nodes(self, source_id: str, target_id: str, rel_type: str) -> None:
        self.edges.append((source_id, target_id, rel_type))

    def query_cypher(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        if "MADE_TOOL_CALL" in cypher:
            tid = params["tid"]
            return [
                dict(row, id=nid)
                for nid, row in self.nodes.items()
                if row.get("label") == "ToolCall" and row.get("_trace_id") == tid
            ]
        if "WorkItem {correlation_id" in cypher:
            cid = params["cid"]
            return [
                {"id": nid}
                for nid, row in self.nodes.items()
                if row.get("label") == "WorkItem" and row.get("correlation_id") == cid
            ]
        return []


def _seed_engine() -> _FakeTwinEngine:
    engine = _FakeTwinEngine()
    engine.add_node(
        "workitem:seam7-1",
        "WorkItem",
        properties={"correlation_id": "run:seam7-demo"},
    )
    engine.add_node(
        "toolcall:seam7-demo:0",
        "ToolCall",
        properties={
            "_trace_id": "trace:run:seam7-demo",
            "tool_name": "kg_query",
            "args": "{}",
            "result_preview": "ok",
            "error": "",
            "sequence": 0,
        },
    )
    return engine


async def test_graph_runvcs_twin_capture_hydrates_from_the_real_kg_and_persists(
    monkeypatch,
):
    engine = _seed_engine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    kg_server.ensure_tools_registered()
    assert kg_server.ACTION_TOOL_ROUTES.get("graph_runvcs") == "/graph/runvcs"

    res = await kg_server._execute_tool(
        "graph_runvcs",
        action="twin_capture",
        run_id="run:seam7-demo",
        agent_name="agent-utilities-expert",
        versions=json.dumps({"model_id": "gpt-test-1", "catalog_epoch": 7}),
        outcome="succeeded",
    )
    payload = json.loads(res)
    assert payload["action"] == "twin_capture"
    twin = payload["twin"]
    # Hydrated from the FAKE KG's real rows via capture_twin_from_kg — not fabricated.
    assert twin["run_id"] == "run:seam7-demo"
    assert twin["work_item_ids"] == ["workitem:seam7-1"]
    assert len(twin["tool_call_ids"]) == 1
    assert twin["versions"]["model_id"] == "gpt-test-1"

    # persist_twin really wrote a durable :AgentDigitalTwin node + reference edges.
    node_id = payload["node_id"]
    assert node_id in engine.nodes
    assert engine.nodes[node_id]["label"] == "AgentDigitalTwin"
    assert (node_id, "trace:run:seam7-demo", "TWIN_OF") in engine.edges
    assert (node_id, "workitem:seam7-1", "REFERENCES") in engine.edges


async def test_graph_runvcs_twin_replay_is_deterministic_through_the_mcp_action(
    monkeypatch,
):
    engine = _seed_engine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    kg_server.ensure_tools_registered()

    captured = await kg_server._execute_tool(
        "graph_runvcs", action="twin_capture", run_id="run:seam7-demo", persist=False
    )
    twin_json = json.dumps(json.loads(captured)["twin"])

    res = await kg_server._execute_tool(
        "graph_runvcs", action="twin_replay", twin=twin_json
    )
    report = json.loads(res)
    assert report["action"] == "twin_replay"
    assert report["deterministic"] is True
    assert report["steps"] == 1  # exactly the one hydrated tool call
    assert report["model_calls"] == 0


async def test_graph_runvcs_twin_counterfactual_recomputes_a_real_policy_decision(
    monkeypatch,
):
    from agent_utilities.orchestration.action_policy import (
        ActionDecision,
        ActionRequest,
    )
    from agent_utilities.orchestration.agent_digital_twin import (
        VersionPins,
        capture_twin,
    )

    twin_obj = capture_twin(
        agent_name="agent-utilities-expert",
        versions=VersionPins(policy_version="1", policy_digest="policyhash1"),
        run_id="run:seam7-cf",
        policy_decisions=[
            ActionDecision(
                decision="allow",
                tier="auto",
                request=ActionRequest(kind="diagnose", target="fleet", source="test"),
                reason="tier auto",
                rule_origin="file",
                audit_id="action_decision:seam7-1",
            )
        ],
    )
    twin_json = json.dumps(twin_obj.to_dict())

    monkeypatch.setattr(kg_server, "_get_engine", lambda: None)
    kg_server.ensure_tools_registered()

    stricter_policy = {
        "version": 2,
        "defaults": {"tier": "approval_required"},
        "rules": [{"kind": "diagnose", "target": "*", "tier": "approval_required"}],
    }
    res = await kg_server._execute_tool(
        "graph_runvcs",
        action="twin_counterfactual",
        twin=twin_json,
        versions=json.dumps({"policy_version": "2", "policy_digest": "policyhash2"}),
        policy_overrides=json.dumps(stricter_policy),
    )
    report = json.loads(res)
    assert report["action"] == "twin_counterfactual"
    assert report["diverged"] is True
    assert report["version_delta"]["policy_version"] == ["1", "2"]
    assert len(report["decision_delta"]) == 1
    delta = report["decision_delta"][0]
    assert delta["original"]["decision"] == "allow"
    # ActionPolicy.decide() genuinely recomputed this — a hardcoded fake could never
    # produce "queue_approval" from a hand-authored ruleset it never actually loaded.
    assert delta["counterfactual"]["decision"] == "queue_approval"


async def test_graph_runvcs_twin_incident_walks_the_real_recorded_run(monkeypatch):
    engine = _seed_engine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    kg_server.ensure_tools_registered()

    captured = await kg_server._execute_tool(
        "graph_runvcs", action="twin_capture", run_id="run:seam7-demo", persist=False
    )
    twin_json = json.dumps(json.loads(captured)["twin"])

    res = await kg_server._execute_tool(
        "graph_runvcs", action="twin_incident", twin=twin_json
    )
    payload = json.loads(res)
    assert payload["action"] == "twin_incident"
    assert payload["run_id"] == "run:seam7-demo"
    assert [s["schema_ref"] for s in payload["steps"]] == ["tool_call:kg_query"]
    assert payload["steps"][0]["declaration"]["tool_name"] == "kg_query"


async def test_graph_runvcs_twin_actions_require_expected_inputs(monkeypatch):
    monkeypatch.setattr(kg_server, "_get_engine", lambda: None)
    kg_server.ensure_tools_registered()

    res = await kg_server._execute_tool("graph_runvcs", action="twin_capture")
    assert "error" in json.loads(res)

    res = await kg_server._execute_tool("graph_runvcs", action="twin_replay")
    assert "error" in json.loads(res)


# ---------------------------------------------------------------------------
# X-4 — ontology_interface explain_routing_eligibility action
# ---------------------------------------------------------------------------


def _make_routing_engine(nodes: dict[str, dict[str, Any]]) -> Any:
    graph = types.SimpleNamespace(
        node_ids=lambda: list(nodes.keys()),
        _get_node_properties=lambda nid: nodes.get(nid, {}),
    )
    return types.SimpleNamespace(graph=graph)


async def test_ontology_interface_explain_routing_eligibility_reaches_the_real_function(
    monkeypatch,
):
    nodes = {
        "mtls_tool": {
            "type": "tool",
            "capabilities": ["EncryptedTransport"],
            "tenant": "tenant-a",
            "policy_tags": ["cleared"],
        },
    }
    engine = _make_routing_engine(nodes)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    kg_server.ensure_tools_registered()
    assert (
        kg_server.ACTION_TOOL_ROUTES.get("ontology_interface") == "/ontology/interface"
    )

    res = await kg_server._execute_tool(
        "ontology_interface",
        action="explain_routing_eligibility",
        entity_id="mtls_tool",
        required_capability_type="TransportCapability",
        tenant="tenant-a",
        policy_tags="cleared",
    )
    report = json.loads(res)
    assert report["action"] == "explain_routing_eligibility"
    assert report["eligible"] is True
    # The real ontology-subsumption walk (EncryptedTransport ⊑ TransportCapability),
    # not a canned answer — only the actual capability_hierarchy could produce this path.
    assert report["subsumption_paths"] == {
        "TransportCapability": ["EncryptedTransport", "TransportCapability"]
    }
    assert report["tenant_match"] is True
    assert report["policy_matched"] is True


async def test_ontology_interface_explain_routing_eligibility_reports_why_ineligible(
    monkeypatch,
):
    nodes = {
        "dns_tool": {"type": "tool", "capabilities": ["DNSCapability"]},
    }
    engine = _make_routing_engine(nodes)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    kg_server.ensure_tools_registered()

    res = await kg_server._execute_tool(
        "ontology_interface",
        action="explain_routing_eligibility",
        entity_id="dns_tool",
        required_capability_type="TransportCapability",
    )
    report = json.loads(res)
    assert report["eligible"] is False
    assert report["missing_caps"] == ["TransportCapability"]


async def test_ontology_interface_explain_routing_eligibility_requires_inputs(
    monkeypatch,
):
    monkeypatch.setattr(kg_server, "_get_engine", lambda: None)
    kg_server.ensure_tools_registered()

    res = await kg_server._execute_tool(
        "ontology_interface", action="explain_routing_eligibility"
    )
    assert "error" in json.loads(res)
