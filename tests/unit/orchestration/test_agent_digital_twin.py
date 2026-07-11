"""CONCEPT:AU-ORCH.twin.agent-digital-twin — Agent Digital Twin + deterministic replay (Codex X-8).

Exercises: capturing a twin from a (mock) run, deterministic regression replay (same
versions -> identical outcome), counterfactual replay (policy swap -> a detectable
delta; model/prompt swap -> a detectable stream delta), querying the run graph/
decisions/evidence, best-effort KG hydration + persistence, and incident-investigation
stepping. No live epistemic-graph engine required — a minimal fake engine stands in for
the two spots this module reads/writes.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.orchestration.action_policy import ActionDecision, ActionRequest
from agent_utilities.orchestration.agent_digital_twin import (
    AgentDigitalTwin,
    VersionPins,
    capture_twin,
    capture_twin_from_kg,
    counterfactual_replay,
    persist_twin,
    replay_twin,
    twin_incident_steps,
)
from agent_utilities.orchestration.work_item import WorkItemStatus


def _versions(**overrides: Any) -> VersionPins:
    base = {
        "model_id": "gpt-test-1",
        "model_provider": "test-provider",
        "prompt_version_id": "prompt:v1",
        "tool_versions": {"kg_query": "1.0.0"},
        "skill_versions": {"kg-query": "1.2.0"},
        "policy_version": "1",
        "policy_digest": "policyhash1",
        "catalog_epoch": 42,
    }
    base.update(overrides)
    return VersionPins(**base)


def _sample_tool_calls() -> list[dict[str, Any]]:
    return [
        {
            "tool_name": "kg_query",
            "args": '{"cypher": "MATCH (n) RETURN n LIMIT 1"}',
            "result": '{"rows": [{"id": "n1"}]}',
            "error": "",
        },
        {
            "tool_name": "kg_search",
            "args": '{"query": "digital twin"}',
            "result": '{"hits": 3}',
            "error": "",
        },
    ]


def _sample_decision() -> ActionDecision:
    return ActionDecision(
        decision="allow",
        tier="auto",
        request=ActionRequest(kind="diagnose", target="fleet", source="test"),
        reason="tier auto",
        rule_origin="file",
        audit_id="action_decision:sample-1",
    )


def _base_twin(**capture_overrides: Any) -> AgentDigitalTwin:
    kwargs: dict[str, Any] = dict(
        agent_name="agent-utilities-expert",
        task="diagnose fleet health",
        versions=_versions(),
        run_id="run:x8-demo",
        budget={"max_tokens": 20000, "max_cost_usd": 1.5},
        work_item_ids=["workitem:aaa111", "workitem:bbb222"],
        tool_calls=_sample_tool_calls(),
        policy_decisions=[_sample_decision()],
        evidence=[{"answer_candidate": "fleet is healthy", "confidence": None}],
        outcome=WorkItemStatus.SUCCEEDED.value,
    )
    kwargs.update(capture_overrides)
    return capture_twin(**kwargs)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def test_capture_twin_from_a_mock_run_records_full_shape():
    twin = _base_twin()

    assert twin.run_id == "run:x8-demo"
    assert twin.agent_name == "agent-utilities-expert"
    assert twin.versions.model_id == "gpt-test-1"
    assert twin.budget == {"max_tokens": 20000, "max_cost_usd": 1.5}
    assert twin.run_graph() == ["workitem:aaa111", "workitem:bbb222"]
    assert len(twin.tool_call_ids) == 2
    assert twin.tool_call_ids[0].startswith("toolcall:x8-demo:")
    assert len(twin.decision_ids) == 1
    assert twin.decision_ids[0] == "action_decision:sample-1"
    assert twin.outcome == WorkItemStatus.SUCCEEDED.value
    # Every tool call + decision was mirrored into the run-VCS event log as a
    # declare/capture pair (2 events each).
    assert len(twin.event_log.events) == (2 + 1) * 2


def test_twin_query_surfaces_decisions_and_evidence():
    twin = _base_twin()
    decisions = twin.decisions()
    assert decisions[0]["decision"] == "allow"
    assert decisions[0]["request"]["kind"] == "diagnose"
    evidence = twin.evidence_bundles()
    assert evidence[0]["answer_candidate"] == "fleet is healthy"


def test_twin_roundtrips_through_to_dict_from_dict():
    twin = _base_twin()
    restored = AgentDigitalTwin.from_dict(twin.to_dict())

    assert restored.run_id == twin.run_id
    assert restored.versions == twin.versions
    assert restored.work_item_ids == twin.work_item_ids
    assert restored.tool_call_ids == twin.tool_call_ids
    assert [e.record_id for e in restored.event_log.events] == [
        e.record_id for e in twin.event_log.events
    ]
    assert [e.ordinal for e in restored.event_log.events] == [
        e.ordinal for e in twin.event_log.events
    ]
    # A restored twin still replays deterministically.
    assert replay_twin(restored).deterministic


# ---------------------------------------------------------------------------
# Deterministic replay — regression
# ---------------------------------------------------------------------------


def test_replay_twin_regression_is_deterministic():
    twin = _base_twin()
    report = replay_twin(twin)

    assert report.deterministic
    assert report.regression.model_calls == 0  # no model_exchange events in this twin
    assert report.regression.steps == 3  # 2 tool calls + 1 policy decision


def test_replay_twin_with_model_exchanges_never_calls_a_live_model():
    twin = _base_twin(
        model_exchanges=[
            {"request": "summarize fleet health", "response": "all green"},
        ]
    )
    report = replay_twin(twin)

    assert report.deterministic
    assert report.regression.model_calls == 1
    assert report.regression.reconstructed[0] == "all green"


# ---------------------------------------------------------------------------
# Counterfactual — policy swap
# ---------------------------------------------------------------------------


def test_counterfactual_policy_swap_produces_detectable_delta():
    twin = _base_twin()  # recorded decision: diagnose/fleet -> allow (auto tier)

    # A stricter counterfactual policy: diagnose now requires approval.
    stricter_policy = {
        "version": 2,
        "defaults": {"tier": "approval_required"},
        "rules": [{"kind": "diagnose", "target": "*", "tier": "approval_required"}],
    }
    report = counterfactual_replay(
        twin,
        versions=_versions(policy_version="2", policy_digest="policyhash2"),
        policy_overrides=stricter_policy,
    )

    assert report.diverged
    assert report.version_delta["policy_version"] == ("1", "2")
    assert len(report.decision_delta) == 1
    delta = report.decision_delta[0]
    assert delta["original"]["decision"] == "allow"
    assert delta["counterfactual"]["decision"] == "queue_approval"


def test_counterfactual_policy_swap_with_identical_rules_has_no_delta():
    twin = _base_twin()
    same_policy = {
        "version": 1,
        "defaults": {"tier": "approval_required"},
        "rules": [{"kind": "diagnose", "target": "*", "tier": "auto"}],
    }
    report = counterfactual_replay(twin, policy_overrides=same_policy)

    assert report.decision_delta == []
    assert report.regression.deterministic


# ---------------------------------------------------------------------------
# Counterfactual — model/prompt swap
# ---------------------------------------------------------------------------


def test_counterfactual_model_swap_surfaces_stream_divergence():
    twin = _base_twin(
        model_exchanges=[{"request": "recommend an action", "response": "scale up"}]
    )
    baseline = replay_twin(twin)
    assert baseline.deterministic

    report = counterfactual_replay(
        twin,
        versions=_versions(model_id="claude-test-2"),
        model_responses={"recommend an action": "restart service"},
    )

    assert report.diverged
    assert report.version_delta["model_id"] == ("gpt-test-1", "claude-test-2")
    assert report.regression.reconstructed[0] == "restart service"
    # The counterfactual stream digest differs from the ORIGINAL recorded stream.
    assert report.regression.replay_digest != report.regression.original_digest


def test_counterfactual_with_no_overrides_matches_baseline_regression():
    """Passing only ``versions`` (no policy/model override) reports the version
    delta for visibility but does not itself change the replay outcome."""
    twin = _base_twin()
    report = counterfactual_replay(twin, versions=_versions(model_id="other-model"))

    assert report.version_delta == {"model_id": ("gpt-test-1", "other-model")}
    assert not report.diverged
    assert report.regression.replay_digest == report.regression.original_digest


# ---------------------------------------------------------------------------
# Incident investigation
# ---------------------------------------------------------------------------


def test_twin_incident_steps_walks_the_recorded_run_in_order():
    twin = _base_twin(
        model_exchanges=[{"request": "diagnose?", "response": "looks fine"}]
    )
    steps = twin_incident_steps(twin)

    assert [s["schema_ref"] for s in steps] == [
        "model_exchange",
        "tool_call:kg_query",
        "tool_call:kg_search",
        "policy_decision",
    ]
    assert steps[0]["capture"] == {"response": "looks fine"}
    assert steps[1]["declaration"]["tool_name"] == "kg_query"
    assert steps[3]["declaration"]["request"]["kind"] == "diagnose"
    assert steps[3]["capture"]["decision"]["decision"] == "allow"
    # Ordinals are strictly increasing (causal order).
    assert [s["ordinal"] for s in steps] == sorted(s["ordinal"] for s in steps)


# ---------------------------------------------------------------------------
# VersionPins
# ---------------------------------------------------------------------------


def test_version_pins_digest_is_stable_and_diff_is_field_precise():
    a = _versions()
    b = _versions()
    assert a.digest() == b.digest()

    c = _versions(model_id="different-model")
    assert a.digest() != c.digest()
    assert a.diff(c) == {"model_id": ("gpt-test-1", "different-model")}


# ---------------------------------------------------------------------------
# KG hydration + persistence (best-effort; a minimal fake engine)
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Minimal engine double covering exactly what ``agent_digital_twin`` reads/writes:
    ``query_cypher`` (2 read shapes), ``add_node``, ``link_nodes``."""

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


def test_capture_twin_from_kg_hydrates_tool_calls_and_work_items():
    engine = _FakeEngine()
    engine.add_node(
        "workitem:hydrated-1",
        "WorkItem",
        properties={"correlation_id": "run:hydrate-me"},
    )
    engine.add_node(
        "toolcall:hydrate-me:0",
        "ToolCall",
        properties={
            "_trace_id": "trace:run:hydrate-me",
            "tool_name": "kg_query",
            "args": "{}",
            "result_preview": "ok",
            "error": "",
            "sequence": 0,
        },
    )

    twin = capture_twin_from_kg(
        engine,
        "run:hydrate-me",
        agent_name="agent-utilities-expert",
        versions=_versions(),
        outcome=WorkItemStatus.SUCCEEDED.value,
    )

    assert twin.work_item_ids == ["workitem:hydrated-1"]
    assert len(twin.tool_call_ids) == 1
    assert replay_twin(twin).deterministic


def test_capture_twin_from_kg_degrades_gracefully_on_a_cold_graph():
    class _ColdEngine:
        def query_cypher(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            raise RuntimeError("backend unavailable")

    twin = capture_twin_from_kg(
        _ColdEngine(), "run:cold", versions=_versions(), agent_name="a"
    )
    assert twin.work_item_ids == []
    assert twin.tool_call_ids == []


def test_persist_twin_writes_node_and_reference_edges():
    engine = _FakeEngine()
    twin = _base_twin()

    node_id = persist_twin(engine, twin)

    assert node_id == twin.twin_id
    assert engine.nodes[node_id]["label"] == "AgentDigitalTwin"
    assert engine.nodes[node_id]["run_id"] == "run:x8-demo"
    assert (node_id, "trace:run:x8-demo", "TWIN_OF") in engine.edges
    assert (node_id, "workitem:aaa111", "REFERENCES") in engine.edges
    assert (node_id, twin.tool_call_ids[0], "REFERENCES") in engine.edges
    assert (node_id, "action_decision:sample-1", "REFERENCES") in engine.edges


def test_persist_twin_is_a_noop_without_an_engine():
    twin = _base_twin()
    assert persist_twin(None, twin) is None
