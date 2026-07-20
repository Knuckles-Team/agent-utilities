"""Escalation-decision policy (``reports/autonomous-sdlc-loop-design.md`` §3.4).

``evaluate_escalation`` fires an :EscalationRequest on each of the six signals and
stays silent (returns None) when the transition is clean — the
fuller-autonomy-with-escalation model. Mirrors the fake-KG style of
``test_observability_incidents.py`` / the lifecycle-orchestrator tests.

@pytest.mark.concept("AU-OS.host.report-only-remediation-proposal")
"""

from __future__ import annotations

from typing import Any

import pytest

import agent_utilities.observability.escalation_policy as ep
import agent_utilities.observability.health_ingest as hi

pytestmark = pytest.mark.concept("AU-OS.host.report-only-remediation-proposal")


class _FakeEngine:
    """Serves node_props (by id) + get_nodes_by_label + out_edges from tables."""

    def __init__(self, nodes=None, by_label=None, edges=None):
        self._nodes = nodes or {}
        self._by_label = by_label or {}
        self._edges = edges or []

    @property
    def graph(self):
        outer = self

        class _G:
            @property
            def nodes(self):
                return outer._nodes

        return _G()

    backend = None

    def get_nodes_by_label(self, label, limit=0):
        return self._by_label.get(label, [])

    def out_edges(self, node_id, data=False):
        return [(s, t, {"rel_type": r}) for (s, r, t) in self._edges if s == node_id]


@pytest.fixture
def _capture_writes(monkeypatch):
    calls: list[dict[str, Any]] = []

    def _fake(entities, relationships=None, *, source, domain, **kw):
        calls.append({"entities": entities, "relationships": relationships or []})
        return {"nodes": len(entities), "edges": len(relationships or [])}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.memory.native_ingest.ingest_entities", _fake
    )
    return calls


# --- silent when clean ---------------------------------------------------- #
def test_clean_transition_does_not_escalate(_capture_writes):
    engine = _FakeEngine()
    out = ep.evaluate_escalation(
        {"entry": "spec1", "transition": "develop_spec", "diff_files": 2},
        engine=engine,
    )
    assert out is None
    assert _capture_writes == []  # nothing written


# --- each signal fires ---------------------------------------------------- #
def test_low_confidence_degraded_run(_capture_writes):
    engine = _FakeEngine(nodes={"trace:x": {"status": "degraded"}})
    out = ep.evaluate_escalation(
        {"entry": "mr1", "transition": "develop_spec", "run_trace": "trace:x"},
        engine=engine,
    )
    assert out is not None
    assert "low_confidence" in out["signals"]
    assert out["autonomy"] == "escalate"


def test_low_confidence_reward_below_floor(_capture_writes):
    out = ep.evaluate_escalation(
        {"entry": "mr1", "transition": "t", "reward": 0.1}, engine=_FakeEngine()
    )
    assert out and "low_confidence" in out["signals"]


def test_large_diff_by_file_count(_capture_writes):
    out = ep.evaluate_escalation(
        {"entry": "mr1", "transition": "t", "diff_files": 25}, engine=_FakeEngine()
    )
    assert out and "large_diff" in out["signals"]


def test_large_diff_by_blast_radius(_capture_writes):
    out = ep.evaluate_escalation(
        {"entry": "mr1", "transition": "t", "blast_radius": 40}, engine=_FakeEngine()
    )
    assert out and "large_diff" in out["signals"]


def test_red_ci_past_retry_cap(_capture_writes):
    out = ep.evaluate_escalation(
        {
            "entry": "mr1",
            "transition": "await_ci",
            "pipeline_status": "failed",
            "attempts": 4,
        },
        engine=_FakeEngine(),
    )
    assert out and "red_ci_past_cap" in out["signals"]


def test_red_ci_below_cap_is_silent(_capture_writes):
    out = ep.evaluate_escalation(
        {
            "entry": "mr1",
            "transition": "await_ci",
            "pipeline_status": "failed",
            "attempts": 1,
        },
        engine=_FakeEngine(),
    )
    assert out is None


def test_governance_gate_always_escalates(_capture_writes):
    for kind in ("owner_signoff", "dpia", "camunda_approval"):
        out = ep.evaluate_escalation(
            {"entry": "spec1", "transition": "gate", "gate_kind": kind},
            engine=_FakeEngine(),
        )
        assert out and "governance_gate" in out["signals"], kind


def test_critical_service_blast_radius(_capture_writes):
    engine = _FakeEngine(nodes={"svc:core": {"criticality": "tier-0"}})
    out = ep.evaluate_escalation(
        {"entry": "inc1", "transition": "deploy", "service": "svc:core"},
        engine=engine,
    )
    assert out and "critical_blast_radius" in out["signals"]


def test_noncritical_service_is_silent(_capture_writes):
    engine = _FakeEngine(nodes={"svc:minor": {"criticality": "tier-3"}})
    out = ep.evaluate_escalation(
        {"entry": "inc1", "transition": "deploy", "service": "svc:minor"},
        engine=engine,
    )
    assert out is None


def test_cold_start_explicit(_capture_writes):
    out = ep.evaluate_escalation(
        {"entry": "inc1", "transition": "novel_op", "cold_start": True},
        engine=_FakeEngine(),
    )
    assert out and "cold_start" in out["signals"]


def test_cold_start_inferred_from_empty_history(_capture_writes):
    engine = _FakeEngine(by_label={"RunTrace": []})
    out = ep.evaluate_escalation(
        {
            "entry": "inc1",
            "transition": "deploy",
            "service": "svc:x",
            "scan_history": True,
        },
        engine=engine,
    )
    assert out and "cold_start" in out["signals"]


def test_cold_start_suppressed_when_prior_run_exists(_capture_writes):
    engine = _FakeEngine(
        by_label={"RunTrace": [("t1", {"transition": "deploy", "service": "svc:x"})]}
    )
    out = ep.evaluate_escalation(
        {
            "entry": "inc1",
            "transition": "deploy",
            "service": "svc:x",
            "scan_history": True,
        },
        engine=engine,
    )
    assert out is None


# --- persistence ---------------------------------------------------------- #
def test_escalation_request_is_written_with_edge(_capture_writes):
    out = ep.evaluate_escalation(
        {"entry": "spec1", "transition": "gate", "gate_kind": "dpia"},
        engine=_FakeEngine(),
    )
    assert out is not None
    assert len(_capture_writes) == 1
    call = _capture_writes[0]
    assert {e["type"] for e in call["entities"]} == {"EscalationRequest"}
    assert {r["type"] for r in call["relationships"]} == {"escalates"}
    assert call["relationships"][0]["source"] == "spec1"


def test_multiple_signals_captured_in_one_request(_capture_writes):
    out = ep.evaluate_escalation(
        {
            "entry": "mr1",
            "transition": "deploy",
            "diff_files": 30,
            "gate_kind": "dpia",
        },
        engine=_FakeEngine(),
    )
    assert out is not None
    assert set(out["signals"]) >= {"large_diff", "governance_gate"}


def test_no_engine_returns_request_unwritten(monkeypatch, _capture_writes):
    """A context-only signal still returns a request even with no engine, but does
    not write (report-only never blocks on persistence)."""
    monkeypatch.setattr(hi, "_engine", lambda: None)
    out = ep.evaluate_escalation(
        {"entry": "spec1", "transition": "gate", "gate_kind": "dpia"}
    )
    assert out and "governance_gate" in out["signals"]
    assert _capture_writes == []  # nothing written without an engine


# --- lifecycle-orchestrator consultable-gate wiring ----------------------- #
def test_lifecycle_proposals_carry_escalation(monkeypatch, _capture_writes):
    """run_lifecycle(consult_escalation=True) stamps autonomy on each proposal, and
    'escalate' when a signal fires (design §3.4)."""
    from agent_utilities.observability import lifecycle_orchestrator as lo

    engine = _FakeEngine()
    # A large-diff context makes every proposed transition escalate.
    out = lo.run_lifecycle(
        {"id": "inc1", "type": "Incident"},
        engine=engine,
        write=False,
        consult_escalation=True,
        escalation_context={"diff_files": 50},
    )
    assert out["steps"], "expected forward-chain proposals"
    assert all(s.get("autonomy") == "escalate" for s in out["steps"])
    assert all("escalation" in s for s in out["steps"])


def test_lifecycle_proposals_auto_when_clean(monkeypatch, _capture_writes):
    from agent_utilities.observability import lifecycle_orchestrator as lo

    engine = _FakeEngine()
    out = lo.run_lifecycle(
        {"id": "inc1", "type": "Incident"},
        engine=engine,
        write=False,
        consult_escalation=True,
    )
    assert out["steps"]
    assert all(s.get("autonomy") == "auto" for s in out["steps"])
