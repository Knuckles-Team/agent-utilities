"""Remediation playbooks on the OS-5.15 triage seam (CONCEPT:OS-5.26).

Covers: idempotent registration on the ``register_playbook()`` seam (critical
and error severities only — warnings keep the default playbook), the
service_down step list (confirm-recovered short-circuit, policy-gated restart
with health-watch verification, escalation on deny/actuation-failure), the
flapping back-off, the never-auto-act resource-pressure path, and the
per-step ``remediation_log`` trail on the originating FleetEvent node.

@pytest.mark.concept("OS-5.26")
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.adaptation import (
    fleet_event_triage,
    remediation_playbooks,
)
from agent_utilities.orchestration.fleet_actuation import (
    DryRunActuator,
    set_fleet_actuator,
)
from agent_utilities.orchestration.fleet_observation import set_fleet_observer

from .fleet_autonomy_fakes import (
    CaptureNotifier,
    FakeEngine,
    FakeObserver,
    obs,
    utc_now_str,
    write_policy,
)

pytestmark = pytest.mark.concept("OS-5.26")

PERMISSIVE = (
    "defaults: {tier: auto, rate_limit: {max: 100, window_s: 60},"
    " blast_radius: {max_targets: 100, window_s: 60}}\n"
    "rules:\n  - {kind: '*', target: '*', tier: auto}\n"
)


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def notifier(monkeypatch):
    sink = CaptureNotifier()
    from agent_utilities.knowledge_graph.actions import dispatch

    monkeypatch.setattr(dispatch, "_DEFAULT_NOTIFIER", sink)
    return sink


@pytest.fixture(autouse=True)
def _clean_seams():
    """Playbook registration + observer/actuator injection are process-global."""
    yield
    set_fleet_observer(None)
    set_fleet_actuator(None)


@pytest.fixture
def permissive_policy(engine, tmp_path, monkeypatch):
    """Route every get_action_policy() in the playbook path to a permissive one."""
    path = write_policy(tmp_path, PERMISSIVE)
    from agent_utilities.orchestration import action_policy as ap

    original = ap.ActionPolicy

    def factory(eng=None):
        return original(engine=eng, policy_path=path)

    monkeypatch.setattr(ap, "get_action_policy", factory)
    monkeypatch.setattr(
        "agent_utilities.orchestration.action_policy.get_action_policy", factory
    )
    return path


def _event(
    engine,
    subject="caddy-mcp",
    source="uptime-kuma",
    severity="critical",
    status="down",
    summary="monitor down",
):
    event_id = f"fleet_event:{subject}"
    engine.add_node(
        event_id,
        "FleetEvent",
        properties={
            "source": source,
            "severity": severity,
            "subject": subject,
            "status": status,
            "summary": summary,
            "received_at": utc_now_str(),
            "triage_status": "pending",
        },
    )
    return event_id


def _log(engine, event_id):
    raw = engine.nodes[event_id].get("remediation_log") or "[]"
    return [e["step"] for e in json.loads(raw)]


# ---------------------------------------------------------------------------
# Registration on the OS-5.15 seam
# ---------------------------------------------------------------------------


def test_registration_is_idempotent_and_scoped():
    remediation_playbooks.ensure_registered()
    remediation_playbooks.ensure_registered()
    keys = set(fleet_event_triage.PLAYBOOKS)
    assert "uptime-kuma:critical" in keys
    assert "alertmanager:error" in keys
    # Warnings keep the OS-5.15 default playbook.
    assert "uptime-kuma:warning" not in keys
    assert (
        fleet_event_triage._resolve_playbook("uptime-kuma", "warning")
        is fleet_event_triage.default_playbook
    )
    assert (
        fleet_event_triage._resolve_playbook("uptime-kuma", "critical")
        is remediation_playbooks.remediation_playbook
    )


# ---------------------------------------------------------------------------
# service_down
# ---------------------------------------------------------------------------


def test_service_down_recovered_short_circuits(engine):
    remediation_playbooks.ensure_registered()
    set_fleet_observer(FakeObserver({"caddy-mcp": obs("caddy-mcp", "up")}))
    event_id = _event(engine)
    report = fleet_event_triage.triage_fleet_event(engine, event_id)
    assert report["resolution"] == "already_recovered"
    assert "confirm" in _log(engine, event_id)
    assert engine.by_type("ActionExecution") == []  # nothing actuated


def test_service_down_restarts_and_schedules_verification(
    engine, permissive_policy, notifier
):
    remediation_playbooks.ensure_registered()
    set_fleet_observer(FakeObserver({"caddy-mcp": obs("caddy-mcp", "down")}))
    actuator = DryRunActuator()
    set_fleet_actuator(actuator)
    event_id = _event(engine)
    report = fleet_event_triage.triage_fleet_event(engine, event_id)
    assert report["decision"] == "allow"
    assert [r.kind for r in actuator.applied] == ["restart_service"]
    # Verification: a durable OS-5.27 watch was queued and recorded as a step.
    assert [t["task_type"] for t in engine.submitted] == ["deploy_watch"]
    steps = _log(engine, event_id)
    assert steps == ["observe", "confirm", "policy", "actuate", "verify"]


def test_service_down_queues_approval_under_default_policy(engine, notifier):
    remediation_playbooks.ensure_registered()
    set_fleet_observer(FakeObserver({"caddy-mcp": obs("caddy-mcp", "down")}))
    actuator = DryRunActuator()
    set_fleet_actuator(actuator)
    event_id = _event(engine)
    report = fleet_event_triage.triage_fleet_event(engine, event_id)
    assert report["decision"] == "queue_approval"
    assert actuator.applied == []
    assert len(engine.by_type("ActionApproval")) == 1
    assert any("awaits approval" in m for m in notifier.messages)


def test_service_down_escalates_on_actuation_failure(
    engine, permissive_policy, notifier
):
    remediation_playbooks.ensure_registered()
    set_fleet_observer(FakeObserver({"caddy-mcp": obs("caddy-mcp", "down")}))

    class FailingActuator:
        name = "failing"

        def apply(self, request):
            return {"ok": False, "dry_run": False, "detail": "no such service"}

    set_fleet_actuator(FailingActuator())
    event_id = _event(engine)
    fleet_event_triage.triage_fleet_event(engine, event_id)
    steps = _log(engine, event_id)
    assert "escalate" in steps
    assert any("escalation" in m for m in notifier.messages)
    assert len(engine.by_type("ActionApproval")) == 1  # escalation queued


# ---------------------------------------------------------------------------
# service_flapping
# ---------------------------------------------------------------------------


def test_flapping_backs_off_and_escalates(engine, permissive_policy, notifier):
    remediation_playbooks.ensure_registered()
    set_fleet_observer(FakeObserver({"caddy-mcp": obs("caddy-mcp", "down", flaps=4)}))
    actuator = DryRunActuator()
    set_fleet_actuator(actuator)
    event_id = _event(engine)
    report = fleet_event_triage.triage_fleet_event(engine, event_id)
    assert report["playbook"] == "service_flapping"
    assert report["resolution"] == "backed_off"
    assert actuator.applied == []  # backed off — no restart even on auto tier
    assert any("flapping" in m for m in notifier.messages)
    approvals = engine.by_type("ActionApproval")
    assert len(approvals) == 1 and approvals[0]["kind"] == "restart_service"


# ---------------------------------------------------------------------------
# resource pressure
# ---------------------------------------------------------------------------


def test_resource_pressure_never_auto_acts(engine, permissive_policy, notifier):
    remediation_playbooks.ensure_registered()
    set_fleet_observer(FakeObserver({}))
    actuator = DryRunActuator()
    set_fleet_actuator(actuator)
    event_id = _event(
        engine,
        subject="r820",
        source="alertmanager",
        status="firing",
        summary="disk usage above 95% on /var",
    )
    report = fleet_event_triage.triage_fleet_event(engine, event_id)
    assert report["playbook"] == "resource_pressure"
    assert report["resolution"] == "proposed_only"
    assert actuator.applied == []  # even a fully-permissive policy never acts
    approvals = engine.by_type("ActionApproval")
    assert [a["kind"] for a in approvals] == ["investigate_resource_pressure"]
    assert any("resource pressure" in m for m in notifier.messages)


# ---------------------------------------------------------------------------
# default-playbook behavior is preserved underneath
# ---------------------------------------------------------------------------


def test_remediation_keeps_failure_gap_escalation(engine, notifier):
    remediation_playbooks.ensure_registered()
    set_fleet_observer(FakeObserver({"caddy-mcp": obs("caddy-mcp", "down")}))
    event_id = _event(engine)
    report = fleet_event_triage.triage_fleet_event(engine, event_id)
    # The OS-5.15 default playbook still filed the failure_gap topic.
    assert report.get("gap_topic")
