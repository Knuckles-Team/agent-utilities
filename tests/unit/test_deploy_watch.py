"""Health-gated deploy watch + policy-gated rollback (CONCEPT:AU-OS.config.health-gated-deploy-rollback).

Covers: durable scheduling (the watch spec rides on the Task node so a
requeued watch resumes its ORIGINAL deadline), the sustained-green success
path, the failure path (default ``on_fail`` = ActionPolicy-gated rollback —
queued under the shipped default policy, actuated under a permissive one —
plus operator escalation), the unobserved path (notify, never roll back on
zero evidence), and custom ``on_fail`` injection.

@pytest.mark.concept("AU-OS.config.health-gated-deploy-rollback")
"""

from __future__ import annotations

import json
import time

import pytest

from agent_utilities.orchestration import deploy_watch as dw
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
    write_policy,
)

pytestmark = pytest.mark.concept("AU-OS.config.health-gated-deploy-rollback")

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
    yield
    set_fleet_observer(None)
    set_fleet_actuator(None)


def _run(engine, service, job_id):
    return dw.run_deploy_watch(engine, service, job_id, sleep=lambda s: None)


# ---------------------------------------------------------------------------
# Scheduling (durability)
# ---------------------------------------------------------------------------


def test_watch_deploy_queues_durable_task_with_spec(engine):
    job_id = dw.watch_deploy(
        engine, "caddy-mcp", version="1.2.3", window_s=120, source="reconciler"
    )
    assert job_id is not None
    task = engine.submitted[0]
    assert task["task_type"] == "deploy_watch"
    spec = json.loads(engine.nodes[job_id][dw.WATCH_PROP])
    assert spec["service"] == "caddy-mcp"
    assert spec["version"] == "1.2.3"
    assert spec["window_s"] == 120
    # The deadline is recorded at schedule time — a resumed watch keeps it.
    assert spec["deadline_unix"] == pytest.approx(time.time() + 120, abs=10)


def test_watch_without_queue_returns_none():
    class NoQueue:
        pass

    assert dw.watch_deploy(NoQueue(), "svc") is None


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


def test_sustained_green_records_success(engine):
    set_fleet_observer(FakeObserver({"caddy-mcp": obs("caddy-mcp", "up")}))
    job_id = dw.watch_deploy(engine, "caddy-mcp", window_s=0.05)
    result = _run(engine, "caddy-mcp", job_id)
    assert result["outcome"] == dw.OUTCOME_SUCCESS
    watches = engine.by_type("DeployWatch")
    assert len(watches) == 1 and watches[0]["outcome"] == "success"


def test_failure_triggers_policy_gated_rollback_queue(engine, notifier):
    set_fleet_observer(FakeObserver({"caddy-mcp": obs("caddy-mcp", "down")}))
    job_id = dw.watch_deploy(engine, "caddy-mcp", window_s=0.05)
    result = _run(engine, "caddy-mcp", job_id)
    assert result["outcome"] == dw.OUTCOME_FAILED
    # Shipped default policy: rollback_service is approval_required ⇒ queued.
    assert result["on_fail"]["rollback_decision"] == "queue_approval"
    approvals = engine.by_type("ActionApproval")
    assert [a["kind"] for a in approvals] == ["rollback_service"]
    assert any("deploy watch FAILED" in m for m in notifier.messages)


def test_failure_rolls_back_under_permissive_policy(
    engine, notifier, tmp_path, monkeypatch
):
    path = write_policy(tmp_path, PERMISSIVE)
    from agent_utilities.orchestration import action_policy as ap

    original = ap.ActionPolicy
    monkeypatch.setattr(
        ap, "get_action_policy", lambda eng=None: original(engine=eng, policy_path=path)
    )
    actuator = DryRunActuator()
    set_fleet_actuator(actuator)
    set_fleet_observer(FakeObserver({"caddy-mcp": obs("caddy-mcp", "down")}))
    job_id = dw.watch_deploy(engine, "caddy-mcp", version="1.2.3", window_s=0.05)
    result = _run(engine, "caddy-mcp", job_id)
    assert result["on_fail"]["rollback_decision"] == "allow"
    assert result["on_fail"]["rollback_execution"]["ok"] is True
    assert [(r.kind, r.params.get("version")) for r in actuator.applied] == [
        ("rollback_service", "1.2.3")
    ]


def test_unobserved_notifies_but_never_rolls_back(engine, notifier):
    set_fleet_observer(FakeObserver({}))
    actuator = DryRunActuator()
    set_fleet_actuator(actuator)
    job_id = dw.watch_deploy(engine, "ghost-svc", window_s=0.05)
    result = _run(engine, "ghost-svc", job_id)
    assert result["outcome"] == dw.OUTCOME_UNOBSERVED
    assert actuator.applied == []
    assert engine.by_type("ActionApproval") == []
    assert any("NO observations" in m for m in notifier.messages)


def test_custom_on_fail_handler(engine):
    set_fleet_observer(FakeObserver({"caddy-mcp": obs("caddy-mcp", "down")}))
    calls = []
    job_id = dw.watch_deploy(engine, "caddy-mcp", window_s=0.05)
    result = dw.run_deploy_watch(
        engine,
        "caddy-mcp",
        job_id,
        on_fail=lambda eng, spec: calls.append(spec) or {"handled": True},
        sleep=lambda s: None,
    )
    assert result["on_fail"] == {"handled": True}
    assert calls and calls[0]["service"] == "caddy-mcp"


def test_resumed_watch_keeps_original_deadline(engine):
    """A watch requeued after a host crash resumes against the recorded deadline."""
    set_fleet_observer(FakeObserver({"caddy-mcp": obs("caddy-mcp", "up")}))
    job_id = dw.watch_deploy(engine, "caddy-mcp", window_s=600)
    # Simulate the original deadline having already passed before resume.
    spec = json.loads(engine.nodes[job_id][dw.WATCH_PROP])
    spec["deadline_unix"] = time.time() - 1
    engine.nodes[job_id][dw.WATCH_PROP] = json.dumps(spec)
    start = time.time()
    result = _run(engine, "caddy-mcp", job_id)
    assert time.time() - start < 5  # did NOT wait a fresh 600s window
    assert result["outcome"] == dw.OUTCOME_SUCCESS
