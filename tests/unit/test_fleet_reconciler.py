"""Desired-state fleet reconciler (CONCEPT:OS-5.25).

Covers: registry + override desired-state parsing, conservative diffing
(down ⇒ restart, replica mismatch ⇒ scale, undesired-up ⇒ stop, unobserved ⇒
skip), the ActionPolicy gate in the convergence path (queue under the shipped
default, actuate + health-watch under a permissive policy), the per-tick
storm guard, the approved-action drain, and ReconcileReport persistence.

@pytest.mark.concept("OS-5.25")
"""

from __future__ import annotations

import pytest

from agent_utilities.orchestration.action_policy import ActionPolicy
from agent_utilities.orchestration.fleet_actuation import DryRunActuator
from agent_utilities.orchestration.fleet_reconciler import (
    FleetReconciler,
    load_desired_state,
    resolve_registry_path,
)

from .fleet_autonomy_fakes import FakeEngine, FakeObserver, obs, write_policy

pytestmark = pytest.mark.concept("OS-5.25")


REGISTRY = """
version: 1
services:
  - name: caddy-mcp
    host_port: 8206
  - name: portainer-mcp
    replicas: 2
"""

OVERRIDE = """
services:
  - name: caddy-mcp
    desired: stopped
  - name: extra-svc
    replicas: 3
"""

PERMISSIVE = (
    "defaults: {tier: auto, rate_limit: {max: 100, window_s: 60},"
    " blast_radius: {max_targets: 100, window_s: 60}}\n"
    "rules:\n  - {kind: '*', target: '*', tier: auto}\n"
)


@pytest.fixture
def engine():
    return FakeEngine()


def _reconciler(engine, observations, tmp_path, policy_body=None, max_actions=5):
    registry = tmp_path / "registry.yml"
    registry.write_text(REGISTRY, encoding="utf-8")
    policy_path = write_policy(tmp_path, policy_body) if policy_body else None
    rec = FleetReconciler(
        engine,
        observer=FakeObserver(observations),
        actuator=DryRunActuator(),
        policy=ActionPolicy(engine=engine, policy_path=policy_path),
        max_actions=max_actions,
    )
    # Pin desired state to the test registry (not the repo's 52-service one).
    import agent_utilities.orchestration.fleet_reconciler as fr

    original = fr.load_desired_state
    fr_load = lambda *a, **k: original(registry_path=str(registry))  # noqa: E731
    return rec, fr, fr_load


@pytest.fixture
def patch_desired(monkeypatch, tmp_path):
    def _patch(rec_tuple):
        rec, fr, fr_load = rec_tuple
        monkeypatch.setattr(fr, "load_desired_state", fr_load)
        return rec

    return _patch


# ---------------------------------------------------------------------------
# Desired-state parsing
# ---------------------------------------------------------------------------


def test_registry_parse(tmp_path):
    registry = tmp_path / "registry.yml"
    registry.write_text(REGISTRY, encoding="utf-8")
    desired = load_desired_state(registry_path=str(registry))
    assert set(desired) == {"caddy-mcp", "portainer-mcp"}
    assert desired["caddy-mcp"].replicas == 1  # default
    assert desired["portainer-mcp"].replicas == 2
    assert desired["caddy-mcp"].desired == "running"


def test_override_layering(tmp_path):
    registry = tmp_path / "registry.yml"
    registry.write_text(REGISTRY, encoding="utf-8")
    override = tmp_path / "state.yml"
    override.write_text(OVERRIDE, encoding="utf-8")
    desired = load_desired_state(
        registry_path=str(registry), override_path=str(override)
    )
    assert desired["caddy-mcp"].desired == "stopped"
    assert desired["extra-svc"].replicas == 3  # override can add services


def test_shipped_registry_resolves_and_parses():
    path = resolve_registry_path()
    assert path is not None and path.is_file()
    desired = load_desired_state(registry_path=str(path))
    assert len(desired) > 40  # the ~52-service fleet registry


# ---------------------------------------------------------------------------
# Diffing (conservative)
# ---------------------------------------------------------------------------


def test_down_service_proposes_restart(engine, tmp_path, patch_desired):
    rec = patch_desired(
        _reconciler(engine, {"caddy-mcp": obs("caddy-mcp", "down")}, tmp_path)
    )
    proposals = rec.diff()
    assert [(p.kind, p.target) for p in proposals] == [("restart_service", "caddy-mcp")]


def test_unobserved_service_is_skipped(engine, tmp_path, patch_desired):
    rec = patch_desired(_reconciler(engine, {}, tmp_path))
    assert rec.diff() == []  # zero evidence ⇒ zero action


def test_replica_mismatch_proposes_scale(engine, tmp_path, patch_desired):
    rec = patch_desired(
        _reconciler(
            engine,
            {"portainer-mcp": obs("portainer-mcp", "up", replicas=1)},
            tmp_path,
        )
    )
    proposals = rec.diff()
    assert [(p.kind, p.target, p.params.get("replicas")) for p in proposals] == [
        ("scale_service", "portainer-mcp", 2)
    ]


def test_healthy_fleet_diffs_nothing(engine, tmp_path, patch_desired):
    rec = patch_desired(
        _reconciler(
            engine,
            {
                "caddy-mcp": obs("caddy-mcp", "up", replicas=1),
                "portainer-mcp": obs("portainer-mcp", "up", replicas=2),
            },
            tmp_path,
        )
    )
    assert rec.diff() == []


# ---------------------------------------------------------------------------
# Convergence through the policy gate
# ---------------------------------------------------------------------------


def test_default_policy_queues_convergence_actions(engine, tmp_path, patch_desired):
    rec = patch_desired(
        _reconciler(engine, {"caddy-mcp": obs("caddy-mcp", "down")}, tmp_path)
    )
    report = rec.reconcile()
    assert report["divergences"] == 1
    assert report["actions"][0]["decision"] == "queue_approval"
    assert rec.actuator.applied == []  # nothing actuated without approval
    assert len(engine.by_type("ActionApproval")) == 1
    assert len(engine.by_type("ReconcileReport")) == 1


def test_permissive_policy_actuates_and_schedules_watch(
    engine, tmp_path, patch_desired
):
    rec = patch_desired(
        _reconciler(
            engine,
            {"caddy-mcp": obs("caddy-mcp", "down")},
            tmp_path,
            policy_body=PERMISSIVE,
        )
    )
    report = rec.reconcile()
    action = report["actions"][0]
    assert action["decision"] == "allow"
    assert action["execution"]["ok"] is True
    assert [r.target for r in rec.actuator.applied] == ["caddy-mcp"]
    # Restart success scheduled an OS-5.27 health watch on the durable queue.
    assert [t["task_type"] for t in engine.submitted] == ["deploy_watch"]
    # And the execution is durably recorded.
    assert len(engine.by_type("ActionExecution")) == 1


def test_storm_guard_defers_beyond_budget(engine, tmp_path, patch_desired):
    rec = patch_desired(
        _reconciler(
            engine,
            {
                "caddy-mcp": obs("caddy-mcp", "down"),
                "portainer-mcp": obs("portainer-mcp", "down"),
            },
            tmp_path,
            policy_body=PERMISSIVE,
            max_actions=1,
        )
    )
    report = rec.reconcile()
    assert report["divergences"] == 2
    assert report["processed"] == 1
    assert len(report["deferred"]) == 1


# ---------------------------------------------------------------------------
# Approved-action drain (closing the human-in-the-loop circle)
# ---------------------------------------------------------------------------


def test_granted_approval_is_executed_and_stamped(engine, tmp_path, patch_desired):
    rec = patch_desired(_reconciler(engine, {}, tmp_path))
    engine.add_node(
        "action_approval:xyz",
        "ActionApproval",
        properties={
            "kind": "restart_service",
            "target": "caddy-mcp",
            "params_json": "{}",
            "status": "approved",
            "source": "reconciler",
        },
    )
    report = rec.reconcile()
    drained = report["approved_drained"]
    assert len(drained) == 1
    assert drained[0]["status"] == "executed"
    assert engine.nodes["action_approval:xyz"]["status"] == "executed"
    assert [r.target for r in rec.actuator.applied] == ["caddy-mcp"]
    # Watched kind ⇒ health watch scheduled after the approved execution too.
    assert [t["task_type"] for t in engine.submitted] == ["deploy_watch"]


def test_denied_approvals_are_not_drained(engine, tmp_path, patch_desired):
    rec = patch_desired(_reconciler(engine, {}, tmp_path))
    engine.add_node(
        "action_approval:no",
        "ActionApproval",
        properties={"kind": "restart_service", "target": "x", "status": "denied"},
    )
    report = rec.reconcile()
    assert report["approved_drained"] == []
    assert rec.actuator.applied == []


# ---------------------------------------------------------------------------
# Daemon wiring (leader-only maintenance tick, opt-in flag)
# ---------------------------------------------------------------------------


def _enabled_maintenance_names() -> set[str]:
    """Names of ENABLED maintenance :Schedule nodes for the current config
    (the unified-scheduler analog of a tick being registered; CONCEPT:OS-5.44)."""
    from agent_utilities.core import schedule_engine as _se
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin

    inst = TaskManagerMixin.__new__(TaskManagerMixin)  # type: ignore[type-abstract]
    inst.backend = EpistemicGraphBackend()
    inst._register_maintenance_schedules()
    return {s.name for s in _se._load_all(inst) if s.enabled}


def test_reconciler_tick_registration_is_flag_gated(monkeypatch):
    from agent_utilities.core.config import config

    monkeypatch.setattr(config, "fleet_reconciler", True)
    assert "fleet_reconciler" in _enabled_maintenance_names()

    monkeypatch.setattr(config, "fleet_reconciler", False)
    assert "fleet_reconciler" not in _enabled_maintenance_names()
