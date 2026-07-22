"""Fleet actuators — the injectable hands of the autonomy control plane.

Covers the W3 gap: the reactive autoscaler (``fleet_autoscaler.py``) only had
``DryRunActuator``/``DockerActuator`` — no actuator could scale a Kubernetes
Deployment, so the W3 k8s worker Deployments (``graph-os-dispatch`` /
``graph-os-ingest`` / ``graph-os-mining`` — see ``services/graph-os``
``k8s/w3-worker-topology.yaml``) could never actually be scaled by it.

This file covers ``KubernetesActuator`` in isolation (the right ``kubectl``
call per action kind, guarded availability, unsafe-target rejection,
namespace resolution) plus ``get_fleet_actuator``'s config-driven selection,
then proves the actuator is exercised END TO END through
``FleetAutoscaler`` — asserting the real (mocked) ``kubectl scale`` call is
issued, that it is gated by ``ActionPolicy`` (queued under the default
conservative policy, actuated under a permissive one), and that repeat
scale-service calls are blocked by the durable cooldown ledger — the exact
same seams ``DockerActuator``/``DryRunActuator`` already prove in
``test_fleet_autoscaler.py``. No live cluster is touched anywhere: every
``subprocess.run`` call is mocked.
"""

from __future__ import annotations

import subprocess

import pytest

from agent_utilities.orchestration.action_policy import ActionPolicy, ActionRequest
from agent_utilities.orchestration.fleet_actuation import (
    DryRunActuator,
    KubernetesActuator,
    execute_action,
    get_fleet_actuator,
    set_fleet_actuator,
)
from agent_utilities.orchestration.fleet_autoscaler import FleetAutoscaler

from .fleet_autonomy_fakes import FakeObserver, FakeSignalProvider, obs, write_policy

pytestmark = pytest.mark.concept("AU-OS.config.desired-state-fleet-reconciler")


class _FakeProc:
    def __init__(self, returncode=0, stdout="ok", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _RecordingRun:
    """Drop-in for ``subprocess.run`` that records argv and returns a canned proc."""

    def __init__(self, returncode=0, stdout="ok", stderr=""):
        self.calls: list[list[str]] = []
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        return _FakeProc(self.returncode, self.stdout, self.stderr)


@pytest.fixture(autouse=True)
def _reset_injected_actuator():
    """``get_fleet_actuator`` caches a process-wide override — never leak it."""
    yield
    set_fleet_actuator(None)


# ---------------------------------------------------------------------------
# KubernetesActuator — unit level (mocked subprocess, no live cluster)
# ---------------------------------------------------------------------------


def _actuator(monkeypatch, **kw):
    act = KubernetesActuator(kubectl_bin="/usr/bin/kubectl", namespace="platform", **kw)
    recorder = _RecordingRun()
    monkeypatch.setattr(subprocess, "run", recorder)
    return act, recorder


def test_available_reflects_kubectl_presence(monkeypatch):
    assert KubernetesActuator(kubectl_bin="/usr/bin/kubectl").available is True
    # ``kubectl_bin=None`` falls back to auto-detection via ``shutil.which`` —
    # this dev box has a REAL kubectl on PATH (pointed at a live cluster), so
    # the "absent" case must force detection off rather than rely on the
    # environment happening to lack kubectl.
    monkeypatch.setattr(
        "agent_utilities.orchestration.fleet_actuation.shutil.which", lambda name: None
    )
    assert KubernetesActuator(kubectl_bin=None).available is False


def test_apply_without_kubectl_is_a_safe_no_op(monkeypatch):
    # Force "no kubectl" regardless of what's actually on this box's PATH,
    # AND make any (unexpected) real subprocess.run call fail loudly instead
    # of silently reaching a live cluster.
    monkeypatch.setattr(
        "agent_utilities.orchestration.fleet_actuation.shutil.which", lambda name: None
    )

    def _must_not_be_called(*a, **k):
        raise AssertionError("subprocess.run must not be invoked when kubectl is unavailable")

    monkeypatch.setattr(subprocess, "run", _must_not_be_called)
    act = KubernetesActuator(kubectl_bin=None)
    result = act.apply(
        ActionRequest(kind="scale_service", target="graph-os-dispatch", params={"replicas": 3})
    )
    assert result == {"ok": False, "dry_run": False, "detail": "kubectl CLI not available"}


def test_apply_rejects_unsafe_target(monkeypatch):
    act, recorder = _actuator(monkeypatch)
    result = act.apply(
        ActionRequest(kind="scale_service", target="; rm -rf /", params={"replicas": 3})
    )
    assert result["ok"] is False
    assert "unsafe target" in result["detail"]
    assert recorder.calls == []  # never reached the CLI


def test_scale_service_issues_real_kubectl_scale_call(monkeypatch):
    """ANTI-CHEATING: the actuator must issue a real (mocked) scale call."""
    act, recorder = _actuator(monkeypatch)
    result = act.apply(
        ActionRequest(
            kind="scale_service",
            target="graph-os-dispatch",
            params={"replicas": 4},
            source="autoscaler",
        )
    )
    assert result == {"ok": True, "dry_run": False, "detail": "ok"}
    assert recorder.calls == [
        [
            "/usr/bin/kubectl",
            "-n",
            "platform",
            "scale",
            "deployment/graph-os-dispatch",
            "--replicas=4",
        ]
    ]


def test_stop_service_scales_to_zero(monkeypatch):
    act, recorder = _actuator(monkeypatch)
    act.apply(ActionRequest(kind="stop_service", target="graph-os-mining"))
    assert recorder.calls == [
        [
            "/usr/bin/kubectl",
            "-n",
            "platform",
            "scale",
            "deployment/graph-os-mining",
            "--replicas=0",
        ]
    ]


def test_restart_service_issues_rollout_restart(monkeypatch):
    act, recorder = _actuator(monkeypatch)
    act.apply(ActionRequest(kind="restart_service", target="graph-os-ingest"))
    assert recorder.calls == [
        ["/usr/bin/kubectl", "-n", "platform", "rollout", "restart", "deployment/graph-os-ingest"]
    ]


def test_rollback_service_issues_rollout_undo(monkeypatch):
    act, recorder = _actuator(monkeypatch)
    act.apply(ActionRequest(kind="rollback_service", target="graph-os-ingest"))
    assert recorder.calls == [
        ["/usr/bin/kubectl", "-n", "platform", "rollout", "undo", "deployment/graph-os-ingest"]
    ]


def test_deploy_service_with_image_sets_image(monkeypatch):
    act, recorder = _actuator(monkeypatch)
    act.apply(
        ActionRequest(
            kind="deploy_service",
            target="graph-os-dispatch",
            params={"image": "registry.local/graph-os:1.2.3", "container": "graph-os-dispatch"},
        )
    )
    assert recorder.calls == [
        [
            "/usr/bin/kubectl",
            "-n",
            "platform",
            "set",
            "image",
            "deployment/graph-os-dispatch",
            "graph-os-dispatch=registry.local/graph-os:1.2.3",
        ]
    ]


def test_deploy_service_without_image_falls_back_to_restart(monkeypatch):
    act, recorder = _actuator(monkeypatch)
    act.apply(ActionRequest(kind="deploy_service", target="graph-os-dispatch"))
    assert recorder.calls == [
        [
            "/usr/bin/kubectl",
            "-n",
            "platform",
            "rollout",
            "restart",
            "deployment/graph-os-dispatch",
        ]
    ]


def test_unsupported_kind_is_reported_not_raised(monkeypatch):
    act, recorder = _actuator(monkeypatch)
    result = act.apply(ActionRequest(kind="nonsense_kind", target="graph-os-dispatch"))
    assert result["ok"] is False
    assert "unsupported action kind" in result["detail"]
    assert recorder.calls == []


def test_nonzero_returncode_is_reported_as_failure(monkeypatch):
    act, recorder = _actuator(monkeypatch)
    recorder.returncode = 1
    recorder.stdout = ""
    recorder.stderr = "deployments.apps \"graph-os-dispatch\" not found"
    result = act.apply(
        ActionRequest(kind="scale_service", target="graph-os-dispatch", params={"replicas": 2})
    )
    assert result["ok"] is False
    assert "not found" in result["detail"]


def test_namespace_defaults_to_platform_from_config(monkeypatch):
    act = KubernetesActuator(kubectl_bin="/usr/bin/kubectl")
    assert act.namespace == "platform"


def test_execute_action_stamps_actuator_name(monkeypatch):
    act, _recorder = _actuator(monkeypatch)
    from .fleet_autonomy_fakes import FakeEngine

    engine = FakeEngine()
    result = execute_action(
        engine,
        ActionRequest(kind="scale_service", target="graph-os-dispatch", params={"replicas": 2}),
        act,
    )
    assert result["ok"] is True
    assert result["actuator"] == "k8s"
    records = engine.by_type("ActionExecution")
    assert len(records) == 1 and records[0]["actuator"] == "k8s"


# ---------------------------------------------------------------------------
# get_fleet_actuator — config-driven, default-safe selection
# ---------------------------------------------------------------------------


def test_default_selection_is_dryrun_even_with_kubectl_present(monkeypatch):
    """FLEET_ACTUATOR unset ⇒ dry-run stays the default (default-safe), even
    when kubectl happens to be on PATH."""
    monkeypatch.setattr(
        "agent_utilities.orchestration.fleet_actuation.shutil.which",
        lambda name: "/usr/bin/kubectl" if name == "kubectl" else None,
    )
    assert isinstance(get_fleet_actuator(), DryRunActuator)


def test_k8s_selected_when_explicitly_configured(monkeypatch):
    class _Cfg:
        fleet_actuator = "k8s"
        fleet_actuator_k8s_namespace = "platform"

    monkeypatch.setattr("agent_utilities.core.config.config", _Cfg())
    monkeypatch.setattr(
        "agent_utilities.orchestration.fleet_actuation.shutil.which",
        lambda name: "/usr/bin/kubectl" if name == "kubectl" else None,
    )
    actuator = get_fleet_actuator()
    assert isinstance(actuator, KubernetesActuator)
    assert actuator.name == "k8s"
    assert actuator.namespace == "platform"


def test_kubernetes_alias_also_selects_k8s_actuator(monkeypatch):
    class _Cfg:
        fleet_actuator = "kubernetes"
        fleet_actuator_k8s_namespace = "platform"

    monkeypatch.setattr("agent_utilities.core.config.config", _Cfg())
    monkeypatch.setattr(
        "agent_utilities.orchestration.fleet_actuation.shutil.which",
        lambda name: "/usr/bin/kubectl" if name == "kubectl" else None,
    )
    assert isinstance(get_fleet_actuator(), KubernetesActuator)


def test_k8s_selection_falls_back_to_dryrun_without_kubectl(monkeypatch):
    class _Cfg:
        fleet_actuator = "k8s"
        fleet_actuator_k8s_namespace = "platform"

    monkeypatch.setattr("agent_utilities.core.config.config", _Cfg())
    monkeypatch.setattr(
        "agent_utilities.orchestration.fleet_actuation.shutil.which", lambda name: None
    )
    assert isinstance(get_fleet_actuator(), DryRunActuator)


# ---------------------------------------------------------------------------
# End to end through FleetAutoscaler — cooldown + ActionPolicy gate
# ---------------------------------------------------------------------------

REGISTRY = """
version: 1
services:
  - name: graph-os-dispatch
    scaling:
      min: 1
      max: 5
      signal: queue_depth
      target: 100
      scale_up_step: 2
      scale_down_step: 1
      cooldown_s: 300
"""

PERMISSIVE = (
    "defaults: {tier: auto, rate_limit: {max: 100, window_s: 60},"
    " blast_radius: {max_targets: 100, window_s: 60}}\n"
    "rules:\n  - {kind: '*', target: '*', tier: auto}\n"
)


def _k8s_autoscaler(engine, observations, tmp_path, monkeypatch, recorder, policy_body=None):
    from agent_utilities.orchestration import fleet_autoscaler as fa

    registry = tmp_path / "registry.yml"
    registry.write_text(REGISTRY, encoding="utf-8")
    policy_path = write_policy(tmp_path, policy_body) if policy_body else None

    monkeypatch.setattr(subprocess, "run", recorder)
    actuator = KubernetesActuator(kubectl_bin="/usr/bin/kubectl", namespace="platform")

    scaler = FleetAutoscaler(
        engine,
        observer=FakeObserver(observations),
        actuator=actuator,
        policy=ActionPolicy(engine=engine, policy_path=policy_path),
        signal_provider=FakeSignalProvider(default=450.0),
    )
    original = fa.load_desired_state
    monkeypatch.setattr(
        fa, "load_desired_state", lambda *a, **k: original(registry_path=str(registry))
    )
    return scaler


def test_default_policy_gates_k8s_actuator_no_kubectl_call(tmp_path, monkeypatch):
    """ANTI-CHEATING (gating): under the shipped conservative default policy
    the scale is only QUEUED — the k8s actuator must never be invoked."""
    from .fleet_autonomy_fakes import FakeEngine

    engine = FakeEngine()
    recorder = _RecordingRun()
    scaler = _k8s_autoscaler(
        engine, {"graph-os-dispatch": obs("graph-os-dispatch", "up", replicas=1)}, tmp_path,
        monkeypatch, recorder,
    )
    report = scaler.evaluate()
    assert report["evaluations"][0]["outcome"] == "proposed"
    assert "queue_approval" in report["evaluations"][0]["reason"]
    assert recorder.calls == []  # gated: no kubectl call issued
    assert len(engine.by_type("ActionApproval")) == 1


def test_permissive_policy_lets_k8s_actuator_issue_real_scale_call(tmp_path, monkeypatch):
    """ANTI-CHEATING (real call): under a permissive policy the loop must
    actually reach ``kubectl scale`` with the right Deployment + replica
    count — not a no-op / dry-run."""
    from .fleet_autonomy_fakes import FakeEngine

    engine = FakeEngine()
    recorder = _RecordingRun()
    scaler = _k8s_autoscaler(
        engine, {"graph-os-dispatch": obs("graph-os-dispatch", "up", replicas=1)}, tmp_path,
        monkeypatch, recorder, policy_body=PERMISSIVE,
    )
    report = scaler.evaluate()
    assert report["scaled"] == 1
    assert report["actuator"] == "k8s"
    assert recorder.calls == [
        [
            "/usr/bin/kubectl",
            "-n",
            "platform",
            "scale",
            "deployment/graph-os-dispatch",
            "--replicas=3",  # up-step capped from raw 5
        ]
    ]
    executions = engine.by_type("ActionExecution")
    assert len(executions) == 1 and executions[0]["ok"] is True
    assert executions[0]["actuator"] == "k8s"


def test_cooldown_blocks_repeat_k8s_scale_call(tmp_path, monkeypatch):
    """ANTI-CHEATING (cooldown): a second tick inside the cooldown window must
    NOT issue a second kubectl call."""
    from .fleet_autonomy_fakes import FakeEngine

    engine = FakeEngine()
    recorder = _RecordingRun()
    scaler = _k8s_autoscaler(
        engine, {"graph-os-dispatch": obs("graph-os-dispatch", "up", replicas=1)}, tmp_path,
        monkeypatch, recorder, policy_body=PERMISSIVE,
    )
    first = scaler.evaluate()
    assert first["scaled"] == 1
    assert len(recorder.calls) == 1

    second = scaler.evaluate()
    assert second["actions"] == 0
    assert "cooldown" in second["evaluations"][0]["reason"]
    assert len(recorder.calls) == 1  # no second kubectl call inside cooldown
