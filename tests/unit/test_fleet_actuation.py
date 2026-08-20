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
from agent_utilities.orchestration.fleet_reconciler import (
    FleetReconciler,
    load_desired_state,
)

from .fleet_autonomy_fakes import (
    FakeObserver,
    FakeSignalProvider,
    healthy_fleet_evidence,
    obs,
    write_policy,
)
from .test_fleet_action_outbox import DurableActionOutbox
from .test_fleet_scale_authority import DurableIntentCAS

pytestmark = pytest.mark.concept("AU-OS.config.desired-state-fleet-reconciler")

# NE-179 identity-bound Kubernetes actuation (checkpoint au-scale-authority-
# conflict, bundled with the NE-226 ScaleIntent redesign in the same commit):
# every KubernetesActuator.apply() call now requires a full registry
# identity (cluster/context/namespace/workload_kind/uid/resource_version/
# controller_mode) bound into the ActionRequest params, and reads the object
# immediately pre-mutation (real kubectl, or an injected ``resource_reader``
# for tests — see test_fleet_k8s_actuation_contract.py, which this mirrors).
_K8S_IDENTITY = {
    "cluster": "production",
    "context": "production-admin",
    "namespace": "platform",
    "workload_kind": "Deployment",
    "uid": "uid-fixture-1",
    "resource_version": "1",
    "controller_mode": "native",
}


def _identity_params(**overrides: object) -> dict[str, object]:
    params = dict(_K8S_IDENTITY)
    params.update(overrides)
    return params


def _matching_reader(current_replicas: int = 1):
    """A ``resource_reader`` that always answers with a record matching
    whatever identity was resolved from the request — keeps these unit tests
    focused on the exact mutating ``kubectl`` argv instead of also having to
    fake the pre-mutation ``kubectl get`` / ``config view`` reads."""

    def _read(identity):
        return {
            "kind": identity.workload_kind,
            "metadata": {
                "name": identity.name,
                "namespace": identity.namespace,
                "uid": identity.uid,
                "resourceVersion": identity.resource_version,
            },
            "spec": {"replicas": current_replicas},
        }

    return _read


class _AlwaysDrainedGuard:
    """Deterministic NE-167 drain/stabilization evidence for scale-DOWN unit
    tests here — proving the exact kubectl argv, not the guard mechanics
    themselves (covered separately in test_fleet_k8s_actuation_contract.py)."""

    def assess(self, identity, current_replicas, desired_replicas, request):
        return {
            "drained": True,
            "stabilized": True,
            "resource_version": identity.resource_version,
            "observed_replicas": current_replicas,
            "remaining_replicas": desired_replicas,
            "quorum_safe": True,
        }


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


def _actuator(monkeypatch, *, current_replicas=1, **kw):
    kw.setdefault("resource_reader", _matching_reader(current_replicas))
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
        raise AssertionError(
            "subprocess.run must not be invoked when kubectl is unavailable"
        )

    monkeypatch.setattr(subprocess, "run", _must_not_be_called)
    act = KubernetesActuator(kubectl_bin=None)
    result = act.apply(
        ActionRequest(
            kind="scale_service", target="graph-os-dispatch", params={"replicas": 3}
        )
    )
    assert result == {
        "ok": False,
        "dry_run": False,
        "detail": "kubectl CLI not available",
    }


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
    act, recorder = _actuator(monkeypatch, current_replicas=1)
    result = act.apply(
        ActionRequest(
            kind="scale_service",
            target="graph-os-dispatch",
            params=_identity_params(replicas=4),
            source="autoscaler",
        )
    )
    assert result == {"ok": True, "dry_run": False, "detail": "ok"}
    assert recorder.calls == [
        [
            "/usr/bin/kubectl",
            "--context",
            "production-admin",
            "-n",
            "platform",
            "scale",
            "deployment/graph-os-dispatch",
            "--replicas=4",
            "--resource-version=1",
        ]
    ]


def test_stop_service_scales_to_zero(monkeypatch):
    # stop_service scales DOWN to zero, so (unlike the other unit-level
    # kinds here) it needs NE-167 drain/stabilization evidence too.
    act, recorder = _actuator(
        monkeypatch,
        current_replicas=3,
        scale_down_guard=_AlwaysDrainedGuard(),
    )
    act.apply(
        ActionRequest(
            kind="stop_service",
            target="graph-os-mining",
            params=_identity_params(),
        )
    )
    assert recorder.calls == [
        [
            "/usr/bin/kubectl",
            "--context",
            "production-admin",
            "-n",
            "platform",
            "scale",
            "deployment/graph-os-mining",
            "--replicas=0",
            "--resource-version=1",
        ]
    ]


def test_restart_service_issues_rollout_restart(monkeypatch):
    act, recorder = _actuator(monkeypatch)
    act.apply(
        ActionRequest(
            kind="restart_service",
            target="graph-os-ingest",
            params=_identity_params(),
        )
    )
    assert recorder.calls == [
        [
            "/usr/bin/kubectl",
            "--context",
            "production-admin",
            "-n",
            "platform",
            "rollout",
            "restart",
            "deployment/graph-os-ingest",
        ]
    ]


def test_rollback_service_issues_rollout_undo(monkeypatch):
    act, recorder = _actuator(monkeypatch)
    act.apply(
        ActionRequest(
            kind="rollback_service",
            target="graph-os-ingest",
            params=_identity_params(),
        )
    )
    assert recorder.calls == [
        [
            "/usr/bin/kubectl",
            "--context",
            "production-admin",
            "-n",
            "platform",
            "rollout",
            "undo",
            "deployment/graph-os-ingest",
        ]
    ]


def test_deploy_service_with_image_sets_image(monkeypatch):
    act, recorder = _actuator(monkeypatch)
    act.apply(
        ActionRequest(
            kind="deploy_service",
            target="graph-os-dispatch",
            params=_identity_params(
                image="registry.local/graph-os:1.2.3",
                container="graph-os-dispatch",
            ),
        )
    )
    assert recorder.calls == [
        [
            "/usr/bin/kubectl",
            "--context",
            "production-admin",
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
    act.apply(
        ActionRequest(
            kind="deploy_service",
            target="graph-os-dispatch",
            params=_identity_params(),
        )
    )
    assert recorder.calls == [
        [
            "/usr/bin/kubectl",
            "--context",
            "production-admin",
            "-n",
            "platform",
            "rollout",
            "restart",
            "deployment/graph-os-dispatch",
        ]
    ]


def test_unsupported_kind_is_reported_not_raised(monkeypatch):
    act, recorder = _actuator(monkeypatch)
    result = act.apply(
        ActionRequest(
            kind="nonsense_kind",
            target="graph-os-dispatch",
            params=_identity_params(),
        )
    )
    assert result["ok"] is False
    assert "unsupported action kind" in result["detail"]
    assert recorder.calls == []


def test_nonzero_returncode_is_reported_as_failure(monkeypatch):
    act, recorder = _actuator(monkeypatch, current_replicas=1)
    recorder.returncode = 1
    recorder.stdout = ""
    recorder.stderr = 'deployments.apps "graph-os-dispatch" not found'
    result = act.apply(
        ActionRequest(
            kind="scale_service",
            target="graph-os-dispatch",
            params=_identity_params(replicas=2),
        )
    )
    assert result["ok"] is False
    assert "not found" in result["detail"]


def test_namespace_defaults_to_platform_from_config(monkeypatch):
    act = KubernetesActuator(kubectl_bin="/usr/bin/kubectl")
    assert act.namespace == "platform"


def test_execute_action_stamps_actuator_name(monkeypatch):
    act, _recorder = _actuator(monkeypatch, current_replicas=1)
    from .fleet_autonomy_fakes import FakeEngine

    engine = FakeEngine()
    # A non-dry-run actuator now requires the durable action-outbox
    # pre-side-effect fence before execute_action will call it at all.
    engine.action_outbox_store = DurableActionOutbox(engine)
    result = execute_action(
        engine,
        ActionRequest(
            kind="scale_service",
            target="graph-os-dispatch",
            params=_identity_params(replicas=2),
        ),
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
    kubernetes:
      cluster: production
      context: production-admin
      namespace: platform
      workload_kind: Deployment
      name: graph-os-dispatch
      uid: uid-dispatch-1
      resource_version: "1"
      controller_mode: native
"""

PERMISSIVE = (
    "defaults: {tier: auto, rate_limit: {max: 100, window_s: 60},"
    " blast_radius: {max_targets: 100, window_s: 60}}\n"
    "rules:\n  - {kind: '*', target: '*', tier: auto}\n"
)


def _k8s_autoscaler_and_reconciler(
    engine, observations, tmp_path, monkeypatch, recorder, policy_body=None
):
    """NE-226: the autoscaler alone can no longer prove a real kubectl call —
    it only proposes/accepts a durable ScaleIntent. The fleet reconciler is
    now the sole native replica actuator, so an end-to-end "real call" proof
    needs BOTH, sharing one engine/observer/policy/intent-store/registry —
    mirroring test_fleet_scale_authority.py's shared-store pattern, plus the
    NE-179 identity-bound KubernetesActuator wiring from
    test_fleet_k8s_actuation_contract.py.
    """
    from agent_utilities.orchestration import fleet_autoscaler as fa
    from agent_utilities.orchestration import fleet_reconciler as fr

    registry = tmp_path / "registry.yml"
    registry.write_text(REGISTRY, encoding="utf-8")
    policy_path = write_policy(tmp_path, policy_body) if policy_body else None
    policy = ActionPolicy(engine=engine, policy_path=policy_path)
    observer = FakeObserver(observations)
    store = DurableIntentCAS()
    # A non-dry-run actuator requires the durable action-outbox
    # pre-side-effect fence before execute_action will call it at all.
    engine.action_outbox_store = DurableActionOutbox(engine)

    monkeypatch.setattr(subprocess, "run", recorder)
    actuator = KubernetesActuator(
        kubectl_bin="/usr/bin/kubectl",
        namespace="platform",
        # Bypass the pre-mutation kubectl "get"/"config view" reads (not
        # what these tests are about) so `recorder.calls` captures only the
        # actual mutating command, matching the pre-NE-179 shape of these
        # assertions.
        resource_reader=_matching_reader(current_replicas=1),
    )

    scaler = FleetAutoscaler(
        engine,
        observer=observer,
        actuator=actuator,
        policy=policy,
        signal_provider=FakeSignalProvider(default=450.0),
        health_provider=healthy_fleet_evidence,
        intent_store=store,
    )
    reconciler = FleetReconciler(
        engine,
        observer=observer,
        actuator=actuator,
        policy=policy,
        health_provider=healthy_fleet_evidence,
        intent_store=store,
    )

    def _loader(*a, **k):
        return load_desired_state(registry_path=str(registry))

    monkeypatch.setattr(fa, "load_desired_state", _loader)
    monkeypatch.setattr(fr, "load_desired_state", _loader)
    return scaler, reconciler


def test_default_policy_gates_k8s_actuator_no_kubectl_call(tmp_path, monkeypatch):
    """ANTI-CHEATING (gating): under the shipped conservative default policy
    the scale is only PROPOSED (queued for approval) — no accepted intent is
    ever created, so the reconciler has nothing to actuate. The k8s actuator
    must never be invoked, at EITHER layer — this is the security guarantee
    that survives the NE-226 redesign (autoscaler never actuates directly;
    the reconciler only actuates an accepted intent)."""
    from .fleet_autonomy_fakes import FakeEngine

    engine = FakeEngine()
    recorder = _RecordingRun()
    scaler, reconciler = _k8s_autoscaler_and_reconciler(
        engine,
        {"graph-os-dispatch": obs("graph-os-dispatch", "up", replicas=1)},
        tmp_path,
        monkeypatch,
        recorder,
    )
    report = scaler.evaluate()
    assert report["evaluations"][0]["outcome"] == "intent_proposed"
    assert "queue_approval" in report["evaluations"][0]["reason"]
    assert recorder.calls == []  # gated: no kubectl call issued
    assert len(engine.by_type("ActionApproval")) == 1

    converged = reconciler.reconcile()
    assert converged["actions"] == []  # no accepted intent ⇒ nothing to converge
    assert recorder.calls == []  # still gated: the reconciler never reaches kubectl


def test_permissive_policy_lets_k8s_actuator_issue_real_scale_call(
    tmp_path, monkeypatch
):
    """ANTI-CHEATING (real call): under a permissive policy the autoscaler's
    accepted intent must be actuated by the RECONCILER — the sole native
    replica writer under NE-226 — reaching a real (mocked) ``kubectl scale``
    with the right Deployment + replica count, not a no-op / dry-run."""
    from .fleet_autonomy_fakes import FakeEngine

    engine = FakeEngine()
    recorder = _RecordingRun()
    scaler, reconciler = _k8s_autoscaler_and_reconciler(
        engine,
        {"graph-os-dispatch": obs("graph-os-dispatch", "up", replicas=1)},
        tmp_path,
        monkeypatch,
        recorder,
        policy_body=PERMISSIVE,
    )
    report = scaler.evaluate()
    assert report["intents_accepted"] == 1
    assert report["actuator"] == "k8s"
    assert recorder.calls == []  # the autoscaler itself never touches kubectl

    converged = reconciler.reconcile()
    action = converged["actions"][0]
    assert action["decision"] == "accepted_intent"
    assert action["state"] == "executed"
    assert recorder.calls == [
        [
            "/usr/bin/kubectl",
            "--context",
            "production-admin",
            "-n",
            "platform",
            "scale",
            "deployment/graph-os-dispatch",
            "--replicas=3",  # up-step capped from raw 5
            "--resource-version=1",
        ]
    ]
    executions = engine.by_type("ActionExecution")
    assert len(executions) == 1 and executions[0]["ok"] is True
    assert executions[0]["actuator"] == "k8s"


def test_cooldown_blocks_repeat_k8s_scale_call(tmp_path, monkeypatch):
    """ANTI-CHEATING (cooldown): once the reconciler has actuated a real
    scale, a second tick inside the cooldown window must NOT issue a second
    kubectl call — through either the autoscaler (re-proposing) or a repeat
    reconciler pass."""
    from .fleet_autonomy_fakes import FakeEngine

    engine = FakeEngine()
    recorder = _RecordingRun()
    scaler, reconciler = _k8s_autoscaler_and_reconciler(
        engine,
        {"graph-os-dispatch": obs("graph-os-dispatch", "up", replicas=1)},
        tmp_path,
        monkeypatch,
        recorder,
        policy_body=PERMISSIVE,
    )
    first = scaler.evaluate()
    assert first["intents_accepted"] == 1
    reconciler.reconcile()
    assert len(recorder.calls) == 1

    # Advance the intent to its second stable observation (verified) exactly
    # as a real reconcile loop would once the actuation becomes visible to
    # the observer — only then does the autoscaler's own cooldown/flap-guard
    # apply again (an executed/observed intent still "belongs" to the
    # reconciler, per fleet_autoscaler._evaluate_service's intent gate).
    scaler.observer.observations["graph-os-dispatch"] = obs(
        "graph-os-dispatch", "up", replicas=3
    )
    reconciler.reconcile()  # executed -> observed
    reconciler.reconcile()  # observed -> verified
    assert len(recorder.calls) == 1  # still just the one real scale call

    second = scaler.evaluate()
    assert second["actions"] == 0
    assert "cooldown" in second["evaluations"][0]["reason"]
    reconciler.reconcile()
    assert len(recorder.calls) == 1  # cooldown blocked a second kubectl call too
