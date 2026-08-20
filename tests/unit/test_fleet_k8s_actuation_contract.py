"""NE-179 fixtures for identity-bound Kubernetes actuation.

These are focused source fixtures for the fleet authority boundary.  They use
only an injected resource-registry reader and an argv recorder; no Kubernetes
client, emulator, cluster, or kubectl process is started by this module.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.orchestration.action_policy import ActionRequest
from agent_utilities.orchestration.fleet_actuation import (
    KubernetesActuator,
    execute_action,
)

from .fleet_autonomy_fakes import FakeEngine
from .test_fleet_action_outbox import DurableActionOutbox

BASE_IDENTITY = {
    "cluster": "production",
    "context": "production-admin",
    "namespace": "platform",
    "workload_kind": "Deployment",
    "resource_name": "graph-os-dispatch",
    "uid": "uid-dispatch-1",
    "resource_version": "42",
    "controller_mode": "native",
}


def _request(kind: str = "scale_service", **params: Any) -> ActionRequest:
    identity = dict(BASE_IDENTITY)
    identity.update(params.pop("identity", {}))
    action = {**identity, **params}
    if kind == "scale_service":
        action.setdefault("replicas", 3)
    return ActionRequest(kind=kind, target=identity["resource_name"], params=action)


def _resource(
    *,
    identity: dict[str, Any] | None = None,
    replicas: int = 1,
    context: str | None = None,
    cluster: str | None = None,
) -> dict[str, Any]:
    expected = dict(BASE_IDENTITY)
    expected.update(identity or {})
    return {
        "kind": expected["workload_kind"],
        "metadata": {
            "name": expected["resource_name"],
            "namespace": expected["namespace"],
            "uid": expected["uid"],
            "resourceVersion": expected["resource_version"],
        },
        "spec": {"replicas": replicas},
        **({"context": context} if context is not None else {}),
        **({"cluster": cluster} if cluster is not None else {}),
    }


class RecordingKubernetesActuator(KubernetesActuator):
    def __init__(self, resource: dict[str, Any], **kwargs: Any) -> None:
        self.calls: list[list[str]] = []
        super().__init__(
            kubectl_bin="/usr/bin/kubectl",
            resource_reader=lambda _identity: resource,
            **kwargs,
        )

    def _run(
        self, *args: str, context: str | None = None, namespace: str | None = None
    ):
        self._last_error = ""
        command = []
        if context:
            command.extend(["--context", context])
        if namespace:
            command.extend(["-n", namespace])
        command.extend(args)
        self.calls.append(command)
        return True, "ok"


class TimeoutKubernetesActuator(RecordingKubernetesActuator):
    def _run(
        self, *args: str, context: str | None = None, namespace: str | None = None
    ):
        self._last_error = "timeout"
        command = []
        if context:
            command.extend(["--context", context])
        if namespace:
            command.extend(["-n", namespace])
        command.extend(args)
        self.calls.append(command)
        return False, "kubectl command timed out"


class EvidenceGuard:
    def __init__(self, *, quorum_safe: bool = True) -> None:
        self.quorum_safe = quorum_safe

    def assess(self, identity, current_replicas, desired_replicas, request):
        return {
            "drained": True,
            "stabilized": True,
            "resource_version": identity.resource_version,
            "observed_replicas": current_replicas,
            "remaining_replicas": desired_replicas,
            "quorum_safe": self.quorum_safe,
        }


def test_wrong_context_and_uid_fail_before_any_mutation():
    wrong_context = RecordingKubernetesActuator(
        _resource(context="wrong-context"),
    )
    context_result = wrong_context.apply(_request())
    assert context_result["ok"] is False
    assert "context" in context_result["detail"]
    assert wrong_context.calls == []

    wrong_uid = RecordingKubernetesActuator(
        _resource(identity={"uid": "uid-from-a-recreated-object"}),
    )
    uid_result = wrong_uid.apply(_request())
    assert uid_result["ok"] is False
    assert "uid" in uid_result["detail"]
    assert wrong_uid.calls == []


def test_concurrent_resource_version_change_is_a_fail_closed_precondition():
    actuator = RecordingKubernetesActuator(
        _resource(identity={"resource_version": "43"}),
    )
    result = actuator.apply(_request())
    assert result["ok"] is False
    assert "resourceVersion" in result["detail"]
    assert actuator.calls == []


def test_statefulset_scale_down_requires_drain_stabilization_and_quorum():
    stateful_identity = {
        "workload_kind": "StatefulSet",
        "resource_name": "epistemic-graph-raft",
        "uid": "uid-raft-1",
        "quorum_required": True,
    }
    request = _request(replicas=2, identity=stateful_identity)
    resource = _resource(identity=stateful_identity, replicas=3)

    no_guard = RecordingKubernetesActuator(resource)
    refused = no_guard.apply(request)
    assert refused["ok"] is False
    assert "NE-167" in refused["detail"]
    assert no_guard.calls == []

    unsafe_quorum = RecordingKubernetesActuator(
        resource,
        scale_down_guard=EvidenceGuard(quorum_safe=False),
    )
    quorum_refused = unsafe_quorum.apply(request)
    assert quorum_refused["ok"] is False
    assert "quorum" in quorum_refused["detail"]
    assert unsafe_quorum.calls == []


def test_hpa_and_keda_ownership_refuse_replica_writes():
    for mode in ("external_hpa", "external_keda"):
        reader_calls: list[str] = []

        def reader(_identity, calls=reader_calls):
            calls.append("read")
            return _resource()

        actuator = KubernetesActuator(
            kubectl_bin="/usr/bin/kubectl", resource_reader=reader
        )
        result = actuator.apply(_request(identity={"controller_mode": mode}))
        assert result["ok"] is False
        assert mode in result["detail"]
        assert reader_calls == []


def test_undeclared_controller_or_identity_never_uses_configured_namespace():
    reader_calls: list[str] = []

    def reader(_identity):
        reader_calls.append("read")
        return _resource()

    actuator = KubernetesActuator(
        kubectl_bin="/usr/bin/kubectl",
        namespace="ambient-platform",
        resource_reader=reader,
    )
    params = {"replicas": 2, "workload_kind": "Deployment"}
    result = actuator.apply(
        ActionRequest(kind="scale_service", target="graph-os-dispatch", params=params)
    )
    assert result["ok"] is False
    assert "identity" in result["detail"]
    assert reader_calls == []


def test_timeout_after_scale_command_is_recovery_pending_and_never_auto_rolls_back():
    request = _request(replicas=3)
    actuator = TimeoutKubernetesActuator(_resource(replicas=1))
    engine = FakeEngine()
    outbox = DurableActionOutbox(engine)

    result = execute_action(engine, request, actuator, outbox_store=outbox)

    assert result["ok"] is False
    assert result["state"] == "recovery_pending"
    assert result["outcome_unknown"] is True
    assert result["rollback_required"] is True
    assert len(actuator.calls) == 1
    assert not any("undo" in call for call in actuator.calls)


def test_explicit_identity_bound_rollback_is_the_only_rollback_path():
    actuator = RecordingKubernetesActuator(_resource())
    result = actuator.apply(_request("rollback_service"))
    assert result["ok"] is True
    assert actuator.calls == [
        [
            "--context",
            "production-admin",
            "-n",
            "platform",
            "rollout",
            "undo",
            "deployment/graph-os-dispatch",
        ]
    ]
