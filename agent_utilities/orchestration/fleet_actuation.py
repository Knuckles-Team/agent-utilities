#!/usr/bin/python
from __future__ import annotations

"""Fleet actuators — the injectable hands of the autonomy control plane.

CONCEPT:AU-OS.config.desired-state-fleet-reconciler — Desired-state fleet reconciler (actuation seam).

agent-utilities deliberately takes NO hard dependency on the ecosystem's MCP
actuators (portainer-mcp, container-manager, …). Actuation is a protocol:

* :class:`FleetActuator` — ``apply(ActionRequest) -> dict`` for one concrete
  action (restart/scale/deploy/rollback/stop).
* :class:`DryRunActuator` — the DEFAULT. Mutates nothing; every intended
  action is recorded by :func:`execute_action` as an ``ActionExecution`` KG
  node and announced through the notification seam. This is what makes the
  reconciler safe to enable before any real actuator exists.
* :class:`DockerActuator` — reference implementation over the local docker
  CLI/socket (guarded: inert when ``docker`` is absent). Standalone
  containers and swarm services both supported, argv-only (no shell).
* :class:`KubernetesActuator` — reference implementation over the ``kubectl``
  CLI (guarded: inert when ``kubectl`` is absent), the same argv-only
  no-shell shape as :class:`DockerActuator`. Targets are bound to a declared
  resource-registry identity (cluster/context/namespace/kind/UID/
  resourceVersion) before a Deployment or StatefulSet mutation; a configured
  namespace alone is never an authority.

A deployment wires real Portainer/Swarm actuation by registering its own
implementation::

    from agent_utilities.orchestration.fleet_actuation import set_fleet_actuator
    set_fleet_actuator(MyPortainerActuator())

Every execution — real or dry-run — flows through :func:`execute_action`.
Real actions first require the engine-native :class:`ActionOutboxStore` fence;
the compatibility ``ActionExecution`` projection is stamped with
``state=executed`` for a successful real call, ``state=simulated`` for
dry-run, ``state=failed`` for a confirmed unsuccessful call, or
``state=recovery_pending`` when the external outcome or durable acknowledgement
is unknown. This makes "what did autonomy do and when" one graph query away
without treating simulation as mutation.
"""

import hashlib
import json
import logging
import re
import shutil
import subprocess  # nosec B404 — argv-only docker CLI calls, no shell
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agent_utilities.orchestration.action_policy import ActionRequest

logger = logging.getLogger(__name__)

_OUTBOX_PREPARED = "prepared"
_OUTBOX_EXECUTING = "executing"
_OUTBOX_SIMULATED = "simulated"
_OUTBOX_EXECUTED = "executed"
_OUTBOX_OBSERVED = "observed"
_OUTBOX_VERIFIED = "verified"
_OUTBOX_FAILED = "failed"
_OUTBOX_RECOVERY_PENDING = "recovery_pending"
_OUTBOX_TERMINAL = frozenset(
    {
        _OUTBOX_SIMULATED,
        _OUTBOX_EXECUTED,
        _OUTBOX_OBSERVED,
        _OUTBOX_VERIFIED,
        _OUTBOX_FAILED,
    }
)
_SAFE_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")

# Service/target names must be plain identifiers before they reach a CLI.
_SAFE_TARGET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Kubernetes is deliberately a small, typed adapter here rather than a
# second control plane.  A request must carry the identity observed by the
# resource registry immediately before the mutation.  In particular, a
# namespace alone is not an authority: a stale context, UID, or
# resourceVersion must fail closed before kubectl is called.
_SAFE_K8S_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}$")
_K8S_WORKLOAD_KINDS = {
    "deployment": "Deployment",
    "deployments": "Deployment",
    "statefulset": "StatefulSet",
    "statefulsets": "StatefulSet",
}
_K8S_CONTROLLER_MODES = frozenset({"native", "external_hpa", "external_keda"})
_K8S_CONTROLLER_ALIASES = {
    "native": "native",
    "hpa": "external_hpa",
    "external_hpa": "external_hpa",
    "keda": "external_keda",
    "external_keda": "external_keda",
}


@dataclass(frozen=True)
class KubernetesResourceIdentity:
    """Immutable registry identity bound to one Kubernetes mutation.

    The identity is intentionally not inferred from a target name or the
    actuator's configured namespace.  ``uid`` protects against delete/recreate
    races and ``resource_version`` is the observed CAS precondition.  The
    controller mode is part of the binding so an AU-native writer cannot race
    an HPA/KEDA owner.
    """

    cluster: str
    context: str
    namespace: str
    workload_kind: str
    name: str
    uid: str
    resource_version: str
    controller_mode: str
    quorum_required: bool = False

    @classmethod
    def from_request(
        cls, request: ActionRequest
    ) -> tuple[KubernetesResourceIdentity | None, str]:
        params = dict(request.params or {})
        nested = params.get("kubernetes") or params.get("k8s_resource")
        if isinstance(nested, dict):
            # Explicit request fields may carry action-specific values, but
            # registry identity fields remain required and are never inferred.
            merged = dict(nested)
            merged.update(params)
            params = merged

        def value(*names: str) -> str:
            for name in names:
                candidate = params.get(name)
                if candidate is not None and str(candidate).strip():
                    return str(candidate).strip()
            return ""

        cluster = value("cluster", "kube_cluster")
        context = value("context", "kube_context")
        namespace = value("namespace", "kube_namespace")
        kind = value("workload_kind", "resource_kind", "kind")
        name = value("resource_name", "workload_name", "name") or request.target
        uid = value("uid", "resource_uid")
        resource_version = value("resource_version", "resourceVersion")
        mode_raw = value("controller_mode", "controller")
        mode = _K8S_CONTROLLER_ALIASES.get(mode_raw.lower())

        required = {
            "cluster": cluster,
            "context": context,
            "namespace": namespace,
            "workload_kind": kind,
            "name": name,
            "uid": uid,
            "resource_version": resource_version,
            "controller_mode": mode or mode_raw,
        }
        missing = [field for field, candidate in required.items() if not candidate]
        if missing:
            return None, "kubernetes resource identity is incomplete: " + ", ".join(
                missing
            )
        canonical_kind = _K8S_WORKLOAD_KINDS.get(kind.lower())
        if canonical_kind is None:
            return None, f"unsupported Kubernetes workload kind {kind!r}"
        if mode is None or mode not in _K8S_CONTROLLER_MODES:
            return None, f"unsupported Kubernetes controller mode {mode_raw!r}"
        if name != request.target:
            return None, "kubernetes resource name does not match action target"
        for field, candidate in required.items():
            if field == "controller_mode":
                continue
            if not _SAFE_K8S_VALUE.fullmatch(str(candidate)):
                return None, f"unsafe Kubernetes identity field {field!r}"

        quorum_raw = params.get("quorum_required", params.get("quorum", False))
        if isinstance(quorum_raw, bool):
            quorum_required = quorum_raw
        elif str(quorum_raw).strip().lower() in {"1", "true", "yes"}:
            quorum_required = True
        elif str(quorum_raw).strip().lower() in {"0", "false", "no", ""}:
            quorum_required = False
        else:
            return None, "quorum_required must be a boolean"
        return (
            cls(
                cluster=cluster,
                context=context,
                namespace=namespace,
                workload_kind=canonical_kind,
                name=name,
                uid=uid,
                resource_version=resource_version,
                controller_mode=mode,
                quorum_required=quorum_required,
            ),
            "",
        )


class KubernetesScaleDownGuard(Protocol):
    """NE-167 drain/stabilization evidence seam for replica reduction."""

    def assess(
        self,
        identity: KubernetesResourceIdentity,
        current_replicas: int,
        desired_replicas: int,
        request: ActionRequest,
    ) -> dict[str, Any]:
        """Return bounded drain/stabilization evidence; never mutate itself."""
        ...  # ABSTRACT-OK


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@runtime_checkable
class FleetActuator(Protocol):
    """Anything that can apply one operational action to the fleet."""

    name: str

    def apply(self, request: ActionRequest) -> dict[str, Any]:
        """Apply ``request``; return ``{ok, detail, dry_run, ...}``. Never raises."""
        ...  # ABSTRACT-OK


class ActionOutboxStore(Protocol):
    """Durable pre-side-effect intent and completion seam.

    ``prepare`` MUST commit the idempotency key and action intent before the
    actuator is called. ``complete`` MUST durably record the outcome and, when
    an approval id is supplied, close that approval in the same authoritative
    transaction. Implementations must return the prior record for a replay;
    they must never implement this contract with an in-memory flag or a
    read-then-add race. ``recovery_pending`` is non-terminal: it means the
    external outcome is ambiguous and may only be advanced by positive
    observation, never by issuing the actuator call again.
    """

    def prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        """Persist a pending action or return its existing durable record."""
        ...  # ABSTRACT-OK

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        """Persist an outcome and atomically close any linked approval."""
        ...  # ABSTRACT-OK


class EngineActionOutboxStore:
    """Adapter for the engine-native action outbox authority.

    There is intentionally no ``add_node`` fallback. A generic graph write
    cannot atomically establish the idempotency fence and approval/outcome
    record, so the absence of the explicit native seam fails closed before a
    real actuator is called.
    """

    def __init__(self, engine: Any):
        self.engine = engine

    def _call(self, name: str, request: dict[str, Any]) -> dict[str, Any]:
        store = getattr(self.engine, "action_outbox_store", None)
        method = getattr(store, name, None) if store is not None else None
        if not callable(method):
            method = getattr(self.engine, f"{name}_action_outbox", None)
        if not callable(method):
            return {
                "accepted": False,
                "durability_available": False,
                "outcome_unknown": False,
                "reason": "native action outbox authority is unavailable",
            }
        try:
            result = method(dict(request))
        except Exception as exc:  # noqa: BLE001 — durability failure blocks actuation
            logger.warning("fleet action outbox %s failed: %s", name, exc)
            return {
                "accepted": False,
                "durability_available": True,
                "outcome_unknown": True,
                "reason": f"native action outbox {name} failed",
            }
        if isinstance(result, dict):
            return result
        return {
            "accepted": bool(result),
            "durability_available": True,
            "outcome_unknown": result is None,
            "reason": "native action outbox returned a non-record result",
        }

    def prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._call("prepare", request)

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._call("complete", request)


class DryRunActuator:
    """Default no-op actuator: records intent, mutates nothing.

    Keeps an in-memory ``applied`` list for tests/inspection; the durable
    record is the ``ActionExecution`` node written by :func:`execute_action`.
    """

    name = "dryrun"

    def __init__(self) -> None:
        self.applied: list[ActionRequest] = []

    def apply(self, request: ActionRequest) -> dict[str, Any]:
        self.applied.append(request)
        return {
            "ok": True,
            "dry_run": True,
            "detail": f"dry-run: would {request.summary()}",
        }


class DockerActuator:
    """Reference actuator over the docker CLI (optional, guarded).

    Maps action kinds to docker commands — swarm services first, standalone
    containers as fallback:

    * ``restart_service``  → ``docker service update --force`` / ``docker restart``
    * ``scale_service``    → ``docker service scale name=N``
    * ``deploy_service`` / ``rollback_service`` → ``docker service update
      [--image ...|--rollback]``
    * ``stop_service``     → ``docker service scale name=0`` / ``docker stop``
    """

    name = "docker"

    def __init__(self, docker_bin: str | None = None, timeout: float = 60.0):
        self.docker_bin = docker_bin or shutil.which("docker")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.docker_bin)

    def _run(self, *args: str) -> tuple[bool, str]:
        try:
            proc = subprocess.run(  # nosec B603 — fixed binary, validated argv
                [str(self.docker_bin), *args],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            out = (proc.stdout or proc.stderr or "").strip()
            return proc.returncode == 0, out[:500]
        except Exception as e:  # noqa: BLE001 — actuators never raise
            return False, str(e)

    def _is_swarm_service(self, name: str) -> bool:
        ok, _ = self._run("service", "inspect", name, "--format", "{{.ID}}")
        return ok

    def apply(self, request: ActionRequest) -> dict[str, Any]:
        if not self.available:
            return {"ok": False, "dry_run": False, "detail": "docker CLI not available"}
        target = request.target
        if not _SAFE_TARGET.match(target or ""):
            return {
                "ok": False,
                "dry_run": False,
                "detail": f"unsafe target name {target!r}",
            }

        kind = request.kind
        swarm = self._is_swarm_service(target)
        if kind == "restart_service":
            ok, out = (
                self._run("service", "update", "--force", target)
                if swarm
                else self._run("restart", target)
            )
        elif kind == "scale_service":
            replicas = int(request.params.get("replicas", 1))
            if swarm:
                ok, out = self._run("service", "scale", f"{target}={replicas}")
            else:
                ok, out = False, "scale_service needs a swarm service"
        elif kind in ("deploy_service", "redeploy_stack"):
            image = str(request.params.get("image") or "")
            if swarm and image:
                ok, out = self._run("service", "update", "--image", image, target)
            elif swarm:
                ok, out = self._run("service", "update", "--force", target)
            else:
                ok, out = False, "deploy_service needs a swarm service"
        elif kind == "rollback_service":
            if swarm:
                ok, out = self._run("service", "update", "--rollback", target)
            else:
                ok, out = self._run("restart", target)
        elif kind == "stop_service":
            ok, out = (
                self._run("service", "scale", f"{target}=0")
                if swarm
                else self._run("stop", target)
            )
        else:
            ok, out = False, f"unsupported action kind {kind!r}"
        return {"ok": ok, "dry_run": False, "detail": out}


class KubernetesActuator:
    """Reference actuator over ``kubectl`` with an identity/CAS fence.

    Kubernetes actuation is deliberately narrower than the Docker adapter.
    Every request must carry a resource-registry identity containing cluster,
    context, namespace, workload kind, object name, UID, resourceVersion, and
    one declared controller mode.  The actuator reads the object immediately
    before a mutation and rejects any mismatch.  This makes a stale registry,
    wrong kube context, and delete/recreate race fail closed instead of turning
    a service name into an ambient-cluster write.

    ``native`` is the only mode that may write replicas.  ``external_hpa`` and
    ``external_keda`` remain valid discovery modes for restart/rollout actions,
    but a scale/stop request is refused because the external controller owns
    the replica field.  Scale-down additionally requires a side-effect-free
    NE-167 :class:`KubernetesScaleDownGuard` result proving drain and
    stabilization (and quorum safety for StatefulSets).

    The optional ``resource_reader`` and ``scale_down_guard`` are narrow
    dependency-injection seams for the engine/resource-registry integration and
    deterministic fixtures.  They are not fallback emulators: production
    defaults read through the selected ``kubectl --context`` command.
    """

    name = "k8s"

    def __init__(
        self,
        kubectl_bin: str | None = None,
        namespace: str | None = None,
        timeout: float = 60.0,
        *,
        resource_reader: Callable[[KubernetesResourceIdentity], dict[str, Any]]
        | None = None,
        scale_down_guard: KubernetesScaleDownGuard
        | Callable[..., dict[str, Any]]
        | None = None,
    ):
        self.kubectl_bin = kubectl_bin or shutil.which("kubectl")
        if namespace:
            self.namespace = namespace
        else:
            self.namespace = "platform"
            try:
                from agent_utilities.core.config import config as _cfg

                self.namespace = str(
                    getattr(_cfg, "fleet_actuator_k8s_namespace", "platform")
                    or "platform"
                )
            except Exception:  # noqa: BLE001
                pass
        self.timeout = timeout
        self.resource_reader = resource_reader
        self.scale_down_guard = scale_down_guard
        self._last_error = ""

    @property
    def available(self) -> bool:
        return bool(self.kubectl_bin)

    def _run(
        self,
        *args: str,
        context: str | None = None,
        namespace: str | None = None,
    ) -> tuple[bool, str]:
        self._last_error = ""
        command = [str(self.kubectl_bin)]
        if context:
            command.extend(["--context", context])
        effective_namespace = namespace if namespace is not None else self.namespace
        if effective_namespace:
            command.extend(["-n", effective_namespace])
        command.extend(args)
        try:
            proc = subprocess.run(  # nosec B603 — fixed binary, validated argv
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            out = (proc.stdout or proc.stderr or "").strip()
            return proc.returncode == 0, out[:65536]
        except subprocess.TimeoutExpired:
            self._last_error = "timeout"
            return False, "kubectl command timed out"
        except Exception as e:  # noqa: BLE001 — actuators never raise
            self._last_error = "command_error"
            return False, str(e)[:500]

    @staticmethod
    def _failure(detail: str, *, outcome_unknown: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "dry_run": False,
            "detail": str(detail)[:500],
        }
        if outcome_unknown:
            # A timeout after a mutating command may have changed the object.
            # The caller must observe/rollback explicitly; this adapter never
            # guesses and never issues an automatic compensating write.
            result.update(
                {
                    "outcome_unknown": True,
                    "error": "timeout",
                    "rollback_required": True,
                }
            )
        return result

    def _result(self, ok: bool, detail: str, *, mutation_started: bool = False):
        return (
            self._failure(
                detail,
                outcome_unknown=mutation_started and self._last_error == "timeout",
            )
            if not ok
            else {
                "ok": True,
                "dry_run": False,
                "detail": str(detail)[:500],
            }
        )

    def _identity(
        self, request: ActionRequest
    ) -> tuple[KubernetesResourceIdentity | None, str]:
        return KubernetesResourceIdentity.from_request(request)

    def _read_resource(
        self, identity: KubernetesResourceIdentity
    ) -> tuple[dict[str, Any] | None, str]:
        """Read bounded metadata for the selected identity immediately pre-write."""

        if self.resource_reader is not None:
            try:
                raw = self.resource_reader(identity)
            except TimeoutError:
                self._last_error = "timeout"
                return None, "resource identity read timed out"
            except Exception as exc:  # noqa: BLE001 — preflight must fail closed
                return None, f"resource identity read failed: {exc}"
            if not isinstance(raw, dict):
                return None, "resource identity reader returned an invalid record"
            return raw, ""

        ok, out = self._run(
            "get",
            f"{identity.workload_kind.lower()}/{identity.name}",
            "-o",
            "json",
            context=identity.context,
            namespace=identity.namespace,
        )
        if not ok:
            return None, out or "resource identity read failed"
        try:
            raw = json.loads(out)
        except (TypeError, ValueError):
            return None, "kubectl returned malformed resource identity JSON"
        if not isinstance(raw, dict):
            return None, "kubectl returned an invalid resource identity record"

        # A context string is not itself proof of the cluster it resolves to.
        # Ask kubectl for that binding before accepting the object metadata.
        ok, actual_cluster = self._run(
            "config",
            "view",
            "--minify",
            f"--context={identity.context}",
            "-o",
            "jsonpath={.contexts[0].context.cluster}",
            context=None,
            namespace=None,
        )
        if not ok:
            return None, actual_cluster or "unable to verify Kubernetes context cluster"
        if actual_cluster.strip() != identity.cluster:
            return None, "Kubernetes context resolves to an unexpected cluster"
        return raw, ""

    @staticmethod
    def _metadata(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        metadata = raw.get("metadata")
        if not isinstance(metadata, dict):
            metadata = raw
        spec = raw.get("spec")
        if not isinstance(spec, dict):
            spec = {}
        return metadata, spec

    def _verify_resource(
        self,
        identity: KubernetesResourceIdentity,
        raw: dict[str, Any],
    ) -> tuple[bool, str, int | None]:
        metadata, spec = self._metadata(raw)
        actual_kind = str(raw.get("kind") or metadata.get("kind") or "")
        if actual_kind:
            canonical_kind = _K8S_WORKLOAD_KINDS.get(actual_kind.lower())
            if canonical_kind != identity.workload_kind:
                return False, "Kubernetes workload kind changed", None
        checks = (
            ("name", metadata.get("name"), identity.name),
            ("namespace", metadata.get("namespace"), identity.namespace),
            ("uid", metadata.get("uid", raw.get("uid")), identity.uid),
            (
                "resourceVersion",
                metadata.get("resourceVersion", raw.get("resource_version")),
                identity.resource_version,
            ),
        )
        for field, actual, expected in checks:
            if str(actual or "") != expected:
                return False, f"Kubernetes {field} precondition mismatch", None

        # An injected registry reader may provide these explicit bindings; a
        # kubectl JSON response cannot, so the default reader already checked
        # the context→cluster relation above.
        for field, expected in (
            ("cluster", identity.cluster),
            ("context", identity.context),
        ):
            actual = raw.get(field)
            if actual is not None and str(actual) != expected:
                return False, f"Kubernetes {field} binding mismatch", None

        replicas_raw = spec.get("replicas", raw.get("replicas"))
        if replicas_raw is None:
            return True, "", None
        try:
            replicas = int(replicas_raw)
        except (TypeError, ValueError):
            return False, "Kubernetes replica count is invalid", None
        if replicas < 0:
            return False, "Kubernetes replica count is negative", None
        return True, "", replicas

    def _drain_stabilization(
        self,
        identity: KubernetesResourceIdentity,
        current_replicas: int | None,
        desired_replicas: int,
        request: ActionRequest,
    ) -> tuple[bool, str]:
        if current_replicas is None:
            return False, "scale-down requires an observed current replica count"
        if desired_replicas >= current_replicas:
            return True, ""
        guard = self.scale_down_guard
        if guard is None:
            return False, "scale-down requires NE-167 drain/stabilization evidence"
        try:
            assessor = getattr(guard, "assess", None)
            evidence = (
                assessor(identity, current_replicas, desired_replicas, request)
                if callable(assessor)
                else guard(identity, current_replicas, desired_replicas, request)
            )
        except Exception as exc:  # noqa: BLE001 — guard errors fail closed
            return False, f"scale-down guard failed: {exc}"
        if not isinstance(evidence, dict):
            return False, "scale-down guard returned invalid evidence"
        if evidence.get("drained") is not True:
            return False, "scale-down drain evidence is not confirmed"
        if evidence.get("stabilized") is not True:
            return False, "scale-down stabilization evidence is not confirmed"
        evidence_rv = evidence.get("resource_version", evidence.get("resourceVersion"))
        if str(evidence_rv or "") != identity.resource_version:
            return False, "scale-down evidence is stale for resourceVersion"
        try:
            if int(evidence.get("observed_replicas")) != current_replicas:
                return False, "scale-down evidence observed replica count changed"
            if int(evidence.get("remaining_replicas")) != desired_replicas:
                return False, "scale-down evidence targets a different replica count"
        except (TypeError, ValueError):
            return False, "scale-down evidence lacks bounded replica counts"
        # Every StatefulSet reduction requires explicit quorum safety.  This
        # also covers raft/engine members without relying on name heuristics.
        if (
            identity.workload_kind == "StatefulSet"
            and evidence.get("quorum_safe") is not True
        ):
            return False, "StatefulSet scale-down lacks quorum safety evidence"
        if identity.quorum_required and evidence.get("quorum_safe") is not True:
            return False, "quorum scale-down is not safe"
        return True, ""

    def apply(self, request: ActionRequest) -> dict[str, Any]:
        if not self.available:
            return self._failure("kubectl CLI not available")
        target = request.target
        if not _SAFE_TARGET.match(target or ""):
            return self._failure(f"unsafe target name {target!r}")

        identity, identity_error = self._identity(request)
        if identity is None:
            return self._failure(identity_error)

        kind = request.kind
        if kind not in {
            "restart_service",
            "scale_service",
            "deploy_service",
            "redeploy_stack",
            "rollback_service",
            "stop_service",
        }:
            return self._failure(f"unsupported action kind {kind!r}")
        if (
            kind in {"scale_service", "stop_service"}
            and identity.controller_mode != "native"
        ):
            return self._failure(
                f"replica ownership delegated to {identity.controller_mode}"
            )

        raw, read_error = self._read_resource(identity)
        if raw is None:
            return self._failure(read_error)
        matches, mismatch, current_replicas = self._verify_resource(identity, raw)
        if not matches:
            return self._failure(mismatch)

        replicas: int | None = None
        if kind == "scale_service":
            try:
                replicas = int(request.params.get("replicas"))
            except (TypeError, ValueError):
                return self._failure("scale_service requires an integer replica count")
            if replicas < 0 or replicas > 1_000_000:
                return self._failure("replica count is outside the bounded range")
            allowed, guard_error = self._drain_stabilization(
                identity, current_replicas, replicas, request
            )
            if not allowed:
                return self._failure(guard_error)
        elif kind == "stop_service":
            allowed, guard_error = self._drain_stabilization(
                identity, current_replicas, 0, request
            )
            if not allowed:
                return self._failure(guard_error)

        workload = f"{identity.workload_kind.lower()}/{identity.name}"
        run_kwargs = {"context": identity.context, "namespace": identity.namespace}
        mutation_started = True
        if kind == "restart_service":
            ok, out = self._run("rollout", "restart", workload, **run_kwargs)
        elif kind == "scale_service":
            ok, out = self._run(
                "scale",
                workload,
                f"--replicas={replicas}",
                f"--resource-version={identity.resource_version}",
                **run_kwargs,
            )
        elif kind in ("deploy_service", "redeploy_stack"):
            image = str(request.params.get("image") or "")
            container = str(request.params.get("container") or identity.name)
            if not _SAFE_TARGET.fullmatch(container):
                return self._failure(f"unsafe Kubernetes container name {container!r}")
            if len(image) > 500 or any(
                char.isspace() or ord(char) < 32 for char in image
            ):
                return self._failure("unsafe Kubernetes image reference")
            if image:
                ok, out = self._run(
                    "set", "image", workload, f"{container}={image}", **run_kwargs
                )
            else:
                ok, out = self._run("rollout", "restart", workload, **run_kwargs)
        elif kind == "rollback_service":
            ok, out = self._run("rollout", "undo", workload, **run_kwargs)
        elif kind == "stop_service":
            ok, out = self._run(
                "scale",
                workload,
                "--replicas=0",
                f"--resource-version={identity.resource_version}",
                **run_kwargs,
            )
        return self._result(ok, out, mutation_started=mutation_started)


# ── registry (deployment injection point) ───────────────────────────

_ACTUATOR: FleetActuator | None = None


def set_fleet_actuator(actuator: FleetActuator | None) -> None:
    """Register the process-wide actuator (``None`` resets to config default)."""
    global _ACTUATOR
    _ACTUATOR = actuator


def get_fleet_actuator() -> FleetActuator:
    """Resolve the active actuator: injected > ``FLEET_ACTUATOR`` config > dry-run."""
    if _ACTUATOR is not None:
        return _ACTUATOR
    selection = "dryrun"
    try:
        from agent_utilities.core.config import config as _cfg

        selection = str(getattr(_cfg, "fleet_actuator", "dryrun") or "dryrun").lower()
    except Exception:  # noqa: BLE001
        pass
    if selection == "docker":
        docker = DockerActuator()
        if docker.available:
            return docker
        logger.warning("FLEET_ACTUATOR=docker but no docker CLI — using dry-run")
    elif selection in ("k8s", "kubernetes"):
        k8s = KubernetesActuator()
        if k8s.available:
            return k8s
        logger.warning(
            "FLEET_ACTUATOR=%s but no kubectl CLI — using dry-run", selection
        )
    return DryRunActuator()


def _action_idempotency_key(request: ActionRequest, supplied: str | None = None) -> str:
    """Return a stable, opaque key for one action declaration.

    Native scale intents and granted approvals pass their own stronger keys.
    Other callers receive a deterministic key over the governed request so a
    replay cannot silently manufacture a second outbox entry.
    """

    explicit = str(supplied or request.params.get("scale_intent_id") or "").strip()
    if explicit:
        # Keys are persisted and echoed in audit records. Keep bounded platform
        # IDs readable, but hash arbitrary caller input so whitespace and
        # unbounded values never become durable identity material.
        if _SAFE_IDEMPOTENCY_KEY.fullmatch(explicit):
            return explicit
        return "action-key:" + hashlib.sha256(explicit.encode("utf-8")).hexdigest()
    body = json.dumps(
        {
            "kind": request.kind,
            "target": request.target,
            "params": request.params,
            "source": request.source,
            "reason": request.reason,
            "actor_id": request.actor_id,
        },
        default=str,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "action:" + hashlib.sha256(body).hexdigest()[:48]


def _action_request_digest(request: ActionRequest) -> str:
    """Hash the complete governed request for outbox identity checking.

    The explicit idempotency key is the replay identity, while this digest is
    the immutable payload bound to that identity. Keeping both lets an engine
    reject a same-key/different-request delivery without exposing request
    details in the durable key itself.
    """

    body = json.dumps(
        {
            "kind": request.kind,
            "target": request.target,
            "params": request.params,
            "source": request.source,
            "reason": request.reason,
            "actor_id": request.actor_id,
        },
        default=str,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _execution_id(idempotency_key: str) -> str:
    return (
        "action_execution:"
        + hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:32]
    )


def _dry_run_actuator(actuator: FleetActuator) -> bool:
    """Recognize the built-in no-side-effect actuator without calling it."""

    return isinstance(actuator, DryRunActuator) or bool(
        getattr(actuator, "dry_run_only", False)
    )


def _outbox_accepted(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("accepted") is False:
        return False
    # A replay carries the STORED record's own outcome fields (status, ok,
    # ...) through unchanged — a replayed ``recovery_pending``/``failed``
    # prior record legitimately has ``ok: False``, but the *prepare* call
    # itself was still accepted (it correctly refused to redo the side
    # effect). Recognize that before the generic ``ok is False`` bail-out
    # below would otherwise misread the replayed action's own outcome as a
    # rejection of this prepare attempt.
    if result.get("replayed") and str(result.get("status") or "") in (
        _OUTBOX_TERMINAL
        | {_OUTBOX_PREPARED, _OUTBOX_EXECUTING, _OUTBOX_RECOVERY_PENDING}
    ):
        return True
    if result.get("ok") is False:
        return False
    return bool(
        result.get("accepted") is True
        or result.get("ok") is True
        or result.get("status") in {_OUTBOX_PREPARED, "accepted"}
    )


def _outbox_transition_accepted(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("accepted") is False or result.get("ok") is False:
        return False
    return bool(
        result.get("accepted") is True
        or result.get("ok") is True
        or result.get("status") in (_OUTBOX_TERMINAL | {_OUTBOX_RECOVERY_PENDING})
    )


def _outbox_completion_matches(result: Any, expected_state: str) -> bool:
    """Accept only an acknowledged transition to the requested state."""

    if not _outbox_transition_accepted(result):
        return False
    status = str(result.get("status") or "")
    return not status or status in {expected_state, "accepted"}


def _bounded_detail(result: dict[str, Any]) -> str:
    return str(result.get("detail", ""))[:500]


def execute_action(
    engine: Any,
    request: ActionRequest,
    actuator: FleetActuator | None = None,
    *,
    outbox_store: ActionOutboxStore | None = None,
    idempotency_key: str | None = None,
    approval_id: str | None = None,
) -> dict[str, Any]:
    """Durably fence, apply, and complete one action.

    The caller MUST have passed the request through the ActionPolicy gate
    first (CONCEPT:AU-OS.deployment.fleet-lifecycle-control). For every
    non-dry-run actuator, a native idempotent outbox record is committed before
    ``apply``. If that authority is unavailable or completion is lost, this
    function fails closed and never retries an ambiguous side effect. A
    successful dry-run is ``simulated`` and never counts as a real execution;
    a successful non-dry-run is ``executed`` and is the only state eligible for
    scale cooldown accounting.
    """
    act = actuator or get_fleet_actuator()
    key = _action_idempotency_key(request, idempotency_key)
    request_digest = _action_request_digest(request)
    record_id = _execution_id(key)
    timestamp = time.time()

    # Dry-run has no external side effect and can be exercised even when the
    # native outbox capability is not installed. Every real actuator must pass
    # the durable preflight below; there is no graph-node fallback.
    store = outbox_store or EngineActionOutboxStore(engine)
    prepared: dict[str, Any] | None = None
    if not _dry_run_actuator(act):
        try:
            prepared = store.prepare(
                {
                    "operation": "prepare",
                    "idempotency_key": key,
                    "execution_id": record_id,
                    "kind": request.kind,
                    "target": request.target,
                    "params": dict(request.params),
                    "source": request.source,
                    "reason": request.reason[:500],
                    "approval_id": str(approval_id or ""),
                    "request_digest": request_digest,
                    "created_unix": timestamp,
                }
            )
        except Exception as exc:  # noqa: BLE001 — injected durability failures fail closed
            logger.warning("fleet action outbox prepare failed: %s", exc)
            prepared = {
                "accepted": False,
                "durability_available": True,
                "outcome_unknown": True,
                "reason": "native action outbox prepare failed",
            }
        if not isinstance(prepared, dict):
            prepared = {
                "accepted": False,
                "durability_available": True,
                "outcome_unknown": True,
                "reason": "native action outbox returned an invalid prepare result",
            }
        if not _outbox_accepted(prepared):
            known_rejection = bool(
                prepared.get("conflict")
                or prepared.get("rejected")
                or prepared.get("outcome_unknown") is False
            )
            return {
                "ok": False,
                "dry_run": False,
                "state": _OUTBOX_FAILED
                if known_rejection or not prepared.get("outcome_unknown")
                else _OUTBOX_RECOVERY_PENDING,
                "real_execution": False,
                "actuator": getattr(act, "name", "?"),
                "execution_id": record_id,
                "idempotency_key": key,
                "request_digest": request_digest,
                "outbox_status": str(prepared.get("status") or "unavailable"),
                # A durable rejection/conflict is still an authoritative
                # outbox response, so do not fall back to a non-transactional
                # approval stamp. A missing authority is distinct below.
                "outbox_prepared": bool(prepared.get("durability_available")),
                "detail": str(
                    prepared.get("reason") or "durable action intent unavailable"
                )[:500],
                "durability_unavailable": not bool(
                    prepared.get("durability_available")
                ),
                "outcome_unknown": bool(prepared.get("outcome_unknown")),
            }
        if prepared.get("replayed"):
            prior_state = str(prepared.get("status") or "")
            if prior_state in _OUTBOX_TERMINAL:
                # A duplicate approval or process restart must not call the
                # actuator again. A completion retry can repair an approval
                # close without redoing the side effect.
                replay_approval_id = str(
                    approval_id or prepared.get("approval_id") or ""
                )
                try:
                    completion = store.complete(
                        {
                            "operation": "complete",
                            "idempotency_key": key,
                            "execution_id": record_id,
                            "state": prior_state,
                            "ok": prior_state != _OUTBOX_FAILED,
                            "dry_run": prior_state == _OUTBOX_SIMULATED,
                            "approval_id": replay_approval_id,
                            "approval_status": prior_state,
                            "request_digest": request_digest,
                            "replay": True,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 — replay completion remains pending
                    logger.warning(
                        "fleet action outbox replay completion failed: %s", exc
                    )
                    completion = {"accepted": False}
                if not isinstance(completion, dict):
                    completion = {"accepted": False}
                approval_committed = bool(completion.get("approval_committed"))
                completion_accepted = _outbox_completion_matches(
                    completion, prior_state
                )
                if replay_approval_id and (
                    not completion_accepted or not approval_committed
                ):
                    return {
                        "ok": False,
                        "dry_run": prior_state == _OUTBOX_SIMULATED,
                        "state": _OUTBOX_RECOVERY_PENDING,
                        "real_execution": False,
                        "actuator": getattr(act, "name", "?"),
                        "execution_id": record_id,
                        "idempotency_key": key,
                        "replayed": True,
                        "approval_committed": False,
                        "outbox_status": prior_state,
                        "outbox_prepared": True,
                        "outcome_unknown": True,
                        "detail": "approval completion requires durable retry",
                    }
                return {
                    "ok": prior_state != _OUTBOX_FAILED,
                    "dry_run": prior_state == _OUTBOX_SIMULATED,
                    "state": prior_state,
                    "real_execution": prior_state
                    in {_OUTBOX_EXECUTED, _OUTBOX_OBSERVED, _OUTBOX_VERIFIED},
                    "actuator": getattr(act, "name", "?"),
                    "execution_id": record_id,
                    "idempotency_key": key,
                    "request_digest": request_digest,
                    "replayed": True,
                    "approval_committed": approval_committed,
                    "outbox_status": prior_state,
                    "outbox_prepared": True,
                }
            # A durable prepared/executing record has an unknown external
            # outcome. Do not call the actuator again; observer reconciliation
            # must settle it or an operator must issue a new explicit key.
            return {
                "ok": False,
                "dry_run": False,
                "state": _OUTBOX_RECOVERY_PENDING,
                "real_execution": False,
                "actuator": getattr(act, "name", "?"),
                "execution_id": record_id,
                "idempotency_key": key,
                "request_digest": request_digest,
                "replayed": True,
                "outbox_status": prior_state or _OUTBOX_PREPARED,
                "outbox_prepared": True,
                "outcome_unknown": True,
                "detail": "durable action outcome requires observation/recovery",
            }

    apply_raised = False
    try:
        result = act.apply(request) or {}
    except Exception as e:  # noqa: BLE001 — a misbehaving actuator never raises out
        apply_raised = True
        result = {"ok": False, "dry_run": False, "detail": f"actuator error: {e}"}
    if not isinstance(result, dict):
        result = {
            "ok": False,
            "dry_run": False,
            "detail": "actuator returned an invalid result",
        }
    dry_run = bool(result.get("dry_run"))
    ok = bool(result.get("ok"))
    # An actuator timeout after issuing a mutating command cannot be treated
    # as an ordinary rejection: the external world may already have changed.
    # Preserve the outbox's no-replay guarantee and let positive observation
    # settle it instead.
    outcome_unknown = apply_raised or bool(result.get("outcome_unknown"))
    execution_state = (
        _OUTBOX_RECOVERY_PENDING
        if outcome_unknown
        else _OUTBOX_SIMULATED
        if dry_run
        else _OUTBOX_EXECUTED
        if ok
        else _OUTBOX_FAILED
    )
    executed_unix = time.time()
    approval_status = execution_state
    approval_committed = True
    completion: dict[str, Any] | None = None
    if prepared is not None:
        try:
            completion = store.complete(
                {
                    "operation": "complete",
                    "idempotency_key": key,
                    "execution_id": record_id,
                    "state": execution_state,
                    "ok": ok and not outcome_unknown,
                    "dry_run": dry_run,
                    "detail": _bounded_detail(result),
                    "executed_unix": executed_unix,
                    "approval_id": str(approval_id or ""),
                    "approval_status": approval_status,
                    "request_digest": request_digest,
                    "outcome_unknown": outcome_unknown,
                }
            )
        except Exception as exc:  # noqa: BLE001 — completion loss is an ambiguous outcome
            logger.warning("fleet action outbox completion failed: %s", exc)
            completion = {"accepted": False}
        if not isinstance(completion, dict):
            completion = {"accepted": False}
        if not _outbox_completion_matches(completion, execution_state):
            # The actuator may already have changed the world, but the
            # durable outcome/approval close is unknown. Never claim success
            # or issue a second side effect; return a recovery marker.
            execution_state = _OUTBOX_RECOVERY_PENDING
            ok = False
            approval_committed = False
        else:
            approval_committed = bool(
                completion.get("approval_committed", not bool(approval_id))
            )
            if approval_id and not approval_committed:
                execution_state = _OUTBOX_RECOVERY_PENDING
                ok = False

    # Keep the historical ActionExecution projection for query compatibility.
    # The durable outbox completion above is the authority; this projection is
    # never used as the pre-side-effect fence.
    if engine is not None:
        try:
            engine.add_node(
                record_id,
                "ActionExecution",
                properties={
                    "kind": request.kind,
                    "target": request.target,
                    "params_json": json.dumps(request.params, default=str)[:2000],
                    "source": request.source,
                    "actuator": getattr(act, "name", act.__class__.__name__),
                    "ok": ok,
                    "dry_run": dry_run,
                    "state": execution_state,
                    "real_execution": ok and not dry_run,
                    "detail": str(result.get("detail", ""))[:500],
                    "idempotency_key": key,
                    "outbox_status": execution_state,
                    "approval_id": str(approval_id or ""),
                    "request_digest": request_digest,
                    "executed_at": _now_iso(),
                    "executed_unix": executed_unix,
                },
            )
        except Exception as e:  # noqa: BLE001 — result["ok"] is already finalized (from act.apply or the actuator-error dict above) before this write; a failure here only drops the ActionExecution audit node, the caller's returned dict is unaffected
            logger.debug("execute_action: record write failed: %s", e)
    return {
        **result,
        "ok": ok,
        "dry_run": dry_run,
        "execution_id": record_id,
        "actuator": getattr(act, "name", "?"),
        "state": execution_state,
        "real_execution": ok and not dry_run,
        "executed_unix": executed_unix,
        "idempotency_key": key,
        "request_digest": request_digest,
        "approval_committed": approval_committed,
        "outbox_prepared": prepared is not None,
        "outcome_unknown": execution_state == _OUTBOX_RECOVERY_PENDING,
        "outbox_status": (
            completion.get("status", execution_state)
            if completion is not None
            else execution_state
        ),
    }
