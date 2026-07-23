#!/usr/bin/python
from __future__ import annotations

"""Fleet actuators — the injectable hands of the autonomy control plane.

CONCEPT:AU-OS.config.desired-state-fleet-reconciler — Desired-state fleet reconciler (actuation seam).

agent-utilities deliberately takes NO hard dependency on the ecosystem's MCP
actuators (portainer-mcp, container-manager, …). Actuation is a protocol:

* :class:`FleetActuator` — ``apply(ActionRequest) -> dict`` for one concrete
  action (restart/scale/deploy/rollback/stop).
* :class:`DryRunActuator` — the DEFAULT. Mutates nothing; every intended
  action is recorded (the caller persists it as an ``ActionExecution`` KG
  node) and announced through the notification seam. This is what makes the
  reconciler safe to enable before any real actuator exists.
* :class:`DockerActuator` — reference implementation over the local docker
  CLI/socket (guarded: inert when ``docker`` is absent). Standalone
  containers and swarm services both supported, argv-only (no shell).
* :class:`KubernetesActuator` — reference implementation over the ``kubectl``
  CLI (guarded: inert when ``kubectl`` is absent), the same argv-only
  no-shell shape as :class:`DockerActuator`. Targets a Deployment in a
  configured namespace (default ``platform``, the W3 worker-topology
  namespace — ``graph-os-dispatch``/``graph-os-ingest``/``graph-os-mining``)
  — this is what lets the W3 k8s worker Deployments actually be scaled by
  the reactive autoscaler (``fleet_autoscaler.py``) instead of only being
  reconciled to a static replica count.

A deployment wires real Portainer/Swarm actuation by registering its own
implementation::

    from agent_utilities.orchestration.fleet_actuation import set_fleet_actuator
    set_fleet_actuator(MyPortainerActuator())

Every execution — real or dry-run — flows through :func:`execute_action`,
which stamps a durable ``ActionExecution`` KG node so "what did autonomy do
and when" is always one graph query away.
"""

import json
import logging
import re
import shutil
import subprocess  # nosec B404 — argv-only docker CLI calls, no shell
import time
import uuid
from typing import Any, Protocol, runtime_checkable

from agent_utilities.orchestration.action_policy import ActionRequest

logger = logging.getLogger(__name__)

# Service/target names must be plain identifiers before they reach a CLI.
_SAFE_TARGET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@runtime_checkable
class FleetActuator(Protocol):
    """Anything that can apply one operational action to the fleet."""

    name: str

    def apply(self, request: ActionRequest) -> dict[str, Any]:
        """Apply ``request``; return ``{ok, detail, dry_run, ...}``. Never raises."""
        ...  # ABSTRACT-OK


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
    """Reference actuator over the ``kubectl`` CLI (optional, guarded).

    Mirrors :class:`DockerActuator`'s shape (argv-only ``subprocess.run``,
    no shell, a fixed binary, a safe-target check) so the two reference
    actuators stay easy to reason about side by side. Every target is
    treated as a ``Deployment`` in ``namespace`` — the shape every W3 worker
    (``graph-os-dispatch``/``graph-os-ingest``/``graph-os-mining``, see
    ``services/graph-os`` ``k8s/w3-worker-topology.yaml``) and the engine
    StatefulSet's peer workloads are declared as:

    * ``scale_service`` / ``stop_service`` (0 replicas) → ``kubectl scale
      deployment/<target> --replicas=N -n <namespace>``
    * ``restart_service`` → ``kubectl rollout restart
      deployment/<target> -n <namespace>``
    * ``deploy_service`` / ``redeploy_stack`` → ``kubectl set image
      deployment/<target> <container>=<image> -n <namespace>`` when an
      image + container are given, else the same rollout-restart as above
    * ``rollback_service`` → ``kubectl rollout undo
      deployment/<target> -n <namespace>``

    No hard dependency on the ``kubernetes`` python client or on
    ``container-manager-mcp``'s ``KubernetesManager`` — same
    no-ecosystem-actuator-dependency posture the module docstring commits
    to for :class:`DockerActuator`, and the same CLI this repo's own
    deployment tooling already shells out to (``deployment/repo_templates.py``'s
    ``kubectl apply`` / ``kubectl rollout status`` CI template).
    """

    name = "k8s"

    def __init__(
        self,
        kubectl_bin: str | None = None,
        namespace: str | None = None,
        timeout: float = 60.0,
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

    @property
    def available(self) -> bool:
        return bool(self.kubectl_bin)

    def _run(self, *args: str) -> tuple[bool, str]:
        try:
            proc = subprocess.run(  # nosec B603 — fixed binary, validated argv
                [str(self.kubectl_bin), "-n", self.namespace, *args],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            out = (proc.stdout or proc.stderr or "").strip()
            return proc.returncode == 0, out[:500]
        except Exception as e:  # noqa: BLE001 — actuators never raise
            return False, str(e)

    def apply(self, request: ActionRequest) -> dict[str, Any]:
        if not self.available:
            return {
                "ok": False,
                "dry_run": False,
                "detail": "kubectl CLI not available",
            }
        target = request.target
        if not _SAFE_TARGET.match(target or ""):
            return {
                "ok": False,
                "dry_run": False,
                "detail": f"unsafe target name {target!r}",
            }

        kind = request.kind
        deployment = f"deployment/{target}"
        if kind == "restart_service":
            ok, out = self._run("rollout", "restart", deployment)
        elif kind == "scale_service":
            replicas = int(request.params.get("replicas", 1))
            ok, out = self._run("scale", deployment, f"--replicas={replicas}")
        elif kind in ("deploy_service", "redeploy_stack"):
            image = str(request.params.get("image") or "")
            container = str(request.params.get("container") or target)
            if image:
                ok, out = self._run("set", "image", deployment, f"{container}={image}")
            else:
                ok, out = self._run("rollout", "restart", deployment)
        elif kind == "rollback_service":
            ok, out = self._run("rollout", "undo", deployment)
        elif kind == "stop_service":
            ok, out = self._run("scale", deployment, "--replicas=0")
        else:
            ok, out = False, f"unsupported action kind {kind!r}"
        return {"ok": ok, "dry_run": False, "detail": out}


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


def execute_action(
    engine: Any,
    request: ActionRequest,
    actuator: FleetActuator | None = None,
) -> dict[str, Any]:
    """Apply ``request`` via ``actuator`` and stamp an ``ActionExecution`` node.

    The caller MUST have passed the request through the ActionPolicy gate
    first (CONCEPT:AU-OS.deployment.fleet-lifecycle-control); this function only actuates + records.
    """
    act = actuator or get_fleet_actuator()
    try:
        result = act.apply(request) or {}
    except Exception as e:  # noqa: BLE001 — a misbehaving actuator never raises out
        result = {"ok": False, "dry_run": False, "detail": f"actuator error: {e}"}
    record_id = f"action_execution:{uuid.uuid4().hex}"
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
                    "ok": bool(result.get("ok")),
                    "dry_run": bool(result.get("dry_run")),
                    "detail": str(result.get("detail", ""))[:500],
                    "executed_at": _now_iso(),
                    "executed_unix": time.time(),
                },
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("execute_action: record write failed: %s", e)
    return {**result, "execution_id": record_id, "actuator": getattr(act, "name", "?")}
