#!/usr/bin/python
from __future__ import annotations

"""Multi-backend deployment planner for the self-composing ``graph-os`` entrypoint.

``graph-os`` (the ``mcp_server()`` entrypoint + :mod:`agent_utilities.mcp.co_service_supervisor`)
already composes its co-services (messaging, the KG host daemon, agent-webui) IN
one process wherever it runs. This module answers the follow-on question —
*where* should that one process run: this process, a docker/podman container, a
Kubernetes pod, or a plain host process managed by systemd — using the SAME
composition detection (:func:`agent_utilities.mcp.co_service_supervisor.detect_composition`)
so every backend agrees on what co-services are configured.

**No new orchestrator client is built here.** Per this repo's "Sprawl boundaries —
extend before you add" rule, the fleet already has real docker/podman/Kubernetes
coverage (``container-manager-mcp``: ``cm_compose_operations``, the ``cm_k8s_*``
tool family), host/service management (``systems-manager``: ``sm_service_operations``,
``sm_file_operations``), and the canonical host inventory + SSH remote-exec
(``tunnel-manager``: ``tm_hosts``/``tm_remote``, reading
``~/.config/agent-utilities/inventory.yaml`` — see ``docs/examples/example_tunnel_inventory.yaml``).
Every backend below is a THIN adapter that renders a :class:`DeploymentPlan` — a
deterministic, reviewable, data-only description of what would be done — over
those existing tools. A plan's :class:`FleetToolCall` entries are never dispatched
by this module; the sanctioned way to actually run one is the same path every
other cross-service action in this codebase uses (``AGENTS.md`` → "Delegate to
the KG + graph-os"): ``graph_orchestrate action=execute_agent`` against the named
server (human-reviewed), or the fleet loader (``load_tools`` then a direct tool
call). This module's job stops at producing a correct plan.

Honesty over completeness (this project has been burned before by a "deployer"
that reported success without doing the work): the four backends are NOT equally
real.

* :class:`InProcessBackend` — REAL. ``apply(dry_run=False)`` genuinely starts the
  composed co-services in THIS process via the same
  :func:`agent_utilities.mcp.co_service_supervisor.start_co_services` the
  ``graph-os`` entrypoint itself calls. Fully unit-tested.
* :class:`NativeShellBackend` — REAL for the artifact it owns (a systemd unit
  file's content) — that generation is unit-tested. The install/start steps are
  PLAN-ONLY ``FleetToolCall``s against ``systems-manager``/``tunnel-manager``;
  this module never runs them (no live host access from here).
* :class:`ContainerBackend` (docker/podman) — PLAN-ONLY. Renders a real compose
  YAML (unit-tested) and the exact ``container-manager-mcp`` ``cm_compose_operations``
  call that would apply it; ``apply()`` always refuses to execute (raises), it
  never calls the fleet tool.
* :class:`KubernetesBackend` — PLAN-ONLY, and further limited: ``container-manager-mcp``
  exposes rich per-kind ``cm_k8s_*`` CRUD (StatefulSet/DaemonSet/Job/CronJob/
  ConfigMap/Secret/...) but **no generic manifest-apply or Deployment-create
  tool**. The plan therefore renders a manifest for the operator's existing
  GitOps path (Portainer / ``kubectl apply`` review) rather than claiming an
  executable step that does not exist. ``apply()`` always refuses.

READ-ONLY / no live mutation from this module, ever — ``apply()`` is a dry-run
planner for every backend except :class:`InProcessBackend` (which mutates only
the CALLING process's own composition, exactly like starting ``graph-os``
locally always has).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from agent_utilities.mcp.co_service_supervisor import (
    CompositionPlan,
    detect_composition,
)

logger = logging.getLogger(__name__)

BackendName = Literal["in_process", "container", "kubernetes", "native_shell"]


class PlanOnlyBackendError(RuntimeError):
    """Raised by a PLAN-ONLY backend's ``apply(dry_run=False)``.

    Deliberate refusal, not an unimplemented stub: ``ContainerBackend``,
    ``KubernetesBackend``, and ``NativeShellBackend`` never execute a
    ``FleetToolCall`` themselves (see the module docstring) — this fails loud
    instead of silently no-op-ing when a caller asks for a live apply anyway.
    """


@dataclass(frozen=True)
class FleetToolCall:
    """One MCP tool call a plan would need — data, never invoked by this module.

    ``server`` names an existing fleet MCP server (``container-manager-mcp``,
    ``tunnel-manager``, ``systems-manager``); ``tool``/``args`` are that server's
    real, documented tool contract (see the module docstring for which ones).
    """

    server: str
    tool: str
    args: dict[str, Any]
    description: str = ""


@dataclass(frozen=True)
class DeploymentStep:
    """One step of a plan: either a local (in-this-module) action or a fleet call."""

    description: str
    fleet_call: FleetToolCall | None = None
    local_action: str | None = None


@dataclass(frozen=True)
class DeploymentPlan:
    """A deterministic, reviewable description of what a backend would do.

    ``live_capable`` is honest about whether :meth:`DeploymentBackend.apply` can
    actually execute this plan (``True`` only for :class:`InProcessBackend`) or
    whether it is dry-run/plan output that an operator (or a reviewed
    ``execute_agent`` run) must carry out.
    """

    backend: BackendName
    target: str
    composition: CompositionPlan
    steps: tuple[DeploymentStep, ...]
    live_capable: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)
    # Rendered file contents this plan produced (e.g. {"compose.yml": "..."},
    # {"manifest.yaml": "..."}, {"graph-os.service": "..."}) — data, not a write.
    artifacts: dict[str, str] = field(default_factory=dict)

    def rendered_summary(self) -> str:
        lines = [f"[{self.backend}] target={self.target!r}"]
        lines.append(
            f"  composition: {', '.join(self.composition.co_service_names()) or '(none configured)'}"
        )
        for i, step in enumerate(self.steps, 1):
            lines.append(f"  {i}. {step.description}")
            if step.fleet_call is not None:
                lines.append(
                    f"     -> {step.fleet_call.server}.{step.fleet_call.tool}"
                    f"({step.fleet_call.args!r})"
                )
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


class DeploymentBackend(Protocol):
    """A deployment target this composition can be projected onto."""

    name: BackendName

    def plan(
        self, *, target: str, engine: Any = None, **kwargs: Any
    ) -> DeploymentPlan: ...

    def apply(
        self, plan: DeploymentPlan, *, dry_run: bool = True, **kwargs: Any
    ) -> dict[str, Any]: ...


# ── in-process (the uvx / stdio / streamable-http case — REAL) ────────────────


class InProcessBackend:
    """The composition already running in THIS interpreter — what ``graph-os`` IS."""

    name: BackendName = "in_process"

    def plan(
        self, *, target: str = "this process", engine: Any = None, **_: Any
    ) -> DeploymentPlan:
        composition = detect_composition(engine)
        steps = (
            DeploymentStep(
                description=(
                    "run `graph-os` (console script, or `uvx agent-utilities "
                    "graph-os`) — the entrypoint self-composes every configured "
                    "co-service in this same process"
                ),
                local_action="agent_utilities.mcp.kg_server.mcp_server",
            ),
        )
        return DeploymentPlan(
            backend=self.name,
            target=target,
            composition=composition,
            steps=steps,
            live_capable=True,
        )

    def apply(
        self,
        plan: DeploymentPlan,
        *,
        dry_run: bool = True,
        session: Any = None,
        engine: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Genuinely start the composed co-services in THIS process.

        This starts co-services only (not the MCP transport loop itself, which
        blocks) — the same real, tested entry points ``kg_server.mcp_server()``
        calls: :func:`bring_up_host_daemon_if_needed` and
        :func:`start_co_services`. The returned ``CoServiceSupervisor`` is the
        caller's to stop.
        """
        if dry_run:
            return {"applied": False, "plan": plan}
        if session is None:
            raise ValueError("apply(dry_run=False) requires a verified session")
        from agent_utilities.mcp.co_service_supervisor import (
            bring_up_host_daemon_if_needed,
            start_co_services,
        )

        bring_up_host_daemon_if_needed()
        supervisor = start_co_services(session, engine)
        return {"applied": True, "supervisor": supervisor}


# ── container (docker/podman via container-manager-mcp — PLAN-ONLY) ───────────


def render_compose(
    composition: CompositionPlan,
    *,
    image: str,
    transport: str = "streamable-http",
    port: int = 8000,
    webui_image: str | None = None,
) -> str:
    """Render the compose YAML for one self-composing ``graph-os`` service.

    Only ONE ``graph-os`` service is ever rendered — messaging and the host
    daemon are co-services of that SAME process (self-composing), not separate
    containers; only ``agent-webui`` is a genuinely separate service (a Node/Vite
    frontend — see ``co_service_supervisor``'s module docstring). This is the
    lightweight single-node case (``agent-utilities-deployment`` tiny/single-node-
    prod profiles); a hardened multi-secret production Swarm profile already
    exists at ``deploy/swarm/graphos.stack.yml`` and should be extended, not
    duplicated, for that use case.
    """
    import yaml

    services: dict[str, Any] = {
        "graph-os": {
            "image": image,
            "command": [
                "graph-os",
                "--transport",
                transport,
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
            ],
            "environment": {"TRANSPORT": transport},
            "ports": [f"{port}:{port}"],
            "restart": "unless-stopped",
        }
    }
    if composition.web_ui_enabled:
        services["agent-webui"] = {
            "image": webui_image or "agent-webui:latest",
            "environment": {"ENABLE_WEB_UI": "true"},
            "ports": ["5173:5173"],
            "restart": "unless-stopped",
        }
    return yaml.safe_dump({"services": services}, sort_keys=False)


class ContainerBackend:
    """docker/podman via ``container-manager-mcp``'s ``cm_compose_operations``. PLAN-ONLY."""

    name: BackendName = "container"

    def plan(
        self,
        *,
        target: str,
        engine: Any = None,
        image: str = "ghcr.io/knuckles-team/agent-utilities:latest",
        manager_type: Literal["docker", "podman"] = "docker",
        compose_file: str = "/opt/graphos/compose.yml",
        **_: Any,
    ) -> DeploymentPlan:
        composition = detect_composition(engine)
        compose_yaml = render_compose(composition, image=image)
        steps = (
            DeploymentStep(
                description=f"render the compose file to {compose_file!r} on {target!r}",
                local_action="render_compose",
            ),
            DeploymentStep(
                description=(
                    f"bring the stack up on host {target!r} "
                    f"(container-manager-mcp resolves `host` via the "
                    "tunnel-manager inventory)"
                ),
                fleet_call=FleetToolCall(
                    server="container-manager-mcp",
                    tool="cm_compose_operations",
                    args={
                        "action": "up",
                        "compose_file": compose_file,
                        "host": target,
                        "manager_type": manager_type,
                    },
                    description="docker/podman compose up",
                ),
            ),
        )
        return DeploymentPlan(
            backend=self.name,
            target=target,
            composition=composition,
            steps=steps,
            live_capable=False,
            warnings=(
                "dry-run/plan only — this backend never calls the fleet tool; "
                "an operator (or a reviewed `execute_agent` run) applies it.",
            ),
            artifacts={"compose.yml": compose_yaml},
        )

    def apply(
        self, plan: DeploymentPlan, *, dry_run: bool = True, **_: Any
    ) -> dict[str, Any]:
        if not dry_run:
            raise PlanOnlyBackendError(
                "ContainerBackend is plan-only by design — dry_run=False is "
                "not supported; hand the plan's FleetToolCall to an operator "
                "or a reviewed execute_agent run."
            )
        return {"applied": False, "plan": plan}


# ── kubernetes (PLAN-ONLY, and honestly limited) ───────────────────────────────


def render_k8s_manifest(
    composition: CompositionPlan,
    *,
    image: str,
    namespace: str = "default",
    port: int = 8000,
    webui_image: str | None = None,
) -> str:
    """Render a plain Deployment+Service manifest for one self-composing ``graph-os``.

    Informational only (see the module docstring: ``container-manager-mcp`` has
    no generic apply/Deployment-create tool) — this is meant for the operator's
    existing GitOps review path (Portainer / ``kubectl apply --dry-run``), not an
    executable step.
    """
    import yaml

    documents: list[dict[str, Any]] = [
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "graph-os", "namespace": namespace},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": "graph-os"}},
                "template": {
                    "metadata": {"labels": {"app": "graph-os"}},
                    "spec": {
                        "containers": [
                            {
                                "name": "graph-os",
                                "image": image,
                                "command": [
                                    "graph-os",
                                    "--transport",
                                    "streamable-http",
                                    "--host",
                                    "0.0.0.0",
                                    "--port",
                                    str(port),
                                ],
                                "env": [
                                    {"name": "TRANSPORT", "value": "streamable-http"}
                                ],
                                "ports": [{"containerPort": port}],
                            }
                        ]
                    },
                },
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "graph-os", "namespace": namespace},
            "spec": {
                "selector": {"app": "graph-os"},
                "ports": [{"port": port, "targetPort": port}],
            },
        },
    ]
    if composition.web_ui_enabled:
        documents.append(
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "agent-webui", "namespace": namespace},
                "spec": {
                    "replicas": 1,
                    "selector": {"matchLabels": {"app": "agent-webui"}},
                    "template": {
                        "metadata": {"labels": {"app": "agent-webui"}},
                        "spec": {
                            "containers": [
                                {
                                    "name": "agent-webui",
                                    "image": webui_image or "agent-webui:latest",
                                    "ports": [{"containerPort": 5173}],
                                }
                            ]
                        },
                    },
                },
            }
        )
    return yaml.safe_dump_all(documents, sort_keys=False)


class KubernetesBackend:
    """Kubernetes, PLAN-ONLY — and honestly limited further than ``ContainerBackend``.

    ``container-manager-mcp`` exposes rich per-kind ``cm_k8s_*`` CRUD but no
    generic manifest-apply or Deployment-create tool. This backend therefore
    never claims a ``FleetToolCall`` that would apply the rendered manifest —
    only ``cm_k8s_config`` actions that genuinely exist (namespace/configmap
    creation) are offered as steps; the Deployment/Service manifest itself is
    informational output for the operator's existing GitOps review path.
    """

    name: BackendName = "kubernetes"

    def plan(
        self,
        *,
        target: str,
        engine: Any = None,
        image: str = "ghcr.io/knuckles-team/agent-utilities:latest",
        namespace: str = "default",
        **_: Any,
    ) -> DeploymentPlan:
        composition = detect_composition(engine)
        manifest = render_k8s_manifest(composition, image=image, namespace=namespace)
        steps = (
            DeploymentStep(
                description=f"render the Deployment+Service manifest for namespace {namespace!r}",
                local_action="render_k8s_manifest",
            ),
            DeploymentStep(
                description=f"ensure namespace {namespace!r} exists on cluster {target!r}",
                fleet_call=FleetToolCall(
                    server="container-manager-mcp",
                    tool="cm_k8s_config",
                    args={"action": "create_namespace", "namespace": namespace},
                    description="create_namespace is idempotent-checked by the operator before use",
                ),
            ),
        )
        return DeploymentPlan(
            backend=self.name,
            target=target,
            composition=composition,
            steps=steps,
            live_capable=False,
            warnings=(
                "PLAN-ONLY: container-manager-mcp has no generic manifest-apply "
                "or Deployment-create tool — the rendered manifest is for the "
                "operator's existing GitOps review path (Portainer / `kubectl "
                "apply --dry-run`), not an executable step from this module.",
                "READ-ONLY on the live cluster by design — this never patches "
                "or applies anything.",
            ),
            artifacts={"manifest.yaml": manifest},
        )

    def apply(
        self, plan: DeploymentPlan, *, dry_run: bool = True, **_: Any
    ) -> dict[str, Any]:
        if not dry_run:
            raise PlanOnlyBackendError(
                "KubernetesBackend is plan-only by design (and read-only on "
                "the live cluster) — dry_run=False is not supported."
            )
        return {"applied": False, "plan": plan}


# ── native shell (bare host via tunnel-manager / systems-manager) ─────────────


def render_systemd_unit(
    *,
    exec_start: str,
    description: str = "graph-os (self-composing agent-utilities entrypoint)",
) -> str:
    """Render a systemd unit that runs the self-composing ``graph-os`` entrypoint.

    Real, unit-tested content generation — this is the artifact
    :class:`NativeShellBackend` owns outright. Installing/starting it on a
    target host stays a PLAN-ONLY ``FleetToolCall`` (this module has no live
    host access).
    """
    return (
        "[Unit]\n"
        f"Description={description}\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={exec_start}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


class NativeShellBackend:
    """A bare host (systemd), reached via tunnel-manager's SSH remote-exec.

    ``target`` is a host alias resolved from
    ``~/.config/agent-utilities/inventory.yaml`` (see
    ``docs/examples/example_tunnel_inventory.yaml`` and tunnel-manager's
    ``tm_hosts``/``tm_remote``) — the same inventory container-manager-mcp's
    ``host`` parameter resolves, so one inventory file drives every backend.
    ``systems-manager``'s ``sm_service_operations``/``sm_file_operations`` only
    operate on the host THEY run on (no remote-host parameter), so reaching an
    arbitrary inventory host for a plain shell/systemd deployment goes through
    tunnel-manager's ``tm_remote`` (SSH ``run_command``), not systems-manager.
    """

    name: BackendName = "native_shell"

    def plan(
        self,
        *,
        target: str,
        engine: Any = None,
        python_bin: str = "python3",
        unit_name: str = "graph-os.service",
        unit_path: str = "/etc/systemd/system/graph-os.service",
        **_: Any,
    ) -> DeploymentPlan:
        composition = detect_composition(engine)
        exec_start = (
            f"{python_bin} -m agent_utilities.cli graph-os --transport streamable-http"
        )
        unit = render_systemd_unit(exec_start=exec_start)
        write_cmd = f"cat > {unit_path} <<'UNIT'\n{unit}UNIT\nsystemctl daemon-reload"
        steps = (
            DeploymentStep(
                description=f"render {unit_name} (ExecStart runs the self-composing entrypoint)",
                local_action="render_systemd_unit",
            ),
            DeploymentStep(
                description=f"write {unit_path!r} on host {target!r} and reload systemd",
                fleet_call=FleetToolCall(
                    server="tunnel-manager",
                    tool="tm_remote",
                    args={"action": "run_command", "host": target, "cmd": write_cmd},
                    description="SSH run_command (host resolved from inventory.yaml)",
                ),
            ),
            DeploymentStep(
                description=f"enable + start {unit_name} on host {target!r}",
                fleet_call=FleetToolCall(
                    server="tunnel-manager",
                    tool="tm_remote",
                    args={
                        "action": "run_command",
                        "host": target,
                        "cmd": f"systemctl enable --now {unit_name}",
                    },
                    description="the local-host equivalent is systems-manager's "
                    "sm_service_operations(action='enable_service'/'start_service')",
                ),
            ),
        )
        return DeploymentPlan(
            backend=self.name,
            target=target,
            composition=composition,
            steps=steps,
            live_capable=False,
            warnings=(
                "dry-run/plan only — this backend never calls the fleet tool; "
                "an operator (or a reviewed `execute_agent` run) applies it.",
            ),
            artifacts={unit_name: unit},
        )

    def apply(
        self, plan: DeploymentPlan, *, dry_run: bool = True, **_: Any
    ) -> dict[str, Any]:
        if not dry_run:
            raise PlanOnlyBackendError(
                "NativeShellBackend is plan-only by design — dry_run=False is "
                "not supported; hand the plan's FleetToolCall to an operator "
                "or a reviewed execute_agent run."
            )
        return {"applied": False, "plan": plan}


# ── registry ────────────────────────────────────────────────────────────────

_BACKENDS: dict[BackendName, DeploymentBackend] = {
    "in_process": InProcessBackend(),
    "container": ContainerBackend(),
    "kubernetes": KubernetesBackend(),
    "native_shell": NativeShellBackend(),
}


def get_backend(name: BackendName) -> DeploymentBackend:
    """Look up a registered :class:`DeploymentBackend` by name."""
    try:
        return _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown deployment backend {name!r}; known: {sorted(_BACKENDS)}"
        ) from None
