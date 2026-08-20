"""CONCEPT:AU-ORCH.sandbox.default-sandbox-registry — Default sandbox registry.

Constructs the standard backend set for the router. Each backend is imported
defensively: a backend whose optional dependency is missing (``pydantic-monty``, ``wasmtime``,
the container manager) simply isn't constructed, and the router never sees it. Every
registered backend is real process/container/VM isolation EXCEPT ``local`` (in-process
``exec``, ORCH-1.38), which is unconditional and deliberately last-ranked: it is the
always-available floor the router falls back to only once every isolating backend above
it is unavailable or has rejected the snippet (see ``local_backend.py``'s module
docstring) — never removing it would leave the RLM loop with zero backends on a host
with no monty/wasm/forkserver/docker/firecracker support.

Backends are added here as the phases land (monty, wasm, forkserver ORCH-1.87,
container_fork ORCH-1.89, docker, firecracker); until a module exists the ``try``
import is skipped, so this file is safe to ship before the others.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from agent_utilities.core.config import setting

from .base import Sandbox

logger = logging.getLogger(__name__)


class _ContainerOptions(TypedDict):
    image: str
    memory: str
    cpus: str
    pids_limit: int
    timeout_secs: float


def _admission_deadline(admission: Any | None) -> float | None:
    limits = getattr(admission, "resource_limits", None)
    if limits is None:
        return None
    deadline = float(limits.deadline_s)
    if admission is not None:
        deadline = min(deadline, float(admission.remaining_seconds()))
    return deadline


def _container_options(admission: Any | None = None) -> _ContainerOptions:
    image = "python:3.12-slim"
    image_ref = str(setting("RLM_CONTAINER_IMAGE_REF", "") or "").strip()
    if image_ref:
        from agent_utilities.security.secrets_client import create_secrets_client

        resolved = create_secrets_client().resolve_ref(image_ref)
        image = (
            resolved.decode("utf-8")
            if isinstance(resolved, bytes)
            else str(resolved or "")
        )
    limits = getattr(admission, "resource_limits", None)
    if limits is not None:
        memory = f"{max(1, int(limits.memory_bytes / (1024 * 1024)))}m"
        cpus = f"{float(limits.cpu_cores):g}"
        pids_limit = int(limits.max_pids)
        deadline = _admission_deadline(admission)
        if deadline is None:
            raise ValueError("admission with resource_limits must resolve a deadline")
        timeout_secs = deadline
    else:
        memory = str(setting("RLM_CONTAINER_MEMORY", "512m"))
        cpus = str(setting("RLM_CONTAINER_CPUS", "1.0"))
        pids_limit = int(setting("RLM_CONTAINER_PIDS_LIMIT", 256))
        timeout_secs = float(setting("RLM_CONTAINER_TIMEOUT_SECONDS", 120.0))
    return {
        "image": image,
        "memory": memory,
        "cpus": cpus,
        "pids_limit": pids_limit,
        "timeout_secs": timeout_secs,
    }


def default_sandboxes(admission: Any | None = None) -> list[Sandbox]:
    """Build the available backend set, cheapest-first by preference rank.

    Construction is cheap (no daemons started, no payloads loaded — that is deferred to each
    backend's ``is_available``/``execute``), so this is safe to call per RLM environment.
    """
    backends: list[Sandbox] = []

    # monty (Phase 3) — fast in-process isolation with native host callbacks.
    try:
        from .monty_backend import MontySandbox

        limits = getattr(admission, "resource_limits", None)
        if limits is None:
            backends.append(MontySandbox())
        else:
            # Monty's checked-in ResourceLimits currently exposes only a wall
            # duration.  Do not route a governed admission to a backend that
            # would silently ignore its memory/PID/CPU-share contract.
            logger.debug(
                "monty sandbox not registered: governed resource limits are unsupported"
            )
    except Exception as exc:  # noqa: BLE001 - optional backend
        logger.debug("monty sandbox not registered: %s", type(exc).__name__)

    # wasm / CPython-WASI (Phase 5) — isolated full-stdlib, no host callbacks (v1).
    try:
        from .wasm_backend import WasmSandbox

        limits = getattr(admission, "resource_limits", None)
        deadline = _admission_deadline(admission)
        backends.append(
            WasmSandbox(
                memory_bytes=int(limits.memory_bytes)
                if limits is not None
                else 1 << 30,
                max_wasm_pages=int(limits.max_wasm_pages)
                if limits is not None
                else None,
                timeout_secs=deadline if deadline is not None else 30.0,
            )
        )
    except Exception as exc:  # noqa: BLE001 - optional backend
        logger.debug("wasm sandbox not registered: %s", type(exc).__name__)

    # forkserver — process isolation via a warmed multiprocessing forkserver;
    # cheaper isolated tier than docker, pricier than wasm (rank 15 vs 20/10).
    try:
        from .forkserver_backend import ForkServerSandbox

        backends.append(
            ForkServerSandbox(
                timeout_secs=_admission_deadline(admission)
                or float(setting("RLM_CONTAINER_TIMEOUT_SECONDS", 120.0))
            )
        )
    except Exception as exc:  # noqa: BLE001 - optional backend
        logger.debug("forkserver sandbox not registered: %s", type(exc).__name__)

    # container_fork — docker/podman exec into a warm, pooled container; warmer
    # (cheaper) than cold docker, heavier than forkserver (rank 18).
    try:
        from .container_fork_backend import ContainerForkSandbox

        backends.append(ContainerForkSandbox(**_container_options(admission)))
    except Exception as exc:  # noqa: BLE001 - optional backend
        logger.debug("container_fork sandbox not registered: %s", type(exc).__name__)

    # docker / podman — full isolation, host callbacks via UDS bridge.
    try:
        from .docker_backend import DockerSandbox

        backends.append(DockerSandbox(**_container_options(admission)))
    except Exception as exc:  # noqa: BLE001 - optional backend
        logger.debug("docker sandbox not registered: %s", type(exc).__name__)

    # firecracker (CONCEPT:AU-ORCH.sandbox.forkd-backed-microvm-strongest) — forkd-backed microVM, the strongest-isolation rung.
    # Registered only where a reachable forkd controller exists (implies x86_64+KVM+forkd);
    # otherwise it never appears and the router uses a cheaper rung.
    try:
        from .firecracker_backend import FirecrackerSandbox

        fc = FirecrackerSandbox(
            resource_limits=getattr(admission, "resource_limits", None)
        )
        if fc.is_available():
            backends.append(fc)
    except Exception as exc:  # noqa: BLE001 - optional backend
        logger.debug("firecracker sandbox not registered: %s", type(exc).__name__)

    # local (in-process exec) — the always-available floor (CONCEPT:AU-ORCH.sandbox.
    # tiered-rlm-sandbox); NOT an isolation boundary, so it is registered last/
    # worst-ranked and only reached when every isolating backend above is
    # unavailable — guarantees the RLM loop is never left with zero backends.
    try:
        from .local_backend import LocalSandbox

        backends.append(LocalSandbox())
    except Exception as exc:  # noqa: BLE001 - optional backend
        logger.debug("local sandbox not registered: %s", type(exc).__name__)
    return backends
