"""CONCEPT:AU-ORCH.sandbox.default-sandbox-registry — Default sandbox registry.

Constructs the standard backend set for the router. Each backend is imported
defensively: a backend whose optional dependency is missing (``pydantic-monty``, ``wasmtime``,
the container manager) simply isn't constructed, and the router never sees it.
Host-process and process-only execution are permanently excluded because neither
confines filesystem, credentials, subprocesses, or network access.

Backends are added here as the phases land (monty, docker, wasm); until a module exists the
``try`` import is skipped, so this file is safe to ship before the others.
"""

from __future__ import annotations

import logging

from agent_utilities.core.config import setting

from .base import Sandbox

logger = logging.getLogger(__name__)


def _container_options() -> dict[str, object]:
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
    return {
        "image": image,
        "memory": str(setting("RLM_CONTAINER_MEMORY", "512m")),
        "cpus": str(setting("RLM_CONTAINER_CPUS", "1.0")),
        "pids_limit": int(setting("RLM_CONTAINER_PIDS_LIMIT", 256)),
        "timeout_secs": float(setting("RLM_CONTAINER_TIMEOUT_SECONDS", 120.0)),
    }


def default_sandboxes() -> list[Sandbox]:
    """Build the available backend set, cheapest-first by preference rank.

    Construction is cheap (no daemons started, no payloads loaded — that is deferred to each
    backend's ``is_available``/``execute``), so this is safe to call per RLM environment.
    """
    backends: list[Sandbox] = []

    # monty (Phase 3) — fast in-process isolation with native host callbacks.
    try:
        from .monty_backend import MontySandbox

        backends.append(MontySandbox())
    except Exception as exc:  # noqa: BLE001 - optional backend
        logger.debug("monty sandbox not registered: %s", type(exc).__name__)

    # wasm / CPython-WASI (Phase 5) — isolated full-stdlib, no host callbacks (v1).
    try:
        from .wasm_backend import WasmSandbox

        backends.append(WasmSandbox())
    except Exception as exc:  # noqa: BLE001 - optional backend
        logger.debug("wasm sandbox not registered: %s", type(exc).__name__)

    # docker / podman — full isolation, host callbacks via UDS bridge.
    try:
        from .docker_backend import DockerSandbox

        backends.append(DockerSandbox(**_container_options()))
    except Exception as exc:  # noqa: BLE001 - optional backend
        logger.debug("docker sandbox not registered: %s", type(exc).__name__)

    # firecracker (CONCEPT:AU-ORCH.sandbox.forkd-backed-microvm-strongest) — forkd-backed microVM, the strongest-isolation rung.
    # Registered only where a reachable forkd controller exists (implies x86_64+KVM+forkd);
    # otherwise it never appears and the router uses a cheaper rung.
    try:
        from .firecracker_backend import FirecrackerSandbox

        fc = FirecrackerSandbox()
        if fc.is_available():
            backends.append(fc)
    except Exception as exc:  # noqa: BLE001 - optional backend
        logger.debug("firecracker sandbox not registered: %s", type(exc).__name__)
    return backends
