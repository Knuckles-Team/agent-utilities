"""CONCEPT:ORCH-1.38 — Default sandbox registry.

Constructs the standard backend set for the router. Each non-local backend is imported
defensively: a backend whose optional dependency is missing (``pydantic-monty``, ``wasmtime``,
the container manager) simply isn't constructed, and the router never sees it. ``local`` is
unconditional — it is the always-available floor that guarantees a non-empty chain.

Backends are added here as the phases land (monty, docker, wasm); until a module exists the
``try`` import is skipped, so this file is safe to ship before the others.
"""

from __future__ import annotations

import logging

from .base import Sandbox
from .local_backend import LocalSandbox

logger = logging.getLogger(__name__)


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
    except Exception as e:  # noqa: BLE001 - optional backend
        logger.debug("monty sandbox not registered: %s", e)

    # wasm / CPython-WASI (Phase 5) — isolated full-stdlib, no host callbacks (v1).
    try:
        from .wasm_backend import WasmSandbox

        backends.append(WasmSandbox())
    except Exception as e:  # noqa: BLE001 - optional backend
        logger.debug("wasm sandbox not registered: %s", e)

    # forkserver (CONCEPT:ORCH-1.87) — native warm-fork via os.fork; isolated, host callbacks
    # via the UDS bridge, zero infra. Available on any Unix host (incl. ARM); the cheap
    # isolated tier between wasm and docker.
    try:
        from .forkserver_backend import ForkServerSandbox

        fork_sb = ForkServerSandbox()
        if fork_sb.is_available():
            backends.append(fork_sb)
    except Exception as e:  # noqa: BLE001 - optional backend
        logger.debug("forkserver sandbox not registered: %s", e)

    # docker / podman (Phase 4) — full isolation, host callbacks via UDS bridge.
    try:
        from .docker_backend import DockerSandbox

        backends.append(DockerSandbox())
    except Exception as e:  # noqa: BLE001 - optional backend
        logger.debug("docker sandbox not registered: %s", e)

    # local — unconditional floor.
    backends.append(LocalSandbox())
    return backends
