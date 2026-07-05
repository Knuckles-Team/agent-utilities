"""CONCEPT:AU-ORCH.sandbox.native-warm-fork-os — forkserver sandbox: native warm-fork via ``os.fork`` (zero infra).

The cheapest *isolated* warm-fork rung, and the one that runs everywhere — any Linux/Unix host,
including ARM (the GB10), with **no daemon, no KVM, no extra dependency** (pure stdlib
``multiprocessing`` ``forkserver``). It is the native equivalent of forkd's Firecracker fork at
the process level: a long-lived forkserver process pays the heavy ``import`` cost once
(``numpy``/``pandas`` + the bridge), then each snippet runs in a child the forkserver
``os.fork``s — which inherits the parent's loaded modules **copy-on-write** through the kernel,
so the per-snippet ``import numpy`` collapses to zero.

It sits between ``wasm`` (rank 10, isolated but no host callbacks) and ``docker`` (rank 20,
heavy container cold-start) at rank 15, so for the common case — third-party libs **and** host
callbacks (``rlm_query`` …) — the router now prefers a cheap warm fork over a fresh container.

Honest scope of the isolation: a forked child is a *separate process* (its own address space —
it cannot corrupt the orchestrator's objects/``vars`` except through the bridge, and a crash or
OOM never takes the host down), but it is **not** filesystem/network-confined like ``docker``
(``--network none``) or ``wasm`` (WASI preopen). It is the right tier for *our own* glue code at
fan-out speed; hostile-input confinement still escalates to ``docker``/``wasm``/``firecracker``.
Host helpers are served exactly like ``docker`` — over the shared UDS bridge (``_bridge``).
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import logging
import multiprocessing
import os
import shutil
import tempfile
import threading
from pathlib import Path

from ..telemetry import SandboxFatalError
from . import _bridge
from .base import (
    ForkableSandbox,
    ParentHandle,
    SandboxCapabilities,
    SandboxEnv,
    SandboxResult,
    WarmSpec,
)

logger = logging.getLogger(__name__)

# Always preloaded so the child can resolve the bridge runner with zero import cost. The heavy
# scientific libs are preloaded only when present on the host (probed without importing them
# into *this* process — find_spec does not execute the module).
_BRIDGE_MODULE = "agent_utilities.rlm.sandboxes._bridge"
_CANDIDATE_PRELOAD = ("numpy", "pandas")


def _available_preload() -> tuple[str, ...]:
    extra = tuple(
        m for m in _CANDIDATE_PRELOAD if importlib.util.find_spec(m) is not None
    )
    return (_BRIDGE_MODULE, *extra)


def _noop() -> None:  # forkserver boot probe target (paid in warm(), not first execute)
    return None


class ForkServerSandbox(ForkableSandbox):
    """Run a snippet in a child ``os.fork``ed from a warmed ``multiprocessing`` forkserver."""

    name = "forkserver"
    capabilities = SandboxCapabilities(
        host_callbacks=True,  # via the UDS bridge (same as docker)
        third_party_libs=True,
        classes=True,
        full_stdlib=True,
        network=True,  # process isolation only — NOT network-confined (cf. docker/wasm)
        isolated=True,  # separate address space; weaker than container/VM, stronger than local
        preference_rank=15,  # cheaper isolated tier than docker (20); pricier than wasm (10)
        warm_fork=True,
    )

    def __init__(
        self, *, timeout_secs: float = 120.0, preload: tuple[str, ...] | None = None
    ) -> None:
        self.timeout_secs = timeout_secs
        # ``preload`` overrides the auto-detected warm set (used by tests to boot a lean
        # forkserver without paying the numpy/pandas import); always keep the bridge module.
        if preload is None:
            self._preload = _available_preload()
        else:
            self._preload = (
                _BRIDGE_MODULE,
                *(m for m in preload if m != _BRIDGE_MODULE),
            )
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is None:
            self._available = "forkserver" in multiprocessing.get_all_start_methods()
        return self._available

    def warm_spec(self) -> WarmSpec:
        return WarmSpec(backend=self.name, preload=self._preload)

    async def warm(self, spec: WarmSpec) -> ParentHandle:
        """Start a forkserver preloaded with ``spec.preload`` and boot it (pay start-up once)."""
        try:
            ctx = multiprocessing.get_context("forkserver")
            ctx.set_forkserver_preload(list(spec.preload))
            # Boot the server now (with the preload imported) via a throwaway probe child, so
            # the FIRST real snippet already forks from a warm parent.
            probe = ctx.Process(target=_noop)
            await asyncio.get_running_loop().run_in_executor(
                None, _start_join, probe, 60.0
            )
        except Exception as e:  # noqa: BLE001
            raise SandboxFatalError(f"forkserver warm-up failed: {e}") from e
        return ParentHandle(
            backend=self.name,
            spec=spec,
            ref=ctx,
            close=lambda: _shutdown_forkserver(ctx),
        )

    async def run_forked(
        self, parent: ParentHandle, code: str, env: SandboxEnv
    ) -> SandboxResult:
        ctx = parent.ref
        run_id = os.urandom(6).hex()
        tmpdir = Path(tempfile.mkdtemp(prefix=f"rlm-fork-{run_id}-"))
        sock_path = tmpdir / "bridge.sock"
        server: asyncio.AbstractServer | None = None
        try:
            _bridge.write_inputs(
                tmpdir,
                code,
                vars_payload=env.vars,
                tool_sources=env.tool_sources,
                helpers=env.helpers,
                runner_data_dir=None,  # child calls _bridge.run_child directly (no injected script)
            )
            server = await _bridge.start_bridge(sock_path, env.helpers)
            proc = ctx.Process(
                target=_bridge.run_child, args=(str(tmpdir), str(sock_path))
            )
            # Fork + run in an executor thread so the event loop stays free to service the
            # bridge callbacks the child makes (rlm_query, FINAL_VAR, …) while it runs.
            killed = await asyncio.get_running_loop().run_in_executor(
                None, _start_join, proc, self.timeout_secs
            )
            stdout, error, wrote = _bridge.read_result(tmpdir)
            if not wrote:
                why = (
                    "timed out"
                    if killed
                    else "died before writing a result (crash/OOM)"
                )
                raise SandboxFatalError(f"forkserver child {why}")
            return SandboxResult(updated_vars={}, stdout=stdout, error=error)
        except SandboxFatalError:
            raise
        except Exception as e:  # noqa: BLE001 - committed to the fork path => infra failure
            raise SandboxFatalError(f"forkserver sandbox failed: {e}") from e
        finally:
            if server is not None:
                server.close()
                with contextlib.suppress(Exception):
                    await server.wait_closed()
            shutil.rmtree(tmpdir, ignore_errors=True)


# The stdlib ``forkserver`` control channel is a process-wide singleton and is NOT
# thread-safe. Warm-fork fan-out starts N children concurrently from executor
# threads (``run_in_executor``), and racing ``Process.start()`` calls corrupt the
# fork-request / child-pid handshake — surfacing intermittently as
# ``'NoneType' object cannot be interpreted as an integer`` on one branch of a
# fan-out. Serialize ONLY the ``start()`` request; the ``join`` (where the snippet
# actually runs) stays concurrent, so fan-out parallelism is preserved.
_FORK_START_LOCK = threading.Lock()


def _start_join(proc: multiprocessing.process.BaseProcess, timeout: float) -> bool:
    """Start ``proc`` and join with a timeout; terminate on overrun. Returns True if killed."""
    with _FORK_START_LOCK:
        proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        if proc.is_alive():  # pragma: no cover - terminate is usually enough
            proc.kill()
        return True
    return False


def _shutdown_forkserver(ctx: object) -> None:
    """Best-effort stop of the forkserver process backing ``ctx`` (idempotent)."""
    with contextlib.suppress(Exception):
        from multiprocessing import forkserver

        fs = forkserver._forkserver  # noqa: SLF001 - stdlib singleton; only clean stop available
        if getattr(fs, "_forkserver_pid", None) is not None:
            fs._stop()  # type: ignore[attr-defined]  # noqa: SLF001 — stdlib-private stop
