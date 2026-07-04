"""CONCEPT:AU-ORCH.sandbox.tiered-rlm-sandbox — Docker/Podman sandbox: full-isolation escalation tier with a host bridge.

This is where code that monty can't run (third-party libs, classes, full stdlib) goes. The
previous ``_execute_container`` had three gaps this backend closes:

* **No resource limits / network isolation** → now ``--network none``, ``--memory``,
  ``--pids-limit``, ``--cpus``, ``--cap-drop ALL``, ``--security-opt no-new-privileges``, and
  a wall-clock timeout (kill on overrun).
* **JSON-string-only context** → now the whole JSON-able REPL namespace + tool sources are
  shipped in, and seeded as globals.
* **Could not serve the RLM host helpers** → now a per-run **UDS bridge**: the host runs an
  asyncio Unix-socket server bound to a socket *inside the bind-mounted run dir*, so a
  ``--network none`` container still reaches ``rlm_query`` etc. over the filesystem socket.
  Async helpers are awaited host-side and the result returned; ``FINAL_VAR`` round-trips and
  mutates the host ``vars`` directly. The container's only egress is that one socket.

The container script (:data:`_RUNNER_SCRIPT`) mirrors :class:`~.local_backend.LocalSandbox`
semantics: tool sources + user code are wrapped in an ``async def`` (so ``await`` works),
stdout is captured, and the helper shims (async for the async host helpers, sync otherwise)
match how the RLM glue calls them. The result (stdout + any in-sandbox error) comes back via a
``result.json`` in the run dir. A container that dies without writing it — daemon gone, OOM
kill, timeout — is an irreversible failure and raises :class:`SandboxFatalError` (ORCH-1.29).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from ..telemetry import SandboxFatalError
from . import _bridge
from .base import Sandbox, SandboxCapabilities, SandboxEnv, SandboxResult

logger = logging.getLogger(__name__)


class DockerSandbox(Sandbox):
    """Run a snippet in a resource-limited, network-isolated container with a host-helper bridge."""

    name = "docker"
    capabilities = SandboxCapabilities(
        host_callbacks=True,  # via the UDS bridge
        third_party_libs=True,
        classes=True,
        full_stdlib=True,
        network=False,  # --network none
        isolated=True,
        preference_rank=20,  # heavy: tried only when cheaper tiers can't
    )

    def __init__(
        self,
        image: str = "python:3.11-slim",
        *,
        memory: str = "512m",
        cpus: str = "1.0",
        pids_limit: int = 256,
        timeout_secs: float = 120.0,
    ):
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.timeout_secs = timeout_secs
        self._runtime: str | None | bool = None  # False = probed-unavailable

    def is_available(self) -> bool:
        return self._resolve_runtime() is not None

    def _resolve_runtime(self) -> str | None:
        """Find a working docker/podman CLI + daemon (cached). ``None`` if neither is usable."""
        if self._runtime is None:
            self._runtime = False
            for rt in ("docker", "podman"):
                if shutil.which(rt) is None:
                    continue
                # `<rt> info` returns 0 only with a reachable daemon.
                try:
                    ok = (
                        subprocess.run(
                            [rt, "info"], capture_output=True, timeout=10
                        ).returncode
                        == 0
                    )
                except Exception:  # noqa: BLE001
                    ok = False
                if ok:
                    self._runtime = rt
                    break
        return self._runtime if isinstance(self._runtime, str) else None

    async def execute(self, code: str, env: SandboxEnv) -> SandboxResult:
        runtime = self._resolve_runtime()
        if runtime is None:
            # Router shouldn't route here if unavailable; if it did, the infra is gone.
            raise SandboxFatalError("docker/podman runtime unavailable")

        run_id = uuid.uuid4().hex[:12]
        tmpdir = Path(tempfile.mkdtemp(prefix=f"rlm-docker-{run_id}-"))
        sock_path = tmpdir / "bridge.sock"
        server: asyncio.AbstractServer | None = None
        try:
            _bridge.write_inputs(
                tmpdir,
                code,
                vars_payload=env.vars,
                tool_sources=env.tool_sources,
                helpers=env.helpers,
                runner_data_dir="/data",
            )
            server = await _bridge.start_bridge(sock_path, env.helpers)
            # The container process may run under a different uid (rootless/userns-remapped
            # docker), so it needs traverse on the dir AND connect (write) on the socket file.
            os.chmod(tmpdir, 0o777)  # nosec B103 — throwaway sandbox dir; container uid differs
            with contextlib.suppress(FileNotFoundError):
                os.chmod(sock_path, 0o777)  # nosec B103 — bridge socket must accept the container uid

            stdout, error, wrote_result = await self._run_container(
                runtime, tmpdir, run_id
            )
            if not wrote_result:
                # No result.json => the container never completed its runner (OOM/timeout/daemon)
                raise SandboxFatalError(
                    f"container produced no result (likely OOM/timeout/daemon death): {stdout[:200]}"
                )
            return SandboxResult(updated_vars={}, stdout=stdout, error=error)
        except SandboxFatalError:
            raise
        except Exception as e:  # noqa: BLE001
            # Any other failure once we've committed to the container path (bridge/subprocess
            # error, resource exhaustion under load) is an irreversible infra failure, not a
            # code rejection — surface it as fatal (ORCH-1.29), never as a wrong-typed escape.
            raise SandboxFatalError(f"docker sandbox failed: {e}") from e
        finally:
            if server is not None:
                server.close()
                with contextlib.suppress(Exception):
                    await server.wait_closed()
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ── container ──────────────────────────────────────────────────────────────
    async def _run_container(
        self, runtime: str, tmpdir: Path, run_id: str
    ) -> tuple[str, str | None, bool]:
        """Run the container with hard isolation + a timeout; return (stdout, error, wrote_result)."""
        name = f"rlm-{run_id}"
        argv = [
            runtime,
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--memory",
            self.memory,
            "--pids-limit",
            str(self.pids_limit),
            "--cpus",
            self.cpus,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "-v",
            f"{tmpdir}:/data:rw",
            self.image,
            "python",
            "/data/runner.py",
        ]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            raw, _ = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout_secs
            )
        except TimeoutError:
            await self._kill_container(runtime, name)
            with contextlib.suppress(Exception):
                await proc.wait()
            container_log = ""
        else:
            container_log = (raw or b"").decode(errors="replace")

        stdout, error, wrote = _bridge.read_result(tmpdir)
        if wrote:
            return stdout, error, True
        return container_log, None, False

    @staticmethod
    async def _kill_container(runtime: str, name: str) -> None:
        with contextlib.suppress(Exception):
            killer = await asyncio.create_subprocess_exec(
                runtime,
                "kill",
                name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
