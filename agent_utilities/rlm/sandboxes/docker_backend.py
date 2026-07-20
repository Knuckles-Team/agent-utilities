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

The container script wraps tool sources + user code in an ``async def`` so ``await`` works,
captures stdout, and exposes helper shims that match how the RLM glue calls them. The result
(stdout + any in-sandbox error) comes back via ``result.json`` in the run dir. A container that
dies without writing it — daemon gone, OOM kill, timeout — is an irreversible failure and
raises :class:`SandboxFatalError` (ORCH-1.29).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from agent_utilities.core.profile_guard import is_production_profile

from ..telemetry import SandboxFatalError
from . import _bridge
from .base import Sandbox, SandboxCapabilities, SandboxEnv, SandboxResult

logger = logging.getLogger(__name__)

_IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:-]{0,254}$")
_DIGEST_IMAGE_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._/:-]{0,180}@sha256:[0-9a-f]{64}$"
)
_MEMORY_RE = re.compile(r"^(\d+)([kKmMgG])$")


def validate_container_configuration(
    image: str,
    memory: str,
    cpus: str | float,
    pids_limit: int,
    timeout_secs: float,
) -> tuple[str, str, str, int, float]:
    """Validate every operator-controlled value before it enters runtime argv."""
    rendered_image = str(image or "").strip()
    if not _IMAGE_RE.fullmatch(rendered_image):
        raise ValueError("sandbox container image is invalid")
    if is_production_profile() and not _DIGEST_IMAGE_RE.fullmatch(rendered_image):
        raise ValueError("production sandbox images require an immutable sha256 digest")
    rendered_memory = str(memory or "").strip()
    match = _MEMORY_RE.fullmatch(rendered_memory)
    if match is None:
        raise ValueError("sandbox memory limit is invalid")
    factor = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}[match.group(2).lower()]
    memory_bytes = int(match.group(1)) * factor
    if not 64 * (1 << 20) <= memory_bytes <= 64 * (1 << 30):
        raise ValueError("sandbox memory limit is out of range")
    try:
        cpu_value = float(cpus)
    except (TypeError, ValueError):
        raise ValueError("sandbox CPU limit is invalid") from None
    if not 0.1 <= cpu_value <= 16.0:
        raise ValueError("sandbox CPU limit is out of range")
    if not isinstance(pids_limit, int) or isinstance(pids_limit, bool):
        raise ValueError("sandbox process limit is invalid")
    if not 16 <= pids_limit <= 1_024:
        raise ValueError("sandbox process limit is out of range")
    timeout = float(timeout_secs)
    if not 1.0 <= timeout <= 600.0:
        raise ValueError("sandbox timeout is out of range")
    return (
        rendered_image,
        rendered_memory.lower(),
        f"{cpu_value:g}",
        pids_limit,
        timeout,
    )


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
        (
            self.image,
            self.memory,
            self.cpus,
            self.pids_limit,
            self.timeout_secs,
        ) = validate_container_configuration(
            image, memory, cpus, pids_limit, timeout_secs
        )
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
                            [rt, "info"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=10,
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

        run_id = uuid.uuid4().hex
        tmpdir = Path(tempfile.mkdtemp(prefix=f"rlm-docker-{run_id}-"))
        sock_path = tmpdir / "bridge.sock"
        bridge_token = _bridge.new_bridge_token()
        server: asyncio.AbstractServer | None = None
        try:
            _bridge.write_inputs(
                tmpdir,
                code,
                vars_payload=env.vars,
                tool_sources=env.tool_sources,
                helpers=env.helpers,
                bridge_token=bridge_token,
                runner_data_dir="/data",
            )
            server = await _bridge.start_bridge(sock_path, env.helpers, bridge_token)

            stdout, error, wrote_result = await self._run_container(
                runtime, tmpdir, run_id
            )
            if not wrote_result:
                # No result.json => the container never completed its runner (OOM/timeout/daemon)
                raise SandboxFatalError(
                    "container produced no result (likely OOM/timeout/daemon death)"
                )
            return SandboxResult(updated_vars={}, stdout=stdout, error=error)
        except SandboxFatalError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Any other failure once we've committed to the container path (bridge/subprocess
            # error, resource exhaustion under load) is an irreversible infra failure, not a
            # code rejection — surface it as fatal (ORCH-1.29), never as a wrong-typed escape.
            raise SandboxFatalError(
                f"docker sandbox failed ({type(exc).__name__})"
            ) from exc
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
            "--read-only",
            "--ipc",
            "none",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--ulimit",
            "nofile=128:128",
            "--ulimit",
            "fsize=8388608:8388608",
            "-v",
            f"{tmpdir}:/data:rw",
            self.image,
            "python",
            "/data/runner.py",
        ]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=self.timeout_secs)
        except TimeoutError:
            await self._kill_container(runtime, name)
            with contextlib.suppress(Exception):
                await proc.wait()

        stdout, error, wrote = _bridge.read_result(tmpdir)
        if wrote:
            return stdout, error, True
        return "", None, False

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
