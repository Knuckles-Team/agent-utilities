"""CONCEPT:OS-5.33 — Docker/Podman developer-workspace backend (isolation tier).

Unlike :class:`~agent_utilities.rlm.sandboxes.docker_backend.DockerSandbox` — which spins up a
fresh ``--rm`` container *per snippet* — a developer workspace needs a **long-lived** container
whose filesystem, installed deps, and build artifacts persist across many actions. We reuse that
backend's proven hardening flags (``--cap-drop ALL``, ``--security-opt no-new-privileges``,
``--memory``, ``--pids-limit``, ``--cpus``) but:

* start the container **detached** running ``sleep infinity`` and exec actions into it via
  ``docker exec`` (state persists), and
* bind-mount a host dir to ``/workspace`` so file read/write/edit happen host-side on
  ``self.root`` with no in-container shim, and
* drop ``--network none`` (cloning repos, installing deps, and running tests need egress) — the
  mutating actions that use the network are still gated by ``ActionPolicy`` (OS-5.24) upstream.

A class-level registry + :meth:`reap_idle` guards against leaked containers (e.g. a crashed run
that never called :meth:`stop`).
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path


class DockerWorkspace:
    name = "docker"
    workdir = "/workspace"

    # run_id -> (container_name, runtime, last_used_ts) for leak reaping.
    _REGISTRY: dict[str, tuple[str, str, float]] = {}

    def __init__(
        self,
        *,
        run_id: str | None = None,
        image: str = "python:3.11-slim",
        memory: str = "2g",
        cpus: str = "2.0",
        pids_limit: int = 1024,
        network: str = "bridge",
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.network = network
        self.root = Path(tempfile.mkdtemp(prefix=f"au-ws-{self.run_id}-"))
        self.container = f"au-ws-{self.run_id}"
        self._runtime: str | None = None

    # ── availability ──────────────────────────────────────────────────────────
    def is_available(self) -> bool:
        return self._resolve_runtime() is not None

    def _resolve_runtime(self) -> str | None:
        if self._runtime is None:
            for rt in ("docker", "podman"):
                if shutil.which(rt) is None:
                    continue
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
        return self._runtime

    # ── lifecycle ──────────────────────────────────────────────────────────────
    async def start(self) -> None:
        runtime = self._resolve_runtime()
        if runtime is None:
            raise RuntimeError("docker/podman runtime unavailable for DockerWorkspace")
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o777)  # nosec B103 - container uid may differ from host uid
        argv = [
            runtime,
            "run",
            "-d",
            "--rm",
            "--name",
            self.container,
            "--network",
            self.network,
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
            f"{self.root}:/workspace:rw",
            "-w",
            "/workspace",
            self.image,
            "sleep",
            "infinity",
        ]
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"failed to start workspace container: {(out or b'').decode(errors='replace')[:300]}"
            )
        self._REGISTRY[self.run_id] = (self.container, runtime, time.time())

    async def exec_shell(
        self, script: str, env: dict[str, str], timeout: float
    ) -> tuple[int, str, str]:
        runtime = self._runtime or "docker"
        env_flags: list[str] = []
        for k, v in env.items():
            env_flags += ["-e", f"{k}={v}"]
        argv = [runtime, "exec", *env_flags, self.container, "bash", "-c", script]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            return 124, "", f"command timed out after {timeout}s"
        finally:
            if self.run_id in self._REGISTRY:
                name, rt, _ = self._REGISTRY[self.run_id]
                self._REGISTRY[self.run_id] = (name, rt, time.time())
        return (
            proc.returncode or 0,
            out.decode(errors="replace"),
            err.decode(errors="replace"),
        )

    async def stop(self) -> None:
        runtime = self._runtime or "docker"
        with contextlib.suppress(Exception):
            killer = await asyncio.create_subprocess_exec(
                runtime,
                "rm",
                "-f",
                self.container,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        self._REGISTRY.pop(self.run_id, None)
        shutil.rmtree(self.root, ignore_errors=True)

    def exposed_url(self, port: int) -> str:
        return f"http://localhost:{port}"

    # ── leak reaping ────────────────────────────────────────────────────────────
    @classmethod
    def reap_idle(cls, max_idle_secs: float = 3600.0) -> list[str]:
        """Force-remove workspace containers idle longer than ``max_idle_secs``.

        Wireable from a maintenance daemon. Returns the run_ids reaped.
        """
        now = time.time()
        reaped: list[str] = []
        for run_id, (name, runtime, last) in list(cls._REGISTRY.items()):
            if now - last <= max_idle_secs:
                continue
            with contextlib.suppress(Exception):
                subprocess.run(
                    [runtime, "rm", "-f", name], capture_output=True, timeout=30
                )
            cls._REGISTRY.pop(run_id, None)
            reaped.append(run_id)
        return reaped
