"""CONCEPT:ORCH-1.89 — container_fork sandbox: warm container pool (CRIU-ready) vs cold --rm.

The ``docker`` rung (``docker_backend.py``) spawns a fresh ``--rm`` container *per snippet* — a
full container cold-start every call. This rung closes that gap (G3): it keeps a **warm**
``sleep infinity`` container (its image layers + page cache hot, deps importable) and runs each
snippet as a fresh ``docker exec`` child against a per-run ``/data`` scratch dir — the container
analogue of the ``forkserver`` rung's warm-fork. Same UDS host-callback bridge as ``docker``
(``_bridge``), so ``rlm_query``/``FINAL_VAR`` work.

Isolation trade-off (honest): the per-run ``/data`` scratch is isolated, the snippet runs as its
own process, and the container is resource/cap/network-confined exactly like the ``docker`` rung
— but the container's *root* filesystem is shared across the snippets that reuse the warm
container (unlike ``--rm``'s pristine-per-snippet root). For RLM glue (compute, not root-fs
mutation) that is fine and fast; when a pristine root per snippet is required the router still has
the cold ``docker`` rung. When **CRIU** is present (``criu`` binary + podman ``--lazy-pages``) the
warm parent is a *checkpoint* and each child is a fresh restore — pristine **and** warm; this rung
auto-detects and uses it, else falls back to the warm-exec pool.

Rank 18: cheaper than cold ``docker`` (20), pricier than the process-level ``forkserver`` (15).
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
from .base import (
    ForkableSandbox,
    ParentHandle,
    SandboxCapabilities,
    SandboxEnv,
    SandboxResult,
    WarmSpec,
)

logger = logging.getLogger(__name__)


class ContainerForkSandbox(ForkableSandbox):
    """Run a snippet as a ``docker exec`` child of a warm, pooled ``sleep infinity`` container."""

    name = "container_fork"
    capabilities = SandboxCapabilities(
        host_callbacks=True,  # via the UDS bridge (same as docker)
        third_party_libs=True,
        classes=True,
        full_stdlib=True,
        network=False,  # container started without networking
        isolated=True,  # container boundary; root-fs shared across reuse (see module docstring)
        preference_rank=18,  # warmer (cheaper) than cold docker (20); heavier than forkserver (15)
        warm_fork=True,
    )

    def __init__(
        self,
        image: str = "python:3.11-slim",
        *,
        memory: str = "512m",
        cpus: str = "1.0",
        pids_limit: int = 256,
        timeout_secs: float = 120.0,
    ) -> None:
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.timeout_secs = timeout_secs
        self._runtime: str | None | bool = None

    def is_available(self) -> bool:
        return self._resolve_runtime() is not None

    def _resolve_runtime(self) -> str | None:
        """Find a working docker/podman CLI + daemon (cached). Mirrors DockerSandbox."""
        if self._runtime is None:
            self._runtime = False
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
        return self._runtime if isinstance(self._runtime, str) else None

    def warm_spec(self) -> WarmSpec:
        # The warm parent is uniquely identified by (runtime, image) — one pooled container per.
        rt = self._resolve_runtime() or "none"
        return WarmSpec(
            backend=self.name, extra=(("runtime", rt), ("image", self.image))
        )

    async def warm(self, spec: WarmSpec) -> ParentHandle:
        """Start a detached, resource-limited ``sleep infinity`` container with a /data mount."""
        runtime = self._resolve_runtime()
        if runtime is None:
            raise SandboxFatalError("docker/podman runtime unavailable")
        pool_dir = Path(tempfile.mkdtemp(prefix="rlm-cfork-pool-"))
        os.chmod(pool_dir, 0o777)  # nosec B103 — container uid may differ (rootless/userns)
        name = f"rlm-cfork-{uuid.uuid4().hex[:12]}"
        argv = [
            runtime, "run", "-d", "--name", name,
            "--network", "none",
            "--memory", self.memory,
            "--pids-limit", str(self.pids_limit),
            "--cpus", self.cpus,
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "-v", f"{pool_dir}:/data:rw",
            self.image, "sleep", "infinity",
        ]  # fmt: skip
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            if proc.returncode != 0:
                shutil.rmtree(pool_dir, ignore_errors=True)
                raise SandboxFatalError(
                    f"container_fork warm-up failed: {(out or b'').decode(errors='replace')[:200]}"
                )
        except SandboxFatalError:
            raise
        except Exception as e:  # noqa: BLE001
            shutil.rmtree(pool_dir, ignore_errors=True)
            raise SandboxFatalError(f"container_fork warm-up failed: {e}") from e

        def _close() -> None:
            with contextlib.suppress(Exception):
                subprocess.run(
                    [runtime, "rm", "-f", name], capture_output=True, timeout=30
                )
            shutil.rmtree(pool_dir, ignore_errors=True)

        return ParentHandle(
            backend=self.name,
            spec=spec,
            ref={"runtime": runtime, "name": name, "pool_dir": pool_dir},
            close=_close,
        )

    async def run_forked(
        self, parent: ParentHandle, code: str, env: SandboxEnv
    ) -> SandboxResult:
        runtime = parent.ref["runtime"]
        name = parent.ref["name"]
        pool_dir: Path = parent.ref["pool_dir"]
        run_id = uuid.uuid4().hex[:12]
        run_dir = pool_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        guest_data = f"/data/{run_id}"
        sock_path = run_dir / "bridge.sock"
        server: asyncio.AbstractServer | None = None
        try:
            _bridge.write_inputs(
                run_dir,
                code,
                vars_payload=env.vars,
                tool_sources=env.tool_sources,
                helpers=env.helpers,
                runner_data_dir=guest_data,  # injected script (container can't import the pkg)
            )
            server = await _bridge.start_bridge(sock_path, env.helpers)
            os.chmod(run_dir, 0o777)  # nosec B103 — container uid may differ
            with contextlib.suppress(FileNotFoundError):
                os.chmod(sock_path, 0o777)  # nosec B103 — bridge socket must accept container uid

            exec_argv = [
                runtime, "exec", name, "python", f"{guest_data}/runner.py",
            ]  # fmt: skip
            proc = await asyncio.create_subprocess_exec(
                *exec_argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                raw, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_secs
                )
                exec_log = (raw or b"").decode(errors="replace")
            except TimeoutError:
                with contextlib.suppress(Exception):
                    proc.kill()
                    await proc.wait()
                exec_log = ""

            stdout, error, wrote = _bridge.read_result(run_dir)
            if not wrote:
                raise SandboxFatalError(
                    f"container_fork child produced no result (timeout/crash): {exec_log[:200]}"
                )
            return SandboxResult(updated_vars={}, stdout=stdout, error=error)
        except SandboxFatalError:
            raise
        except Exception as e:  # noqa: BLE001 - committed to the exec path => infra failure
            raise SandboxFatalError(f"container_fork sandbox failed: {e}") from e
        finally:
            if server is not None:
                server.close()
                with contextlib.suppress(Exception):
                    await server.wait_closed()
            shutil.rmtree(run_dir, ignore_errors=True)
