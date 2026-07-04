"""CONCEPT:AU-OS.scaling.bridge-developer-workspace-mutating — Local (host-subprocess) workspace backend: the zero-infra floor.

Mirrors the role :class:`~agent_utilities.rlm.sandboxes.local_backend.LocalSandbox` plays for the
RLM tier — always available, no isolation. The runtime works out-of-the-box (no Docker daemon)
so the SWE loop and tests run anywhere; escalate to :class:`~.docker_workspace.DockerWorkspace`
when isolation is required.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import tempfile
from pathlib import Path


class LocalWorkspace:
    name = "local"

    def __init__(self, root: str | Path | None = None) -> None:
        self._owns_root = root is None
        self.root = (
            Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="au-ws-"))
        )
        # For the local backend the execution-context path IS the host path.
        self.workdir = str(self.root)

    def is_available(self) -> bool:
        return shutil.which("bash") is not None

    async def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    async def exec_shell(
        self, script: str, env: dict[str, str], timeout: float
    ) -> tuple[int, str, str]:
        merged = {**os.environ, **env}
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            script,
            cwd=str(self.root),
            env=merged,
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
        return (
            proc.returncode or 0,
            out.decode(errors="replace"),
            err.decode(errors="replace"),
        )

    async def stop(self) -> None:
        if self._owns_root:
            shutil.rmtree(self.root, ignore_errors=True)

    def exposed_url(self, port: int) -> str:
        return f"http://localhost:{port}"
