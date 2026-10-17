"""CONCEPT:AU-OS.scaling.bridge-developer-workspace-mutating — Host-subprocess developer workspace.

This backend is intentionally scoped to an operator-authorized developer workspace; it is not
an RLM code sandbox and does not claim an isolation boundary. Use
:class:`~.docker_workspace.DockerWorkspace` when workspace isolation is required.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .workspace import run_shell_process


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
        return await run_shell_process(
            ["bash", "-c", script],
            timeout,
            cwd=str(self.root),
            env={**os.environ, **env},
        )

    async def stop(self) -> None:
        if self._owns_root:
            shutil.rmtree(self.root, ignore_errors=True)

    def exposed_url(self, port: int) -> str:
        return f"http://localhost:{port}"
