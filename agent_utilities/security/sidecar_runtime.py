"""CONCEPT:ORCH-1.35 / OS-5.5 — Sidecar runtime: typed process stamps + per-run UDS isolation.

Assimilated from open-design's sidecar-proto 5-field stamp (app/mode/namespace/ipc/source) and
socket-path isolation. Each run/agent gets a unique stamp that deterministically resolves to a UDS
path under ``$TMPDIR/agent-utilities/<namespace>/...`` so concurrent runs cannot cross-talk. POSIX
sockets first (the deployment target is Linux daemons/containers); Windows named pipes are deferred.

Pure stdlib; the stamp + path derivation is fully deterministic and unit-testable without spawning.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agent_utilities.core.config import setting


@dataclass(frozen=True, slots=True)
class ProcessStamp:
    """The 5-field identity of a sidecar process (data contract, not an implementation detail)."""

    app: str = "agent-utilities"
    mode: str = "run"
    namespace: str = "default"
    ipc: str = "uds"
    source: str = "engine"

    def key(self) -> str:
        """Stable, filesystem-safe identity key."""
        return f"{self.app}.{self.mode}.{self.namespace}.{self.ipc}.{self.source}"


def _runtime_root(namespace: str) -> Path:
    base = setting("AGENT_UTILITIES_RUNTIME_DIR") or os.path.join(
        tempfile.gettempdir(), "agent-utilities"
    )
    return Path(base) / namespace


def socket_path(stamp: ProcessStamp, run_id: str) -> Path:
    """Deterministic UDS path for ``(stamp, run_id)`` under the namespace's runtime root.

    Two runs in different namespaces (or with different ids) always resolve to different sockets, which
    is the isolation guarantee. The parent directory is created with private (0700) permissions.
    """
    root = _runtime_root(stamp.namespace) / "ipc"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    safe_run = run_id.replace("/", "_").replace(":", "_")
    return root / f"{stamp.source}.{safe_run}.sock"


class SidecarRuntime:
    """Allocates isolated UDS endpoints per run.

    The spawn/IPC transport is intentionally thin here; EPIC 2's mid-turn loop and any future
    out-of-process plugin execution build on these isolated endpoints. Allocation is the part that
    must be correct and is fully tested.
    """

    def __init__(self, stamp: ProcessStamp | None = None) -> None:
        self.stamp = stamp or ProcessStamp()
        self._allocated: dict[str, Path] = {}

    def allocate(self, run_id: str) -> Path:
        """Reserve and return the isolated socket path for ``run_id`` (idempotent per run)."""
        if run_id not in self._allocated:
            self._allocated[run_id] = socket_path(self.stamp, run_id)
        return self._allocated[run_id]

    def release(self, run_id: str) -> None:
        """Release a run's socket path (and unlink the socket file if present)."""
        path = self._allocated.pop(run_id, None)
        if path and path.exists():
            try:
                path.unlink()
            except OSError:
                pass

    def isolated_from(self, other: SidecarRuntime, run_id: str) -> bool:
        """True if this runtime's socket for ``run_id`` differs from ``other``'s (isolation check)."""
        return self.allocate(run_id) != other.allocate(run_id)
