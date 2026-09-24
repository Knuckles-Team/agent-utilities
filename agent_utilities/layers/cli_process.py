"""Child-process launching for CLI harness adapters (Claude Code, Codex, Grok).

:class:`ProcessLauncher` is the one seam between an adapter and the operating
system. :class:`SubprocessLauncher` runs the real binary; the conformance kit's
recorded-transcript double implements the same protocol and replays captured
JSONL, so an adapter's parsing, sequencing and failure typing are exercised
identically either way.

The child environment is an explicit allowlist: the parent's credentials,
tokens and unrelated configuration are never inherited implicitly
(RF-ADR-010 §3 "no run inherits an implicit host or credential context").
"""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from agent_utilities.layers.contracts import HarnessNotInstalled, SandboxBoundaryError

#: Parent variables a CLI harness needs to locate itself and its own login
#: state; nothing else crosses into the child.
INHERITED_CHILD_ENV: tuple[str, ...] = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "XDG_RUNTIME_DIR",
    "NODE_EXTRA_CA_CERTS",
    "SSL_CERT_FILE",
)
#: Terminate grace period before SIGKILL on cancellation.
TERMINATE_GRACE_S = 5.0
#: Retained tail of a child's stderr for typed error messages.
STDERR_TAIL_BYTES = 8_192
#: Longest single JSONL line accepted from a harness.
MAX_LINE_BYTES = 16 * 1024 * 1024


@runtime_checkable
class LaunchedProcess(Protocol):
    """A running harness child: line stream, termination and exit status."""

    def lines(self) -> AsyncIterator[str]: ...

    async def terminate(self) -> None: ...

    async def wait(self) -> int: ...

    def stderr_tail(self) -> str: ...


@runtime_checkable
class ProcessLauncher(Protocol):
    """Resolves and launches a harness binary."""

    def resolve(self, binary: str) -> str: ...

    async def launch(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str],
        stdin_text: str,
    ) -> LaunchedProcess: ...


def child_environment(extra: Mapping[str, str]) -> dict[str, str]:
    """The allowlisted parent environment plus the run's explicit variables."""
    inherited = {
        key: value for key, value in os.environ.items() if key in INHERITED_CHILD_ENV
    }
    return {**inherited, **extra}


def confine(workspace: str | None, path: Path) -> Path:
    """``path`` resolved, refusing anything outside the leased workspace."""
    if workspace is None:
        raise SandboxBoundaryError("a CLI harness run requires a leased workspace")
    root = Path(workspace).resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise SandboxBoundaryError(f"{path} escapes the leased workspace")
    return resolved


async def _drain_tail(stream: asyncio.StreamReader | None, tail: bytearray) -> None:
    """Keep only the last :data:`STDERR_TAIL_BYTES` of a stream, so a chatty
    child can never block on a full stderr pipe."""
    if stream is None:
        return
    while chunk := await stream.read(65_536):
        tail.extend(chunk)
        del tail[:-STDERR_TAIL_BYTES]


@dataclass(slots=True)
class _Subprocess:
    process: asyncio.subprocess.Process
    stderr_buffer: bytearray
    stderr_task: asyncio.Task[None]

    async def lines(self) -> AsyncIterator[str]:
        stdout = self.process.stdout
        if stdout is None:
            return
        async for raw in stdout:
            yield raw.decode("utf-8", errors="replace").rstrip("\n")

    async def terminate(self) -> None:
        if self.process.returncode is not None:
            return
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=TERMINATE_GRACE_S)
        except TimeoutError:
            self.process.kill()
            await self.process.wait()

    async def wait(self) -> int:
        code = await self.process.wait()
        await self.stderr_task
        return code

    def stderr_tail(self) -> str:
        return bytes(self.stderr_buffer).decode("utf-8", errors="replace")


class SubprocessLauncher:
    """Launches the real harness binary with piped stdin/stdout/stderr."""

    def resolve(self, binary: str) -> str:
        found = shutil.which(binary)
        if found is None:
            raise HarnessNotInstalled(f"{binary!r} is not installed on this host")
        return found

    async def launch(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str],
        stdin_text: str,
    ) -> LaunchedProcess:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=dict(env),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=MAX_LINE_BYTES,
        )
        stdin = process.stdin
        if stdin is not None:
            stdin.write(stdin_text.encode("utf-8"))
            await stdin.drain()
            stdin.close()
        tail = bytearray()
        drain = asyncio.create_task(_drain_tail(process.stderr, tail))
        return _Subprocess(process=process, stderr_buffer=tail, stderr_task=drain)


__all__ = [
    "INHERITED_CHILD_ENV",
    "LaunchedProcess",
    "ProcessLauncher",
    "SubprocessLauncher",
    "child_environment",
    "confine",
]
