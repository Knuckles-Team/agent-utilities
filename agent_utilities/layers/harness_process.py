"""Child-process execution for CLI harness adapters.

One bounded launch: pinned argv, a required working directory, prompt on
stdin, a wall-clock timeout that kills the whole process group, and an
allowlisted child environment. The parent's tokens and unrelated settings never
cross into the child implicitly; an adapter passes what the child needs.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from agent_utilities.layers.harness_port import DiffStat

logger = logging.getLogger(__name__)

#: Parent variables a CLI harness needs to find itself and its own login state.
INHERITED_CHILD_ENV: frozenset[str] = frozenset(
    {
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
    }
)
#: Retained tail of a child's stderr for typed error messages.
STDERR_TAIL_CHARS = 8_192
#: Wall-clock bound on the read-only diff probe.
DIFF_TIMEOUT_S = 30.0

_SHORTSTAT = re.compile(
    r"(?P<files>\d+) files? changed"
    r"(?:, (?P<ins>\d+) insertions?\(\+\))?"
    r"(?:, (?P<dels>\d+) deletions?\(-\))?"
)


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """What one child run produced."""

    exit_code: int | None
    stdout: str
    stderr_tail: str
    timed_out: bool = False


def child_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The allowlisted parent environment plus the run's explicit variables."""
    inherited = {k: v for k, v in os.environ.items() if k in INHERITED_CHILD_ENV}
    return {**inherited, **(extra or {})}


async def _kill_group(proc: asyncio.subprocess.Process) -> None:
    """Kill the child and every process it started (its own session)."""
    if proc.returncode is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError as exc:
            logger.warning("harness child exited before the timeout kill: %s", exc)
    await proc.wait()


async def run_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    stdin_text: str,
    timeout_s: float,
    env: Mapping[str, str],
) -> ProcessResult:
    """Run ``argv`` in ``cwd`` once; kill its process group on timeout."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        env=dict(env),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(stdin_text.encode("utf-8")), timeout_s
        )
    except TimeoutError:
        await _kill_group(proc)
        return ProcessResult(exit_code=None, stdout="", stderr_tail="", timed_out=True)
    return ProcessResult(
        exit_code=proc.returncode,
        stdout=out.decode("utf-8", errors="replace"),
        stderr_tail=err.decode("utf-8", errors="replace")[-STDERR_TAIL_CHARS:],
    )


def parse_shortstat(text: str) -> DiffStat:
    """``git diff --shortstat`` output as a :class:`DiffStat` (empty = no change)."""
    match = _SHORTSTAT.search(text)
    if match is None:
        return DiffStat()
    return DiffStat(
        files_changed=int(match["files"]),
        insertions=int(match["ins"] or 0),
        deletions=int(match["dels"] or 0),
    )


async def read_diff_stat(workspace: Path) -> DiffStat | None:
    """Read-only change summary of ``workspace`` against its ``HEAD``.

    Returns ``None`` when the workspace is not a Git work tree or Git is
    absent. This never writes to the repository.
    """
    git = shutil.which("git")
    if git is None:
        return None
    result = await run_process(
        [git, "diff", "--shortstat", "HEAD"],
        cwd=workspace,
        stdin_text="",
        timeout_s=DIFF_TIMEOUT_S,
        env=child_environment(),
    )
    if result.exit_code != 0:
        return None
    return parse_shortstat(result.stdout)


__all__ = [
    "INHERITED_CHILD_ENV",
    "ProcessResult",
    "child_environment",
    "parse_shortstat",
    "read_diff_stat",
    "run_process",
]
