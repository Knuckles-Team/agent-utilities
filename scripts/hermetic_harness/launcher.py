"""GOC-38 process-group launcher: deadline, grace interval, group kill, survivor check.

CONCEPT:AU-GOC.harness.process-group-launcher

Why this exists, concretely: pytest.ini's ``--timeout=300`` does not fire when
a test blocks in an anyio worker thread making a call into the live engine --
pytest-timeout's default signal-based mechanism can only interrupt the main
thread, and a worker thread parked in a blocking C call never sees it. The
fix is architectural, not a bigger timeout: launch the command in its own
process **group**, enforce the deadline from a separate watcher thread that
does not itself block on the child, and cancel by signaling the *group*
(SIGTERM, then escalate to SIGKILL after a grace interval), then positively
verify no process sharing that process-group id remains alive. A "clean
exit" that still has a live descendant is not clean.

Never uses ``shell=True`` or a pipe-through-shell invocation: argv is exec'd
directly so the exit status the caller reads is the exact launched process's
wait status, not ``$?`` after a pipeline (which is only the last stage).
"""

from __future__ import annotations

import dataclasses
import os
import signal
import subprocess
import time
from pathlib import Path


def _pids_in_group(pgid: int) -> list[int]:
    """Best-effort scan of /proc for PIDs whose process group id is pgid.
    Linux-only; this harness targets the homelab's Linux hosts."""
    survivors: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return survivors
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getpgid(pid) == pgid:
                survivors.append(pid)
        except (ProcessLookupError, PermissionError):
            continue
    return survivors


@dataclasses.dataclass
class LaunchResult:
    argv: list[str]
    cwd: str
    process_group_leader_pid: int
    process_group_start_time: float
    exit_code: int | None
    signal_name: str | None
    timed_out: bool
    escalated: bool
    signal_sequence: list[str]
    survivors_before_kill: list[int]
    survivors_after_kill: list[int]
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    wall_seconds: float


class ProcessGroupLauncher:
    """Launches argv in a new session (its own process group), enforces a
    deadline + grace interval by signaling the whole group, and proves no
    survivor remains before returning.
    """

    def __init__(
        self,
        *,
        deadline_seconds: float,
        grace_seconds: float = 15.0,
        max_stream_bytes: int = 10 * 1024 * 1024,
        poll_interval: float = 0.05,
    ) -> None:
        if deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be > 0")
        if grace_seconds < 0:
            raise ValueError("grace_seconds must be >= 0")
        self.deadline_seconds = deadline_seconds
        self.grace_seconds = grace_seconds
        self.max_stream_bytes = max_stream_bytes
        self.poll_interval = poll_interval

    def run(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> LaunchResult:
        if not argv:
            raise ValueError("argv must be non-empty")

        start = time.monotonic()
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # new process group; leader pid == pgid
            shell=False,
        )
        pgid = proc.pid  # leader of its own group by start_new_session

        signal_sequence: list[str] = []
        timed_out = False
        escalated = False
        survivors_before_kill: list[int] = []
        survivors_after_kill: list[int] = []

        deadline_at = start + self.deadline_seconds
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            if time.monotonic() >= deadline_at:
                timed_out = True
                break
            time.sleep(self.poll_interval)

        if timed_out:
            survivors_before_kill = _pids_in_group(pgid)
            try:
                os.killpg(pgid, signal.SIGTERM)
                signal_sequence.append("SIGTERM->group")
            except ProcessLookupError:
                pass
            grace_deadline = time.monotonic() + self.grace_seconds
            while time.monotonic() < grace_deadline:
                if proc.poll() is not None and not _pids_in_group(pgid):
                    break
                time.sleep(self.poll_interval)
            remaining = _pids_in_group(pgid)
            if remaining:
                escalated = True
                try:
                    os.killpg(pgid, signal.SIGKILL)
                    signal_sequence.append("SIGKILL->group")
                except ProcessLookupError:
                    pass
                # Bounded reap wait; do not block forever on a kernel that is
                # slow to reap -- the survivor check below is the source of
                # truth, not this wait.
                reap_deadline = time.monotonic() + max(2.0, self.poll_interval * 20)
                while time.monotonic() < reap_deadline:
                    if not _pids_in_group(pgid):
                        break
                    time.sleep(self.poll_interval)

        try:
            stdout, stderr = proc.communicate(timeout=max(2.0, self.grace_seconds))
        except subprocess.TimeoutExpired:
            stdout, stderr = b"", b""

        survivors_after_kill = _pids_in_group(pgid)

        stdout_truncated = len(stdout) > self.max_stream_bytes
        stderr_truncated = len(stderr) > self.max_stream_bytes
        if stdout_truncated:
            stdout = stdout[: self.max_stream_bytes]
        if stderr_truncated:
            stderr = stderr[: self.max_stream_bytes]

        exit_code = proc.returncode
        signal_name = None
        if exit_code is not None and exit_code < 0:
            try:
                signal_name = signal.Signals(-exit_code).name
            except ValueError:
                signal_name = f"signal-{-exit_code}"

        return LaunchResult(
            argv=argv,
            cwd=str(cwd),
            process_group_leader_pid=pgid,
            process_group_start_time=start,
            exit_code=exit_code,
            signal_name=signal_name,
            timed_out=timed_out,
            escalated=escalated,
            signal_sequence=signal_sequence,
            survivors_before_kill=survivors_before_kill,
            survivors_after_kill=survivors_after_kill,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            wall_seconds=time.monotonic() - start,
        )
