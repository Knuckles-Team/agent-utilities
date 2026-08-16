"""Background-run wrapper (measurement harness, capability G).

CONCEPT:AU-OS.measurement.background-run

Direct response to a pair of incidents where long-running work was launched
as ``systemd-run --user --collect --unit=<name> /bin/bash -c "cmd"`` WITHOUT
putting the output redirect INSIDE the quoted command. Without an inner
redirect, the child process's stdout/stderr go wherever systemd sends
unit output by default — the systemd JOURNAL, not any file — while the
operator tails a log file expecting output and sees nothing, and reads the
empty file as the run having died or hung. The run was alive the whole
time; the wrong place was being watched.

:func:`run_background` is the one function that launches long work
correctly every time: it always builds the inner command as
``cmd > log_path 2>&1`` and passes THAT whole string to ``bash -c``, so the
redirect happens inside the process systemd is supervising, landing in a
real file :func:`poll` can read — never only in the journal.
"""

from __future__ import annotations

import dataclasses
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path


class SystemdRunUnavailableError(Exception):
    """Raised when ``systemd-run`` is not on PATH."""


@dataclasses.dataclass(frozen=True)
class BackgroundRun:
    unit_name: str
    log_path: Path
    launch_returncode: (
        int  # exit status of `systemd-run` itself (unit accepted, not unit's own exit)
    )


def run_background(
    cmd: str,
    *,
    unit_name: str | None = None,
    log_dir: Path | None = None,
    extra_systemd_run_args: list[str] | None = None,
) -> BackgroundRun:
    """Launch ``cmd`` (a shell command string) in a transient user systemd unit.

    The output redirect is built INSIDE the command handed to ``bash -c``
    (``cmd > log_path 2>&1``), never appended outside the
    ``systemd-run ... bash -c "..."`` invocation — that placement is the
    entire fix for the journal-vs-file incident this function exists to
    prevent. Redirecting outside would apply to ``systemd-run``'s own
    (near-empty) output, not the child process's.

    ``--collect`` (always passed) tells systemd to garbage-collect the
    transient unit's metadata once it exits, instead of leaving a stopped
    unit around forever — it does not affect where output goes.

    Returns a :class:`BackgroundRun` with the unit name and the log path to
    poll. Raises :class:`SystemdRunUnavailableError` if ``systemd-run`` is
    not available (e.g. no systemd user session) — never silently falls
    back to a different launcher that would change the run's semantics.
    """
    systemd_run = shutil.which("systemd-run")
    if systemd_run is None:
        raise SystemdRunUnavailableError(
            "systemd-run not found on PATH; run_background() requires a "
            "systemd user session (systemctl --user status) and will not "
            "silently substitute a different backgrounding mechanism"
        )

    unit = unit_name or f"measurement-{uuid.uuid4().hex[:12]}"
    log_dir = Path(log_dir) if log_dir is not None else Path("/var/tmp")  # nosec B108 — /var/tmp default, not tmpfs-backed /tmp (see tmp-is-tmpfs incident); overridable via log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{unit}.log"

    # The redirect MUST be part of the string passed to `bash -c`, not
    # appended after the systemd-run argv — that is the fix, spelled out so
    # a future edit cannot "simplify" it back into the incident's shape.
    #
    # `cmd` is wrapped in a `{ ...; }` group before the redirect is
    # appended. Without the group, `cmd1 && cmd2 > log` binds the redirect
    # to `cmd2` ALONE (shell operator precedence), silently dropping
    # `cmd1`'s output even though the redirect textually looks like it
    # covers the whole command — a second, subtler version of the same
    # journal-vs-file class of bug this function exists to prevent.
    inner_cmd = f"{{ {cmd} ; }} > {shlex.quote(str(log_path))} 2>&1"

    argv = [
        systemd_run,
        "--user",
        "--collect",
        f"--unit={unit}",
        *(extra_systemd_run_args or []),
        "/bin/bash",
        "-c",
        inner_cmd,
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    return BackgroundRun(
        unit_name=unit, log_path=log_path, launch_returncode=proc.returncode
    )


def poll(log_path: Path, *, tail_lines: int | None = None) -> str:
    """Read the current contents of ``log_path`` (the real file, never the journal).

    Returns ``""`` if the file does not exist yet (the unit may not have
    started writing). With ``tail_lines`` set, returns only the last N
    lines — still a real read of the file, not a piped ``tail`` whose exit
    status could be mistaken for the measured process's (see
    :mod:`.run` capability D for that specific antipattern).
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return ""
    text = log_path.read_text(errors="replace")
    if tail_lines is None:
        return text
    lines = text.splitlines()
    return "\n".join(lines[-tail_lines:])


def wait_for_unit(
    unit_name: str, *, timeout_s: float = 300.0, poll_interval_s: float = 1.0
) -> bool:
    """Block until the transient user unit ``unit_name`` is no longer active.

    Returns ``True`` if the unit finished within ``timeout_s``, ``False`` on
    timeout (the unit may still be running). Uses ``systemctl --user
    is-active`` rather than parsing the journal, consistent with this
    module's file-based (not journal-based) philosophy for anything a
    caller needs to act on programmatically.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        proc = subprocess.run(
            ["systemctl", "--user", "is-active", unit_name],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.stdout.strip() != "active":
            return True
        time.sleep(poll_interval_s)
    return False
