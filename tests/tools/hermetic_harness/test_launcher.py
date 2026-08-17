"""GOC-38-W03 process/resource control tests.

Known-bad proof required by the lane: a single-PID kill would leave a
grandchild (and an ignore-SIGTERM child) alive. These tests prove the
process-GROUP kill takes the whole tree out and that the launcher reports
that positively (``survivor_check.clean``) rather than assuming clean exit
from the parent's return code alone.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from scripts.hermetic_harness.launcher import ProcessGroupLauncher


def test_normal_command_completes_without_timeout():
    launcher = ProcessGroupLauncher(deadline_seconds=5.0, grace_seconds=1.0, poll_interval=0.02)
    result = launcher.run([sys.executable, "-c", "print('ok')"], cwd=Path.cwd(), env=dict(os.environ))
    assert not result.timed_out
    assert not result.escalated
    assert result.exit_code == 0
    assert result.survivors_after_kill == []
    assert b"ok" in result.stdout


def test_signal_alone_does_not_prove_survivor_clean_but_group_kill_does():
    """A child that ignores SIGTERM AND forks a grandchild that also ignores
    SIGTERM: a bare `os.kill(leader_pid, SIGTERM)` (signal-only, single
    process) would leave both alive past the deadline -- the documented
    `--timeout=300` failure mode. The launcher must escalate to SIGKILL on
    the whole GROUP and prove zero survivors remain.
    """
    grandchild_script = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(30)"
    )
    child_script = (
        "import signal, subprocess, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"subprocess.Popen([sys.executable, '-c', {grandchild_script!r}]); "
        "time.sleep(30)"
    )
    launcher = ProcessGroupLauncher(deadline_seconds=0.5, grace_seconds=0.5, poll_interval=0.02)
    start = time.monotonic()
    result = launcher.run([sys.executable, "-c", child_script], cwd=Path.cwd(), env=dict(os.environ))
    elapsed = time.monotonic() - start

    assert result.timed_out is True
    assert result.escalated is True, "SIGTERM-ignoring tree must force SIGKILL escalation"
    assert "SIGKILL->group" in result.signal_sequence
    assert result.survivors_before_kill != [], "expected at least parent+grandchild alive at deadline"
    assert result.survivors_after_kill == [], "process-group kill must leave zero survivors"
    # Bounded: deadline (0.5) + grace (0.5) + small overhead, never anywhere
    # near the 30s sleep the script requested -- proves the kill actually
    # interrupted it rather than the process exiting on its own schedule.
    assert elapsed < 10.0


def test_shell_is_never_used_argv_is_exec_d_directly():
    launcher = ProcessGroupLauncher(deadline_seconds=5.0, grace_seconds=1.0)
    # A pipeline-only construct like `false | true` would exit 0 under a
    # shell (only the last stage's status survives in $?). Passed as a
    # single argv element with shell=False, exec(2) must look for a literal
    # binary named "false | true" (which does not exist) rather than a shell
    # interpreting it as a pipeline -- proving no shell/pipeline ever
    # mediates the exit-status provenance this envelope records.
    try:
        launcher.run(["false | true"], cwd=Path.cwd(), env=dict(os.environ))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError -- shell must never interpret argv")
