"""Capability G proof: background-run wrapper (the journal-vs-file incident, "incident 9").

Incident (occurred twice): work was launched as
`systemd-run ... /bin/bash -c "cmd"` WITHOUT the output redirect inside the
quoted command. Output went to the systemd JOURNAL, not the log file the
operator tailed, so the run looked dead/hung when it was actually alive.

Proves: (1) run_background() builds the redirect INSIDE the command handed
to bash -c (a string-shape assertion against the actual systemd-run argv);
(2) end-to-end on this host (which has a working `systemd-run --user`
session), the launched unit's real output lands in the file poll() reads,
not nowhere.
"""

from __future__ import annotations

import shutil
import time

import pytest

from agent_utilities.measurement.background import poll, run_background, wait_for_unit

pytestmark = pytest.mark.skipif(
    shutil.which("systemd-run") is None, reason="no systemd-run on PATH in this environment"
)


def test_redirect_is_built_inside_the_bash_c_command(monkeypatch, tmp_path):
    """Assert the redirect placement directly against the argv systemd-run
    would receive -- the exact fix for the journal-vs-file incident."""
    captured_argv = {}

    class FakeCompleted:
        returncode = 0

    def fake_run(argv, **kwargs):
        captured_argv["argv"] = argv
        return FakeCompleted()

    monkeypatch.setattr("agent_utilities.measurement.background.subprocess.run", fake_run)

    result = run_background("echo hello", unit_name="test-unit-xyz", log_dir=tmp_path)

    argv = captured_argv["argv"]
    assert argv[0].endswith("systemd-run")
    assert argv[1:4] == ["--user", "--collect", "--unit=test-unit-xyz"]
    assert argv[-2] == "-c"
    inner_cmd = argv[-1]
    # The redirect must be PART of the string handed to `bash -c`, not
    # appended after the systemd-run invocation -- and it must wrap the
    # WHOLE command (not just its last `&&`-joined stage; see the
    # background.py docstring for why the `{ ...; }` grouping matters).
    assert inner_cmd.startswith("{ echo hello ; } >")
    assert str(result.log_path) in inner_cmd
    assert "2>&1" in inner_cmd


def test_end_to_end_output_lands_in_the_log_file_not_nowhere(tmp_path):
    """Real systemd-run, real unit, real file -- proves the fix actually
    works on this host, not just that the string looks right."""
    unit = f"measurement-harness-test-{int(time.time())}"
    result = run_background(
        "echo provenance-harness-canary && sleep 0.2",
        unit_name=unit,
        log_dir=tmp_path,
    )
    assert result.launch_returncode == 0

    finished = wait_for_unit(unit, timeout_s=15.0, poll_interval_s=0.2)
    assert finished, "unit did not finish within timeout"

    contents = poll(result.log_path)
    assert "provenance-harness-canary" in contents


def test_poll_of_nonexistent_log_returns_empty_not_an_error(tmp_path):
    assert poll(tmp_path / "never-written.log") == ""
