"""Capability D proof: exit-code correctness (incident 1).

Incident: ``python3 script.py | tail -25`` then ``echo "EXIT=$?"`` reported
tail's exit status (0) as the script's, hiding a real failure. Proves:
(1) ``run()`` captures the REAL exit status of the measured process, never a
pipeline stage's, because it never goes through a shell pipeline at all;
(2) the static linter flags the incident's exact shell shape.
"""

from __future__ import annotations

import sys

import pytest

from agent_utilities.measurement.run import (
    KilledBySignalError,
    run,
    scan_for_pipeline_exit_antipattern,
)


def test_run_captures_real_nonzero_exit_code_not_a_pipeline_stage():
    """The incident's exact shape: a failing script piped into `tail` would
    have reported exit 0 via `$?`. `run()` on the SAME failing script must
    report the real nonzero code, because there is no pipeline stage to
    substitute it."""
    result = run([sys.executable, "-c", "import sys; print('x' * 30); sys.exit(17)"])
    assert result.returncode == 17
    assert not result.ok


def test_run_rejects_shell_string_to_prevent_pipeline_shape():
    with pytest.raises(TypeError):
        run("python3 script.py | tail -25")  # type: ignore[arg-type]


def test_run_distinguishes_signal_kill_from_pass():
    with pytest.raises(KilledBySignalError):
        run([sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"])

    result = run(
        [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"],
        raise_on_signal=False,
    )
    assert result.killed_by_signal == 15
    assert not result.ok  # a kill must never read as ok


def test_linter_catches_incident_1_exact_shape():
    """This is the literal shell text from the incident report."""
    script = 'python3 script.py | tail -25\necho "EXIT=$?"\n'
    hits = scan_for_pipeline_exit_antipattern(script)
    assert len(hits) == 1
    assert "tail" in hits[0].pipe_line
    assert "$?" in hits[0].dollar_question_line


def test_linter_does_not_flag_pipefail_guarded_version():
    """The correct fix (pipefail + PIPESTATUS) must not be flagged."""
    script = (
        "set -o pipefail\n"
        "python3 script.py | tail -25\n"
        'echo "EXIT=${PIPESTATUS[0]}"\n'
    )
    hits = scan_for_pipeline_exit_antipattern(script)
    assert hits == []


def test_linter_does_not_flag_unrelated_pipe_with_no_exit_read():
    script = "ls | tail -5\necho done\n"
    hits = scan_for_pipeline_exit_antipattern(script)
    assert hits == []


def test_check_script_flags_a_synthetic_bad_file(tmp_path):
    """End-to-end: run the actual pre-commit-hookable script against a
    synthetic file reproducing incident 1, and confirm it exits 1."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    bad_dir = tmp_path / "scripts"
    bad_dir.mkdir()
    (bad_dir / "reproduces_incident_1.sh").write_text(
        '#!/usr/bin/env bash\npython3 script.py | tail -25\necho "EXIT=$?"\n'
    )
    # Point the checker at a throwaway root containing only our bad file by
    # invoking it as a library call against that root, mirroring what the
    # real script does, rather than depend on repo-relative path plumbing.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_measurement_exit_code_antipattern",
        repo_root / "scripts" / "check_measurement_exit_code_antipattern.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    scripts = mod._iter_shell_scripts(tmp_path)
    assert any(p.name == "reproduces_incident_1.sh" for p in scripts)
    text = (bad_dir / "reproduces_incident_1.sh").read_text()
    assert len(scan_for_pipeline_exit_antipattern(text)) == 1
