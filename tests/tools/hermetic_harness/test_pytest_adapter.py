"""GOC-38-W04 pytest adapter tests, run against real subprocesses.

The second half of this file is the lane's required known-bad proof: a
deliberately vacuous run (an empty directory with no collectible tests) run
through the actual adapter -- not a pure-function stand-in -- must produce a
non-green, VACUOUS-outcome, schema-valid envelope.

These tests deliberately launch the REPO's own real, uv-managed ``.venv``
rather than synthesizing a fake venv directory wrapped in an extra symlink
hop: an experiment while building this suite confirmed that an extra symlink
layer around a uv venv's ``bin/python3`` breaks CPython's own venv detection
(``sys.prefix`` resolves to the base interpreter's install, not the venv,
and site-packages/pygments goes missing) -- i.e. it reproduces the exact
"realpath resolves OUT of the venv" defect class this harness exists to
catch, rather than exercising a normal invocation. So the "correct
interpreter" fixture is the real ``.venv`` untouched, and the "wrong
interpreter" fixture is a genuinely different, real interpreter
(``/usr/bin/python3``), not a synthesized stand-in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.hermetic_harness.envelope import validate_envelope
from scripts.hermetic_harness.manifest import build_manifest
from scripts.hermetic_harness.pytest_adapter import (
    normalize_outcome,
    parse_assertion_summary,
    parse_collection,
    run_pytest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_VENV = REPO_ROOT / ".venv"
REAL_LOCKFILE = REPO_ROOT / "uv.lock"
SYSTEM_PYTHON3 = Path("/usr/bin/python3")

_venv_available = (REAL_VENV / "pyvenv.cfg").is_file() and REAL_LOCKFILE.is_file()
_system_python_available = SYSTEM_PYTHON3.is_file()


def test_parse_collection_handles_plain_and_deselected():
    assert parse_collection("3 tests collected in 0.01s") == (3, 0)
    assert parse_collection("5 tests collected (2 deselected) in 0.01s") == (5, 2)
    assert parse_collection("no tests ran in 0.00s") == (0, 0)
    assert parse_collection("") == (0, 0)
    assert parse_collection("1 test collected in 0.24s") == (1, 0)


def test_parse_assertion_summary():
    line = "===== 3 failed, 5 passed, 1 skipped, 2 errors in 1.23s ====="
    summary = parse_assertion_summary(line)
    assert summary == {"passed": 5, "failed": 3, "errors": 2, "skipped": 1}


def test_normalize_outcome_maps_exit_5_to_vacuous():
    assert normalize_outcome(5, timed_out=False, collection_count=0) == "VACUOUS"
    assert normalize_outcome(0, timed_out=False, collection_count=3) == "PASSED"
    assert normalize_outcome(1, timed_out=False, collection_count=3) == "FAILED"
    assert normalize_outcome(None, timed_out=True, collection_count=0) == "TIMEOUT"


@pytest.mark.skipif(not _venv_available, reason="repo's real .venv/uv.lock not present in this checkout")
def test_run_pytest_end_to_end_real_pass_is_green(tmp_path):
    fixture_dir = tmp_path / "fixture_tests"
    fixture_dir.mkdir()
    (fixture_dir / "test_ok.py").write_text("def test_trivially_true():\n    assert 1 == 1\n")

    manifest = build_manifest(
        repo="agent-utilities",
        repo_path=REPO_ROOT,
        test_paths=[str(fixture_dir)],
        lockfile_path=REAL_LOCKFILE,
        venv_path=REAL_VENV,
        timeout_seconds=60,
        grace_seconds=10,
    )
    envelope = run_pytest(
        manifest,
        repo_path=REPO_ROOT,
        branch="goc/goc-38-hermetic-evidence",
        dirty=True,
        python_path=REAL_VENV / "bin" / "python3",
        stdout_dir=tmp_path / "evidence",
    )
    validate_envelope(envelope)  # schema-valid by construction, re-checked here
    assert envelope["collection"]["collection_count"] == 1
    assert envelope["environment"]["venv_identity_match"] is True
    assert envelope["verdict"]["outcome"] == "PASSED"
    assert envelope["verdict"]["green"] is True
    assert envelope["verdict"]["falsifiable"] is True
    assert envelope["resources"]["survivor_check"]["clean"] is True


@pytest.mark.skipif(not _venv_available, reason="repo's real .venv/uv.lock not present in this checkout")
def test_run_pytest_end_to_end_vacuous_run_is_never_green(tmp_path):
    """KNOWN-BAD PROOF (real subprocess, not a stand-in): an empty directory
    with zero collectible tests must come back FAILED/VACUOUS, never green,
    even though pytest itself may exit non-fatally and produce clean-looking
    output. This is the exact 'measured nothing but reported confidently'
    shape recorded for ~20 gate helpers hit by the git-ls-files
    root-relative-path defect."""
    empty_dir = tmp_path / "empty_tests"
    empty_dir.mkdir()

    manifest = build_manifest(
        repo="agent-utilities",
        repo_path=REPO_ROOT,
        test_paths=[str(empty_dir)],
        lockfile_path=REAL_LOCKFILE,
        venv_path=REAL_VENV,
        timeout_seconds=60,
        grace_seconds=10,
    )
    envelope = run_pytest(
        manifest,
        repo_path=REPO_ROOT,
        branch="goc/goc-38-hermetic-evidence",
        dirty=True,
        python_path=REAL_VENV / "bin" / "python3",
        stdout_dir=tmp_path / "evidence",
    )
    validate_envelope(envelope)
    assert envelope["collection"]["collection_count"] == 0
    assert envelope["verdict"]["green"] is False
    assert envelope["verdict"]["outcome"] == "VACUOUS"
    assert envelope["verdict"]["falsifiable"] is False
    assert any("zero tests collected" in r for r in envelope["verdict"]["reasons"])


@pytest.mark.skipif(
    not (_venv_available and _system_python_available),
    reason="repo's real .venv or /usr/bin/python3 not present in this checkout",
)
def test_run_pytest_end_to_end_wrong_interpreter_is_never_green(tmp_path):
    """KNOWN-BAD PROOF: manifest pins the repo's real venv, but the adapter
    is pointed at a genuinely different, real interpreter (the system
    python3) to run -- the shape of `uv run pytest` silently resolving the
    system interpreter instead of the project's. venv_identity_match must
    come back False and the verdict must not be green, independent of
    whether the wrong interpreter happens to pass the tests it can run."""
    fixture_dir = tmp_path / "fixture_tests"
    fixture_dir.mkdir()
    (fixture_dir / "test_ok.py").write_text("def test_trivially_true():\n    assert 1 == 1\n")

    manifest = build_manifest(
        repo="agent-utilities",
        repo_path=REPO_ROOT,
        test_paths=[str(fixture_dir)],
        lockfile_path=REAL_LOCKFILE,
        venv_path=REAL_VENV,
        timeout_seconds=60,
        grace_seconds=10,
    )
    envelope = run_pytest(
        manifest,
        repo_path=REPO_ROOT,
        branch="goc/goc-38-hermetic-evidence",
        dirty=True,
        python_path=SYSTEM_PYTHON3,  # deliberately the WRONG interpreter
        stdout_dir=tmp_path / "evidence",
    )
    validate_envelope(envelope)
    assert envelope["environment"]["venv_identity_match"] is False
    assert envelope["verdict"]["green"] is False
    assert any("interpreter identity" in r for r in envelope["verdict"]["reasons"])
