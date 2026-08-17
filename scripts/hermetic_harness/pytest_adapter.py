"""GOC-38 pytest adapter (GOC-38-W04): normalizes a pytest invocation into the
shared evidence envelope while preserving native evidence.

CONCEPT:AU-GOC.harness.pytest-adapter

Two-phase run, deliberately not one:

1. **Collection phase** -- an authoritative ``--collect-only -q`` pass records
   ``collection_count``/``deselection_count`` *before* any assertion runs, so
   a run that silently measured an empty universe (the ``git -C <subdir>
   ls-files`` root-relative-path class of defect, which made ~20 gate
   helpers report a confident wrong verdict over zero real files) is visible
   as a distinct, checkable number rather than folded into a single exit
   code.
2. **Execution phase** -- the real run, launched through
   :class:`~scripts.hermetic_harness.launcher.ProcessGroupLauncher` so a
   hang is killed by process group rather than relying on
   ``pytest.ini``'s ``--timeout=300``, which does not fire for a test
   blocked in an anyio worker thread.

Both phases exec the venv's interpreter directly (``<venv>/bin/python3 -m
pytest ...``) -- never a bare ``pytest`` off ``$PATH`` and never through
``uv run`` without first pinning the interpreter, which is exactly how ``uv
run pytest`` has previously silently run the system pytest instead of the
project's.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .envelope import build_envelope
from .launcher import LaunchResult, ProcessGroupLauncher
from .manifest import Manifest, filtered_env, sha256_bytes, venv_identity_digest

_COLLECTED_RE = re.compile(
    r"(?P<collected>\d+) tests? collected(?: \((?P<deselected>\d+) deselected\))?"
)
_NO_TESTS_RE = re.compile(r"no tests ran")
_SUMMARY_COUNT_RE = re.compile(
    r"(?P<count>\d+) (?P<label>passed|failed|error|errors|skipped|deselected)"
)

# pytest's own documented exit codes (never inferred from a shell $?).
EXIT_OK = 0
EXIT_TESTS_FAILED = 1
EXIT_INTERRUPTED = 2
EXIT_INTERNAL_ERROR = 3
EXIT_USAGE_ERROR = 4
EXIT_NO_TESTS_COLLECTED = 5


def parse_collection(stdout: str) -> tuple[int, int]:
    """Returns (collection_count, deselection_count) from pytest's
    ``--collect-only -q`` output. 0/0 for "no tests ran" rather than raising
    -- a vacuous collection is data the envelope must carry, not an
    exception the adapter swallows."""
    if _NO_TESTS_RE.search(stdout):
        return 0, 0
    m = _COLLECTED_RE.search(stdout)
    if not m:
        return 0, 0
    collected = int(m.group("collected"))
    deselected = int(m.group("deselected") or 0)
    return collected, deselected


def parse_assertion_summary(stdout: str) -> dict[str, int]:
    summary = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for m in _SUMMARY_COUNT_RE.finditer(stdout):
        label = m.group("label")
        count = int(m.group("count"))
        if label == "passed":
            summary["passed"] = count
        elif label == "failed":
            summary["failed"] = count
        elif label in ("error", "errors"):
            summary["errors"] = count
        elif label == "skipped":
            summary["skipped"] = count
    return summary


def normalize_outcome(exit_code: int | None, timed_out: bool, collection_count: int) -> str:
    if timed_out:
        return "TIMEOUT"
    if exit_code is None:
        return "CRASHED"
    if exit_code == EXIT_OK:
        return "PASSED" if collection_count > 0 else "VACUOUS"
    if exit_code == EXIT_NO_TESTS_COLLECTED:
        return "VACUOUS"
    if exit_code == EXIT_TESTS_FAILED:
        return "FAILED"
    if exit_code in (EXIT_INTERRUPTED, EXIT_INTERNAL_ERROR, EXIT_USAGE_ERROR):
        return "CRASHED"
    return "FAILED"


def run_pytest(
    manifest: Manifest,
    *,
    repo_path: Path,
    branch: str,
    dirty: bool,
    python_path: Path,
    extra_args: list[str] | None = None,
    stdout_dir: Path,
    lane: str = "GOC-38",
) -> dict[str, Any]:
    """Runs the manifest's declared test selection under the two-phase
    protocol above and returns a schema-valid evidence envelope."""
    child_env, env_rejected_vars = filtered_env(manifest.env_allowlist)
    test_paths = manifest.test_selection["paths"]

    launcher = ProcessGroupLauncher(
        deadline_seconds=manifest.timeout_seconds,
        grace_seconds=manifest.grace_seconds,
    )

    # Phase 1: collection.
    collect_argv = [str(python_path), "-m", "pytest", "--collect-only", "-q", *test_paths]
    collect_result = launcher.run(collect_argv, cwd=repo_path, env=child_env)
    collect_stdout = collect_result.stdout.decode("utf-8", errors="replace")
    collection_count, deselection_count = parse_collection(collect_stdout)
    selection_digest = sha256_bytes(collect_stdout.encode("utf-8"))

    # Interpreter identity: compare the venv actually referenced by
    # python_path against the manifest's frozen digest. Never realpath().
    venv_root = python_path.parent.parent
    actual_digest = venv_identity_digest(venv_root)
    venv_identity_match = actual_digest == manifest.interpreter_digest

    # Phase 2: execution (only if phase 1 found something to run; a
    # vacuous collection still produces a full envelope, not a short-circuit,
    # so the VACUOUS verdict itself carries raw evidence).
    argv = [str(python_path), "-m", "pytest", "-q", *test_paths, *(extra_args or [])]
    launch: LaunchResult = launcher.run(argv, cwd=repo_path, env=child_env)
    run_stdout = launch.stdout.decode("utf-8", errors="replace")
    assertion_summary = parse_assertion_summary(run_stdout)
    normalized_outcome = normalize_outcome(launch.exit_code, launch.timed_out, collection_count)

    stdout_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = stdout_dir / "stdout.raw"
    stderr_path = stdout_dir / "stderr.raw"
    stdout_path.write_bytes(launch.stdout)
    stderr_path.write_bytes(launch.stderr)

    from . import manifest as manifest_mod  # local import to avoid cycle at module load

    package_count = manifest_mod.installed_package_count(venv_root)

    return build_envelope(
        manifest=manifest,
        repo_path=repo_path,
        branch=branch,
        dirty=dirty,
        launch=launch,
        adapter="pytest",
        venv_identity_match=venv_identity_match,
        package_count=package_count,
        env_rejected_vars=env_rejected_vars,
        collection_count=collection_count,
        deselection_count=deselection_count,
        quarantine_count=0,
        selection_digest=selection_digest,
        collection_source="pytest --collect-only -q",
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        stdout_truncation_marker="truncated at max_stream_bytes" if launch.stdout_truncated else None,
        stderr_truncation_marker="truncated at max_stream_bytes" if launch.stderr_truncated else None,
        assertion_summary=assertion_summary,
        normalized_outcome=normalized_outcome,
        exit_provenance="direct_waitpid",
        lane=lane,
    )
