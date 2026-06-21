#!/usr/bin/python
"""Diff-scoped change validator — the agent's fast inner loop (CONCEPT:OS-5.50).

Runs the quality bar on **only what this change touched**, turning the minutes-long
full pre-commit into a seconds-long loop: ruff on the changed Python files, the
pinned mypy hook on them, the guardrail gates (concepts/stub/sprawl/env), and pytest
on the changed test files plus the tests that mirror changed modules.

Usage:
    python scripts/validate_change.py            # vs origin/main + working tree
    python scripts/validate_change.py --base HEAD~1
    python scripts/validate_change.py --no-tests # skip pytest (lint/type/gates only)

Exit code is non-zero if any stage fails — wire it into a loop while iterating.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], label: str) -> bool:
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO)
    ok = proc.returncode == 0
    print(f"--- {label}: {'PASS' if ok else 'FAIL'}")
    return ok


def _changed_files(base: str) -> list[str]:
    out: set[str] = set()
    for args in (
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        ["git", "diff", "--name-only"],  # unstaged
        ["git", "diff", "--name-only", "--cached"],  # staged
        ["git", "ls-files", "--others", "--exclude-standard"],  # untracked
    ):
        try:
            res = subprocess.run(args, cwd=REPO, capture_output=True, text=True)
            out.update(ln for ln in res.stdout.splitlines() if ln.strip())
        except Exception:
            pass
    return sorted(f for f in out if (REPO / f).exists())


def _mirror_tests(py_files: list[str]) -> list[str]:
    """Map a changed module to its test file(s) by name, if present."""
    tests: set[str] = set()
    for f in py_files:
        if f.startswith("tests/"):
            tests.add(f)
            continue
        stem = Path(f).stem
        for cand in REPO.glob(f"tests/**/test_{stem}.py"):
            tests.add(str(cand.relative_to(REPO)))
    return sorted(tests)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--no-tests", action="store_true")
    args = ap.parse_args()

    changed = _changed_files(args.base)
    py = [f for f in changed if f.endswith(".py")]
    if not py:
        print("No changed Python files — nothing to validate.")
        return 0
    print(f"Changed Python files ({len(py)}):")
    for f in py:
        print(f"  {f}")

    results: dict[str, bool] = {}
    results["ruff"] = _run(["ruff", "check", *py], "ruff (changed files)")
    results["mypy"] = _run(
        ["pre-commit", "run", "mypy", "--files", *py], "pinned mypy (changed files)"
    )
    for gate in (
        "check_concepts.py",
        "check_no_stub.py",
        "check_sprawl.py",
        "check_no_env_sprawl.py",
    ):
        if (REPO / "scripts" / gate).exists():
            results[gate] = _run([sys.executable, f"scripts/{gate}"], f"gate: {gate}")

    if not args.no_tests:
        tests = _mirror_tests(py)
        if tests:
            results["pytest"] = _run(
                [sys.executable, "-m", "pytest", "-q", *tests], "pytest (mirrored)"
            )
        else:
            print("\n(no mirrored test files for the changed modules)")

    print("\n================ SUMMARY ================")
    for k, v in results.items():
        print(f"  {'✅' if v else '❌'} {k}")
    failed = [k for k, v in results.items() if not v]
    print("=========================================")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("All change-scoped checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
