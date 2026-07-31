#!/usr/bin/env python3
"""Safe wrapper for ``pre-commit run --all-files`` (D-OB-12).

CONCEPT:AU-OS.governance.precommit-all-files-safety.

``pre-commit run --all-files`` internally ``git stash``es every UNSTAGED
change before running hooks and restores it after. When a file-rewriting
hook (``ruff-format``, ``turtle-format``, ``guardrail-docs-contract --write``,
...) touches a path that ALSO had unstaged edits, the restore can silently
DROP those edits instead of merging them — this repo hit it for real during
the fastmcp-4 migration, eating a full round of regenerated docs.

It is especially dangerous here because ``docs/concept_reservations.yaml`` is
a shared, cross-session coordination ledger that is DELIBERATELY left
unstaged by concurrent sessions (see AGENTS.md's concept-reservation
workflow) — one careless ``--all-files`` run can destroy another session's
in-flight reservations.

This wrapper is the mechanical guard (a guard beats a paragraph):

1. Backs up the FULL unstaged diff before the run, so a drop is recoverable
   even in the worst case.
2. Prints an explicit, named warning when ``docs/concept_reservations.yaml``
   (or another tracked file matching a known shared-ledger pattern) is
   unstaged, since that is the highest-risk case this exists to catch.
3. Runs ``pre-commit run --all-files`` (forwarding any extra CLI args).
4. Verifies afterward that the backed-up diff still applies — i.e. nothing
   was silently dropped — and loudly points at the recovery command if not.

Usage: ``python3 scripts/safe_precommit_all_files.py [-- pre-commit args...]``
mirrors ``pre-commit run --all-files [args...]`` and exits with the same
status pre-commit would have.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Tracked files known to be a shared, cross-session coordination ledger that
# is deliberately left unstaged — named here (not just concept_reservations.yaml)
# so a future addition to this class only needs one line, not a rewrite.
_SHARED_UNSTAGED_LEDGERS = ("docs/concept_reservations.yaml",)


def _repo_root(cwd: Path | None = None) -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return Path(out.stdout.strip())


def _unstaged_diff(root: Path) -> str:
    out = subprocess.run(
        ["git", "diff", "--", "."],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    )
    return out.stdout


def _unstaged_paths(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", "--", "."],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    )
    return [line for line in out.stdout.splitlines() if line]


def _diff_still_applies(root: Path, backup: Path) -> bool:
    """True if every change in ``backup`` is still present in the working tree."""
    check = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(backup)],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return check.returncode == 0


def _run_precommit(root: Path, argv: list[str]) -> int:
    """Isolated so tests can stub the actual ``pre-commit`` invocation."""
    return subprocess.run(
        ["pre-commit", "run", "--all-files", *argv], cwd=root
    ).returncode


def main(argv: list[str], *, cwd: Path | None = None) -> int:
    root = _repo_root(cwd)
    diff = _unstaged_diff(root)
    backup: Path | None = None

    if diff.strip():
        touched = set(_unstaged_paths(root))
        backup_dir = root / ".git" / "precommit-all-files-backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = backup_dir / f"unstaged-{stamp}.patch"
        backup.write_text(diff, encoding="utf-8")
        print(f"[safe-precommit] backed up unstaged changes to {backup}")

        risky = sorted(touched & set(_SHARED_UNSTAGED_LEDGERS))
        if risky:
            print(
                "[safe-precommit] WARNING: the following shared, cross-session "
                "coordination ledger(s) have UNSTAGED changes: "
                + ", ".join(risky)
                + ". A file-rewriting hook touching the same path during "
                "--all-files could drop them (D-OB-12). The backup above is "
                "your recovery path if that happens."
            )
    else:
        print("[safe-precommit] no unstaged changes — nothing to protect.")

    status = _run_precommit(root, argv)

    if backup is not None:
        if _diff_still_applies(root, backup):
            print("[safe-precommit] unstaged changes survived the run intact.")
        else:
            print(
                "[safe-precommit] WARNING: your pre-run unstaged changes no "
                "longer apply cleanly — a hook may have altered or dropped "
                f"them. Recover with: git apply --3way '{backup}'"
            )

    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
