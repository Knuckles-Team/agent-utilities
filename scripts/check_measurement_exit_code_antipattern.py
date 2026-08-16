#!/usr/bin/env python3
"""Reject the `cmd | tail` + `$?` exit-status antipattern in shell scripts (measurement harness, D).

CONCEPT:AU-OS.measurement.exit-code-correctness

One incident that motivated the whole measurement harness
(``agent_utilities/measurement/``): a gate was run as
``python3 script.py | tail -25`` in a shell, followed by
``echo "EXIT=$?"`` — ``$?`` there is ``tail``'s exit status (almost always
0), not the script's. The gate was reported "exit 0" (pass) when the thing
actually being measured had never had its exit status observed.

This script statically scans every ``*.sh`` file in the repo for that exact
shape, using :func:`agent_utilities.measurement.run.scan_for_pipeline_exit_antipattern`:
a pipeline ending in a filter command (``tail``, ``head``, ``grep``, ``sed``,
``awk``, ...) followed within a few lines by a bare ``$?`` read with no
``PIPESTATUS``/``pipefail`` guard in between.

Usage:
  python3 scripts/check_measurement_exit_code_antipattern.py

Exit 0 = no antipattern found, 1 = violation(s) found (printed with file:line).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_utilities.measurement.run import (
    scan_for_pipeline_exit_antipattern,  # noqa: E402
)

# Directories never worth scanning: vendored/build output, not code we own.
_EXCLUDE_DIR_PARTS = {
    ".git",
    "node_modules",
    "build-artifacts",
    "__pycache__",
    ".venv",
    "venv",
}


def _iter_shell_scripts(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*.sh")
        if p.is_file() and not _EXCLUDE_DIR_PARTS.intersection(p.parts)
    )


def main() -> int:
    violations = 0
    for path in _iter_shell_scripts(ROOT):
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            print(f"WARN: could not read {path}: {exc}", file=sys.stderr)
            continue
        hits = scan_for_pipeline_exit_antipattern(text)
        for hit in hits:
            violations += 1
            rel = path.relative_to(ROOT)
            print(
                f"{rel}:{hit.line_no}: pipeline into a filter command "
                f"({hit.pipe_line.strip()!r}) whose exit status is then read as "
                f"$? at line {hit.dollar_question_line_no} "
                f"({hit.dollar_question_line.strip()!r}) — this measures the "
                "filter's exit status, not the piped command's. Use "
                "`set -o pipefail` + `${PIPESTATUS[0]}`, or restructure to "
                "avoid the pipe, or capture output to a variable instead of "
                "piping to a filter."
            )

    if violations:
        print(f"\nFAIL: {violations} exit-code antipattern violation(s) found.", file=sys.stderr)
        return 1
    print("OK: no `cmd | filter` + bare `$?` antipattern found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
