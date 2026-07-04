#!/usr/bin/env python3
"""Gate: zero legacy ``CONCEPT:<PREFIX>-<digit>`` markers survive (CONCEPT:AU-OS.governance.concept-2).

The big-bang completion invariant: after the OKF-CIS cutover, every marker must be
the new ``<SLUG>-<PILLAR>.<domain>.<concept>`` form. Any surviving numeric-pillar
marker (``CONCEPT:AU-KG.query.vendor-agnostic-traversal``) is a missed rewrite. Exit non-zero listing them.

Usage: python scripts/check_no_legacy_markers.py [ROOT ...]  (default: cwd)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# A legacy marker = CONCEPT: then LETTERS-DIGIT (the new grammar has a word pillar,
# never a digit immediately after the dash).
_LEGACY_RE = re.compile(r"CONCEPT:[A-Z]+-[0-9]")
_EXT = {".py", ".rs", ".md"}
_SKIP = {"__pycache__", ".git", ".venv", "node_modules", "target", "build", "dist"}


def scan(root: Path) -> list[str]:
    hits: list[str] = []
    for p in root.rglob("*"):
        if p.suffix not in _EXT or any(s in p.parts for s in _SKIP):
            continue
        # skip files that legitimately record legacy ids: this gate (documents the
        # pattern), CHANGELOG/concept_map (history), and the generated registries.
        if p.name in {
            "check_no_legacy_markers.py", "CHANGELOG.md", "concept_map.md",
            "concepts.yaml", "concept_reservations.yaml",
        }:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for ln, line in enumerate(text.splitlines(), 1):
            if _LEGACY_RE.search(line):
                hits.append(f"{p.relative_to(root)}:{ln}: {line.strip()[:100]}")
    return hits


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv] or [Path.cwd()]
    all_hits: list[str] = []
    for r in roots:
        all_hits.extend(scan(r))
    if all_hits:
        print(f"FAIL: {len(all_hits)} legacy concept marker(s) survive the cutover:")
        for h in all_hits[:60]:
            print("  " + h)
        if len(all_hits) > 60:
            print(f"  … and {len(all_hits) - 60} more")
        return 1
    print("OK: no legacy CONCEPT: markers remain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
